"""Benchmark runner.

    python -m src.benchmark --config configs/smoke.yaml                  # train + measure
    python -m src.benchmark --config configs/config.yaml --phase train   # train only
    python -m src.benchmark --phase measure --run-dir results/<run>      # measure only

For every CV split x feature count k x model x precision (FP32 / INT8) it trains the model,
exports it to ONNX, evaluates macro-F1 on the test fold, measures latency / throughput /
RAM in a fresh process and checks guardrails. Rows are appended to metrics.csv as they come,
so an interrupted run keeps its partial results.

Two-phase mode separates training (needs RAM, no limits) from measurement (must run under
the device limits, e.g. `docker run --cpus=1 --memory=1g`, or on a Raspberry Pi to which the
results/<run> folder was copied). Quality metrics do not depend on the device; cost metrics do.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from . import analyze  # noqa: E402
from .data import CLASSES, FEATURES, load_dataset  # noqa: E402
from .features import rank_features  # noqa: E402
from .measure import cost_metrics, predict_onnx, quality_metrics  # noqa: E402
from .models import (MODEL_NAMES, NEURAL, build_model, check_classes,  # noqa: E402
                     onnx_label_output, to_onnx)
from .quantize import quantize_int8  # noqa: E402
from .utils import (capture_env, load_config, make_run_dir, resolve,  # noqa: E402
                    set_seed, setup_logger, write_json)

FIELDS = (["run_id", "split", "repeat", "fold", "model", "n_features", "precision",
           "is_baseline", "status", "error", "train_time_s", "n_train", "n_test",
           "macro_f1", "accuracy", "benign_fpr"]
          + [f"recall_{c}" for c in CLASSES]
          + ["latency_p50_ms", "latency_p95_ms", "latency_mean_ms", "latency_std_ms",
             "throughput_rps", "session_load_ms", "model_size_kb", "ram_peak_mb",
             "ram_model_mb", "guardrails_pass", "guardrail_violations", "features",
             "output_name", "artifact", "timing_rows"])


def _num(value: Any) -> float | None:
    """Values read back from metrics.csv are strings; '' means not measured."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_guardrails(row: dict[str, Any], g: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = [("latency_p50_ms", "max_latency_p50_ms"), ("model_size_kb", "max_model_size_kb"),
             ("ram_peak_mb", "max_peak_ram_mb"), ("benign_fpr", "max_benign_fpr")]
    violations = []
    if row.get("status") != "ok":
        violations.append("run_failed")
    for metric, limit_key in rules:
        value, limit = _num(row.get(metric)), g.get(limit_key)
        if limit is not None and (value is None or not np.isfinite(value)):
            violations.append(f"{metric}:missing")
        elif limit is not None and value > limit:
            violations.append(f"{metric}>{limit}")
    return not violations, violations


def run(cfg: dict[str, Any], max_splits: int | None = None, phase: str = "all") -> Path:
    warnings.filterwarnings("once", category=ConvergenceWarning)
    exp = cfg["experiment"]
    seed = int(exp["seed"])
    set_seed(seed)
    run_dir = make_run_dir(exp.get("output_dir", "results"), exp["name"])
    log = setup_logger(run_dir)
    run_id = run_dir.name
    shutil.copy(cfg["_config_path"], run_dir / "config_snapshot.yaml")
    write_json(capture_env(), run_dir / "env.json")
    log.info("Run %s | phase %s | config %s", run_id, phase, cfg["_config_path"])
    measure_now = phase == "all"

    X, y = load_dataset(cfg, seed)
    counts = sorted({int(k) for k in cfg["features"]["counts"]}, reverse=True)
    if not all(1 <= k <= len(FEATURES) for k in counts):
        raise ValueError(f"features.counts must be within 1..{len(FEATURES)}")
    model_cfgs = {n: c for n, c in cfg["models"].items() if c.get("enabled", True)}
    unknown = set(model_cfgs) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models in config: {unknown}")
    qcfg, mcfg = cfg.get("quantization", {}), cfg["measurement"]
    g, base = cfg.get("guardrails", {}), cfg.get("baseline", {})

    cv_cfg = cfg["cv"]
    cv = RepeatedStratifiedKFold(n_splits=int(cv_cfg["n_splits"]),
                                 n_repeats=int(cv_cfg["n_repeats"]), random_state=seed)
    total = int(cv_cfg["n_splits"]) * int(cv_cfg["n_repeats"])
    if max_splits:
        total = min(total, max_splits)
    work = run_dir / "artifacts"
    work.mkdir()
    keep_models = bool(exp.get("keep_models", False)) or not measure_now
    rng = np.random.default_rng(seed)

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()

        def emit(row: dict[str, Any]) -> None:
            writer.writerow(row)
            fh.flush()

        t_start = time.perf_counter()
        for split, (tr, te) in enumerate(cv.split(X, y)):
            if split >= total:
                break
            repeat, fold = divmod(split, int(cv_cfg["n_splits"]))
            tag = f"[split {split + 1}/{total} r{repeat} f{fold}]"
            ranking = rank_features(X[tr], y[tr], cfg["features"]["selector"],
                                    cfg["features"].get("mi_max_rows"), seed)
            log.info("%s MI ranking (top 10): %s", tag, [FEATURES[i] for i in ranking[:10]])

            dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
            emit({"run_id": run_id, "split": split, "repeat": repeat, "fold": fold,
                  "model": "dummy", "n_features": len(FEATURES), "precision": "-",
                  "status": "ok", "is_baseline": False,
                  **quality_metrics(y[te], dummy.predict(X[te]))})

            for k in counts:
                cols = np.sort(ranking[:k])          # keep original column order for the CNN
                Xtr, Xte = X[tr][:, cols], X[te][:, cols]
                n_rows = min(len(Xte), int(mcfg["timed_runs"]))
                rows_path = work / f"s{split}_k{k}_timing_rows.npy"
                np.save(rows_path, Xte[rng.choice(len(Xte), size=n_rows, replace=False)])
                calib = Xtr[rng.choice(len(Xtr), size=min(len(Xtr),
                                       int(qcfg.get("calibration_rows", 1000))), replace=False)]

                for name, mc in model_cfgs.items():
                    common = {"run_id": run_id, "split": split, "repeat": repeat, "fold": fold,
                              "model": name, "n_features": k, "n_train": len(tr),
                              "n_test": len(te), "features": "|".join(FEATURES[i] for i in cols)}
                    variants: list[tuple[str, Path]] = []
                    try:
                        t0 = time.perf_counter()
                        model = build_model(name, mc.get("params"), seed, len(CLASSES))
                        model.fit(Xtr, y[tr])
                        common["train_time_s"] = time.perf_counter() - t0
                        check_classes(model, name, len(CLASSES))
                        onx = to_onnx(name, model, k)
                        out_name = onnx_label_output(onx)
                        fp32 = work / f"s{split}_k{k}_{name}_fp32.onnx"
                        onnx.save_model(onx, str(fp32))
                        variants.append(("fp32", fp32))
                        if qcfg.get("enabled") and name in NEURAL and name in qcfg.get(
                                "apply_to", NEURAL):
                            int8 = work / f"s{split}_k{k}_{name}_int8.onnx"
                            quantize_int8(fp32, int8, calib, qcfg)
                            variants.append(("int8", int8))
                    except Exception as exc:  # training/export failure -> one failed row
                        log.exception("%s k=%d %s: training/export failed", tag, k, name)
                        row = {**common, "precision": "fp32", "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"[:300]}
                        row["guardrails_pass"], v = check_guardrails(row, g)
                        row["guardrail_violations"] = ";".join(v)
                        emit(row)
                        continue

                    for precision, path in variants:
                        row = {**common, "precision": precision, "status": "ok",
                               "output_name": out_name,
                               "artifact": str(path.relative_to(run_dir)),
                               "timing_rows": str(rows_path.relative_to(run_dir)),
                               "model_size_kb": path.stat().st_size / 1024.0,
                               "is_baseline": (name == base.get("model")
                                               and k == int(base.get("n_features", -1))
                                               and precision == base.get("precision", "fp32"))}
                        try:
                            row.update(quality_metrics(y[te], predict_onnx(path, Xte, out_name)))
                        except Exception as exc:
                            log.exception("%s k=%d %s %s: evaluation failed",
                                          tag, k, name, precision)
                            row.update(status="failed",
                                       error=f"{type(exc).__name__}: {exc}"[:300])
                        label = f"{tag} k={k:<2d} {name:<13s} {precision}"
                        if measure_now:
                            _measure_row(row, run_dir, mcfg, g, log, label)
                        elif row["status"] == "ok":
                            log.info("%s | F1=%.4f size=%.1fKB | cost: pending (measure phase)",
                                     label, row["macro_f1"], row["model_size_kb"])
                        if not keep_models:
                            path.unlink(missing_ok=True)
                            row["artifact"] = row["timing_rows"] = ""
                        emit(row)

    log.info("All splits done in %.1fs -> %s", time.perf_counter() - t_start, csv_path)
    if not keep_models:
        shutil.rmtree(work, ignore_errors=True)
    if measure_now:
        analyze.write_reports(run_dir, cfg)
        log.info("Reports: %s, %s", run_dir / "summary.md", run_dir / "stats.md")
    else:
        analyze.write_reports(run_dir, cfg, stats=False)
        log.info("Training done. Next, under device limits:\n  python -m src.benchmark "
                 "--phase measure --run-dir %s", run_dir)
    return run_dir


def _measure_row(row: dict[str, Any], run_dir: Path, mcfg: dict[str, Any],
                 g: dict[str, Any], log, label: str) -> None:
    if row["status"] == "ok":
        try:
            row.update(cost_metrics(run_dir / row["artifact"], run_dir / row["timing_rows"],
                                    row["output_name"], {
                                        **mcfg,
                                        "raw_path": str(run_dir / "raw" / (Path(row["artifact"]).stem + ".jsonl")),
                                        "context": {"schema_version": 1, "timing_origin": "measured",
                                            "data_origin": mcfg.get("data_origin", "unverified"),
                                            "model": row["model"], "n_features": int(row["n_features"]),
                                            "precision": row["precision"], "split": int(row["split"]),
                                            "run_id": run_dir.name,
                                            "unit_id": run_dir.name + ":" + str(row["split"]),
                                            "batch_size": 1}
                                    }))
        except Exception as exc:
            log.exception("%s: measurement failed", label)
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    ok, v = check_guardrails(row, g)
    row["guardrails_pass"], row["guardrail_violations"] = ok, ";".join(v)
    if row["status"] == "ok":
        log.info("%s | F1=%.4f p50=%.4fms size=%.1fKB RAM=%.0fMB | %s", label,
                 float(row["macro_f1"]), row["latency_p50_ms"], row["model_size_kb"],
                 row["ram_peak_mb"], "guardrails OK" if ok else "VIOLATES " + ";".join(v))


def measure_run(run_dir: str | Path, max_rows: int | None = None) -> Path:
    """Phase 2: measure inference cost of every model saved by a `--phase train` run."""
    run_dir = resolve(run_dir)
    cfg = load_config(run_dir / "config_snapshot.yaml")
    log = setup_logger(run_dir)
    write_json(capture_env(), run_dir / "env_measure.json")
    log.info("Measure phase for %s", run_dir.name)
    src_csv, tmp_csv = run_dir / "metrics.csv", run_dir / "metrics.measuring.csv"
    with src_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with tmp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        done = 0
        for row in rows:
            if row["model"] != "dummy" and row.get("artifact") and (
                    max_rows is None or done < max_rows):
                row["model_size_kb"] = float(row["model_size_kb"])
                _measure_row(row, run_dir, cfg["measurement"], cfg.get("guardrails", {}), log,
                             f"[split {row['split']}] k={row['n_features']:<2s} "
                             f"{row['model']:<13s} {row['precision']}")
                done += 1
            writer.writerow(row)
            fh.flush()
    os.replace(tmp_csv, src_csv)
    analyze.write_reports(run_dir, cfg)
    log.info("Measured %d models. Reports: %s, %s", done, run_dir / "summary.md",
             run_dir / "stats.md")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-splits", type=int, default=None,
                        help="stop after N CV splits (quick checks)")
    parser.add_argument("--phase", choices=["all", "train", "measure"], default="all",
                        help="all = train + measure; train = no cost metrics; "
                             "measure = cost metrics for an existing --run-dir")
    parser.add_argument("--run-dir", help="results/<run> folder of a train-phase run")
    args = parser.parse_args(argv)
    if args.phase == "measure":
        if not args.run_dir:
            parser.error("--phase measure requires --run-dir")
        measure_run(args.run_dir)
    else:
        run(load_config(args.config), args.max_splits, args.phase)


if __name__ == "__main__":
    main()
