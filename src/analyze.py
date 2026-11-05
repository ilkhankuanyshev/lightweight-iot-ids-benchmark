"""Aggregate metrics.csv and test H1.1-H1.3.

    python -m src.analyze results/<run_dir>      # re-run the analysis on an existing run

H1.1  Friedman test across models (k = h1_features, FP32) + Nemenyi critical difference.
H1.2  per model, k_from -> k_to: one-sided Wilcoxon that latency drops by >= min_reduction,
      and Wilcoxon non-inferiority test on macro-F1 with margin m. Holm across models.
H1.3  per neural model, FP32 -> INT8: size ratio >= min_size_ratio on every split,
      one-sided Wilcoxon on latency, non-inferiority on macro-F1. Holm across models.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from .utils import load_config

# Nemenyi critical values q_0.05 (Demsar, 2006), index = number of models
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def _wilcoxon(d: np.ndarray, alternative: str) -> float:
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _pivot(df: pd.DataFrame, metric: str, index: str = "split", col: str = "model"):
    return df.pivot_table(index=index, columns=col, values=metric, aggfunc="first").dropna()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[(df["status"] == "ok") & (df["model"] != "dummy")]
    agg = ok.groupby(["model", "n_features", "precision"]).agg(
        n=("macro_f1", "size"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        p50_ms_mean=("latency_p50_ms", "mean"), p95_ms_mean=("latency_p95_ms", "mean"),
        throughput_rps=("throughput_rps", "mean"), size_kb=("model_size_kb", "mean"),
        ram_peak_mb=("ram_peak_mb", "mean"), benign_fpr=("benign_fpr", "mean"),
        guardrails_pass_rate=("guardrails_pass", lambda s: s.astype(str).eq("True").mean()),
        baseline=("is_baseline", lambda s: s.astype(str).eq("True").any()),
    ).reset_index()
    return agg.sort_values(["n_features", "macro_f1_mean"], ascending=[False, False])


def test_h1(df: pd.DataFrame, s: dict[str, Any]) -> list[str]:
    sub = df[(df["n_features"] == s["h1_features"]) & (df["precision"] == "fp32")
             & (df["status"] == "ok") & (df["model"] != "dummy")]
    out = [f"### H1.1 — model type (k = {s['h1_features']}, FP32)", ""]
    for metric in ("macro_f1", "latency_p50_ms"):
        piv = _pivot(sub, metric)
        k, n = piv.shape[1], piv.shape[0]
        if k < 3 or n < 2:
            out.append(f"- {metric}: need >= 3 models and >= 2 complete splits (have {k}, {n}).")
            continue
        stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        higher_better = metric == "macro_f1"
        ranks = piv.rank(axis=1, ascending=not higher_better).mean().sort_values()
        cd = _Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
        verdict = "reject H0.1" if p < s["alpha"] else "fail to reject H0.1"
        out.append(f"- **{metric}**: Friedman chi2 = {stat:.3f}, p = {p:.4g}, n = {n} -> "
                   f"{verdict}. Mean ranks (1 = best): "
                   + ", ".join(f"{m} {r:.2f}" for m, r in ranks.items())
                   + f". Nemenyi CD = {cd:.2f} (rank gaps larger than CD are significant).")
    return out + [""]


def test_h2(df: pd.DataFrame, s: dict[str, Any]) -> list[str]:
    kf, kt, red, m = s["h2_from"], s["h2_to"], s["h2_min_latency_reduction"], s[
        "noninferiority_margin_f1"]
    sub = df[(df["precision"] == "fp32") & (df["status"] == "ok") & (df["model"] != "dummy")]
    rows = []
    for model, g in sub.groupby("model"):
        lat = _pivot(g, "latency_p50_ms", col="n_features")
        f1 = _pivot(g, "macro_f1", col="n_features")
        if kf not in lat or kt not in lat:
            continue
        rel = 1 - (lat[kt] / lat[kf]).median()
        rows.append({"model": model, "n": len(lat), "median_latency_reduction": rel,
                     "median_dF1_pp": 100 * (f1[kt] - f1[kf]).median(),
                     "p_latency": _wilcoxon(lat[kt] - (1 - red) * lat[kf], "less"),
                     "p_f1_noninf": _wilcoxon(f1[kt] - f1[kf] + m, "greater")})
    return _family_report(rows, s, f"### H1.2 — features {kf} -> {kt}",
                          f"latency reduction >= {red:.0%} AND macro-F1 drop < {100 * m:.1f} pp")


def test_h3(df: pd.DataFrame, s: dict[str, Any]) -> list[str]:
    k, ratio_min, m = s["h3_features"], s["h3_min_size_ratio"], s["noninferiority_margin_f1"]
    sub = df[(df["n_features"] == k) & (df["status"] == "ok") & (df["model"] != "dummy")]
    rows = []
    for model, g in sub.groupby("model"):
        if not {"fp32", "int8"} <= set(g["precision"]):
            continue
        piv = {mt: _pivot(g, mt, col="precision") for mt in
               ("latency_p50_ms", "macro_f1", "model_size_kb")}
        ratio = piv["model_size_kb"]["fp32"] / piv["model_size_kb"]["int8"]
        lat, f1 = piv["latency_p50_ms"], piv["macro_f1"]
        rows.append({"model": model, "n": len(lat), "min_size_ratio": ratio.min(),
                     "median_latency_ratio": (lat["int8"] / lat["fp32"]).median(),
                     "median_dF1_pp": 100 * (f1["int8"] - f1["fp32"]).median(),
                     "p_latency": _wilcoxon(lat["int8"] - lat["fp32"], "less"),
                     "p_f1_noninf": _wilcoxon(f1["int8"] - f1["fp32"] + m, "greater"),
                     "size_ok": bool(ratio.min() >= ratio_min)})
    return _family_report(rows, s, f"### H1.3 — FP32 -> INT8 (k = {k})",
                          f"size ratio >= {ratio_min}x on every split AND latency lower "
                          f"AND macro-F1 drop < {100 * m:.1f} pp")


def _family_report(rows: list[dict], s: dict[str, Any], title: str, rule: str) -> list[str]:
    out = [title, "", f"Supported for a model if: {rule} (Holm-adjusted, alpha = {s['alpha']}).", ""]
    if not rows:
        return out + ["Not enough data for this test in this run.", ""]
    res = pd.DataFrame(rows)
    res["p_latency_holm"] = holm(res["p_latency"].tolist())
    res["p_f1_noninf_holm"] = holm(res["p_f1_noninf"].tolist())
    ok = (res["p_latency_holm"] < s["alpha"]) & (res["p_f1_noninf_holm"] < s["alpha"])
    if "size_ok" in res:
        ok &= res["size_ok"]
    res["H1_supported"] = ok
    n_min = int(res["n"].min())
    note = []
    if 0.5 ** n_min > s["alpha"]:
        note = [f"> n = {n_min} paired splits: the smallest attainable one-sided Wilcoxon p "
                f"is {0.5 ** n_min:.3g} > alpha, so H1 cannot be supported. Use more splits.", ""]
    return out + [_md_table(res), ""] + note


def write_reports(run_dir: Path, cfg: dict[str, Any], stats: bool = True) -> None:
    df = pd.read_csv(run_dir / "metrics.csv")
    smoke = bool(cfg["data"].get("smoke_augment"))
    banner = (["> **SMOKE RUN on augmented 10-row sample — numbers below only prove that "
               "the pipeline works; they have no scientific meaning.**", ""] if smoke else [])

    summary = summarize(df)
    summary.to_csv(run_dir / "summary.csv", index=False)
    failed = int((df["status"] != "ok").sum())
    (run_dir / "summary.md").write_text("\n".join(
        [f"# Summary — {run_dir.name}", ""] + banner
        + [f"Rows: {len(df)} | failed: {failed} | splits: {df['split'].nunique()}", "",
           _md_table(summary), ""]), encoding="utf-8")

    if not stats:
        return
    if smoke:
        (run_dir / "stats.md").write_text(
            "# Hypothesis tests\n\nNOT TESTED: synthetic smoke input cannot support H1.1-H1.3.\n",
            encoding="utf-8")
        return
    s = cfg["stats"]
    lines = [f"# Hypothesis tests — {run_dir.name}", ""] + banner
    for fn in (test_h1, test_h2, test_h3):
        try:
            lines += fn(df, s)
        except Exception as exc:  # analysis must never lose the raw metrics
            lines += [f"{fn.__name__} failed: {type(exc).__name__}: {exc}", ""]
    (run_dir / "stats.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1])
    write_reports(target, load_config(target / "config_snapshot.yaml"))
    print((target / "stats.md").read_text(encoding="utf-8"))
