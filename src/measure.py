"""Quality metrics (from the exported ONNX model) and cost metrics (isolated process)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, recall_score

from .data import BENIGN_ID, CLASSES
from .utils import REPO_ROOT


def predict_onnx(model_path: Path, X: np.ndarray, output_name: str,
                 chunk: int = 20_000) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = [sess.run([output_name], {in_name: X[i:i + chunk].astype(np.float32)})[0]
           for i in range(0, len(X), chunk)]
    return np.concatenate(out).astype(np.int64).ravel()


def quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = list(range(len(CLASSES)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    benign = y_true == BENIGN_ID
    fpr = float((y_pred[benign] != BENIGN_ID).mean()) if benign.any() else float("nan")
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "benign_fpr": fpr,
    }
    out.update({f"recall_{c}": float(r) for c, r in zip(CLASSES, rec)})
    return out


def cost_metrics(model_path: Path, rows_path: Path, output_name: str,
                 mcfg: dict[str, Any]) -> dict[str, Any]:
    """Latency / throughput / RAM of one ONNX model, measured by src.latency_worker."""
    params = {k: mcfg[k] for k in ("warmup_runs", "timed_runs", "throughput_batch_size",
                                   "throughput_repeats", "intra_op_threads", "inter_op_threads")}
    params["output_name"] = output_name
    cmd = [sys.executable, "-m", "src.latency_worker", str(model_path), str(rows_path),
           json.dumps(params)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=float(mcfg["timeout_s"]))
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(f"latency worker exit {proc.returncode}: {' | '.join(tail)}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["model_size_kb"] = model_path.stat().st_size / 1024.0
    return result
