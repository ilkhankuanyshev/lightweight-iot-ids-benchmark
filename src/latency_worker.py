"""Inference cost measurement in a FRESH Python process.

Run as:  python -m src.latency_worker <model.onnx> <rows.npy> '<json params>'
Prints one JSON line with latency / throughput / memory. Only numpy, onnxruntime and psutil
are imported, so the memory footprint is not polluted by training libraries.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import psutil  # noqa: E402


def peak_rss_mb() -> float:
    """Peak RSS of THIS process image. /proc VmHWM is reset on exec(); getrusage's
    ru_maxrss is not (it would include the parent's memory), so it is only a fallback."""
    try:
        for line in open("/proc/self/status", encoding="ascii"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure(model_path: str, rows_path: str, p: dict) -> dict:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(p.get("intra_op_threads", 1))
    so.inter_op_num_threads = int(p.get("inter_op_threads", 1))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1e3
    in_name, out_name = sess.get_inputs()[0].name, p["output_name"]

    X = np.load(rows_path).astype(np.float32)
    rows = [X[i:i + 1] for i in range(len(X))]
    for i in range(int(p["warmup_runs"])):
        sess.run([out_name], {in_name: rows[i % len(rows)]})

    n = int(p["timed_runs"])
    lat_ns = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = rows[i % len(rows)]
        t = time.perf_counter_ns()
        sess.run([out_name], {in_name: r})
        lat_ns[i] = time.perf_counter_ns() - t
    lat_ms = lat_ns / 1e6

    bs = int(p["throughput_batch_size"])
    batch = np.resize(X, (bs, X.shape[1])).astype(np.float32)
    sess.run([out_name], {in_name: batch})
    reps = int(p["throughput_repeats"])
    t = time.perf_counter()
    for _ in range(reps):
        sess.run([out_name], {in_name: batch})
    throughput = bs * reps / (time.perf_counter() - t)

    rss_after = proc.memory_info().rss
    return {
        "latency_p50_ms": float(np.percentile(lat_ms, 50)),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "latency_mean_ms": float(lat_ms.mean()),
        "latency_std_ms": float(lat_ms.std(ddof=1)) if n > 1 else 0.0,
        "throughput_rps": float(throughput),
        "session_load_ms": float(load_ms),
        "ram_peak_mb": peak_rss_mb(),
        "ram_model_mb": (rss_after - rss_before) / 2**20,
        "ort_version": ort.__version__,
    }


if __name__ == "__main__":
    result = measure(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
    print(json.dumps(result))
