"""Shared helpers: configuration, seeding, logging and environment capture."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
    "onnx", "onnxruntime", "skl2onnx", "onnxmltools", "psutil", "PyYAML",
]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config; relative paths inside it are resolved against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_run_dir(output_dir: str | Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resolve(output_dir) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _cgroup_limits() -> dict[str, str]:
    """CPU/RAM limits visible inside a container (cgroup v2), if any."""
    out: dict[str, str] = {}
    for key, fname in (("cpu.max", "/sys/fs/cgroup/cpu.max"),                    # cgroup v2
                       ("memory.max", "/sys/fs/cgroup/memory.max"),
                       ("cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),  # cgroup v1
                       ("memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            out[key] = Path(fname).read_text().strip()
        except OSError:
            out[key] = "n/a"
    return out


def capture_env() -> dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_os": os.cpu_count(),
        "cgroup": _cgroup_limits(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "packages": versions,
    }


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
