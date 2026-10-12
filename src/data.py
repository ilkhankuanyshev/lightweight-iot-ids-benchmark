"""CICIoT2023 loading: schema check, 34 -> 8 class mapping, per-class cap, smoke augmentation."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .utils import resolve

log = logging.getLogger("bench")

# The 46 flow features of CICIoT2023 (Neto et al., 2023), in the original column order.
FEATURES: list[str] = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue",
    "Radius", "Covariance", "Variance", "Weight",
]
assert len(FEATURES) == 46

# 8-class scheme: 7 attack categories + benign. Index order is fixed (it is the label id).
CLASSES: list[str] = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
BENIGN_ID = 0

_EXPLICIT = {
    "benigntraffic": "Benign",
    "vulnerabilityscan": "Recon",
    "mitm-arpspoofing": "Spoofing",
    "dns_spoofing": "Spoofing",
    "dictionarybruteforce": "BruteForce",
    "sqlinjection": "Web",
    "commandinjection": "Web",
    "xss": "Web",
    "uploading_attack": "Web",
    "backdoor_malware": "Web",
    "browserhijacking": "Web",
}
_PREFIX = [("ddos-", "DDoS"), ("dos-", "DoS"), ("mirai-", "Mirai"), ("recon-", "Recon")]


def map_label(raw: str) -> str:
    """Map an original CICIoT2023 label (34 classes) to one of the 8 categories."""
    key = str(raw).strip().lower()
    if key in _EXPLICIT:
        return _EXPLICIT[key]
    for prefix, cat in _PREFIX:
        if key.startswith(prefix):
            return cat
    raise KeyError(raw)


def encode_labels(raw: pd.Series) -> np.ndarray:
    mapped, unknown = [], set()
    for value in raw:
        try:
            mapped.append(CLASSES.index(map_label(value)))
        except KeyError:
            unknown.add(value)
            mapped.append(-1)
    if unknown:
        raise ValueError(f"Unknown labels (extend data.map_label): {sorted(unknown)}")
    return np.asarray(mapped, dtype=np.int64)


def _smoke_augment(X: np.ndarray, y: np.ndarray, rows_per_class: int, noise: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Resample each class with small multiplicative jitter so CV is mechanically possible.

    Only for the smoke test on the 10-row sample: the numbers it produces carry no
    scientific meaning; the point is to execute the exact code path of a full run.
    """
    xs, ys = [], []
    for cls in np.unique(y):
        idx = rng.choice(np.flatnonzero(y == cls), size=rows_per_class, replace=True)
        base = X[idx]
        jitter = 1.0 + rng.normal(0.0, noise, size=base.shape)
        xs.append(np.clip(base * jitter + rng.normal(0.0, noise, size=base.shape), 0, None))
        ys.append(np.full(rows_per_class, cls, dtype=np.int64))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


def load_dataset(cfg: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    dcfg = cfg["data"]
    path = resolve(dcfg["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md (run src.prepare_data first).")
    df = pd.read_csv(path)
    label_col = dcfg.get("label_column", "label")
    missing = [c for c in FEATURES + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Schema mismatch, missing columns: {missing}")

    X = df[FEATURES].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    if (~finite).any():
        log.warning("Dropping %d rows with NaN/inf", int((~finite).sum()))
    X, labels = X[finite], df.loc[finite, label_col]
    y = encode_labels(labels)
    X = X.astype(np.float32)

    rng = np.random.default_rng(seed)
    cap = dcfg.get("max_rows_per_class")
    if cap:
        keep = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            keep.append(idx if len(idx) <= cap else rng.choice(idx, size=cap, replace=False))
        keep = np.sort(np.concatenate(keep))
        X, y = X[keep], y[keep]

    aug = dcfg.get("smoke_augment")
    if aug:
        log.warning("SMOKE MODE: augmenting %d sample rows by jittered resampling; "
                    "metrics are NOT meaningful", len(y))
        X, y = _smoke_augment(X, y, int(aug["rows_per_class"]), float(aug["noise"]), rng)

    present = sorted(set(y.tolist()))
    if len(present) != len(CLASSES):
        log.warning("Only %d of %d classes present: %s", len(present), len(CLASSES),
                    [CLASSES[i] for i in present])
    counts = {CLASSES[c]: int((y == c).sum()) for c in present}
    log.info("Loaded %d rows x %d features from %s; class counts: %s",
             len(y), X.shape[1], path.name, counts)
    return X, y
