"""Build the experiment dataset from the raw CICIoT2023 CSV files.

    python -m src.prepare_data --raw-dir data/raw --out data/processed/ciciot2023_capped.csv \
        --cap 50000 --seed 42

Streams every *.csv in --raw-dir in chunks and keeps a uniform random sample of at most
--cap rows per 8-class category (bottom-k of i.i.d. random keys, so the result does not
depend on file order). Peak memory stays around 8 * cap rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, FEATURES, map_label
from .utils import resolve


def build(raw_dir: Path, out: Path, cap: int, seed: int, label_col: str = "label",
          chunksize: int = 500_000) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")
    rng = np.random.default_rng(seed)
    kept: dict[str, pd.DataFrame] = {}
    for i, f in enumerate(files, 1):
        for chunk in pd.read_csv(f, chunksize=chunksize, usecols=FEATURES + [label_col]):
            chunk["_cat"] = chunk[label_col].map(map_label)
            chunk["_key"] = rng.random(len(chunk))
            for cat, part in chunk.groupby("_cat"):
                merged = pd.concat([kept[cat], part]) if cat in kept else part
                kept[cat] = merged.nsmallest(cap, "_key")
        print(f"[{i}/{len(files)}] {f.name}: " +
              ", ".join(f"{c}={len(kept.get(c, []))}" for c in CLASSES), flush=True)
    df = pd.concat(kept.values()).sort_values("_key").drop(columns=["_cat", "_key"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Build capped CICIoT2023 subset")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/processed/ciciot2023_capped.csv")
    p.add_argument("--cap", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-col", default="label")
    a = p.parse_args()
    build(resolve(a.raw_dir), resolve(a.out), a.cap, a.seed, a.label_col)


if __name__ == "__main__":
    main()
