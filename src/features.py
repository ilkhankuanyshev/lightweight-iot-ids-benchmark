"""Feature selection: rank features by mutual information on the TRAINING fold only."""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def rank_features(X_train: np.ndarray, y_train: np.ndarray, method: str,
                  max_rows: int | None, seed: int) -> np.ndarray:
    """Return feature indices sorted from most to least informative."""
    if method != "mutual_info":
        raise ValueError(f"Unsupported feature selector: {method}")
    rng = np.random.default_rng(seed)
    if max_rows and len(y_train) > max_rows:
        idx = rng.choice(len(y_train), size=max_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    scores = mutual_info_classif(X_train, y_train, random_state=seed)
    # stable sort: ties keep the original column order -> deterministic ranking
    return np.argsort(-scores, kind="stable")
