"""베이스라인 분류기 — 단순 기준점."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_baseline(random_state: int = 42) -> Pipeline:
    """L2 로지스틱 회귀 — 선형 기준점."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_rf_baseline(random_state: int = 42) -> RandomForestClassifier:
    """Random Forest — 비선형 기준점."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def adjust_proba_to_prior(
    proba_pos: np.ndarray,
    *,
    train_prior: float,
    target_prior: float,
) -> np.ndarray:
    """학습 prior 대비 목표 prior 로 확률 보정 (class_weight='balanced' 보정용)."""
    if train_prior <= 0 or target_prior <= 0:
        return proba_pos
    lo = (1 - target_prior) / (1 - train_prior)
    hi = target_prior / train_prior
    odds = proba_pos * hi / (proba_pos * hi + (1 - proba_pos) * lo)
    return np.clip(odds, 0, 1)
