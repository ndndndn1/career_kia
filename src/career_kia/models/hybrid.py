"""하이브리드 모델 (성능 + 해석).

LightGBM의 예측 확률과 원 피처의 부분집합을 결합해 **pyGAM** (Generalized
Additive Model) 메타 학습기에 투입한다. 이 구조는 다음 두 목적을 동시에 만족::

    - LightGBM: 비선형/상호작용 포착으로 최대 성능
    - pyGAM:    각 변수의 비선형 성분(편향 없는 부분의존성)을 명시적으로
                보여주는 가법 모델. 현장 엔지니어가 해석하기 쉽다.

        LGBM proba ──┐
                     ├── → pyGAM (logistic link) ─→ 최종 확률
        원 피처(선택)┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pygam import LogisticGAM, s
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    # pyGAM 메타에 넘길 해석 피처 (가법 모델이라 적을수록 좋음)
    interpret_features: list[str] = field(
        default_factory=lambda: [
            "Tool wear [min]",
            "Torque [Nm]",
            "Rotational speed [rpm]",
            "Process temperature [K]",
            "Air temperature [K]",
        ]
    )
    # 희소 라벨(Machine failure ~3%) 대응
    class_weight: str | None = "balanced"


class HybridModel(BaseEstimator, ClassifierMixin):
    """LightGBM + pyGAM 스태킹.

    Usage
    -----
    >>> model = HybridModel(config=HybridConfig())
    >>> model.fit(X_df, y)
    >>> proba = model.predict_proba(X_df)[:, 1]
    """

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self.lgbm_: LGBMClassifier | None = None
        self.gam_: LogisticGAM | None = None
        self.feature_names_: list[str] | None = None
        self.interpret_idx_: list[int] | None = None
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sanitize_columns(cols: Iterable[str]) -> list[str]:
        """LightGBM 가 허용하지 않는 특수문자 제거."""
        out: list[str] = []
        for c in cols:
            s = str(c)
            for ch in "[](){}\"':,":
                s = s.replace(ch, "")
            s = s.replace(" ", "_")
            out.append(s)
        return out

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()
        X2.columns = self._sanitize_columns(X2.columns)
        return X2

    # scikit-learn API
    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> "HybridModel":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 는 pandas DataFrame 이어야 합니다.")
        y_arr = np.asarray(y).astype(int)
        self.feature_names_ = list(X.columns)
        X_sanitized = self._prepare_X(X)

        # 1) LightGBM 학습
        lgbm = LGBMClassifier(
            **self.config.lgbm_params,
            class_weight=self.config.class_weight,
        )
        lgbm.fit(X_sanitized, y_arr)
        self.lgbm_ = lgbm

        # 2) LightGBM 예측 확률 + 해석 피처 → pyGAM
        p_lgbm = lgbm.predict_proba(X_sanitized)[:, 1]
        # logit 변환으로 GAM의 logistic link 와 결합이 더 안정적
        z = np.clip(p_lgbm, 1e-4, 1 - 1e-4)
        lgbm_logit = np.log(z / (1 - z))

        interp_cols = [c for c in self.config.interpret_features if c in X.columns]
        self.interpret_idx_ = [X.columns.get_loc(c) for c in interp_cols]

        meta_X = np.column_stack([lgbm_logit, X[interp_cols].to_numpy()])
        # 각 변수마다 스무스 항
        terms = s(0)
        for i in range(1, meta_X.shape[1]):
            terms += s(i)
        gam = LogisticGAM(terms)
        gam.fit(meta_X, y_arr)
        self.gam_ = gam
        logger.info(
            "HybridModel 학습 완료: LGBM features=%d, GAM meta=%d",
            X.shape[1],
            meta_X.shape[1],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.lgbm_ is None or self.gam_ is None:
            raise RuntimeError("아직 학습되지 않았습니다. fit() 먼저.")
        X_sanitized = self._prepare_X(X)
        p_lgbm = self.lgbm_.predict_proba(X_sanitized)[:, 1]
        z = np.clip(p_lgbm, 1e-4, 1 - 1e-4)
        lgbm_logit = np.log(z / (1 - z))
        interp_cols = [self.feature_names_[i] for i in self.interpret_idx_]
        meta_X = np.column_stack([lgbm_logit, X[interp_cols].to_numpy()])
        p_final = self.gam_.predict_proba(meta_X)
        return np.column_stack([1 - p_final, p_final])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # 해석 편의
    def partial_dependence(self, feature: str) -> tuple[np.ndarray, np.ndarray]:
        """pyGAM 메타 모델의 부분의존성 (feature: 'lgbm_logit' 또는 interpret_features 중 하나)."""
        if self.gam_ is None:
            raise RuntimeError("아직 학습되지 않았습니다.")
        interp_cols = [self.feature_names_[i] for i in self.interpret_idx_]
        names = ["lgbm_logit", *interp_cols]
        if feature not in names:
            raise ValueError(f"{feature} 없음. 선택 가능: {names}")
        idx = names.index(feature)
        XX = self.gam_.generate_X_grid(term=idx)
        pdep = self.gam_.partial_dependence(term=idx, X=XX)
        return XX[:, idx], pdep
