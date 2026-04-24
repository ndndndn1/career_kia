"""이상치 탐지.

단변량(IQR, MAD)과 다변량(Isolation Forest), 시계열(이동창 Z-score)을 제공한다.
제조 현장에서는 측정 순간 이상치 / 장비 고장 발생 / 실제 불량 이벤트가
섞여 있어 "제거"가 아닌 "마스킹 후 도메인 확인"이 원칙이다. 본 모듈은
`mask_only=True`(기본)로 마스크만 반환하고, 실제 처리는 호출자가 결정한다.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

Method = Literal["iqr", "mad", "rolling_z", "isolation_forest"]


def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR 기반 이상치 마스크 (True = 이상치)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (s < lower) | (s > upper)


def mad_mask(s: pd.Series, threshold: float = 3.5) -> pd.Series:
    """MAD(중위값 절대편차) 기반 로버스트 Z-score 마스크."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)
    modz = 0.6745 * (s - med) / mad
    return modz.abs() > threshold


def rolling_zscore_mask(
    s: pd.Series,
    window: int = 50,
    threshold: float = 3.0,
) -> pd.Series:
    """이동창 Z-score 기반 시계열 이상치 마스크."""
    mean = s.rolling(window=window, min_periods=1, center=True).mean()
    std = s.rolling(window=window, min_periods=1, center=True).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.abs() > threshold


def isolation_forest_mask(
    df: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.Series:
    """Isolation Forest 기반 다변량 이상치 마스크."""
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(df.to_numpy())
    return pd.Series(preds == -1, index=df.index)


def apply_mask(
    s: pd.Series,
    mask: pd.Series,
    *,
    fill: Literal["nan", "clip", "median"] = "nan",
) -> pd.Series:
    """이상치 마스크 적용.

    - nan   : 결측치로 만들어 후속 보간에 위임
    - clip  : IQR whisker 값으로 클리핑
    - median: 중앙값으로 대체
    """
    out = s.copy().astype(float)
    if fill == "nan":
        out[mask] = np.nan
    elif fill == "clip":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[mask & (s < lower)] = lower
        out[mask & (s > upper)] = upper
    elif fill == "median":
        out[mask] = s.median()
    else:
        raise ValueError(f"알 수 없는 fill: {fill}")
    return out
