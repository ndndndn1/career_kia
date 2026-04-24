"""결측치 보간.

제조 현장 데이터 특성에 맞춰 4가지 전략을 제공한다.
    - linear : 공정 파라미터의 물리적 연속성 가정 (가장 흔함)
    - spline : 진동 신호처럼 곡률이 중요한 고주파 gap
    - knn    : 여러 변수가 동시 결측일 때 변수 간 상관 활용
    - seasonal : 계절성(사이클) 성분을 보존해야 할 시계열
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer

ImputeMethod = Literal["linear", "spline", "knn", "seasonal"]


def impute_series(
    s: pd.Series,
    method: ImputeMethod = "linear",
    *,
    limit: int | None = None,
) -> pd.Series:
    """단일 시계열 보간.

    Parameters
    ----------
    s
        값 인덱스가 시간 또는 정수인 1D 시리즈.
    method
        보간 방법.
    limit
        연속 결측이 이 값을 초과하면 보간하지 않음(안전장치).
    """
    if method == "linear":
        return s.interpolate(method="linear", limit=limit, limit_direction="both")

    if method == "spline":
        values = s.to_numpy(dtype=float, copy=True)
        mask = ~np.isnan(values)
        if mask.sum() < 4:
            return s.interpolate(method="linear", limit=limit, limit_direction="both")
        x_known = np.flatnonzero(mask).astype(float)
        y_known = values[mask]
        cs = CubicSpline(x_known, y_known, extrapolate=True)
        missing_idx = np.flatnonzero(~mask)
        values[missing_idx] = cs(missing_idx.astype(float))
        return pd.Series(values, index=s.index, name=s.name)

    if method == "seasonal":
        # 일별 주기(예: 24시간)를 가정하지 않고, 단순 이동평균으로 추세 제거 후
        # 잔차는 선형 보간. 필요 시 `statsmodels.seasonal_decompose`로 교체 가능.
        rolling = s.rolling(window=11, min_periods=1, center=True).mean()
        resid = (s - rolling).interpolate(method="linear", limit=limit, limit_direction="both")
        return rolling + resid

    raise ValueError(f"시리즈 단위에서 지원하지 않는 method: {method}")


def impute_dataframe(
    df: pd.DataFrame,
    *,
    method: ImputeMethod = "linear",
    columns: list[str] | None = None,
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """DataFrame 단위 보간.

    `method == 'knn'` 일 때만 다변량 보간이며 나머지는 컬럼별 독립 보간.
    """
    cols = columns if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    out = df.copy()

    if method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        out[cols] = imputer.fit_transform(out[cols])
        return out

    for c in cols:
        out[c] = impute_series(out[c], method=method)
    return out


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측 요약."""
    total = len(df)
    missing = df.isna().sum()
    pct = (missing / total * 100).round(2)
    return (
        pd.DataFrame({"missing": missing, "pct": pct})
        .loc[missing > 0]
        .sort_values("missing", ascending=False)
    )
