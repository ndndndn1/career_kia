"""피처 매트릭스 로더 — mlflow-free.

`train.py` 는 mlflow 등 무거운 의존성을 끌어오므로, 대시보드/추론처럼
학습이 아닌 곳에서는 본 모듈만 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from career_kia.config import PROCESSED_DIR


META_DROP: list[str] = [
    "batch_id",
    "timestamp",
    "shift",
    "operator_id",
    "line_id",
    "vibration_fault_type",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_feature_matrix(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """피처 테이블 → (X, y, groups). groups는 누수 방지용 operator_id."""
    path = Path(path) if path else PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(path)
    y = df["Machine failure"].astype(int)
    groups = df["operator_id"].astype(str)
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)
    X = df.drop(columns=[c for c in META_DROP if c in df.columns])
    return X, y, groups
