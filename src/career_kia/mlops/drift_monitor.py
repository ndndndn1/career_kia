"""피처 드리프트 감지.

- 수치형: Kolmogorov-Smirnov 검정 + PSI (Population Stability Index)
- 범주형: χ² 검정 + PSI
- 배치 단위로 `data/processed/features.parquet` 의 최근 슬라이스와
  학습 스냅샷을 비교하여 드리프트 레포트 생성.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from career_kia.config import PROCESSED_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature: str
    statistic: float
    p_value: float
    psi: float
    drift: bool


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """PSI = Σ (actual_i − expected_i) × ln(actual_i / expected_i)."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    bins_edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(bins_edges) < 2:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=bins_edges)
    cur_hist, _ = np.histogram(cur, bins=bins_edges)
    ref_pct = ref_hist / (ref_hist.sum() + eps)
    cur_pct = cur_hist / (cur_hist.sum() + eps)
    ref_pct = np.clip(ref_pct, eps, 1)
    cur_pct = np.clip(cur_pct, eps, 1)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
) -> DriftResult:
    ks_stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    psi = population_stability_index(reference.to_numpy(), current.to_numpy())
    return DriftResult(
        feature=str(reference.name),
        statistic=float(ks_stat),
        p_value=float(p_value),
        psi=float(psi),
        drift=bool(psi > psi_threshold or p_value < ks_alpha),
    )


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    chi2_alpha: float = 0.01,
) -> DriftResult:
    table = pd.crosstab(
        pd.Index(reference.dropna(), name="ref"),
        pd.Index(current.dropna(), name="cur"),
    )
    if table.size == 0:
        return DriftResult(str(reference.name), float("nan"), 1.0, 0.0, False)
    chi2, p, _, _ = chi2_contingency(table)
    return DriftResult(
        feature=str(reference.name),
        statistic=float(chi2),
        p_value=float(p),
        psi=0.0,
        drift=bool(p < chi2_alpha),
    )


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    num = numeric_cols or reference.select_dtypes(include=[np.number]).columns.tolist()
    cat = categorical_cols or []
    rows: list[DriftResult] = []
    for c in num:
        if c in current.columns:
            rows.append(detect_numeric_drift(reference[c], current[c]))
    for c in cat:
        if c in current.columns:
            rows.append(detect_categorical_drift(reference[c], current[c]))
    return pd.DataFrame(
        [r.__dict__ for r in rows]
    ).sort_values("psi", ascending=False).reset_index(drop=True)


def demo_split_and_report(
    features_path: Path | None = None,
    split_frac: float = 0.5,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """데모: features.parquet 를 시간순으로 반으로 잘라 드리프트 레포트.

    전반부 = 학습 스냅샷, 후반부 = 운영 데이터로 가정.
    """
    features_path = Path(features_path) if features_path else PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(features_path).sort_values("timestamp")
    split = int(len(df) * split_frac)
    ref, cur = df.iloc[:split], df.iloc[split:]
    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "t_rms_mean",
        "t_kurtosis_max",
        "f_env_BPFO_max",
    ]
    categorical = ["Type", "shift", "line_id"]
    report = drift_report(ref, cur, numeric_cols=numeric_cols, categorical_cols=categorical)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        logger.info("드리프트 레포트 저장: %s", output_path)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = PROJECT_ROOT / "mlops_artifacts" / "drift_report.csv"
    report = demo_split_and_report(output_path=out)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
