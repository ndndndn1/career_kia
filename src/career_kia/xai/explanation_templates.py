"""설명 템플릿 표준화.

JD 의 "설명 템플릿 표준화(변수 기여도, 임계 구간, 상호작용 효과)" 에 대응.
이 모듈은 SHAP 결과를 세 가지 표준 구조로 정리한다.

    1) ContributionExplanation : 변수 기여도 (개별 샘플 top-K)
    2) ThresholdExplanation    : 임계 구간 (PDP/ICE 에서 도출)
    3) InteractionExplanation  : 상호작용 효과 (SHAP interaction top 쌍)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from career_kia.xai.shap_utils import ShapBundle, top_contributors, top_interactions


# ---------------------------------------------------------------------------
# 표준 구조체
# ---------------------------------------------------------------------------

@dataclass
class ContributionExplanation:
    """개별 샘플의 변수 기여도 요약."""

    sample_id: str
    prediction: float
    base_value: float
    contributions: list[dict]  # [{feature, value, shap, direction}]


@dataclass
class ThresholdExplanation:
    """변수별 위험 급증 구간."""

    feature: str
    threshold: float
    direction: str  # 'above' | 'below'
    risk_delta: float  # 평균 대비 risk 상승량


@dataclass
class InteractionExplanation:
    """상호작용 상위 쌍."""

    pairs: pd.DataFrame  # columns: feature_a, feature_b, mean_abs_interaction


@dataclass
class BatchExplanation:
    """한 배치 전체 설명 패키지 — 대시보드·리포트에 그대로 전달."""

    contribution: ContributionExplanation
    thresholds: list[ThresholdExplanation] = field(default_factory=list)
    interactions: InteractionExplanation | None = None


# ---------------------------------------------------------------------------
# 생성 유틸
# ---------------------------------------------------------------------------

def build_contribution(
    bundle: ShapBundle,
    sample_idx: int,
    *,
    prediction: float,
    sample_id: str,
    k: int = 5,
) -> ContributionExplanation:
    top = top_contributors(bundle, sample_idx, k=k)
    contributions = []
    for _, row in top.iterrows():
        contributions.append(
            {
                "feature": row["feature"],
                "value": float(row["value"]),
                "shap": float(row["shap"]),
                "direction": "증가" if row["shap"] > 0 else "감소",
            }
        )
    return ContributionExplanation(
        sample_id=sample_id,
        prediction=prediction,
        base_value=bundle.base_value,
        contributions=contributions,
    )


def infer_thresholds(
    bundle: ShapBundle,
    *,
    features: list[str],
    bins: int = 20,
    min_shap: float = 0.1,
) -> list[ThresholdExplanation]:
    """SHAP 기반 단순 임계 구간 추정.

    변수 값 구간별 평균 SHAP 이 `min_shap` 을 처음 초과하는 경계를 반환.
    """
    out: list[ThresholdExplanation] = []
    df_X = bundle.X
    for feat in features:
        if feat not in df_X.columns:
            continue
        idx = bundle.feature_names.index(feat)
        vals = df_X[feat].to_numpy()
        shaps = bundle.values[:, idx]
        try:
            qs = pd.qcut(vals, q=bins, duplicates="drop")
        except ValueError:
            continue
        grouped = pd.Series(shaps).groupby(qs, observed=False).mean()
        # 위험 증가 방향
        for interval, m in grouped.items():
            if m > min_shap:
                out.append(
                    ThresholdExplanation(
                        feature=feat,
                        threshold=float(interval.left),
                        direction="above",
                        risk_delta=float(m),
                    )
                )
                break
        # 위험 감소 방향 (보호 효과)
        for interval, m in grouped.items():
            if m < -min_shap:
                out.append(
                    ThresholdExplanation(
                        feature=feat,
                        threshold=float(interval.right),
                        direction="below",
                        risk_delta=float(m),
                    )
                )
                break
    return out


def build_interactions(
    inter_values: np.ndarray,
    feature_names: list[str],
    *,
    k: int = 5,
) -> InteractionExplanation:
    return InteractionExplanation(pairs=top_interactions(inter_values, feature_names, k=k))
