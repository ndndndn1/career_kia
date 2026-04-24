"""개입 효과(ATE) 추정 — DoWhy 4 단계 워크플로.

사용 예::

    est = estimate_ate(
        df, treatment='Tool_wear', outcome='Machine_failure',
        treatment_value=200, control_value=50,
    )
    refute = refute_estimate(est)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dowhy import CausalModel

from career_kia.causal.dag import NODE_TO_COLUMN, get_default_graph

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    treatment: str
    outcome: str
    method: str
    estimate: float
    control_value: float
    treatment_value: float
    p_value: float | None = None
    refutation: dict[str, float] | None = None


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DoWhy 는 공백·대괄호 없는 컬럼명을 선호."""
    col_map = {v: k for k, v in NODE_TO_COLUMN.items()}
    cols_needed = [c for c in col_map if c in df.columns]
    sub = df[cols_needed].rename(columns=col_map).copy()
    # Type 을 ordinal 로 변환 (L=0, M=1, H=2)
    if "Type" in sub.columns:
        mapping = {"L": 0, "M": 1, "H": 2}
        sub["Type"] = sub["Type"].map(mapping).fillna(0).astype(int)
    return sub


def estimate_ate(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str = "Machine_failure",
    treatment_value: float,
    control_value: float,
    method: str = "backdoor.linear_regression",
    graph: str | None = None,
) -> tuple[InterventionResult, CausalModel, object]:
    """ATE 추정.

    Parameters
    ----------
    treatment, outcome
        DAG 노드명.
    treatment_value, control_value
        개입값 vs 기준값. 연속변수의 반사실 비교 (예: Tool_wear = 200 vs 50).
    method
        DoWhy estimation method (예: 'backdoor.linear_regression',
        'backdoor.propensity_score_matching', 'backdoor.propensity_score_weighting').
    """
    g = graph or get_default_graph()
    data = _prepare_dataframe(df)

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=g,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name=method,
        control_value=control_value,
        treatment_value=treatment_value,
        target_units="ate",
    )
    result = InterventionResult(
        treatment=treatment,
        outcome=outcome,
        method=method,
        estimate=float(estimate.value),
        control_value=float(control_value),
        treatment_value=float(treatment_value),
    )
    logger.info(
        "ATE [%s → %s]: %.4f (method=%s, %.1f vs %.1f)",
        treatment, outcome, result.estimate, method, treatment_value, control_value,
    )
    return result, model, estimate


def refute_estimate(
    model: CausalModel,
    estimate,
    *,
    methods: tuple[str, ...] = (
        "random_common_cause",
        "placebo_treatment_refuter",
        "data_subset_refuter",
    ),
) -> dict[str, float]:
    """여러 refutation 검증을 한 번에 수행.

    - random_common_cause: 무의미한 공통 원인을 추가해도 효과가 유지되는지
    - placebo_treatment_refuter: 무작위 처치로 바꾸면 효과가 0 근처로 가는지
    - data_subset_refuter: 데이터 서브셋에서도 효과가 안정적인지
    """
    out: dict[str, float] = {}
    for m in methods:
        try:
            ref = model.refute_estimate(
                model.identify_effect(proceed_when_unidentifiable=True),
                estimate,
                method_name=m,
            )
            out[m] = float(ref.new_effect)
        except Exception as exc:  # noqa: BLE001
            logger.warning("refutation %s 실패: %s", m, exc)
            out[m] = float("nan")
    return out


def whatif_dose_response(
    df: pd.DataFrame,
    *,
    treatment: str,
    grid: np.ndarray,
    baseline: float,
    method: str = "backdoor.linear_regression",
) -> pd.DataFrame:
    """개입값을 그리드로 스윕해 dose-response 곡선 산출.

    대시보드의 What-if 슬라이더 뒷단에 사용된다.
    """
    rows = []
    for v in grid:
        res, _, _ = estimate_ate(
            df, treatment=treatment, treatment_value=float(v), control_value=baseline, method=method
        )
        rows.append({"treatment_value": v, "ate": res.estimate})
    return pd.DataFrame(rows)
