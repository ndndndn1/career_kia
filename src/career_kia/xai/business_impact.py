"""비즈니스 임팩트 번역 레이어.

SHAP/ATE 같은 통계량을 경영진이 해석 가능한 단위(₩/시간/건수)로
변환한다. 가정치는 ``configs/business.yaml`` 의 플레이스홀더 값에서
로드하며, 실제 라인 도입 시 라인별 값으로 교체한다.

핵심 공식
---------
- 배치 기대 손실 = P(고장) × (불량 비용 + 다운타임 손실)
- ATE → 연 절감 = ΔP(고장) × 연간 배치수 × 배치당 손실
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from career_kia.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "business.yaml"


@dataclass
class BusinessAssumptions:
    """플레이스홀더 가정치 컨테이너."""

    revenue_per_hour_krw: float
    batches_per_hour: float
    defect_cost_per_batch_krw: float
    downtime_per_failure_hours: float
    downtime_cost_per_hour_krw: float
    tool_replacement_cost_krw: float
    bearing_replacement_cost_krw: float
    operating_hours_per_year: float
    action_map: dict[str, dict[str, str]]

    @property
    def batches_per_year(self) -> float:
        return self.batches_per_hour * self.operating_hours_per_year

    @property
    def cost_per_failure_krw(self) -> float:
        """고장 1건당 총 손실 = 불량 비용 + 다운타임 손실."""
        return (
            self.defect_cost_per_batch_krw
            + self.downtime_per_failure_hours * self.downtime_cost_per_hour_krw
        )


def load_assumptions(path: Path | None = None) -> BusinessAssumptions:
    """``business.yaml`` 로드."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return BusinessAssumptions(
        revenue_per_hour_krw=float(raw["revenue_per_hour_krw"]),
        batches_per_hour=float(raw["batches_per_hour"]),
        defect_cost_per_batch_krw=float(raw["defect_cost_per_batch_krw"]),
        downtime_per_failure_hours=float(raw["downtime_per_failure_hours"]),
        downtime_cost_per_hour_krw=float(raw["downtime_cost_per_hour_krw"]),
        tool_replacement_cost_krw=float(raw["tool_replacement_cost_krw"]),
        bearing_replacement_cost_krw=float(raw["bearing_replacement_cost_krw"]),
        operating_hours_per_year=float(raw["operating_hours_per_year"]),
        action_map=raw.get("action_map", {}),
    )


def translate_batch_risk_to_krw(
    risk: float, assumptions: BusinessAssumptions
) -> float:
    """단일 배치 예측 리스크 → 기대 손실 (₩)."""
    risk = max(0.0, min(1.0, float(risk)))
    return risk * assumptions.cost_per_failure_krw


def translate_ate_to_krw(
    ate: float,
    assumptions: BusinessAssumptions,
    *,
    coverage: float = 1.0,
) -> float:
    """ATE(고장 확률 변화량) → 연 기대 절감 (₩).

    Parameters
    ----------
    ate
        개입에 따른 P(고장) 변화량. 음수일수록 더 많이 절감.
    coverage
        해당 개입을 적용하는 배치 비율 (0~1).
    """
    saved_per_batch = -ate * assumptions.cost_per_failure_krw
    return saved_per_batch * assumptions.batches_per_year * coverage


def format_krw(value: float) -> str:
    """원/만/억 단위 친화 포맷."""
    abs_v = abs(value)
    sign = "-" if value < 0 else ""
    if abs_v >= 1e8:
        return f"{sign}{abs_v / 1e8:.1f}억 원"
    if abs_v >= 1e4:
        return f"{sign}{abs_v / 1e4:.0f}만 원"
    return f"{sign}{abs_v:,.0f}원"


def action_recommendation(
    feature: str, assumptions: BusinessAssumptions
) -> dict[str, str] | None:
    """피처명 → 권장 조치 카드.

    Returns
    -------
    {"label", "description", "estimated_cost_krw"} 형식 dict.
    매핑이 없으면 None.
    """
    spec = assumptions.action_map.get(feature)
    if spec is None:
        return None
    cost_category = spec.get("cost_category", "")
    cost_lookup = {
        "tool_replacement": assumptions.tool_replacement_cost_krw,
        "bearing_replacement": assumptions.bearing_replacement_cost_krw,
        "setup_adjustment": 0.0,
        "maintenance": 0.0,
    }
    cost = cost_lookup.get(cost_category, 0.0)
    return {
        "feature": feature,
        "label": spec["label"],
        "description": spec["description"],
        "estimated_cost_krw": cost,
        "estimated_cost_text": format_krw(cost) if cost > 0 else "직접 비용 미미",
    }
