"""비즈니스 임팩트 번역 레이어 테스트."""

from __future__ import annotations

import textwrap

from career_kia.xai import business_impact


def _write_yaml(tmp_path, body: str):
    path = tmp_path / "business.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_load_assumptions_returns_dataclass(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        revenue_per_hour_krw: 10_000_000
        batches_per_hour: 60
        defect_cost_per_batch_krw: 200_000
        downtime_per_failure_hours: 3
        downtime_cost_per_hour_krw: 5_000_000
        tool_replacement_cost_krw: 100_000
        bearing_replacement_cost_krw: 1_000_000
        operating_hours_per_year: 6_000
        action_map:
          "Tool wear [min]":
            label: "공구 교체"
            description: "교체 권장"
            cost_category: tool_replacement
        """,
    )
    a = business_impact.load_assumptions(path)
    assert a.revenue_per_hour_krw == 10_000_000
    assert a.batches_per_year == 60 * 6_000
    # cost_per_failure_krw = 200_000 + 3*5_000_000 = 15_200_000
    assert a.cost_per_failure_krw == 15_200_000


def test_translate_batch_risk_to_krw_clamps_and_scales(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        revenue_per_hour_krw: 1
        batches_per_hour: 1
        defect_cost_per_batch_krw: 100
        downtime_per_failure_hours: 1
        downtime_cost_per_hour_krw: 900
        tool_replacement_cost_krw: 0
        bearing_replacement_cost_krw: 0
        operating_hours_per_year: 1
        action_map: {}
        """,
    )
    a = business_impact.load_assumptions(path)
    # cost_per_failure = 100 + 900 = 1000
    assert business_impact.translate_batch_risk_to_krw(0.5, a) == 500
    # 0~1 범위 밖은 클램핑
    assert business_impact.translate_batch_risk_to_krw(-0.1, a) == 0
    assert business_impact.translate_batch_risk_to_krw(2.0, a) == 1000


def test_translate_ate_to_krw_negative_ate_means_savings(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        revenue_per_hour_krw: 1
        batches_per_hour: 10
        defect_cost_per_batch_krw: 100
        downtime_per_failure_hours: 0
        downtime_cost_per_hour_krw: 0
        tool_replacement_cost_krw: 0
        bearing_replacement_cost_krw: 0
        operating_hours_per_year: 100
        action_map: {}
        """,
    )
    a = business_impact.load_assumptions(path)
    # cost_per_failure = 100, batches_per_year = 1_000
    # ate = -0.05 → 배치당 5원 절감 × 1000 = 5_000원
    saved = business_impact.translate_ate_to_krw(-0.05, a)
    assert saved == 5_000
    # 양의 ATE 는 음의 절감
    assert business_impact.translate_ate_to_krw(0.05, a) == -5_000
    # coverage 적용
    assert business_impact.translate_ate_to_krw(-0.05, a, coverage=0.5) == 2_500


def test_format_krw_units():
    assert business_impact.format_krw(500) == "500원"
    assert business_impact.format_krw(50_000) == "5만 원"
    assert business_impact.format_krw(5_000_000_000) == "50.0억 원"
    assert business_impact.format_krw(-200_000_000) == "-2.0억 원"


def test_action_recommendation_returns_card(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        revenue_per_hour_krw: 1
        batches_per_hour: 1
        defect_cost_per_batch_krw: 1
        downtime_per_failure_hours: 1
        downtime_cost_per_hour_krw: 1
        tool_replacement_cost_krw: 180_000
        bearing_replacement_cost_krw: 1_200_000
        operating_hours_per_year: 1
        action_map:
          "Tool wear [min]":
            label: "공구 교체"
            description: "교체 권장"
            cost_category: tool_replacement
          "Torque [Nm]":
            label: "토크 재조정"
            description: "재조정 권장"
            cost_category: setup_adjustment
        """,
    )
    a = business_impact.load_assumptions(path)
    card = business_impact.action_recommendation("Tool wear [min]", a)
    assert card["label"] == "공구 교체"
    assert card["estimated_cost_krw"] == 180_000
    assert "만" in card["estimated_cost_text"]

    card2 = business_impact.action_recommendation("Torque [Nm]", a)
    assert card2["estimated_cost_krw"] == 0
    assert card2["estimated_cost_text"] == "직접 비용 미미"

    assert business_impact.action_recommendation("UNKNOWN_FEATURE", a) is None
