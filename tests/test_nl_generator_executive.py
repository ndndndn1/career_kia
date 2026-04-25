"""nl_generator Executive 레이어 테스트.

핵심 계약
~~~~~~~~~
1. WHAT 본문에 SHAP 숫자/부호/소수점 표현이 노출되지 않는다.
2. executive_summary 에 ₩ 또는 '원' 단위가 포함된다.
3. recommended_actions 는 양의 SHAP 변수만 매핑하고 라벨 중복을 제거한다.
"""

from __future__ import annotations

import textwrap

import pytest

from career_kia.xai import nl_generator
from career_kia.xai.business_impact import load_assumptions
from career_kia.xai.explanation_templates import ContributionExplanation


@pytest.fixture
def assumptions(tmp_path):
    p = tmp_path / "business.yaml"
    p.write_text(
        textwrap.dedent(
            """
            revenue_per_hour_krw: 12_000_000
            batches_per_hour: 60
            defect_cost_per_batch_krw: 250_000
            downtime_per_failure_hours: 4
            downtime_cost_per_hour_krw: 8_500_000
            tool_replacement_cost_krw: 180_000
            bearing_replacement_cost_krw: 1_200_000
            operating_hours_per_year: 6_000
            action_map:
              "Tool wear [min]":
                label: "공구 교체"
                description: "교체 권장"
                cost_category: tool_replacement
              "Torque [Nm]":
                label: "토크 재조정"
                description: "재조정"
                cost_category: setup_adjustment
              "f_env_BPFI_max":
                label: "베어링 내륜 점검"
                description: "베어링 검토"
                cost_category: bearing_replacement
            """
        ),
        encoding="utf-8",
    )
    return load_assumptions(p)


def _make_contrib(risk: float = 0.72) -> ContributionExplanation:
    return ContributionExplanation(
        sample_id="B000123",
        prediction=risk,
        base_value=-1.0,
        contributions=[
            {"feature": "Tool wear [min]", "value": 200.0, "shap": 0.42, "direction": "증가"},
            {"feature": "Torque [Nm]", "value": 75.0, "shap": 0.21, "direction": "증가"},
            {"feature": "f_env_BPFI_max", "value": 0.12, "shap": 0.18, "direction": "증가"},
            {"feature": "Process temperature [K]", "value": 308.0, "shap": -0.05, "direction": "감소"},
        ],
    )


def test_what_happened_bullets_describe_deviation_and_share():
    """본문은 (1) 정상 범위 대비 평문 + (2) 위험 기여 % 둘 다 포함한다."""
    contrib = _make_contrib()
    bullets = nl_generator.what_happened_bullets(contrib)
    text = " ".join(bullets)
    # 정상 범위 대비 평문 (현장 엔지니어 납득용)
    assert "공구 마모" in text
    assert "초과" in text or "범위" in text
    # 정량 영향 — 경영진/현장이 우선순위를 가늠할 수 있도록 % 노출
    assert "%" in text
    assert "위험 기여" in text
    # 합산이 100% 를 초과하지 않는지 (top_k=3 기준)
    import re
    shares = [int(x) for x in re.findall(r"위험 기여 \+(\d+)%", text)]
    assert sum(shares) <= 100


def test_what_happened_bullets_no_positive_returns_default():
    contrib = ContributionExplanation(
        sample_id="B0",
        prediction=0.05,
        base_value=-2.0,
        contributions=[
            {"feature": "Tool wear [min]", "value": 30.0, "shap": -0.1, "direction": "감소"},
        ],
    )
    bullets = nl_generator.what_happened_bullets(contrib)
    assert any("비정상" in b or "감지되지" in b for b in bullets)


def test_executive_summary_contains_quantitative_signals(assumptions):
    contrib = _make_contrib()
    text = nl_generator.executive_summary(contrib, assumptions)
    # ₩ 단위
    assert "원" in text or "₩" in text
    assert "B000123" in text
    # 위험 표현
    assert any(token in text for token in ["높음", "보통", "낮음"])
    # 확률(%) 가 명시적으로 노출 (경영진이 숫자를 볼 수 있도록)
    assert "%" in text
    # 기준 확률 대비 비교가 포함되어야 함
    assert "기준 확률" in text


def test_recommended_actions_maps_positive_only_and_dedupes(assumptions):
    contrib = _make_contrib()
    cards = nl_generator.recommended_actions(contrib, assumptions, top_k=3)
    labels = [c["label"] for c in cards]
    assert "공구 교체" in labels
    assert "토크 재조정" in labels
    assert "베어링 내륜 점검" in labels
    # 라벨 중복 없음
    assert len(labels) == len(set(labels))
    # 모든 카드에 expected_loss_text 가 포함
    assert all("expected_loss_text" in c for c in cards)
    # 음의 SHAP (Process temperature) 는 제외
    assert all(c["feature"] != "Process temperature [K]" for c in cards if c["feature"])


def test_recommended_actions_no_match_falls_back_to_monitoring(assumptions):
    contrib = ContributionExplanation(
        sample_id="B0",
        prediction=0.4,
        base_value=-1.0,
        contributions=[
            {"feature": "completely_unknown_feat", "value": 1.0, "shap": 0.5, "direction": "증가"},
        ],
    )
    cards = nl_generator.recommended_actions(contrib, assumptions)
    assert len(cards) == 1
    assert cards[0]["label"] == "추가 모니터링"


def test_technical_details_still_uses_shap_numbers():
    """분석가용 상세는 SHAP 값 노출 OK (계약 유지)."""
    contrib = _make_contrib()
    lines = nl_generator.technical_details_sentences(contrib)
    text = " ".join(lines)
    assert "기여도" in text
    assert "+0.420" in text or "+0.42" in text


def test_contribution_to_sentences_alias_preserved():
    """노트북/리포트가 사용하는 기존 함수명이 별칭으로 유지되어야 한다."""
    assert nl_generator.contribution_to_sentences is nl_generator.technical_details_sentences
