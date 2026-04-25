"""현장용 자연어 설명 생성 — WHAT → ACTION → WHY 계층.

Phase 11 (Revision 2) 재설계 사양
---------------------------------
경영진/현장 엔지니어가 SHAP 숫자 없이 상황을 이해할 수 있도록
**정상 범위 대비 평문 + ₩ 영향 + 권장 조치** 를 우선 제공한다.

함수 계층
~~~~~~~~~
- ``what_happened_bullets``       : SHAP 숫자 없이 정상 범위 대비 평문 (WHAT)
- ``executive_summary``           : 3-4 문장 요약 + ₩ 기대 손실
- ``recommended_actions``         : 상위 2-3 조치 카드 (₩ 절감 추정 포함)
- ``technical_details_sentences`` : 분석가용 SHAP 기반 상세 (WHY, expander 안)

기존 함수 ``contribution_to_sentences`` 는 ``technical_details_sentences``
의 별칭으로 보존한다.
"""

from __future__ import annotations

import math

from career_kia.xai.business_impact import (
    BusinessAssumptions,
    action_recommendation,
    format_krw,
    translate_batch_risk_to_krw,
)
from career_kia.xai.explanation_templates import (
    BatchExplanation,
    ContributionExplanation,
    InteractionExplanation,
    ThresholdExplanation,
)
from career_kia.xai.normal_ranges import describe_deviation, is_out_of_range

# ---------------------------------------------------------------------------
# 변수명 → 한국어 친화명
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "Air temperature [K]": "대기 온도",
    "Process temperature [K]": "공정 온도",
    "Rotational speed [rpm]": "회전 속도",
    "Torque [Nm]": "토크",
    "Tool wear [min]": "공구 마모 시간",
    "t_rms_mean": "진동 RMS 평균",
    "t_rms_max": "진동 RMS 최대",
    "t_kurtosis_mean": "진동 첨도 평균",
    "t_kurtosis_max": "진동 첨도 최대",
    "t_crest_factor_mean": "크레스트 팩터 평균",
    "t_crest_factor_max": "크레스트 팩터 최대",
    "t_impulse_factor_max": "임펄스 팩터 최대",
    "f_env_BPFI_mean": "BPFI(내륜) 포락선 평균",
    "f_env_BPFI_max": "BPFI(내륜) 포락선 최대",
    "f_env_BPFO_mean": "BPFO(외륜) 포락선 평균",
    "f_env_BPFO_max": "BPFO(외륜) 포락선 최대",
    "f_env_BSF_mean": "BSF(볼) 포락선 평균",
    "f_env_BSF_max": "BSF(볼) 포락선 최대",
    "f_env_FTF_mean": "FTF(케이지) 포락선 평균",
    "f_env_FTF_max": "FTF(케이지) 포락선 최대",
    "f_spec_entropy_mean": "스펙트럴 엔트로피",
    "f_spec_centroid_mean": "스펙트럴 중심",
}

FEATURE_UNITS: dict[str, str] = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "분",
}


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _unit(feature: str) -> str:
    return FEATURE_UNITS.get(feature, "")


def _fmt_value(value: float, feature: str) -> str:
    unit = _unit(feature)
    if abs(value) >= 100:
        return f"{value:.0f}{unit}"
    if abs(value) >= 1:
        return f"{value:.1f}{unit}"
    return f"{value:.3f}{unit}"


def _risk_level(prob: float) -> str:
    if prob >= 0.66:
        return "🔴 높음"
    if prob >= 0.33:
        return "🟡 보통"
    return "🟢 낮음"


# ---------------------------------------------------------------------------
# WHAT: 정상 범위 대비 평문 (SHAP 숫자 노출 금지)
# ---------------------------------------------------------------------------

def what_happened_bullets(
    contribution: ContributionExplanation,
    *,
    top_k: int = 3,
) -> list[str]:
    """리스크를 끌어올린 상위 변수들을 정상 범위 대비 평문으로 서술.

    SHAP 값/부호/소수점 표현은 본문에 일절 포함하지 않는다.
    """
    pos_contribs = [c for c in contribution.contributions if c["shap"] > 0]
    if not pos_contribs:
        return ["현재 비정상 신호가 감지되지 않았습니다."]

    bullets: list[str] = []
    for c in pos_contribs[:top_k]:
        feat, val = c["feature"], float(c["value"])
        if is_out_of_range(feat, val):
            bullets.append(describe_deviation(feat, val))
        else:
            bullets.append(
                f"{_label(feat)} = {_fmt_value(val, feat)} 가 평소와 다른 패턴"
            )
    return bullets


# ---------------------------------------------------------------------------
# Executive summary: 3-4 문장 + ₩
# ---------------------------------------------------------------------------

def executive_summary(
    contribution: ContributionExplanation,
    assumptions: BusinessAssumptions,
) -> str:
    """경영진용 3-4 문장 요약 (₩ 기대 손실 포함, SHAP 숫자 없음)."""
    risk = max(0.0, min(1.0, contribution.prediction))
    risk_pct = risk * 100
    expected_loss = translate_batch_risk_to_krw(risk, assumptions)
    level = _risk_level(risk)

    lines = [
        f"배치 **{contribution.sample_id}** 의 고장 위험은 {level} ({risk_pct:.0f}%) 입니다.",
        f"이 배치에서 발생할 수 있는 기대 손실은 약 **{format_krw(expected_loss)}** 입니다.",
    ]
    bullets = what_happened_bullets(contribution, top_k=2)
    if bullets and bullets[0] != "현재 비정상 신호가 감지되지 않았습니다.":
        lines.append("주된 원인: " + " · ".join(bullets))
    else:
        lines.append("뚜렷한 위험 신호는 관찰되지 않습니다.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 권장 조치 카드
# ---------------------------------------------------------------------------

def recommended_actions(
    contribution: ContributionExplanation,
    assumptions: BusinessAssumptions,
    *,
    top_k: int = 3,
) -> list[dict]:
    """상위 위험 변수 → 조치 카드 매핑.

    Returns
    -------
    list of dict with keys: feature, label, description, estimated_cost_krw,
    estimated_cost_text, expected_loss_text.
    """
    expected_loss = translate_batch_risk_to_krw(contribution.prediction, assumptions)
    expected_loss_text = format_krw(expected_loss)

    cards: list[dict] = []
    seen_labels: set[str] = set()
    for c in contribution.contributions:
        if c["shap"] <= 0:
            continue
        card = action_recommendation(c["feature"], assumptions)
        if card is None:
            continue
        if card["label"] in seen_labels:
            continue
        seen_labels.add(card["label"])
        card = {**card, "expected_loss_text": expected_loss_text}
        cards.append(card)
        if len(cards) >= top_k:
            break

    if not cards:
        cards.append(
            {
                "feature": "",
                "label": "추가 모니터링",
                "description": "직접 매칭되는 조치가 없습니다. 다음 배치까지 관찰을 유지하세요.",
                "estimated_cost_krw": 0,
                "estimated_cost_text": "비용 없음",
                "expected_loss_text": expected_loss_text,
            }
        )
    return cards


# ---------------------------------------------------------------------------
# 분석가용 (WHY) — 기존 SHAP 기반 상세
# ---------------------------------------------------------------------------

def technical_details_sentences(exp: ContributionExplanation) -> list[str]:
    """SHAP 기반 상세 — expander 내부에서만 사용."""
    risk_pct = max(0.0, min(1.0, exp.prediction)) * 100
    base_pct = max(0.0, min(1.0, _sigmoid(exp.base_value))) * 100
    head = (
        f"배치 {exp.sample_id} 의 예측 고장 확률은 {risk_pct:.1f}% "
        f"(기준 확률 {base_pct:.1f}% 대비)."
    )

    bullets: list[str] = [head]
    if not exp.contributions:
        bullets.append("유의미한 기여 변수가 감지되지 않았습니다.")
        return bullets

    pos = [c for c in exp.contributions if c["shap"] > 0]
    neg = [c for c in exp.contributions if c["shap"] < 0]

    if pos:
        bullets.append("리스크를 **증가**시킨 주요 변수:")
        for c in pos[:3]:
            bullets.append(
                f"  - {_label(c['feature'])} = {_fmt_value(c['value'], c['feature'])} "
                f"(기여도 {c['shap']:+.3f})"
            )
    if neg:
        bullets.append("리스크를 **감소**시킨 주요 변수:")
        for c in neg[:3]:
            bullets.append(
                f"  - {_label(c['feature'])} = {_fmt_value(c['value'], c['feature'])} "
                f"(기여도 {c['shap']:+.3f})"
            )
    return bullets


# 하위 호환 — 기존 노트북/리포트 호출처
contribution_to_sentences = technical_details_sentences


def thresholds_to_sentences(thresholds: list[ThresholdExplanation]) -> list[str]:
    if not thresholds:
        return []
    out = ["주요 임계 구간:"]
    for t in thresholds[:5]:
        sign = "초과" if t.direction == "above" else "미만"
        direction_word = "위험 상승" if t.risk_delta > 0 else "위험 완화"
        out.append(
            f"  - {_label(t.feature)} 이(가) {_fmt_value(t.threshold, t.feature)} {sign} 구간에서 "
            f"{direction_word} 관찰 (SHAP 평균 {t.risk_delta:+.3f})"
        )
    return out


def interactions_to_sentences(inter: InteractionExplanation | None) -> list[str]:
    if inter is None or inter.pairs.empty:
        return []
    out = ["상위 상호작용 효과:"]
    for _, row in inter.pairs.head(3).iterrows():
        out.append(
            f"  - {_label(row['feature_a'])} ↔ {_label(row['feature_b'])} "
            f"(상호작용 강도 {row['mean_abs_interaction']:.3f})"
        )
    return out


def batch_explanation_to_paragraph(batch_exp: BatchExplanation) -> str:
    """기존 호환 — 분석가 모드에서 BatchExplanation 전체 상세 출력."""
    lines: list[str] = []
    lines.extend(technical_details_sentences(batch_exp.contribution))
    lines.append("")
    lines.extend(thresholds_to_sentences(batch_exp.thresholds))
    lines.append("")
    lines.extend(interactions_to_sentences(batch_exp.interactions))
    return "\n".join(lines)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
