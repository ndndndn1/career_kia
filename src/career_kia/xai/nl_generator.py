"""현장용 자연어 설명 생성.

SHAP 기반 ContributionExplanation / ThresholdExplanation / InteractionExplanation 을
현장 엔지니어·경영진이 읽기 쉬운 **한국어 문장**으로 변환한다. 규칙 기반이라
외부 LLM 의존 없이 재현 가능하며, 배포 환경에서도 안전하게 동작한다.
"""

from __future__ import annotations

from career_kia.xai.explanation_templates import (
    BatchExplanation,
    ContributionExplanation,
    InteractionExplanation,
    ThresholdExplanation,
)

# ---------------------------------------------------------------------------
# 변수명 → 한국어 친화명 매핑
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
    "f_env_BPFI_mean": "BPFI(내륜 결함 주파수) 포락선 평균",
    "f_env_BPFI_max": "BPFI(내륜 결함 주파수) 포락선 최대",
    "f_env_BPFO_mean": "BPFO(외륜 결함 주파수) 포락선 평균",
    "f_env_BPFO_max": "BPFO(외륜 결함 주파수) 포락선 최대",
    "f_env_BSF_mean": "BSF(볼 자전 주파수) 포락선 평균",
    "f_env_BSF_max": "BSF(볼 자전 주파수) 포락선 최대",
    "f_env_FTF_mean": "FTF(케이지 주파수) 포락선 평균",
    "f_env_FTF_max": "FTF(케이지 주파수) 포락선 최대",
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


# ---------------------------------------------------------------------------
# 문장 생성
# ---------------------------------------------------------------------------

def contribution_to_sentences(exp: ContributionExplanation) -> list[str]:
    """개별 샘플 기여도 → 한국어 문장 리스트."""
    risk_pct = max(0.0, min(1.0, exp.prediction)) * 100
    head = (
        f"배치 {exp.sample_id} 의 예측 고장 확률은 {risk_pct:.1f}% 입니다 "
        f"(기준 확률 {max(0.0, min(1.0, _sigmoid(exp.base_value)))*100:.1f}% 대비)."
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
    """BatchExplanation 전체를 한 문단 스토리로 구성."""
    lines: list[str] = []
    lines.extend(contribution_to_sentences(batch_exp.contribution))
    lines.append("")
    lines.extend(thresholds_to_sentences(batch_exp.thresholds))
    lines.append("")
    lines.extend(interactions_to_sentences(batch_exp.interactions))
    return "\n".join(lines)


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))
