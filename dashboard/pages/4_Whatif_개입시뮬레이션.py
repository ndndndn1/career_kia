"""Page 4 — 조건 변경 시뮬레이션 (DoWhy 인과추론 기반).

Phase 11 (Revision 2)
- "dose-response" 라벨 → "조건 변경 시뮬레이션"
- ATE %p 옆에 ₩ 연 절감 추정 병기
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from career_kia.config import PROJECT_ROOT
from career_kia.xai import business_impact
from dashboard._helpers import render_confidence_badge


st.set_page_config(page_title="조건 변경 시뮬", page_icon="🧪", layout="wide")
st.title("🧪 What-if · 공정 조건 변경 시뮬레이션")
st.caption(
    "공정 변수를 바꾸면 고장 위험과 연 절감이 어떻게 달라지는지 인과 추정 결과로 시뮬레이션합니다."
)


ARTIFACTS = PROJECT_ROOT / "causal_artifacts"


@st.cache_data
def _load_summary():
    return json.loads((ARTIFACTS / "ate_summary.json").read_text(encoding="utf-8"))


@st.cache_data
def _load_dose_response(treatment: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / f"doseresp_{treatment}.csv")


@st.cache_resource
def _load_assumptions():
    return business_impact.load_assumptions()


try:
    summary = _load_summary()
    assumptions = _load_assumptions()
except FileNotFoundError:
    st.error("`make causal` 를 먼저 실행하여 `causal_artifacts/` 를 생성하세요.")
    st.stop()

treatment = st.selectbox(
    "개입 변수",
    list(summary.keys()),
    format_func=lambda x: {
        "Tool_wear": "공구 마모 시간 (분)",
        "Torque": "토크 (Nm)",
        "Rotational_speed": "회전 속도 (rpm)",
    }.get(x, x),
)

dr = _load_dose_response(treatment)
baseline = float(summary[treatment]["control_value"])

target = st.slider(
    "개입 목표값",
    float(dr["treatment_value"].min()),
    float(dr["treatment_value"].max()),
    baseline,
)

# 선형 보간으로 곡선 값 추출
ate_at_target = float(np.interp(target, dr["treatment_value"], dr["ate"]))
ate_at_baseline = float(np.interp(baseline, dr["treatment_value"], dr["ate"]))
delta = ate_at_target - ate_at_baseline
annual_savings = business_impact.translate_ate_to_krw(delta, assumptions)

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("현재 운영값", f"{baseline:.1f}")
c2.metric("개입 목표", f"{target:.1f}")
c3.metric("고장 확률 변화", f"{delta*100:+.2f}%p")
c4.metric(
    "연 기대 절감",
    business_impact.format_krw(annual_savings),
    delta="절감" if annual_savings > 0 else "증가",
)

# 인과추정 신뢰도 — DoWhy refutation 결과 기반
ref = summary[treatment].get("refutation", {})
ate_est = float(summary[treatment]["ate"])

def _stable(name: str, *, rel_tol: float = 0.20) -> bool:
    val = ref.get(name)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if abs(ate_est) < 1e-9:
        return abs(val) < 1e-3
    return abs(val - ate_est) / abs(ate_est) <= rel_tol

passes = [n for n in ("random_common_cause", "data_subset_refuter") if _stable(n)]
total = 2
causal_conf = len(passes) / total
ref_str = " · ".join(
    f"{k}={ref[k]:+.4f}" for k in ("random_common_cause", "data_subset_refuter")
    if k in ref and not (isinstance(ref[k], float) and np.isnan(ref[k]))
)
render_confidence_badge(
    max(causal_conf, 0.5),
    prefix=f"인과 추정 신뢰도 ({len(passes)}/{total} 반박 통과)",
)
st.caption(f"반박 검정 결과: {ref_str or '값 없음'}  ·  ATE 자체 = {ate_est:+.4f}")

# ---------------------------------------------------------------------------
# 곡선 + 우측 ₩ 보조축
# ---------------------------------------------------------------------------
krw_per_batch = -dr["ate"] * assumptions.cost_per_failure_krw
annual_savings_curve = krw_per_batch * assumptions.batches_per_year

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dr["treatment_value"], y=dr["ate"] * 100,
        mode="lines+markers", name="고장 확률 변화 (%p)",
        line=dict(color="#3498db", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=dr["treatment_value"], y=annual_savings_curve / 1e8,
        mode="lines", name="연 기대 절감 (억 원)",
        line=dict(color="#27ae60", width=2, dash="dot"),
        yaxis="y2",
    )
)
fig.add_vline(x=baseline, line_dash="dot", line_color="gray", annotation_text="현재")
fig.add_vline(x=target, line_dash="dash", line_color="red", annotation_text="개입")
fig.update_layout(
    xaxis_title=treatment,
    yaxis=dict(title="고장 확률 변화 (%p)"),
    yaxis2=dict(title="연 기대 절감 (억 원)", overlaying="y", side="right"),
    height=440,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 해석
# ---------------------------------------------------------------------------
direction = "감소" if delta < 0 else "증가"
saving_word = "절감" if annual_savings > 0 else "추가 손실"
st.markdown(
    f"""
    #### 해석
    - **{treatment}** 를 현재값 **{baseline:.1f}** 에서 **{target:.1f}** 로 변경하면
      예상 고장 확률이 약 **{abs(delta)*100:.2f}%p {direction}** 합니다.
    - 라인 가정치 기준 연 약 **{business_impact.format_krw(abs(annual_savings))} {saving_word}** 으로 환산됩니다.
    - 가정치는 `configs/business.yaml` 의 플레이스홀더로, 실제 라인 값으로 교체하면 그대로 갱신됩니다.
    """
)

with st.expander("📊 인과추론 방법 (분석가용)"):
    st.markdown(
        """
        - DoWhy 4 단계 워크플로 (model → identify → estimate → refute) 적용
        - 식별: 도메인 DAG 의 backdoor 경로를 통제 (`src/career_kia/causal/dag.py`)
        - 추정: `backdoor.linear_regression` (선형 효과)
        - 반박: `random_common_cause`, `data_subset_refuter` 통과 추정치만 사용
        - 라벨 'ATE %p' 는 평균 처치 효과(Average Treatment Effect, percentage points) 를 의미
        """
    )
