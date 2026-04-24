"""Page 4 — What-if 개입 시뮬레이션 (DoWhy ATE).

슬라이더로 공정 조건을 바꾸면 사전 계산된 dose-response 곡선 위에서
예상 고장 확률 변화를 보여준다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from career_kia.config import PROJECT_ROOT


st.set_page_config(page_title="What-if 개입 시뮬", page_icon="🧪", layout="wide")
st.title("🧪 What-if · 공정 개입 시뮬레이션")
st.caption("DoWhy 로 추정한 dose-response 곡선 기반. `make causal` 실행 필요.")


ARTIFACTS = PROJECT_ROOT / "causal_artifacts"


@st.cache_data
def _load_summary():
    return json.loads((ARTIFACTS / "ate_summary.json").read_text(encoding="utf-8"))


@st.cache_data
def _load_dose_response(treatment: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / f"doseresp_{treatment}.csv")


try:
    summary = _load_summary()
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

# 선형 보간으로 dose-response 값 추출
ate_at_target = float(np.interp(target, dr["treatment_value"], dr["ate"]))
ate_at_baseline = float(np.interp(baseline, dr["treatment_value"], dr["ate"]))
delta = ate_at_target - ate_at_baseline

c1, c2, c3 = st.columns(3)
c1.metric("기준값", f"{baseline:.1f}")
c2.metric("개입 목표", f"{target:.1f}")
c3.metric("예상 고장 확률 변화", f"{delta*100:+.2f}%p")

# 그래프
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dr["treatment_value"],
        y=dr["ate"] * 100,
        mode="lines+markers",
        name="Dose-response",
    )
)
fig.add_vline(x=baseline, line_dash="dot", line_color="gray", annotation_text="기준")
fig.add_vline(x=target, line_dash="dash", line_color="red", annotation_text="개입")
fig.update_layout(
    xaxis_title=treatment,
    yaxis_title="ATE (%p, 기준 대비)",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
    #### 해석
    - **{treatment}** 를 기준값 **{baseline:.1f}** 에서 **{target:.1f}** 로 변경할 때,
      모델이 추정한 평균 처치 효과(ATE) 는 **{delta*100:+.2f} %p** 입니다.
    - 이 값은 관측 데이터 기반 추정이며, 교란변수는 도메인 DAG 로 통제되었습니다
      (`src/career_kia/causal/dag.py` 참조).
    - `make causal` 실행 시 `random_common_cause`, `data_subset_refuter` refutation 을 통과한 추정치만 사용됩니다.
    """
)
