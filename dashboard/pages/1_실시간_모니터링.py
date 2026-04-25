"""Page 1 — 실시간 모니터링.

배치 스트림을 시뮬레이션하여 시계열 센서·공정 신호와 예측 리스크를 표시.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from career_kia.config import PROCESSED_DIR
from career_kia.models.train import load_feature_matrix
from dashboard._helpers import confidence_from_proba


st.set_page_config(page_title="실시간 모니터링", page_icon="📈", layout="wide")
st.title("📈 실시간 모니터링")


@st.cache_data
def _load_all():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, y, _ = load_feature_matrix()
    proba = model.predict_proba(X)[:, 1]
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    feat["risk_score"] = proba[feat.index]
    feat["confidence"] = confidence_from_proba(feat["risk_score"].values)
    return feat


df = _load_all()

# 사이드바 — 라인 / 시프트 필터
with st.sidebar:
    st.markdown("### 필터")
    line_options = sorted(df["line_id"].unique())
    lines = st.multiselect("라인", line_options, default=line_options)
    shift_options = sorted(df["shift"].unique())
    shifts = st.multiselect("교대", shift_options, default=shift_options)
    window = st.slider("표시 배치 수", 100, 3000, 1000, step=100)

filtered = df[df["line_id"].isin(lines) & df["shift"].isin(shifts)].tail(window)

# 상단 KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("표시 배치", f"{len(filtered):,}")
c2.metric("고장 건수", int(filtered["Machine failure"].sum()))
c3.metric("고위험(>50%)", int((filtered["risk_score"] > 0.5).sum()))
c4.metric("평균 리스크", f"{filtered['risk_score'].mean()*100:.1f}%")
c5.metric(
    "평균 신뢰도",
    f"{filtered['confidence'].mean()*100:.0f}%",
    help="max(p, 1-p) — 모델이 얼마나 확신하는지의 평균",
)

# 리스크 시계열 (+ 신뢰도 표시)
st.subheader("배치별 예측 리스크 · 신뢰도")
st.caption("점 색상이 진할수록 모델이 그 예측을 강하게 확신합니다 (max(p, 1-p) 기준).")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=filtered["timestamp"], y=filtered["risk_score"] * 100,
        mode="lines", line=dict(color="lightsteelblue", width=1),
        name="예측 리스크 (%)",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=filtered["timestamp"], y=filtered["risk_score"] * 100,
        mode="markers",
        marker=dict(
            color=filtered["confidence"] * 100,
            colorscale="Blues",
            cmin=50, cmax=100,
            size=5,
            colorbar=dict(title="신뢰도 (%)", thickness=12),
        ),
        name="배치별 리스크",
        customdata=filtered["confidence"] * 100,
        hovertemplate="시각=%{x}<br>리스크=%{y:.1f}%<br>신뢰도=%{customdata:.0f}%<extra></extra>",
    )
)
fail_points = filtered[filtered["Machine failure"] == 1]
fig.add_trace(
    go.Scatter(
        x=fail_points["timestamp"], y=fail_points["risk_score"] * 100,
        mode="markers", marker=dict(color="crimson", size=8, symbol="x"),
        name="실제 고장",
    )
)
fig.update_layout(
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="시각", yaxis_title="리스크 (%)",
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig, use_container_width=True)

# 신뢰도 분포 — 작게
with st.expander("예측 신뢰도 분포 (히스토그램)", expanded=False):
    hist = px.histogram(
        filtered, x=filtered["confidence"] * 100, nbins=20,
        labels={"x": "신뢰도 (%)"},
    )
    hist.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(hist, use_container_width=True)

# 공정 파라미터 시계열
st.subheader("공정 파라미터")
cols_plot = [
    "Torque [Nm]",
    "Rotational speed [rpm]",
    "Process temperature [K]",
    "Tool wear [min]",
]
for c in cols_plot:
    fig = px.line(filtered, x="timestamp", y=c, height=200)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("좌측에서 라인/교대를 필터하여 특정 영역만 볼 수 있습니다.")
