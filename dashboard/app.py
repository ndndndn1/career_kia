"""SDF-Xplain 대시보드 — Executive Summary 홈.

Phase 11 (Revision 2) 재설계
----------------------------
경영진 3초 의사결정 화면. KPI / Gauge / TOP 3 조치 / 7일 sparkline.

실행::

    make dash
    # 또는
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from career_kia.config import PROCESSED_DIR
from career_kia.models.train import load_feature_matrix
from career_kia.xai import business_impact, explanation_templates, nl_generator, shap_utils
from dashboard._helpers import confidence_from_proba, render_confidence_badge

st.set_page_config(
    page_title="SDF-Xplain · Executive Summary",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def _bootstrap():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, _y, _ = load_feature_matrix()
    proba = model.predict_proba(X)[:, 1]
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    feat["risk_score"] = proba[feat.index]
    assumptions = business_impact.load_assumptions()
    return model, X, feat, assumptions


def _gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 33], "color": "#2ecc71"},
                    {"range": [33, 66], "color": "#f1c40f"},
                    {"range": [66, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": value * 100,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    return fig


try:
    model, X, feat, assumptions = _bootstrap()
    artifacts_ready = True
except (FileNotFoundError, OSError) as e:
    artifacts_ready = False
    bootstrap_error = str(e)


st.title("⚙️ SDF-Xplain · 오늘의 라인 상태")

if not artifacts_ready:
    st.warning(
        "모델/데이터 산출물이 아직 없습니다. 먼저 `make data → make preprocess → "
        "make features → make train` 을 실행해 주세요."
    )
    with st.expander("자세한 오류"):
        st.code(bootstrap_error)
    st.stop()


# ---------------------------------------------------------------------------
# 분석 대상: 가장 최근 24시간(또는 가용 마지막 1000건)
# ---------------------------------------------------------------------------

today_window = feat.tail(1000)
yesterday_window = feat.iloc[-2000:-1000] if len(feat) >= 2000 else today_window

today_window = today_window.copy()
today_window["confidence"] = confidence_from_proba(today_window["risk_score"].values)
avg_confidence_today = float(today_window["confidence"].mean())

high_risk = today_window[today_window["risk_score"] > 0.5]
expected_loss_today = sum(
    business_impact.translate_batch_risk_to_krw(r, assumptions)
    for r in high_risk["risk_score"]
)
avg_risk_today = today_window["risk_score"].mean()
avg_risk_yesterday = yesterday_window["risk_score"].mean()
delta_pp = (avg_risk_today - avg_risk_yesterday) * 100

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("고위험 배치", f"{len(high_risk):,} 건")
k2.metric("예측 손실", business_impact.format_krw(expected_loss_today))
k3.metric("평균 리스크", f"{avg_risk_today*100:.1f}%")
k4.metric("전일 대비", f"{delta_pp:+.1f}%p", delta=f"{delta_pp:+.1f}")
k5.metric("평균 신뢰도", f"{avg_confidence_today*100:.0f}%", help="max(p, 1-p) 기준 — 높을수록 모델이 확신")

st.markdown("---")

# ---------------------------------------------------------------------------
# Gauge + 7일 sparkline
# ---------------------------------------------------------------------------
g1, g2 = st.columns([1, 2])

with g1:
    st.markdown("##### 현재 평균 리스크")
    st.plotly_chart(_gauge(avg_risk_today), use_container_width=True)

with g2:
    st.markdown("##### 최근 7일 평균 리스크 추이")
    feat["date"] = pd.to_datetime(feat["timestamp"]).dt.date
    daily = feat.groupby("date")["risk_score"].mean().reset_index().tail(7)
    fig = go.Figure(
        go.Scatter(
            x=daily["date"], y=daily["risk_score"] * 100,
            mode="lines+markers", line=dict(color="#3498db", width=3),
            fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.15)",
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="평균 리스크 (%)", range=[0, max(40, daily["risk_score"].max() * 100 * 1.2)]),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# TOP 3 권장 조치 — 오늘 가장 위험한 배치 1건의 SHAP 기반
# ---------------------------------------------------------------------------
st.markdown("### 🛠️ 오늘의 권장 조치 TOP 3")

if len(high_risk) == 0:
    st.success("현재 고위험 배치가 없습니다. 평소 운영을 유지하세요.")
else:
    top_idx = high_risk["risk_score"].idxmax()
    top_row = feat.loc[top_idx]
    top_confidence = float(confidence_from_proba(float(top_row["risk_score"])))
    local = shap_utils.explain_batch(model, X.iloc[[top_idx]])
    contribution = explanation_templates.build_contribution(
        local,
        sample_idx=0,
        prediction=float(top_row["risk_score"]),
        sample_id=str(top_row["batch_id"]),
        k=8,
    )
    actions = nl_generator.recommended_actions(contribution, assumptions, top_k=3)

    st.caption(
        f"가장 위험한 배치 **{top_row['batch_id']}** (리스크 "
        f"{top_row['risk_score']*100:.0f}%) 기반 권장 조치"
    )
    render_confidence_badge(top_confidence, prefix="이 예측 신뢰도")

    cols = st.columns(len(actions))
    icons = ["🔧", "⚙️", "🔍"]
    for i, (col, card) in enumerate(zip(cols, actions)):
        with col:
            icon = icons[i] if i < len(icons) else "•"
            st.markdown(f"#### {icon} {card['label']}")
            st.write(card["description"])
            st.caption(
                f"예상 조치 비용: {card['estimated_cost_text']}  \n"
                f"미조치 시 배치당 기대 손실: {card['expected_loss_text']}"
            )

st.markdown("---")

# ---------------------------------------------------------------------------
# 좌측 메뉴 안내 + 프로젝트 소개 (접힘)
# ---------------------------------------------------------------------------
st.info(
    "더 자세한 분석은 좌측 사이드바에서 페이지를 선택하세요 — "
    "**1. 실시간 모니터링** · **2. 불량원인 설명** · **3. 변수중요도 트렌드** · "
    "**4. What-if 개입 시뮬레이션** · **5. 데이터 출처 & 사용처**"
)

with st.expander("📘 프로젝트 소개 (Software Defined Factory · SDF-Xplain)"):
    st.markdown(
        """
        **Software Defined Factory(SDF)** 참조 구현. 자동차 파워트레인 가공 라인의
        스핀들 베어링/공정 이상에 의한 품질 저하를 조기 예측하고, 현장에서 납득할 수
        있는 설명과 인과 기반의 What-if 답을 제공합니다.

        **데이터/모델 산출물**
        - 피처 테이블: `data/processed/features.parquet`
        - 하이브리드 모델: `models_artifacts/hybrid_model.joblib`
        - SHAP 산출물: `xai_artifacts/`
        - 인과 ATE: `causal_artifacts/`

        **참조 가정치**: `configs/business.yaml` (실 라인 도입 시 교체)
        """
    )
