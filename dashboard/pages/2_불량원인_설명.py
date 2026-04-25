"""Page 2 — 불량원인 설명 (WHAT → ACTION → WHY).

Phase 11 (Revision 2) 재설계
- 상단: 위험 수준 + 평문 원인 + 권장 조치 카드 + ₩ 기대 손실
- 하단(접힘): 분석가용 SHAP Waterfall + 자연어 상세
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from career_kia.config import PROCESSED_DIR
from career_kia.models.train import load_feature_matrix
from career_kia.xai import (
    business_impact,
    explanation_templates,
    nl_generator,
    shap_utils,
)
from career_kia.xai.nl_generator import FEATURE_LABELS


st.set_page_config(page_title="불량원인 설명", page_icon="🔍", layout="wide")
st.title("🔍 불량원인 설명")
st.caption("배치를 선택하면 무엇이 위험한지 → 어떻게 할지 → 왜 그렇게 판단했는지 순서로 보여줍니다.")


@st.cache_resource
def _load_model_and_data():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, y, _ = load_feature_matrix()
    assumptions = business_impact.load_assumptions()
    return model, X, y, feat, assumptions


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
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10))
    return fig


model, X, y, feat, assumptions = _load_model_and_data()
proba = model.predict_proba(X)[:, 1]

# ---------------------------------------------------------------------------
# 사이드바 — 배치 선택
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 배치 선택")
    mode = st.radio(
        "선택 방식",
        ["고위험 상위 10", "실제 고장", "직접 입력"],
        index=0,
    )
    if mode == "고위험 상위 10":
        top_idx = np.argsort(proba)[::-1][:10]
        options = [f"{feat.iloc[i]['batch_id']} (리스크 {proba[i]*100:.0f}%)" for i in top_idx]
        sel = st.selectbox("배치", options)
        sel_idx = int(top_idx[options.index(sel)])
    elif mode == "실제 고장":
        fail_idx = feat.index[feat["Machine failure"] == 1].to_numpy()
        options = [f"{feat.iloc[i]['batch_id']} (리스크 {proba[i]*100:.0f}%)" for i in fail_idx]
        sel = st.selectbox("고장 배치", options)
        sel_idx = int(fail_idx[options.index(sel)])
    else:
        sel_batch = st.text_input("배치 ID (예: B000001)")
        if sel_batch and sel_batch in feat["batch_id"].values:
            sel_idx = int(feat.index[feat["batch_id"] == sel_batch][0])
        else:
            st.stop()

# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
row = feat.iloc[sel_idx]
risk = float(proba[sel_idx])
local = shap_utils.explain_batch(model, X.iloc[[sel_idx]])
contribution = explanation_templates.build_contribution(
    local,
    sample_idx=0,
    prediction=risk,
    sample_id=str(row["batch_id"]),
    k=8,
)
batch_exp = explanation_templates.BatchExplanation(contribution=contribution)
expected_loss = business_impact.translate_batch_risk_to_krw(risk, assumptions)

# ---------------------------------------------------------------------------
# WHAT (상단, 항상 노출)
# ---------------------------------------------------------------------------
top_left, top_right = st.columns([2, 1])

with top_left:
    st.markdown("### 🚨 무슨 일이 일어나고 있나")
    st.markdown(nl_generator.executive_summary(contribution, assumptions))

    st.markdown("##### 주요 위험 신호")
    bullets = nl_generator.what_happened_bullets(contribution, top_k=3)
    for b in bullets:
        st.markdown(f"- {b}")

with top_right:
    st.markdown("### 위험 수준")
    st.plotly_chart(_gauge(risk), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("실제 고장", "Yes" if row["Machine failure"] else "No")
    c2.metric("기대 손실", business_impact.format_krw(expected_loss))

st.markdown("---")

# ---------------------------------------------------------------------------
# ACTION (중단)
# ---------------------------------------------------------------------------
st.markdown("### 🛠️ 권장 조치")
actions = nl_generator.recommended_actions(contribution, assumptions, top_k=3)
icons = ["🔧", "⚙️", "🔍"]
cols = st.columns(len(actions))
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
# WHY (기본 노출) — 변수별 위험 기여도 Waterfall
# ---------------------------------------------------------------------------
st.markdown("### 📊 변수별 위험 기여도 (Waterfall)")
st.caption(
    "각 막대 길이는 해당 변수가 평균 대비 위험을 얼마나 끌어올리거나(빨강) 낮추는지(파랑) 를 의미합니다."
)
feature_values = local.X.iloc[0].values
sorted_idx = np.argsort(-np.abs(local.values[0]))[:10]
feature_labels = [
    f"{FEATURE_LABELS.get(local.feature_names[i], local.feature_names[i])} "
    f"= {feature_values[i]:.2f}"
    for i in sorted_idx
]
shap_values = local.values[0][sorted_idx]
fig = go.Figure(
    go.Waterfall(
        orientation="h",
        y=feature_labels[::-1],
        x=shap_values[::-1],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "steelblue"}},
        increasing={"marker": {"color": "crimson"}},
    )
)
fig.update_layout(
    height=450, margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="위험 기여도 (SHAP 값, +면 위험↑ / -면 위험↓)",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 분석가 상세 (선택적, expander)
# ---------------------------------------------------------------------------
with st.expander("🔬 분석가용 상세 — SHAP 통계 표현"):
    paragraph = nl_generator.batch_explanation_to_paragraph(batch_exp)
    st.markdown(paragraph.replace("\n", "  \n"))
