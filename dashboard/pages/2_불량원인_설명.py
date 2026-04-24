"""Page 2 — 불량원인 설명 (SHAP + 자연어).

배치 ID 를 선택하면 SHAP waterfall 과 한국어 자연어 설명을 출력.
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
from career_kia.xai import explanation_templates, nl_generator, shap_utils


st.set_page_config(page_title="불량원인 설명", page_icon="🔍", layout="wide")
st.title("🔍 불량원인 설명")


@st.cache_resource
def _load_model_and_data():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, y, _ = load_feature_matrix()
    return model, X, y, feat


model, X, y, feat = _load_model_and_data()
proba = model.predict_proba(X)[:, 1]

# 사이드바 — 배치 선택
with st.sidebar:
    st.markdown("### 배치 선택")
    mode = st.radio(
        "선택 방식",
        ["고위험 상위 10", "실제 고장", "직접 입력"],
        index=0,
    )
    if mode == "고위험 상위 10":
        top_idx = np.argsort(proba)[::-1][:10]
        options = [f"{feat.iloc[i]['batch_id']} (예측 {proba[i]*100:.1f}%)" for i in top_idx]
        sel = st.selectbox("배치", options)
        sel_idx = int(top_idx[options.index(sel)])
    elif mode == "실제 고장":
        fail_idx = feat.index[feat["Machine failure"] == 1].to_numpy()
        options = [f"{feat.iloc[i]['batch_id']} (예측 {proba[i]*100:.1f}%)" for i in fail_idx]
        sel = st.selectbox("고장 배치", options)
        sel_idx = int(fail_idx[options.index(sel)])
    else:
        sel_batch = st.text_input("배치 ID (예: B000001)")
        if sel_batch and sel_batch in feat["batch_id"].values:
            sel_idx = int(feat.index[feat["batch_id"] == sel_batch][0])
        else:
            st.stop()

# 헤더 정보
row = feat.iloc[sel_idx]
c1, c2, c3, c4 = st.columns(4)
c1.metric("배치 ID", row["batch_id"])
c2.metric("예측 리스크", f"{proba[sel_idx]*100:.1f}%")
c3.metric("실제 고장", "Yes" if row["Machine failure"] else "No")
c4.metric("진동 유형", row["vibration_fault_type"])

# SHAP 계산
local = shap_utils.explain_batch(model, X.iloc[[sel_idx]])
contribution = explanation_templates.build_contribution(
    local,
    sample_idx=0,
    prediction=float(proba[sel_idx]),
    sample_id=str(row["batch_id"]),
    k=8,
)

# waterfall 시각화
st.subheader("SHAP 기여도 (Waterfall)")
feature_values = local.X.iloc[0].values
sorted_idx = np.argsort(-np.abs(local.values[0]))[:10]
feature_labels = [
    f"{local.feature_names[i]} = {feature_values[i]:.2f}"
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
fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="SHAP 값")
st.plotly_chart(fig, use_container_width=True)

# 자연어 설명
st.subheader("📝 자연어 설명")
batch_exp = explanation_templates.BatchExplanation(contribution=contribution)
paragraph = nl_generator.batch_explanation_to_paragraph(batch_exp)
st.markdown(paragraph.replace("\n", "  \n"))
