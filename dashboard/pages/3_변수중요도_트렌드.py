"""Page 3 — 변수중요도 트렌드 + 공정별 히트맵.

Phase 11 (Revision 2) 톤 다운
- 상단: 이번 주 TOP 3 원인 평문
- 라벨: "mean |SHAP|" → "원인 기여도"
- 분석가 모드 토글로 SHAP 원본 노출
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
import streamlit as st

from career_kia.config import PROCESSED_DIR
from career_kia.models.train import load_feature_matrix
from career_kia.xai import shap_utils
from career_kia.xai.nl_generator import FEATURE_LABELS
from dashboard._helpers import confidence_from_proba


st.set_page_config(page_title="변수중요도 트렌드", page_icon="📊", layout="wide")
st.title("📊 변수중요도 트렌드 + 공정별 히트맵")


@st.cache_resource
def _load():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, y, _ = load_feature_matrix()
    bundle = shap_utils.explain_batch(model, X, max_samples=3000)
    return feat, X, y, model, bundle


def _label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


feat, X, y, model, bundle = _load()

# 사이드바 — 분석가 모드
with st.sidebar:
    st.markdown("### 표시 옵션")
    analyst_mode = st.toggle("분석가 모드 (SHAP 원본값)", value=False)

# ---------------------------------------------------------------------------
# 상단: 이번 주 가장 큰 리스크 원인 TOP 3 (평문)
# ---------------------------------------------------------------------------
st.markdown("### 🔝 이번 주 가장 큰 리스크 원인 TOP 3")
top_global = bundle.top_k(k=3)
top_features = list(top_global.index)
cols = st.columns(3)
for i, (col, feat_name) in enumerate(zip(cols, top_features), start=1):
    with col:
        st.markdown(f"#### {i}. {_label(feat_name)}")
        st.caption("이 변수가 평소와 다를 때 가장 큰 위험 신호로 작용합니다.")

st.markdown("---")

# ---------------------------------------------------------------------------
# 전역 기여도 (라벨 톤다운)
# ---------------------------------------------------------------------------
st.subheader("전역 원인 기여도 상위 피처")
k = st.slider("표시 개수", 5, 30, 15)
top = bundle.top_k(k=k)
labelled = top[::-1].rename(index=_label)
x_label = "mean |SHAP|" if analyst_mode else "원인 기여도"
fig = px.bar(
    labelled,
    orientation="h",
    labels={"value": x_label, "index": "변수"},
    color=labelled.values,
    color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
)
fig.update_layout(height=450, showlegend=False, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 교대 × 상위 피처 히트맵 (Green-Yellow-Red)
# ---------------------------------------------------------------------------
heat_label = "교대별 SHAP 평균" if analyst_mode else "교대별 원인 기여도"
st.subheader(heat_label)
shap_matrix = pd.DataFrame(bundle.values, columns=bundle.feature_names)
shap_matrix["shift"] = feat.loc[bundle.X.index, "shift"].values
top_feats = list(top.head(10).index)
heat = shap_matrix.groupby("shift")[top_feats].mean()
heat_disp = heat if analyst_mode else heat.rename(columns=_label)
color_scale = "RdBu_r" if analyst_mode else [[0, "#2ecc71"], [0.5, "#f1c40f"], [1, "#e74c3c"]]
fig = px.imshow(
    heat_disp,
    aspect="auto",
    color_continuous_scale=color_scale,
    origin="lower",
)
fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 시간축 리스크 트렌드
# ---------------------------------------------------------------------------
st.subheader("시간축 리스크 신호 (이동평균) · 신뢰도")
proba = model.predict_proba(X)[:, 1]
conf = confidence_from_proba(proba)
feat_sorted = feat.sort_values("timestamp").reset_index(drop=True)
feat_sorted["리스크 (%)"] = proba[feat_sorted.index] * 100
feat_sorted["이동평균"] = feat_sorted["리스크 (%)"].rolling(100, min_periods=10).mean()
feat_sorted["신뢰도 이동평균 (%)"] = (
    pd.Series(conf[feat_sorted.index]).rolling(100, min_periods=10).mean() * 100
)
fig = px.line(
    feat_sorted,
    x="timestamp",
    y=["리스크 (%)", "이동평균", "신뢰도 이동평균 (%)"],
    labels={"value": "값 (%)", "variable": ""},
)
fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"전체 평균 신뢰도: **{float(np.mean(conf))*100:.0f}%** — 신뢰도가 일시적으로 떨어지는 구간은 "
    "모델이 판단을 망설인 시점이므로 추가 점검이 필요합니다."
)

if not analyst_mode:
    st.caption(
        "💡 좌측 사이드바에서 '분석가 모드' 를 켜면 SHAP 원본값과 통계 라벨로 전환됩니다."
    )
