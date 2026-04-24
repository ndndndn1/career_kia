"""Page 3 — 변수중요도 트렌드 + 공정별 히트맵."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from career_kia.config import PROCESSED_DIR, PROJECT_ROOT
from career_kia.models.train import load_feature_matrix
from career_kia.xai import shap_utils


st.set_page_config(page_title="변수중요도 트렌드", page_icon="📊", layout="wide")
st.title("📊 변수중요도 트렌드 + 공정별 히트맵")


@st.cache_resource
def _load():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    model = joblib.load(_ROOT / "models_artifacts" / "hybrid_model.joblib")
    X, y, _ = load_feature_matrix()
    bundle = shap_utils.explain_batch(model, X, max_samples=3000)
    return feat, X, y, model, bundle


feat, X, y, model, bundle = _load()

# --- 상단: 전역 상위 K 피처 중요도 ---
st.subheader("전역 SHAP 상위 피처")
k = st.slider("표시 개수", 5, 30, 15)
top = bundle.top_k(k=k)
fig = px.bar(top[::-1], orientation="h", labels={"value": "mean |SHAP|", "index": "feature"})
fig.update_layout(height=450, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- 중단: 공정별 기여도 히트맵 (교대 × 피처) ---
st.subheader("교대 × 상위 피처 평균 SHAP 히트맵")
sample_df = feat.iloc[bundle.X.index].copy().reset_index(drop=True)
sample_df["__key"] = sample_df.index
shap_matrix = pd.DataFrame(bundle.values, columns=bundle.feature_names)
shap_matrix["shift"] = feat.loc[bundle.X.index, "shift"].values
top_feats = list(top.head(10).index)
heat = shap_matrix.groupby("shift")[top_feats].mean()
fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- 하단: 시간축 리스크 트렌드 (이동평균) ---
st.subheader("시간축 리스크 신호 (이동평균)")
proba = model.predict_proba(X)[:, 1]
feat_sorted = feat.sort_values("timestamp").reset_index(drop=True)
feat_sorted["risk"] = proba[feat_sorted.index] * 100
feat_sorted["risk_ma"] = feat_sorted["risk"].rolling(100, min_periods=10).mean()
fig = px.line(
    feat_sorted,
    x="timestamp",
    y=["risk", "risk_ma"],
    labels={"value": "리스크 (%)"},
)
fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
