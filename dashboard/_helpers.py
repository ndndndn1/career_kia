"""대시보드 공통 헬퍼.

- 예측 결정 명확도(decision clarity, A) — 평균 위험률(prior) 대비 정보량
- 모델 흔들림(logit-space SE, B) — pyGAM 메타 모델의 표준오차
- 데이터 출처(source) ↔ 예측 보드 사용처 레지스트리

배경 — 본 데이터는 양성 비율 ~3.4% 의 극심한 불균형이므로 max(p, 1-p) 형태의
"신뢰도" 는 모든 배치에서 ≈100% 가 되어 정보를 주지 못한다. 그래서:
  · A: 평균 위험률 prior 대비 |p - prior| 를 정규화 → "이 예측이 평소와 얼마나
        다른가" 를 0~1 로 표현
  · B: pyGAM 의 logit 공간 SE — 확률이 saturated 된 영역에서도 흔들림 측정 가능
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse


# ---------------------------------------------------------------------------
# A. 결정 명확도 (decision clarity vs prior)
# ---------------------------------------------------------------------------

def decision_clarity(
    proba: float | np.ndarray, prior: float
) -> float | np.ndarray:
    """평균 위험률 prior 대비 |p - prior| / max(prior, 1-prior) 반환.

    - p == prior 면 0 (정보 없음)
    - p 가 0 또는 1 로 갈수록 1 (강한 신호)
    - 불균형 데이터(prior 0.034)에서도 분포가 펼쳐짐 — max(p,1-p) 의 saturation 회피.
    """
    p = np.asarray(proba, dtype=float)
    denom = max(float(prior), 1.0 - float(prior))
    clarity = np.minimum(np.abs(p - float(prior)) / denom, 1.0)
    return float(clarity) if clarity.ndim == 0 else clarity


def clarity_label(clarity: float) -> tuple[str, str]:
    """결정 명확도 등급 (라벨, 색상). 분포가 양극단에 몰리는 특성을 고려한 컷오프."""
    if clarity >= 0.70:
        return "강한 신호", "#27ae60"
    if clarity >= 0.30:
        return "중간 신호", "#2ecc71"
    if clarity >= 0.10:
        return "약한 신호", "#f1c40f"
    return "평균과 유사", "#95a5a6"


def render_clarity_badge(
    clarity: float, *, prefix: str = "결정 명확도"
) -> None:
    """결정 명확도 배지 (A)."""
    label, color = clarity_label(clarity)
    st.markdown(
        f"<div style='display:inline-block;padding:2px 8px;border-radius:8px;"
        f"background:{color}22;color:{color};font-size:0.85rem;font-weight:600;'>"
        f"{prefix} {clarity*100:.0f}% · {label}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# B. 모델 흔들림 (logit-space SE from pyGAM)
# ---------------------------------------------------------------------------

def compute_logit_se(model, X: pd.DataFrame) -> np.ndarray:
    """HybridModel(LightGBM + LogisticGAM) 의 logit 공간 표준오차 (per-sample).

    Returns array of shape (n,). 값이 클수록 모델이 흔들리는 예측.
    """
    gam = model.gam_
    X_san = model._prepare_X(X)
    p_lgbm = model.lgbm_.predict_proba(X_san)[:, 1]
    z = np.clip(p_lgbm, 1e-4, 1.0 - 1e-4)
    lgbm_logit = np.log(z / (1.0 - z))
    cols = [model.feature_names_[i] for i in model.interpret_idx_]
    meta_X = np.column_stack([lgbm_logit, X[cols].to_numpy()])
    mm = gam._modelmat(meta_X)
    mm = mm.toarray() if sparse.issparse(mm) else np.asarray(mm)
    cov = gam.statistics_["cov"]
    var = np.einsum("ij,jk,ik->i", mm, cov, mm)
    return np.sqrt(np.maximum(var, 0.0))


def se_to_stability(se_array: np.ndarray, se_value: float | None = None):
    """SE 분포를 0~1 안정성 점수로 변환 (1 = 가장 안정).

    벡터(se_array) 만 받으면 전체 안정성 벡터, se_value 도 주면 단일 값 반환.
    """
    se_array = np.asarray(se_array, dtype=float)
    rank = pd.Series(se_array).rank(pct=True).to_numpy()  # 0..1
    stability = 1.0 - rank
    if se_value is None:
        return stability
    # se_value 의 분위 기반 안정성
    pct = float((se_array <= se_value).mean())  # 작은 SE 비율
    # SE 가 작으면 stability ↑
    return 1.0 - float((se_array < se_value).mean())


def stability_label(stability: float) -> tuple[str, str]:
    if stability >= 0.75:
        return "매우 안정", "#27ae60"
    if stability >= 0.50:
        return "안정", "#2ecc71"
    if stability >= 0.25:
        return "흔들림", "#f1c40f"
    return "매우 흔들림", "#e74c3c"


def render_stability_badge(stability: float, se_value: float) -> None:
    label, color = stability_label(stability)
    st.markdown(
        f"<div style='display:inline-block;padding:2px 8px;border-radius:8px;"
        f"background:{color}22;color:{color};font-size:0.8rem;font-weight:600;'>"
        f"모델 안정성 {stability*100:.0f}% · {label} (logit SE={se_value:.1f})</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 데이터 출처 ↔ 사용처 레지스트리
# 페이지가 늘거나 산출물이 바뀌면 여기만 수정.
# ---------------------------------------------------------------------------

DATA_SOURCES: list[dict] = [
    {
        "id": "ai4i",
        "name": "UCI AI4I 2020 Predictive Maintenance",
        "kind": "공정 파라미터 (정형)",
        "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        "what": "회전수(rpm), 토크(Nm), 공정/대기 온도(K), 공구 마모(min), Type, Machine failure 라벨",
        "license": "CC BY 4.0",
        "fallback": "네트워크 차단 시 동일 스키마 합성 폴백 (`_synthesize_ai4i`)",
        "raw_path": "data/raw/ai4i/ai4i2020.csv",
    },
    {
        "id": "cwru",
        "name": "CWRU Bearing Fault Data",
        "kind": "진동 신호 (비정형, 12 kHz)",
        "url": "https://engineering.case.edu/bearingdatacenter",
        "what": "정상/IR/OR/Ball 결함 베어링 가속도 신호 → 시간/주파수 도메인 피처",
        "license": "Case Western Reserve University 학술 사용 허용",
        "fallback": "기본은 합성 폴백 (`synthetic_index.parquet`, 클래스당 40개 × 1초)",
        "raw_path": "data/raw/cwru/",
    },
    {
        "id": "mes",
        "name": "합성 MES 메타",
        "kind": "이벤트 메타 (조인 키)",
        "url": None,
        "what": "batch_id · timestamp · shift · operator_id · line_id — AI4I 행을 가상 라인 배치로 정렬",
        "license": "프로젝트 내 생성",
        "fallback": "—",
        "raw_path": "src/career_kia/data/loaders.py",
    },
    {
        "id": "business",
        "name": "비즈니스 가정치",
        "kind": "환산 파라미터",
        "url": None,
        "what": "배치당 손실 ₩, 연 배치 수, 조치 비용 — 리스크/ATE 를 ₩ 임팩트로 환산",
        "license": "프로젝트 내 플레이스홀더 — 실제 라인 값으로 교체",
        "fallback": "—",
        "raw_path": "configs/business.yaml",
    },
]


# 산출물 (raw → processed → 모델 → XAI/인과)
ARTIFACTS: list[dict] = [
    {
        "id": "features",
        "label": "피처 테이블",
        "path": "data/processed/features.parquet",
        "from": ["ai4i", "cwru", "mes"],
        "produced_by": "make preprocess → make features",
    },
    {
        "id": "hybrid_model",
        "label": "하이브리드 모델 (LightGBM + pyGAM)",
        "path": "models_artifacts/hybrid_model.joblib",
        "from": ["features"],
        "produced_by": "make train",
    },
    {
        "id": "shap",
        "label": "SHAP 산출물 (전역/지역 기여도)",
        "path": "xai_artifacts/",
        "from": ["hybrid_model", "features"],
        "produced_by": "make xai",
    },
    {
        "id": "ate",
        "label": "DoWhy ATE / dose-response",
        "path": "causal_artifacts/",
        "from": ["features"],
        "produced_by": "make causal",
    },
]


# 페이지가 어떤 출처/산출물을 쓰는지
PAGE_USAGE: list[dict] = [
    {
        "page": "🏠 홈 · Executive Summary",
        "file": "dashboard/app.py",
        "uses": ["features", "hybrid_model", "shap", "business"],
        "predictions": "최근 1,000 배치 리스크, 고위험 카운트, TOP3 권장조치",
    },
    {
        "page": "📈 1. 실시간 모니터링",
        "file": "dashboard/pages/1_실시간_모니터링.py",
        "uses": ["features", "hybrid_model"],
        "predictions": "배치별 리스크 시계열, 평균 리스크 KPI",
    },
    {
        "page": "🔍 2. 불량원인 설명",
        "file": "dashboard/pages/2_불량원인_설명.py",
        "uses": ["features", "hybrid_model", "shap", "business"],
        "predictions": "선택 배치의 리스크 + SHAP 기여도 + ₩ 기대손실",
    },
    {
        "page": "📊 3. 변수중요도 트렌드",
        "file": "dashboard/pages/3_변수중요도_트렌드.py",
        "uses": ["features", "hybrid_model", "shap"],
        "predictions": "전역 원인 기여도 TOP-K, 교대별 히트맵, 시간축 이동평균",
    },
    {
        "page": "🧪 4. What-if 개입 시뮬",
        "file": "dashboard/pages/4_Whatif_개입시뮬레이션.py",
        "uses": ["ate", "business"],
        "predictions": "공구마모/토크/회전수 개입 시 ATE %p + 연 ₩ 절감 곡선",
    },
]
