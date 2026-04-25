"""대시보드 공통 헬퍼.

- 예측 신뢰도(confidence) 계산 / 표시 유틸
- 데이터 출처(source) ↔ 예측 보드 사용처 레지스트리
"""

from __future__ import annotations

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# 신뢰도 (probabilistic confidence)
# ---------------------------------------------------------------------------

def confidence_from_proba(proba: float | np.ndarray) -> float | np.ndarray:
    """이진 확률 p 에서 신뢰도 = max(p, 1-p) 반환.

    p=0.5 (가장 불확실) → 0.5,  p=0.99 → 0.99.
    벡터/스칼라 모두 지원.
    """
    p = np.asarray(proba, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    return float(conf) if conf.ndim == 0 else conf


def confidence_label(conf: float) -> tuple[str, str]:
    """(라벨, 색상 hex) — 한글 등급 + Streamlit metric 호환 색."""
    if conf >= 0.90:
        return "매우 높음", "#27ae60"
    if conf >= 0.75:
        return "높음", "#2ecc71"
    if conf >= 0.60:
        return "중간", "#f1c40f"
    return "낮음", "#e74c3c"


def render_confidence_badge(conf: float, *, prefix: str = "신뢰도") -> None:
    """metric 옆에 두면 어울리는 짧은 배지."""
    label, color = confidence_label(conf)
    st.markdown(
        f"<div style='display:inline-block;padding:2px 8px;border-radius:8px;"
        f"background:{color}22;color:{color};font-size:0.85rem;font-weight:600;'>"
        f"{prefix} {conf*100:.0f}% · {label}</div>",
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
