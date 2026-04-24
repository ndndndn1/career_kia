"""SDF-Xplain 대시보드 — 메인 엔트리.

실행::

    make dash
    # 또는
    streamlit run dashboard/app.py

페이지:
    1. 실시간 모니터링
    2. 불량원인 설명
    3. 변수중요도 트렌드
    4. What-if 개입 시뮬레이션
"""

from __future__ import annotations

import sys
from pathlib import Path

# src 모듈 import 경로 확보
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

st.set_page_config(
    page_title="SDF-Xplain · 제조 AI 대시보드",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ SDF-Xplain · 제조 센서 기반 품질/고장 원인 분석 대시보드")

st.markdown(
    """
    **Software Defined Factory(SDF)** 참조 구현.
    좌측 사이드바에서 페이지를 선택하세요.

    | 페이지 | 설명 |
    |---|---|
    | 1. 실시간 모니터링 | 배치별 센서·공정 시계열 + 리스크 스코어 |
    | 2. 불량원인 설명 | 샘플별 SHAP 기여도 + 자연어 설명 |
    | 3. 변수중요도 트렌드 | 공정별 SHAP 히트맵, 시간축 트렌드 |
    | 4. What-if 개입 시뮬 | Causal ATE 기반 공정 조건 변경 효과 |

    **데이터/모델 산출물**
    - 피처 테이블: `data/processed/features.parquet`
    - 하이브리드 모델: `models_artifacts/hybrid_model.joblib`
    - SHAP 산출물: `xai_artifacts/`
    - 인과 ATE: `causal_artifacts/`
    """
)

st.info(
    "먼저 `make data → make preprocess → make features → make train → make xai → make causal` 를 실행해 "
    "대시보드가 읽을 산출물을 생성하세요."
)
