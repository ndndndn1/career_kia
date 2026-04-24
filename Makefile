# SDF-Xplain 제조 AI 포트폴리오 Makefile

.PHONY: help setup data preprocess features train xai causal dash test clean

help:
	@echo "사용 가능한 명령어:"
	@echo "  make setup      - uv 로 가상환경 구성 및 의존성 설치"
	@echo "  make data       - CWRU / AI4I 데이터 다운로드 및 합성 batch 매핑"
	@echo "  make preprocess - 전처리 파이프라인 실행 (보간/필터/이상치/동기화)"
	@echo "  make features   - 윈도잉 기반 피처 엔지니어링"
	@echo "  make train      - 하이브리드 모델 학습 + MLflow 로깅"
	@echo "  make xai        - SHAP/LIME 설명 산출"
	@echo "  make causal     - 인과 DAG 및 개입 효과 추정"
	@echo "  make dash       - Streamlit 대시보드 실행"
	@echo "  make test       - pytest 단위 테스트"
	@echo "  make clean      - 캐시/중간 산출물 제거"

setup:
	uv venv
	uv pip install -e ".[dev]"

data:
	uv run python -m career_kia.data.download

preprocess:
	uv run python -m career_kia.preprocessing.run_pipeline

features:
	uv run python -m career_kia.features.run_pipeline

train:
	uv run python -m career_kia.models.train

xai:
	uv run python -m career_kia.xai.generate_explanations

causal:
	uv run python -m career_kia.causal.run_analysis

dash:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/interim/* data/processed/*
	rm -rf mlruns/ mlartifacts/
