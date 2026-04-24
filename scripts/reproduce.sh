#!/usr/bin/env bash
# SDF-Xplain 재현 스크립트
# 처음부터 끝까지 파이프라인을 순차 실행한다.

set -euo pipefail

echo "[1/6] 환경 구성"
make setup

echo "[2/6] 데이터 다운로드 및 합성 batch 매핑"
make data

echo "[3/6] 전처리"
make preprocess

echo "[4/6] 피처 엔지니어링"
make features

echo "[5/6] 모델 학습 (+ MLflow 로깅)"
make train

echo "[6/6] XAI · 인과 분석"
make xai
make causal

echo ""
echo "모든 단계 완료."
echo "대시보드 실행: make dash"
echo "MLflow UI: uv run mlflow ui"
