# SDF-Xplain · 제조 센서 기반 품질/고장 원인 분석 XAI 포트폴리오

> Software Defined Factory(SDF) 구현을 위한 제조 AI 포트폴리오.
> 자동차 파워트레인 가공 라인의 **스핀들 베어링 상태**와 **CNC 공정 파라미터**를 함께 분석하여,
> 품질 저하를 **예측**하고 · **원인을 설명**하며 · **무엇을 바꾸면 불량이 줄어드는가를 인과적으로 추정**한다.

## 🚀 라이브 데모

**👉 <https://duty-kia-260426.streamlit.app/>**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://duty-kia-260426.streamlit.app/)

- **로컬 실행**: `make dash` (또는 `streamlit run dashboard/app.py`)
- **포함 보드 (6)**: 🏠 Executive Summary · 📈 실시간 모니터링 · 🔍 불량원인 설명 · 📊 변수중요도 트렌드 · 🧪 What-if 개입 시뮬 · 🗂️ 데이터 출처 & 사용처
- **신뢰도 표시**: 모든 예측 옆에 `confidence = max(p, 1-p)` 기반 신뢰도 등급(매우 높음/높음/중간/낮음) 노출
- **호스팅**: Streamlit Community Cloud · 의존성은 `requirements.txt` + `.streamlit/config.toml` 자동 인식

---

## 1. 프로젝트 배경과 목표

제조 현장에서 불량/고장은 단일 센서·단일 변수로 설명되지 않는다. 진동 같은 **비정형 신호**와 회전수·토크·공구 마모 같은 **정형 공정 파라미터**가 상호작용하여 품질을 결정하며, 현장 엔지니어가 실제 조치를 취하려면 다음 세 가지가 모두 필요하다.

1. **조기 예측** — 센서와 공정 이상을 실시간으로 감지
2. **설명 가능한 원인 분석** — 모델이 왜 그렇게 판단했는지, 어느 변수·어느 시점이 기여했는지
3. **인과 기반 처방** — "공구 마모를 30분 빨리 교체하면 불량률이 얼마나 줄어드는가?" 같은 개입 효과 추정

본 프로젝트는 위 세 가지를 하나의 end-to-end 파이프라인으로 엮어 SDF 관점의 참조 구현을 제공한다.

## 2. 시나리오 및 데이터

**가상 시나리오**: 자동차 파워트레인 부품(샤프트·기어) 가공 라인. 스핀들 베어링 진동과 CNC 공정 조건이 함께 기록되며, 배치 단위 품질 검사 결과를 라벨로 사용한다.

| 레이어 | 공개 데이터셋 | 실제 매핑 |
|---|---|---|
| 진동 신호 | [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter) | 스핀들 베어링 가속도 (정상/IR/OR/Ball) |
| 공정 파라미터 | [UCI AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 회전수, 토크, 공구 마모, 공정/대기 온도 |
| MES 메타 | 합성 `batch_id`, `timestamp`, `shift`, `operator_id` | 배치 단위 조인 키 |

> 공개 데이터셋이 실제 자동차 제조 데이터는 아니지만, **각 레이어의 신호 처리·모델링·인과 분석 방법론은 현장에 그대로 이식 가능한 형태**로 구현한다.

## 3. 기술 스택

- **언어/환경**: Python 3.11, [uv](https://github.com/astral-sh/uv)
- **신호 처리**: NumPy, SciPy, PyWavelets
- **머신러닝**: scikit-learn, LightGBM, pyGAM (성능 + 해석 혼합)
- **XAI**: SHAP, LIME
- **인과추론**: DoWhy, EconML, causal-learn
- **MLOps**: MLflow (실험 추적 · 모델 버저닝 · 드리프트 모니터링)
- **대시보드**: Streamlit, Plotly

## 4. 디렉터리 구조

```
career_kia/
├── data/                    # 원본/중간/가공 데이터 (원본은 gitignore)
├── docs/                    # 한글 기술 문서 (개요/데이터/방법론/결과)
├── notebooks/               # 7편의 노트북 (EDA→전처리→피처→모델→XAI→인과→리포트)
├── src/career_kia/
│   ├── data/                # 데이터 다운로드 · 로더
│   ├── preprocessing/       # 보간 · 필터 · 이상치 · 동기화
│   ├── features/            # 윈도잉 · 시간/주파수 도메인 피처
│   ├── models/              # 베이스라인 · 하이브리드(LightGBM + pyGAM)
│   ├── xai/                 # SHAP · LIME · 설명 템플릿 · 자연어 생성
│   ├── causal/              # DAG · 개입 효과 · 시계열 인과
│   └── mlops/               # MLflow 래퍼 · 드리프트 감지
├── dashboard/               # Streamlit 4페이지 대시보드
├── tests/                   # pytest 단위 테스트
├── scripts/reproduce.sh     # 재현 스크립트
├── Makefile
└── pyproject.toml
```

## 5. 실행 방법

```bash
# 1) 환경 구성
make setup

# 2) 데이터 다운로드 + batch 매핑
make data

# 3) 전처리 → 피처 → 학습 순차 실행
make preprocess
make features
make train

# 4) XAI · 인과 분석 (선택)
make xai
make causal

# 5) 대시보드 실행
make dash        # http://localhost:8501

# 6) MLflow UI
uv run mlflow ui # http://localhost:5000

# 7) 단위 테스트
make test
```

## 6. 핵심 결과

### 모델 성능 (5-fold GroupKFold, 양성 3.4%)

| 모델 | ROC-AUC | PR-AUC | F1 | P@R=0.9 |
|---|---|---|---|---|
| Logistic Regression | 0.934 | 0.528 | 0.395 | 0.136 |
| Random Forest | 0.960 | 0.680 | 0.621 | 0.270 |
| **Hybrid LGBM + pyGAM** | **0.962** | **0.857** | **0.776** | **0.611** |

### 공정 개입 효과 (DoWhy ATE, backdoor 조정)

| 처치 | 기준 → 개입 | ATE | Refutation 통과 |
|---|---|---|---|
| Tool wear | 50 → 200 분 | +4.47 %p | RCC · Subset 모두 ±0.01%p 이내 |
| Torque | 40 → 60 Nm | +6.76 %p | RCC · Subset 모두 ±0.1%p 이내 |
| Rotational speed | 1600 → 1300 rpm | +5.91 %p | RCC · Subset 모두 ±0.01%p 이내 |

### 테스트
- `pytest tests/` — 전처리 18개 + 피처 15개 + 모델 6개 + XAI 10개 + 인과 7개 + MLOps 5개 = **총 61개 통과**

## 7. JD → 산출물 매핑

| JD 문구 | 증빙 |
|---|---|
| 결측치 보간, 노이즈 필터링, 이상치 탐지 | `src/career_kia/preprocessing/*`, `notebooks/02_*` |
| 다중 센서 시간 동기화 및 리샘플링 | `preprocessing/synchronization.py` |
| Windowing 기반 피처 생성 | `features/windowing.py`, `time_domain.py`, `freq_domain.py` |
| 센서/MES/공정 파라미터 기반 불량·고장 탐지 | `notebooks/04_*`, 합성 batch 조인 |
| 공정 특성 반영 특징공학, 변수 중요도 프레임 | `features/freq_domain.py`, `xai/shap_utils.py` |
| 성능 ML + 해석 통계 혼합 모델링 | `models/hybrid.py` (LightGBM + pyGAM 스태킹) |
| SHAP · LIME 개별/집합 설명 | `xai/*`, `notebooks/05_*` |
| 설명 템플릿 표준화 | `xai/explanation_templates.py` |
| 대시보드 시각화(중요도 트렌드/히트맵/리스크) | `dashboard/pages/*` |
| 비전문가용 설명문 · 스토리텔링 | `xai/nl_generator.py`, `notebooks/07_*` |
| 인과 그래프, 교란변수 통제 | `causal/dag.py`, `notebooks/06_*` |
| 개입 효과 추정 | `causal/intervention.py`, 대시보드 What-if 페이지 |
| 시계열 인과추론 신기술 검토 | `causal/time_series.py` |
| MLflow 버저닝 · 모니터링 · 재학습 (우대) | `mlops/*` |
| Streamlit 대시보드 (우대) | `dashboard/` |

## 8. 라이선스 및 데이터 출처

- CWRU Bearing Dataset — Case Western Reserve University, 연구·교육용 공개
- AI4I 2020 Predictive Maintenance — UCI ML Repository (CC BY 4.0)

본 저장소의 코드는 개인 포트폴리오 목적이며, 외부 데이터셋은 각 제공처의 라이선스를 따른다.
