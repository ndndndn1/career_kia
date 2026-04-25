"""Page 5 — 데이터 출처 & 사용처 + 신뢰도 안내.

대시보드의 모든 예측이 어떤 원본 데이터에서 나왔고, 어떤 산출물을 거쳐
어느 페이지에서 쓰이는지 한 눈에 보여준다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import streamlit as st

from dashboard._helpers import ARTIFACTS, DATA_SOURCES, PAGE_USAGE


st.set_page_config(page_title="데이터 출처 & 사용처", page_icon="🗂️", layout="wide")
st.title("🗂️ 데이터 출처 & 사용처")
st.caption(
    "대시보드의 모든 예측은 아래 4개 데이터 소스 → 산출물 파이프라인을 거쳐 만들어집니다. "
    "어느 보드가 어떤 데이터를 쓰는지 추적할 수 있도록 구성했습니다."
)


# ---------------------------------------------------------------------------
# 1. 데이터 소스 카드
# ---------------------------------------------------------------------------
st.markdown("## 1. 원본 데이터 소스")

cols = st.columns(2)
for i, src in enumerate(DATA_SOURCES):
    with cols[i % 2]:
        link = f"[{src['name']}]({src['url']})" if src["url"] else f"**{src['name']}**"
        st.markdown(
            f"""
            #### {link}
            - **종류**: {src['kind']}
            - **포함**: {src['what']}
            - **라이선스**: {src['license']}
            - **로컬 경로**: `{src['raw_path']}`
            - **폴백**: {src['fallback']}
            """
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. 산출물 파이프라인
# ---------------------------------------------------------------------------
st.markdown("## 2. 산출물 파이프라인")
st.caption("원본 → 가공 산출물 → 모델 → 설명/인과. 빠진 산출물이 있으면 해당 단계의 `make` 타겟을 실행하세요.")

art_df = pd.DataFrame(
    [
        {
            "산출물": a["label"],
            "경로": a["path"],
            "재료(상위)": ", ".join(a["from"]),
            "생성 명령": a["produced_by"],
            "현재 상태": "✅ 존재" if (_ROOT / a["path"]).exists() else "⚠️ 없음",
        }
        for a in ARTIFACTS
    ]
)
st.dataframe(art_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 3. 페이지별 사용처 매핑
# ---------------------------------------------------------------------------
st.markdown("## 3. 페이지별 사용처")
st.caption("각 보드가 어떤 데이터/산출물 위에서 동작하는지 — 만약 보드가 실패하면 해당 칼럼의 산출물부터 점검하세요.")

# 헤더 = 산출물 + 데이터소스 합집합
header_ids = [a["id"] for a in ARTIFACTS] + [s["id"] for s in DATA_SOURCES]
header_labels = {a["id"]: a["label"] for a in ARTIFACTS}
header_labels.update({s["id"]: s["name"] for s in DATA_SOURCES})

rows = []
for page in PAGE_USAGE:
    row = {"페이지": page["page"], "예측 출력": page["predictions"]}
    used = set(page["uses"])
    # 페이지가 직접 쓰는 것 + 그 산출물의 상위 의존성도 ●로 칠하지 않고 ◌ 로
    direct = used.copy()
    indirect: set[str] = set()
    for a in ARTIFACTS:
        if a["id"] in direct:
            indirect.update(a["from"])
    indirect -= direct
    for hid in header_ids:
        if hid in direct:
            row[header_labels[hid]] = "●"
        elif hid in indirect:
            row[header_labels[hid]] = "◌"
        else:
            row[header_labels[hid]] = ""
    rows.append(row)

usage_df = pd.DataFrame(rows)
st.dataframe(usage_df, use_container_width=True, hide_index=True)
st.caption("● 직접 사용 · ◌ 간접 의존 (상위 산출물 통해)")

st.markdown("---")

# ---------------------------------------------------------------------------
# 4. 예측 명확도/안정성 정의
# ---------------------------------------------------------------------------
st.markdown("## 4. 예측 옆에 표시되는 두 지표")

st.markdown(
    """
    이 대시보드는 두 가지 보조 지표를 **함께** 표시합니다. 단일 `max(p, 1-p)` 형태의
    "신뢰도" 는 양성 비율 ~3.4% 의 불균형 데이터에서 항상 ≈100% 가 되어 정보를 주지
    못하기 때문입니다.
    """
)

a_col, b_col = st.columns(2)
with a_col:
    st.markdown(
        """
        ### A. 결정 명확도 (Decision Clarity)
        ```
        clarity = |p - prior| / max(prior, 1 - prior)
        ```
        - **prior**: 학습 데이터 양성 비율 (≈3.4%)
        - **0%** = 평균과 동일 → 정보 없음
        - **100%** = 평균과 가장 다른 극단 → 강한 신호
        - **등급**: 강한 ≥ 70% · 중간 ≥ 30% · 약한 ≥ 10% · 평균과 유사 < 10%
        - 적용 페이지: 🏠 홈 · 📈 1번 · 🔍 2번 · 📊 3번
        """
    )
with b_col:
    st.markdown(
        """
        ### B. 모델 안정성 (Logit-space SE)
        - pyGAM `LogisticGAM` 의 logit 공간 표준오차
        - `SE = sqrt(x.T · cov · x)` (per-sample) 의 **분위 기반 안정성**
        - 확률이 saturated 된 영역(p≈0 or 1)에서도 흔들림 측정 가능
        - **등급**: 매우 안정 ≥ 75% · 안정 ≥ 50% · 흔들림 ≥ 25% · 매우 흔들림 < 25%
        - 적용: 🏠 홈 (TOP 위험 배치) · 🔍 2번 (선택 배치)
        """
    )

st.info(
    "**왜 두 가지인가?** A 는 *예측이 평균과 얼마나 다른가* (신호의 세기), "
    "B 는 *이 추정 자체가 얼마나 흔들리는가* (모델 불확실성). 둘은 독립적입니다 — "
    "강한 신호이면서도 모델이 흔들릴 수 있고, 그 반대도 가능합니다."
)

with st.expander("🔬 What-if 페이지의 인과 신뢰도는 다릅니다"):
    st.markdown(
        """
        - 인과 추정에는 분류 확률이 없으므로 위 지표가 적용되지 않습니다.
        - 대신 DoWhy `random_common_cause` · `data_subset_refuter` 두 반박 검정의
          **통과율** 을 *인과 추정 안정성* 으로 표기합니다 (2/2 → 100%).
        """
    )
