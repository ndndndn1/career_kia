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
# 4. 예측 신뢰도 정의
# ---------------------------------------------------------------------------
st.markdown("## 4. 예측 신뢰도(confidence) 표기 방법")
st.markdown(
    """
    각 보드의 예측 옆에 표시되는 **신뢰도** 는 다음과 같이 정의합니다.

    ```
    confidence = max(p, 1 - p)        # p = 모델이 출력한 고장 확률
    ```

    - **범위**: 0.5 (가장 불확실, 동전 던지기 수준) ~ 1.0 (완전 확신)
    - **등급**: 매우 높음 ≥ 90% · 높음 ≥ 75% · 중간 ≥ 60% · 낮음 < 60%
    - **모델 출력**: LightGBM(트리 부스팅) → logit → pyGAM(LogisticGAM) 의 최종 확률
    - **What-if 페이지의 ATE 신뢰도**: DoWhy `random_common_cause` · `data_subset_refuter`
      두 반박 검정을 통과한 추정치만 사용하므로, 신뢰도는 *통계적 안정성* 의미로 사용합니다.

    > 신뢰도가 낮은(중간 이하) 예측은 단독 의사결정 근거로 쓰지 말고, 인접 배치/SHAP 기여도와
    > 함께 검토하는 것을 권장합니다.
    """
)

with st.expander("🔬 더 정교한 불확실성을 원할 경우 (개발자용)"):
    st.markdown(
        """
        - LightGBM tree-level variance(`predict(..., pred_contrib=True)` 기반 분산) 사용 가능
        - pyGAM `prediction_intervals(X)` 로 95% PI 추정 가능
        - 본 대시보드는 직관적 비교를 위해 `max(p, 1-p)` 단일 지표를 표준으로 사용
        """
    )
