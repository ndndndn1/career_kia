"""시계열 인과 검토 — Granger + PCMCI 소개 코드.

JD 의 "시계열 데이터에 적합한 인과 추론 기법 신기술 검토" 항목 대응.
완전한 PCMCI 구현은 tigramite 에 있으며 본 모듈은 개념적 데모·연결 지점을
제공한다. 실제 현장 데이터에서는 `tigramite` 를 사용할 것.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


def granger_causality_matrix(
    df: pd.DataFrame,
    *,
    max_lag: int = 3,
    test_name: str = "ssr_chi2test",
) -> pd.DataFrame:
    """쌍별 Granger 인과성 p-value 행렬.

    행 → 열 : 행이 열의 선행 변수인지 (p<0.05 면 인과적 선행 의심)
    """
    columns = df.columns.tolist()
    p_matrix = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns)
    for col_src in columns:
        for col_tgt in columns:
            if col_src == col_tgt:
                continue
            try:
                test_data = df[[col_tgt, col_src]].dropna()
                result = grangercausalitytests(
                    test_data, maxlag=max_lag, verbose=False
                )
                # maxlag 에서의 p-value
                p_values = [result[lag + 1][0][test_name][1] for lag in range(max_lag)]
                p_matrix.loc[col_src, col_tgt] = float(min(p_values))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Granger 실패 %s→%s: %s", col_src, col_tgt, exc)
                p_matrix.loc[col_src, col_tgt] = 1.0
    return p_matrix


def pcmci_note() -> str:
    """PCMCI 적용 시 권장 스텝 메모 — 실제 구현은 tigramite 참조."""
    return (
        "PCMCI 권장 절차:\n"
        "1) 정상성 확보 (차분 / deseasonalize)\n"
        "2) tigramite.data_processing.DataFrame 으로 래핑\n"
        "3) CMIknn 또는 ParCorr 로 조건부 독립성 검정\n"
        "4) PCMCI.run_pcmci(tau_max=..., pc_alpha=...) 실행\n"
        "5) q_matrix (FDR 보정) 기준으로 유의한 시차 엣지 선택\n"
        "→ 제조 공정에서는 작업주기/셋업 변경을 regime 변수로 노출하여\n"
        "   비정상성을 명시적으로 처리하는 것이 권장됨."
    )
