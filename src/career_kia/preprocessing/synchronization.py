"""멀티 센서 시간 동기화 및 리샘플링.

제조 현장에서는 센서마다 샘플링 주파수가 다르고(예: 진동 12kHz/48kHz,
온도 1Hz, MES 이벤트 비동기), 타임스탬프 정렬과 공통 타임베이스 리샘플링이
필수다. 본 모듈은 다음을 제공한다.

    - resample_signal    : 단일 센서 신호의 공통 주파수 리샘플 (polyphase)
    - align_to_timegrid  : 여러 센서를 공통 시간 그리드로 리샘플
    - merge_asof_aligned : 비동기 이벤트(MES)를 센서 프레임에 근사 조인
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def resample_signal(
    signal: np.ndarray,
    fs_in: int,
    fs_out: int,
) -> np.ndarray:
    """polyphase 필터 기반 리샘플 (anti-aliasing 포함)."""
    if fs_in == fs_out:
        return signal.astype(float, copy=True)
    # 최대공약수로 up/down 비율 단순화
    gcd = np.gcd(fs_in, fs_out)
    up, down = fs_out // gcd, fs_in // gcd
    return resample_poly(signal, up, down)


def align_to_timegrid(
    frames: dict[str, pd.DataFrame],
    freq: str,
    *,
    how: str = "mean",
) -> pd.DataFrame:
    """여러 DatetimeIndex DataFrame을 공통 타임그리드로 리샘플.

    Parameters
    ----------
    frames
        {센서명: 시계열 프레임}. 각 프레임은 DatetimeIndex를 가져야 한다.
    freq
        타겟 빈도 (pandas offset alias, 예: '1s', '100ms').
    how
        집계 방식 ('mean' | 'last' | 'first' | 'max').
    """
    aligned = {}
    for name, df in frames.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name} 프레임이 DatetimeIndex가 아닙니다")
        resampler = df.resample(freq)
        aligned[name] = getattr(resampler, how)()
    # 컬럼 접두사로 충돌 방지
    renamed = {name: df.add_prefix(f"{name}__") for name, df in aligned.items()}
    return pd.concat(renamed.values(), axis=1).sort_index()


def merge_asof_aligned(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_time: str = "timestamp",
    right_time: str = "timestamp",
    tolerance: str | None = "5min",
    direction: str = "backward",
) -> pd.DataFrame:
    """비동기 이벤트 기반 근사 조인.

    MES/SCADA 이벤트를 센서 배치 테이블에 "가장 가까운 과거" 기준으로 조인할 때
    사용. `tolerance`를 지정해 너무 먼 매칭을 방지한다.
    """
    left_sorted = left.sort_values(left_time)
    right_sorted = right.sort_values(right_time)
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_time,
        right_on=right_time,
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
        direction=direction,
    )
