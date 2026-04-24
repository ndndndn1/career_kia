"""슬라이딩 윈도우 생성.

진동 신호를 일정 길이의 윈도우로 쪼개 피처를 산출한다.
- 오버랩 지원 (기본 50%)
- (N, win_size) 2D 배열 반환 — 벡터화된 시간/주파수 피처 계산에 적합
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from career_kia.config import WINDOW_SIZE, WINDOW_STRIDE


def make_windows(
    signal: np.ndarray,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    drop_last: bool = True,
) -> np.ndarray:
    """1D 신호 → (n_windows, window_size) 2D.

    Parameters
    ----------
    drop_last
        True 이면 window 에 맞지 않는 꼬리 버림. False 면 0 패딩.
    """
    if signal.ndim != 1:
        raise ValueError(f"1D 신호가 필요합니다. shape={signal.shape}")
    if len(signal) < window_size:
        if drop_last:
            return np.empty((0, window_size), dtype=signal.dtype)
        padded = np.zeros(window_size, dtype=signal.dtype)
        padded[: len(signal)] = signal
        return padded.reshape(1, -1)

    windows = sliding_window_view(signal, window_shape=window_size)[::stride]
    if not drop_last:
        tail_start = 1 + (len(signal) - window_size) // stride * stride
        if tail_start + window_size < len(signal):
            padded = np.zeros(window_size, dtype=signal.dtype)
            remainder = signal[tail_start + stride :]
            padded[: len(remainder)] = remainder
            windows = np.vstack([windows, padded[None, :]])
    return np.ascontiguousarray(windows)


def window_count(
    signal_length: int,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> int:
    """몇 개의 윈도우가 나올지 예측."""
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // stride
