"""신호 필터링.

Butterworth 대역통과/고역/저역 필터와 Wavelet denoising을 제공한다.
베어링 결함 분석에서는 충격성 성분을 보존하는 것이 중요하므로
- zero-phase 필터(`filtfilt`) 기본 사용
- Wavelet은 soft thresholding (VisuShrink)으로 결함 임펄스 보존
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pywt
from scipy.signal import butter, filtfilt

FilterType = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def butterworth(
    signal: np.ndarray,
    fs: int,
    *,
    cutoff: float | tuple[float, float],
    filter_type: FilterType = "bandpass",
    order: int = 4,
) -> np.ndarray:
    """Butterworth 필터 (zero-phase).

    Parameters
    ----------
    signal
        1D 입력 신호.
    fs
        샘플링 주파수 (Hz).
    cutoff
        저역/고역은 스칼라, 대역통과/저지대는 (low, high).
    filter_type
        'lowpass' / 'highpass' / 'bandpass' / 'bandstop'
    order
        필터 차수.
    """
    nyq = fs / 2.0
    if filter_type in ("bandpass", "bandstop"):
        if not isinstance(cutoff, tuple):
            raise TypeError("bandpass/bandstop 은 (low, high) 튜플 cutoff 필요")
        wn = (cutoff[0] / nyq, cutoff[1] / nyq)
    else:
        if isinstance(cutoff, tuple):
            raise TypeError("lowpass/highpass 는 단일 스칼라 cutoff 필요")
        wn = cutoff / nyq
    b, a = butter(order, wn, btype=filter_type)
    return filtfilt(b, a, signal)


def wavelet_denoise(
    signal: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
    mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    """웨이블릿 기반 잡음 제거 (VisuShrink threshold).

    Notes
    -----
    임펄스 성분(베어링 결함의 특성)을 보존하기 위해 soft thresholding을 기본으로 한다.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 최상세 계수의 MAD 기반 노이즈 표준편차 추정
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]
