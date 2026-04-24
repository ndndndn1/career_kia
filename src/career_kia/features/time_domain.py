"""시간 도메인 피처.

베어링 진단 표준 피처 세트::

    RMS, Peak, Peak-to-Peak, Mean, Std, Variance, Skewness, Kurtosis,
    Crest Factor, Impulse Factor, Shape Factor, Margin Factor,
    Clearance Factor, Zero Crossings, Energy

모든 함수는 (N, W) 형태의 윈도우 배열에 대해 벡터화 동작한다.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x**2, axis=-1))


def peak(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x), axis=-1)


def peak_to_peak(x: np.ndarray) -> np.ndarray:
    return np.ptp(x, axis=-1)


def crest_factor(x: np.ndarray) -> np.ndarray:
    r = rms(x)
    return np.where(r > 0, peak(x) / r, 0.0)


def impulse_factor(x: np.ndarray) -> np.ndarray:
    m = np.mean(np.abs(x), axis=-1)
    return np.where(m > 0, peak(x) / m, 0.0)


def shape_factor(x: np.ndarray) -> np.ndarray:
    m = np.mean(np.abs(x), axis=-1)
    return np.where(m > 0, rms(x) / m, 0.0)


def margin_factor(x: np.ndarray) -> np.ndarray:
    """Peak / (mean(sqrt(|x|)))^2 — 임펄스성에 더 민감한 지표.

    문헌에 따라 clearance factor 라고도 불린다.
    """
    denom = np.mean(np.sqrt(np.abs(x)), axis=-1) ** 2
    return np.where(denom > 0, peak(x) / denom, 0.0)


def zero_crossing_rate(x: np.ndarray) -> np.ndarray:
    signs = np.sign(x)
    return np.sum(signs[..., :-1] * signs[..., 1:] < 0, axis=-1).astype(float) / x.shape[-1]


def energy(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def skewness(x: np.ndarray) -> np.ndarray:
    return skew(x, axis=-1)


def kurt(x: np.ndarray) -> np.ndarray:
    """첨도 (Fisher 정의: 정규분포가 0)."""
    return kurtosis(x, axis=-1, fisher=True)


def compute_time_features(windows: np.ndarray) -> dict[str, np.ndarray]:
    """(N, W) 윈도우에서 시간 도메인 피처 dict 산출."""
    return {
        "rms": rms(windows),
        "peak": peak(windows),
        "p2p": peak_to_peak(windows),
        "mean": np.mean(windows, axis=-1),
        "std": np.std(windows, axis=-1),
        "var": np.var(windows, axis=-1),
        "skew": skewness(windows),
        "kurtosis": kurt(windows),
        "crest_factor": crest_factor(windows),
        "impulse_factor": impulse_factor(windows),
        "shape_factor": shape_factor(windows),
        "margin_factor": margin_factor(windows),
        "zero_crossing_rate": zero_crossing_rate(windows),
        "energy": energy(windows),
    }


TIME_FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "peak",
    "p2p",
    "mean",
    "std",
    "var",
    "skew",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "margin_factor",
    "zero_crossing_rate",
    "energy",
)
