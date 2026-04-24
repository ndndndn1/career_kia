"""주파수 도메인 피처.

베어링 진단의 핵심인 **Envelope Spectrum**과 일반 스펙트럴 피처를 제공한다.

- fft_spectrum      : 단일 사이드 진폭 스펙트럼
- band_power        : 주파수 대역별 에너지
- envelope_spectrum : 힐버트 변환 → 포락선의 FFT (BPFI/BPFO 감지에 핵심)
- bearing_fault_amplitudes : 베어링 특성 주파수 진폭
- spectral_entropy  : 스펙트럴 엔트로피
- spectral_centroid : 스펙트럴 중심
- psd_welch         : Welch 기반 PSD (노이즈 강건)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch

# CWRU 1797rpm 회전축의 대표 베어링 특성 주파수 (Hz) — 6205 2RS JEM SKF 기준
DEFAULT_FAULT_FREQS_HZ: dict[str, float] = {
    "BPFI": 157.94,  # Inner race
    "BPFO": 104.56,  # Outer race
    "BSF": 69.52,    # Ball spin
    "FTF": 11.85,    # Cage (Fundamental Train)
}


def fft_spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """단일 사이드 진폭 스펙트럼.

    Parameters
    ----------
    x
        (N, W) 또는 (W,) 형태 신호.
    fs
        샘플링 주파수.

    Returns
    -------
    freqs : (W//2+1,)
    amps  : x 와 같은 선행 차원 + 마지막 축이 freqs 와 동일한 길이
    """
    w = x.shape[-1]
    freqs = np.fft.rfftfreq(w, d=1 / fs)
    spec = np.abs(np.fft.rfft(x, axis=-1)) * 2 / w
    return freqs, spec


def band_power(
    x: np.ndarray,
    fs: int,
    bands: list[tuple[float, float]],
) -> np.ndarray:
    """주파수 대역별 에너지 합.

    Returns
    -------
    (N, len(bands)) — 각 윈도우 × 각 대역의 에너지
    """
    freqs, amps = fft_spectrum(x, fs)
    power = amps**2
    out = np.empty((*x.shape[:-1], len(bands)))
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        out[..., i] = power[..., mask].sum(axis=-1)
    return out


def envelope_spectrum(
    x: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert 포락선의 FFT.

    베어링 결함은 고주파 공진이 저주파 결함 주파수로 변조되므로, 포락선을
    취한 후 FFT 하면 결함 주파수(BPFI 등)가 뚜렷한 피크로 나타난다.
    """
    analytic = hilbert(x, axis=-1)
    envelope = np.abs(analytic)
    # DC 제거
    envelope = envelope - envelope.mean(axis=-1, keepdims=True)
    w = x.shape[-1]
    freqs = np.fft.rfftfreq(w, d=1 / fs)
    env_spec = np.abs(np.fft.rfft(envelope, axis=-1)) * 2 / w
    return freqs, env_spec


def bearing_fault_amplitudes(
    x: np.ndarray,
    fs: int,
    *,
    fault_freqs: dict[str, float] | None = None,
    tolerance_hz: float = 3.0,
) -> dict[str, np.ndarray]:
    """포락선 스펙트럼에서 특성 주파수 주변 최대 진폭.

    Returns
    -------
    {'BPFI': (N,), 'BPFO': (N,), ...}
    """
    ff = fault_freqs if fault_freqs is not None else DEFAULT_FAULT_FREQS_HZ
    freqs, env = envelope_spectrum(x, fs)
    out: dict[str, np.ndarray] = {}
    for name, target in ff.items():
        mask = (freqs >= target - tolerance_hz) & (freqs <= target + tolerance_hz)
        if not mask.any():
            out[name] = np.zeros(x.shape[:-1])
            continue
        out[name] = env[..., mask].max(axis=-1)
    return out


def spectral_entropy(x: np.ndarray, fs: int, *, eps: float = 1e-12) -> np.ndarray:
    """스펙트럴 엔트로피 (Shannon, 로그2 기반, 정규화)."""
    _, amps = fft_spectrum(x, fs)
    psd = amps**2
    psd_norm = psd / (psd.sum(axis=-1, keepdims=True) + eps)
    psd_norm = np.clip(psd_norm, eps, 1.0)
    ent = -np.sum(psd_norm * np.log2(psd_norm), axis=-1)
    return ent / np.log2(psd.shape[-1])


def spectral_centroid(x: np.ndarray, fs: int, *, eps: float = 1e-12) -> np.ndarray:
    """스펙트럴 중심 주파수 (Hz)."""
    freqs, amps = fft_spectrum(x, fs)
    total = amps.sum(axis=-1) + eps
    return (amps * freqs).sum(axis=-1) / total


def psd_welch(
    x: np.ndarray,
    fs: int,
    *,
    nperseg: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD — 스펙트럼이 노이즈가 많을 때 강건함."""
    freqs, psd = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return freqs, psd


# ---------------------------------------------------------------------------
# 통합 주파수 피처
# ---------------------------------------------------------------------------

DEFAULT_BANDS_HZ: list[tuple[float, float]] = [
    (0, 100),
    (100, 500),
    (500, 1500),
    (1500, 3000),
    (3000, 5000),
    (5000, 6000),
]


def compute_freq_features(
    windows: np.ndarray,
    fs: int,
    *,
    bands: list[tuple[float, float]] | None = None,
    fault_freqs: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """(N, W) 윈도우에서 주파수 도메인 피처 dict 산출."""
    bands = bands if bands is not None else DEFAULT_BANDS_HZ

    features: dict[str, np.ndarray] = {
        "spec_entropy": spectral_entropy(windows, fs),
        "spec_centroid": spectral_centroid(windows, fs),
    }

    bp = band_power(windows, fs, bands)
    for i, (lo, hi) in enumerate(bands):
        features[f"band_{int(lo)}_{int(hi)}"] = bp[..., i]

    fault_amps = bearing_fault_amplitudes(windows, fs, fault_freqs=fault_freqs)
    for name, amp in fault_amps.items():
        features[f"env_{name}"] = amp

    return features
