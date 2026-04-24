"""피처 엔지니어링 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from career_kia.features import freq_domain, time_domain, windowing


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------

def test_make_windows_shape():
    sig = np.arange(10_000, dtype=float)
    w = windowing.make_windows(sig, window_size=1000, stride=500)
    expected = 1 + (10_000 - 1000) // 500
    assert w.shape == (expected, 1000)


def test_make_windows_values():
    sig = np.arange(20, dtype=float)
    w = windowing.make_windows(sig, window_size=5, stride=2)
    # 첫 윈도우: [0,1,2,3,4], 두 번째: [2,3,4,5,6], ...
    np.testing.assert_array_equal(w[0], np.arange(5))
    np.testing.assert_array_equal(w[1], np.arange(2, 7))


def test_make_windows_short_signal_drop_last():
    sig = np.arange(100, dtype=float)
    w = windowing.make_windows(sig, window_size=1000, stride=500, drop_last=True)
    assert w.shape == (0, 1000)


def test_make_windows_short_signal_pad():
    sig = np.arange(100, dtype=float)
    w = windowing.make_windows(sig, window_size=1000, stride=500, drop_last=False)
    assert w.shape == (1, 1000)
    assert w[0, :100].tolist() == list(range(100))
    assert w[0, 100:].sum() == 0


def test_window_count_agrees():
    n = 12_345
    assert windowing.window_count(n, window_size=1000, stride=500) == windowing.make_windows(
        np.arange(n, dtype=float), window_size=1000, stride=500
    ).shape[0]


# ---------------------------------------------------------------------------
# time_domain
# ---------------------------------------------------------------------------

def test_rms_of_sine_is_amplitude_over_sqrt2():
    amp = 2.0
    t = np.linspace(0, 1, 10_000, endpoint=False)
    sig = amp * np.sin(2 * np.pi * 10 * t)
    w = sig.reshape(1, -1)
    np.testing.assert_allclose(time_domain.rms(w), amp / np.sqrt(2), rtol=1e-2)


def test_time_features_output_keys():
    w = np.random.RandomState(0).normal(size=(5, 1024))
    feats = time_domain.compute_time_features(w)
    for key in time_domain.TIME_FEATURE_NAMES:
        assert key in feats
        assert feats[key].shape == (5,)


def test_kurtosis_flags_impulsive_signal():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2048)
    impulsive = base.copy()
    impulsive[::100] += 10.0  # 명확한 임펄스
    w = np.stack([base, impulsive])
    k = time_domain.kurt(w)
    assert k[1] > k[0]


def test_crest_factor_high_for_impulsive():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2048)
    impulsive = base.copy()
    impulsive[::100] += 10.0
    w = np.stack([base, impulsive])
    cf = time_domain.crest_factor(w)
    assert cf[1] > cf[0]


# ---------------------------------------------------------------------------
# freq_domain
# ---------------------------------------------------------------------------

def test_fft_spectrum_peak_location():
    fs = 2000
    f0 = 100
    t = np.arange(0, 1, 1 / fs)
    sig = np.sin(2 * np.pi * f0 * t).reshape(1, -1)
    freqs, amps = freq_domain.fft_spectrum(sig, fs=fs)
    peak_idx = np.argmax(amps[0])
    assert abs(freqs[peak_idx] - f0) < 2


def test_band_power_in_correct_band():
    fs = 2000
    f0 = 150
    t = np.arange(0, 1, 1 / fs)
    sig = np.sin(2 * np.pi * f0 * t).reshape(1, -1)
    bp = freq_domain.band_power(sig, fs=fs, bands=[(0, 100), (100, 200), (200, 400)])
    # 150Hz 신호는 100-200Hz 대역에 집중
    assert bp[0, 1] > bp[0, 0]
    assert bp[0, 1] > bp[0, 2]


def test_envelope_spectrum_detects_modulation():
    """AM 변조된 고주파 — 포락선 스펙트럼이 변조 주파수를 잡아야 함."""
    fs = 12_000
    t = np.arange(0, 1, 1 / fs)
    carrier = np.cos(2 * np.pi * 3000 * t)
    modulation = 1 + 0.5 * np.cos(2 * np.pi * 157 * t)  # 변조 주파수 = BPFI 근접
    sig = (carrier * modulation).reshape(1, -1).astype(float)
    freqs, env = freq_domain.envelope_spectrum(sig, fs=fs)
    # 변조 주파수 주변 피크
    idx = np.argmax(env[0, (freqs > 140) & (freqs < 180)])
    band_freqs = freqs[(freqs > 140) & (freqs < 180)]
    assert abs(band_freqs[idx] - 157) < 3


def test_compute_freq_features_keys():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 2048))
    feats = freq_domain.compute_freq_features(w, fs=12_000)
    assert "spec_entropy" in feats
    assert "env_BPFI" in feats
    assert "env_BPFO" in feats
    assert all(v.shape == (4,) for v in feats.values())


def test_spectral_entropy_bounded():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 2048))
    ent = freq_domain.spectral_entropy(w, fs=12_000)
    assert np.all(ent >= 0)
    assert np.all(ent <= 1.0)


def test_bearing_fault_amplitude_higher_for_fault():
    """포락선 피크가 BPFI 주변에 실린 신호가 백색잡음보다 큰 BPFI 진폭."""
    fs = 12_000
    t = np.arange(0, 1, 1 / fs)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(t))
    modulated = noise * (1 + 0.8 * np.cos(2 * np.pi * 157.94 * t))
    w = np.stack([noise, modulated])
    out = freq_domain.bearing_fault_amplitudes(w, fs=fs)
    assert out["BPFI"][1] > out["BPFI"][0]
