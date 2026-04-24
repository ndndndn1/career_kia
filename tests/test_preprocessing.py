"""전처리 모듈 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_kia.preprocessing import filtering, imputation, outliers, synchronization


# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------

def test_linear_impute_fills_internal_nan():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    out = imputation.impute_series(s, method="linear")
    assert out.isna().sum() == 0
    # 선형: 1 → 2 → 3 → 4 → 5 → 6
    np.testing.assert_allclose(out.to_numpy(), [1, 2, 3, 4, 5, 6], atol=1e-6)


def test_spline_impute_preserves_curvature():
    x = np.arange(20, dtype=float)
    y = np.sin(x / 3.0)
    # pd.Series(y) 가 y 와 메모리 공유하지 않도록 복사
    s = pd.Series(y.copy())
    s.iloc[[5, 6, 7, 12, 13]] = np.nan
    out = imputation.impute_series(s, method="spline")
    assert out.isna().sum() == 0
    # 큐빅 스플라인은 원 함수에 매우 가까워야 한다
    assert np.max(np.abs(out.to_numpy() - y)) < 0.1


def test_knn_dataframe_impute():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
            "c": rng.normal(size=50),
        }
    )
    df.iloc[3, 0] = np.nan
    df.iloc[7, 1] = np.nan
    out = imputation.impute_dataframe(df, method="knn")
    assert out.isna().sum().sum() == 0


def test_summarize_missing():
    df = pd.DataFrame({"x": [1, None, 3], "y": [1, 2, 3]})
    summary = imputation.summarize_missing(df)
    assert "x" in summary.index
    assert "y" not in summary.index


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def test_butterworth_bandpass_attenuates_out_of_band():
    fs = 2000
    t = np.arange(0, 1, 1 / fs)
    low = np.sin(2 * np.pi * 10 * t)   # 10 Hz (통과대역 밖)
    high = np.sin(2 * np.pi * 500 * t)  # 500 Hz (통과대역 내)
    sig = low + high
    out = filtering.butterworth(
        sig, fs=fs, cutoff=(100, 800), filter_type="bandpass", order=4
    )
    # 10 Hz 성분이 거의 제거되어야 하므로 RMS가 원 신호보다 작아야 함
    assert np.std(out) < np.std(sig)
    # 필터링된 신호는 500 Hz 사인에 더 가까워야 함
    assert np.corrcoef(out, high)[0, 1] > 0.9


def test_wavelet_denoise_reduces_noise():
    rng = np.random.default_rng(0)
    n = 2048
    clean = np.sin(2 * np.pi * 5 * np.arange(n) / 1000)
    noise = rng.normal(0, 0.3, size=n)
    noisy = clean + noise
    denoised = filtering.wavelet_denoise(noisy, level=4)
    # 잡음 에너지가 줄어야 함
    assert np.std(denoised - clean) < np.std(noisy - clean)


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------

def test_iqr_mask_flags_extremes():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    mask = outliers.iqr_mask(s)
    assert mask.iloc[-1]
    assert not mask.iloc[:-1].any()


def test_mad_mask_robust_to_single_outlier():
    rng = np.random.default_rng(0)
    s = pd.Series(np.concatenate([rng.normal(10, 0.5, size=50), [1000.0]]))
    mask = outliers.mad_mask(s, threshold=3.5)
    assert mask.iloc[-1]
    assert mask.iloc[:-1].sum() == 0


def test_mad_mask_all_identical_returns_no_outliers():
    # MAD=0 경계: 전부 동일한 값이면 이상치 없음으로 취급
    s = pd.Series([10.0] * 50)
    mask = outliers.mad_mask(s)
    assert not mask.any()


def test_rolling_zscore_detects_spike():
    n = 200
    s = pd.Series(np.sin(np.arange(n) / 10.0))
    s.iloc[100] += 10
    mask = outliers.rolling_zscore_mask(s, window=30, threshold=3.0)
    assert mask.iloc[100]


def test_isolation_forest_flags_cluster_outlier():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 3)))
    df.iloc[0] = [10.0, 10.0, 10.0]  # 명확한 이상치
    mask = outliers.isolation_forest_mask(df, contamination=0.02)
    assert mask.iloc[0]


@pytest.mark.parametrize("fill", ["nan", "clip", "median"])
def test_apply_mask_variants(fill: str):
    s = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
    mask = outliers.iqr_mask(s)
    out = outliers.apply_mask(s, mask, fill=fill)
    if fill == "nan":
        assert np.isnan(out.iloc[-1])
    else:
        assert out.iloc[-1] < 100


# ---------------------------------------------------------------------------
# synchronization
# ---------------------------------------------------------------------------

def test_resample_signal_identity():
    sig = np.arange(100, dtype=float)
    out = synchronization.resample_signal(sig, fs_in=1000, fs_out=1000)
    np.testing.assert_allclose(out, sig)


def test_resample_signal_downsamples():
    sig = np.sin(2 * np.pi * 10 * np.arange(48_000) / 48_000)
    out = synchronization.resample_signal(sig, fs_in=48_000, fs_out=12_000)
    assert len(out) == 12_000


def test_align_to_timegrid_combines_two_sensors():
    idx1 = pd.date_range("2026-01-01", periods=10, freq="100ms")
    idx2 = pd.date_range("2026-01-01", periods=5, freq="200ms")
    df1 = pd.DataFrame({"v": range(10)}, index=idx1)
    df2 = pd.DataFrame({"v": range(5)}, index=idx2)
    merged = synchronization.align_to_timegrid(
        {"s1": df1, "s2": df2}, freq="200ms", how="mean"
    )
    assert "s1__v" in merged.columns
    assert "s2__v" in merged.columns


def test_merge_asof_aligned_backward_join():
    left = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 00:01", "2026-01-01 00:05"])}
    )
    right = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:03"]),
            "event": ["A", "B"],
        }
    )
    out = synchronization.merge_asof_aligned(left, right, tolerance="10min")
    assert list(out["event"]) == ["A", "B"]
