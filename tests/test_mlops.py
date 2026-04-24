"""MLOps 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_kia.mlops import drift_monitor


def test_psi_no_drift_is_small():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    cur = rng.normal(size=1000)
    psi = drift_monitor.population_stability_index(ref, cur)
    assert psi < 0.1


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=1000)
    cur = rng.normal(2, 1, size=1000)  # 평균 2 이동
    psi = drift_monitor.population_stability_index(ref, cur)
    assert psi > 0.3


def test_numeric_drift_result_flags_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, size=1000), name="x")
    cur = pd.Series(rng.normal(3, 1, size=1000), name="x")
    result = drift_monitor.detect_numeric_drift(ref, cur)
    assert result.drift is True
    assert result.psi > 0.2


def test_numeric_drift_same_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, size=2000), name="x")
    cur = pd.Series(rng.normal(0, 1, size=2000), name="x")
    result = drift_monitor.detect_numeric_drift(ref, cur)
    assert result.drift is False


def test_drift_report_returns_dataframe():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": rng.normal(size=500),
            "Type": rng.choice(["L", "M", "H"], size=500),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(2, 1, size=500),  # drift
            "b": rng.normal(size=500),         # no drift
            "Type": rng.choice(["L", "M", "H"], size=500, p=[0.1, 0.1, 0.8]),  # drift
        }
    )
    report = drift_monitor.drift_report(
        ref, cur,
        numeric_cols=["a", "b"],
        categorical_cols=["Type"],
    )
    assert "feature" in report.columns
    assert set(report["feature"]) == {"a", "b", "Type"}
    # 'a' 는 drift, 'b' 는 not drift
    drift_map = dict(zip(report["feature"], report["drift"]))
    assert drift_map["a"] is True
    assert drift_map["b"] is False
