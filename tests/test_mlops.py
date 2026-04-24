"""MLOps 단위 테스트."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from career_kia.mlops import drift_monitor, mlflow_utils


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


def test_configure_sets_tracking_uri(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    mlflow_utils.configure(tracking_uri=tracking_uri, experiment="test-exp")
    import mlflow
    assert mlflow.get_tracking_uri() == tracking_uri


def test_register_model_creates_version(tmp_path):
    """register_model → list_model_versions 왕복으로 버저닝 확인."""
    mlflow_utils.configure(
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        experiment="test-register",
    )
    # 더미 모델 파일
    model_file = tmp_path / "dummy_model.joblib"
    joblib.dump({"a": 1}, model_file)

    uri1 = mlflow_utils.register_model(
        model_file, "test-dummy-model", tags={"k": "v"}
    )
    assert uri1.startswith("models:/test-dummy-model/")

    versions = mlflow_utils.list_model_versions("test-dummy-model")
    assert len(versions) == 1
    assert int(versions[0]["version"]) == 1

    # 동일 모델 재등록 → version 2
    uri2 = mlflow_utils.register_model(model_file, "test-dummy-model")
    assert uri2.endswith("/2")
    versions = mlflow_utils.list_model_versions("test-dummy-model")
    assert len(versions) == 2
    assert {int(v["version"]) for v in versions} == {1, 2}


def test_compare_runs_ranks_by_metric(tmp_path):
    import mlflow
    mlflow_utils.configure(
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        experiment="test-compare",
    )
    # 3 run with different metric values
    for name, score in [("a", 0.7), ("b", 0.9), ("c", 0.8)]:
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("roc_auc_mean", score)

    leaderboard = mlflow_utils.compare_runs(
        experiment_name="test-compare", metric="roc_auc_mean", top_k=3
    )
    assert len(leaderboard) == 3
    assert leaderboard[0]["run_name"] == "b"
    assert leaderboard[-1]["run_name"] == "a"


def test_compare_runs_nonexistent_experiment_returns_empty():
    assert mlflow_utils.compare_runs(experiment_name="definitely-not-exists-xyz") == []


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
