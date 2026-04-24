"""인과추론 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_kia.causal import dag, intervention, time_series


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    n = 600
    tool_wear = rng.integers(0, 250, size=n).astype(float)
    torque = 40 + 0.05 * tool_wear + rng.normal(0, 5, size=n)
    rot_speed = rng.normal(1500, 200, size=n)
    # Machine failure 가 tool_wear 와 torque 에 의해 결정되는 합성
    logit = -5 + 0.02 * tool_wear + 0.08 * torque
    proba = 1 / (1 + np.exp(-logit))
    failure = (rng.random(n) < proba).astype(int)

    return pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Tool wear [min]": tool_wear,
            "Air temperature [K]": rng.normal(300, 2, size=n),
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Process temperature [K]": 310 + 0.01 * torque + rng.normal(0, 1, size=n),
            "t_rms_mean": rng.normal(0.01, 0.005, size=n),
            "Machine failure": failure,
        }
    )


def test_dag_has_expected_nodes():
    graph = dag.get_default_graph()
    for node in ["Tool_wear", "Torque", "Machine_failure", "Process_temperature"]:
        assert node in graph


def test_column_mapping_consistent():
    for node, col in dag.NODE_TO_COLUMN.items():
        assert isinstance(col, str)
        assert " " in col or "_" in col or col == "Type"


def test_estimate_ate_tool_wear_positive(toy_df):
    # 합성 라벨이 tool_wear 양의 계수 → ATE 가 양수여야 함
    res, _, _ = intervention.estimate_ate(
        toy_df,
        treatment="Tool_wear",
        treatment_value=200.0,
        control_value=50.0,
        method="backdoor.linear_regression",
    )
    assert res.treatment == "Tool_wear"
    assert res.outcome == "Machine_failure"
    assert res.estimate > 0  # 마모가 커지면 고장 확률 상승


def test_refutation_random_common_cause_preserves_sign(toy_df):
    """무의미한 공통 원인을 추가해도 원 효과의 부호가 유지되어야 한다."""
    res, model, estimate = intervention.estimate_ate(
        toy_df,
        treatment="Tool_wear",
        treatment_value=200.0,
        control_value=50.0,
        method="backdoor.linear_regression",
    )
    refute = intervention.refute_estimate(
        model, estimate, methods=("random_common_cause",)
    )
    assert not np.isnan(refute["random_common_cause"])
    # 부호는 유지
    assert np.sign(refute["random_common_cause"]) == np.sign(res.estimate)
    # 상대 변화는 과도하지 않아야 함 (50% 이내)
    rel = abs(refute["random_common_cause"] - res.estimate) / max(abs(res.estimate), 1e-9)
    assert rel < 0.5


def test_whatif_dose_response_monotone(toy_df):
    dr = intervention.whatif_dose_response(
        toy_df,
        treatment="Tool_wear",
        grid=np.array([50.0, 100.0, 150.0, 200.0]),
        baseline=50.0,
    )
    assert len(dr) == 4
    # 선형 회귀라 tool wear 증가에 따라 ATE 도 증가
    assert dr["ate"].is_monotonic_increasing


def test_granger_matrix_shape(toy_df):
    sub = toy_df[["Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"]].iloc[:200]
    pmat = time_series.granger_causality_matrix(sub, max_lag=2)
    assert pmat.shape == (3, 3)
    assert (np.diag(pmat.to_numpy()) == 1.0).all()


def test_pcmci_note_non_empty():
    assert len(time_series.pcmci_note()) > 20
