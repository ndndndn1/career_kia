"""모델 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_kia.models.baselines import make_logistic_baseline, make_rf_baseline
from career_kia.models.hybrid import HybridConfig, HybridModel


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {
            "Tool wear [min]": rng.integers(0, 250, size=n),
            "Torque [Nm]": rng.normal(40, 10, size=n),
            "Rotational speed [rpm]": rng.normal(1500, 200, size=n),
            "Process temperature [K]": rng.normal(310, 2, size=n),
            "Air temperature [K]": rng.normal(300, 2, size=n),
            "t_rms_mean": rng.normal(0, 1, size=n),
        }
    )
    # 간단한 합성 라벨 — Tool wear와 Torque 기반
    lin = 0.01 * X["Tool wear [min]"] + 0.05 * X["Torque [Nm]"] - 3.0
    proba = 1 / (1 + np.exp(-lin))
    y = (rng.random(n) < proba).astype(int)
    return X, y


def test_logistic_baseline_fit_predict(toy_dataset):
    X, y = toy_dataset
    model = make_logistic_baseline().fit(X.to_numpy(), y)
    proba = model.predict_proba(X.to_numpy())[:, 1]
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_rf_baseline_fit_predict(toy_dataset):
    X, y = toy_dataset
    model = make_rf_baseline().fit(X.to_numpy(), y)
    proba = model.predict_proba(X.to_numpy())[:, 1]
    assert proba.shape == (len(y),)


def test_hybrid_model_end_to_end(toy_dataset):
    X, y = toy_dataset
    model = HybridModel(config=HybridConfig()).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_hybrid_partial_dependence(toy_dataset):
    X, y = toy_dataset
    model = HybridModel(config=HybridConfig()).fit(X, y)
    grid, pdep = model.partial_dependence("Tool wear [min]")
    assert len(grid) == len(pdep)
    assert len(grid) > 0


def test_hybrid_requires_dataframe():
    model = HybridModel(config=HybridConfig())
    with pytest.raises(TypeError):
        model.fit(np.zeros((10, 5)), np.zeros(10))


def test_hybrid_raises_before_fit():
    model = HybridModel(config=HybridConfig())
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"a": [1]}))
