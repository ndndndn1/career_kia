"""XAI 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_kia.models.hybrid import HybridConfig, HybridModel
from career_kia.xai import explanation_templates, nl_generator, shap_utils


@pytest.fixture(scope="module")
def trained_model():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "Tool wear [min]": rng.integers(0, 250, size=n).astype(float),
            "Torque [Nm]": rng.normal(40, 10, size=n),
            "Rotational speed [rpm]": rng.normal(1500, 200, size=n),
            "Process temperature [K]": rng.normal(310, 2, size=n),
            "Air temperature [K]": rng.normal(300, 2, size=n),
            "t_rms_mean": rng.normal(0, 1, size=n),
        }
    )
    lin = 0.01 * X["Tool wear [min]"] + 0.05 * X["Torque [Nm]"] - 3.0
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    model = HybridModel(config=HybridConfig()).fit(X, y)
    return model, X, y


def test_shap_bundle_shape(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(50))
    assert bundle.values.shape == (50, X.shape[1])
    assert len(bundle.feature_names) == X.shape[1]


def test_shap_top_k(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(30))
    top = bundle.top_k(k=3)
    assert len(top) == 3
    assert top.iloc[0] >= top.iloc[-1]


def test_top_contributors_sorted(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(30))
    df = shap_utils.top_contributors(bundle, sample_idx=0, k=3)
    assert len(df) == 3
    assert df["abs_shap"].is_monotonic_decreasing


def test_interaction_top_pairs(trained_model):
    model, X, _ = trained_model
    inter = shap_utils.interaction_values(model, X, max_samples=80)
    pairs = shap_utils.top_interactions(inter, list(X.columns), k=3)
    assert len(pairs) == 3
    assert pairs["mean_abs_interaction"].is_monotonic_decreasing


def test_build_contribution_explanation(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(5))
    exp = explanation_templates.build_contribution(
        bundle, sample_idx=0, prediction=0.7, sample_id="B000001"
    )
    assert exp.sample_id == "B000001"
    assert exp.prediction == 0.7
    assert len(exp.contributions) <= 5


def test_infer_thresholds(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(200))
    thresholds = explanation_templates.infer_thresholds(
        bundle, features=["Tool wear [min]", "Torque [Nm]"], bins=5, min_shap=0.01
    )
    # 학습 라벨이 Tool wear/Torque 로 만들어졌으므로 최소 하나는 나와야 함
    assert len(thresholds) > 0


def test_nl_generator_contribution_sentences(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(3))
    exp = explanation_templates.build_contribution(
        bundle, sample_idx=0, prediction=0.85, sample_id="B000001"
    )
    sentences = nl_generator.contribution_to_sentences(exp)
    joined = "\n".join(sentences)
    assert "B000001" in joined
    assert "%" in joined


def test_nl_generator_full_paragraph(trained_model):
    model, X, _ = trained_model
    bundle = shap_utils.explain_batch(model, X.head(20))
    exp = explanation_templates.build_contribution(
        bundle, sample_idx=0, prediction=0.9, sample_id="B000002"
    )
    thresholds = explanation_templates.infer_thresholds(
        bundle, features=["Tool wear [min]"], bins=5, min_shap=0.01
    )
    batch_exp = explanation_templates.BatchExplanation(
        contribution=exp, thresholds=thresholds
    )
    paragraph = nl_generator.batch_explanation_to_paragraph(batch_exp)
    assert isinstance(paragraph, str)
    assert len(paragraph) > 50


def test_nl_feature_label_falls_back_to_raw():
    # 매핑 없는 피처명이면 원 이름 그대로
    assert nl_generator._label("unknown_feature_xyz") == "unknown_feature_xyz"


def test_nl_feature_label_maps_known():
    assert nl_generator._label("Tool wear [min]") == "공구 마모 시간"
