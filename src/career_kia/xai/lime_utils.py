"""LIME 유틸 — SHAP 과 교차 검증용 국소 설명.

LIME 은 샘플 근방에서 sparse linear model 을 적합하므로, SHAP 과
해석이 일치하면 설명 신뢰도가 높아진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from career_kia.models.hybrid import HybridModel


@dataclass
class LimeExplanation:
    feature_values: list[tuple[str, float, float]]  # (rule, value, weight)
    intercept: float


def build_lime_explainer(
    model: HybridModel,
    X_background: pd.DataFrame,
    *,
    training_labels: np.ndarray | None = None,
) -> LimeTabularExplainer:
    """백그라운드 분포에서 LIME 설명기 생성."""
    X_san = model._prepare_X(X_background)
    return LimeTabularExplainer(
        training_data=X_san.to_numpy(),
        feature_names=list(X_san.columns),
        class_names=["정상", "고장"],
        training_labels=training_labels,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def explain_instance(
    model: HybridModel,
    instance: pd.Series,
    *,
    explainer: LimeTabularExplainer,
    num_features: int = 10,
) -> LimeExplanation:
    """단일 인스턴스에 대한 LIME 설명."""
    X_san = model._prepare_X(pd.DataFrame([instance]))

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=X_san.columns)
        # LIME 은 sanitize 된 컬럼을 넘김 — HybridModel 에는 원래 이름이 필요
        df.columns = model.feature_names_
        return model.predict_proba(df)

    exp = explainer.explain_instance(
        data_row=X_san.iloc[0].to_numpy(),
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=1500,
    )
    # exp.as_list() 는 [(rule_str, weight), ...]
    rules = []
    for rule, weight in exp.as_list(label=1):
        rules.append((rule, 0.0, float(weight)))
    return LimeExplanation(feature_values=rules, intercept=float(exp.intercept[1]))
