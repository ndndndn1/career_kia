"""SHAP 유틸 — HybridModel 의 LightGBM 단계에 TreeExplainer 적용.

왜 LightGBM 단계인가?
    하이브리드 모델은 LGBM → pyGAM 스태킹. 피처 레벨 설명은 LGBM 쪽이
    책임지고(상호작용 포착), pyGAM 은 글로벌 해석을 제공. 변수 기여도·임계
    구간·상호작용 모두 LGBM SHAP 에서 산출한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from career_kia.models.hybrid import HybridModel


@dataclass
class ShapBundle:
    """SHAP 결과 묶음 — 대시보드/설명 템플릿 공통 입력."""

    values: np.ndarray               # (N, F) SHAP 값
    base_value: float                # 기준값 (확률이 아닌 logit 공간)
    feature_names: list[str]
    X: pd.DataFrame                  # 설명 대상 샘플

    @property
    def mean_abs(self) -> pd.Series:
        return pd.Series(
            np.abs(self.values).mean(axis=0), index=self.feature_names
        ).sort_values(ascending=False)

    def top_k(self, k: int = 10) -> pd.Series:
        return self.mean_abs.head(k)


def build_tree_explainer(model: HybridModel) -> shap.TreeExplainer:
    """HybridModel 의 내부 LGBM 을 감싼 TreeExplainer."""
    if model.lgbm_ is None:
        raise RuntimeError("모델이 학습되지 않았습니다.")
    return shap.TreeExplainer(model.lgbm_)


def explain_batch(
    model: HybridModel,
    X: pd.DataFrame,
    *,
    explainer: shap.TreeExplainer | None = None,
    max_samples: int | None = None,
) -> ShapBundle:
    """한 배치(또는 서브셋)에 대한 SHAP 값 산출.

    Returns
    -------
    ShapBundle — 양성 클래스 기준 SHAP 값.
    """
    if max_samples is not None and len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    explainer = explainer or build_tree_explainer(model)
    X_san = model._prepare_X(X)
    shap_values = explainer.shap_values(X_san)
    # LightGBM 이진분류: shap_values 는 (N, F) 또는 [(N,F),(N,F)] 구조
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # 새 shap 에서는 (N, F, 2) 로 올 수 있음
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    return ShapBundle(
        values=shap_values,
        base_value=float(base),
        feature_names=list(X.columns),
        X=X.reset_index(drop=True),
    )


def top_contributors(
    bundle: ShapBundle,
    sample_idx: int,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """단일 샘플의 SHAP 기여도 상위 k 개."""
    vals = bundle.values[sample_idx]
    row = bundle.X.iloc[sample_idx]
    df = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "value": row.values,
            "shap": vals,
            "abs_shap": np.abs(vals),
        }
    )
    return df.sort_values("abs_shap", ascending=False).head(k).reset_index(drop=True)


def interaction_values(
    model: HybridModel,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> np.ndarray:
    """SHAP 상호작용 값 — (N, F, F). 비용이 커서 샘플링으로 제한."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    explainer = build_tree_explainer(model)
    X_san = model._prepare_X(X)
    inter = explainer.shap_interaction_values(X_san)
    if isinstance(inter, list):
        inter = inter[1]
    if inter.ndim == 4:
        inter = inter[..., 1]
    return inter


def top_interactions(
    inter: np.ndarray,
    feature_names: list[str],
    *,
    k: int = 10,
) -> pd.DataFrame:
    """상호작용 강도 상위 쌍."""
    abs_mean = np.abs(inter).mean(axis=0)  # (F, F)
    # 대각선 제외, 상삼각만
    n = abs_mean.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((feature_names[i], feature_names[j], float(abs_mean[i, j] * 2)))
    df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "mean_abs_interaction"])
    return df.sort_values("mean_abs_interaction", ascending=False).head(k).reset_index(drop=True)
