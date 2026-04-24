"""Phase 5 학습 엔트리 — `make train` 에서 호출.

모든 모델을 동일한 스플릿에서 학습·평가·MLflow 로깅한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from career_kia.config import PROCESSED_DIR, PROJECT_ROOT
from career_kia.mlops.mlflow_utils import (
    compare_runs,
    configure as configure_mlflow,
    list_model_versions,
    register_model,
)
from career_kia.models.baselines import make_logistic_baseline, make_rf_baseline
from career_kia.models.hybrid import HybridConfig, HybridModel

logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "models_artifacts"
REGISTERED_MODEL_NAME = "sdf-xplain-hybrid"

META_DROP = [
    "batch_id",
    "timestamp",
    "shift",
    "operator_id",
    "line_id",
    "vibration_fault_type",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_feature_matrix(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """피처 테이블 → (X, y, groups). groups는 누수 방지용 일자 단위 batch_id 접두."""
    path = Path(path) if path else PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(path)
    y = df["Machine failure"].astype(int)
    # operator_id 단위 GroupKFold — 작업자 간 누수 방지
    groups = df["operator_id"].astype(str)
    # Type(L/M/H)은 카테고리 → 원핫
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)
    X = df.drop(columns=[c for c in META_DROP if c in df.columns])
    return X, y, groups


def precision_at_recall(y_true: np.ndarray, y_proba: np.ndarray, recall_target: float = 0.9) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    # recall >= target 중 precision 최대
    mask = recalls[:-1] >= recall_target
    return float(precisions[:-1][mask].max()) if mask.any() else float("nan")


def eval_model(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision_at_recall_0.9": precision_at_recall(y_true, y_proba, 0.9),
    }


def cross_val_evaluate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    needs_dataframe: bool = False,
) -> dict[str, float]:
    """GroupKFold 교차검증 평가 (집계 평균 + 표준편차)."""
    gkf = GroupKFold(n_splits=n_splits)
    metrics = {"roc_auc": [], "pr_auc": [], "f1": [], "precision_at_recall_0.9": []}
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = model_factory()
        X_tr = X.iloc[tr] if needs_dataframe else X.iloc[tr].to_numpy()
        X_te = X.iloc[te] if needs_dataframe else X.iloc[te].to_numpy()
        model.fit(X_tr, y.iloc[tr].to_numpy())
        proba = model.predict_proba(X_te)[:, 1]
        fold_metrics = eval_model(y.iloc[te].to_numpy(), proba)
        for k, v in fold_metrics.items():
            metrics[k].append(v)
        logger.info("fold %d: %s", fold, {k: round(v, 4) for k, v in fold_metrics.items()})
    return {
        **{f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in metrics.items()},
    }


def run(
    input_path: Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "sdf-xplain",
) -> dict[str, dict[str, float]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configure_mlflow(tracking_uri=tracking_uri, experiment=experiment_name)

    X, y, groups = load_feature_matrix(input_path)
    logger.info("피처 행렬: %s, 양성 비율: %.3f", X.shape, y.mean())
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, float]] = {}

    # --- 1) Logistic baseline ---
    with mlflow.start_run(run_name="logistic_baseline"):
        res = cross_val_evaluate(make_logistic_baseline, X, y, groups, needs_dataframe=False)
        mlflow.log_metrics(res)
        mlflow.log_param("model", "logistic")
        all_results["logistic"] = res

    # --- 2) Random Forest baseline ---
    with mlflow.start_run(run_name="rf_baseline"):
        res = cross_val_evaluate(make_rf_baseline, X, y, groups, needs_dataframe=False)
        mlflow.log_metrics(res)
        mlflow.log_param("model", "random_forest")
        all_results["random_forest"] = res

    # --- 3) Hybrid (LightGBM + pyGAM) ---
    hybrid_run_id: str | None = None
    with mlflow.start_run(run_name="hybrid_lgbm_gam") as active_run:
        hybrid_run_id = active_run.info.run_id
        res = cross_val_evaluate(
            lambda: HybridModel(config=HybridConfig()),
            X,
            y,
            groups,
            needs_dataframe=True,
        )
        mlflow.log_metrics(res)
        mlflow.log_param("model", "hybrid_lgbm_gam")
        all_results["hybrid_lgbm_gam"] = res

        # 최종 전체 데이터 학습한 모델 저장 (XAI/대시보드용)
        final_model = HybridModel(config=HybridConfig()).fit(X, y)
        model_path = ARTIFACT_DIR / "hybrid_model.joblib"
        joblib.dump(final_model, model_path)
        logger.info("최종 하이브리드 모델 저장: %s", model_path)

    # --- 4) 모델 레지스트리 등록 (버저닝) ---
    try:
        uri = register_model(
            model_path,
            REGISTERED_MODEL_NAME,
            tags={
                "roc_auc_mean": f"{res['roc_auc_mean']:.4f}",
                "pr_auc_mean": f"{res['pr_auc_mean']:.4f}",
                "feature_count": str(X.shape[1]),
            },
            existing_run_id=hybrid_run_id,
        )
        logger.info("레지스트리 등록 URI: %s", uri)
        versions = list_model_versions(REGISTERED_MODEL_NAME)
        logger.info("현재 '%s' 버전 수: %d", REGISTERED_MODEL_NAME, len(versions))
    except Exception as exc:  # noqa: BLE001
        # 로컬 파일 백엔드는 registry 기능이 제한적일 수 있음 — 실패해도 학습 산출물은 유지
        logger.warning("모델 레지스트리 등록 실패 (%s) — 로컬 joblib 은 정상 저장됨", exc)

    # --- 5) 실험 leaderboard (재학습 이력 비교) ---
    leaderboard = compare_runs(metric="roc_auc_mean", top_k=5)
    if leaderboard:
        logger.info("최근 상위 run (roc_auc_mean 기준):")
        for entry in leaderboard:
            logger.info("  %s -> %.4f", entry["run_name"], entry.get("roc_auc_mean") or 0.0)

    # 요약 저장
    summary_path = ARTIFACT_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("CV 요약: %s", summary_path)
    return all_results


def main() -> None:
    run()


if __name__ == "__main__":
    main()
