"""MLflow 실험 추적/모델 레지스트리 래퍼.

- 경로·실험명 표준화
- 모델 버저닝 편의 함수
- 재학습 시 이전 버전과의 지표 비교
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from career_kia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
DEFAULT_EXPERIMENT = "sdf-xplain"


def configure(
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment or DEFAULT_EXPERIMENT)


def register_model(
    local_path: Path,
    model_name: str,
    *,
    tags: dict | None = None,
) -> str:
    """로컬 joblib 을 모델 레지스트리에 등록.

    Returns
    -------
    등록된 모델 URI (`runs:/<run_id>/model` 또는 `models:/<name>/<version>`)
    """
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(local_path), artifact_path="model")
        if tags:
            mlflow.set_tags(tags)
        uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(uri, model_name)
    logger.info(
        "모델 등록: %s version %s",
        result.name,
        result.version,
    )
    return f"models:/{result.name}/{result.version}"


def list_model_versions(model_name: str) -> list[dict]:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "status": v.status,
            "run_id": v.run_id,
        }
        for v in versions
    ]


def compare_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    *,
    metric: str = "roc_auc_mean",
    top_k: int = 10,
):
    """최근 실험들의 특정 지표 비교."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_k,
    )
    return [
        {
            "run_name": r.data.tags.get("mlflow.runName", "?"),
            metric: r.data.metrics.get(metric),
            "start_time": r.info.start_time,
        }
        for r in runs
    ]
