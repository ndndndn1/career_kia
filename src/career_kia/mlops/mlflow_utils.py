"""MLflow 실험 추적 · 모델 레지스트리 래퍼.

- `configure` : 추적 URI / 실험명 표준화
- `register_model` : 로컬 joblib 을 레지스트리에 등록하여 버전 부여
- `list_model_versions` : 등록된 모델의 버전 이력 조회
- `compare_runs` : 최근 run 들의 특정 지표 leaderboard
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
    existing_run_id: str | None = None,
) -> str:
    """로컬 joblib 을 모델 레지스트리에 등록.

    MLflow 3.x 에서는 `mlflow.register_model(uri)` 이 log_artifact 기반 URI 를
    인식하지 못하므로, `MlflowClient.create_model_version` 으로 직접 버전을
    생성한다.

    Parameters
    ----------
    local_path
        등록할 모델 파일 (.joblib).
    model_name
        레지스트리에 등록할 이름. 없으면 자동 생성.
    tags
        모델 버전에 부여할 태그.
    existing_run_id
        이미 진행 중인 run 에 등록할 경우 그 run id. None 이면 새 run 생성.

    Returns
    -------
    `models:/<name>/<version>` 형식의 모델 URI.
    """
    client = MlflowClient()

    if existing_run_id is not None:
        mlflow.log_artifact(str(local_path), artifact_path="model", run_id=existing_run_id)
        run_id = existing_run_id
    else:
        with mlflow.start_run() as run:
            mlflow.log_artifact(str(local_path), artifact_path="model")
            run_id = run.info.run_id

    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/model"

    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # 이미 존재 (RESOURCE_ALREADY_EXISTS)

    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        tags={k: str(v) for k, v in (tags or {}).items()},
    )
    logger.info("모델 등록: %s version %s", mv.name, mv.version)
    return f"models:/{mv.name}/{mv.version}"


def list_model_versions(model_name: str) -> list[dict]:
    """등록된 모델의 버전 이력."""
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
) -> list[dict]:
    """최근 실험들의 특정 지표 leaderboard."""
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
