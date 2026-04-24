"""MLflow 실험 추적 설정.

추적 URI·실험명 기본값을 프로젝트 루트에 고정한다. `models/train.py` 에서
호출되어 모든 학습 run 이 같은 경로·같은 실험에 기록되도록 한다.
"""

from __future__ import annotations

import mlflow

from career_kia.config import PROJECT_ROOT

DEFAULT_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
DEFAULT_EXPERIMENT = "sdf-xplain"


def configure(
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment or DEFAULT_EXPERIMENT)
