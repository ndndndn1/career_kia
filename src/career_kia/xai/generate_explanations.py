"""Phase 6 XAI 산출 엔트리 — `make xai` 에서 호출.

파이프라인::

    hybrid_model.joblib + features.parquet
        │
        ├─ 전역 SHAP 요약 → data/processed/shap_global.parquet
        ├─ 대표 샘플(고위험 10개) 국소 설명 → JSON
        ├─ 상호작용 상위 쌍 → CSV
        └─ 자연어 설명문 샘플 → 텍스트
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from career_kia.config import PROCESSED_DIR, PROJECT_ROOT
from career_kia.models.train import load_feature_matrix
from career_kia.xai import explanation_templates, nl_generator, shap_utils

logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "xai_artifacts"


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models_artifacts" / "hybrid_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 없음. `make train` 먼저 실행.")
    model = joblib.load(model_path)
    logger.info("모델 로드: %s", model_path)

    X, y, _ = load_feature_matrix()
    batch_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # 1) 전역 SHAP
    bundle = shap_utils.explain_batch(model, X, max_samples=2000)
    global_importance = bundle.mean_abs
    global_importance.to_frame("mean_abs_shap").to_parquet(
        ARTIFACT_DIR / "shap_global.parquet"
    )
    logger.info("전역 SHAP 상위 10:\n%s", global_importance.head(10).to_string())

    # 2) 고위험 대표 샘플 선정 — 모델 예측 확률 상위 10
    proba = model.predict_proba(X)[:, 1]
    top_risk_idx = np.argsort(proba)[::-1][:10]

    # 3) 상호작용
    inter_values = shap_utils.interaction_values(model, X, max_samples=500)
    inter = explanation_templates.build_interactions(
        inter_values, bundle.feature_names, k=10
    )
    inter.pairs.to_csv(ARTIFACT_DIR / "shap_interactions_top10.csv", index=False)
    logger.info("상호작용 상위 5:\n%s", inter.pairs.head(5).to_string())

    # 4) 개별 설명 + 자연어
    all_narratives: list[dict] = []
    for rank, idx in enumerate(top_risk_idx):
        local_bundle = shap_utils.explain_batch(model, X.iloc[[idx]])
        contribution = explanation_templates.build_contribution(
            local_bundle,
            sample_idx=0,
            prediction=float(proba[idx]),
            sample_id=str(batch_df.iloc[idx]["batch_id"]),
        )
        thresholds = explanation_templates.infer_thresholds(
            bundle,
            features=list(global_importance.head(8).index),
            bins=10,
            min_shap=0.05,
        )
        batch_exp = explanation_templates.BatchExplanation(
            contribution=contribution,
            thresholds=thresholds,
            interactions=inter,
        )
        paragraph = nl_generator.batch_explanation_to_paragraph(batch_exp)
        all_narratives.append(
            {
                "rank": int(rank + 1),
                "batch_id": contribution.sample_id,
                "prediction": contribution.prediction,
                "narrative": paragraph,
            }
        )

    with open(ARTIFACT_DIR / "top_risk_narratives.json", "w", encoding="utf-8") as f:
        json.dump(all_narratives, f, ensure_ascii=False, indent=2)
    logger.info("고위험 배치 설명문 %d 개 저장", len(all_narratives))

    # 5) 읽기 쉬운 텍스트 버전
    txt_lines = []
    for n in all_narratives:
        txt_lines.append(f"=== Rank {n['rank']}: {n['batch_id']} (예측 {n['prediction']*100:.1f}%) ===")
        txt_lines.append(n["narrative"])
        txt_lines.append("")
    (ARTIFACT_DIR / "top_risk_narratives.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    logger.info("XAI 산출 완료: %s", ARTIFACT_DIR)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
