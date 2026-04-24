"""Phase 7 인과추론 엔트리 — `make causal` 에서 호출.

주요 공정 개입(Tool wear, Torque, Rotational speed) 의 고장 확률에 대한
평균 처치 효과(ATE) 를 추정하고 refutation 검증까지 수행한다.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from career_kia.config import PROCESSED_DIR, PROJECT_ROOT
from career_kia.causal.intervention import (
    estimate_ate,
    refute_estimate,
    whatif_dose_response,
)

logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "causal_artifacts"


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    logger.info("피처 로드: %s", df.shape)

    # 1) Tool wear: 200 vs 50 (공구 교체 주기 개입)
    results = {}

    for treatment, treat_val, ctrl_val, grid in [
        ("Tool_wear", 200.0, 50.0, np.linspace(20, 240, 12)),
        ("Torque", 60.0, 40.0, np.linspace(20, 75, 12)),
        ("Rotational_speed", 1300.0, 1600.0, np.linspace(1200, 2800, 9)),
    ]:
        logger.info("===== %s 개입 분석 =====", treatment)
        res, model, estimate = estimate_ate(
            df,
            treatment=treatment,
            treatment_value=treat_val,
            control_value=ctrl_val,
            method="backdoor.linear_regression",
        )
        refute = refute_estimate(model, estimate)
        res.refutation = refute
        logger.info("refutation: %s", refute)

        # dose-response 곡선 (linear regression 이라 내삽이 선형이지만 데모용)
        dr = whatif_dose_response(df, treatment=treatment, grid=grid, baseline=float(ctrl_val))
        dr.to_csv(ARTIFACT_DIR / f"doseresp_{treatment}.csv", index=False)

        results[treatment] = {
            "treatment_value": res.treatment_value,
            "control_value": res.control_value,
            "ate": res.estimate,
            "refutation": refute,
        }

    (ARTIFACT_DIR / "ate_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("ATE 결과 저장: %s", ARTIFACT_DIR)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
