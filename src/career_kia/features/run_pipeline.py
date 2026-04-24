"""Phase 4 피처 엔지니어링 엔트리 — `make features` 에서 호출.

배치 × 윈도우 × 피처 → 배치 × 피처(윈도우 통계 집계)로 축약한다.

산출물::

    data/processed/features.parquet
        - 공정 파라미터(Air/Process temp, RPM, Torque, Tool wear, Type)
        - 진동 시간 도메인 피처(배치 내 윈도우 평균·최대·표준편차)
        - 진동 주파수 도메인 피처(배치 내 윈도우 평균·최대)
        - 베어링 특성 주파수(BPFI/BPFO/BSF/FTF) 포락선 진폭
        - 메타(batch_id, timestamp, shift, operator_id, line_id, Type)
        - 라벨(Machine failure, vibration_fault_type)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from career_kia.config import (
    CWRU_FS_12K,
    INTERIM_DIR,
    PROCESSED_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from career_kia.features import freq_domain, time_domain
from career_kia.features.windowing import make_windows

logger = logging.getLogger(__name__)

META_COLS = ["batch_id", "timestamp", "shift", "operator_id", "line_id", "Type"]
PROC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
LABEL_COLS = ["Machine failure", "vibration_fault_type", "TWF", "HDF", "PWF", "OSF", "RNF"]


def aggregate_window_features(
    windows: np.ndarray,
    fs: int,
) -> dict[str, float]:
    """단일 배치의 (N_window, W) → 집계 스칼라 dict.

    각 피처에 대해 mean / max / std 3가지 집계를 산출하여 배치 특성 반영.
    """
    out: dict[str, float] = {}
    time_f = time_domain.compute_time_features(windows)
    for name, arr in time_f.items():
        out[f"t_{name}_mean"] = float(arr.mean())
        out[f"t_{name}_max"] = float(arr.max())
        out[f"t_{name}_std"] = float(arr.std())

    freq_f = freq_domain.compute_freq_features(windows, fs=fs)
    for name, arr in freq_f.items():
        out[f"f_{name}_mean"] = float(arr.mean())
        out[f"f_{name}_max"] = float(arr.max())
    return out


def build_features(
    batch_df: pd.DataFrame,
    cwru_clean_dir: Path,
    fs: int = CWRU_FS_12K,
) -> pd.DataFrame:
    """배치 테이블 전체에 대해 피처 생성."""
    rows: list[dict[str, object]] = []
    signal_cache: dict[str, np.ndarray] = {}

    for i, row in enumerate(batch_df.itertuples(index=False)):
        fname = row.vibration_file
        if fname not in signal_cache:
            signal_cache[fname] = np.load(cwru_clean_dir / fname).astype(np.float32)
        sig = signal_cache[fname]
        windows = make_windows(sig, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
        if len(windows) == 0:
            continue
        feats = aggregate_window_features(windows, fs=fs)
        record: dict[str, object] = {c: getattr(row, c.replace(" ", "_").replace("[", "").replace("]", "")) for c in []}
        # 메타 직접 복사
        for c in META_COLS + PROC_COLS + LABEL_COLS:
            # itertuples 는 공백/대괄호를 '_' 로 바꾼 name 을 가짐 — 인덱스 접근이 안전
            record[c] = batch_df.iloc[i][c]
        record.update(feats)
        rows.append(record)

        if (i + 1) % 2000 == 0:
            logger.info("피처 생성: %d / %d", i + 1, len(batch_df))

    feat_df = pd.DataFrame(rows)
    logger.info("최종 피처 테이블: %s", feat_df.shape)
    return feat_df


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path) if input_path else INTERIM_DIR / "preprocessed_batch.parquet"
    output_path = Path(output_path) if output_path else PROCESSED_DIR / "features.parquet"
    cwru_clean_dir = INTERIM_DIR / "cwru_clean"

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 없음. `make preprocess` 먼저 실행.")
    if not cwru_clean_dir.exists():
        raise FileNotFoundError(f"{cwru_clean_dir} 없음. `make preprocess` 먼저 실행.")

    batch = pd.read_parquet(input_path)
    logger.info("전처리 배치 로드: %s", batch.shape)

    feat_df = build_features(batch, cwru_clean_dir=cwru_clean_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(output_path, index=False)
    logger.info("피처 저장: %s", output_path)
    return feat_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()


if __name__ == "__main__":
    main()
