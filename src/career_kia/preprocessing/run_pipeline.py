"""Phase 3 전처리 파이프라인 엔트리 — `make preprocess` 에서 호출.

흐름::

    batch_master.parquet (Phase 2 산출)
        │
        ├─ 공정 파라미터: 결측 보간 + 이상치 마스킹→재보간
        ├─ 진동 신호: 배치마다 wavelet denoise + bandpass 후 파일 갱신
        └─ 결과를 `interim/preprocessed_batch.parquet` 로 저장
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from career_kia.config import CWRU_FS_12K, INTERIM_DIR, RAW_DIR
from career_kia.preprocessing import filtering, imputation, outliers

logger = logging.getLogger(__name__)

PROC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def preprocess_process_params(df: pd.DataFrame) -> pd.DataFrame:
    """공정 파라미터: IQR 이상치 → NaN → 선형 보간."""
    out = df.copy()
    for c in PROC_COLS:
        mask = outliers.iqr_mask(out[c], k=3.0)  # 너무 공격적이지 않은 기준
        out[c] = outliers.apply_mask(out[c], mask, fill="nan")
    out[PROC_COLS] = imputation.impute_dataframe(out[PROC_COLS], method="linear")
    # 보간 후에도 남은 결측(경계 등)은 KNN으로 채움
    if out[PROC_COLS].isna().any().any():
        out[PROC_COLS] = imputation.impute_dataframe(out[PROC_COLS], method="knn")
    return out


def preprocess_vibration_file(
    src_path: Path,
    *,
    fs: int = CWRU_FS_12K,
    bandpass: tuple[float, float] = (500.0, 5000.0),
    wavelet_level: int = 4,
) -> np.ndarray:
    """단일 진동 파일을 로드해 bandpass + wavelet denoise 적용 후 반환."""
    sig = np.load(src_path).astype(float)
    sig = filtering.butterworth(
        sig, fs=fs, cutoff=bandpass, filter_type="bandpass", order=4
    )
    sig = filtering.wavelet_denoise(sig, wavelet="db4", level=wavelet_level)
    return sig.astype(np.float32)


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    write_cleaned_signals: bool = True,
) -> pd.DataFrame:
    """전체 파이프라인 실행."""
    input_path = Path(input_path) if input_path else INTERIM_DIR / "batch_master.parquet"
    output_path = Path(output_path) if output_path else INTERIM_DIR / "preprocessed_batch.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} 가 없습니다. `make data`를 먼저 실행하세요."
        )

    logger.info("배치 마스터 로드: %s", input_path)
    df = pd.read_parquet(input_path)
    n = len(df)

    # 1) 공정 파라미터 전처리
    logger.info("공정 파라미터 전처리 (n=%d)", n)
    df = preprocess_process_params(df)

    # 2) 진동 신호 전처리
    if write_cleaned_signals:
        cwru_dir = RAW_DIR / "cwru"
        clean_dir = INTERIM_DIR / "cwru_clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        processed_files = set()
        for fname in df["vibration_file"].unique():
            if fname in processed_files:
                continue
            src = cwru_dir / fname
            dst = clean_dir / fname
            if dst.exists():
                processed_files.add(fname)
                continue
            cleaned = preprocess_vibration_file(src)
            np.save(dst, cleaned)
            processed_files.add(fname)
        logger.info("진동 신호 전처리 완료: %d개 파일 → %s", len(processed_files), clean_dir)
        df["vibration_clean_file"] = df["vibration_file"]

    # 3) 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("전처리 결과 저장: %s (%d 행)", output_path, len(df))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()


if __name__ == "__main__":
    main()
