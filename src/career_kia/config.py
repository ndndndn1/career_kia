"""프로젝트 전역 경로/설정."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

ARTIFACT_DIR: Path = PROJECT_ROOT / "mlruns"

# CWRU 베어링 샘플링 주파수 (12k / 48k 두 종류 존재)
CWRU_FS_12K: int = 12_000
CWRU_FS_48K: int = 48_000

# 시나리오 기준 "공통 리샘플 주파수" — 멀티 센서 동기화 타겟
TARGET_FS: int = 12_000

# 기본 윈도우 설정
WINDOW_SIZE: int = 2_048   # 약 170ms @ 12kHz
WINDOW_STRIDE: int = 1_024 # 50% 오버랩

RANDOM_SEED: int = 42
