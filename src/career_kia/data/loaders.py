"""데이터 로딩 및 합성 batch_id 매핑.

CWRU 진동 신호(비정형)와 AI4I 공정 파라미터(정형)를 같은 가상 생산라인 관점의
`batch_id` / `timestamp`로 연결하여 MES 스타일 이벤트 로그를 만든다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from career_kia.config import INTERIM_DIR, RANDOM_SEED, RAW_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AI4I
# ---------------------------------------------------------------------------

def load_ai4i(csv_path: Path | None = None) -> pd.DataFrame:
    """AI4I 2020 CSV 로딩."""
    csv_path = Path(csv_path) if csv_path else RAW_DIR / "ai4i" / "ai4i2020.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"AI4I 파일이 없습니다: {csv_path}. 먼저 `make data` 를 실행하세요."
        )
    df = pd.read_csv(csv_path)
    logger.info("AI4I 로드: %d 행, %d 컬럼", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CWRURecord:
    filename: str
    fault_type: str
    fault_id: int
    fs: int
    signal: np.ndarray


def load_cwru_signals(cwru_dir: Path | None = None) -> list[CWRURecord]:
    """CWRU(합성) 진동 신호 전체를 메모리에 로드.

    합성본 규모(클래스당 40개 × 1초 × 12kHz)가 작아서 문제가 없지만,
    실데이터로 확장 시에는 generator 버전으로 교체할 것.
    """
    cwru_dir = Path(cwru_dir) if cwru_dir else RAW_DIR / "cwru"
    index_path = cwru_dir / "synthetic_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"CWRU 인덱스가 없습니다: {index_path}. 먼저 `make data` 를 실행하세요."
        )

    idx = pd.read_parquet(index_path)
    records: list[CWRURecord] = []
    for _, row in idx.iterrows():
        sig = np.load(cwru_dir / row["filename"])
        records.append(
            CWRURecord(
                filename=row["filename"],
                fault_type=row["fault_type"],
                fault_id=int(row["fault_id"]),
                fs=int(row["fs"]),
                signal=sig,
            )
        )
    logger.info("CWRU 로드: %d 개 신호", len(records))
    return records


# ---------------------------------------------------------------------------
# 합성 batch 매핑 — MES 스타일
# ---------------------------------------------------------------------------

def synthesize_mes_metadata(
    n_batches: int,
    start_time: str = "2026-01-01 00:00:00",
    batch_minutes: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """가상 MES 메타데이터 생성.

    Parameters
    ----------
    n_batches
        만들어낼 배치 수 (AI4I 행 수와 동일하게 설정하는 것이 기본).
    batch_minutes
        한 배치 간격 (분).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=start_time, periods=n_batches, freq=f"{batch_minutes}min"
    )

    # 교대조: 8시간씩 A/B/C
    hours = timestamps.hour
    shift = np.where(
        hours < 8, "A", np.where(hours < 16, "B", "C")
    )

    operators = rng.choice([f"OP{i:02d}" for i in range(1, 11)], size=n_batches)
    lines = rng.choice(["PL-01", "PL-02"], size=n_batches, p=[0.6, 0.4])

    return pd.DataFrame(
        {
            "batch_id": [f"B{i:06d}" for i in range(n_batches)],
            "timestamp": timestamps,
            "shift": shift,
            "operator_id": operators,
            "line_id": lines,
        }
    )


def build_joined_dataset(
    ai4i_df: pd.DataFrame,
    cwru_records: list[CWRURecord],
    save_path: Path | None = None,
    seed: int = RANDOM_SEED,
    *,
    p_false_negative_vibration: float = 0.25,
    p_early_warning_vibration: float = 0.05,
) -> pd.DataFrame:
    """AI4I 행별로 CWRU 신호 1개를 매핑하여 통합 배치 테이블 생성.

    **현실적인 라벨 노이즈** 를 반영한 확률적 매핑::

        - 고장 배치(Machine failure == 1) 중 `p_false_negative_vibration` 비율은
          진동으로는 잡히지 않는 전기/열/공정 원인으로 가정하여 Normal 진동 배정
        - 정상 배치(Machine failure == 0) 중 `p_early_warning_vibration` 비율은
          조기 경고(아직 고장은 아니나 결함 진행 중)로 결함 진동 배정

    이 노이즈가 없으면 `vibration_fault_type` 이 `Machine failure` 를 완벽히
    결정하여 학습이 trivial 하게 풀리고 XAI/인과추론 의미가 없어진다.
    """
    rng = np.random.default_rng(seed)
    n = len(ai4i_df)
    mes = synthesize_mes_metadata(n_batches=n, seed=seed)

    by_cls: dict[str, list[CWRURecord]] = {}
    for r in cwru_records:
        by_cls.setdefault(r.fault_type, []).append(r)

    def _pick(fault_type: str) -> str:
        pool = by_cls[fault_type]
        return rng.choice([p.filename for p in pool])

    fault_pool = ["IR", "OR", "Ball"]
    mapped_fault: list[str] = []
    mapped_file: list[str] = []
    for failure in ai4i_df["Machine failure"].to_numpy():
        if failure == 1:
            # 고장 배치: 일부는 진동으로 안 잡히는 원인 (false negative)
            if rng.random() < p_false_negative_vibration:
                ft = "Normal"
            else:
                ft = rng.choice(fault_pool)
        else:
            # 정상 배치: 일부는 조기 경고 진동 (early warning)
            if rng.random() < p_early_warning_vibration:
                ft = rng.choice(fault_pool)
            else:
                ft = "Normal"
        mapped_fault.append(ft)
        mapped_file.append(_pick(ft))

    joined = ai4i_df.copy().reset_index(drop=True)
    joined = pd.concat(
        [mes.reset_index(drop=True), joined],
        axis=1,
    )
    joined["vibration_fault_type"] = mapped_fault
    joined["vibration_file"] = mapped_file

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(save_path, index=False)
        logger.info("조인 테이블 저장: %s (%d 행)", save_path, len(joined))

    return joined


def build_default_joined() -> pd.DataFrame:
    """기본 경로로 AI4I + CWRU 조인 테이블을 생성/저장."""
    ai4i = load_ai4i()
    cwru = load_cwru_signals()
    save_path = INTERIM_DIR / "batch_master.parquet"
    return build_joined_dataset(ai4i, cwru, save_path=save_path)
