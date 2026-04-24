"""공개 데이터셋(CWRU, AI4I 2020) 다운로드.

외부 네트워크 접근이 불가능한 환경(방화벽/오프라인)에서도 파이프라인 검증이
가능하도록, 실제 다운로드 실패 시 **합성 폴백 데이터**를 생성하는 구조로
설계했다. 실제 현장 배포 시에는 `synthesize_if_missing=False`로 호출하여
폴백 경로를 차단하면 된다.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from career_kia.config import RANDOM_SEED, RAW_DIR

logger = logging.getLogger(__name__)

AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)


# ---------------------------------------------------------------------------
# AI4I 2020
# ---------------------------------------------------------------------------

def download_ai4i(
    target_dir: Path | None = None,
    *,
    synthesize_if_missing: bool = True,
    timeout: int = 30,
) -> Path:
    """AI4I 2020 Predictive Maintenance CSV 다운로드.

    Returns
    -------
    Path
        저장된 CSV 파일 경로.
    """
    target_dir = Path(target_dir) if target_dir else RAW_DIR / "ai4i"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "ai4i2020.csv"

    if csv_path.exists():
        logger.info("AI4I 이미 존재: %s", csv_path)
        return csv_path

    try:
        logger.info("AI4I 다운로드 시도: %s", AI4I_URL)
        resp = requests.get(AI4I_URL, timeout=timeout)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # zip 안의 csv 파일을 찾는다
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as src, open(csv_path, "wb") as dst:
                dst.write(src.read())
        logger.info("AI4I 저장 완료: %s", csv_path)
        return csv_path
    except Exception as exc:  # noqa: BLE001
        if not synthesize_if_missing:
            raise
        logger.warning("AI4I 다운로드 실패 (%s) — 합성 폴백 생성", exc)
        df = _synthesize_ai4i(n_rows=10_000, seed=RANDOM_SEED)
        df.to_csv(csv_path, index=False)
        return csv_path


def _synthesize_ai4i(n_rows: int = 10_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """AI4I 유사 합성 데이터 — 스키마와 분포를 실제에 근접시킨다.

    실제 AI4I 논문/데이터 분포를 참고했지만, 본 프로젝트의 자체 생성본이라는
    점을 명시하기 위해 `UDI`를 음수 대역에서 시작시킨다.
    """
    rng = np.random.default_rng(seed)

    type_probs = {"L": 0.5, "M": 0.3, "H": 0.2}
    types = rng.choice(
        list(type_probs.keys()),
        size=n_rows,
        p=list(type_probs.values()),
    )

    air_temp = rng.normal(300.0, 2.0, size=n_rows)
    process_temp = air_temp + rng.normal(10.0, 1.0, size=n_rows)
    rot_speed = rng.normal(1538.0, 179.0, size=n_rows).clip(1168, 2886)
    torque = rng.normal(40.0, 9.9, size=n_rows).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 253, size=n_rows)

    # 고장 메커니즘
    twf = (tool_wear >= 200) & (tool_wear <= 240) & (rng.random(n_rows) < 0.4)
    hdf = ((process_temp - air_temp) < 8.6) & (rot_speed < 1380) & (rng.random(n_rows) < 0.6)
    power = (torque * 2 * np.pi * rot_speed / 60).clip(0)  # W
    pwf = ((power < 3500) | (power > 9000)) & (rng.random(n_rows) < 0.5)
    osf_thresh = {"L": 11000, "M": 12000, "H": 13000}
    thr = np.array([osf_thresh[t] for t in types])
    osf = (tool_wear * torque > thr) & (rng.random(n_rows) < 0.5)
    rnf = rng.random(n_rows) < 0.001

    machine_failure = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame(
        {
            "UDI": -np.arange(1, n_rows + 1),  # 합성 표식
            "Product ID": [f"{t}{i:05d}" for t, i in zip(types, range(n_rows), strict=True)],
            "Type": types,
            "Air temperature [K]": air_temp.round(1),
            "Process temperature [K]": process_temp.round(1),
            "Rotational speed [rpm]": rot_speed.round().astype(int),
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": tool_wear,
            "Machine failure": machine_failure,
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )


# ---------------------------------------------------------------------------
# CWRU 베어링
# ---------------------------------------------------------------------------

def download_cwru(
    target_dir: Path | None = None,
    *,
    synthesize_if_missing: bool = True,
    n_samples_per_class: int = 40,
) -> Path:
    """CWRU 베어링 데이터 확보.

    실제 CWRU 저장소는 로그인/정규식 기반으로 파일을 제공하여 자동화가
    불안정하므로, 본 프로젝트에서는 **합성 진동 신호**로 폴백을 기본으로 한다.
    실제 데이터를 사용하려면 수동으로 `data/raw/cwru/` 아래에 `.mat` 파일을
    배치하고 `loaders.load_cwru_signals(synthetic=False)`를 호출하면 된다.
    """
    target_dir = Path(target_dir) if target_dir else RAW_DIR / "cwru"
    target_dir.mkdir(parents=True, exist_ok=True)
    index_path = target_dir / "synthetic_index.parquet"

    if index_path.exists():
        logger.info("CWRU 합성본 이미 존재: %s", target_dir)
        return target_dir

    if not synthesize_if_missing:
        raise FileNotFoundError(
            "실제 CWRU 데이터가 없고 합성 폴백이 비활성화되어 있습니다."
        )

    logger.info("CWRU 합성 진동 신호 생성 (클래스당 %d개)", n_samples_per_class)
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED)

    classes = ["Normal", "IR", "OR", "Ball"]
    for cls_idx, cls in enumerate(classes):
        for sample_idx in range(n_samples_per_class):
            sig = _synthesize_cwru_signal(fault_type=cls, rng=rng)
            fname = f"{cls}_{sample_idx:03d}.npy"
            np.save(target_dir / fname, sig)
            records.append(
                {
                    "filename": fname,
                    "fault_type": cls,
                    "fault_id": cls_idx,
                    "fs": 12_000,
                    "n_samples": len(sig),
                }
            )

    idx = pd.DataFrame(records)
    idx.to_parquet(index_path, index=False)
    logger.info("CWRU 합성본 인덱스: %s (%d 행)", index_path, len(idx))
    return target_dir


def _synthesize_cwru_signal(
    fault_type: str,
    rng: np.random.Generator,
    fs: int = 12_000,
    duration_sec: float = 1.0,
) -> np.ndarray:
    """베어링 결함 유형별 진동 신호 모사.

    정상은 광대역 노이즈, IR/OR/Ball은 각각 다른 특성 주파수(BPFI/BPFO/BSF)에서
    주기적 임펄스를 갖도록 합성한다. 본 모델은 "진단 가능한 패턴"을 담는 데
    목적이 있으며, 실제 베어링 동역학 모델은 아니다.
    """
    n = int(fs * duration_sec)
    t = np.arange(n) / fs

    # 기본 회전 성분 (회전수 ~ 1750 rpm = 29.17 Hz)
    shaft_freq = 29.17
    base = 0.3 * np.sin(2 * np.pi * shaft_freq * t)
    noise = 0.4 * rng.normal(size=n)
    sig = base + noise

    # 결함 특성 주파수 (Hz)
    fault_freqs = {
        "Normal": None,
        "IR": 157.94,
        "OR": 104.56,
        "Ball": 61.32,
    }

    ff = fault_freqs[fault_type]
    if ff is not None:
        # 결함은 임펄스성 (공진 주파수 ~3kHz를 감쇠 진동으로)
        period_samples = int(fs / ff)
        resonance = 3000
        decay = np.exp(-t[: period_samples // 2] * 400)
        impulse = 2.5 * decay * np.sin(2 * np.pi * resonance * t[: period_samples // 2])
        # 주기적으로 반복
        impulse_train = np.zeros(n)
        for start in range(0, n - len(impulse), period_samples):
            jitter = rng.integers(-20, 21)
            s = max(0, start + jitter)
            e = min(n, s + len(impulse))
            impulse_train[s:e] += impulse[: e - s]
        sig = sig + impulse_train * (0.6 + 0.2 * rng.random())

    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# 통합 엔트리
# ---------------------------------------------------------------------------

def main() -> None:
    """Makefile에서 `make data` 로 호출되는 엔트리포인트.

    1) AI4I / CWRU 원본 확보
    2) 가상 MES batch_id 로 진동 ↔ 공정 조인 → `data/interim/batch_master.parquet`
    """
    from career_kia.data.loaders import build_default_joined

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ai4i = download_ai4i()
    cwru = download_cwru()
    logger.info("AI4I: %s", ai4i)
    logger.info("CWRU: %s", cwru)

    joined = build_default_joined()
    logger.info("조인 테이블: %d 행 × %d 컬럼", *joined.shape)


if __name__ == "__main__":
    main()
