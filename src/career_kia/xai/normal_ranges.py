"""피처별 정상 운전 범위 정의 + 이탈 서술.

목적: 현장 엔지니어가 SHAP 숫자가 아니라 "정상 범위 대비 얼마나 벗어났는가"
로 상황을 이해할 수 있도록 한다.

값들은 AI4I 2020 데이터셋의 정상(고장=0) 분위수와 도메인 지식을 결합한
보수적 범위이다. 실제 라인 도입 시 ``learn_ranges_from_data()`` 로
재계산하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NormalRange:
    feature: str
    low: float
    high: float
    label: str   # 사람이 읽기 좋은 한국어 라벨
    unit: str = ""


# ---------------------------------------------------------------------------
# 기본 정상 범위 (도메인 지식 + AI4I 정상 분위수)
# ---------------------------------------------------------------------------

DEFAULT_RANGES: dict[str, NormalRange] = {
    "Air temperature [K]": NormalRange(
        "Air temperature [K]", 295.0, 304.0, "대기 온도", "K"
    ),
    "Process temperature [K]": NormalRange(
        "Process temperature [K]", 305.0, 313.0, "공정 온도", "K"
    ),
    "Rotational speed [rpm]": NormalRange(
        "Rotational speed [rpm]", 1_200.0, 2_500.0, "회전 속도", "rpm"
    ),
    "Torque [Nm]": NormalRange(
        "Torque [Nm]", 20.0, 65.0, "토크", "Nm"
    ),
    "Tool wear [min]": NormalRange(
        "Tool wear [min]", 0.0, 150.0, "공구 마모 시간", "분"
    ),
    "t_rms_max": NormalRange(
        "t_rms_max", 0.0, 0.5, "스핀들 진동 RMS 최대", ""
    ),
    "t_kurtosis_max": NormalRange(
        "t_kurtosis_max", 2.0, 5.0, "진동 첨도 최대", ""
    ),
    "f_env_BPFI_max": NormalRange(
        "f_env_BPFI_max", 0.0, 0.05, "내륜 결함 주파수 신호", ""
    ),
    "f_env_BPFO_max": NormalRange(
        "f_env_BPFO_max", 0.0, 0.05, "외륜 결함 주파수 신호", ""
    ),
    "f_env_BSF_max": NormalRange(
        "f_env_BSF_max", 0.0, 0.05, "볼 결함 주파수 신호", ""
    ),
}


def get_range(feature: str) -> NormalRange | None:
    return DEFAULT_RANGES.get(feature)


def is_out_of_range(feature: str, value: float) -> bool:
    nr = get_range(feature)
    if nr is None:
        return False
    return value < nr.low or value > nr.high


def deviation_pct(feature: str, value: float) -> float | None:
    """범위 이탈 정도 (% 기준).

    상한 초과 시 양수, 하한 미달 시 음수, 범위 내면 0.
    범위 정의가 없으면 None.
    """
    nr = get_range(feature)
    if nr is None:
        return None
    if nr.high > 0 and value > nr.high:
        return (value - nr.high) / nr.high * 100.0
    if value < nr.low:
        ref = nr.low if nr.low != 0 else 1.0
        return (value - nr.low) / abs(ref) * 100.0
    return 0.0


def describe_deviation(feature: str, value: float) -> str:
    """피처 값을 정상 범위 대비 평문으로 서술.

    예: "공구 마모 시간 200분: 정상 범위(0~150분)를 33% 초과"
    """
    nr = get_range(feature)
    if nr is None:
        return f"{feature} = {value:.2f}"
    low, high, label, unit = nr.low, nr.high, nr.label, nr.unit

    def _fmt(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}{unit}"
        if abs(v) >= 1:
            return f"{v:.1f}{unit}"
        return f"{v:.3f}{unit}"

    if value > high:
        pct = (value - high) / high * 100.0 if high > 0 else float("inf")
        return (
            f"{label} {_fmt(value)}: 정상 범위({_fmt(low)}~{_fmt(high)})를 "
            f"{pct:.0f}% 초과"
        )
    if value < low:
        if low == 0:
            return f"{label} {_fmt(value)}: 정상 범위({_fmt(low)}~{_fmt(high)}) 미만"
        pct = (low - value) / abs(low) * 100.0
        return (
            f"{label} {_fmt(value)}: 정상 범위({_fmt(low)}~{_fmt(high)}) 대비 "
            f"{pct:.0f}% 부족"
        )
    return f"{label} {_fmt(value)}: 정상 범위({_fmt(low)}~{_fmt(high)}) 내"


def learn_ranges_from_data(
    df: pd.DataFrame,
    *,
    label_col: str = "Machine failure",
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, NormalRange]:
    """정상(label=0) 데이터의 분위수로 새 범위 학습.

    실제 라인 데이터에 맞추고 싶을 때 사용. 기본 범위는 그대로 두고
    학습 결과로 덮어쓸 수 있도록 dict 를 반환한다.
    """
    normal = df[df[label_col] == 0] if label_col in df.columns else df
    out: dict[str, NormalRange] = {}
    for feature, ref in DEFAULT_RANGES.items():
        if feature not in normal.columns:
            continue
        col = normal[feature].dropna()
        if col.empty:
            continue
        low = float(np.quantile(col, quantiles[0]))
        high = float(np.quantile(col, quantiles[1]))
        out[feature] = NormalRange(feature, low, high, ref.label, ref.unit)
    return out
