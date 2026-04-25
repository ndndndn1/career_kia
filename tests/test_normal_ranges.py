"""정상 범위 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_kia.xai import normal_ranges as nr


def test_get_range_known_and_unknown():
    assert nr.get_range("Tool wear [min]") is not None
    assert nr.get_range("Tool wear [min]").high == 150.0
    assert nr.get_range("nonexistent_feature") is None


def test_is_out_of_range():
    assert nr.is_out_of_range("Tool wear [min]", 200.0) is True
    assert nr.is_out_of_range("Tool wear [min]", 100.0) is False
    # 범위 정의 없는 피처는 항상 False
    assert nr.is_out_of_range("nonexistent", 999.0) is False


def test_deviation_pct():
    # 200 분 vs 상한 150 → 약 33% 초과
    pct = nr.deviation_pct("Tool wear [min]", 200.0)
    assert pct is not None and 30 < pct < 40

    # 범위 내
    assert nr.deviation_pct("Tool wear [min]", 100.0) == 0.0

    # 정의 없음
    assert nr.deviation_pct("unknown", 1.0) is None


def test_describe_deviation_above_range():
    text = nr.describe_deviation("Tool wear [min]", 200.0)
    assert "공구 마모 시간" in text
    assert "초과" in text
    assert "150" in text
    # SHAP 같은 통계 용어가 없어야 함
    assert "SHAP" not in text
    assert "기여도" not in text


def test_describe_deviation_within_range():
    text = nr.describe_deviation("Tool wear [min]", 100.0)
    assert "내" in text


def test_describe_deviation_below_range():
    text = nr.describe_deviation("Torque [Nm]", 5.0)
    assert "Torque" in text or "토크" in text
    assert "부족" in text or "미만" in text


def test_describe_deviation_unknown_feature():
    text = nr.describe_deviation("xyz", 1.234)
    assert "xyz" in text


def test_learn_ranges_from_data_uses_normal_quantiles():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Tool wear [min]": np.concatenate(
                [rng.uniform(0, 100, 1000), rng.uniform(180, 250, 100)]
            ),
            "Machine failure": np.concatenate(
                [np.zeros(1000, dtype=int), np.ones(100, dtype=int)]
            ),
        }
    )
    learned = nr.learn_ranges_from_data(df)
    assert "Tool wear [min]" in learned
    # 정상(label=0) 의 95% 분위수는 100 이하여야 함
    assert learned["Tool wear [min]"].high < 110
