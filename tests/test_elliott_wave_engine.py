import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.elliott_wave_engine import Pivot, detect_pivots


def _make_series(prices: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


class TestDetectPivots:
    def test_returns_list_of_pivots(self):
        prices = [100, 110, 105, 115, 108, 120, 112]
        result = detect_pivots(_make_series(prices))
        assert isinstance(result, list)
        assert all(isinstance(p, Pivot) for p in result)

    def test_alternates_high_low(self):
        # Up-down-up-down pattern should give alternating high/low
        prices = [100, 115, 105, 120, 108, 130, 115]
        result = detect_pivots(_make_series(prices))
        for i in range(len(result) - 1):
            assert result[i].type != result[i + 1].type

    def test_insufficient_data_returns_empty(self):
        prices = [100, 101, 99]
        result = detect_pivots(_make_series(prices))
        assert result == []

    def test_returns_empty_for_flat_data(self):
        prices = [100.0] * 50
        result = detect_pivots(_make_series(prices))
        assert result == []

    def test_pivot_has_required_fields(self):
        prices = [100, 110, 90, 115, 85, 120, 95, 125]
        result = detect_pivots(_make_series(prices))
        if result:
            p = result[0]
            assert hasattr(p, "date")
            assert hasattr(p, "price")
            assert p.type in ("high", "low")
