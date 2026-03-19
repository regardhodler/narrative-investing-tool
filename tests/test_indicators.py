import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.indicators import rsi, macd, obv


def _make_close(n: int = 100) -> pd.Series:
    np.random.seed(42)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.normal(0.05, 1.0))
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="B"))


class TestRSI:
    def test_values_in_range(self):
        close = _make_close(200)
        result = rsi(close)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_output_length_matches_input(self):
        close = _make_close(100)
        result = rsi(close)
        assert len(result) == len(close)

    def test_nan_only_in_warmup(self):
        close = _make_close(100)
        result = rsi(close, period=14)
        # First 14 values should be NaN (period for diff + rolling)
        assert result.iloc[14:].notna().all()

    def test_custom_period(self):
        close = _make_close(100)
        result = rsi(close, period=7)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestMACD:
    def test_returns_three_series(self):
        close = _make_close(100)
        result = macd(close)
        assert len(result) == 3
        assert all(isinstance(s, pd.Series) for s in result)

    def test_histogram_equals_macd_minus_signal(self):
        close = _make_close(200)
        macd_line, signal_line, histogram = macd(close)
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected, check_names=False)

    def test_output_length_matches_input(self):
        close = _make_close(100)
        macd_line, signal_line, histogram = macd(close)
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)


def _make_volume(n: int = 100) -> pd.Series:
    np.random.seed(99)
    return pd.Series(
        np.random.randint(1_000_000, 10_000_000, n),
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
        dtype=float,
    )


class TestOBV:
    def test_output_length_matches_input(self):
        close = _make_close(100)
        vol = _make_volume(100)
        result = obv(close, vol)
        assert len(result) == len(close)

    def test_all_up_days_monotonically_increasing(self):
        n = 50
        close = pd.Series(range(100, 100 + n), index=pd.date_range("2024-01-01", periods=n, freq="B"), dtype=float)
        vol = pd.Series([1_000_000.0] * n, index=close.index)
        result = obv(close, vol)
        # After first bar, should be monotonically increasing
        diffs = result.diff().iloc[1:]
        assert (diffs >= 0).all()

    def test_flat_price_obv_stays_zero(self):
        n = 50
        close = pd.Series([100.0] * n, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        vol = pd.Series([1_000_000.0] * n, index=close.index)
        result = obv(close, vol)
        assert (result == 0).all()
