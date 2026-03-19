import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.elliott_wave_engine import (
    Pivot, WaveSequence, BestCount,
    detect_pivots, find_wave_sequences, score_sequence, get_best_wave_count,
)


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


def _make_pivots(prices: list[float], types: list[str]) -> list[Pivot]:
    """Build a pivot list from prices and types."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return [Pivot(date=d, price=p, type=t) for d, p, t in zip(dates, prices, types)]


class TestFindWaveSequences:
    def test_returns_list_of_wave_sequences(self):
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        assert isinstance(result, list)
        assert all(isinstance(s, WaveSequence) for s in result)

    def test_impulse_has_6_pivots(self):
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        impulses = [s for s in result if s.wave_type == "impulse"]
        assert all(len(s.pivots) == 6 for s in impulses)

    def test_corrective_has_4_pivots(self):
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        correctives = [s for s in result if s.wave_type == "corrective"]
        assert all(len(s.pivots) == 4 for s in correctives)

    def test_insufficient_pivots_returns_empty(self):
        pivots = _make_pivots([100, 110, 95], ["low", "high", "low"])
        assert find_wave_sequences(pivots) == []

    def test_search_bound_last_30_pivots(self):
        prices = [100 + i * 3 if i % 2 == 0 else 100 + i * 3 - 5 for i in range(40)]
        types = ["low" if i % 2 == 0 else "high" for i in range(40)]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        cutoff_date = pivots[-30].date
        for seq in result:
            assert seq.pivots[0].date >= cutoff_date


class TestScoreSequence:
    def _bullish_impulse(self) -> WaveSequence:
        """Well-formed bullish impulse with near-perfect Fibonacci ratios."""
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100.00, "low"),
            Pivot(pd.Timestamp("2024-02-01"), 110.00, "high"),  # W1=10
            Pivot(pd.Timestamp("2024-03-01"), 103.82, "low"),   # retraces 0.618 of W1
            Pivot(pd.Timestamp("2024-04-01"), 120.00, "high"),  # W3=16.18 ~ 1.618*W1
            Pivot(pd.Timestamp("2024-05-01"), 113.82, "low"),   # retraces 0.382 of W3
            Pivot(pd.Timestamp("2024-06-01"), 123.82, "high"),  # W5~10
        ]
        return WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])

    def test_valid_impulse_is_valid(self):
        seq = self._bullish_impulse()
        is_valid, confidence, hits = score_sequence(seq)
        assert is_valid is True

    def test_valid_impulse_has_positive_confidence(self):
        seq = self._bullish_impulse()
        is_valid, confidence, hits = score_sequence(seq)
        assert confidence > 0

    def test_wave3_shortest_rule_rejects(self):
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),  # W1=20
            Pivot(pd.Timestamp("2024-03-01"), 112, "low"),
            Pivot(pd.Timestamp("2024-04-01"), 117, "high"),  # W3=5 (shortest — INVALID)
            Pivot(pd.Timestamp("2024-05-01"), 110, "low"),
            Pivot(pd.Timestamp("2024-06-01"), 135, "high"),  # W5=25
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_wave4_overlap_rule_rejects(self):
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),  # W1 high=120
            Pivot(pd.Timestamp("2024-03-01"), 108, "low"),
            Pivot(pd.Timestamp("2024-04-01"), 140, "high"),
            Pivot(pd.Timestamp("2024-05-01"), 115, "low"),   # W4 low=115 < W1 high=120 — INVALID
            Pivot(pd.Timestamp("2024-06-01"), 150, "high"),
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_wave2_beyond_wave0_rejects(self):
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),   # wave 0 low=100
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),
            Pivot(pd.Timestamp("2024-03-01"), 95, "low"),    # wave2 low=95 < wave0 low=100 — INVALID
            Pivot(pd.Timestamp("2024-04-01"), 140, "high"),
            Pivot(pd.Timestamp("2024-05-01"), 125, "low"),
            Pivot(pd.Timestamp("2024-06-01"), 160, "high"),
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_fibonacci_hits_format(self):
        seq = self._bullish_impulse()
        _, _, hits = score_sequence(seq)
        for hit in hits:
            assert "pts" in hit

    def test_returns_tuple_of_three(self):
        seq = self._bullish_impulse()
        result = score_sequence(seq)
        assert len(result) == 3
        is_valid, confidence, hits = result
        assert isinstance(is_valid, bool)
        assert isinstance(confidence, int)
        assert isinstance(hits, list)


class TestGetBestWaveCount:
    def _spy_like_series(self) -> pd.Series:
        np.random.seed(42)
        prices = [400.0]
        for i in range(251):
            change = np.random.normal(0.05, 1.5)
            prices.append(max(300.0, prices[-1] + change))
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        return pd.Series(prices, index=dates)

    def test_returns_best_count_or_none(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        assert result is None or isinstance(result, BestCount)

    def test_confidence_in_range(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert 0 <= result.confidence <= 100

    def test_invalidation_level_is_float(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.invalidation_level, float)

    def test_wave_position_is_string(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.wave_position, str)
            assert len(result.wave_position) > 0

    def test_too_few_bars_returns_none(self):
        short_series = pd.Series(
            [100 + i for i in range(30)],
            index=pd.date_range("2024-01-01", periods=30, freq="B")
        )
        result = get_best_wave_count(short_series)
        assert result is None

    def test_fibonacci_hits_is_list_of_strings(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.fibonacci_hits, list)
            for hit in result.fibonacci_hits:
                assert isinstance(hit, str)
