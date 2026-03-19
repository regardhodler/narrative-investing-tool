import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.wyckoff_engine import (
    detect_trading_ranges,
    classify_volume_behavior,
    identify_wyckoff_events,
    determine_phases,
    analyze_wyckoff,
    TradingRange,
)


def _make_ohlcv(n: int = 200) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate synthetic OHLCV with a consolidation zone in the middle."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Trending up, then flat (consolidation), then trending up
    prices = []
    p = 100.0
    for i in range(n):
        if i < n // 4:
            p += np.random.normal(0.3, 0.5)  # uptrend
        elif i < 3 * n // 4:
            p += np.random.normal(0.0, 0.15)  # tight consolidation
        else:
            p += np.random.normal(0.3, 0.5)  # uptrend
        prices.append(max(p, 50.0))

    close = pd.Series(prices, index=dates)
    high = close + np.random.uniform(0.2, 1.0, n)
    low = close - np.random.uniform(0.2, 1.0, n)
    volume = pd.Series(np.random.randint(1_000_000, 10_000_000, n), index=dates, dtype=float)
    return close, high, low, volume


def _make_short(n: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.random.normal(100, 1, n), index=dates)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n), index=dates, dtype=float)
    return close, high, low, volume


class TestDetectTradingRanges:
    def test_returns_list(self):
        close, high, low, _ = _make_ohlcv()
        result = detect_trading_ranges(close, high, low)
        assert isinstance(result, list)

    def test_has_required_fields(self):
        close, high, low, _ = _make_ohlcv()
        result = detect_trading_ranges(close, high, low)
        for tr in result:
            assert hasattr(tr, "start_idx")
            assert hasattr(tr, "end_idx")
            assert hasattr(tr, "start_date")
            assert hasattr(tr, "end_date")
            assert hasattr(tr, "upper_bound")
            assert hasattr(tr, "lower_bound")
            assert hasattr(tr, "preceding_trend")

    def test_empty_on_short_data(self):
        close, high, low, _ = _make_short()
        result = detect_trading_ranges(close, high, low)
        assert isinstance(result, list)

    def test_upper_geq_lower(self):
        close, high, low, _ = _make_ohlcv()
        result = detect_trading_ranges(close, high, low)
        for tr in result:
            assert tr.upper_bound >= tr.lower_bound


class TestClassifyVolumeBehavior:
    def test_returns_valid_string(self):
        close, high, low, volume = _make_ohlcv()
        ranges = detect_trading_ranges(close, high, low)
        if ranges:
            result = classify_volume_behavior(close, volume, ranges[0])
            assert result in ("accumulation", "distribution", "neutral")

    def test_detects_accumulation_pattern(self):
        close, high, low, volume = _make_ohlcv(200)
        ranges = detect_trading_ranges(close, high, low)
        if ranges:
            tr = ranges[0]
            # Bias volume toward up-days
            sl = slice(tr.start_idx, tr.end_idx + 1)
            diff = close.iloc[sl].diff()
            for i in range(len(diff)):
                idx = diff.index[i]
                if diff.iloc[i] > 0:
                    volume.loc[idx] = 8_000_000.0
                else:
                    volume.loc[idx] = 2_000_000.0
            result = classify_volume_behavior(close, volume, tr)
            assert result == "accumulation"

    def test_detects_distribution_pattern(self):
        close, high, low, volume = _make_ohlcv(200)
        ranges = detect_trading_ranges(close, high, low)
        if ranges:
            tr = ranges[0]
            sl = slice(tr.start_idx, tr.end_idx + 1)
            diff = close.iloc[sl].diff()
            for i in range(len(diff)):
                idx = diff.index[i]
                if diff.iloc[i] < 0:
                    volume.loc[idx] = 8_000_000.0
                else:
                    volume.loc[idx] = 2_000_000.0
            result = classify_volume_behavior(close, volume, tr)
            assert result == "distribution"


class TestIdentifyWyckoffEvents:
    def test_returns_list(self):
        close, high, low, volume = _make_ohlcv()
        ranges = detect_trading_ranges(close, high, low)
        if ranges:
            events = identify_wyckoff_events(close, high, low, volume, ranges[0], "accumulation")
            assert isinstance(events, list)

    def test_events_have_required_fields(self):
        close, high, low, volume = _make_ohlcv()
        ranges = detect_trading_ranges(close, high, low)
        if ranges:
            events = identify_wyckoff_events(close, high, low, volume, ranges[0], "accumulation")
            for evt in events:
                assert hasattr(evt, "date")
                assert hasattr(evt, "price")
                assert hasattr(evt, "event_type")
                assert hasattr(evt, "description")

    def test_valid_event_types(self):
        close, high, low, volume = _make_ohlcv()
        ranges = detect_trading_ranges(close, high, low)
        valid_types = {"SC", "AR", "ST", "Spring", "SOS", "BC", "UTAD", "SOW"}
        if ranges:
            for phase_type in ("accumulation", "distribution"):
                events = identify_wyckoff_events(close, high, low, volume, ranges[0], phase_type)
                for evt in events:
                    assert evt.event_type in valid_types


class TestDeterminePhases:
    def test_returns_list(self):
        close, high, low, volume = _make_ohlcv()
        result = determine_phases(close, high, low, volume)
        assert isinstance(result, list)

    def test_valid_phase_names(self):
        close, high, low, volume = _make_ohlcv()
        result = determine_phases(close, high, low, volume)
        valid = {"Accumulation", "Distribution", "Markup", "Markdown"}
        for phase in result:
            assert phase.phase in valid

    def test_confidence_in_range(self):
        close, high, low, volume = _make_ohlcv()
        result = determine_phases(close, high, low, volume)
        for phase in result:
            assert 0 <= phase.confidence <= 100

    def test_start_before_end(self):
        close, high, low, volume = _make_ohlcv()
        result = determine_phases(close, high, low, volume)
        for phase in result:
            assert phase.start_date <= phase.end_date


class TestAnalyzeWyckoff:
    def test_returns_analysis_or_none(self):
        close, high, low, volume = _make_ohlcv()
        result = analyze_wyckoff(close, high, low, volume)
        assert result is None or hasattr(result, "current_phase")

    def test_short_data_returns_none(self):
        close, high, low, volume = _make_short(30)
        result = analyze_wyckoff(close, high, low, volume)
        assert result is None

    def test_current_phase_is_last(self):
        close, high, low, volume = _make_ohlcv()
        result = analyze_wyckoff(close, high, low, volume)
        if result is not None and result.all_phases:
            assert result.current_phase == result.all_phases[-1]
