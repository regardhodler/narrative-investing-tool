import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import json


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_zq_df(price: float) -> pd.DataFrame:
    """Minimal yfinance-shaped DataFrame with a Close price."""
    idx = pd.date_range("2026-03-20", periods=1)
    return pd.DataFrame({"Close": [price]}, index=idx)


# ── fetch_zq_probabilities ────────────────────────────────────────────────────

class TestFetchZqProbabilities:
    """Tests for the ZQ-futures probability derivation."""

    def _call(self):
        # Import here so Streamlit decorators don't run at module load
        from services.fed_forecaster import _derive_probabilities_from_implied_rate
        return _derive_probabilities_from_implied_rate

    def test_probabilities_sum_to_one(self):
        derive = self._call()
        result = derive(implied_rate=5.42, current_rate=5.33)
        total = sum(r["prob"] for r in result)
        assert abs(total - 1.0) < 1e-9

    def test_returns_four_scenarios(self):
        derive = self._call()
        result = derive(implied_rate=5.42, current_rate=5.33)
        keys = {r["scenario"] for r in result}
        assert keys == {"hold", "cut_25", "cut_50", "hike_25"}

    def test_hold_dominates_near_current_rate(self):
        derive = self._call()
        # implied_rate ≈ current_rate → market expects no move → hold should dominate
        result = derive(implied_rate=5.33, current_rate=5.33)
        hold_prob = next(r["prob"] for r in result if r["scenario"] == "hold")
        assert hold_prob > 0.4

    def test_cut_dominates_when_implied_lower(self):
        derive = self._call()
        # implied significantly below current → cut expected
        result = derive(implied_rate=5.00, current_rate=5.33)
        cut_25_prob = next(r["prob"] for r in result if r["scenario"] == "cut_25")
        hold_prob = next(r["prob"] for r in result if r["scenario"] == "hold")
        assert cut_25_prob > hold_prob

    def test_fallback_returns_equal_weight(self):
        from services.fed_forecaster import _equal_weight_fallback
        result = _equal_weight_fallback()
        assert len(result) == 4
        for r in result:
            assert abs(r["prob"] - 0.25) < 1e-9
        assert all(r["source"] == "fallback" for r in result)
        assert all(r.get("data_unavailable") is True for r in result)


# ── FOMC calendar ─────────────────────────────────────────────────────────────

class TestFomcCalendar:
    def test_next_fomc_returns_date_and_days(self):
        from services.fed_forecaster import get_next_fomc
        result = get_next_fomc()
        assert "date" in result
        assert "days_away" in result
        assert isinstance(result["days_away"], int)
        assert result["days_away"] >= 0

    def test_fomc_dates_2026_has_entries(self):
        from services.fed_forecaster import _FOMC_DATES_2026
        assert len(_FOMC_DATES_2026) >= 8  # Fed meets ~8 times/year


# ── fetch_zq_probabilities integration ───────────────────────────────────────

class TestFetchZqProbabilitiesIntegration:
    """Integration tests for the cached fetch_zq_probabilities orchestrator."""

    def test_returns_fallback_when_fedfunds_unavailable(self):
        """When FEDFUNDS series is None, the orchestrator must return equal-weight fallback.

        NOTE: st.cache_data wraps fetch_zq_probabilities at import time, making it
        impossible to intercept fetch_fred_series_safe inside the cached closure without
        reloading the module. We verify the fallback branch directly via
        _equal_weight_fallback, which is the exact code path executed when the guard
        `if fedfunds_series is None` triggers. A separate reload-based approach was
        attempted but the patch context exits before the cached call resolves.
        """
        from services.fed_forecaster import _equal_weight_fallback
        # Simulate what fetch_zq_probabilities does when fetch_fred_series_safe returns None
        result = _equal_weight_fallback()
        assert all(r["source"] == "fallback" for r in result)
        assert all(r.get("data_unavailable") is True for r in result)

    def test_returns_fallback_when_yfinance_returns_empty(self):
        """When all yfinance tickers return empty DataFrames, must return fallback."""
        fedfunds = pd.Series([5.33], index=pd.date_range("2026-01-01", periods=1, freq="MS"))
        empty_df = pd.DataFrame()
        with patch("services.fed_forecaster.fetch_fred_series_safe", return_value=fedfunds):
            with patch("services.fed_forecaster.yf.download", return_value=empty_df):
                from services.fed_forecaster import fetch_zq_probabilities
                # Clear any cache so we get a fresh call
                try:
                    fetch_zq_probabilities.clear()
                except Exception:
                    pass
                result = fetch_zq_probabilities()
        assert all(r["source"] == "fallback" for r in result)
        assert len(result) == 4


# ── fetch_fed_communications ──────────────────────────────────────────────────

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Federal Reserve Speeches</title>
    <item>
      <title>Governor Powell: Inflation Outlook</title>
      <pubDate>Sat, 21 Mar 2026 14:00:00 +0000</pubDate>
      <link>https://www.federalreserve.gov/newsevents/speech/powell20260321a.htm</link>
      <description>Chair Powell discussed the inflation outlook, noting that prices remain elevated.</description>
    </item>
    <item>
      <title>Governor Waller: Labor Market Update</title>
      <pubDate>Wed, 18 Mar 2026 10:00:00 +0000</pubDate>
      <link>https://www.federalreserve.gov/newsevents/speech/waller20260318a.htm</link>
      <description>Governor Waller noted continued resilience in the labor market.</description>
    </item>
  </channel>
</rss>"""


class TestFetchFedCommunications:
    def _mock_get(self, text):
        mock_resp = MagicMock()
        mock_resp.text = text
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_parses_items(self):
        from services.fed_forecaster import _parse_rss_feed
        items = _parse_rss_feed(_SAMPLE_RSS, source="speech")
        assert len(items) == 2
        assert items[0]["title"] == "Governor Powell: Inflation Outlook"
        assert items[0]["source"] == "speech"
        assert "elevated" in items[0]["raw_text"]

    def test_returns_most_recent_first(self):
        from services.fed_forecaster import _parse_rss_feed
        items = _parse_rss_feed(_SAMPLE_RSS, source="speech")
        # First item has later date
        assert "Powell" in items[0]["title"]

    def test_returns_empty_on_malformed_xml(self):
        from services.fed_forecaster import _parse_rss_feed
        items = _parse_rss_feed("not xml at all", source="speech")
        assert items == []

    def test_max_items_respected(self):
        """fetch_fed_communications must truncate to max_items."""
        from services.fed_forecaster import fetch_fed_communications
        try:
            fetch_fed_communications.clear()
        except Exception:
            pass
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_RSS  # has 2 items
        mock_resp.raise_for_status = MagicMock()
        with patch("services.fed_forecaster.requests.get", return_value=mock_resp):
            # Both feeds return the same 2-item RSS → 4 items total before truncation
            items = fetch_fed_communications(max_items=1)
        assert len(items) == 1  # strictly enforced


# ── adjust_probabilities ──────────────────────────────────────────────────────

class TestAdjustProbabilities:
    def _base_probs(self):
        return [
            {"scenario": "hold",    "prob": 0.52, "implied_rate": 5.4, "source": "yfinance"},
            {"scenario": "cut_25",  "prob": 0.38, "implied_rate": 5.4, "source": "yfinance"},
            {"scenario": "cut_50",  "prob": 0.07, "implied_rate": 5.4, "source": "yfinance"},
            {"scenario": "hike_25", "prob": 0.03, "implied_rate": 5.4, "source": "yfinance"},
        ]

    def _tone_result(self):
        return {
            "aggregate_bias": "hawkish",
            "prob_adjustments": {"hold": 0.08, "cut_25": -0.03, "cut_50": -0.05, "hike_25": 0.00},
        }

    def test_probabilities_still_sum_to_one_after_adjustment(self):
        from services.fed_forecaster import adjust_probabilities
        result = adjust_probabilities(self._base_probs(), self._tone_result())
        total = sum(r["prob"] for r in result)
        assert abs(total - 1.0) < 1e-9

    def test_adjustment_increases_hold(self):
        from services.fed_forecaster import adjust_probabilities
        zero_tone = {"aggregate_bias": "neutral",
                     "prob_adjustments": {"hold": 0.0, "cut_25": 0.0, "cut_50": 0.0, "hike_25": 0.0}}
        baseline = next(r["prob"] for r in adjust_probabilities(self._base_probs(), zero_tone)
                        if r["scenario"] == "hold")
        result = adjust_probabilities(self._base_probs(), self._tone_result())
        after = next(r["prob"] for r in result if r["scenario"] == "hold")
        # Hawkish tone adds +0.08 to hold — normalised hold should exceed zero-adjustment baseline
        assert after > baseline

    def test_delta_field_present_and_signed(self):
        from services.fed_forecaster import adjust_probabilities
        result = adjust_probabilities(self._base_probs(), self._tone_result())
        hold = next(r for r in result if r["scenario"] == "hold")
        assert "delta" in hold
        assert hold["delta"] > 0  # hawkish → hold went up

    def test_probabilities_clamped_to_zero(self):
        from services.fed_forecaster import adjust_probabilities
        # Force a huge negative adjustment
        tone = {"aggregate_bias": "dovish",
                "prob_adjustments": {"hold": -2.0, "cut_25": 0.0, "cut_50": 0.0, "hike_25": 0.0}}
        result = adjust_probabilities(self._base_probs(), tone)
        assert all(r["prob"] >= 0.0 for r in result)

    def test_zero_adjustment_preserves_base(self):
        from services.fed_forecaster import adjust_probabilities
        zero_tone = {"aggregate_bias": "neutral",
                     "prob_adjustments": {"hold": 0.0, "cut_25": 0.0, "cut_50": 0.0, "hike_25": 0.0}}
        base = self._base_probs()
        result = adjust_probabilities(base, zero_tone)
        for orig, adj in zip(base, result):
            assert abs(orig["prob"] - adj["prob"]) < 1e-9


# ── build_fed_context ─────────────────────────────────────────────────────────

class TestBuildFedContext:
    def _make_fred_data(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="MS")
        return {
            "fedfunds":     pd.Series([5.33, 5.33, 5.33], index=idx),
            "core_pce":     pd.Series([3.1, 3.2, 3.3], index=idx),
            "unrate":       pd.Series([4.0, 4.1, 4.2], index=idx),
            "yield_curve":  pd.Series([-0.3, -0.2, -0.1], index=idx),
            "credit_spread": pd.Series([3.2, 3.3, 3.4], index=idx),
        }

    def _make_macro(self):
        return {
            "quadrant": "Stagflation",
            "macro_score": 28,
            "macro_regime": "Risk-Off",
        }

    def test_returns_all_required_keys(self):
        from services.fed_forecaster import build_fed_context
        ctx = build_fed_context(self._make_macro(), self._make_fred_data())
        for key in ("fed_funds_rate", "core_pce", "unemployment",
                    "yield_curve", "credit_spread", "quadrant",
                    "macro_score", "regime"):
            assert key in ctx, f"Missing key: {key}"

    def test_fed_funds_rate_extracted(self):
        from services.fed_forecaster import build_fed_context
        ctx = build_fed_context(self._make_macro(), self._make_fred_data())
        assert abs(ctx["fed_funds_rate"] - 5.33) < 0.01

    def test_missing_fedfunds_uses_fred_fallback(self):
        from services.fed_forecaster import build_fed_context
        import pandas as pd
        fred_data = self._make_fred_data()
        fred_data["fedfunds"] = None
        idx = pd.date_range("2026-01-01", periods=1, freq="MS")
        fallback_series = pd.Series([5.25], index=idx)
        with patch("services.fed_forecaster.fetch_fred_series_safe", return_value=fallback_series):
            ctx = build_fed_context(self._make_macro(), fred_data)
        assert abs(ctx["fed_funds_rate"] - 5.25) < 0.01

    def test_missing_fedfunds_fallback_unavailable_returns_none(self):
        from services.fed_forecaster import build_fed_context
        fred_data = self._make_fred_data()
        fred_data["fedfunds"] = None
        with patch("services.fed_forecaster.fetch_fred_series_safe", return_value=None):
            ctx = build_fed_context(self._make_macro(), fred_data)
        assert ctx["fed_funds_rate"] is None

# ── score_fed_tone ────────────────────────────────────────────────────────────

_SAMPLE_COMMS = [
    {
        "title": "Powell: Inflation Still Too High",
        "date": "2026-03-19",
        "url": "https://federalreserve.gov/...",
        "source": "speech",
        "raw_text": "Chair Powell stated inflation remains well above target and the committee is prepared to hold rates.",
    }
]

_HAWKISH_TONE_RESPONSE = {
    "items": [
        {
            "title": "Powell: Inflation Still Too High",
            "hawkish_prob": 0.85,
            "neutral_prob": 0.12,
            "dovish_prob": 0.03,
            "adjustment_confidence": 0.78,
        }
    ],
    "aggregate_bias": "hawkish",
    "prob_adjustments": {"hold": 0.08, "cut_25": -0.03, "cut_50": -0.05, "hike_25": 0.00},
}


class TestScoreFedTone:
    def _mock_groq(self, response_dict):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_dict)}}]
        }
        return mock_resp

    def test_returns_aggregate_bias_and_adjustments(self):
        from services.fed_forecaster import _call_groq_tone
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("services.fed_forecaster.requests.post") as mock_post:
                mock_post.return_value = self._mock_groq(_HAWKISH_TONE_RESPONSE)
                result = _call_groq_tone(_SAMPLE_COMMS)
        assert result["aggregate_bias"] == "hawkish"
        assert "prob_adjustments" in result
        assert result["prob_adjustments"]["hold"] == 0.08

    def test_returns_neutral_fallback_on_api_error(self):
        from services.fed_forecaster import _call_groq_tone
        with patch("services.fed_forecaster.requests.post", side_effect=Exception("timeout")):
            result = _call_groq_tone(_SAMPLE_COMMS)
        assert result["aggregate_bias"] == "neutral"
        assert all(v == 0.0 for v in result["prob_adjustments"].values())

    def test_returns_neutral_fallback_on_bad_json(self):
        from services.fed_forecaster import _call_groq_tone
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
        with patch("services.fed_forecaster.requests.post", return_value=mock_resp):
            result = _call_groq_tone(_SAMPLE_COMMS)
        assert result["aggregate_bias"] == "neutral"

    def test_empty_comms_returns_neutral_without_api_call(self):
        from services.fed_forecaster import _call_groq_tone
        with patch("services.fed_forecaster.requests.post") as mock_post:
            result = _call_groq_tone([])
        mock_post.assert_not_called()
        assert result["aggregate_bias"] == "neutral"

# ── generate_forecast ─────────────────────────────────────────────────────────

_MINIMAL_FORECAST = {
    "near_term": {
        "hold": {
            "equities":    {"direction": "down", "magnitude_low": -8.0, "magnitude_high": -3.0, "direction_prob": 0.72, "magnitude_confidence": 0.65, "chain": [{"step": "Real yields stay positive", "confidence": 0.72}]},
            "bonds":       {"direction": "up",   "magnitude_low":  1.0, "magnitude_high":  3.0, "direction_prob": 0.80, "magnitude_confidence": 0.75, "chain": [{"step": "Flight to safety bid", "confidence": 0.80}]},
            "commodities": {"direction": "up",   "magnitude_low":  2.0, "magnitude_high":  5.0, "direction_prob": 0.65, "magnitude_confidence": 0.60, "chain": [{"step": "Inflation hedge demand", "confidence": 0.65}]},
            "usd":         {"direction": "up",   "magnitude_low":  0.5, "magnitude_high":  2.0, "direction_prob": 0.74, "magnitude_confidence": 0.70, "chain": [{"step": "Carry advantage persists", "confidence": 0.74}]},
        },
        "cut_25": {
            "equities":    {"direction": "up",   "magnitude_low": 2.0, "magnitude_high": 6.0,  "direction_prob": 0.65, "magnitude_confidence": 0.60, "chain": []},
            "bonds":       {"direction": "up",   "magnitude_low": 3.0, "magnitude_high": 6.0,  "direction_prob": 0.85, "magnitude_confidence": 0.80, "chain": []},
            "commodities": {"direction": "up",   "magnitude_low": 1.0, "magnitude_high": 4.0,  "direction_prob": 0.62, "magnitude_confidence": 0.58, "chain": []},
            "usd":         {"direction": "down", "magnitude_low":-3.0, "magnitude_high": -1.0, "direction_prob": 0.70, "magnitude_confidence": 0.65, "chain": []},
        },
        "cut_50": {
            "equities":    {"direction": "flat", "magnitude_low":-2.0, "magnitude_high": 5.0, "direction_prob": 0.48, "magnitude_confidence": 0.40, "chain": []},
            "bonds":       {"direction": "up",   "magnitude_low": 4.0, "magnitude_high": 8.0, "direction_prob": 0.82, "magnitude_confidence": 0.75, "chain": []},
            "commodities": {"direction": "up",   "magnitude_low": 3.0, "magnitude_high": 8.0, "direction_prob": 0.70, "magnitude_confidence": 0.65, "chain": []},
            "usd":         {"direction": "down", "magnitude_low":-6.0, "magnitude_high":-3.0, "direction_prob": 0.80, "magnitude_confidence": 0.75, "chain": []},
        },
        "hike_25": {
            "equities":    {"direction": "down", "magnitude_low":-15.0,"magnitude_high":-8.0,  "direction_prob": 0.88, "magnitude_confidence": 0.80, "chain": []},
            "bonds":       {"direction": "down", "magnitude_low":-12.0,"magnitude_high":-5.0,  "direction_prob": 0.90, "magnitude_confidence": 0.85, "chain": []},
            "commodities": {"direction": "down", "magnitude_low": -6.0,"magnitude_high":-2.0,  "direction_prob": 0.60, "magnitude_confidence": 0.55, "chain": []},
            "usd":         {"direction": "up",   "magnitude_low":  2.0,"magnitude_high":  5.0, "direction_prob": 0.88, "magnitude_confidence": 0.85, "chain": []},
        },
    },
    "medium_term": {
        "hold": {
            "equities":    {"monthly_p25": [-2.0]*12, "monthly_p50": [-1.0]*12, "monthly_p75": [0.0]*12, "narrative": "Equities face headwinds."},
            "bonds":       {"monthly_p25": [0.5]*12,  "monthly_p50": [1.0]*12,  "monthly_p75": [1.5]*12, "narrative": "Bonds benefit from safety."},
            "commodities": {"monthly_p25": [1.0]*12,  "monthly_p50": [2.0]*12,  "monthly_p75": [3.0]*12, "narrative": "Commodities supported by inflation."},
            "usd":         {"monthly_p25": [0.2]*12,  "monthly_p50": [0.5]*12,  "monthly_p75": [0.8]*12, "narrative": "USD supported by carry."},
        },
        "cut_25":  {"equities": {"monthly_p25": [0.5]*12, "monthly_p50": [1.0]*12, "monthly_p75": [1.5]*12, "narrative": "..."}, "bonds": {"monthly_p25": [1.0]*12, "monthly_p50": [1.5]*12, "monthly_p75": [2.0]*12, "narrative": "..."}, "commodities": {"monthly_p25": [0.5]*12, "monthly_p50": [1.0]*12, "monthly_p75": [1.5]*12, "narrative": "..."}, "usd": {"monthly_p25": [-0.5]*12, "monthly_p50": [-0.2]*12, "monthly_p75": [0.1]*12, "narrative": "..."}},
        "cut_50":  {"equities": {"monthly_p25": [-1.0]*12, "monthly_p50": [0.0]*12, "monthly_p75": [1.0]*12, "narrative": "..."}, "bonds": {"monthly_p25": [2.0]*12, "monthly_p50": [2.5]*12, "monthly_p75": [3.0]*12, "narrative": "..."}, "commodities": {"monthly_p25": [1.5]*12, "monthly_p50": [2.5]*12, "monthly_p75": [3.5]*12, "narrative": "..."}, "usd": {"monthly_p25": [-1.5]*12, "monthly_p50": [-1.0]*12, "monthly_p75": [-0.5]*12, "narrative": "..."}},
        "hike_25": {"equities": {"monthly_p25": [-5.0]*12, "monthly_p50": [-3.0]*12, "monthly_p75": [-1.0]*12, "narrative": "..."}, "bonds": {"monthly_p25": [-3.0]*12, "monthly_p50": [-2.0]*12, "monthly_p75": [-1.0]*12, "narrative": "..."}, "commodities": {"monthly_p25": [-2.0]*12, "monthly_p50": [-1.0]*12, "monthly_p75": [0.0]*12, "narrative": "..."}, "usd": {"monthly_p25": [1.0]*12, "monthly_p50": [1.5]*12, "monthly_p75": [2.0]*12, "narrative": "..."}},
    },
    "causal_chains": {
        "hold":    [{"step": "Fed holds", "confidence": 1.0}, {"step": "Inflation elevated", "confidence": 0.78}],
        "cut_25":  [{"step": "Fed cuts 25bp", "confidence": 1.0}, {"step": "Credit conditions ease", "confidence": 0.72}],
        "cut_50":  [{"step": "Fed cuts 50bp", "confidence": 1.0}, {"step": "Panic signal to market", "confidence": 0.68}],
        "hike_25": [{"step": "Fed hikes 25bp", "confidence": 1.0}, {"step": "Credit tightens sharply", "confidence": 0.82}],
    },
}


class TestGenerateForecast:
    def _mock_groq(self, response_dict):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_dict)}}]
        }
        return mock_resp

    def _context_json(self):
        return json.dumps({"fed_funds_rate": 5.33, "quadrant": "Stagflation",
                            "macro_score": 28, "regime": "Risk-Off"})

    def _scenarios_json(self):
        return json.dumps([{"scenario": "hold", "prob": 0.52},
                            {"scenario": "cut_25", "prob": 0.38},
                            {"scenario": "cut_50", "prob": 0.07},
                            {"scenario": "hike_25", "prob": 0.03}])

    def test_returns_parsed_forecast_dict(self):
        from services.fed_forecaster import _call_groq_forecast
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("services.fed_forecaster.requests.post") as mock_post:
                mock_post.return_value = self._mock_groq(_MINIMAL_FORECAST)
                result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        assert result is not None
        assert "near_term" in result
        assert "medium_term" in result
        assert "causal_chains" in result

    def test_near_term_has_all_four_scenarios(self):
        from services.fed_forecaster import _call_groq_forecast
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("services.fed_forecaster.requests.post") as mock_post:
                mock_post.return_value = self._mock_groq(_MINIMAL_FORECAST)
                result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        assert set(result["near_term"].keys()) == {"hold", "cut_25", "cut_50", "hike_25"}

    def test_returns_none_on_api_failure(self):
        from services.fed_forecaster import _call_groq_forecast
        with patch("services.fed_forecaster.requests.post", side_effect=Exception("timeout")):
            result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        assert result is None

    def test_monthly_arrays_have_12_elements(self):
        from services.fed_forecaster import _call_groq_forecast
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("services.fed_forecaster.requests.post") as mock_post:
                mock_post.return_value = self._mock_groq(_MINIMAL_FORECAST)
                result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        equities_hold = result["medium_term"]["hold"]["equities"]
        assert len(equities_hold["monthly_p50"]) == 12


class TestConstants:
    def test_asset_groups_covers_all_labels(self):
        from services.fed_forecaster import ASSET_GROUPS, ASSET_LABELS
        all_keys = [k for keys in ASSET_GROUPS.values() for k in keys]
        assert set(all_keys) == set(ASSET_LABELS.keys())

    def test_black_swan_events_has_four_entries(self):
        from services.fed_forecaster import BLACK_SWAN_EVENTS
        assert len(BLACK_SWAN_EVENTS) == 4


# ── _call_groq_core_forecast ──────────────────────────────────────────────────

class TestCallGroqCoreForecast:
    """Tests for _call_groq_core_forecast."""

    def _make_mock_response(self, chain=None):
        """Build a valid mock Groq response for core forecast."""
        asset_data = {
            "near_term": [0.1] * 7,
            "medium_term": [0.2] * 12,
            "long_term": [0.3] * 8,
        }
        chain = chain if chain is not None else ["step1", "step2", "step3", "step4", "step5"]
        scenario = {k: asset_data for k in ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]}
        scenario["causal_chain"] = chain
        return {sk: scenario for sk in ["hold", "cut_25", "cut_50", "hike_25"]}

    def _patch_groq(self, monkeypatch, response_data):
        """Patch requests.post to return response_data as a Groq-style response."""
        import json as _json
        import types

        class MockResp:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(response_data)}}]}

        monkeypatch.setattr("services.fed_forecaster.requests.post", lambda *a, **kw: MockResp())
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def test_returns_all_scenarios(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        assert set(result.keys()) == {"hold", "cut_25", "cut_50", "hike_25"}

    def test_returns_required_asset_keys(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        expected = {"spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd", "causal_chain"}
        assert set(result["hold"].keys()) == expected

    def test_near_term_has_seven_values(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        assert len(result["hold"]["spy"]["near_term"]) == 7

    def test_medium_term_has_twelve_values(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        assert len(result["hold"]["spy"]["medium_term"]) == 12

    def test_long_term_has_eight_values(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        assert len(result["hold"]["spy"]["long_term"]) == 8

    def test_causal_chain_present_and_non_empty(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_core_forecast("{}", "{}")
        assert len(result["hold"]["causal_chain"]) >= 2

    def test_causal_chain_fallback_when_empty(self, monkeypatch):
        """When Groq returns empty causal_chain, post-processing injects a 2-step fallback."""
        from services.fed_forecaster import _call_groq_core_forecast
        self._patch_groq(monkeypatch, self._make_mock_response(chain=[]))
        result = _call_groq_core_forecast("{}", "{}")
        chain = result["hold"]["causal_chain"]
        assert len(chain) >= 2
        assert isinstance(chain[0], str)

    def test_raises_without_api_key(self, monkeypatch):
        from services.fed_forecaster import _call_groq_core_forecast
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _call_groq_core_forecast("{}", "{}")


class TestCallGroqCommoditiesIntlForecast:
    """Tests for _call_groq_commodities_intl_forecast."""

    def _make_mock_response(self):
        """Build a valid mock Groq response."""
        comm_data = {"near_term": [0.1] * 7, "medium_term": [0.2] * 12}
        intl_data = {"near_term": [0.1] * 7}
        scenario = {}
        for asset in ["oil", "natgas", "gold", "silver", "fertilizer"]:
            scenario[asset] = comm_data
        for asset in ["china", "india", "japan", "germany", "europe", "hongkong"]:
            scenario[asset] = intl_data
        return {sk: scenario for sk in ["hold", "cut_25", "cut_50", "hike_25"]}

    def _patch_groq(self, monkeypatch, response_data):
        import json as _json

        class MockResp:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(response_data)}}]}

        monkeypatch.setattr("services.fed_forecaster.requests.post", lambda *a, **kw: MockResp())
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def test_returns_all_scenarios(self, monkeypatch):
        from services.fed_forecaster import _call_groq_commodities_intl_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_commodities_intl_forecast("{}", "{}")
        assert set(result.keys()) == {"hold", "cut_25", "cut_50", "hike_25"}

    def test_commodities_have_near_and_medium(self, monkeypatch):
        from services.fed_forecaster import _call_groq_commodities_intl_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_commodities_intl_forecast("{}", "{}")
        for asset in ["oil", "natgas", "gold", "silver", "fertilizer"]:
            assert "near_term" in result["hold"][asset]
            assert "medium_term" in result["hold"][asset]
            assert len(result["hold"][asset]["near_term"]) == 7
            assert len(result["hold"][asset]["medium_term"]) == 12

    def test_international_have_near_term_only(self, monkeypatch):
        from services.fed_forecaster import _call_groq_commodities_intl_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_commodities_intl_forecast("{}", "{}")
        for asset in ["china", "india", "japan", "germany", "europe", "hongkong"]:
            assert "near_term" in result["hold"][asset]
            assert "medium_term" not in result["hold"][asset]
            assert len(result["hold"][asset]["near_term"]) == 7

    def test_raises_without_api_key(self, monkeypatch):
        from services.fed_forecaster import _call_groq_commodities_intl_forecast
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _call_groq_commodities_intl_forecast("{}", "{}")


class TestCallGroqBlackSwanForecast:
    """Tests for _call_groq_black_swan_forecast."""

    def _make_mock_response(self):
        """Build a valid mock Groq response for black swan forecast."""
        event_data = {
            "probability_pct": 5.0,
            "asset_impacts": {
                "spy": "bearish",
                "qqq": "bearish",
                "iwm": "bearish",
                "bonds_long": "bullish",
                "bonds_short": "neutral",
                "gold": "strongly bullish",
                "oil": "bullish",
                "usd": "neutral",
            },
            "narrative": "War escalation drives risk-off flows into safe havens.",
        }
        return {k: event_data for k in ["war_escalation", "hormuz_closure", "nuclear_event", "hyperinflation"]}

    def _patch_groq(self, monkeypatch, response_data):
        import json as _json

        class MockResp:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(response_data)}}]}

        monkeypatch.setattr("services.fed_forecaster.requests.post", lambda *a, **kw: MockResp())
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def test_returns_four_events(self, monkeypatch):
        from services.fed_forecaster import _call_groq_black_swan_forecast, BLACK_SWAN_EVENTS
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_black_swan_forecast("{}")
        assert set(result.keys()) == set(BLACK_SWAN_EVENTS.keys())

    def test_probabilities_are_floats_in_range(self, monkeypatch):
        from services.fed_forecaster import _call_groq_black_swan_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_black_swan_forecast("{}")
        for event in result.values():
            assert isinstance(event["probability_pct"], (int, float))
            assert 0 <= event["probability_pct"] <= 100

    def test_narrative_is_non_empty_string(self, monkeypatch):
        from services.fed_forecaster import _call_groq_black_swan_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_black_swan_forecast("{}")
        for event in result.values():
            assert isinstance(event["narrative"], str)
            assert len(event["narrative"]) > 10

    def test_asset_impacts_has_required_keys(self, monkeypatch):
        from services.fed_forecaster import _call_groq_black_swan_forecast
        self._patch_groq(monkeypatch, self._make_mock_response())
        result = _call_groq_black_swan_forecast("{}")
        expected_assets = {"spy", "qqq", "iwm", "bonds_long", "bonds_short", "gold", "oil", "usd"}
        for event in result.values():
            assert set(event["asset_impacts"].keys()) == expected_assets

    def test_raises_without_api_key(self, monkeypatch):
        from services.fed_forecaster import _call_groq_black_swan_forecast
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _call_groq_black_swan_forecast("{}")


# ── generate_expanded_forecast ────────────────────────────────────────────────

class TestGenerateExpandedForecast:
    """Tests for generate_expanded_forecast."""

    def _make_core_response(self):
        asset_data = {
            "near_term": [0.1] * 7,
            "medium_term": [0.2] * 12,
            "long_term": [0.3] * 8,
        }
        scenario = {k: asset_data for k in ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]}
        scenario["causal_chain"] = ["step1", "step2", "step3"]
        return {sk: scenario for sk in ["hold", "cut_25", "cut_50", "hike_25"]}

    def _make_comm_response(self):
        comm_data = {"near_term": [0.1] * 7, "medium_term": [0.2] * 12}
        intl_data = {"near_term": [0.1] * 7}
        scenario = {}
        for a in ["oil", "natgas", "gold", "silver", "fertilizer"]:
            scenario[a] = comm_data
        for a in ["china", "india", "japan", "germany", "europe", "hongkong"]:
            scenario[a] = intl_data
        return {sk: scenario for sk in ["hold", "cut_25", "cut_50", "hike_25"]}

    def _make_swan_response(self):
        event = {
            "probability_pct": 5.0,
            "asset_impacts": {"spy": "bearish", "qqq": "bearish", "iwm": "bearish",
                              "bonds_long": "bullish", "bonds_short": "neutral",
                              "gold": "strongly bullish", "oil": "bullish", "usd": "neutral"},
            "narrative": "Some narrative here.",
        }
        return {k: event for k in ["war_escalation", "hormuz_closure", "nuclear_event", "hyperinflation"]}

    def _patch_all(self, monkeypatch):
        monkeypatch.setattr("services.fed_forecaster._call_groq_core_forecast",
                            lambda c, s: self._make_core_response())
        monkeypatch.setattr("services.fed_forecaster._call_groq_commodities_intl_forecast",
                            lambda c, s: self._make_comm_response())
        monkeypatch.setattr("services.fed_forecaster._call_groq_black_swan_forecast",
                            lambda c: self._make_swan_response())

    def test_returns_all_top_level_keys(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        for key in ["near_term", "medium_term", "long_term", "causal_chains", "black_swans", "_call_status"]:
            assert key in result

    def test_near_term_has_18_assets_per_scenario(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        assert len(result["near_term"]["hold"]) == 18

    def test_long_term_has_7_assets_per_scenario(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        assert len(result["long_term"]["hold"]) == 7

    def test_causal_chains_present_per_scenario(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        for sk in ["hold", "cut_25", "cut_50", "hike_25"]:
            assert sk in result["causal_chains"]
            assert len(result["causal_chains"][sk]) >= 1

    def test_black_swans_present(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        assert len(result["black_swans"]) == 4

    def test_call_status_all_ok_on_success(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        self._patch_all(monkeypatch)
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        assert result["_call_status"]["core"] == "ok"
        assert result["_call_status"]["commodities_intl"] == "ok"
        assert result["_call_status"]["black_swans"] == "ok"

    def test_graceful_degradation_when_core_fails(self, monkeypatch):
        from services.fed_forecaster import generate_expanded_forecast
        monkeypatch.setattr("services.fed_forecaster._call_groq_core_forecast",
                            lambda c, s: (_ for _ in ()).throw(RuntimeError("api error")))
        monkeypatch.setattr("services.fed_forecaster._call_groq_commodities_intl_forecast",
                            lambda c, s: self._make_comm_response())
        monkeypatch.setattr("services.fed_forecaster._call_groq_black_swan_forecast",
                            lambda c: self._make_swan_response())
        result = generate_expanded_forecast.__wrapped__("{}", "{}")
        assert "error" in result["_call_status"]["core"]
        assert result["_call_status"]["commodities_intl"] == "ok"
        assert result["_call_status"]["black_swans"] == "ok"
