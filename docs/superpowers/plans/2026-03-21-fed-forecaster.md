# Fed Policy Forecasting Machine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Fed Forecaster" tab to the Risk Regime module that shows probability-weighted Fed policy scenarios with full causal asset-class chains, derived from live ZQ futures data and Fed RSS tone scoring.

**Architecture:** New `services/fed_forecaster.py` handles all data fetching (ZQ futures via yfinance, Fed RSS feeds, Groq tone scoring and forecast generation). `modules/risk_regime.py` gains a tab system wrapping existing content in Tab 1 and adding a new `_render_fed_forecaster()` function in Tab 2. FRED `FEDFUNDS` series is added at three locations in risk_regime.py.

**Tech Stack:** Python, Streamlit, Plotly, yfinance, requests (stdlib xml.etree), Groq API (requests.post), pytest, unittest.mock

---

## Chunk 1: Pure data functions in `services/fed_forecaster.py`

Covers: FOMC calendar constant, ZQ probability derivation, RSS parsing, `adjust_probabilities`, `build_fed_context`. All pure or easily mockable — test-first.

### Task 1: FOMC calendar constant + `fetch_zq_probabilities`

**Files:**
- Create: `services/fed_forecaster.py`
- Create: `tests/test_fed_forecaster.py`

- [ ] **Step 1: Write failing tests for `fetch_zq_probabilities`**

Create `tests/test_fed_forecaster.py`:

```python
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


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
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd "C:/Users/16476/claude projects/narrative-investing-tool"
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py -v 2>&1 | head -40
```

Expected: `ModuleNotFoundError: No module named 'services.fed_forecaster'`

- [ ] **Step 3: Create `services/fed_forecaster.py` with FOMC calendar, `_derive_probabilities_from_implied_rate`, and `_equal_weight_fallback`**

```python
"""
Fed Policy Forecasting Machine — data layer.

Functions:
  fetch_zq_probabilities()         — ZQ futures → 4-scenario probabilities
  fetch_fed_communications()       — Fed RSS feeds → tone items
  score_fed_tone()                 — Groq tone scoring of Fed comms
  adjust_probabilities()           — Apply tone adjustment to base probs
  build_fed_context()              — Package regime signals for Groq prompt
  generate_forecast()              — Groq causal chain + asset matrix
  get_next_fomc()                  — Days to next FOMC meeting
"""

import json
import os
import hashlib
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from services.market_data import fetch_fred_series_safe


# ─────────────────────────────────────────────────────────────────────────────
# FOMC CALENDAR  (update each January from federalreserve.gov/monetarypolicy/fomccalendars.htm)
# ─────────────────────────────────────────────────────────────────────────────

_FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 10),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def get_next_fomc() -> dict:
    """Return the next upcoming FOMC meeting date and days away."""
    today = date.today()
    future = [d for d in _FOMC_DATES_2026 if d >= today]
    if not future:
        # Off-calendar year — return last known date
        last = _FOMC_DATES_2026[-1]
        return {"date": last.strftime("%b %d, %Y"), "days_away": 0}
    nxt = future[0]
    return {
        "date": nxt.strftime("%b %d, %Y"),
        "days_away": (nxt - today).days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL SCENARIO KEYS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_KEYS = ["hold", "cut_25", "cut_50", "hike_25"]
SCENARIO_LABELS = {
    "hold":    "Fed Holds",
    "cut_25":  "Cut 25bp",
    "cut_50":  "Cut 50bp",
    "hike_25": "Hike 25bp",
}
# Rate change implied by each scenario (in percentage points)
_SCENARIO_DELTAS = {
    "cut_50":  -0.50,
    "cut_25":  -0.25,
    "hold":     0.00,
    "hike_25": +0.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# ZQ PROBABILITY DERIVATION (pure, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_probabilities_from_implied_rate(implied_rate: float, current_rate: float) -> list[dict]:
    """
    Distribute probability across 4 scenarios using a normal distribution
    centred on (implied_rate - current_rate) with σ=0.15.

    Returns list of {scenario, prob, implied_rate, source} dicts summing to 1.0.
    """
    delta = implied_rate - current_rate
    sigma = 0.15

    scenario_deltas = [_SCENARIO_DELTAS[k] for k in SCENARIO_KEYS]
    # Gaussian probability mass at each scenario's rate delta
    raw = np.array([
        np.exp(-0.5 * ((delta - sd) / sigma) ** 2)
        for sd in scenario_deltas
    ])
    probs = raw / raw.sum()

    return [
        {
            "scenario": key,
            "prob": float(probs[i]),
            "implied_rate": implied_rate,
            "source": "yfinance",
        }
        for i, key in enumerate(SCENARIO_KEYS)
    ]


def _equal_weight_fallback() -> list[dict]:
    """Return equal 25% probability for all 4 scenarios."""
    return [
        {
            "scenario": key,
            "prob": 0.25,
            "implied_rate": None,
            "source": "fallback",
            "data_unavailable": True,
        }
        for key in SCENARIO_KEYS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FETCH ZQ PROBABILITIES  (cached, tiered)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=14400)
def fetch_zq_probabilities() -> list[dict]:
    """
    Derive 4-scenario Fed policy probabilities from Fed Funds Futures.

    Tier 1: yfinance ZQ=F (front-month generic)
    Tier 2: yfinance named contracts ZQH26, ZQK26, ZQM26
    Tier 3: equal-weight fallback (data_unavailable=True)
    """
    fedfunds_series = fetch_fred_series_safe("FEDFUNDS")
    if fedfunds_series is None or fedfunds_series.empty:
        return _equal_weight_fallback()
    current_rate = float(fedfunds_series.dropna().iloc[-1])

    # Tier 1 — generic front-month
    try:
        df = yf.download("ZQ=F", period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        if df is not None and not df.empty and "Close" in df.columns:
            price = float(df["Close"].dropna().iloc[-1])
            implied = 100.0 - price
            result = _derive_probabilities_from_implied_rate(implied, current_rate)
            result[0]["source"] = "yfinance"
            return result
    except Exception:
        pass

    # Tier 2 — named contracts
    for ticker in ("ZQH26", "ZQK26", "ZQM26"):
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            if df is not None and not df.empty and "Close" in df.columns:
                price = float(df["Close"].dropna().iloc[-1])
                implied = 100.0 - price
                result = _derive_probabilities_from_implied_rate(implied, current_rate)
                for r in result:
                    r["source"] = "yfinance"
                return result
        except Exception:
            continue

    # Tier 3 — fallback
    return _equal_weight_fallback()
```

- [ ] **Step 4: Run tests — should pass**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestFetchZqProbabilities tests/test_fed_forecaster.py::TestFomcCalendar -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/fed_forecaster.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): ZQ probability engine + FOMC calendar"
```

---

### Task 2: `fetch_fed_communications`

**Files:**
- Modify: `services/fed_forecaster.py`
- Modify: `tests/test_fed_forecaster.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fed_forecaster.py`:

```python
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
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_RSS  # has 2 items
        mock_resp.raise_for_status = MagicMock()
        with patch("services.fed_forecaster.requests.get", return_value=mock_resp):
            # Both feeds return the same 2-item RSS → 4 items total before truncation
            items = fetch_fed_communications(max_items=1)
        assert len(items) == 1  # strictly enforced — would fail if truncation removed
```

- [ ] **Step 2: Run to verify failure**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestFetchFedCommunications -v
```

Expected: `ImportError: cannot import name '_parse_rss_feed'`

- [ ] **Step 3: Add `_parse_rss_feed` and `fetch_fed_communications` to `services/fed_forecaster.py`**

```python
# ─────────────────────────────────────────────────────────────────────────────
# FED RSS COMMUNICATIONS
# ─────────────────────────────────────────────────────────────────────────────

_FED_RSS_FEEDS = {
    "release": "https://www.federalreserve.gov/rss/releases.xml",
    "speech":  "https://www.federalreserve.gov/rss/speeches.xml",
}


def _parse_rss_feed(xml_text: str, source: str) -> list[dict]:
    """Parse Federal Reserve RSS XML text into a list of communication dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()

        if not title:
            continue

        # Parse date to sortable datetime
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub_date)
            date_str = dt.strftime("%Y-%m-%d")
            sort_key = dt.timestamp()
        except Exception:
            date_str = pub_date
            sort_key = 0.0

        items.append({
            "title": title,
            "date": date_str,
            "url": link,
            "source": source,
            "raw_text": description,
            "_sort_key": sort_key,
        })

    # Most recent first
    items.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in items:
        item.pop("_sort_key", None)
    return items


@st.cache_data(ttl=3600)
def fetch_fed_communications(max_items: int = 5) -> list[dict]:
    """
    Fetch and merge Fed press releases and speeches from official RSS feeds.
    Returns up to max_items most recent items, sorted by date descending.
    Falls back to [] on any error.
    """
    all_items = []
    for source, url in _FED_RSS_FEEDS.items():
        try:
            resp = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "NarrativeInvestingTool/1.0"},
            )
            resp.raise_for_status()
            all_items.extend(_parse_rss_feed(resp.text, source=source))
        except Exception:
            continue

    # Sort merged list by date descending, return top max_items
    all_items.sort(key=lambda x: x["date"], reverse=True)
    return all_items[:max_items]
```

- [ ] **Step 4: Run tests — should pass**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestFetchFedCommunications -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/fed_forecaster.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): Fed RSS feed parser"
```

---

### Task 3: `adjust_probabilities` and `build_fed_context`

**Files:**
- Modify: `services/fed_forecaster.py`
- Modify: `tests/test_fed_forecaster.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fed_forecaster.py`:

```python
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
        result = adjust_probabilities(self._base_probs(), self._tone_result())
        before = 0.52
        after = next(r["prob"] for r in result if r["scenario"] == "hold")
        assert after > before

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

    def test_missing_fedfunds_does_not_raise(self):
        from services.fed_forecaster import build_fed_context
        fred_data = self._make_fred_data()
        fred_data["fedfunds"] = None
        ctx = build_fed_context(self._make_macro(), fred_data)
        # Should not raise; fed_funds_rate may be None or a fallback float
        assert "fed_funds_rate" in ctx
```

- [ ] **Step 2: Run to verify failure**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestAdjustProbabilities tests/test_fed_forecaster.py::TestBuildFedContext -v
```

Expected: `ImportError: cannot import name 'adjust_probabilities'`

- [ ] **Step 3: Add `adjust_probabilities` and `build_fed_context` to `services/fed_forecaster.py`**

```python
# ─────────────────────────────────────────────────────────────────────────────
# ADJUST PROBABILITIES (pure)
# ─────────────────────────────────────────────────────────────────────────────

def adjust_probabilities(base_probs: list[dict], tone_result: dict) -> list[dict]:
    """
    Apply tone-derived probability adjustments to base ZQ probabilities.
    Clamps to [0, 1] then re-normalises to sum to 1.0.
    Adds a signed `delta` field to each item.
    """
    adjustments = tone_result.get("prob_adjustments", {})

    adjusted = []
    for item in base_probs:
        key = item["scenario"]
        raw_adj = adjustments.get(key, 0.0)
        new_prob = max(0.0, min(1.0, item["prob"] + raw_adj))
        adjusted.append({**item, "prob": new_prob, "delta": raw_adj})

    # Re-normalise
    total = sum(r["prob"] for r in adjusted)
    if total > 0:
        for r in adjusted:
            r["prob"] = r["prob"] / total

    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FED CONTEXT (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_last(series) -> float | None:
    """Extract last non-null float from a pandas Series, or None."""
    if series is None:
        return None
    try:
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])
    except Exception:
        return None


def build_fed_context(macro: dict, fred_data: dict) -> dict:
    """
    Package current macro regime signals into a serialisable dict for the Groq prompt.
    Does not make network calls. Falls back gracefully when series are None.
    """
    fedfunds_series = fred_data.get("fedfunds")
    fed_rate = _safe_last(fedfunds_series)
    if fed_rate is None:
        # Last-resort: try disk cache directly
        fallback = fetch_fred_series_safe("FEDFUNDS")
        fed_rate = _safe_last(fallback)

    return {
        "fed_funds_rate":  fed_rate,
        "core_pce":        _safe_last(fred_data.get("core_pce")),
        "unemployment":    _safe_last(fred_data.get("unrate")),
        "yield_curve":     _safe_last(fred_data.get("yield_curve")),
        "credit_spread":   _safe_last(fred_data.get("credit_spread")),
        "quadrant":        macro.get("quadrant", "Unknown"),
        "macro_score":     macro.get("macro_score", 50),
        "regime":          macro.get("macro_regime", "Unknown"),
    }
```

- [ ] **Step 4: Run tests — should pass**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestAdjustProbabilities tests/test_fed_forecaster.py::TestBuildFedContext -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```
venv/Scripts/python.exe -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add services/fed_forecaster.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): adjust_probabilities + build_fed_context"
```

---

## Chunk 2: Groq AI functions in `services/fed_forecaster.py`

Covers: `score_fed_tone` and `generate_forecast` — both make Groq API calls, tested with mocks.

### Task 4: `score_fed_tone`

**Files:**
- Modify: `services/fed_forecaster.py`
- Modify: `tests/test_fed_forecaster.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fed_forecaster.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestScoreFedTone -v
```

Expected: `ImportError: cannot import name '_call_groq_tone'`

- [ ] **Step 3: Add `_call_groq_tone`, `_neutral_tone_fallback`, and `score_fed_tone` to `services/fed_forecaster.py`**

```python
# ─────────────────────────────────────────────────────────────────────────────
# GROQ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"


def _groq_headers() -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from Groq response if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _neutral_tone_fallback() -> dict:
    return {
        "items": [],
        "aggregate_bias": "neutral",
        "prob_adjustments": {k: 0.0 for k in SCENARIO_KEYS},
    }


# ─────────────────────────────────────────────────────────────────────────────
# FED TONE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_tone(communications: list[dict]) -> dict:
    """
    Internal: call Groq to score Fed communication tone.
    Not cached — called by the cached score_fed_tone wrapper.
    """
    if not communications:
        return _neutral_tone_fallback()

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _neutral_tone_fallback()

    items_text = "\n\n".join(
        f"[{i+1}] {c['title']} ({c['date']})\n{c['raw_text']}"
        for i, c in enumerate(communications)
    )

    prompt = f"""You are a Federal Reserve communication analyst. Score the following Fed statements for monetary policy tone.

Statements:
{items_text}

Return ONLY valid JSON (no markdown fences) matching this schema:
{{
  "items": [
    {{
      "title": "<title>",
      "hawkish_prob": <0.0-1.0>,
      "neutral_prob": <0.0-1.0>,
      "dovish_prob": <0.0-1.0>,
      "adjustment_confidence": <0.0-1.0>
    }}
  ],
  "aggregate_bias": "hawkish" | "neutral" | "dovish",
  "prob_adjustments": {{
    "hold": <-0.15 to +0.15>,
    "cut_25": <-0.15 to +0.15>,
    "cut_50": <-0.15 to +0.15>,
    "hike_25": <-0.15 to +0.15>
  }}
}}

Rules:
- hawkish_prob + neutral_prob + dovish_prob must sum to 1.0 per item
- prob_adjustments must sum to 0 (they are redistributions)
- Hawkish = signals higher rates or holding; dovish = signals cuts
- adjustment_confidence is how confident you are in the adjustment (0=none, 1=certain)"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.2,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = _strip_fences(resp.json()["choices"][0]["message"]["content"])
        return json.loads(text)
    except Exception:
        return _neutral_tone_fallback()


@st.cache_data(ttl=3600)
def score_fed_tone(comm_key: str, _communications: list[dict]) -> dict:
    """
    Score Fed communications tone via Groq.
    comm_key: stable hash of [(title, date)] — used as cache discriminator.
    _communications: leading underscore = Streamlit skips hashing this arg.
    """
    return _call_groq_tone(_communications)
```

- [ ] **Step 4: Run tests — should pass**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestScoreFedTone -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/fed_forecaster.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): Groq tone scoring for Fed communications"
```

---

### Task 5: `generate_forecast`

**Files:**
- Modify: `services/fed_forecaster.py`
- Modify: `tests/test_fed_forecaster.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fed_forecaster.py`:

```python
# ── generate_forecast ─────────────────────────────────────────────────────────

import json as _json

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
            "choices": [{"message": {"content": _json.dumps(response_dict)}}]
        }
        return mock_resp

    def _context_json(self):
        return _json.dumps({"fed_funds_rate": 5.33, "quadrant": "Stagflation",
                            "macro_score": 28, "regime": "Risk-Off"})

    def _scenarios_json(self):
        return _json.dumps([{"scenario": "hold", "prob": 0.52},
                            {"scenario": "cut_25", "prob": 0.38},
                            {"scenario": "cut_50", "prob": 0.07},
                            {"scenario": "hike_25", "prob": 0.03}])

    def test_returns_parsed_forecast_dict(self):
        from services.fed_forecaster import _call_groq_forecast
        with patch("services.fed_forecaster.requests.post") as mock_post:
            mock_post.return_value = self._mock_groq(_MINIMAL_FORECAST)
            result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        assert result is not None
        assert "near_term" in result
        assert "medium_term" in result
        assert "causal_chains" in result

    def test_near_term_has_all_four_scenarios(self):
        from services.fed_forecaster import _call_groq_forecast
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
        with patch("services.fed_forecaster.requests.post") as mock_post:
            mock_post.return_value = self._mock_groq(_MINIMAL_FORECAST)
            result = _call_groq_forecast(self._context_json(), self._scenarios_json())
        equities_hold = result["medium_term"]["hold"]["equities"]
        assert len(equities_hold["monthly_p50"]) == 12
```

- [ ] **Step 2: Run to verify failure**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestGenerateForecast -v
```

Expected: `ImportError: cannot import name '_call_groq_forecast'`

- [ ] **Step 3: Add `_call_groq_forecast` and `generate_forecast` to `services/fed_forecaster.py`**

```python
# ─────────────────────────────────────────────────────────────────────────────
# FORECAST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_forecast(context_json: str, scenarios_json: str) -> dict | None:
    """
    Internal: single Groq call covering all 4 scenarios, both time horizons.
    Not cached — called by the cached generate_forecast wrapper.
    Returns parsed dict or None on failure.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    context = json.loads(context_json)
    scenarios = json.loads(scenarios_json)

    scenarios_text = "\n".join(
        f"- {SCENARIO_LABELS[s['scenario']]}: {s['prob']*100:.0f}%"
        for s in scenarios
    )

    prompt = f"""You are a senior macro strategist. Given the current economic regime and Fed policy scenarios below, provide a probability-weighted forecast for 4 asset classes.

CURRENT REGIME:
- Fed Funds Rate: {context.get('fed_funds_rate', 'N/A')}%
- Core PCE Inflation: {context.get('core_pce', 'N/A')}%
- Unemployment: {context.get('unemployment', 'N/A')}%
- Yield Curve (10Y-2Y): {context.get('yield_curve', 'N/A')}%
- Credit Spread (HY): {context.get('credit_spread', 'N/A')}%
- Dalio Quadrant: {context.get('quadrant', 'N/A')}
- Macro Score: {context.get('macro_score', 'N/A')}/100 ({context.get('regime', 'N/A')})

FED POLICY SCENARIOS (market-implied probabilities):
{scenarios_text}

Return ONLY valid JSON (no markdown fences) with this EXACT structure for all 4 scenarios (hold, cut_25, cut_50, hike_25):

{{
  "near_term": {{
    "<scenario_key>": {{
      "equities":    {{"direction": "up|down|flat", "magnitude_low": <float>, "magnitude_high": <float>, "direction_prob": <0-1>, "magnitude_confidence": <0-1>, "chain": [{{"step": "<text>", "confidence": <0-1>}}]}},
      "bonds":       {{...same structure...}},
      "commodities": {{...same structure...}},
      "usd":         {{...same structure...}}
    }}
  }},
  "medium_term": {{
    "<scenario_key>": {{
      "equities":    {{"monthly_p25": [<12 floats>], "monthly_p50": [<12 floats>], "monthly_p75": [<12 floats>], "narrative": "<1-2 sentences>"}},
      "bonds":       {{...same structure...}},
      "commodities": {{...same structure...}},
      "usd":         {{...same structure...}}
    }}
  }},
  "causal_chains": {{
    "<scenario_key>": [{{"step": "<text>", "confidence": <0-1 cumulative decay>}}]
  }}
}}

Rules:
- magnitude values are percentage returns (e.g. -8.0 means -8%)
- chain confidence values are CUMULATIVE (each hop lower than the previous)
- monthly arrays have EXACTLY 12 values (months 1-12 from today), cumulative % returns
- p25 < p50 < p75 for each month
- Use scenario keys: hold, cut_25, cut_50, hike_25"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
            timeout=45,
        )
        resp.raise_for_status()
        text = _strip_fences(resp.json()["choices"][0]["message"]["content"])
        return json.loads(text)
    except Exception:
        return None


@st.cache_data(ttl=14400)
def generate_forecast(context_json: str, scenarios_json: str) -> dict | None:
    """
    Generate probability-weighted Fed policy forecast via Groq.
    Both args are JSON strings (hashable by st.cache_data).
    Returns parsed forecast dict or None on failure.
    """
    return _call_groq_forecast(context_json, scenarios_json)
```

- [ ] **Step 4: Run tests — should pass**

```
venv/Scripts/python.exe -m pytest tests/test_fed_forecaster.py::TestGenerateForecast -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Run full test suite**

```
venv/Scripts/python.exe -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add services/fed_forecaster.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): Groq forecast generation (causal chain + asset matrix)"
```

---

## Chunk 3: Tab system + UI Sections 1–3 in `modules/risk_regime.py`

Covers: FRED additions, tab wrapping, `_render_fed_forecaster` skeleton, Section 1 (FOMC strip), Section 2 (comms tracker), Section 3 (probability bars).

### Task 6: Add FEDFUNDS to FRED series + wrap render in tabs

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Add `"FEDFUNDS"` to `_FRED_SERIES_IDS`**

In `modules/risk_regime.py` around line 1243, add `"FEDFUNDS"` to the list:

```python
_FRED_SERIES_IDS = [
    "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
    "PCEPILFE", "PNFI", "THREEFYTP10",
    "INDPRO", "NFCI", "DGS10", "ICSA", "USSLIND",
    "UMCSENT", "PERMIT", "FEDFUNDS",
]
```

- [ ] **Step 2: Add `"fedfunds": "FEDFUNDS"` to `fred_ids` dict in `render()`**

Around line 1277, add the entry:

```python
fred_ids = {
    "yield_curve": "T10Y2Y",
    "credit_spread": "BAMLH0A0HYM2",
    "m2": "M2SL",
    "sahm": "SAHMREALTIME",
    "unrate": "UNRATE",
    "core_pce": "PCEPILFE",
    "capex": "PNFI",
    "icsa": "ICSA",
    "lei": "USSLIND",
    "term_premium": "THREEFYTP10",
    "ism": "INDPRO",
    "fci": "NFCI",
    "dgs10": "DGS10",
    "umcsent": "UMCSENT",
    "permit": "PERMIT",
    "fedfunds": "FEDFUNDS",   # ← add
}
```

- [ ] **Step 3: Wrap the existing render body in `tab1` and add empty `tab2`**

Locate the comment `# ── Ticker Bar ──` in `render()` (around line 1315). Insert the tab creation immediately before that comment, then indent everything from the ticker bar through the end of the function under `with tab1:`:

```python
    tab1, tab2 = st.tabs(["📊 Macro Dashboard", "🏦 Fed Forecaster"])

    with tab1:
        # ── Ticker Bar ──
        # ... [all existing content from here to end of render stays indented here]

    with tab2:
        _render_fed_forecaster(macro, fred_data)
```

The insertion point is the `# ── Ticker Bar ──` comment line. Everything before it (status block, load timing metrics) stays outside the tabs. Everything from the ticker bar onward moves into `with tab1:`.

Add at the bottom of the file (before the final line if any):

```python
def _render_fed_forecaster(macro: dict, fred_data: dict):
    """Render Tab 2: Fed Forecaster — placeholder stub."""
    st.info("Fed Forecaster loading...")
```

- [ ] **Step 4: Verify the app renders without error**

```
venv/Scripts/python.exe -c "import ast, sys; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 5: Run existing tests**

```
venv/Scripts/python.exe -m pytest tests/test_risk_regime_labels.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(risk-regime): add FEDFUNDS to FRED series, wrap render in tabs"
```

---

### Task 7: Section 1 — FOMC context strip + Section 2 — Communications tracker

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Replace the `_render_fed_forecaster` stub with Sections 1 and 2**

Replace the stub function with:

```python
def _render_fed_forecaster(macro: dict, fred_data: dict):
    """Render Tab 2: Fed Forecaster."""
    from services.fed_forecaster import (
        fetch_zq_probabilities, fetch_fed_communications, score_fed_tone,
        adjust_probabilities, build_fed_context, generate_forecast,
        get_next_fomc, SCENARIO_KEYS, SCENARIO_LABELS,
    )
    import hashlib, json as _json
    from datetime import datetime as _dt

    # ── Section 1: FOMC Context Strip ────────────────────────────────────────
    fomc = get_next_fomc()
    fedfunds_series = fred_data.get("fedfunds")
    current_rate_str = "N/A"
    if fedfunds_series is not None and not fedfunds_series.empty:
        current_rate_str = f"{fedfunds_series.dropna().iloc[-1]:.2f}%"

    regime_label = macro.get("macro_regime", "Unknown")
    quadrant = macro.get("quadrant", "")
    regime_color = COLORS["red"] if "Risk-Off" in regime_label else (
        COLORS["green"] if "Risk-On" in regime_label else COLORS["yellow"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🗓 Next FOMC", fomc["date"], f"{fomc['days_away']} days away")
    c2.metric("🏦 Fed Funds Rate", current_rate_str)
    c3.markdown(
        f'<div style="padding:8px 0;">'
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};font-family:\'JetBrains Mono\',monospace;'
        f'text-transform:uppercase;letter-spacing:0.06em;">Regime</div>'
        f'<div style="font-size:18px;font-weight:700;color:{regime_color};">'
        f'{regime_label} · {quadrant}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 2: Fed Communications Tracker ────────────────────────────────
    _section_header("Fed Communications")

    comms = fetch_fed_communications(max_items=5)
    comms_updated = _dt.now().strftime("%H:%M")

    if not comms:
        st.markdown(
            f'<div style="color:{COLORS["text_dim"]};font-size:13px;">'
            f'Fed communications unavailable — tone adjustment skipped</div>',
            unsafe_allow_html=True,
        )
        tone_result = {"aggregate_bias": "neutral",
                       "prob_adjustments": {k: 0.0 for k in SCENARIO_KEYS}}
    else:
        comm_key = hashlib.md5(
            str([(c["title"], c["date"]) for c in comms]).encode()
        ).hexdigest()
        tone_result = score_fed_tone(comm_key, comms)

        tone_colors = {
            "hawkish": COLORS["red"],
            "neutral": COLORS["yellow"],
            "dovish":  COLORS["green"],
        }

        for item_data, scored in zip(comms, tone_result.get("items", [])):
            h = scored.get("hawkish_prob", 0.0)
            n = scored.get("neutral_prob", 1.0)
            d = scored.get("dovish_prob", 0.0)
            if h >= n and h >= d:
                tone_label, tone_emoji = "Hawkish", "🔴"
                tone_pct = int(round(h * 100))
            elif d >= h and d >= n:
                tone_label, tone_emoji = "Dovish", "🟢"
                tone_pct = int(round(d * 100))
            else:
                tone_label, tone_emoji = "Neutral", "🟡"
                tone_pct = int(round(n * 100))

            adj_conf = scored.get("adjustment_confidence", 0.0)
            adj_html = ""
            prob_adj = tone_result.get("prob_adjustments", {})
            dominant_scenario = max(SCENARIO_KEYS, key=lambda k: abs(prob_adj.get(k, 0)))
            dominant_adj = prob_adj.get(dominant_scenario, 0.0)
            if abs(dominant_adj) > 0.01:
                sign = "+" if dominant_adj > 0 else ""
                pp = int(round(abs(dominant_adj) * 100))
                adj_html = (
                    f'<span style="color:{COLORS["text_dim"]};font-size:11px;"> '
                    f'Δ {SCENARIO_LABELS[dominant_scenario]} {sign}{pp}pp '
                    f'[{int(round(adj_conf*100))}% conf]</span>'
                )

            title_short = item_data["title"][:80] + ("…" if len(item_data["title"]) > 80 else "")
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid {COLORS["border"]};">'
                f'{tone_emoji} <b style="color:{tone_colors.get(tone_label.lower(), COLORS["text"])};">'
                f'{tone_label} [{tone_pct}%]</b> '
                f'<span style="color:{COLORS["text_dim"]};font-size:12px;">{item_data["source"].upper()} · {item_data["date"]}</span>'
                f'<br><span style="font-size:13px;">{title_short}</span>{adj_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption(f"Comms as of {comms_updated}")
    st.markdown("---")

    # ── Sections 3-6 placeholder ─────────────────────────────────────────────
    _render_fed_probability_bars(macro, fred_data, tone_result)
```

Add the probability bars stub:

```python
def _render_fed_probability_bars(macro: dict, fred_data: dict, tone_result: dict):
    """Sections 3-6: placeholder stub."""
    st.info("Probability bars — coming in next step")
```

- [ ] **Step 2: Verify syntax**

```
venv/Scripts/python.exe -c "import ast; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(fed-forecaster): FOMC context strip + Fed comms tracker (Sections 1-2)"
```

---

### Task 8: Section 3 — Scenario probability bars

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Replace `_render_fed_probability_bars` stub with real implementation**

```python
def _render_fed_probability_bars(macro: dict, fred_data: dict, tone_result: dict):
    """Sections 3–6: probability bars, asset matrix, causal chain, fan charts."""
    from services.fed_forecaster import (
        fetch_zq_probabilities, build_fed_context, generate_forecast,
        adjust_probabilities, SCENARIO_KEYS, SCENARIO_LABELS,
    )
    import json as _json
    from datetime import datetime as _dt

    # ── Section 3: Scenario Probability Bars ─────────────────────────────────
    _section_header("Scenario Probabilities (Fed Funds Futures)")

    base_probs = fetch_zq_probabilities()
    futures_updated = _dt.now().strftime("%H:%M")

    # Show fallback warning if data unavailable
    if any(r.get("data_unavailable") for r in base_probs):
        st.warning("⚠ Futures data unavailable — showing equal-weight 25% per scenario")

    # Apply tone adjustment
    adj_probs = adjust_probabilities(base_probs, tone_result)

    # Source label
    source = (base_probs[0].get("source", "fallback") if base_probs else "fallback")
    source_label = {"yfinance": "Futures: yfinance ZQ", "fallback": "⚠ Fallback: equal-weight"}.get(source, source)
    st.caption(f"{source_label}  |  Futures as of {futures_updated}")

    # Horizontal bar chart
    scenario_colors = {
        "hold":    COLORS.get("yellow", "#f0c040"),
        "cut_25":  COLORS.get("green",  "#40c080"),
        "cut_50":  COLORS.get("green",  "#40c080"),
        "hike_25": COLORS.get("red",    "#e05050"),
    }

    import plotly.graph_objects as go

    labels = [SCENARIO_LABELS[k] for k in SCENARIO_KEYS]
    probs  = [next((r["prob"] for r in adj_probs if r["scenario"] == k), 0.25) for k in SCENARIO_KEYS]
    deltas = [next((r.get("delta", 0.0) for r in adj_probs if r["scenario"] == k), 0.0) for k in SCENARIO_KEYS]
    colors = [scenario_colors[k] for k in SCENARIO_KEYS]

    # Build text labels with delta badges
    text_labels = []
    for p, d in zip(probs, deltas):
        pct = int(round(p * 100))
        if abs(d) > 0.005:
            sign = "▲" if d > 0 else "▼"
            pp = int(round(abs(d) * 100))
            text_labels.append(f"{pct}%  {sign}{pp}pp")
        else:
            text_labels.append(f"{pct}%")

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        text=text_labels,
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
    ))
    apply_dark_layout(fig)
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Refresh Forecaster", key="refresh_forecaster"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Sections 4-6 placeholder ─────────────────────────────────────────────
    _render_fed_asset_matrix(macro, fred_data, adj_probs)
```

Add stub:

```python
def _render_fed_asset_matrix(macro: dict, fred_data: dict, adj_probs: list[dict]):
    """Sections 4-6 stub."""
    st.info("Asset matrix — coming in next step")
```

- [ ] **Step 2: Verify syntax**

```
venv/Scripts/python.exe -c "import ast; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(fed-forecaster): scenario probability bars (Section 3)"
```

---

## Chunk 4: UI Sections 4–6 in `modules/risk_regime.py`

Covers: near-term asset impact matrix, full causal chain, medium-term fan charts.

### Task 9: Section 4 — Near-term asset impact matrix

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Replace `_render_fed_asset_matrix` stub**

```python
def _render_fed_asset_matrix(macro: dict, fred_data: dict, adj_probs: list[dict]):
    """Sections 4-6: asset matrix, causal chain, fan charts."""
    from services.fed_forecaster import (
        build_fed_context, generate_forecast, SCENARIO_KEYS, SCENARIO_LABELS,
    )
    import json as _json

    # Build and call forecast
    context = build_fed_context(macro, fred_data)
    context_json   = _json.dumps(context)
    scenarios_json = _json.dumps(adj_probs)
    forecast = generate_forecast(context_json, scenarios_json)

    if forecast is None:
        st.error("Fed forecast unavailable — Groq API error or key not set.")
        return

    near = forecast.get("near_term", {})
    medium = forecast.get("medium_term", {})
    chains = forecast.get("causal_chains", {})

    # ── Section 4: Near-Term Asset Impact Matrix ──────────────────────────────
    _section_header("Near-Term Asset Impact (0–3 months)")

    ASSET_KEYS   = ["equities", "bonds", "commodities", "usd"]
    ASSET_LABELS = {"equities": "Equities", "bonds": "Bonds",
                    "commodities": "Commodities", "usd": "USD"}

    DIR_COLORS = {
        "up":   COLORS.get("green", "#40c080"),
        "down": COLORS.get("red",   "#e05050"),
        "flat": COLORS.get("text_dim", "#888"),
    }
    DIR_ARROWS = {"up": "▲ UP", "down": "▼ DOWN", "flat": "— FLAT"}

    # Header row
    header_cols = st.columns([2, 1, 1, 1, 1])
    header_cols[0].markdown(
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};'
        f'font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;'
        f'letter-spacing:0.06em;">Asset</div>', unsafe_allow_html=True
    )
    for i, key in enumerate(SCENARIO_KEYS):
        prob = next((r["prob"] for r in adj_probs if r["scenario"] == key), 0.25)
        header_cols[i+1].markdown(
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
            f'font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;'
            f'letter-spacing:0.06em;">{SCENARIO_LABELS[key]}<br>'
            f'<span style="color:{COLORS["text"]};">{int(round(prob*100))}%</span></div>',
            unsafe_allow_html=True
        )

    # Data rows
    for asset in ASSET_KEYS:
        row_cols = st.columns([2, 1, 1, 1, 1])
        row_cols[0].markdown(
            f'<div style="font-size:13px;font-weight:600;padding:8px 0;">'
            f'{ASSET_LABELS[asset]}</div>', unsafe_allow_html=True
        )
        for i, scenario_key in enumerate(SCENARIO_KEYS):
            cell = (near.get(scenario_key) or {}).get(asset, {})
            if not cell:
                row_cols[i+1].markdown("—")
                continue
            direction  = cell.get("direction", "flat")
            dir_prob   = int(round(cell.get("direction_prob", 0.5) * 100))
            mag_low    = cell.get("magnitude_low", 0.0)
            mag_high   = cell.get("magnitude_high", 0.0)
            mag_conf   = int(round(cell.get("magnitude_confidence", 0.5) * 100))
            color = DIR_COLORS.get(direction, COLORS["text_dim"])
            arrow = DIR_ARROWS.get(direction, "— FLAT")
            row_cols[i+1].markdown(
                f'<div style="background:{COLORS["surface"]};border-radius:4px;'
                f'padding:6px 8px;margin:2px;">'
                f'<div style="font-size:13px;font-weight:700;color:{color};">'
                f'{arrow} [{dir_prob}%]</div>'
                f'<div style="font-size:11px;color:{COLORS["text_dim"]};">'
                f'{mag_low:+.0f}% to {mag_high:+.0f}%</div>'
                f'<div style="font-size:10px;color:{COLORS["text_dim"]};">'
                f'[{mag_conf}% CI]</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    _render_fed_causal_chain(chains, adj_probs, medium)


def _render_fed_causal_chain(chains: dict, adj_probs: list[dict], medium: dict):
    """Section 5 stub."""
    st.info("Causal chain — coming in next step")
```

- [ ] **Step 2: Verify syntax**

```
venv/Scripts/python.exe -c "import ast; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(fed-forecaster): near-term asset impact matrix (Section 4)"
```

---

### Task 10: Section 5 — Full causal chain

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Replace `_render_fed_causal_chain` stub**

```python
def _render_fed_causal_chain(chains: dict, adj_probs: list[dict], medium: dict):
    """Section 5: full causal chain with cumulative confidence decay."""
    from services.fed_forecaster import SCENARIO_KEYS, SCENARIO_LABELS

    _section_header("Causal Chain")

    # Dominant scenario = highest adjusted probability
    dominant_key = max(SCENARIO_KEYS, key=lambda k: next(
        (r["prob"] for r in adj_probs if r["scenario"] == k), 0.0
    ))

    for scenario_key in SCENARIO_KEYS:
        chain_steps = chains.get(scenario_key, [])
        prob = next((r["prob"] for r in adj_probs if r["scenario"] == scenario_key), 0.25)
        label = f"{SCENARIO_LABELS[scenario_key]} [{int(round(prob*100))}%]"
        is_dominant = scenario_key == dominant_key

        with st.expander(label, expanded=is_dominant):
            if not chain_steps:
                st.caption("Chain data unavailable.")
                continue

            start_conf = chain_steps[0]["confidence"]
            end_conf   = chain_steps[-1]["confidence"]

            # Render chain steps with indented confidence
            for step_data in chain_steps:
                step = step_data["step"]
                conf = step_data["confidence"]
                conf_pct = int(round(conf * 100))
                color = (
                    COLORS.get("green",    "#40c080") if conf >= 0.70 else
                    COLORS.get("yellow",   "#f0c040") if conf >= 0.55 else
                    COLORS.get("text_dim", "#888888")
                )
                st.markdown(
                    f'<div style="padding:3px 0 3px 16px;border-left:2px solid {color};">'
                    f'<span style="font-size:13px;">{step}</span>'
                    f'<span style="float:right;font-size:11px;color:{color};">'
                    f'[{conf_pct}%]</span></div>',
                    unsafe_allow_html=True,
                )

            # Confidence decay bar
            if start_conf > 0:
                decay_pct = int(round((end_conf / start_conf) * 100))
                filled = int(round(decay_pct / 10))
                bar = "█" * filled + "░" * (10 - filled)
                st.markdown(
                    f'<div style="font-size:11px;color:{COLORS["text_dim"]};'
                    f'font-family:\'JetBrains Mono\',monospace;margin-top:8px;">'
                    f'Confidence: {bar} {int(round(start_conf*100))}% → '
                    f'{int(round(end_conf*100))}%</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    _render_fed_fan_charts(medium, adj_probs)
```

Add stub:

```python
def _render_fed_fan_charts(medium: dict, adj_probs: list[dict]):
    """Section 6 stub."""
    st.info("Fan charts — coming in next step")
```

- [ ] **Step 2: Verify syntax**

```
venv/Scripts/python.exe -c "import ast; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(fed-forecaster): full causal chain with confidence decay bar (Section 5)"
```

---

### Task 11: Section 6 — Medium-term fan charts

**Files:**
- Modify: `modules/risk_regime.py`

- [ ] **Step 1: Replace `_render_fed_fan_charts` stub**

```python
def _render_fed_fan_charts(medium: dict | None, adj_probs: list[dict]):
    """Section 6: probability-weighted medium-term fan charts per asset class."""
    from services.fed_forecaster import SCENARIO_KEYS, SCENARIO_LABELS
    import plotly.graph_objects as go
    import numpy as np

    _section_header("Medium-Term Outlook (3–12 months)")

    if not medium:
        st.info("Medium-term forecast unavailable.")
        return

    ASSET_KEYS   = ["equities", "bonds", "commodities", "usd"]
    ASSET_LABELS = {"equities": "Equities", "bonds": "Bonds",
                    "commodities": "Commodities", "usd": "USD"}
    months = list(range(1, 13))

    # Weight scenarios by adjusted probability
    prob_map = {r["scenario"]: r["prob"] for r in adj_probs}

    col_left, col_right = st.columns(2)
    col_map = {
        "equities":    col_left,
        "bonds":       col_right,
        "commodities": col_left,
        "usd":         col_right,
    }

    for asset in ASSET_KEYS:
        # Compute probability-weighted p25, p50, p75 across all scenarios
        w_p25 = np.zeros(12)
        w_p50 = np.zeros(12)
        w_p75 = np.zeros(12)
        total_weight = 0.0

        for key in SCENARIO_KEYS:
            sc_data = (medium.get(key) or {}).get(asset, {})
            p25 = sc_data.get("monthly_p25", [0.0] * 12)
            p50 = sc_data.get("monthly_p50", [0.0] * 12)
            p75 = sc_data.get("monthly_p75", [0.0] * 12)
            w = prob_map.get(key, 0.25)
            if len(p25) == 12 and len(p50) == 12 and len(p75) == 12:
                w_p25 += w * np.array(p25)
                w_p50 += w * np.array(p50)
                w_p75 += w * np.array(p75)
                total_weight += w

        if total_weight > 0:
            w_p25 /= total_weight
            w_p50 /= total_weight
            w_p75 /= total_weight

        fig = go.Figure()

        # Band fill (p25 to p75)
        fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=list(w_p75) + list(w_p25[::-1]),
            fill="toself",
            fillcolor="rgba(100,149,237,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p25–p75",
            showlegend=False,
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=months,
            y=list(w_p50),
            line=dict(color=COLORS.get("bloomberg_orange", "#f0a040"), width=2),
            name="Median",
            showlegend=False,
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS.get("border", "#444"), line_width=1)

        apply_dark_layout(fig)
        fig.update_layout(
            title=dict(text=ASSET_LABELS[asset], font_size=13),
            height=220,
            margin=dict(l=0, r=10, t=30, b=20),
            xaxis=dict(title="Month", tickmode="linear", dtick=3),
            yaxis=dict(title="Cumulative return (%)"),
        )

        col_map[asset].plotly_chart(fig, use_container_width=True)

        # Narrative from dominant scenario
        dominant_key = max(SCENARIO_KEYS, key=lambda k: prob_map.get(k, 0.0))
        narrative = (medium.get(dominant_key) or {}).get(asset, {}).get("narrative", "")
        if narrative and narrative != "...":
            col_map[asset].caption(narrative)
```

- [ ] **Step 2: Verify syntax**

```
venv/Scripts/python.exe -c "import ast; ast.parse(open('modules/risk_regime.py').read()); print('syntax OK')"
```

- [ ] **Step 3: Run full test suite**

```
venv/Scripts/python.exe -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add modules/risk_regime.py
git commit -m "feat(fed-forecaster): medium-term probability fan charts (Section 6)"
```

---

### Task 12: End-to-end smoke test

**Files:**
- No changes — verify integration

- [ ] **Step 1: Check all imports resolve cleanly**

```
venv/Scripts/python.exe -c "
import sys, os
sys.path.insert(0, '.')
# Patch streamlit to avoid page config issues
import unittest.mock as m
with m.patch('streamlit.cache_data', lambda **kw: (lambda f: f)):
    from services import fed_forecaster
    print('fetch_zq_probabilities:', fed_forecaster.fetch_zq_probabilities)
    print('fetch_fed_communications:', fed_forecaster.fetch_fed_communications)
    print('score_fed_tone:', fed_forecaster.score_fed_tone)
    print('adjust_probabilities:', fed_forecaster.adjust_probabilities)
    print('build_fed_context:', fed_forecaster.build_fed_context)
    print('generate_forecast:', fed_forecaster.generate_forecast)
    print('get_next_fomc:', fed_forecaster.get_next_fomc)
    print('All imports OK')
"
```

Expected: prints all function names, then `All imports OK`

- [ ] **Step 2: Run full test suite one final time**

```
venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all tests pass, no regressions.

- [ ] **Step 3: Final commit**

```bash
git add services/fed_forecaster.py modules/risk_regime.py tests/test_fed_forecaster.py
git commit -m "feat(fed-forecaster): complete Fed Policy Forecasting Machine

- services/fed_forecaster.py: ZQ futures probabilities, Fed RSS parsing,
  Groq tone scoring, adjust_probabilities, build_fed_context, generate_forecast
- modules/risk_regime.py: tab system, Sections 1-6 (FOMC strip, comms tracker,
  probability bars, asset matrix, causal chain, fan charts)
- FEDFUNDS added to FRED warm-cache and fred_ids
- tests/test_fed_forecaster.py: 25+ tests covering all pure functions"
```
