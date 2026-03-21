# Fed Policy Forecasting Machine — Design Spec

**Date:** 2026-03-21
**Module:** Risk Regime (Module 0)
**Status:** Approved

---

## Overview

Add a "Fed Forecaster" tab to the existing Risk Regime module. Given the current macro regime (stagflation, risk-off), the forecaster:

1. Derives Fed scenario probabilities from ZQ (Fed Funds Futures) via yfinance
2. Tracks Fed communications in real-time via Federal Reserve RSS feeds
3. Adjusts scenario probabilities using AI-scored tone from Fed speeches/statements
4. Generates a full probability-weighted causal chain (asset cascade) via Groq
5. Displays near-term (0–3 month) and medium-term (3–12 month) asset class forecasts with explicit percentage probabilities on every element

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Probability source | Hybrid: ZQ futures + Fed RSS tone adjustment | Market-implied base anchored in real consensus; RSS adjusts for Fed communication signal |
| Fed scenarios | 4: Hold / Cut 25bp / Cut 50bp / Hike 25bp | Cut 50bp vs 25bp tells opposite stagflation stories; 50bp hike is <2% prob, excluded |
| Time horizon | Both near-term (0–3m) and medium-term (3–12m) | Near-term = actionable; medium-term = portfolio positioning |
| Causal chain depth | Full chain with per-hop probability weights | Stagflation requires full transmission mechanism; confidence decay shown explicitly |
| UI placement | New tab alongside existing Macro Dashboard tab | Full screen real estate; clean separation from regime signals |
| Refresh cadence | Auto: ZQ 4h, RSS 1h, Groq 4h | Live feel without hammering APIs; manual Refresh button available |

---

## Architecture

```
ZQ Futures (yfinance)     ─┐
Fed RSS Feeds (requests)   ├──► services/fed_forecaster.py ──► Groq ──► modules/risk_regime.py
FRED + regime signals       ─┘                                            (Tab 2: Fed Forecaster)
```

### Files Changed

| File | Change |
|---|---|
| `services/fed_forecaster.py` | **New** — all data fetching, tone scoring, forecast generation |
| `modules/risk_regime.py` | **Modified** — wrap existing content in Tab 1, add Tab 2 render function |

---

## `services/fed_forecaster.py` — Function Contracts

### `fetch_zq_probabilities() -> list[dict]`
- **Cache:** `@st.cache_data(ttl=14400)` (4h)
- Fetches next 2 ZQ futures contracts (e.g. `ZQK26`, `ZQM26`) via yfinance
- Computes `implied_rate = 100 - futures_price`
- Derives 4-scenario probabilities using Fed Funds futures math
- Returns: `[{scenario, prob, implied_rate, contract}]` — probabilities sum to 1.0
- **Fallback:** Equal-weight 25% each, with a `data_unavailable: True` flag

### `fetch_fed_communications(max_items: int = 5) -> list[dict]`
- **Cache:** `@st.cache_data(ttl=3600)` (1h)
- Polls Federal Reserve RSS feeds:
  - Press releases: `https://www.federalreserve.gov/rss/releases.xml`
  - Speeches: `https://www.federalreserve.gov/rss/speeches.xml`
- Returns: `[{title, date, url, source, raw_text}]` — most recent `max_items` items
- **Fallback:** Empty list (tone adjustment skipped)

### `score_fed_tone(communications: list[dict]) -> dict`
- **Cache:** `@st.cache_data(ttl=3600)` (1h)
- Single Groq call scoring all communications at once
- Returns per-item tone: `{hawkish_prob, neutral_prob, dovish_prob}` (sum to 1.0)
- Also returns aggregate tone signal and probability adjustments:
  ```json
  {
    "items": [{"title": "...", "hawkish_prob": 0.85, "neutral_prob": 0.12, "dovish_prob": 0.03, "adjustment_confidence": 0.78}],
    "aggregate_bias": "hawkish",
    "prob_adjustments": {"hold": +0.08, "cut_25": -0.03, "cut_50": -0.05, "hike_25": 0.00}
  }
  ```
- **Fallback:** Zero adjustments (raw ZQ probabilities used as-is)

### `adjust_probabilities(base_probs: list[dict], tone_result: dict) -> list[dict]`
- Pure math, no cache, no network
- Applies `prob_adjustments` from tone scoring to base ZQ probabilities
- Clamps all values to [0, 1] and re-normalises to sum to 1.0
- Adds `delta` field showing the adjustment made per scenario

### `build_fed_context(macro: dict, fred_data: dict) -> dict`
- Pure function, no cache, no network
- Packages already-computed regime signals into AI prompt context:
  - Current Fed Funds Rate (from FRED `FEDFUNDS`)
  - Core PCE (from FRED `PCEPILFE`)
  - Unemployment rate (from FRED `UNRATE`)
  - Yield curve (T10Y2Y)
  - Credit spread (BAMLH0A0HYM2)
  - Dalio quadrant + macro score + regime label

### `generate_forecast(scenarios: list[dict], context: dict) -> dict`
- **Cache:** `@st.cache_data(ttl=14400)` (4h)
- Single Groq call covering all 4 scenarios, both time horizons
- Structured JSON response (see schema below)
- **Fallback:** Returns `None`; UI shows "Analysis unavailable, using cached data" or degraded state

#### Forecast JSON Schema

```json
{
  "near_term": {
    "<scenario>": {
      "equities":    {"direction": "up|down|flat", "magnitude_low": -8.0, "magnitude_high": -3.0, "direction_prob": 0.72, "magnitude_confidence": 0.65, "chain": ["step 1 [prob]", "step 2 [prob]"]},
      "bonds":       { ... },
      "commodities": { ... },
      "usd":         { ... }
    }
  },
  "medium_term": {
    "<scenario>": {
      "equities":    {"p25": -15.0, "p50": -8.0, "p75": -2.0, "narrative": "..."},
      "bonds":       { ... },
      "commodities": { ... },
      "usd":         { ... }
    }
  },
  "causal_chains": {
    "<scenario>": [
      {"step": "Fed holds at 5.25–5.50%", "confidence": 1.0},
      {"step": "Inflation stays elevated above 3%", "confidence": 0.78},
      {"step": "Real yields remain positive (+2.1%)", "confidence": 0.72},
      {"step": "Growth expectations compress", "confidence": 0.65},
      {"step": "Credit spreads widen ~30bp", "confidence": 0.60},
      {"step": "Defensives outperform cyclicals", "confidence": 0.55}
    ]
  }
}
```

---

## `modules/risk_regime.py` — Changes

### Tab System
Wrap the existing `render()` body in `tab1, tab2 = st.tabs(["Macro Dashboard", "Fed Forecaster"])` with existing content under `tab1`.

### `_render_fed_forecaster(macro: dict, fred_data: dict)`
New private function rendering Tab 2 in 6 sections:

**Section 1 — FOMC Context Strip**
Single-row metrics: Next FOMC date + days away, Current Fed Funds Rate, Current Regime badge.

**Section 2 — Fed Communications Tracker**
- Shows last 3–5 Fed communications with:
  - Tone badge: 🔴 Hawkish `[85%]` / 🟡 Neutral `[71%]` / 🟢 Dovish `[63%]`
  - Speaker name, date, truncated headline
  - Probability delta: `Δ Hold +8pp [78% confidence]`
- Last updated timestamp

**Section 3 — Scenario Probability Bars**
- 4 horizontal bars using Plotly (matches existing dark theme)
- Each bar shows: adjusted probability + base probability + delta indicator (▲/▼ Xpp)
- Data sourced from `adjust_probabilities()` output
- Fallback banner if ZQ data unavailable

**Section 4 — Near-Term Asset Impact Matrix** (0–3 months)
- 4×4 grid (4 scenarios × 4 asset classes) using `st.columns`
- Each cell shows:
  - Direction arrow + label: `▼ DOWN`
  - Direction probability: `[72%]`
  - Magnitude range: `-3% to -8%`
  - Magnitude confidence: `[65% CI]`
- Color coded: red = bearish, green = bullish, grey = flat

**Section 5 — Full Causal Chain**
- `st.tabs` or `st.expander` per scenario (dominant scenario expanded by default)
- Vertical chain with confidence at each hop
- Confidence decay bar at bottom: `████████░░ Start 78% → End 55%`
- Two sub-branches where chain diverges (e.g. USD path vs equity path)

**Section 6 — Medium-Term Fan Chart** (3–12 months)
- Plotly fan chart per asset class using `go.Scatter` with fill between percentile bands
  - Dark fill = 25th–75th percentile range
  - Lighter line = median (p50)
- X-axis: months 1–12
- One chart per asset class in a 2×2 grid
- Below each chart: AI narrative paragraph (from `generate_forecast`)

---

## Probability Rules

- All probabilities displayed as percentages rounded to nearest integer
- All probabilities within a scenario sum to 100%
- Causal chain confidences are standalone per-hop probabilities (not cumulative)
- Confidence decay bar shows: `start_confidence * decay_factor^n_hops`
- Magnitude confidence intervals are 65% CI (1 standard deviation equivalent)
- Medium-term percentile bands: p25/p50/p75 across scenario-weighted distribution

---

## Caching & Refresh

| Function | TTL | Scope |
|---|---|---|
| `fetch_zq_probabilities` | 4h | st.cache_data |
| `fetch_fed_communications` | 1h | st.cache_data |
| `score_fed_tone` | 1h | st.cache_data |
| `generate_forecast` | 4h | st.cache_data |
| Manual Refresh button | clears all | `st.cache_data.clear()` |

The existing Tab 1 Refresh button clears all caches including Tab 2. A second Refresh button is added within Tab 2 for users who only want to refresh the forecaster.

---

## Error Handling & Fallbacks

| Failure | Behaviour |
|---|---|
| ZQ futures unavailable | Equal-weight 25% per scenario, orange warning banner |
| Fed RSS unreachable | Skip tone adjustment, grey "Comms unavailable" notice |
| Groq API error | Show last cached forecast if age < 24h, else "Analysis unavailable" notice |
| FRED current rate missing | Use last known rate from `DGS10` approximation |
| Partial Groq response | Render available sections, show "Partial data" badge on missing cells |

---

## Out of Scope

- Push notifications or background polling (Streamlit doesn't support true background tasks)
- Backtesting the forecast accuracy
- User-configurable scenario weights
- More than 4 asset classes (equities, bonds, commodities, USD only)
- Options/derivatives sub-asset analysis
