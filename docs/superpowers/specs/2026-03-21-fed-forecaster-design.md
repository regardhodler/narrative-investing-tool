# Fed Policy Forecasting Machine — Design Spec

**Date:** 2026-03-21
**Module:** Risk Regime (Module 0)
**Status:** Approved

---

## Overview

Add a "Fed Forecaster" tab to the existing Risk Regime module. Given the current macro regime (stagflation, risk-off), the forecaster:

1. Derives Fed scenario probabilities from Fed Funds Futures via a tiered fetch strategy (yfinance → CME FedWatch HTML fallback)
2. Tracks Fed communications in real-time via Federal Reserve RSS feeds
3. Adjusts scenario probabilities using AI-scored tone from Fed speeches/statements
4. Generates a full probability-weighted causal chain (asset cascade) via Groq
5. Displays near-term (0–3 month) and medium-term (3–12 month) asset class forecasts with explicit percentage probabilities on every element

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Probability source | Hybrid: Fed Funds Futures + Fed RSS tone adjustment | Market-implied base anchored in real consensus; RSS adjusts for Fed communication signal |
| Fed scenarios | 4: Hold / Cut 25bp / Cut 50bp / Hike 25bp | Cut 50bp vs 25bp tells opposite stagflation stories; 50bp hike is <2% prob, excluded |
| Time horizon | Both near-term (0–3m) and medium-term (3–12m) | Near-term = actionable; medium-term = portfolio positioning |
| Causal chain depth | Full chain with per-hop confidence weights (cumulative decay model) | Stagflation requires full transmission mechanism; confidence decay shown explicitly |
| UI placement | New tab alongside existing Macro Dashboard tab | Full screen real estate; clean separation from regime signals |
| Refresh cadence | Auto: Futures 4h, RSS 1h, Groq 4h | Live feel without hammering APIs; manual Refresh button available |

---

## Architecture

```
Fed Funds Futures (yfinance → CME fallback) ─┐
Fed RSS Feeds (requests)                      ├──► services/fed_forecaster.py ──► Groq ──► modules/risk_regime.py
FRED FEDFUNDS + regime signals                ─┘                                            (Tab 2: Fed Forecaster)
```

### Files Changed

| File | Change |
|---|---|
| `services/fed_forecaster.py` | **New** — all data fetching, tone scoring, forecast generation |
| `modules/risk_regime.py` | **Modified** — add `FEDFUNDS` to FRED series list; wrap render in tabs; add Tab 2 render function |

---

## Canonical Scenario Keys

All three functions (`fetch_zq_probabilities`, `score_fed_tone`, `generate_forecast`) and the Groq prompt/response must use these exact string keys:

| Key | Display label |
|---|---|
| `"hold"` | Fed Holds |
| `"cut_25"` | Cut 25bp |
| `"cut_50"` | Cut 50bp |
| `"hike_25"` | Hike 25bp |

---

## `modules/risk_regime.py` — FRED Series Addition

**Three locations** must be updated:

1. **`_FRED_SERIES_IDS`** (warm-cache list, ~line 1243) — add `"FEDFUNDS"` so `warm_fred_cache` pre-fetches it at startup:
   ```python
   _FRED_SERIES_IDS = [
       "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
       "PCEPILFE", "PNFI", "THREEFYTP10", "INDPRO", "NFCI", "DGS10",
       "ICSA", "USSLIND", "UMCSENT", "PERMIT", "FEDFUNDS",  # ← add
   ]
   ```

2. **`fred_ids` dict in `render()`** (~line 1277) — so `fred_data["fedfunds"]` is populated and passed to `_render_fed_forecaster`:
   ```python
   "fedfunds": "FEDFUNDS",   # Effective Federal Funds Rate
   ```

3. **`_build_macro_dashboard`** — does **not** need updating. `build_fed_context` is called from `_render_fed_forecaster`, not from `_build_macro_dashboard`.

The `FEDFUNDS` series is monthly; its last value is the most recent effective rate set by the Fed (e.g. 5.33%).

---

## `services/fed_forecaster.py` — Function Contracts

### `fetch_zq_probabilities() -> list[dict]`
- **Cache:** `@st.cache_data(ttl=14400)` (4h)
- Internally calls `fetch_fred_series_safe("FEDFUNDS")` to get `current_rate`. This is itself a cached call (TTL 12h) so it will be a cache hit after Tab 1 warms it. No `fred_data` parameter is needed or passed.

**Tiered fetch strategy (in order):**

1. **yfinance `ZQ=F`** — front-month Fed Funds Futures generic ticker. `implied_rate = 100 - last_price`. Proceed to tier 2 if yfinance returns empty DataFrame or raises.
2. **yfinance named contracts** — try `ZQH26`, `ZQK26`, `ZQM26` in sequence; use the first non-empty result. Proceed to tier 3 if all return empty.
3. **Equal-weight fallback** — 25% per scenario, `data_unavailable: True` flag set, `source: "fallback"`.

Note: CME FedWatch is a JavaScript SPA; a plain `requests.get()` against its URL does not return parseable probability data. It is excluded from the tiered strategy. The equal-weight fallback is the de facto result when yfinance ZQ data is unavailable.

**Probability derivation from implied rate (tiers 1–2):**
```
current_rate = fetch_fred_series_safe("FEDFUNDS").iloc[-1]  # e.g. 5.33
implied_rate = 100 - futures_price                           # e.g. 5.42
delta = implied_rate - current_rate                          # e.g. +0.09

# Distribute across 4 scenarios using normal distribution centred on delta
# with σ = 0.15 (typical intra-meeting uncertainty)
# Scenarios: -0.50, -0.25, 0.00, +0.25 (cut_50, cut_25, hold, hike_25)
```

- Returns: `[{scenario: str, prob: float, implied_rate: float, source: "yfinance"|"fallback"}]`
- `scenario` values use canonical keys: `"hold"`, `"cut_25"`, `"cut_50"`, `"hike_25"`
- Probabilities sum to 1.0

### `fetch_fed_communications(max_items: int = 5) -> list[dict]`
- **Cache:** `@st.cache_data(ttl=3600)` (1h)
- Polls two Federal Reserve RSS feeds (unauthenticated, public):
  - Press releases: `https://www.federalreserve.gov/rss/releases.xml`
  - Speeches: `https://www.federalreserve.gov/rss/speeches.xml`
- Parses with `xml.etree.ElementTree` (stdlib)
- `raw_text` = RSS `<description>` element (1–3 sentence summary published by the Fed). Full speech text is **not** fetched — the summary is sufficient for tone scoring.
- Returns: `[{title: str, date: str, url: str, source: "release"|"speech", raw_text: str}]` — most recent `max_items` items merged and sorted by date descending
- **Fallback:** Empty list; tone adjustment step is skipped entirely

### `score_fed_tone(comm_key: str, _communications: list[dict]) -> dict`
- **Cache:** `@st.cache_data(ttl=3600)` (1h)
- `comm_key` is a stable string hash of `[(item["title"], item["date"]) for item in communications]`, computed by the caller before the cache boundary. It is a plain `str` and is always hashable by Streamlit.
- `_communications` uses a leading underscore — Streamlit's `@st.cache_data` skips hashing arguments whose names start with `_`. This prevents an `UnhashableParamError` on the `list[dict]` argument while still passing it into the function body for use in the Groq call.
- Single Groq call scoring all communications at once using their `raw_text` (RSS summaries)
- Returns:
  ```json
  {
    "items": [
      {
        "title": "...",
        "hawkish_prob": 0.85,
        "neutral_prob": 0.12,
        "dovish_prob": 0.03,
        "adjustment_confidence": 0.78
      }
    ],
    "aggregate_bias": "hawkish",
    "prob_adjustments": {"hold": 0.08, "cut_25": -0.03, "cut_50": -0.05, "hike_25": 0.00}
  }
  ```
- **Fallback:** Zero adjustments dict; `aggregate_bias: "neutral"`

### `adjust_probabilities(base_probs: list[dict], tone_result: dict) -> list[dict]`
- Pure math, no cache, no network
- Applies `prob_adjustments` from tone scoring to base probabilities
- Clamps all values to [0, 1] and re-normalises to sum to 1.0
- Adds `delta` field (float, signed) showing adjustment per scenario
- **Timestamp contract:** Caller is responsible for displaying the age of the oldest input. The UI section renders two timestamps: "Futures as of X" and "Comms as of Y". `adjust_probabilities` itself has no timestamp concept.

### `build_fed_context(macro: dict, fred_data: dict) -> dict`
- Pure function, no cache, no network
- Reads `fred_data["fedfunds"]` for current Fed Funds Rate. If `None`, falls back to last known value from disk cache via `fetch_fred_series_safe("FEDFUNDS")`. Does **not** use `DGS10` as a proxy (numerically different instrument).
- Packages regime signals into a serialisable dict for the Groq prompt:
  - `fed_funds_rate`: float (from `FEDFUNDS`)
  - `core_pce`: float (from `PCEPILFE`)
  - `unemployment`: float (from `UNRATE`)
  - `yield_curve`: float (T10Y2Y)
  - `credit_spread`: float (BAMLH0A0HYM2)
  - `quadrant`: str (e.g. "Stagflation")
  - `macro_score`: int (0–100)
  - `regime`: str (e.g. "Risk-Off")

### `generate_forecast(context_json: str, scenarios_json: str) -> dict`
- **Cache:** `@st.cache_data(ttl=14400)` (4h)
- Both arguments are **JSON strings** (serialised from `context` dict and `scenarios` list by the caller before the cache boundary). Plain strings are always hashable by Streamlit.
- Single Groq call covering all 4 scenarios and both time horizons
- Structured JSON response parsed from Groq output
- **Fallback:** Returns `None`; UI shows "Analysis unavailable" notice

#### Forecast JSON Schema

```json
{
  "near_term": {
    "<scenario>": {
      "equities": {
        "direction": "up|down|flat",
        "magnitude_low": -8.0,
        "magnitude_high": -3.0,
        "direction_prob": 0.72,
        "magnitude_confidence": 0.65,
        "chain": [
          {"step": "Real yields unchanged", "confidence": 0.72},
          {"step": "Risk premium stays elevated", "confidence": 0.65},
          {"step": "Defensives outperform", "confidence": 0.60}
        ]
      },
      "bonds":       { "...": "same structure" },
      "commodities": { "...": "same structure" },
      "usd":         { "...": "same structure" }
    }
  },
  "medium_term": {
    "<scenario>": {
      "equities": {
        "monthly_p25": [-2.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0],
        "monthly_p50": [-1.0, -2.0, -3.0, -4.0, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -8.0],
        "monthly_p75": [0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0],
        "narrative": "Equities face persistent headwinds as..."
      },
      "bonds":       { "...": "same structure — 12-element arrays" },
      "commodities": { "...": "same structure" },
      "usd":         { "...": "same structure" }
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

**`medium_term` arrays** are 12 floats, one per month (months 1–12 from today). These are cumulative return estimates (%) used as Y-axis values in the fan chart. Groq is instructed to return exactly 12 values per array.

---

## Causal Chain Confidence Model

Chain confidences use a **cumulative decay model**:

- Each hop's `confidence` field is the **cumulative** probability that the full chain up to and including that step is correct (not a standalone hop probability)
- Confidence always decreases monotonically down the chain
- The decay bar shows: first step confidence → last step confidence, e.g. `Start 78% → End 55%`
- Formula: `confidence[n] = confidence[0] * decay_factor^n` where `decay_factor` is tuned per scenario by Groq (typically 0.92–0.97)
- This model is consistent: the bar formula and the chain values use the same cumulative interpretation

---

## `modules/risk_regime.py` — Tab 2 UI Sections

### Tab System
```python
tab1, tab2 = st.tabs(["📊 Macro Dashboard", "🏦 Fed Forecaster"])
```
Existing `render()` body moves under `with tab1`. `_render_fed_forecaster(macro, fred_data)` called under `with tab2`.

### Section 1 — FOMC Context Strip
Three `st.metric` columns:
- **Next FOMC:** Date + days away. Source: hardcoded list of 2026 FOMC meeting dates (published annually by the Fed at `federalreserve.gov/monetarypolicy/fomccalendars.htm`). Stored as a constant `_FOMC_DATES_2026` in `fed_forecaster.py`. Updated manually each January when the Fed publishes the next year's calendar.
- **Current Rate:** `FEDFUNDS` latest value (e.g. "5.33%")
- **Regime:** Regime badge from `macro["macro_regime"]` (e.g. "🔴 Risk-Off · Stagflation")

### Section 2 — Fed Communications Tracker
- Displays last 3–5 items from `fetch_fed_communications()`
- Per item:
  - Tone badge: 🔴 Hawkish `[85%]` / 🟡 Neutral `[71%]` / 🟢 Dovish `[63%]`
  - Speaker/source, date (relative: "3 days ago"), truncated headline (80 chars)
  - Probability delta (from `score_fed_tone`): `Δ Hold +8pp [78% confidence]`
- Two timestamps below: `Futures as of HH:MM` and `Comms as of HH:MM`
- If `fetch_fed_communications()` returns empty list: grey notice "Fed communications unavailable — tone adjustment skipped"

### Section 3 — Scenario Probability Bars
- 4 Plotly horizontal bar charts (one per scenario) using existing dark theme (`apply_dark_layout`)
- Each bar: adjusted probability (large) + base probability label + delta badge (▲/▼ Xpp)
- Source indicator: "Futures: yfinance" or "Futures: CME" or "⚠ Fallback: equal-weight" (orange banner)

### Section 4 — Near-Term Asset Impact Matrix (0–3 months)
- Header row + 4 scenario columns + 4 asset class rows = `st.columns([2,1,1,1,1])`
- Each cell:
  - Direction arrow + label: `▼ DOWN`
  - Direction probability: `[72%]`
  - Magnitude range: `-3% to -8%`
  - Magnitude confidence: `[65% CI]`
- Background color coded per direction (dark red / dark green / grey, consistent with `COLORS` theme)

### Section 5 — Full Causal Chain
- `st.expander` per scenario; dominant scenario (highest prob) expanded by default
- Indented chain steps with cumulative confidence at each hop
- Confidence decay bar rendered as Unicode blocks: `████████░░ 78% → 55%`
- Two sub-branches rendered as indented sub-lists where chain diverges

### Section 6 — Medium-Term Fan Chart (3–12 months)
- 2×2 Plotly subplot grid, one chart per asset class (equities, bonds, commodities, USD)
- Per chart: scenario-probability-weighted p25/p50/p75 arrays (weighted average across all 4 scenarios using their adjusted probabilities)
- `go.Scatter` with `fill='tonexty'` between p25 and p75 bands
- X-axis: months 1–12; Y-axis: cumulative return %
- Below each chart: AI narrative paragraph from `generate_forecast`

---

## Probability Rules

- All probabilities displayed as percentages rounded to nearest integer
- Scenario probabilities sum to 100%
- Causal chain confidences are **cumulative** (monotonically decreasing down the chain)
- Confidence decay bar: first step → last step confidence
- Magnitude confidence intervals are 65% CI
- Medium-term fan chart uses scenario-weighted percentile bands (p25/p50/p75 arrays of length 12)

---

## Caching & Refresh

| Function | Cache arg type | TTL |
|---|---|---|
| `fetch_zq_probabilities` | none | 4h |
| `fetch_fed_communications` | `max_items: int` | 1h |
| `score_fed_tone` | `comm_key: str` (hashed), `_communications: list[dict]` (unhashed, leading `_`) | 1h |
| `generate_forecast` | `context_json: str`, `scenarios_json: str` | 4h |

**Refresh buttons:**
- Tab 1 "Refresh Data" button: calls `st.cache_data.clear()` — clears everything including Tab 2
- Tab 2 "Refresh Forecaster" button: calls `st.cache_data.clear()` — same behaviour; documented in UI as "refreshes all data"

---

## Error Handling & Fallbacks

| Failure | Behaviour |
|---|---|
| yfinance ZQ=F and all named contracts empty | Equal-weight 25% per scenario, orange "⚠ Futures unavailable" banner |
| Fed RSS unreachable | Empty comms list; tone adjustment skipped; grey notice shown |
| Groq API error | Return `None`; UI renders "Analysis unavailable" in affected sections |
| `FEDFUNDS` series missing | Call `fetch_fred_series_safe("FEDFUNDS")` directly; if still None, omit from context and note in Groq prompt |
| Partial Groq JSON | Parse available keys; render "—" in missing matrix cells with grey "Partial data" badge |

---

## Out of Scope

- Push notifications or background polling (Streamlit doesn't support true background tasks)
- Backtesting forecast accuracy
- User-configurable scenario weights
- More than 4 asset classes (equities, bonds, commodities, USD only)
- Options/derivatives sub-asset analysis
- Full speech text fetching (RSS description summaries used for tone scoring)
