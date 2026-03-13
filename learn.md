# Module 0 Upgrade Guide (Macro Strategist + Engineering)

## Objective
Transform Module 0 into a daily macro decision engine with concise output:
- 15 core indicators classified as Risk-On / Neutral / Risk-Off
- Dalio macro quadrant
- Valuation + cycle-stage interpretation
- Macro Score (0-100)
- Portfolio bias by asset bucket
- SPY options sentiment mode (gamma zone + walls + flip)

## Architecture Updates

### 1) Data Layer (services)
Use shared data services instead of embedding network calls in UI:
- `fetch_batch()` for market proxies (SPY, QQQ, UUP, USO, CPER, ^DJI)
- `fetch_fred_series()` for macro time series (yield curve, credit spreads, ISM, FCI, etc.)
- `fetch_options_chain_snapshot()` for SPY strike-level options aggregates

Why this matters:
- Keeps Module 0 composable and testable
- Enables reuse by valuation/options modules
- Centralizes caching and provider fallback

### 2) Macro Engine (module-local compute)
`_build_macro_dashboard()` should:
- Convert each indicator into a normalized score in [-1, +1]
- Classify each score into:
  - 🟢 Risk-On
  - 🟡 Neutral
  - 🔴 Risk-Off
- Aggregate to `Macro Score (0-100)`
- Derive `Risk-On / Neutral / Risk-Off` headline regime

### 3) Regime Framework
Use two layers:
- **Risk environment score** (portfolio posture)
- **Dalio quadrant** (growth/inflation mix):
  - Reflation, Goldilocks, Stagflation, Deflation

This prevents false certainty from a single scalar regime.

### 4) Options Sentiment Mode
For SPY options:
- Compute strike-level net gamma proxy from OI + IV
- Identify:
  - Current gamma zone (positive/negative)
  - Gamma flip
  - Call wall
  - Put wall
- Plot Strike (x) vs Net Gamma Proxy (y), with vertical markers for spot/flip/walls

## Engineering Principles
- Keep UI concise and decision-first (daily workflow)
- Isolate computation from rendering
- Handle missing data gracefully (default to Neutral instead of failure)
- Cache heavy calls (`st.cache_data`) with realistic TTL
- Avoid adding dependencies unless strictly needed

## Confidence Layer (Added)
Each indicator now carries a confidence score (Low/Medium/High):
- Market-based indicators: confidence from ticker freshness/staleness
- FRED macro indicators: confidence from series recency vs expected release cadence
- Composite indicators (Buffett, CAPEX vs Liquidity): blended confidence from source series

Why it helps:
- Prevents over-weighting stale/lagged macro releases
- Keeps daily decision flow concise while adding reliability context

## Streamlit Free-Tier Guardrails
To reduce compute/API usage:
- FRED data cache TTL increased to **6h**
- SPY options chain cache TTL increased to **3h**
- SPY gamma mode uses fewer expiries (`max_expiries=2`) to reduce options-chain requests
- Keep refresh manual and only use when needed (forced refresh clears cache)

### Low Compute Mode (UI Toggle)
Module 0 now includes a **Low Compute Mode** toggle (default ON):
- Still shows textual SPY options sentiment (price, gamma zone, flip, call wall, put wall)
- Uses lighter options processing (`max_expiries=1`)
- Disables gamma chart rendering to reduce front-end and compute overhead

Operational recommendation:
- For daily use, open dashboard 1–2 times/day and avoid repeated forced refreshes.

## Future Enhancements (Optional)
- Add confidence score per indicator based on data freshness/frequency
- Add historical Macro Score time series persistence (daily snapshots)
- Add user-selectable regions (US vs global)
- Add scenario templates (soft landing, recession, inflation shock)

## Validation Checklist
- Module loads without errors
- All 15 indicators appear with signal label
- Macro score and quadrant render consistently
- SPY options sentiment panel shows price/zone/flip/walls
- Chart renders with strike on x-axis and gamma proxy on y-axis
