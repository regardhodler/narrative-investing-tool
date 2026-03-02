# Narrative Investing Tool

## Project Overview
Streamlit-based multi-module investment intelligence dashboard. Entry point: `app.py`.

## Architecture
- `app.py` — Streamlit app with sidebar module routing (6 modules)
- `modules/` — Each module is a standalone `render()` function
  - `narrative_discovery.py` — Module 1: Ticker/narrative discovery
  - `narrative_pulse.py` — Module 2: Narrative sentiment tracking
  - `edgar_scanner.py` — Module 3: SEC EDGAR filing search
  - `institutional.py` — Module 4: 13F institutional holdings analysis
  - `insider_congress.py` — Module 5: Form 4 insider trades & Congress trades
  - `options_activity.py` — Module 6: Options flow analysis
- `services/` — API clients
  - `sec_client.py` — SEC EDGAR API (filings, 13F, insider trades, CIK mapping)
  - `claude_client.py` — Claude AI integration
  - `congress_client.py` — Congress trading data
  - `ibkr_client.py` — Interactive Brokers connection
  - `trends_client.py` — Google Trends
- `utils/` — Shared utilities
  - `session.py` — Streamlit session state helpers (ticker, narrative, IBKR status)
  - `theme.py` — Dark theme colors (`COLORS` dict) and `apply_dark_layout()` for Plotly

## Key Conventions
- All SEC requests use `SEC_HEADERS` with User-Agent and go through `_rate_limit()` (10 req/sec max)
- Heavy API calls are cached with `@st.cache_data(ttl=3600)` (1hr) or `ttl=86400` (24hr)
- Concurrent SEC fetches use `ThreadPoolExecutor(max_workers=5)` to stay under rate limits
- Plotly charts use `apply_dark_layout()` from `utils/theme.py`
- CIK-to-ticker mapping via `get_cik_ticker_map()` is the foundation for SEC lookups

## Running
```bash
streamlit run app.py
```

## Dependencies
See `requirements.txt`. Key: streamlit, plotly, pandas, requests, python-dotenv, ib_insync.
