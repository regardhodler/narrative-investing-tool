# Narrative Investing Tool

## Project Overview
Streamlit-based multi-module investment intelligence dashboard. Entry point: `app.py`.

## Architecture
- `app.py` — Streamlit app with sidebar module routing, password auth gate via APP_PASSWORD env var

### modules/
Each module is a standalone `render()` function.
- `quick_run.py` — **QIR (Quick Intelligence Report)**: master dashboard combining all signals into one view. Contains: LL-anchored Crisis Detection (CI% system), HMM Brain State card, conviction scoring, Elliott Wave, Wyckoff, GEX, entry signal
- `backtesting.py` — Historical snapshot viewer with CI% LL-anchored block, Wyckoff pill, regime overlays
- `risk_regime.py` — Module 0: Cross-asset risk-on/risk-off regime indicator (z-score based, 17+ signals, daily history)
- `narrative_discovery.py` — Module 1: Ticker/narrative discovery (AI-grouped by narrative themes)
- `narrative_pulse.py` — Module 2: Narrative sentiment tracking
- `edgar_scanner.py` — Module 3: SEC EDGAR filing search
- `institutional.py` — Module 4: 13F institutional holdings analysis
- `insider_congress.py` — Module 5: Form 4 insider trades & Congress trades
- `options_activity.py` — Module 6: Options flow analysis (unusual activity sentiment verdict + chart)
- `valuation.py` — Module 7: AI Valuation & Recommendation (aggregates all signals, Groq-powered rating)
- `elliott_wave.py` — Elliott Wave analysis module
- `wyckoff.py` — Wyckoff phase analysis module
- `macro_scorecard.py` — Macro signal scorecard
- `stress_signals.py` — Stress/crisis signal dashboard
- `tail_risk_studio.py` — Tail risk analysis
- `fed_forecaster.py` — Fed policy forecasting
- `signal_scorecard.py` — Signal performance scorecard
- `signal_audit.py` — Signal audit and validation
- `forecast_accuracy.py` — Forecast accuracy tracking
- `performance.py` — Portfolio performance module
- `trade_journal.py` — Trade journal
- `whale_buyers.py` — Whale/large buyer detection
- `current_events.py` — Current events / news context
- `export_hub.py` — Data export hub
- `alerts_settings.py` — Alert configuration UI

### services/
- `hmm_regime.py` — **Core**: GaussianHMM trained on FRED + VIX. Infers latent market regimes (Bull/Neutral/Stress/Late Cycle/Crisis). Stores brain in `data/hmm_brain.json`, history in `data/hmm_state_history.json`. `lookback_years` stored in brain JSON — must match live scoring
- `market_data.py` — Shared market data layer (yfinance batch fetch, z-scores, AssetSnapshot dataclass)
- `backtest_engine.py` — Backtesting engine for signal validation
- `sec_client.py` — SEC EDGAR API (filings, 13F, insider trades, CIK mapping)
- `claude_client.py` — Claude AI integration
- `congress_client.py` — Congress trading data
- `ibkr_client.py` — Interactive Brokers connection
- `trends_client.py` — Google Trends
- `scoring.py` — Signal scoring logic
- `indicators.py` — Technical indicators
- `wyckoff_engine.py` — Wyckoff phase detection engine
- `elliott_wave_engine.py` — Elliott Wave detection engine
- `signals_cache.py` — Signal caching layer
- `signal_quantifier.py` — Signal quantification
- `qir_history.py` — QIR run history persistence
- `tactical_history.py` — Tactical score history
- `portfolio_sizing.py` — Kelly criterion / position sizing
- `sector_rotation.py` — Sector rotation signals
- `turning_point.py` — Market turning point detection
- `whale_screener.py` — Whale activity screener
- `activism_screener.py` — Activist investor screener
- `alerts_service.py` — Alert delivery service
- `telegram_client.py` — Telegram notifications
- `stocktwits_client.py` — StockTwits sentiment
- `stress_client.py` — Stress signal data client
- `news_feed.py` — News feed aggregation
- `free_data.py` — Free data sources
- `fed_forecaster.py` — Fed forecasting service
- `forecast_tracker.py` — Forecast accuracy tracking
- `play_log.py` — Trade play logging

### utils/
- `session.py` — Streamlit session state helpers (ticker, narrative, IBKR status)
- `theme.py` — Dark theme colors (`COLORS` dict) and `apply_dark_layout()` for Plotly
- `components.py` — Shared UI components
- `signal_block.py` — Reusable signal block renderer
- `styles.py` — CSS/style constants
- `auth.py` — Authentication helpers
- `state_keys.py` — Session state key constants
- `ai_tier.py` — AI tier/model selection
- `alerts_config.py` — Alert configuration helpers
- `api_helpers.py` — API utility functions
- `watchlist.py` — Watchlist management
- `journal.py` — Journal utilities
- `options_history.py` — Options history helpers
- `debate_record.py` — Debate/thesis record

## CI% Crisis Detection System
The LL-anchored Crisis Intensity system is the core crisis signal:
- **Formula**: `CI% = abs(ll_zscore) / 0.467 * 100` (uncapped — COVID in-sample = 100%)
- **Zone 1** (CI < 22%): Normal — conviction signals suppressed
- **Zone 2** (CI 22–67%): Model Stress — signals shown as context
- **Zone 3** (CI ≥ 67%): Crisis Confirmed — 100% precision, 0 false alarms in 3,408 days
- **Zone 4** (CI > 100%): Beyond Training Range — purple, model seeing post-training extremes
- **Gate**: z < -0.30 = 67% CI (Volmageddon=76%, Fed Panic=96%, COVID=100%)
- **After retraining**: run `python ll_gate_backtest_live_brain.py` and update `0.467` anchor if COVID peak z changed

## Key Conventions
- All SEC requests use `SEC_HEADERS` with User-Agent and go through `_rate_limit()` (10 req/sec max)
- Heavy API calls are cached with `@st.cache_data(ttl=3600)` (1hr) or `ttl=86400` (24hr)
- Concurrent SEC fetches use `ThreadPoolExecutor(max_workers=5)` to stay under rate limits
- Plotly charts use `apply_dark_layout()` from `utils/theme.py`
- CIK-to-ticker mapping via `get_cik_ticker_map()` is the foundation for SEC lookups
- HMM brain `lookback_years` must match between training and live scoring (stored in `data/hmm_brain.json`)

## Running
```bash
streamlit run app.py
```

## Key Data Files
- `data/hmm_brain.json` — Trained HMM model (transmat, means, covars, baseline LL stats, lookback_years)
- `data/hmm_state_history.json` — Daily HMM state log (last 500 entries)
- `data/signals_cache.json` — Cached signal values
- `ll_gate_backtest_live_brain.py` — Backtest script (run after every brain retrain)
- `ll_gate_backtest_live_brain.json` — Latest backtest results

## Dependencies
See `requirements.txt`. Key: streamlit, plotly, pandas, requests, python-dotenv, ib_insync, hmmlearn, scipy.

