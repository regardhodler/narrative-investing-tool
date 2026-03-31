"""
Morning QIR headless runner — for GitHub Actions cron.

Mocks Streamlit (no UI), calls the same service functions QIR uses,
then writes data/signals_cache.json and updates the Gist.

Usage:
    python tools/morning_run.py
"""

# ── 1. Mock Streamlit BEFORE any project imports ─────────────────────────────
import json
import sys
import types

class _SessionState(dict):
    """dict that also supports attribute-style access (st.session_state.foo)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, val):
        self[key] = val


class _MockStreamlit(types.ModuleType):
    session_state = _SessionState()

    # Passthrough decorators ───────────────────────────────────────────────────
    @staticmethod
    def cache_data(*dargs, **dkwargs):
        def decorator(fn):
            return fn
        # Handle both @st.cache_data and @st.cache_data(ttl=…)
        if dargs and callable(dargs[0]):
            return dargs[0]
        return decorator

    @staticmethod
    def cache_resource(*dargs, **dkwargs):
        def decorator(fn):
            return fn
        if dargs and callable(dargs[0]):
            return dargs[0]
        return decorator

    # No-op UI calls ───────────────────────────────────────────────────────────
    @staticmethod
    def error(*a, **k): pass
    @staticmethod
    def warning(*a, **k): pass
    @staticmethod
    def info(*a, **k): pass
    @staticmethod
    def success(*a, **k): pass
    @staticmethod
    def write(*a, **k): pass
    @staticmethod
    def spinner(*a, **k):
        import contextlib
        return contextlib.nullcontext()

    # secrets — return empty dict so .get() calls don't crash
    class secrets:
        @staticmethod
        def get(key, default=None):
            return default
        def __getitem__(self, key):
            raise KeyError(key)


_mock_st = _MockStreamlit("streamlit")
sys.modules["streamlit"] = _mock_st
# Also stub sub-modules that some imports expect
sys.modules["streamlit.runtime"] = types.ModuleType("streamlit.runtime")
sys.modules["streamlit.runtime.scriptrunner"] = types.ModuleType("streamlit.runtime.scriptrunner")


# ── 2. Load environment ───────────────────────────────────────────────────────
import os
from pathlib import Path

_root = Path(__file__).parent.parent
_env = _root / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)
    print(f"[morning_run] Loaded .env from {_env}")
else:
    print("[morning_run] No .env file found — relying on environment variables")


# ── 3. Run the data-fetching pipeline ────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(_root))

print("[morning_run] Starting QIR data pipeline...")


def _run_regime():
    """Fetch regime + tactical context and store in mock session_state."""
    print("[morning_run] [1/3] Fetching regime context...")
    try:
        from datetime import datetime as _dt
        from modules.risk_regime import run_quick_regime
        _val = run_quick_regime(use_claude=False)
        if _val:
            _macro_ctx, _fred_data, _tac_data, _tac_text, _regime_ctx, _plays, _tier, _dq = _val
            _mock_st.session_state["_regime_context"]    = _regime_ctx
            _mock_st.session_state["_regime_context_ts"] = _dt.now()
            _mock_st.session_state["_rp_plays_result"]   = _plays
            _mock_st.session_state["_rp_plays_last_tier"] = _tier
            if _tac_data:
                _mock_st.session_state["_tactical_context"]    = _tac_data
                _mock_st.session_state["_tactical_context_ts"] = _dt.now()
            if _tac_text:
                _mock_st.session_state["_tactical_analysis"]    = _tac_text
                _mock_st.session_state["_tactical_analysis_ts"] = _dt.now()
            if _dq:
                _mock_st.session_state["_data_quality"]    = _dq
                _mock_st.session_state["_data_quality_ts"] = _dt.now()
            regime = _regime_ctx.get("regime", "unknown")
            score  = _regime_ctx.get("score", 0.0)
            print(f"[morning_run]   Regime: {regime} (score={score:+.2f})")
            return True
        print("[morning_run]   WARNING: run_quick_regime returned nothing — check FRED_API_KEY secret")
        return False
    except Exception as e:
        print(f"[morning_run]   ERROR in regime: {e}")
        import traceback; traceback.print_exc()
        return False


def _run_digest():
    """Fetch current-events digest and store in mock session_state."""
    print("[morning_run] [2/3] Fetching current events digest...")
    try:
        from modules.current_events import run_quick_digest
        _val = run_quick_digest(use_claude=False)
        if _val:
            for _k, _v in _val.items():
                _mock_st.session_state[_k] = _v
            digest = _val.get("_current_events_digest", "")
            preview = digest[:120].replace("\n", " ") if digest else "(empty)"
            print(f"[morning_run]   Digest: {preview}...")
            return True
        print("[morning_run]   WARNING: digest empty — check GROQ_API_KEY / NEWSAPI_KEY secrets")
        return False
    except Exception as e:
        print(f"[morning_run]   ERROR in digest: {e}")
        import traceback; traceback.print_exc()
        return False


def _run_fed():
    """Fetch ZQ / FedWatch probabilities and store in mock session_state."""
    print("[morning_run] [3/3] Fetching Fed rate-path probabilities...")
    try:
        from services.fed_forecaster import fetch_zq_probabilities
        probs = fetch_zq_probabilities()
        if probs:
            _mock_st.session_state["_rate_path_probs"] = probs
            src = probs[0].get("source", "?") if probs else "none"
            top = max(probs, key=lambda x: x["prob"])
            print(f"[morning_run]   Rate-path source={src}  top={top['scenario']} ({top['prob']:.1%})")
        return bool(probs)
    except Exception as e:
        print(f"[morning_run]   ERROR in fed: {e}")
        import traceback; traceback.print_exc()
        return False


_run_regime()
_run_digest()
_run_fed()


# ── 4. Persist to file + Gist ─────────────────────────────────────────────────
print("[morning_run] Saving signals cache...")
try:
    # Override the st import inside signals_cache so it sees our mock
    from services import signals_cache as _sc
    _sc.st = _mock_st  # point the module at our mock

    # save_signals reads from st.session_state — which IS our mock dict
    _sc.save_signals()

    cache_path = _root / "data" / "signals_cache.json"
    if cache_path.exists():
        size = cache_path.stat().st_size
        print(f"[morning_run] signals_cache.json written ({size:,} bytes)")
    else:
        print("[morning_run] WARNING: signals_cache.json not found after save")

    gist_id = os.getenv("SIGNALS_GIST_ID", "")
    if gist_id:
        print(f"[morning_run] Gist updated: {gist_id[:8]}…")
    else:
        print("[morning_run] No SIGNALS_GIST_ID set — Gist not updated")

except Exception as e:
    print(f"[morning_run] ERROR saving signals: {e}")
    import traceback
    traceback.print_exc()

# ── 5. Write AI export files ──────────────────────────────────────────────────
print("[morning_run] Writing AI export files...")
try:
    from datetime import datetime as _dt
    _now_iso = _dt.now().strftime("%Y-%m-%dT%H:%M:%S")
    _ss = _mock_st.session_state

    # ── regard_brief.json — personal morning brief ────────────────────────────
    _rc  = _ss.get("_regime_context") or {}
    _tac = _ss.get("_tactical_context") or {}
    _probs = _ss.get("_rate_path_probs") or []
    _top_rate = max(_probs, key=lambda x: x["prob"]) if _probs else {}
    _pattern_name = None
    try:
        from modules.quick_run import _classify_signals
        _cls = _classify_signals(_rc, _tac, _ss.get("_options_flow_context") or {})
        _pattern_name = _cls.get("pattern")
        _pattern_interp = _cls.get("interpretation", "")
        _pattern_buy = _cls.get("buy_tier", "")
        _pattern_short = _cls.get("short_tier", "")
    except Exception:
        _pattern_interp = _pattern_buy = _pattern_short = ""

    # Load open positions from trade_journal.json
    _positions = []
    try:
        _tj_path = _root / "data" / "trade_journal.json"
        if _tj_path.exists():
            _trades = json.load(open(_tj_path, encoding="utf-8"))
            _positions = [
                {
                    "ticker":       t["ticker"],
                    "direction":    t["direction"],
                    "entry_price":  t["entry_price"],
                    "entry_date":   t["entry_date"],
                    "position_size": t.get("position_size"),
                    "thesis":       t.get("thesis", ""),
                }
                for t in _trades if t.get("status") == "open"
            ]
    except Exception:
        pass

    _brief = {
        "source":       "Regard Terminals — Morning QIR Run",
        "generated_at": _now_iso,
        "regime": {
            "label":          _rc.get("regime", ""),
            "score":          round(_rc.get("score", 0.0), 3),
            "quadrant":       _rc.get("quadrant", ""),
            "signal_summary": _rc.get("signal_summary", ""),
        },
        "tactical": {
            "score":       _tac.get("tactical_score"),
            "label":       _tac.get("label", ""),
            "action_bias": _tac.get("action_bias", ""),
        },
        "qir_pattern": {
            "name":           _pattern_name,
            "interpretation": _pattern_interp,
            "buy_tier":       _pattern_buy,
            "short_tier":     _pattern_short,
        },
        "rate_path": {
            "top_scenario": _top_rate.get("scenario", ""),
            "top_prob":     round(_top_rate.get("prob", 0.0), 3),
            "source":       _top_rate.get("source", ""),
        },
        "current_events_digest": _ss.get("_current_events_digest", ""),
        "open_positions":        _positions,
        "earnings_risk":         _ss.get("_qir_earnings_risk") or [],
    }

    _brief_path = _root / "data" / "regard_brief.json"
    with open(_brief_path, "w", encoding="utf-8") as _f:
        json.dump(_brief, _f, indent=2, ensure_ascii=False, default=str)
    print(f"[morning_run] regard_brief.json written ({_brief_path.stat().st_size:,} bytes)")

    # ── regard_system.json — system showcase, no personal data ───────────────
    _system = {
        "system":             "Regard Terminals",
        "description":        "Multi-module investment intelligence dashboard. Regime-aware, AI-powered, built for independent investors.",
        "generated_at":       _now_iso,
        "current_regime":     _rc.get("regime", ""),
        "current_qir_pattern": _pattern_name,
        "modules": [
            {"id": 0, "name": "Risk Regime",       "description": "Cross-asset risk-on/off indicator. 17+ z-score signals, daily history."},
            {"id": 1, "name": "Narrative Discovery","description": "AI-grouped ticker discovery by narrative themes. Auto-refreshes every 4h."},
            {"id": 2, "name": "Narrative Pulse",    "description": "Sentiment tracking across narratives."},
            {"id": 3, "name": "EDGAR Scanner",      "description": "SEC filing search and digest."},
            {"id": 4, "name": "Institutional",      "description": "13F institutional holdings analysis."},
            {"id": 5, "name": "Insider & Congress", "description": "Form 4 insider trades and Congress trading activity."},
            {"id": 6, "name": "Options Activity",   "description": "Options flow analysis with unusual activity sentiment."},
            {"id": 7, "name": "Valuation",          "description": "AI rating + 2-stage levered DCF with CAPM discount rate, sector profiles, Kelly sizing."},
        ],
        "qir_patterns": [
            {"name": "BULLISH_CONFIRMATION",    "description": "All three timing layers aligned bullish — highest-conviction long entry."},
            {"name": "BEARISH_CONFIRMATION",    "description": "All three layers aligned bearish — highest-conviction short or cash environment."},
            {"name": "PULLBACK_IN_UPTREND",     "description": "Regime + Options Flow confirm bull trend — Tactical dip is a buy-the-dip setup."},
            {"name": "OPTIONS_FLOW_DIVERGENCE", "description": "Regime + Tactical bullish but options crowd hedging — smart money buying protection."},
            {"name": "BEAR_MARKET_BOUNCE",      "description": "Regime bearish but Tactical + Options show short-term bounce — trade the bounce, not the trend."},
            {"name": "LATE_CYCLE_SQUEEZE",      "description": "Regime bearish, Tactical neutral, Options bullish — gamma squeeze risk, stay defensive."},
            {"name": "GENUINE_UNCERTAINTY",     "description": "No clear alignment across layers — reduce size, wait for confluence."},
        ],
        "signal_layers": {
            "regime":       "Structural macro (weeks–months): FRED data, z-scores, yield curve, credit spreads, VIX, M2, Sahm rule.",
            "tactical":     "Medium-term (days–weeks): price momentum, breadth, RSI, moving average crossovers.",
            "options_flow": "Short-term (hours–days): put/call ratio, unusual activity, gamma exposure.",
        },
        "dcf_methodology": (
            "2-stage levered DCF. CAPM discount rate using live market beta + Damodaran sector unlevered beta "
            "re-levered with D/E ratio. Regime-adjusted ERP: 5.0% risk-on, 5.5% neutral, 6.0% risk-off. "
            "Bear/Base/Bull scenarios ±35% growth adjustment. "
            "Sensitivity grid: WACC ±100bps × terminal growth ±50bps."
        ),
        "ai_engines": ["Groq LLaMA 3.3 70B (free)", "Grok 4.1 via xAI", "Claude Sonnet 4.6 via Anthropic"],
        "data_sources": ["yfinance", "FRED", "SEC EDGAR", "CME FedWatch", "NewsAPI RSS", "Google Trends"],
    }

    _sys_path = _root / "data" / "regard_system.json"
    with open(_sys_path, "w", encoding="utf-8") as _f:
        json.dump(_system, _f, indent=2, ensure_ascii=False, default=str)
    print(f"[morning_run] regard_system.json written ({_sys_path.stat().st_size:,} bytes)")

except Exception as e:
    print(f"[morning_run] ERROR writing AI exports: {e}")
    import traceback
    traceback.print_exc()

print("[morning_run] Done.")
