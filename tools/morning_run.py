"""
Morning QIR headless runner — for GitHub Actions cron.

Mocks Streamlit (no UI), calls the same service functions QIR uses,
then writes data/signals_cache.json and updates the Gist.

Usage:
    python tools/morning_run.py
"""

# ── 1. Mock Streamlit BEFORE any project imports ─────────────────────────────
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
        from modules.risk_regime import run_quick_regime
        ok = run_quick_regime(use_claude=False)
        rc = _mock_st.session_state.get("_regime_context") or {}
        regime = rc.get("regime", "unknown")
        score  = rc.get("score", 0.0)
        print(f"[morning_run]   Regime: {regime} (score={score:+.2f})")
        return ok
    except Exception as e:
        print(f"[morning_run]   ERROR in regime: {e}")
        return False


def _run_digest():
    """Fetch current-events digest and store in mock session_state."""
    print("[morning_run] [2/3] Fetching current events digest...")
    try:
        from modules.current_events import run_quick_digest
        ok = run_quick_digest(use_claude=False)
        digest = _mock_st.session_state.get("_current_events_digest") or ""
        preview = digest[:120].replace("\n", " ") if digest else "(empty)"
        print(f"[morning_run]   Digest: {preview}...")
        return ok
    except Exception as e:
        print(f"[morning_run]   ERROR in digest: {e}")
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

print("[morning_run] Done.")
