"""Shared API call utilities: error capture decorator and retry helper.

Usage
-----
Error capture (surfaces failures to the scorecard ⚠ badge):

    from utils.api_helpers import capture_api_error

    @capture_api_error("FRED")
    def fetch_fred_series(...):
        ...  # exceptions are caught, stored in _api_errors, function returns None

Retry (transient 429/5xx with exponential backoff):

    from utils.api_helpers import post_with_retry

    resp = post_with_retry(url, headers=headers, payload=payload, timeout=30)
"""

import functools
import time

import requests


# ---------------------------------------------------------------------------
# Error capture decorator
# ---------------------------------------------------------------------------

def capture_api_error(service_name: str, fallback=None):
    """Decorator: catches exceptions, stores message in session_state._api_errors,
    returns fallback value. Works with or without Streamlit (no-ops outside a session).

    Args:
        service_name: Human-readable label shown in the ⚠ scorecard badge.
        fallback: Value to return on failure. Can be a callable (called with no args).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                _store_error(service_name, e)
                return fallback() if callable(fallback) else fallback
        return wrapper
    return decorator


def _store_error(service_name: str, exc: Exception) -> None:
    """Write error to session_state._api_errors if Streamlit is active."""
    try:
        import streamlit as st
        errs = st.session_state.setdefault("_api_errors", {})
        errs[service_name] = str(exc)[:140]
    except Exception:
        pass  # Outside a Streamlit session — fail silently


def clear_api_errors() -> None:
    """Clear all stored API errors (call at start of QIR run)."""
    try:
        import streamlit as st
        st.session_state.pop("_api_errors", None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Retry helper (extracted from fed_forecaster._groq_post_with_retry)
# ---------------------------------------------------------------------------

def post_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = 30,
    max_retries: int = 2,
) -> requests.Response:
    """POST with exponential backoff on 429 / 5xx responses.

    Waits 2s then 4s between retries (doubles each attempt).
    Raises requests.HTTPError on final failure — callers should catch.

    Args:
        url: Full endpoint URL.
        headers: HTTP headers dict (Authorization, Content-Type, etc.).
        payload: JSON-serializable request body.
        timeout: Per-attempt socket timeout in seconds.
        max_retries: Total attempts before giving up (default 2 = 1 retry).
    """
    delay = 2
    last_resp = None
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        last_resp = resp
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        return resp
    # Exhausted retries — raise on the last response
    if last_resp is not None:
        last_resp.raise_for_status()
    raise RuntimeError(f"post_with_retry: no response after {max_retries} attempts")
