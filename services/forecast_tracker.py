"""Forecast Accuracy Tracker service.

Logs AI-generated predictions (valuation ratings, regime calls, squeeze theses, etc.)
and evaluates their outcomes against actual price moves or user-marked results.
All data persists via the signals_cache Gist mechanism under _forecast_log.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


# ── Correctness thresholds ─────────────────────────────────────────────────────
_THRESHOLDS = {
    "valuation": {
        "Strong Buy":  ("up",   0.02),   # needs +2% to be correct
        "Buy":         ("up",   0.02),
        "Hold":        ("flat", 0.05),   # correct if within ±5%
        "Sell":        ("down", 0.02),
        "Strong Sell": ("down", 0.02),
    },
    "squeeze": {
        "default": ("up", 0.05),         # squeeze thesis → needs +5%
    },
}

MODEL_LABELS = {
    "llama-3.3-70b-versatile": "Groq Llama 3.3",
    "grok-4-1-fast-reasoning": "xAI Grok",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_log() -> list[dict]:
    return st.session_state.get("_forecast_log") or []


def _set_log(log: list[dict]) -> None:
    st.session_state["_forecast_log"] = log
    st.session_state["_forecast_log_ts"] = datetime.now()


def _fetch_price(ticker: str) -> Optional[float]:
    """Return current close price for ticker via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price or info.previous_close)
    except Exception:
        return None


def _auto_outcome(entry: dict, current_price: float) -> tuple[str, float]:
    """Determine outcome and return_pct given current price."""
    price_at = entry.get("price_at_forecast")
    if not price_at or price_at <= 0:
        return "unknown", 0.0

    ret = (current_price - price_at) / price_at  # decimal
    sig_type = entry.get("signal_type", "")
    prediction = entry.get("prediction", "")

    if sig_type == "valuation":
        direction, threshold = _THRESHOLDS["valuation"].get(
            prediction, ("up", 0.02)
        )
    elif sig_type == "squeeze":
        direction, threshold = _THRESHOLDS["squeeze"]["default"]
    else:
        return "pending", ret * 100

    if direction == "up":
        outcome = "correct" if ret >= threshold else "incorrect"
    elif direction == "down":
        outcome = "correct" if ret <= -threshold else "incorrect"
    else:  # flat / hold
        outcome = "correct" if abs(ret) <= threshold else "incorrect"

    return outcome, round(ret * 100, 2)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_forecast(
    signal_type: str,
    prediction: str,
    confidence: int,
    summary: str,
    model: str,
    ticker: Optional[str] = None,
    target_price: Optional[float] = None,
    horizon_days: int = 30,
    notes: str = "",
) -> str:
    """Log a new forecast. Returns the forecast ID."""
    price_at = _fetch_price(ticker) if ticker else None

    entry = {
        "id": str(uuid.uuid4())[:8],
        "signal_type": signal_type,
        "ticker": ticker or "",
        "prediction": prediction,
        "confidence": confidence,
        "summary": summary,
        "price_at_forecast": price_at,
        "target_price": target_price,
        "horizon_days": horizon_days,
        "timestamp": datetime.now(),
        "model": MODEL_LABELS.get(model, model),
        "evaluated_at": None,
        "price_at_eval": None,
        "outcome": None,
        "return_pct": None,
        "notes": notes,
    }

    log = _get_log()
    log.insert(0, entry)
    _set_log(log)
    return entry["id"]


def evaluate_pending(force: bool = False) -> int:
    """Auto-evaluate all past-horizon forecasts. Returns count updated."""
    log = _get_log()
    now = datetime.now()
    updated = 0

    for entry in log:
        if entry.get("outcome") not in (None, "pending"):
            continue

        ts = entry.get("timestamp")
        if not ts:
            continue
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        horizon = entry.get("horizon_days", 30)
        eval_due = ts + timedelta(days=horizon)

        if not force and now < eval_due:
            continue

        sig_type = entry.get("signal_type", "")
        ticker = entry.get("ticker", "")

        if sig_type in ("valuation", "squeeze") and ticker:
            price = _fetch_price(ticker)
            if price is None:
                continue
            outcome, ret = _auto_outcome(entry, price)
            entry["outcome"] = outcome
            entry["return_pct"] = ret
            entry["price_at_eval"] = price
            entry["evaluated_at"] = now
            updated += 1
        # regime/fed/manual require user marking — skip auto-eval

    if updated:
        _set_log(log)
    return updated


def mark_outcome(forecast_id: str, outcome: str, notes: str = "") -> bool:
    """Manually mark a forecast correct/incorrect. Returns True if found."""
    log = _get_log()
    for entry in log:
        if entry.get("id") == forecast_id:
            entry["outcome"] = outcome
            entry["evaluated_at"] = datetime.now()
            if notes:
                entry["notes"] = notes
            _set_log(log)
            return True
    return False


def delete_forecast(forecast_id: str) -> bool:
    """Remove a forecast from the log. Returns True if found."""
    log = _get_log()
    new_log = [e for e in log if e.get("id") != forecast_id]
    if len(new_log) < len(log):
        _set_log(new_log)
        return True
    return False


def get_stats() -> dict:
    """Return accuracy statistics across all resolved forecasts."""
    log = _get_log()
    resolved = [e for e in log if e.get("outcome") in ("correct", "incorrect")]
    pending = [e for e in log if e.get("outcome") in (None, "pending")]
    expired = [e for e in log if e.get("outcome") == "expired"]

    total_resolved = len(resolved)
    correct = sum(1 for e in resolved if e["outcome"] == "correct")
    accuracy = round(correct / total_resolved * 100, 1) if total_resolved else 0.0

    # By signal type
    by_type: dict[str, dict] = {}
    for e in resolved:
        st_ = e.get("signal_type", "unknown")
        if st_ not in by_type:
            by_type[st_] = {"correct": 0, "total": 0}
        by_type[st_]["total"] += 1
        if e["outcome"] == "correct":
            by_type[st_]["correct"] += 1
    for st_, d in by_type.items():
        d["accuracy"] = round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0.0

    # By model
    by_model: dict[str, dict] = {}
    for e in resolved:
        m = e.get("model", "Unknown")
        if m not in by_model:
            by_model[m] = {"correct": 0, "total": 0}
        by_model[m]["total"] += 1
        if e["outcome"] == "correct":
            by_model[m]["correct"] += 1
    for m, d in by_model.items():
        d["accuracy"] = round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0.0

    # Confidence calibration buckets (0-9, 10-19, …, 90-100)
    calibration: list[dict] = []
    for low in range(0, 100, 10):
        high = low + 9
        bucket = [e for e in resolved if low <= e.get("confidence", 0) <= high]
        if bucket:
            acc = round(sum(1 for e in bucket if e["outcome"] == "correct") / len(bucket) * 100, 1)
            calibration.append({"label": f"{low}-{high}", "accuracy": acc, "n": len(bucket)})

    # Average return on correct calls
    correct_returns = [e["return_pct"] for e in resolved if e["outcome"] == "correct" and e.get("return_pct") is not None]
    avg_return_correct = round(sum(correct_returns) / len(correct_returns), 2) if correct_returns else None

    return {
        "total": len(log),
        "total_resolved": total_resolved,
        "pending": len(pending),
        "expired": len(expired),
        "correct": correct,
        "accuracy": accuracy,
        "by_type": by_type,
        "by_model": by_model,
        "calibration": calibration,
        "avg_return_correct": avg_return_correct,
    }
