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
    "llama-3.3-70b-versatile":     "Groq Llama 3.3",
    "grok-4-1-fast-reasoning":     "xAI Grok",
    "claude-sonnet-4-6":           "Claude Sonnet 4.6",
    "claude-haiku-4-5-20251001":   "Claude Haiku 4.5",
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


def _fetch_spy_price() -> Optional[float]:
    return _fetch_price("SPY")


# ── ATR trailing stop engine ───────────────────────────────────────────────────

_ATR_MULT_STOP   = 2.0   # trailing stop = high_watermark - N×ATR
_ATR_MULT_TARGET = 3.0   # profit target = entry + N×ATR  (1.5:1 R:R)
_ATR_PERIOD      = 14

def _fetch_atr_and_history(ticker: str, since: datetime) -> Optional[dict]:
    """Fetch ATR(14) and daily OHLC history since log date.

    Returns dict with:
        atr          float   ATR at log date (used for stop/target levels)
        highs        list    daily highs since log date
        lows         list    daily lows since log date
        closes       list    daily closes since log date
        dates        list    date strings
    """
    try:
        import yfinance as yf
        import pandas as pd

        ticker_obj = yf.Ticker(ticker)
        # Fetch enough history for ATR(14) + all days since log
        lookback_start = since - timedelta(days=_ATR_PERIOD * 2 + 10)
        hist = ticker_obj.history(start=lookback_start.strftime("%Y-%m-%d"), auto_adjust=True)
        if hist is None or len(hist) < _ATR_PERIOD + 2:
            return None

        # Compute ATR(14) using True Range
        high = hist["High"]
        low  = hist["Low"]
        close_prev = hist["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low  - close_prev).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(_ATR_PERIOD).mean()

        # ATR at the log date (last value before/at since)
        pre_log = atr_series[atr_series.index <= pd.Timestamp(since).tz_localize(atr_series.index.tzinfo)]
        if pre_log.empty:
            pre_log = atr_series
        atr_at_log = float(pre_log.iloc[-1])

        # History AFTER log date
        since_ts = pd.Timestamp(since).tz_localize(hist.index.tzinfo) if hist.index.tzinfo else pd.Timestamp(since)
        post = hist[hist.index > since_ts]
        if post.empty:
            return {"atr": atr_at_log, "highs": [], "lows": [], "closes": [], "dates": []}

        return {
            "atr":    atr_at_log,
            "highs":  post["High"].tolist(),
            "lows":   post["Low"].tolist(),
            "closes": post["Close"].tolist(),
            "dates":  [str(d)[:10] for d in post.index],
        }
    except Exception:
        return None


def _evaluate_atr_trailing(entry: dict) -> Optional[dict]:
    """Walk price history with ATR trailing stop + profit target.

    Returns dict with outcome, return_pct, exit_date, exit_price, exit_reason
    or None if insufficient data.
    """
    ticker = entry.get("ticker", "")
    if not ticker:
        return None

    ts = entry.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if not ts:
        return None

    price_at = entry.get("price_at_forecast")
    if not price_at or price_at <= 0:
        return None

    data = _fetch_atr_and_history(ticker, ts)
    if not data or not data["closes"]:
        return None

    atr = data["atr"]
    if atr <= 0:
        return None

    prediction = entry.get("prediction", "")
    sig_type    = entry.get("signal_type", "")
    is_short    = prediction in ("Sell", "Strong Sell")

    stop_dist   = _ATR_MULT_STOP   * atr
    target_dist = _ATR_MULT_TARGET * atr

    if is_short:
        stop_level   = price_at + stop_dist    # initial stop above entry
        target_level = price_at - target_dist  # target below entry
    else:
        stop_level   = price_at - stop_dist    # initial stop below entry
        target_level = price_at + target_dist  # target above entry

    watermark = price_at  # tracks best price in our direction
    trailing_stop = stop_level

    highs, lows, closes, dates = data["highs"], data["lows"], data["closes"], data["dates"]

    for i, (h, l, c, d) in enumerate(zip(highs, lows, closes, dates)):
        if is_short:
            # Update watermark (lowest low)
            if l < watermark:
                watermark = l
                trailing_stop = watermark + stop_dist

            # Check profit target (intraday low)
            if l <= target_level:
                ret = round((price_at - target_level) / price_at * 100, 2)
                return {"outcome": "correct", "return_pct": ret, "exit_date": d,
                        "exit_price": target_level, "exit_reason": "profit_target",
                        "atr_at_log": atr, "stop_at_log": stop_level, "target_at_log": target_level}

            # Check trailing stop (intraday high)
            if h >= trailing_stop:
                ret = round((price_at - trailing_stop) / price_at * 100, 2)
                outcome = "correct" if ret > 0 else "incorrect"
                return {"outcome": outcome, "return_pct": ret, "exit_date": d,
                        "exit_price": trailing_stop, "exit_reason": "trailing_stop",
                        "atr_at_log": atr, "stop_at_log": stop_level, "target_at_log": target_level}
        else:
            # Update watermark (highest high)
            if h > watermark:
                watermark = h
                trailing_stop = watermark - stop_dist

            # Check profit target (intraday high)
            if h >= target_level:
                ret = round((target_level - price_at) / price_at * 100, 2)
                return {"outcome": "correct", "return_pct": ret, "exit_date": d,
                        "exit_price": target_level, "exit_reason": "profit_target",
                        "atr_at_log": atr, "stop_at_log": stop_level, "target_at_log": target_level}

            # Check trailing stop (intraday low)
            if l <= trailing_stop:
                ret = round((trailing_stop - price_at) / price_at * 100, 2)
                outcome = "correct" if ret > 0 else "incorrect"
                return {"outcome": outcome, "return_pct": ret, "exit_date": d,
                        "exit_price": trailing_stop, "exit_reason": "trailing_stop",
                        "atr_at_log": atr, "stop_at_log": stop_level, "target_at_log": target_level}

    # Neither stop nor target triggered — trade is still open
    # Return None so the caller keeps the entry as pending
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

def backtest_atr(ticker: str, date_str: str, direction: str, confidence: int = 70) -> Optional[dict]:
    """Simulate an ATR trailing stop trade on any ticker from any past date.

    Args:
        ticker:     e.g. "SPY"
        date_str:   ISO format "YYYY-MM-DD"
        direction:  "Buy" or "Sell"
        confidence: 0-100

    Returns dict with:
        outcome, return_pct, exit_date, exit_price, exit_reason,
        atr_at_log, stop_at_log, target_at_log,
        price_at_entry, highs, lows, closes, dates   (full history for charting)
    or None on data failure.
    """
    try:
        ts = datetime.fromisoformat(date_str)
    except ValueError:
        return None

    import yfinance as yf
    import pandas as pd

    # Fetch entry price (close on that date)
    lookback = ts - timedelta(days=_ATR_PERIOD * 2 + 10)
    hist = yf.Ticker(ticker).history(start=lookback.strftime("%Y-%m-%d"), auto_adjust=True)
    if hist is None or hist.empty:
        return None

    # Find entry price = close on or after date_str
    since_ts = pd.Timestamp(ts).tz_localize(hist.index.tzinfo) if hist.index.tzinfo else pd.Timestamp(ts)
    entry_row = hist[hist.index >= since_ts]
    if entry_row.empty:
        return None
    price_at = float(entry_row["Close"].iloc[0])

    synthetic = {
        "ticker":            ticker,
        "timestamp":         ts.isoformat(),
        "price_at_forecast": price_at,
        "prediction":        direction,
        "signal_type":       "valuation",
        "confidence":        confidence,
    }

    result = _evaluate_atr_trailing(synthetic)
    if result is None:
        # Trade still open — return partial with full history
        data = _fetch_atr_and_history(ticker, ts)
        if not data:
            return None
        return {
            "outcome":       "open",
            "return_pct":    None,
            "exit_date":     None,
            "exit_price":    None,
            "exit_reason":   "still_open",
            "atr_at_log":    data.get("atr"),
            "stop_at_log":   price_at - _ATR_MULT_STOP * data["atr"] if direction == "Buy" else price_at + _ATR_MULT_STOP * data["atr"],
            "target_at_log": price_at + _ATR_MULT_TARGET * data["atr"] if direction == "Buy" else price_at - _ATR_MULT_TARGET * data["atr"],
            "price_at_entry": price_at,
            "highs":  data["highs"],
            "lows":   data["lows"],
            "closes": data["closes"],
            "dates":  data["dates"],
        }

    data = _fetch_atr_and_history(ticker, ts)
    result["price_at_entry"] = price_at
    result["highs"]   = data["highs"]  if data else []
    result["lows"]    = data["lows"]   if data else []
    result["closes"]  = data["closes"] if data else []
    result["dates"]   = data["dates"]  if data else []
    return result


def _capture_market_context() -> dict:
    """Snapshot key market signals from session state at time of logging."""
    ctx: dict = {}
    regime = st.session_state.get("_regime_context")
    if isinstance(regime, dict):
        ctx["regime"] = regime.get("regime", "")
        ctx["quadrant"] = regime.get("quadrant", "")
        ctx["regime_score"] = regime.get("score", None)

    fg = st.session_state.get("_fear_greed")
    if isinstance(fg, dict):
        ctx["fear_greed_score"] = fg.get("score")
        ctx["fear_greed_label"] = fg.get("label")

    vix = st.session_state.get("_vix_curve")
    if isinstance(vix, dict):
        ctx["vix_spot"] = vix.get("vix_spot")
        ctx["vix_structure"] = vix.get("structure")

    rp = st.session_state.get("_dominant_rate_path")
    if isinstance(rp, dict):
        ctx["fed_path"] = rp.get("scenario")
        ctx["fed_path_prob"] = rp.get("prob_pct")

    return ctx


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
    spy_price_at = _fetch_spy_price()

    # ATR-based stop and target levels at log time
    atr_at_log    = None
    stop_at_log   = None
    target_at_log = None
    if ticker and price_at:
        try:
            import yfinance as yf, pandas as pd
            _h = yf.Ticker(ticker).history(period=f"{_ATR_PERIOD * 2 + 10}d", auto_adjust=True)
            if _h is not None and len(_h) >= _ATR_PERIOD + 2:
                _tr = pd.concat([
                    _h["High"] - _h["Low"],
                    (_h["High"] - _h["Close"].shift(1)).abs(),
                    (_h["Low"]  - _h["Close"].shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr_at_log = round(float(_tr.rolling(_ATR_PERIOD).mean().iloc[-1]), 4)
                is_short = prediction in ("Sell", "Strong Sell")
                if is_short:
                    stop_at_log   = round(price_at + _ATR_MULT_STOP   * atr_at_log, 4)
                    target_at_log = round(price_at - _ATR_MULT_TARGET * atr_at_log, 4)
                else:
                    stop_at_log   = round(price_at - _ATR_MULT_STOP   * atr_at_log, 4)
                    target_at_log = round(price_at + _ATR_MULT_TARGET * atr_at_log, 4)
        except Exception:
            pass

    entry = {
        "id": str(uuid.uuid4())[:8],
        "signal_type": signal_type,
        "ticker": ticker or "",
        "prediction": prediction,
        "confidence": confidence,
        "summary": summary,
        "price_at_forecast": price_at,
        "spy_price_at_forecast": spy_price_at,
        "target_price": target_price,
        "atr_at_log": atr_at_log,
        "stop_at_log": stop_at_log,
        "target_at_log": target_at_log,
        "horizon_days": horizon_days,
        "timestamp": datetime.now(),
        "model": MODEL_LABELS.get(model, model),
        "evaluated_at": None,
        "price_at_eval": None,
        "spy_price_at_eval": None,
        "outcome": None,
        "return_pct": None,
        "spy_return_pct": None,
        "alpha_pct": None,
        "exit_reason": None,
        "exit_date": None,
        "notes": notes,
        "market_context": _capture_market_context(),
    }

    log = _get_log()
    log.insert(0, entry)
    _set_log(log)
    return entry["id"]


def evaluate_pending(force: bool = False) -> int:
    """Auto-evaluate all past-horizon forecasts using ATR trailing stop/target engine.

    For valuation/squeeze with a ticker:
      - Walks full price history since log date
      - Trailing stop: high_watermark - 2×ATR (longs) or low_watermark + 2×ATR (shorts)
      - Profit target: entry + 3×ATR (longs) or entry - 3×ATR (shorts)
      - If neither triggered, evaluates at horizon end price

    Regime/fed/manual still evaluate at horizon end (no price data).
    """
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

        sig_type = entry.get("signal_type", "")
        ticker   = entry.get("ticker", "")

        if sig_type in ("valuation", "squeeze") and ticker:
            # ATR trailing stop/target — only exit when stop or target fires
            days_elapsed = (now - ts).days
            if not force and days_elapsed < 1:
                continue  # need at least 1 trading day

            atr_result = _evaluate_atr_trailing(entry)
            if atr_result is None:
                # No trigger yet — keep pending, update ATR levels if newly available
                continue

            # Stop or target triggered — finalize
            entry["outcome"]       = atr_result["outcome"]
            entry["return_pct"]    = atr_result["return_pct"]
            entry["price_at_eval"] = atr_result["exit_price"]
            entry["exit_reason"]   = atr_result["exit_reason"]
            entry["exit_date"]     = atr_result["exit_date"]
            if atr_result.get("atr_at_log") and not entry.get("atr_at_log"):
                entry["atr_at_log"]    = atr_result["atr_at_log"]
                entry["stop_at_log"]   = atr_result["stop_at_log"]
                entry["target_at_log"] = atr_result["target_at_log"]

            entry["evaluated_at"] = now

            # SPY benchmark — compute alpha
            spy_now = _fetch_spy_price()
            spy_at  = entry.get("spy_price_at_forecast")
            if spy_now and spy_at and spy_at > 0:
                spy_ret = round((spy_now - spy_at) / spy_at * 100, 2)
                entry["spy_price_at_eval"] = spy_now
                entry["spy_return_pct"]    = spy_ret
                prediction = entry.get("prediction", "")
                if prediction in ("Sell", "Strong Sell"):
                    entry["alpha_pct"] = round(-(entry["return_pct"] or 0) - spy_ret, 2)
                else:
                    entry["alpha_pct"] = round((entry["return_pct"] or 0) - spy_ret, 2)

            updated += 1
        elif sig_type == "regime":
            # Auto-eval: compare predicted quadrant vs current quadrant
            ctx = entry.get("market_context", {})
            predicted_quadrant = ctx.get("quadrant") or entry.get("prediction", "")
            current_regime = st.session_state.get("_regime_context") or {}
            current_quadrant = current_regime.get("quadrant", "")
            if current_quadrant:
                outcome = "correct" if predicted_quadrant.lower() in current_quadrant.lower() or current_quadrant.lower() in predicted_quadrant.lower() else "incorrect"
                entry["outcome"] = outcome
                entry["evaluated_at"] = now
                updated += 1
        elif sig_type == "fed":
            # Auto-eval: compare predicted scenario vs current dominant path
            ctx = entry.get("market_context", {})
            predicted_path = ctx.get("fed_path") or entry.get("prediction", "")
            current_rp = st.session_state.get("_dominant_rate_path") or {}
            current_path = current_rp.get("scenario", "")
            if current_path:
                outcome = "correct" if predicted_path.lower() == current_path.lower() else "incorrect"
                entry["outcome"] = outcome
                entry["evaluated_at"] = now
                updated += 1
        elif not ticker and sig_type not in ("regime", "fed", "manual"):
            # Can't auto-eval without a ticker — mark expired
            entry["outcome"] = "expired"
            entry["evaluated_at"] = now
            updated += 1

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

    # Streak tracking (chronological order — oldest first)
    chrono = sorted(resolved, key=lambda e: e.get("timestamp") or "", reverse=False)
    outcomes_seq = [e["outcome"] for e in chrono]
    current_streak = 0
    current_streak_type = None
    best_correct = 0
    worst_incorrect = 0
    tmp = 0
    tmp_type = None
    for o in outcomes_seq:
        if o == tmp_type:
            tmp += 1
        else:
            tmp = 1
            tmp_type = o
        if tmp_type == "correct":
            best_correct = max(best_correct, tmp)
        else:
            worst_incorrect = max(worst_incorrect, tmp)
    # current streak = last N in a row
    if outcomes_seq:
        current_streak_type = outcomes_seq[-1]
        current_streak = 0
        for o in reversed(outcomes_seq):
            if o == current_streak_type:
                current_streak += 1
            else:
                break

    # Split: price-based (valuation/squeeze) vs macro (regime/fed/manual)
    _price_types = ("valuation", "squeeze")
    price_resolved  = [e for e in resolved if e.get("signal_type") in _price_types]
    macro_resolved  = [e for e in resolved if e.get("signal_type") not in _price_types]

    price_correct   = sum(1 for e in price_resolved if e["outcome"] == "correct")
    macro_correct   = sum(1 for e in macro_resolved if e["outcome"] == "correct")
    price_accuracy  = round(price_correct / len(price_resolved) * 100, 1) if price_resolved else None
    macro_accuracy  = round(macro_correct / len(macro_resolved) * 100, 1) if macro_resolved else None

    # Alpha vs SPY — price-based only
    alphas = [e["alpha_pct"] for e in price_resolved if e.get("alpha_pct") is not None]
    avg_alpha = round(sum(alphas) / len(alphas), 2) if alphas else None
    positive_alpha_rate = round(sum(1 for a in alphas if a > 0) / len(alphas) * 100, 1) if alphas else None

    # Returns — price-based only
    correct_returns   = [e["return_pct"] for e in price_resolved if e["outcome"] == "correct"   and e.get("return_pct") is not None]
    incorrect_returns = [e["return_pct"] for e in price_resolved if e["outcome"] == "incorrect" and e.get("return_pct") is not None]
    avg_return_correct   = round(sum(correct_returns)   / len(correct_returns),   2) if correct_returns   else None
    avg_return_incorrect = round(sum(incorrect_returns) / len(incorrect_returns), 2) if incorrect_returns else None

    # Win rate by regime context (VIX bucket + quadrant stored at log time)
    by_regime: dict[str, dict] = {}
    for e in price_resolved:
        ctx = e.get("market_context") or {}
        quadrant = ctx.get("quadrant") or "Unknown"
        vix = ctx.get("vix")
        vix_bucket = (
            "VIX<15 (calm)"       if vix and vix < 15  else
            "VIX 15–20 (normal)"  if vix and vix < 20  else
            "VIX 20–30 (elevated)" if vix and vix < 30 else
            "VIX>30 (stress)"     if vix               else
            "VIX unknown"
        )
        key = f"{quadrant} | {vix_bucket}"
        if key not in by_regime:
            by_regime[key] = {"correct": 0, "total": 0, "quadrant": quadrant, "vix_bucket": vix_bucket}
        by_regime[key]["total"] += 1
        if e["outcome"] == "correct":
            by_regime[key]["correct"] += 1
    for d in by_regime.values():
        d["accuracy"] = round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0.0

    return {
        "total": len(log),
        "total_resolved": total_resolved,
        "pending": len(pending),
        "expired": len(expired),
        "correct": correct,
        "accuracy": accuracy,
        # Split accuracy
        "price_accuracy":       price_accuracy,
        "price_resolved":       len(price_resolved),
        "macro_accuracy":       macro_accuracy,
        "macro_resolved":       len(macro_resolved),
        "by_type":    by_type,
        "by_model":   by_model,
        "by_regime":  by_regime,
        "calibration": calibration,
        "avg_return_correct":   avg_return_correct,
        "avg_return_incorrect": avg_return_incorrect,
        "avg_alpha":            avg_alpha,
        "positive_alpha_rate":  positive_alpha_rate,
        "current_streak":          current_streak,
        "current_streak_type":     current_streak_type,
        "best_correct_streak":     best_correct,
        "worst_incorrect_streak":  worst_incorrect,
        "log": log,
    }
