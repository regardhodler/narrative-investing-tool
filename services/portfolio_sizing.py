"""
3-Factor Institutional Position Sizing Engine

Factors:
  1. Regime Fit      (40%): quadrant sensitivity dot-product, 0-100
  2. ATR Risk Budget (35%): Turtle-style sizing — how much portfolio risk consumed vs 1% target
  3. Conviction Decay(25%): entry thesis freshness + regime consistency + price distance

Output per position:
  {
    "ticker":          str,
    "current_weight":  float,   # % of portfolio
    "target_weight":   float,   # % based on ATR stop sizing
    "regime_fit":      int,     # 0-100
    "conviction":      int,     # 0-100
    "composite_score": int,     # weighted 0-100
    "atr_stop":        float|None,
    "risk_pct_used":   float,   # actual % at risk
    "action":          str,     # ADD / HOLD / REDUCE / EXIT
    "add_amount":      int|None,
    "add_pct":         float|None,
    "reduce_amount":   int|None,
    "reduce_to_pct":   float|None,
    "hold_condition":  str|None,
  }
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ── Regime Sensitivity Map ────────────────────────────────────────────────────
# Each ticker maps to [growth, inflation, liquidity, credit] sensitivities (-1 to +1)
_SENSITIVITY: dict[str, list[float]] = {
    # Physical Gold / Gold ETFs
    "GLD":     [-0.2,  0.9,  0.1,  0.3],
    "IAU":     [-0.2,  0.9,  0.1,  0.3],
    "PHYS":    [-0.2,  0.9,  0.1,  0.3],
    "PHYS.TO": [-0.2,  0.9,  0.1,  0.3],
    "GC=F":    [-0.2,  0.9,  0.1,  0.3],
    # Silver
    "SLV":     [ 0.2,  0.8,  0.1,  0.2],
    "PSLV":    [ 0.2,  0.8,  0.1,  0.2],
    "PSLV.TO": [ 0.2,  0.8,  0.1,  0.2],
    "SI=F":    [ 0.2,  0.8,  0.1,  0.2],
    # Copper / Industrial Metals
    "FCX":     [ 0.7,  0.4,  0.2, -0.1],
    "COPX":    [ 0.7,  0.4,  0.2, -0.1],
    "HG=F":    [ 0.7,  0.4,  0.2, -0.1],
    # Energy
    "XLE":     [ 0.3,  0.6,  0.0, -0.1],
    "CVX":     [ 0.3,  0.6,  0.0, -0.1],
    "XOM":     [ 0.3,  0.6,  0.0, -0.1],
    "CHEV":    [ 0.3,  0.6,  0.0, -0.1],
    "CHEV.TO": [ 0.3,  0.6,  0.0, -0.1],
    "CL=F":    [ 0.3,  0.6,  0.0, -0.1],
    # JPY / Risk-off hedge
    "XTLH":    [-0.4,  0.3, -0.7,  0.5],
    "XTLH.TO": [-0.4,  0.3, -0.7,  0.5],
    "FXY":     [-0.5,  0.2, -0.6,  0.6],
    # Healthcare — defensive
    "UNH":     [ 0.1,  0.2,  0.1,  0.1],
    "UNH.TO":  [ 0.1,  0.2,  0.1,  0.1],
    "XLV":     [ 0.1,  0.2,  0.1,  0.1],
    # Broad market / growth
    "FMKT":    [ 0.8,  0.1,  0.3, -0.1],
    "SPY":     [ 0.7,  0.0,  0.3,  0.0],
    "QQQ":     [ 0.9, -0.1,  0.3,  0.0],
    "IWM":     [ 0.8,  0.0,  0.2,  0.0],
    "VTI":     [ 0.7,  0.0,  0.3,  0.0],
    # Bonds — deflation / safety
    "TLT":     [-0.5, -0.8,  0.5,  0.5],
    "IEF":     [-0.3, -0.5,  0.4,  0.4],
    "ZB=F":    [-0.5, -0.8,  0.5,  0.5],
    # Utilities — defensive
    "XLU":     [-0.1,  0.1,  0.2,  0.3],
    # Volatility / tail hedge
    "VXX":     [-0.8, -0.1, -0.6, -0.5],
    "UVXY":    [-0.8, -0.1, -0.6, -0.5],
    # Crypto — high-beta growth
    "BTC-USD": [ 0.9, -0.2,  0.5, -0.3],
    "ETH-USD": [ 0.9, -0.2,  0.5, -0.3],
    "IBIT":    [ 0.9, -0.2,  0.5, -0.3],
}

# ── Dynamic Factor Sensitivity ───────────────────────────────────────────────
# ETF proxies for each factor (all daily via yfinance, already cached)
#   growth    = SPY  — broad equity market tracks the growth cycle
#   inflation = TIP  — TIPS ETF directly tracks inflation expectations
#   liquidity = IEI  — 3-7yr treasuries; flight-to-safety when credit tightens
#   credit    = HYG  — high yield bonds track the credit cycle
_FACTOR_PROXIES = ["SPY", "TIP", "IEI", "HYG"]
_FACTOR_NAMES   = ["growth", "inflation", "liquidity", "credit"]


@st.cache_data(ttl=86400)
def compute_dynamic_sensitivity(ticker: str, lookback_days: int = 126) -> list[float] | None:
    """OLS regression of ticker daily log-returns vs 4 factor proxy returns.

    Returns [growth, inflation, liquidity, credit] each normalized to [-1, 1]
    via tanh(beta / 2).  Returns None if insufficient data (< 63 trading days).

    Cached for 24 hours per ticker — recomputes once daily.
    """
    try:
        from services.market_data import _fetch_single
        frames = {t: _fetch_single(t, period="1y") for t in [ticker] + _FACTOR_PROXIES}
        if any(f is None or f.empty for f in frames.values()):
            return None

        closes = {}
        for t, df in frames.items():
            col = df.get("Close", df.iloc[:, 0])
            closes[t] = col.squeeze()

        price_df = pd.DataFrame(closes).dropna()
        if len(price_df) < 63:
            return None

        log_rets = np.log(price_df / price_df.shift(1)).dropna().tail(lookback_days)
        y = log_rets[ticker].values
        X = log_rets[_FACTOR_PROXIES].values

        # OLS via least squares: beta = (X'X)^-1 X'y
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # Normalize to [-1, 1] via tanh(beta / 2)
        normalized = [float(np.tanh(b / 2)) for b in betas]
        return [round(v, 3) for v in normalized]
    except Exception:
        return None


def get_sensitivity(ticker: str) -> list[float]:
    """Get factor sensitivities for a ticker.

    Priority: dynamic OLS (if ≥63 days data) → static _SENSITIVITY → [0, 0, 0, 0].
    Use this instead of _SENSITIVITY.get() for all new code.
    """
    dynamic = compute_dynamic_sensitivity(ticker.upper())
    if dynamic is not None:
        return dynamic
    return _SENSITIVITY.get(ticker.upper(), [0.0, 0.0, 0.0, 0.0])


# Quadrant → [growth, inflation, liquidity, credit] environment vector
_QUADRANT_VECTOR: dict[str, list[float]] = {
    "Goldilocks":  [ 0.9, -0.2,  0.6,  0.4],
    "Reflation":   [ 0.6,  0.7,  0.3,  0.0],
    "Stagflation": [-0.5,  0.9, -0.3, -0.4],
    "Deflation":   [-0.8, -0.6, -0.7, -0.5],
}


# ── Factor 1: Regime Fit ──────────────────────────────────────────────────────

def _regime_fit_score(ticker: str, regime_ctx: dict) -> float:
    """Return regime fit 0-100. 50 = neutral."""
    sens = get_sensitivity(ticker)
    quadrant = regime_ctx.get("quadrant", "")
    qvec = _QUADRANT_VECTOR.get(quadrant)

    if qvec is None:
        # No quadrant: use macro_score as risk-on/risk-off proxy
        raw_score = float(regime_ctx.get("score", 0.5))
        # Normalize to -1..+1 if given as 0-1 or 0-100
        if raw_score > 1.0:
            raw_score = (raw_score - 50.0) / 50.0
        elif raw_score <= 1.0 and raw_score >= 0.0:
            raw_score = (raw_score - 0.5) * 2.0
        avg_sens = sum(sens) / len(sens) if sens else 0.0
        fit_raw = avg_sens * raw_score
        return max(0.0, min(100.0, 50.0 + fit_raw * 50.0))

    # Dot product of ticker sensitivity vs quadrant environment vector
    dot = sum(s * q for s, q in zip(sens, qvec))
    # dot ranges ~-3.6 to +3.6 (4 products each ≤ 1.0)
    fit_pct = (dot / 3.6 + 1.0) / 2.0
    return max(0.0, min(100.0, fit_pct * 100.0))


# ── Factor 2: ATR Risk Budget ─────────────────────────────────────────────────

def _fetch_atr(ticker: str, period: int = 20) -> float | None:
    """Fetch 20-day ATR for ticker. Returns absolute ATR in price units or None."""
    try:
        from services.market_data import fetch_ohlcv_single
        ohlcv = fetch_ohlcv_single(ticker, period="3mo", interval="1d")
        if ohlcv is None or ohlcv.empty or len(ohlcv) < period + 1:
            return None
        high = ohlcv["High"].squeeze()
        low  = ohlcv["Low"].squeeze()
        prev_close = ohlcv["Close"].squeeze().shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.rolling(period).mean().iloc[-1])
        return atr_val if atr_val > 0 else None
    except Exception:
        return None


def _risk_budget_score(
    position_size: float,
    current_price: float,
    atr: float | None,
    portfolio_value: float,
    target_risk_pct: float = 1.0,
) -> dict:
    """
    Turtle-style risk budget calculation.

    Returns dict with:
      risk_pct_used:    actual % of portfolio at risk (ATR-stop-based)
      target_weight_pct: optimal position size % based on risk budget
      score:            0-100 (50 = at target; >50 = room to add; <50 = overweight risk)
      atr_stop:         suggested stop price (2× ATR below current)
    """
    position_value = current_price * position_size
    current_weight = (position_value / portfolio_value * 100.0) if portfolio_value > 0 else 0.0

    if atr is None or atr <= 0 or portfolio_value <= 0 or current_price <= 0:
        return {
            "risk_pct_used":    round(current_weight, 2),
            "target_weight_pct": round(current_weight, 1),
            "score":            50.0,
            "atr_stop":         None,
        }

    stop_distance = 2.0 * atr                          # $ distance to stop
    atr_stop      = current_price - stop_distance       # long stop price

    # Optimal shares so that 1% of portfolio is at risk
    dollar_risk_target = portfolio_value * (target_risk_pct / 100.0)
    optimal_shares     = dollar_risk_target / stop_distance
    target_weight      = (optimal_shares * current_price / portfolio_value) * 100.0

    # Actual risk consumed
    actual_dollar_risk = position_size * stop_distance
    actual_risk_pct    = (actual_dollar_risk / portfolio_value) * 100.0

    # Score based on current vs target weight ratio
    ratio = current_weight / target_weight if target_weight > 0 else 1.0
    if   ratio <= 0.5:  score = 78.0   # significantly underweight — strong add signal
    elif ratio <= 0.85: score = 63.0   # underweight — mild add
    elif ratio <= 1.15: score = 50.0   # at target — hold
    elif ratio <= 1.5:  score = 32.0   # overweight — reduce
    else:               score = 18.0   # significantly overweight — reduce urgently

    return {
        "risk_pct_used":    round(actual_risk_pct, 2),
        "target_weight_pct": round(max(0.5, min(25.0, target_weight)), 1),  # cap 0.5-25%
        "score":            score,
        "atr_stop":         round(atr_stop, 2) if atr_stop > 0 else None,
    }


# ── Factor 3: Conviction Decay ────────────────────────────────────────────────

def _conviction_score(trade: dict, regime_ctx: dict, current_price: float) -> float:
    """
    Conviction 0-100:
      40pts — regime consistency (entry regime vs current)
      30pts — hold duration (fresh = full, stale = decaying)
      30pts — price distance from entry (near = thesis active)
    """
    from datetime import date as _date
    score = 0.0

    # 1. Regime consistency (40 pts)
    entry_regime   = (trade.get("regime_at_entry") or "").split("(")[0].strip()
    current_regime = regime_ctx.get("quadrant") or regime_ctx.get("regime") or ""
    if not entry_regime or not current_regime:
        score += 20.0
    elif entry_regime.lower() == current_regime.lower():
        score += 40.0
    else:
        score += 8.0   # regime flip — thesis under pressure

    # 2. Duration decay (30 pts) — targets 90-day hold window
    try:
        entry_d   = _date.fromisoformat(str(trade.get("entry_date", "")))
        days_held = (_date.today() - entry_d).days
        if   days_held <= 30:  score += 30.0
        elif days_held <= 90:  score += 25.0
        elif days_held <= 180: score += 15.0
        elif days_held <= 365: score += 8.0
        else:                  score += 3.0
    except Exception:
        score += 15.0

    # 3. Distance from entry (30 pts)
    entry_px = float(trade.get("entry_price") or 0)
    if entry_px <= 0:
        score += 15.0
    else:
        direction = (trade.get("direction") or "long").lower()
        move_pct  = ((current_price - entry_px) / entry_px * 100.0)
        if direction == "short":
            move_pct = -move_pct  # positive = working for short
        if   move_pct >= 20:   score += 10.0  # large winner — consider booking
        elif move_pct >= 5:    score += 20.0  # modest gain
        elif move_pct >= -5:   score += 30.0  # near entry
        elif move_pct >= -15:  score += 15.0  # moderate loss
        else:                  score += 5.0   # large loss — review thesis

    return min(100.0, score)


# ── Main Scoring Functions ────────────────────────────────────────────────────

def score_position(
    trade: dict,
    regime_ctx: dict,
    portfolio_value: float,
    current_price: float,
) -> dict:
    """
    Score a single open position. Returns structured action dict.

    trade must have: ticker, entry_price, position_size, direction, entry_date, status
    """
    ticker        = trade["ticker"].upper()
    entry_price   = float(trade.get("entry_price")   or 0)
    position_size = float(trade.get("position_size") or 0)

    # ── Three factors ─────────────────────────────────────────────────────────
    rf_score  = _regime_fit_score(ticker, regime_ctx)                           # 0-100
    atr       = _fetch_atr(ticker)
    rb        = _risk_budget_score(position_size, current_price, atr,
                                   portfolio_value)                              # dict
    cv_score  = _conviction_score(trade, regime_ctx, current_price)             # 0-100

    composite = rf_score * 0.40 + rb["score"] * 0.35 + cv_score * 0.25

    # ── Weights ───────────────────────────────────────────────────────────────
    position_value = current_price * position_size
    current_weight = (position_value / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
    target_weight  = rb["target_weight_pct"]
    delta          = target_weight - current_weight

    # ── Action ────────────────────────────────────────────────────────────────
    if composite < 25:
        action = "EXIT"
    elif delta > 2.0:
        action = "ADD"
    elif delta < -2.0:
        action = "REDUCE"
    else:
        action = "HOLD"

    # Dollar amounts
    add_amount    = round((delta / 100.0) * portfolio_value)        if action == "ADD"    else None
    reduce_amount = round((-delta / 100.0) * portfolio_value)       if action == "REDUCE" else None

    # HOLD condition
    hold_condition = None
    if action == "HOLD":
        quadrant = regime_ctx.get("quadrant", "")
        parts = []
        if quadrant:
            parts.append(f"quadrant flips from {quadrant}")
        atr_stop = rb.get("atr_stop")
        if atr_stop:
            parts.append(f"price breaks ${atr_stop:.2f}")
        hold_condition = "Re-evaluate if " + " or ".join(parts) if parts else "Monitor signals"

    return {
        "ticker":           ticker,
        "direction":        trade.get("direction", "Long"),
        "entry_price":      entry_price,
        "current_price":    round(current_price, 2),
        "position_size":    position_size,
        "position_value":   round(position_value, 2),
        "current_weight":   round(current_weight, 1),
        "target_weight":    round(target_weight, 1),
        "regime_fit":       round(rf_score),
        "conviction":       round(cv_score),
        "risk_pct_used":    rb.get("risk_pct_used"),
        "composite_score":  round(composite),
        "atr_stop":         rb.get("atr_stop"),
        "action":           action,
        "add_amount":       add_amount,
        "add_pct":          round(delta, 1)          if action == "ADD"    else None,
        "reduce_amount":    reduce_amount,
        "reduce_to_pct":    round(target_weight, 1)  if action == "REDUCE" else None,
        "hold_condition":   hold_condition,
    }


def score_portfolio(
    trades: list[dict],
    regime_ctx: dict,
    portfolio_value: float,
    live_prices: dict[str, float],
) -> dict:
    """
    Score all open positions. Returns:
      {
        "positions":         list[dict],
        "total_add":         int,      # $ to deploy
        "total_reduce":      int,      # $ to trim
        "exits":             list[str],
        "rebalance_summary": str,
        "portfolio_value":   float,
      }
    """
    scored = []
    for t in trades:
        if t.get("status") != "open":
            continue
        tk  = t["ticker"].upper()
        px  = live_prices.get(tk) or live_prices.get(t["ticker"]) or float(t.get("entry_price") or 0)
        if px <= 0:
            px = float(t.get("entry_price") or 0)
        s = score_position(t, regime_ctx, portfolio_value, px)
        scored.append(s)

    total_add    = sum(s["add_amount"]    for s in scored if s.get("add_amount"))
    total_reduce = sum(s["reduce_amount"] for s in scored if s.get("reduce_amount"))
    exits        = [s["ticker"]           for s in scored if s["action"] == "EXIT"]

    parts = []
    if total_add    > 0: parts.append(f"Deploy ${total_add:,.0f}")
    if total_reduce > 0: parts.append(f"Trim ${total_reduce:,.0f}")
    if exits:            parts.append(f"Exit: {', '.join(exits)}")
    summary = " · ".join(parts) if parts else "Portfolio balanced — no rebalancing needed"

    return {
        "positions":          scored,
        "total_add":          total_add,
        "total_reduce":       total_reduce,
        "exits":              exits,
        "rebalance_summary":  summary,
        "portfolio_value":    portfolio_value,
    }


# ── Portfolio-Level Factor Exposure ──────────────────────────────────────────

_FACTORS = ["growth", "inflation", "liquidity", "credit"]

def aggregate_factor_exposure(
    positions: list[dict],   # list of score_position() results (has ticker + current_weight)
) -> dict:
    """
    Weight each ticker's factor sensitivity vector by its portfolio weight.

    Returns:
      {
        "factors":   {"growth": float, "inflation": float, "liquidity": float, "credit": float},
        "dominant":  str,      # factor with largest absolute exposure
        "warnings":  list[str] # e.g. ["Overweight inflation (1.3x)"]
      }
    """
    exposure = {f: 0.0 for f in _FACTORS}
    total_weight = sum(p.get("current_weight", 0) for p in positions)
    if total_weight <= 0:
        return {"factors": exposure, "dominant": "", "warnings": []}

    for p in positions:
        tk = p.get("ticker", "").upper()
        wt = p.get("current_weight", 0) / 100.0  # convert % to decimal
        sens = get_sensitivity(tk)
        for i, f in enumerate(_FACTORS):
            exposure[f] += wt * sens[i]

    # Normalize to total_weight scale (so 100% invested = full exposure)
    scale = 100.0 / total_weight if total_weight > 0 else 1.0
    exposure = {f: round(v * scale, 3) for f, v in exposure.items()}

    dominant = max(exposure, key=lambda f: abs(exposure[f]))

    warnings = []
    if abs(exposure["inflation"]) > 0.6:
        direction = "over" if exposure["inflation"] > 0 else "short"
        warnings.append(f"{direction.capitalize()}weight inflation ({exposure['inflation']:+.2f}x)")
    if abs(exposure["growth"]) > 0.7:
        direction = "over" if exposure["growth"] > 0 else "short"
        warnings.append(f"{direction.capitalize()}weight growth ({exposure['growth']:+.2f}x)")
    if exposure["credit"] < -0.4:
        warnings.append(f"Significant credit risk-off exposure ({exposure['credit']:+.2f}x)")

    return {"factors": exposure, "dominant": dominant, "warnings": warnings}


# ── Pre-Trade What-If Simulator ───────────────────────────────────────────────

def simulate_add(
    ticker: str,
    dollar_amount: float,
    existing_positions: list[dict],   # open trades (raw trade dicts)
    regime_ctx: dict,
    portfolio_value: float,
    live_prices: dict[str, float],
) -> dict:
    """
    Simulate adding a new position and return impact on portfolio metrics.

    Returns:
      {
        "ticker":            str,
        "proposed_weight":   float,   # % of portfolio
        "new_portfolio_value": float,
        "sizing_score":      dict,    # score_position() result for the proposed trade
        "factor_before":     dict,    # aggregate_factor_exposure() before add
        "factor_after":      dict,    # aggregate_factor_exposure() after add
        "factor_delta":      dict,    # change per factor
        "corr_to_portfolio": float|None,  # avg correlation to existing tickers
        "warnings":          list[str],
      }
    """
    tk = ticker.upper()
    cur_px = live_prices.get(tk)

    # Compute existing factor exposure
    scored_existing = []
    for t in existing_positions:
        if t.get("status") != "open":
            continue
        t_tk = t["ticker"].upper()
        px = live_prices.get(t_tk) or float(t.get("entry_price") or 0)
        if px > 0:
            scored_existing.append(score_position(t, regime_ctx, portfolio_value, px))

    factor_before = aggregate_factor_exposure(scored_existing)

    # Build a synthetic trade dict for the proposed position
    proposed_trade = {
        "ticker": tk,
        "direction": "long",
        "entry_price": cur_px or 0,
        "position_size": int(dollar_amount / cur_px) if cur_px and cur_px > 0 else 0,
        "entry_date": "",
        "regime_at_entry": regime_ctx.get("quadrant", ""),
        "status": "open",
    }
    new_portfolio_value = portfolio_value + dollar_amount
    proposed_weight = (dollar_amount / new_portfolio_value * 100) if new_portfolio_value > 0 else 0

    # Score the proposed position
    if cur_px and cur_px > 0:
        sizing = score_position(proposed_trade, regime_ctx, new_portfolio_value, cur_px)
    else:
        sizing = {"action": "UNKNOWN", "composite_score": None, "regime_fit": None,
                  "conviction": None, "atr_stop": None, "target_weight": None}

    # Factor exposure after add — scale existing weights to new portfolio value
    rescaled = []
    for p in scored_existing:
        p2 = dict(p)
        p2["current_weight"] = p2["current_weight"] * portfolio_value / new_portfolio_value
        rescaled.append(p2)
    proposed_scored = dict(sizing)
    proposed_scored["ticker"] = tk
    proposed_scored["current_weight"] = proposed_weight
    rescaled.append(proposed_scored)
    factor_after = aggregate_factor_exposure(rescaled)

    factor_delta = {
        f: round(factor_after["factors"][f] - factor_before["factors"][f], 3)
        for f in _FACTORS
    }

    # Correlation to portfolio (avg of proposed ticker vs each existing ticker)
    corr_avg = None
    try:
        from services.market_data import fetch_correlation_matrix
        all_tickers = tuple(
            sorted({t["ticker"].upper() for t in existing_positions if t.get("status") == "open"} | {tk})
        )
        if len(all_tickers) >= 2:
            corr_df = fetch_correlation_matrix(all_tickers)
            if corr_df is not None and tk in corr_df.columns:
                others = [c for c in corr_df.columns if c != tk]
                corr_vals = [corr_df.loc[tk, c] for c in others if c in corr_df.index]
                if corr_vals:
                    corr_avg = round(float(sum(corr_vals) / len(corr_vals)), 3)
    except Exception:
        pass

    warnings = list(factor_after["warnings"])
    if corr_avg is not None and corr_avg > 0.75:
        warnings.append(f"High avg correlation to portfolio ({corr_avg:.2f}) — limited diversification benefit")
    if sizing.get("composite_score") is not None and sizing["composite_score"] < 40:
        warnings.append(f"Low sizing score ({sizing['composite_score']}) — poor regime fit")

    return {
        "ticker":              tk,
        "proposed_weight":     round(proposed_weight, 2),
        "new_portfolio_value": round(new_portfolio_value, 2),
        "sizing_score":        sizing,
        "factor_before":       factor_before,
        "factor_after":        factor_after,
        "factor_delta":        factor_delta,
        "corr_to_portfolio":   corr_avg,
        "warnings":            warnings,
    }


# ── Dynamic Kelly Criterion ───────────────────────────────────────────────────

_REGIME_B_IMPLIED: dict[str, float] = {
    "Goldilocks":  1.8,
    "Reflation":   1.4,
    "Stagflation": 0.8,
    "Deflation":   0.6,
}


def get_trade_kelly_stats() -> dict:
    """Read trade_journal.json and compute win/loss stats for Kelly calculation.

    Returns:
        n_closed    — number of closed trades with both entry and exit price
        n_wins      — count of profitable closed trades
        n_losses    — count of losing closed trades
        avg_win_pct — average return % of winning trades (positive float)
        avg_loss_pct— average return % of losing trades (negative float, or 0)
    """
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_journal.json")
    try:
        with open(path) as f:
            trades = json.load(f)
    except Exception:
        return {"n_closed": 0, "n_wins": 0, "n_losses": 0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0}

    win_pcts, loss_pcts = [], []
    for t in trades:
        if t.get("status") != "closed":
            continue
        ep = t.get("entry_price") or 0
        xp = t.get("exit_price") or 0
        if not ep or not xp:
            continue
        direction = (t.get("direction") or "Long").lower()
        if direction == "long":
            ret_pct = (xp - ep) / ep * 100
        else:
            ret_pct = (ep - xp) / ep * 100
        if ret_pct > 0:
            win_pcts.append(ret_pct)
        else:
            loss_pcts.append(ret_pct)

    n_closed = len(win_pcts) + len(loss_pcts)
    return {
        "n_closed":     n_closed,
        "n_wins":       len(win_pcts),
        "n_losses":     len(loss_pcts),
        "avg_win_pct":  sum(win_pcts) / len(win_pcts) if win_pcts else 0.0,
        "avg_loss_pct": sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0,
    }


def compute_qir_kelly(
    conviction_score: int,
    fear_composite: dict,
    regime_ctx: dict,
    options_score: int | None = None,
    tactical_score: int | None = None,
    hmm_state_label: str | None = None,
) -> dict:
    """Compute dynamic half-Kelly position size for the QIR dashboard.

    p (win probability): Bayesian shrinkage blend of historical win rate and
    current conviction score. With few trades, heavily conviction-weighted;
    converges to historical rate at ~20 closed trades (max 60% weight).

    b (win/loss ratio): From realized avg_win_pct / abs(avg_loss_pct) when
    both wins and losses exist in trade history. Otherwise regime-implied.

    stress discount: fear_composite.score / 100 * 0.30 (up to 30% reduction).

    alignment multiplier: fraction of 4 signals agreeing with verdict direction.
    0/4=×0.25, 1/4=×0.50, 2/4=×0.75, 3/4=×0.90, 4/4=×1.00.

    hmm multiplier: Bull=×1.10, Neutral=×1.00, Stress=×0.85, Late Cycle=×0.75, Crisis=×0.60.

    Returns dict with keys:
        kelly_half_pct, kelly_full_pct, p, b, p_source, b_source,
        n_closed, stress_discount_pct, capped, viable,
        alignment_score, n_signals_agree, n_signals_total,
        align_multiplier, hmm_multiplier, signal_dirs
    """
    stats = get_trade_kelly_stats()
    n_closed = stats["n_closed"]
    n_wins   = stats["n_wins"]
    n_losses = stats["n_losses"]

    # ── p: Bayesian shrinkage ─────────────────────────────────────────────────
    shrink_weight  = 5.0
    p_hist_shrunk  = (n_wins + shrink_weight * 0.5) / (n_closed + shrink_weight)
    history_weight = min(n_closed / 20.0, 0.6)
    p_conviction   = max(0.01, min(0.99, (conviction_score if conviction_score is not None else 50) / 100.0))
    p = p_hist_shrunk * history_weight + p_conviction * (1.0 - history_weight)

    if n_closed == 0:
        p_source = "Conviction only"
    else:
        p_source = f"Historical (n={n_closed}) + Conviction"

    # ── b: win/loss ratio ─────────────────────────────────────────────────────
    quadrant = (regime_ctx or {}).get("quadrant", "")
    b_implied = _REGIME_B_IMPLIED.get(quadrant, 1.2)

    # Verdict direction from p for b-flip (before alignment computation)
    _p_dir = 1 if p > 0.55 else (-1 if p < 0.45 else 0)

    # For bearish verdicts, flip the regime-implied b:
    # Best shorting envs (Stagflation/Deflation) get high b; worst (Goldilocks) get low b.
    # Mirrors the long-side logic: short in Stagflation = long in Goldilocks.
    _SHORT_B_IMPLIED: dict[str, float] = {
        "Stagflation": 1.8,   # best short env — mirror of Goldilocks long
        "Deflation":   1.4,   # good short env — mirror of Reflation long
        "Reflation":   0.8,   # poor short env — mirror of Stagflation long
        "Goldilocks":  0.6,   # worst short env — mirror of Deflation long
    }
    if _p_dir == -1:
        b_implied = _SHORT_B_IMPLIED.get(quadrant, 1.2)

    avg_win  = stats["avg_win_pct"]
    avg_loss = abs(stats["avg_loss_pct"])
    if n_wins >= 5 and n_losses >= 5 and avg_loss > 0:
        # Enough real history on both sides — use actual win/loss ratio
        b = avg_win / avg_loss
        b_source = f"Historical (w={n_wins}, l={n_losses})"
    elif n_wins >= 1 and n_losses >= 1 and avg_loss > 0:
        # Some history but not enough for full confidence — blend ATR rule with real data
        _atr_b = 1.5  # ATR weekly ×3 target / ×2 stop = 1.5
        _real_b = avg_win / avg_loss
        _blend_w = min((n_wins + n_losses) / 10.0, 0.5)  # up to 50% real at 10 trades each side
        b = _atr_b * (1 - _blend_w) + _real_b * _blend_w
        b_source = f"ATR 2:3 rule + partial history (n={n_wins+n_losses})"
    else:
        # No real history yet — use ATR weekly ×3/×2 = 1.5 as bootstrap
        b = 1.5
        b_source = "ATR weekly 2:3 rule (bootstrap)"

    # ── Kelly formula ─────────────────────────────────────────────────────────
    q = 1.0 - p
    kelly_full = (b * p - q) / b if b > 0 else 0.0
    kelly_half = max(kelly_full * 0.5, 0.0)

    # ── Stress discount ───────────────────────────────────────────────────────
    fear_score       = float((fear_composite or {}).get("score", 50))
    stress_discount  = (fear_score / 100.0) * 0.30
    kelly_half       = max(kelly_half - stress_discount * kelly_half, 0.0)

    # Save base kelly (after stress, before alignment/HMM) for reference table
    kelly_half_base = kelly_half

    # ── Cross-timeframe alignment ─────────────────────────────────────────────
    # Verdict direction from p: bull if p > 0.55, bear if p < 0.45, neutral otherwise
    _verdict_dir = 1 if p > 0.55 else (-1 if p < 0.45 else 0)

    def _to_dir(score: int | None, neutral_band: int = 10) -> int | None:
        if score is None:
            return None
        if score >= 50 + neutral_band:
            return 1
        if score <= 50 - neutral_band:
            return -1
        return 0

    # regime score: macro_score is already 0-100 if present, else use score (0-1)
    _rc = regime_ctx or {}
    _macro_raw = _rc.get("macro_score")
    if _macro_raw is None:
        _macro_raw = float(_rc.get("score", 0.5)) * 100.0
    _regime_dir = _to_dir(int(_macro_raw))

    _signal_dirs: dict[str, int | None] = {
        "options":   _to_dir(options_score),
        "tactical":  _to_dir(tactical_score),
        "regime":    _regime_dir,
        "conviction": _verdict_dir,
    }

    _n_total = sum(1 for v in _signal_dirs.values() if v is not None)
    _n_agree = sum(1 for v in _signal_dirs.values() if v is not None and v == _verdict_dir)

    _alignment = _n_agree / _n_total if _n_total > 0 else 0.5

    # Alignment multiplier — neutral verdict skips alignment penalty
    if _verdict_dir == 0:
        _align_mult = 0.75
    else:
        _align_mult = {0: 0.25, 1: 0.50, 2: 0.75, 3: 0.90}.get(_n_agree, 1.00)

    kelly_half = kelly_half * _align_mult

    # ── HMM state multiplier (final gate) ────────────────────────────────────
    _HMM_MULT = {
        "Bull":         1.10,
        "Neutral":      1.00,
        "Early Stress": 0.90,
        "Stress":       0.85,
        "Late Cycle":   0.75,
        "Crisis":       0.60,
    }
    _hmm_mult = 1.00
    if hmm_state_label:
        for key, mult in _HMM_MULT.items():
            if key.lower() in hmm_state_label.lower():
                _hmm_mult = mult
                break
    kelly_half = kelly_half * _hmm_mult

    # ── Cap at 15% ────────────────────────────────────────────────────────────
    capped = kelly_half > 0.15
    kelly_half = min(kelly_half, 0.15)

    return {
        "kelly_half_pct":      round(kelly_half * 100, 1),
        "kelly_full_pct":      round(max(kelly_full, 0.0) * 100, 1),
        "kelly_half_raw_pct":  round(max(kelly_full * 0.5, 0.0) * 100, 1),  # before stress/alignment/HMM
        "p":                   round(p, 3),
        "b":                   round(b, 2),
        "p_source":            p_source,
        "b_source":            b_source,
        "n_closed":            n_closed,
        "stress_discount_pct": round(stress_discount * 100, 1),
        "fear_score":          round(fear_score, 1),
        "capped":              capped,
        "viable":              round(kelly_half * 100, 1) >= 0.1,
        "alignment_score":     round(_alignment * 100),
        "n_signals_agree":     _n_agree,
        "n_signals_total":     _n_total,
        "align_multiplier":    round(_align_mult, 2),
        "hmm_multiplier":      round(_hmm_mult, 2),
        "signal_dirs":         _signal_dirs,
        "kelly_half_base_pct": round(kelly_half_base * 100, 1),
    }


_SHORT_B_IMPLIED: dict[str, float] = {
    "Stagflation": 1.8,
    "Deflation":   1.4,
    "Reflation":   0.8,
    "Goldilocks":  0.6,
}

_HMM_MULT_MAP: dict[str, float] = {
    "Bull":         1.10,
    "Neutral":      1.00,
    "Early Stress": 0.90,
    "Stress":       0.85,
    "Late Cycle":   0.75,
    "Crisis":       0.60,
}


def compute_triple_kelly(
    fear_composite: dict,
    regime_ctx: dict,
    hmm_state_label: str | None,
    forced_lean: str,       # "BULLISH" or "BEARISH"
    lean_pct: float,        # e.g. 53.0
    uncertainty_score: int, # 0-100
    macro_score: int = 50,
    leading_score: int = 50,
) -> dict:
    """Triple Kelly for GENUINE_UNCERTAINTY — bimodal sizing across 3 horizons.

    1. Structural Long  — HMM regime persistence + macro. Weeks/months.
    2. Tactical Short   — Forced bearish lean + fear. Days/weeks.
    3. Tactical Long    — Momentum scalp, penalised by uncertainty. 1-3 days.

    Returns dict with keys: structural, tactical_short, tactical_long.
    Each sub-dict: half_pct, full_pct, p, b, label, timeframe, color.
    """
    quadrant   = (regime_ctx or {}).get("quadrant", "")
    fear_score = float((fear_composite or {}).get("score", 50))

    # HMM multiplier
    _hmm_mult = 1.00
    if hmm_state_label:
        for key, mult in _HMM_MULT_MAP.items():
            if key.lower() in hmm_state_label.lower():
                _hmm_mult = mult
                break

    def _kelly(p, b):
        q = 1.0 - p
        full = (b * p - q) / b if b > 0 else 0.0
        return max(full, 0.0), max(full * 0.5, 0.0)

    # ── 1. Structural Long ────────────────────────────────────────────────────
    # p: macro_score (long-term regime confidence)
    # b: always long-side regime-implied — structural is a regime trade, not a lean trade
    # Direction is determined by macro/HMM, NOT the short-term forced lean
    # HMM multiplier applies — this IS the regime trade
    p_struct = max(0.01, min(0.99, macro_score / 100))
    b_struct  = _REGIME_B_IMPLIED.get(quadrant, 1.2)
    kf_s, kh_s = _kelly(p_struct, b_struct)
    kh_s = kh_s * _hmm_mult
    kh_s = min(kh_s, 0.15)

    # ── 2. Tactical Short ─────────────────────────────────────────────────────
    # p: BEARISH scenario probability — opposite side of the forced lean
    # When lean is BULLISH, bearish prob = (100 - lean_pct) / 100
    # When lean is BEARISH, bearish prob = lean_pct / 100
    # Fear AMPLIFIES short edge (high fear = better short environment)
    # No HMM mult — this is a short-term hedge, not a regime trade
    _bear_pct = (100 - lean_pct) if forced_lean == "BULLISH" else lean_pct
    p_short = max(0.01, min(0.99, _bear_pct / 100))
    b_short  = _SHORT_B_IMPLIED.get(quadrant, 1.2)
    kf_sh, kh_sh = _kelly(p_short, b_short)
    fear_boost = 1.0 + (fear_score / 100) * 0.20   # fear helps shorts, up to +20%
    kh_sh = kh_sh * fear_boost

    # Lean alignment: suppress or penalise short leg when lean is clearly bullish
    _is_bullish_lean = forced_lean == "BULLISH"
    if _is_bullish_lean and lean_pct > 55:
        # Clear BUY signal -- suppress the short leg entirely
        kh_sh = 0.0
        kf_sh = 0.0
        _short_label = "Bear Hedge"
        _short_note = f"Suppressed -- BUY lean {lean_pct:.0f}% exceeds 55% threshold"
        _short_color = "#475569"
    elif _is_bullish_lean:
        # Weakly bullish -- allow a small hedge penalised by bear confidence
        kh_sh = kh_sh * (_bear_pct / 100)
        kh_sh = min(kh_sh, 0.05)
        kf_sh = kh_sh * 2
        _short_label = "Bear Hedge"
        _short_note = f"Minority hedge -- bear {_bear_pct:.0f}% scenario, lean penalty applied"
        _short_color = "#f97316"
    else:
        kh_sh = min(kh_sh, 0.15)
        _short_label = "Tactical Short"
        _short_note = f"Bear scenario {_bear_pct:.0f}% · fear {fear_score:.0f}/100 boost"
        _short_color = "#ef4444"

    # ── 3. Tactical Long (scalp) ──────────────────────────────────────────────
    # p: leading momentum score, heavily penalised by uncertainty
    # b: tight (momentum scalps have narrow edge)
    # Uncertainty penalty shrinks size — high uncertainty = tiny scalp only
    _uncert_penalty = 1.0 - (uncertainty_score / 200)   # 0-100 → 0.50-1.00 penalty range
    p_scalp = max(0.01, min(0.99, (leading_score / 100) * _uncert_penalty))
    b_scalp  = 1.1
    kf_sc, kh_sc = _kelly(p_scalp, b_scalp)
    kh_sc = kh_sc * _uncert_penalty   # double penalty — momentum scalp in uncertainty is risky
    kh_sc = min(kh_sc, 0.08)          # hard cap 8% — scalps never go large

    return {
        "structural": {
            "half_pct":  round(kh_s  * 100, 1),
            "full_pct":  round(kf_s  * 100, 1),
            "p": round(p_struct, 3), "b": round(b_struct, 2),
            "label":     "Structural Long",
            "timeframe": "weeks/months",
            "color":     "#22c55e",
            "note":      f"HMM {hmm_state_label or 'N/A'} ×{_hmm_mult:.2f} · macro {macro_score}/100",
        },
        "tactical_short": {
            "half_pct":  round(kh_sh * 100, 1),
            "full_pct":  round(kf_sh * 100, 1),
            "p": round(p_short, 3), "b": round(b_short, 2),
            "label":     _short_label,
            "timeframe": "1-3 days",
            "color":     _short_color,
            "note":      _short_note,
        },
        "tactical_long": {
            "half_pct":  round(kh_sc * 100, 1),
            "full_pct":  round(kf_sc * 100, 1),
            "p": round(p_scalp, 3), "b": round(b_scalp, 2),
            "label":     "Tactical Long",
            "timeframe": "days/weeks",
            "color":     "#f59e0b",
            "note":      f"Momentum {leading_score}/100 · uncertainty penalty ×{_uncert_penalty:.2f}",
        },
    }
