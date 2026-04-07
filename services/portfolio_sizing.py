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
