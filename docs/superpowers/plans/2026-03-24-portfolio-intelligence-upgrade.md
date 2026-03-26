# Portfolio Intelligence Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Macro Scorecard with an institutional-grade 3-factor position sizing engine wired directly into Portfolio Intelligence, giving specific ADD/REDUCE/EXIT amounts per holding; also move the AI tier selector to the top of Elliott Wave and Wyckoff pages.

**Architecture:** New pure-Python scoring service (`services/portfolio_sizing.py`) computes Regime Fit × ATR Risk Budget × Conviction Decay per position and returns structured action dicts. `trade_journal.py` calls this service to render a regime strip + upgraded per-position cards + rebalance summary. Elliott Wave and Wyckoff tier selectors move from the bottom expander to the top controls row.

**Tech Stack:** Python, yfinance (ATR via OHLCV), Streamlit session_state (regime_context), existing `fetch_ohlcv_single` from `services/market_data.py`.

---

## Chunk 1: Scoring Engine

### Task 1: `services/portfolio_sizing.py` — Core scoring service

**Files:**
- Create: `services/portfolio_sizing.py`

The service must be pure Python — no `st.*` calls. It returns plain dicts that the UI layer renders.

#### Regime Sensitivity Map

Every ticker gets a 4-element sensitivity vector `[growth, inflation, liquidity, credit]` on a -1 to +1 scale. Unknown tickers fall back to a neutral `[0, 0, 0, 0]`.

The **quadrant** from `_regime_context` maps to a 4-element regime vector:
- `"Goldilocks"` → `[+0.9, -0.2, +0.6, +0.4]`
- `"Reflation"` → `[+0.6, +0.7, +0.3, +0.0]`
- `"Stagflation"` → `[-0.5, +0.9, -0.3, -0.4]`
- `"Deflation"` → `[-0.8, -0.6, -0.7, -0.5]`
- `""` / unknown → use macro_score only (described below)

**Dot product** of ticker sensitivity vs regime vector gives raw fit (-1 to +1), scaled to 0–100.

#### Factor 1 — Regime Fit (40% weight)

```python
_SENSITIVITY: dict[str, list[float]] = {
    # Gold / Physical Gold ETFs
    "GLD":     [-0.2,  0.9,  0.1,  0.3],
    "PHYS":    [-0.2,  0.9,  0.1,  0.3],
    "PHYS.TO": [-0.2,  0.9,  0.1,  0.3],
    "IAU":     [-0.2,  0.9,  0.1,  0.3],
    # Silver
    "SLV":     [ 0.2,  0.8,  0.1,  0.2],
    "PSLV":    [ 0.2,  0.8,  0.1,  0.2],
    "PSLV.TO": [ 0.2,  0.8,  0.1,  0.2],
    # Copper / Industrial Metals
    "FCX":     [ 0.7,  0.4,  0.2, -0.1],
    "COPX":    [ 0.7,  0.4,  0.2, -0.1],
    "HG=F":    [ 0.7,  0.4,  0.2, -0.1],
    # Energy
    "XLE":     [ 0.3,  0.6,  0.0, -0.1],
    "CVX":     [ 0.3,  0.6,  0.0, -0.1],
    "CHEV":    [ 0.3,  0.6,  0.0, -0.1],  # assuming Chevron variant
    "CHEV.TO": [ 0.3,  0.6,  0.0, -0.1],
    # JPY / Risk-off hedges
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
    # Bonds
    "TLT":     [-0.5, -0.8,  0.5,  0.5],
    "IEF":     [-0.3, -0.5,  0.4,  0.4],
    # Utilities
    "XLU":     [-0.1,  0.1,  0.2,  0.3],
    # Volatility
    "VXX":     [-0.8, -0.1, -0.6, -0.5],
    "UVXY":    [-0.8, -0.1, -0.6, -0.5],
}

_QUADRANT_VECTOR: dict[str, list[float]] = {
    "Goldilocks":  [ 0.9, -0.2,  0.6,  0.4],
    "Reflation":   [ 0.6,  0.7,  0.3,  0.0],
    "Stagflation": [-0.5,  0.9, -0.3, -0.4],
    "Deflation":   [-0.8, -0.6, -0.7, -0.5],
}
```

```python
def _regime_fit_score(ticker: str, regime_ctx: dict) -> float:
    """Return regime fit 0-100. 50 = neutral."""
    sens = _SENSITIVITY.get(ticker.upper(), [0.0, 0.0, 0.0, 0.0])
    quadrant = regime_ctx.get("quadrant", "")
    qvec = _QUADRANT_VECTOR.get(quadrant)
    if qvec is None:
        # No quadrant: use macro_score as a rough risk-on/risk-off proxy
        macro_score = float(regime_ctx.get("score", 0.5))  # 0-1 or -1 to +1
        # Normalize to -1 to +1
        if macro_score > 1:
            macro_score = (macro_score - 50) / 50.0
        # For risk-on assets (positive growth sensitivity), regime_fit rises with score
        fit_raw = sum(s * macro_score for s in sens) / max(len(sens), 1)
        return max(0.0, min(100.0, 50.0 + fit_raw * 50.0))
    # Dot product of sensitivity vs quadrant vector, normalized to 0-100
    dot = sum(s * q for s, q in zip(sens, qvec))
    # dot ranges roughly -3 to +3 (sum of 4 products each up to ±1)
    fit_pct = (dot / 3.6 + 1.0) / 2.0  # 0 to 1
    return max(0.0, min(100.0, fit_pct * 100.0))
```

#### Factor 2 — ATR Risk Budget (35% weight)

```python
@st.cache_data(ttl=3600)  # Note: this one CAN import st since it caches — but keep pure
def _fetch_atr(ticker: str, period: int = 20) -> float | None:
    """Fetch 20-day ATR for a ticker. Returns absolute price ATR or None."""
    try:
        from services.market_data import fetch_ohlcv_single
        ohlcv = fetch_ohlcv_single(ticker, period="3mo", interval="1d")
        if ohlcv is None or ohlcv.empty or len(ohlcv) < period + 1:
            return None
        high = ohlcv["High"].squeeze()
        low  = ohlcv["Low"].squeeze()
        close_prev = ohlcv["Close"].squeeze().shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low  - close_prev).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return None
```

```python
def _risk_budget_score(
    position_size: float,
    entry_price: float,
    current_price: float,
    atr: float | None,
    portfolio_value: float,
    target_risk_pct: float = 1.0,
) -> dict:
    """
    Returns:
      risk_pct_used:  actual risk % of portfolio consumed by this position
      target_pct:     ideal position size % based on ATR stop
      score:          0-100 (100 = exactly at target; <50 = overweight risk; >50 = room to add)
      atr_stop:       suggested stop price
    """
    position_value = current_price * position_size
    current_weight = position_value / portfolio_value * 100 if portfolio_value > 0 else 0

    if atr is None or atr <= 0 or portfolio_value <= 0:
        return {
            "risk_pct_used": current_weight,
            "target_weight_pct": current_weight,  # can't compute, treat as neutral
            "score": 50.0,
            "atr_stop": None,
        }

    # ATR stop: 2× ATR below current price (long position)
    atr_stop = current_price - 2.0 * atr
    stop_distance = current_price - atr_stop  # in $ per share

    # Turtle-style sizing: target risk = portfolio_value * target_risk_pct%
    dollar_risk_target = portfolio_value * (target_risk_pct / 100.0)
    # Optimal shares = dollar_risk_target / stop_distance
    optimal_shares = dollar_risk_target / stop_distance if stop_distance > 0 else position_size
    target_weight = (optimal_shares * current_price / portfolio_value) * 100

    # Actual risk being used ($ at risk if stop hit)
    actual_dollar_risk = position_size * stop_distance
    actual_risk_pct = actual_dollar_risk / portfolio_value * 100

    # Score: 100 = at target, drops as you go overweight
    if target_weight <= 0:
        score = 50.0
    else:
        ratio = current_weight / target_weight
        if ratio <= 0.5:   score = 75.0  # underweight — room to add
        elif ratio <= 0.9: score = 60.0  # slightly underweight
        elif ratio <= 1.1: score = 50.0  # at target
        elif ratio <= 1.5: score = 35.0  # overweight — reduce
        else:              score = 20.0  # significantly overweight

    return {
        "risk_pct_used": round(actual_risk_pct, 2),
        "target_weight_pct": round(target_weight, 1),
        "score": score,
        "atr_stop": round(atr_stop, 2) if atr_stop > 0 else None,
    }
```

#### Factor 3 — Conviction Decay (25% weight)

```python
def _conviction_score(trade: dict, regime_ctx: dict) -> float:
    """
    Conviction 0-100 based on:
      - Regime consistency: regime at entry vs current (40pts)
      - Hold duration vs 90-day target (30pts): fresh = full, >180d = decaying
      - Distance from entry (30pts): still near entry = thesis active
    """
    from datetime import date
    score = 0.0

    # 1. Regime consistency (40 pts)
    entry_regime  = trade.get("regime_at_entry", "")
    current_regime = regime_ctx.get("regime", "") or regime_ctx.get("quadrant", "")
    if not entry_regime or not current_regime:
        score += 20.0   # unknown — neutral
    elif entry_regime == current_regime:
        score += 40.0   # regime unchanged — full conviction
    else:
        score += 10.0   # regime changed — thesis under pressure

    # 2. Duration decay (30 pts) — target hold 90 days
    try:
        entry_d = date.fromisoformat(str(trade.get("entry_date", "")))
        days_held = (date.today() - entry_d).days
        if days_held <= 90:
            score += 30.0
        elif days_held <= 180:
            score += 20.0
        elif days_held <= 365:
            score += 10.0
        else:
            score += 5.0  # very stale position
    except Exception:
        score += 15.0  # unknown date — neutral

    # 3. Distance from entry (30 pts)
    entry_px  = trade.get("entry_price", 0) or 0
    current_px = trade.get("_current_price", entry_px)  # injected by caller
    if entry_px <= 0:
        score += 15.0
    else:
        move_pct = abs((current_px - entry_px) / entry_px * 100)
        if move_pct < 5:
            score += 30.0   # near entry — thesis just starting
        elif move_pct < 15:
            score += 20.0   # moved, still reasonable
        elif move_pct < 30:
            score += 10.0   # large move — consider booking
        else:
            score += 5.0    # very large move — exit or review

    return min(100.0, score)
```

#### Main scoring function

```python
def score_position(
    trade: dict,
    regime_ctx: dict,
    portfolio_value: float,
    current_price: float,
    all_trades: list[dict],
) -> dict:
    """
    Returns a full scoring dict for one position.
    trade must have: ticker, entry_price, position_size, direction, entry_date
    """
    ticker        = trade["ticker"].upper()
    entry_price   = float(trade.get("entry_price", 0) or 0)
    position_size = float(trade.get("position_size", 0) or 0)

    # Inject current price for conviction calc
    trade_with_px = {**trade, "_current_price": current_price}

    # Factor 1: Regime Fit (40%)
    rf  = _regime_fit_score(ticker, regime_ctx)

    # Factor 2: ATR Risk Budget (35%)
    atr = _fetch_atr(ticker)
    rb  = _risk_budget_score(position_size, entry_price, current_price, atr, portfolio_value)

    # Factor 3: Conviction Decay (25%)
    cv  = _conviction_score(trade_with_px, regime_ctx)

    # Composite score (0-100)
    composite = rf * 0.40 + rb["score"] * 0.35 + cv * 0.25

    # Current weight
    position_value  = current_price * position_size
    current_weight  = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
    target_weight   = rb["target_weight_pct"]

    # Action logic
    delta = target_weight - current_weight
    if composite < 25:
        action = "EXIT"
    elif delta > 2.0:
        action = "ADD"
    elif delta < -2.0:
        action = "REDUCE"
    else:
        action = "HOLD"

    # Dollar amounts
    add_amount     = max(0.0, (target_weight - current_weight) / 100 * portfolio_value) if action == "ADD" else None
    reduce_amount  = max(0.0, (current_weight - target_weight) / 100 * portfolio_value) if action == "REDUCE" else None

    # HOLD condition
    quadrant = regime_ctx.get("quadrant", "")
    hold_condition = None
    if action == "HOLD":
        stop = rb.get("atr_stop")
        hold_parts = []
        if quadrant:
            hold_parts.append(f"quadrant flips from {quadrant}")
        if stop:
            hold_parts.append(f"price breaks ${stop:.2f}")
        hold_condition = "Re-evaluate if " + " or ".join(hold_parts) if hold_parts else "Monitor"

    return {
        "ticker":          ticker,
        "direction":       trade.get("direction", "Long"),
        "entry_price":     entry_price,
        "current_price":   current_price,
        "position_size":   position_size,
        "position_value":  round(position_value, 2),
        "current_weight":  round(current_weight, 1),
        "target_weight":   round(target_weight, 1),
        "regime_fit":      round(rf),
        "conviction":      round(cv),
        "composite_score": round(composite),
        "atr_stop":        rb.get("atr_stop"),
        "risk_pct_used":   rb.get("risk_pct_used"),
        "action":          action,
        "add_amount":      round(add_amount) if add_amount else None,
        "add_pct":         round(delta, 1) if action == "ADD" else None,
        "reduce_amount":   round(reduce_amount) if reduce_amount else None,
        "reduce_to_pct":   round(target_weight, 1) if action == "REDUCE" else None,
        "hold_condition":  hold_condition,
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
        "positions": [...],            # list of score_position dicts
        "total_add": float,            # total $ to deploy
        "total_reduce": float,         # total $ to trim
        "rebalance_summary": str,      # human-readable one-liner
        "portfolio_value": float,
      }
    """
    scored = []
    for t in trades:
        if t.get("status") != "open":
            continue
        tk    = t["ticker"].upper()
        px    = live_prices.get(tk) or float(t.get("entry_price", 0) or 0)
        s     = score_position(t, regime_ctx, portfolio_value, px, trades)
        scored.append(s)

    total_add    = sum(s["add_amount"]    for s in scored if s["add_amount"])
    total_reduce = sum(s["reduce_amount"] for s in scored if s["reduce_amount"])
    exits        = [s["ticker"] for s in scored if s["action"] == "EXIT"]

    parts = []
    if total_add    > 0: parts.append(f"Deploy ${total_add:,.0f}")
    if total_reduce > 0: parts.append(f"Trim ${total_reduce:,.0f}")
    if exits:            parts.append(f"Exit: {', '.join(exits)}")
    rebalance_summary = " · ".join(parts) if parts else "Portfolio balanced — no rebalancing needed"

    return {
        "positions":          scored,
        "total_add":          round(total_add),
        "total_reduce":       round(total_reduce),
        "exits":              exits,
        "rebalance_summary":  rebalance_summary,
        "portfolio_value":    portfolio_value,
    }
```

- [ ] **Step 1: Create `services/portfolio_sizing.py`** with all functions above. Full file — no tests required (pure math/data functions, validated visually in app).

- [ ] **Step 2: Verify import works**
  ```python
  # Quick sanity check — run in Python shell
  from services.portfolio_sizing import score_portfolio
  print("import ok")
  ```

---

## Chunk 2: Portfolio Intelligence UI Upgrade

### Task 2: Regime Strip (always-visible, no button press needed)

**Files:**
- Modify: `modules/trade_journal.py` — inside `with tab_intel:`, insert BEFORE "Section A — Context freshness panel"

Add a compact regime strip showing:
```
[ Stagflation · Score 42 · Stable 18 sessions ]  [ Growth ▼ | Inflation ▲ | Liquidity ▼ | Credit ▼ ]
```

```python
# ── Regime Strip ──────────────────────────────────────────────────────────
_rc = st.session_state.get("_regime_context") or {}
_quadrant = _rc.get("quadrant", "")
_regime_score = _rc.get("score", 0)
# Normalize score to 0-100 if it's -1..+1
if isinstance(_regime_score, float) and -1 <= _regime_score <= 1:
    _regime_score_100 = int((_regime_score + 1) * 50)
else:
    _regime_score_100 = int(_regime_score)

# Stability from regime_history.json
import json as _json, os as _os
_REGIME_HIST = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "regime_history.json")
_stability = 0
try:
    with open(_REGIME_HIST) as _f:
        _hist = _json.load(_f)
    if _hist:
        _hist = sorted(_hist, key=lambda r: r.get("date", ""))
        _cur_q = _hist[-1].get("quadrant", _hist[-1].get("regime", ""))
        for _hr in reversed(_hist):
            _hq = _hr.get("quadrant", _hr.get("regime", ""))
            if _hq == _cur_q:
                _stability += 1
            else:
                break
except Exception:
    pass

_qcolors = {
    "Stagflation": "#a855f7", "Goldilocks": "#22c55e",
    "Reflation": "#f59e0b", "Deflation": "#3b82f6",
}
_qc = _qcolors.get(_quadrant, "#888")
_stab_color = "#22c55e" if _stability >= 10 else ("#f59e0b" if _stability >= 5 else "#94a3b8")

if _quadrant or _rc:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;'
        f'background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;'
        f'padding:8px 16px;margin-bottom:12px;flex-wrap:wrap;">'
        f'<span style="color:{_qc};font-weight:700;font-size:13px;">{_quadrant or "—"}</span>'
        f'<span style="color:#888;font-size:11px;">Score {_regime_score_100}/100</span>'
        f'<span style="color:{_stab_color};font-size:11px;">· Stable {_stability} session{"s" if _stability != 1 else ""}</span>'
        f'<span style="color:#555;font-size:11px;">|</span>'
        f'<span style="color:#888;font-size:11px;">Run <b style="color:#fff;">Risk Regime</b> to refresh</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Run Risk Regime → Regime Overview to load macro context, then return here.")
```

- [ ] **Step 3: Add regime strip** at the top of `with tab_intel:` block (line ~576), before the Context Freshness section.

---

### Task 3: Wire sizing engine into per-position cards

**Files:**
- Modify: `modules/trade_journal.py` — Section D (per-position cards, ~line 746)

- [ ] **Step 4: Import and call scoring engine**

After `_pa_positions` is built (~line 747), add:

```python
# Compute institutional sizing scores for all open positions
from services.portfolio_sizing import score_portfolio as _score_portfolio
_total_portfolio_val = sum(
    (_pi_live.get(t["ticker"], t["entry_price"]) * t["position_size"])
    for t in open_trades
)
_sizing_result = {}
if _total_portfolio_val > 0:
    _sz = _score_portfolio(open_trades, _rc, _total_portfolio_val, _pi_live)
    _sizing_result = {p["ticker"]: p for p in _sz["positions"]}
    _sz_summary = _sz
else:
    _sz_summary = None
```

- [ ] **Step 5: Upgrade per-position card HTML**

In the per-position loop, after `_action` is retrieved from `_pos_data`, add the sizing overlay:

```python
_sz_pos = _sizing_result.get(_tk.upper(), {})
_sz_action  = _sz_pos.get("action", "")
_sz_score   = _sz_pos.get("composite_score", "—")
_sz_cw      = _sz_pos.get("current_weight")
_sz_tw      = _sz_pos.get("target_weight")
_sz_stop    = _sz_pos.get("atr_stop")
_sz_add     = _sz_pos.get("add_amount")
_sz_add_pct = _sz_pos.get("add_pct")
_sz_red     = _sz_pos.get("reduce_amount")
_sz_red_pct = _sz_pos.get("reduce_to_pct")
_sz_hold    = _sz_pos.get("hold_condition")
_sz_rf      = _sz_pos.get("regime_fit")
_sz_cv      = _sz_pos.get("conviction")

# Build action detail string
_action_detail = ""
if _sz_action == "ADD" and _sz_add:
    _action_detail = f'<b style="color:#22c55e;">ADD ${_sz_add:,} (+{_sz_add_pct:.1f}%)</b>'
elif _sz_action == "REDUCE" and _sz_red:
    _action_detail = f'<b style="color:#f59e0b;">REDUCE by ${_sz_red:,} → {_sz_red_pct:.1f}% weight</b>'
elif _sz_action == "EXIT":
    _action_detail = f'<b style="color:#ef4444;">EXIT</b>'
elif _sz_action == "HOLD" and _sz_hold:
    _action_detail = f'<b style="color:#22c55e;">HOLD</b> <span style="color:#888;font-size:10px;">· {_sz_hold}</span>'

# Sizing row HTML
_sizing_html = ""
if _sz_pos:
    _score_color = "#22c55e" if _sz_score >= 65 else ("#f59e0b" if _sz_score >= 40 else "#ef4444")
    _weight_html = ""
    if _sz_cw is not None and _sz_tw is not None:
        _wdelta = _sz_tw - _sz_cw
        _wdc = "#22c55e" if _wdelta > 0 else ("#ef4444" if _wdelta < 0 else "#888")
        _weight_html = (
            f'<span style="color:#888;font-size:10px;">'
            f'Wt: {_sz_cw:.1f}% → '
            f'<span style="color:{_wdc};">{_sz_tw:.1f}%</span>'
            f'</span>'
        )
    _stop_html = f'<span style="color:#94a3b8;font-size:10px;">Stop ${_sz_stop:.2f}</span>' if _sz_stop else ""
    _score_html = (
        f'<span style="background:{_score_color}22;border:1px solid {_score_color}44;'
        f'border-radius:3px;padding:1px 6px;font-size:10px;color:{_score_color};">'
        f'Score {_sz_score}</span>'
    )
    _factors_html = ""
    if _sz_rf is not None and _sz_cv is not None:
        _factors_html = (
            f'<span style="color:#64748b;font-size:10px;margin-left:6px;">'
            f'RegimeFit {_sz_rf} · Conviction {_sz_cv}</span>'
        )
    _sizing_html = (
        f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #222;'
        f'display:flex;flex-wrap:wrap;gap:8px;align-items:center;">'
        f'{_score_html}{_factors_html}'
        f'<span style="color:#555;">|</span>'
        f'{_weight_html}'
        f'{"&nbsp;·&nbsp;" + _stop_html if _stop_html else ""}'
        f'</div>'
        f'<div style="margin-top:4px;font-size:12px;">{_action_detail}</div>'
        if _action_detail else
        f'</div>'
    )
```

Then inject `{_sizing_html}` at the end of each position card's `st.markdown(...)` HTML, before the closing `</div>`.

---

### Task 4: Rebalance Summary strip

**Files:**
- Modify: `modules/trade_journal.py` — after the per-position loop, inside Section D

```python
if _sz_summary:
    _tot_add = _sz_summary.get("total_add", 0)
    _tot_red = _sz_summary.get("total_reduce", 0)
    _exits   = _sz_summary.get("exits", [])
    _rb_sum  = _sz_summary.get("rebalance_summary", "")
    _rb_color = "#ef4444" if _exits else ("#f59e0b" if _tot_red > 0 else "#22c55e")
    st.markdown(
        f'<div style="background:#0E1E2E;border:1px solid {_rb_color}44;border-radius:6px;'
        f'padding:10px 16px;margin-top:12px;">'
        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;'
        f'letter-spacing:0.08em;">REBALANCE SUMMARY · </span>'
        f'<span style="color:#ccc;font-size:12px;">{_rb_sum}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 6: Add rebalance summary** after the per-position loop closes.

---

### Task 5: Remove Macro Scorecard from sidebar

**Files:**
- Modify: `app.py`

- [ ] **Step 7: Remove "Macro Scorecard"** from the sidebar options list and its `elif` routing block.

---

## Chunk 3: AI Tier Selector to Top

### Task 6: Elliott Wave — move tier selector above chart

**Files:**
- Modify: `modules/elliott_wave.py`

Currently the `st.radio(key="ew_narrative_tier")` lives inside the "Elliott Wave AI Narrative" expander at the bottom of the page. The Claude analysis already reads from session_state before the chart renders, but the widget to change it is buried.

**Target position:** Add the radio immediately below the interval selector row (after `chart_height = st.slider(...)`, before `st.button("Refresh Data")`).

- [ ] **Step 8: Add top-level tier selector in `modules/elliott_wave.py`**

Insert after the `chart_height` slider (~line 784):

```python
# ── AI Engine Tier — controls wave count analysis and narrative ────────────
_has_anthropic_ew = bool(os.getenv("ANTHROPIC_API_KEY"))
_ew_tier_top = st.radio(
    "AI Engine",
    ["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"],
    index=["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"].index(
        st.session_state.get("ew_narrative_tier", "⚡ Standard")
    ),
    horizontal=True,
    key="ew_narrative_tier",   # same key — stays in sync with expander
    disabled=not _has_anthropic_ew,
    help="Standard = Groq LLaMA · Regard = Claude Haiku · Highly Regarded = Claude Sonnet (overrides wave count)",
)
if not _has_anthropic_ew and _ew_tier_top != "⚡ Standard":
    st.caption("Set ANTHROPIC_API_KEY to unlock Regard modes.")
```

- [ ] **Step 9: Remove the duplicate radio from inside the AI Narrative expander**

In the AI Narrative expander (~line 1019), replace the `st.radio(key="ew_narrative_tier")` block with a status label:

```python
# Replace the radio widget with a read-only status label
_ew_tier = st.session_state.get("ew_narrative_tier", "⚡ Standard")
st.caption(f"Engine: {_ew_tier} — change at top of page")
```

Remove lines:
```python
_ew_tier_options = ["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
_ew_tier_map = { ... }
_ew_tier = st.radio("AI Engine", _ew_tier_options, ...)
if not _has_anthropic and _ew_tier != "⚡ Standard": ...
```

Keep `_ew_tier_map` definition so `_ew_use_claude, _ew_model = _ew_tier_map[_ew_tier]` still works.

---

### Task 7: Wyckoff — move tier selector above chart

**Files:**
- Modify: `modules/wyckoff.py`

Currently `st.radio(key="wy_narrative_tier")` lives inside the Wyckoff AI Narrative expander. Move it to just after `st.button("Refresh Data")` (~line 661), before the ticker guard.

- [ ] **Step 10: Add top-level tier selector in `modules/wyckoff.py`**

Insert after `st.button("Refresh Data")` but BEFORE `if not ticker: return`:

```python
# ── AI Engine Tier ─────────────────────────────────────────────────────────
_has_anthropic_wy = bool(os.getenv("ANTHROPIC_API_KEY"))
_wy_tier_top = st.radio(
    "AI Engine",
    ["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"],
    index=["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"].index(
        st.session_state.get("wy_narrative_tier", "⚡ Standard")
    ),
    horizontal=True,
    key="wy_narrative_tier",
    disabled=not _has_anthropic_wy,
    help="Standard = Groq LLaMA · Regard = Claude Haiku · Highly Regarded = Claude Sonnet (overrides Wyckoff phase)",
)
if not _has_anthropic_wy and _wy_tier_top != "⚡ Standard":
    st.caption("Set ANTHROPIC_API_KEY to unlock Regard modes.")
```

- [ ] **Step 11: Remove duplicate radio from Wyckoff AI Narrative expander**

Same pattern as Elliott Wave — replace the `st.radio(key="wy_narrative_tier")` block with:

```python
_wy_tier = st.session_state.get("wy_narrative_tier", "⚡ Standard")
st.caption(f"Engine: {_wy_tier} — change at top of page")
```

---

## Verification Checklist

- [ ] Open My Regarded Portfolio → Portfolio Intelligence tab → regime strip visible at top showing quadrant + score + stability
- [ ] Click "Run Portfolio Analysis" → per-position cards show `Score 78`, `Wt: 8.2% → 11.5%`, `ADD $1,240 (+3.3%)`, `Stop $54.20`
- [ ] Rebalance Summary row appears after last position card
- [ ] Macro Scorecard no longer appears in sidebar
- [ ] Elliott Wave page: tier radio visible at top, before chart renders
- [ ] Wyckoff page: same
- [ ] In Standard mode: wave count is algorithm-only (Groq narrative)
- [ ] In Regard/Highly Regarded mode: `✦ CLAUDE WAVE OVERRIDE` badge appears on metrics
- [ ] Graceful degradation: if ANTHROPIC_API_KEY missing, radios disabled with caption
- [ ] Graceful degradation: if yfinance ATR fetch fails, score defaults to 50 and action logic still works
