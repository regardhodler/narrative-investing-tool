# QIR Intelligence Dashboard — Design Spec
**Date:** 2026-03-29
**Status:** Approved

---

## Overview

Replace the current two-column Short/Buy panels in Quick Intel Run with a single fused **QIR Intelligence Dashboard** — a command-center panel that synthesizes all timing signals, macro events, earnings risk, and buy/short guidance into one glanceable view.

The dashboard is always rendered in the persistent QIR section (immediately below the Score Interpretation Reference expander, above the Run QIR button): greyed-out shell before QIR runs, fully activated (glowing border) after.

---

## Layout

```
┌─ QIR INTELLIGENCE DASHBOARD ─────────────────────────────────────────┐
│                                                                        │
│  TIMING STACK          MACRO EVENTS           EARNINGS RISK           │
│  ▲ Regime  Risk-On     FOMC  12d  Apr 10      ⚠ NVDA   3d  ±8.2%    │
│  ▲ Tactical  71/100    CPI    4d  Apr 2       📅 TSLA   9d  ±6.1%    │
│  ▲ Opt Flow  68/100    NFP   18d  Apr 17      ✓ AMD   22d  ±5.4%     │
│                                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                        │
│  PULLBACK IN UPTREND — TACTICAL WEAKNESS IS A BUY-THE-DIP SETUP      │
│  Regime and Options Flow confirm bull trend. Tactical dip = entry.    │
│                                                                        │
│  BUY SETUP                      SHORT SETUP                           │
│  ▶ STRONG BUYING ENV            ▶ NOT A SHORTING ENV                  │
│  Instruments: QQQ, SPY, XLK     No short edge — trend is up.         │
│  Entry: Scale on red days,      Avoid fighting the regime.            │
│  stop below 20d MA              If forced: very tight stops only      │
│                                                                        │
│  ⚠ NVDA earnings in 3d — size to survive ±8.2% gap before print     │
└──────────────────────────────────────────────────────────────────────┘
```

**Shell state (before QIR):** Dark border `#1e293b`, all values `—`, footer reads "Run QIR to activate"
**Active state (after QIR):** Border color reflects verdict — green (bullish), red (bearish), amber (divergent), grey (uncertain)

**Render position:** Immediately below the Score Interpretation Reference expander (the `📖 How to read...` expander), above the Run QIR button. This replaces the current `if _sh_rc or _sh_tac or _sh_of:` block at line ~213 of `quick_run.py`.

---

## Signal Classification Logic

Helper function `_classify_signals(regime_ctx, tac_ctx, of_ctx)` in `quick_run.py`.

**Direction thresholds** — identical to the existing Short/Buy panel logic at lines 221–226:
- Regime: `"Risk-On" in label OR score > 0.3` = bullish; `"Risk-Off" in label OR score < -0.3` = bearish
- Tactical: `tactical_score >= 65` = bullish; `< 38` = bearish; else neutral
- Options Flow: `options_score >= 65` = bullish; `< 38` = bearish; else neutral

Note: `_hr_gate_check()` continues to use `score < -0.3` unchanged — these thresholds are already consistent.

**Seven patterns:**

| Pattern | Condition | Label | Border |
|---------|-----------|-------|--------|
| All 3 bullish | R▲ T▲ OF▲ | BULLISH CONFIRMATION | `#22c55e` |
| All 3 bearish | R▼ T▼ OF▼ | BEARISH CONFIRMATION | `#ef4444` |
| Regime ▲, Tactical ▼, OptFlow ▲ | R▲ T▼ OF▲ | PULLBACK IN UPTREND | `#f59e0b` |
| Regime ▲, Tactical ▲, OptFlow ▼ | R▲ T▲ OF▼ | OPTIONS FLOW DIVERGENCE | `#f59e0b` |
| Regime ▼, Tactical ▲, OptFlow ▲ | R▼ T▲ OF▲ | BEAR MARKET BOUNCE | `#f97316` |
| Regime ▼, Tactical ▼, OptFlow ▲ | R▼ T▼ OF▲ | LATE CYCLE SQUEEZE | `#ef4444` |
| All others | — | GENUINE UNCERTAINTY | `#475569` |

**Returns dict:**
```python
{
  "pattern": str,
  "label": str,
  "color": str,           # hex border color
  "interpretation": str,  # one sentence
  "buy_tier": str,        # STRONG / MODERATE / SELECTIVE / NOT A BUYING ENV
  "short_tier": str,      # STRONG / MODERATE / SELECTIVE / NOT A SHORTING ENV
  "instruments_buy": list[str],
  "instruments_short": list[str],
  "entry_buy": str,
  "entry_short": str,
}
```

---

## Pattern Interpretations & Instruments

### BULLISH CONFIRMATION
- **Interpretation:** All three timing layers aligned bullish — highest-conviction long entry.
- **Buy:** STRONG | QQQ, SPY, XLK, high-beta growth | Buy breakouts, add on dips, stop below 50d MA
- **Short:** NOT A SHORTING ENV | Avoid fighting the trend | If forced: only extreme overbought bounces, very tight stops

### BEARISH CONFIRMATION
- **Interpretation:** All three layers aligned bearish — highest-conviction short or cash environment.
- **Buy:** NOT A BUYING ENV | Stay cash or hedge | No new longs until at least Tactical turns
- **Short:** STRONG | SH, PSQ, SQQQ (small size), puts on weak sectors | Sell bounces, stop above last swing high

### PULLBACK IN UPTREND
- **Interpretation:** Regime and Options Flow confirm bull trend — Tactical dip is a buy-the-dip setup.
- **Buy:** STRONG | QQQ, SPY, sector leaders | Scale in on red days, stop below 20d MA, target previous high
- **Short:** NOT A SHORTING ENV | Trend is up — pullbacks are entries, not reversals

### OPTIONS FLOW DIVERGENCE
- **Interpretation:** Regime and Tactical bullish but options crowd is hedging — smart money buying protection.
- **Buy:** MODERATE | SPY, defensive growth (MSFT, GOOGL) | Smaller size, wait for Options Flow to confirm ≥65
- **Short:** SELECTIVE | Only weakest sector ETFs | Treat as hedge, not directional trade

### BEAR MARKET BOUNCE
- **Interpretation:** Short-term momentum against the macro trend — take profits quickly, don't chase.
- **Buy:** SELECTIVE | Short-term momentum only (QQQ, ARKK) | Tight stops, take 50% profits at first resistance
- **Short:** MODERATE | SH, weak sectors (XLE, XLF) | Wait for bounce to fade, sell into strength

### LATE CYCLE SQUEEZE
- **Interpretation:** Options crowd squeezing higher against a bearish regime and tactical — high risk of reversal.
- **Buy:** NOT A BUYING ENV | Don't chase the squeeze | Wait for regime to confirm before entering longs
- **Short:** STRONG | Build short position in tranches | Stop above squeeze high, target regime-implied support

### GENUINE UNCERTAINTY
- **Interpretation:** No edge — signals are conflicting with no clear majority direction.
- **Buy:** NOT A BUYING ENV | Wait for 2 of 3 layers to align
- **Short:** NOT A SHORTING ENV | No directional edge in either direction

---

## Earnings Risk

**New session state key:** `_qir_earnings_risk` — list of dicts for held positions with earnings ≤21 days:
```python
[{"ticker": "NVDA", "days_away": 3, "expected_move_pct": 8.2, "expected_move_dollar": 46.0, "date": "Apr 1"}]
```

**Data source:** `fetch_earnings_intelligence(ticker)` from `services/market_data.py` (not `fetch_earnings_date`). Uses `result["next_earnings"]` for date/days and `result["expected_move"]` for pct/dollar. This function is `@st.cache_data(ttl=3600)` — safe to call per ticker in Round 5 without hitting API limits.

**Fetched in QIR Round 5** — appended to existing `run_quick_risk_snapshot` call. Loops open trades from `load_journal()`, calls `fetch_earnings_intelligence` per unique ticker, filters to `days_away <= 21`.

**Display thresholds in Earnings Risk column:**
- `days_away <= 3`: ⚠ red `#ef4444` — high urgency
- `days_away <= 7`: ⚠ amber `#f59e0b` — caution
- `days_away <= 21`: 📅 grey `#475569` — on radar

**Footer earnings caveat** appended to verdict section for any position with `days_away <= 14`:
> "⚠ {TICKER} earnings in {N}d — size to survive ±{X}% gap before the print"

Positions in the 15–21 day window appear in the Earnings Risk panel column only, not in the footer caveat.

**Downstream injection — Portfolio Intelligence:**
`_qir_earnings_risk` is formatted as a text string and added to `_upstream["earnings_risk"]` in `trade_journal.py`. The `_upstream` dict is passed as context to the AI prompt function (`analyze_portfolio` / portfolio intelligence AI call). The AI reads it as text — no schema change needed in `scoring.py`, as unknown dict keys are already formatted into the prompt as-is.

**Downstream injection — Valuation:**
If the currently selected ticker appears in `_qir_earnings_risk`, inject into `signals_text`:
```
Earnings Risk: {ticker} reports in {N}d, options pricing ±{X}% move (${Y})
```

---

## Macro Events Column

Calls existing `get_next_fomc()`, `get_next_cpi()`, `get_next_nfp()` from `services/fed_forecaster` inline during dashboard render — no new session state needed.

**Color thresholds** match existing `render_macro_events()` in `utils/components.py` exactly:
- 0 days → red `#ef4444`
- 1 day → orange `#f97316`
- ≤5 days → amber `#f59e0b`
- else → grey `#475569`

The dashboard does NOT call `render_macro_events()` directly (it would render a full Streamlit block in the wrong layout context). Instead, the dashboard renders its own inline cells using the same color logic.

**QIR AI prompt injection:** Next FOMC/CPI/NFP dates added to `_sig_parts` (Round 4 completion block in `quick_run.py`) so AI-generated plays are calendar-aware.

---

## Files Changed

| File | Change |
|------|--------|
| `modules/quick_run.py` | Remove Short/Buy columns; add `_render_qir_dashboard()` and `_classify_signals()` helpers; fetch earnings in Round 5; inject macro events into `_sig_parts` |
| `services/signals_cache.py` | Add `_qir_earnings_risk`, `_qir_earnings_risk_ts` to `_SIGNAL_KEYS` |
| `modules/valuation.py` | Inject earnings risk into `signals_text` for current ticker |
| `modules/trade_journal.py` | Add `earnings_risk` text string to Portfolio Intelligence `_upstream` |

---

## What Gets Removed

- The `if _sh_rc or _sh_tac or _sh_of:` block and everything inside it (~lines 213–450 of `quick_run.py`)
- `_layer_html_row()` helper function
- Duplicate layer status block that was rendered above the two columns

---

## Success Criteria

1. Dashboard shell renders greyed-out on page load with no QIR data
2. After QIR runs, border glows with correct verdict color
3. All 7 signal patterns produce correct label, interpretation, buy/short tiers
4. Earnings risk correctly flags held positions with earnings ≤21 days (footer warning ≤14 days)
5. Macro events column shows live FOMC/CPI/NFP countdowns with correct color thresholds
6. Downstream: Portfolio Intelligence and Valuation receive earnings risk context as text
7. `_hr_gate_check()` threshold (`-0.3`) unchanged
