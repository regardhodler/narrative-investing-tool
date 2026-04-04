# QIR Dashboard Layout Redesign

**Date:** 2026-04-04
**Status:** Approved

## Problem

The current QIR Intelligence Dashboard card mixes signals (Regime, Tactical, Options Flow) into a compressed "Timing Stack" one-liner column alongside unrelated columns (Macro Events, Earnings Risk). The Entry Recommendation card (Buy the Dip / Wait / Hold / Sell the Rip) is buried below the pattern verdict and conviction bar. The reading order is: header → signals (tiny) → pattern verdict → conviction → warnings → recommendation → buy/short detail. This doesn't match how a trader thinks: signals first, then what to do, then the detail.

## Goal

Reorganize `_render_qir_dashboard()` in `modules/quick_run.py` into a clear top-to-bottom reading flow:

1. **Signals** — what the market is saying (Regime / Tactical / Options Flow)
2. **Recommendation** — what to do about it (Buy the Dip / Wait / Hold / Sell the Rip)
3. **Pattern detail** — the full pattern + conviction + buy/short panels
4. **Context footer** — Macro Events + Earnings Risk compact inline
5. **Judge Judy debate** — already positioned below the card; no change needed

## Layout Design

### Zone 1 — Signal Strip (full width, 3 equal cells)

Replaces the current `_t1` (Timing Stack) column. The 3-col grid now spans full width and each cell is a self-contained signal panel.

**Regime cell:**
- Label: `📡 REGIME`
- Direction arrow + label: `▲ Risk-On` (colored)
- `Composite   61 /100`
- `Leading     73 /100`
- Divergence badge: `+12 pts · EARLY RISK-ON` (green pill) or `-8 pts · EARLY RISK-OFF` (red pill) or `Aligned` (gray)

The current standalone leading-divergence warning bar (`_leading_warning`) is removed — absorbed into this cell.

**Tactical cell:**
- Label: `⚡ TACTICAL`
- Direction arrow + label: `▼ Oversold` (colored)
- `Score   38 /100`
- Key signal lines from `_tac["signals"]`: match by `s["Signal"]` string (e.g. `"SMA Cross"`, `"RSI"`) — do NOT use index, order is not guaranteed
- VIX note if present (replaces the current standalone vix_note bar): extract VIX value by matching `s["Signal"]` against the VIX signal name string — do NOT use index 0, same rule as SMA/RSI

**Options Flow cell:**
- Label: `📊 OPTIONS FLOW`
- Direction arrow + label: `◆ Neutral` (colored)
- `Score   55 /100`
- P/C Ratio: `_of.get("pc_ratio")`
- Gamma zone text: extracted from `_of.get("signals", [])` by matching `s["Signal"] == "Gamma Zone"`, not by index
- Unusual call%: from `st.session_state.get("_unusual_activity_sentiment", {}).get("call_pct")` — this is a separate session key, not in `_of`

**Placeholder state (pre-run):** When `_rc`, `_tac`, `_of` are empty (QIR not yet run), each cell shows a dim one-liner consistent with the existing pattern: `◌ 📡 Regime — run QIR`, `◌ ⚡ Tactical — run QIR`, `◌ 📊 Opt Flow — run QIR` in `color:#374151`. The cell borders are still rendered but use the muted `#1e293b` border color. No scores or divergence badge shown. This matches the current greyed shell behavior.

**Options Flow third state:** If `_of.get("data_unavailable")` is true (market closed), show `◆ 📊 Opt Flow — market closed` in `color:#475569` — distinct from the run-QIR placeholder color `#374151`. This preserves the existing three-state logic for the Options Flow cell.

### Zone 2 — Entry Recommendation (full width)

The existing `_entry_rec` card, promoted to full-width prominence directly below the signal strip. No structural change to `_classify_entry_recommendation()` — only position changes.

Layout:
```
ENTRY SIGNAL                           [9px label]
▲  BUY THE DIP                         [20px bold, verdict color]
Leading 73/100 · Macro 61/100 · +12 pts [EARLY RISK-ON badge]
▌ Reasoning text (2 sentences)
```

### Zone 3 — Pattern + Conviction + Buy/Short (unchanged content)

- Pattern label + interpretation text
- Conviction score bar
- Buy setup panel | Short setup panel (2-col grid)
- Earnings caveats

### Zone 4 — Compact footer (single line)

Macro Events and Earnings Risk condensed into one horizontal line:

```
FOMC 12d · CPI 5d · NFP 22d  |  ⚠ NVDA 3d ±4.2%  AAPL 7d ±2.1%
```

- Macro events: FOMC, CPI, NFP with days-away colored by urgency (red=today, orange=tomorrow, yellow=≤5d, gray=beyond)
- Earnings: up to 3 tickers with days and expected move %, separator `|` between the two groups
- If no earnings: only the macro events line, no separator
- If a single macro event fetch fails, show `FOMC —` (dash) for that event, consistent with current `_t2` fallback behavior — do not omit the event entirely

## Implementation Scope

**One file only:** `modules/quick_run.py`

### Functions to change

| Function | Change |
|---|---|
| `_render_qir_dashboard()` | Restructure `_verdict_html` assembly: signal strip → entry rec → pattern → footer |
| `_t1`, `_t2`, `_t3` HTML strings | Replace `_t1` with richer regime cell; `_t2`/`_t3` repurposed into compact footer |

### Functions unchanged

| Function | Reason |
|---|---|
| `_classify_entry_recommendation()` | Logic untouched; only its position in the HTML changes |
| `_classify_signals()` | No change |
| `_build_uncertainty_profile()` | No change |
| `_render_genuine_uncertainty_panel()` | No change |
| All debate/Judge Judy rendering | Stays below the card as-is |

## Data Sources (all already in session state)

| Signal | Session key | Fields used |
|---|---|---|
| Regime composite | `_regime_context` | `score`, `regime`, `macro_score` |
| Regime leading | `_regime_context` | `leading_score`, `leading_divergence`, `leading_label` |
| Tactical | `_tactical_context` | `tactical_score`, `label`, `signals` (SMA, RSI) |
| Options flow | `_options_flow_context` | `options_score`, `label`, `pc_ratio`; gamma zone from `signals` list by name match; call% from `_unusual_activity_sentiment` session key |
| Macro events | `fed_forecaster` service | `get_next_fomc`, `get_next_cpi`, `get_next_nfp` |
| Earnings risk | `_qir_earnings_risk` | `ticker`, `days_away`, `expected_move_pct` |

## Removed Elements

- Standalone `_leading_warning` orange bar → merged into Regime cell divergence badge
- Standalone `_vix_note` bar → merged into Tactical cell
- Macro Events and Earnings Risk as standalone columns in the 3-col header grid → replaced by compact footer line

## Visual Style

Consistent with existing dark theme:
- Cell borders: `1px solid #1e293b`, `border-radius:5px`, `padding:8px 10px`
- Orange labels: `color:#f59e0b`, `font-size:9px`, `font-weight:700`, `letter-spacing:0.08em`
- Values: `color:#f1f5f9`, `font-size:13px`, `font-weight:800`
- Divergence pills: green `#22c55e` bg for risk-on, red `#ef4444` bg for risk-off, gray `#1e293b` for aligned
- Monospace font: `'JetBrains Mono', Consolas, monospace` for scores

## Verification

1. Run `streamlit run app.py`, navigate to QIR
2. Before running QIR: card shows greyed placeholder state in all 4 zones
3. Run QIR: verify signal strip populates with correct values matching regime/tactical/options sections elsewhere in the app
4. Verify leading divergence is no longer a floating orange bar — only shown in Regime cell
5. Verify VIX note is no longer a floating bar — only shown in Tactical cell
6. Verify Entry Recommendation card appears directly below signal strip
7. Verify Macro Events + Earnings Risk appear as compact single-line footer
8. Verify Judge Judy debate section is visually below the recommendation (unchanged position)
