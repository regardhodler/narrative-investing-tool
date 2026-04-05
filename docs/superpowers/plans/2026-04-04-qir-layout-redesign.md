# QIR Dashboard Layout Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `_render_qir_dashboard()` so the reading order is Signal Strip → Entry Recommendation → Pattern Detail → Footer, replacing the current mixed-column layout.

**Architecture:** Pure HTML string refactor inside one function in one file. No logic changes — `_classify_entry_recommendation()` and `_classify_signals()` are untouched. The new layout builds four HTML string variables (`_sig_strip_html`, `_entry_rec_html`, `_verdict_html`, `_footer_html`) and assembles them in order in the final `st.markdown` call.

**Tech Stack:** Streamlit, inline HTML/CSS (dark theme), Python f-strings.

**Spec:** `docs/superpowers/specs/2026-04-04-qir-layout-redesign.md`

---

## File Map

| File | Change |
|---|---|
| `modules/quick_run.py` | Only file modified. All changes inside `_render_qir_dashboard()`. |

**Signal names confirmed from source (do not guess):**

Tactical signals (`_tac["signals"]`, match by `s["Signal"]`):
- `"VIX Level + 5d Trend"` — VIX value for note
- `"SPY vs 20d/50d MA"` — price vs moving averages
- `"SPY Momentum (5d vs 20d ROC)"` — momentum direction

Options flow signals (`_of["signals"]`, match by `s["Signal"]`):
- `"P/C Ratio"` — put/call ratio
- `"Gamma Zone"` — dealer gamma positioning

Call% source: `st.session_state.get("_unusual_activity_sentiment", {}).get("call_pct")` — NOT in `_of`.

---

## Chunk 1: Zone 1 — Signal Strip

### Task 1: Build `_sig_strip_html`

**Files:**
- Modify: `modules/quick_run.py` — inside `_render_qir_dashboard()`, after line 656 (after `_of_score` is set), replacing the `_t1` block (lines 658–693)

- [ ] **Step 1: Extract all data needed for the three signal cells**

Replace the `_t1` block (lines 658–693) and the `_t2` / `_t3` blocks (lines 695–734) with this data extraction. Keep `_regime_score`, `_regime_label`, `_tac_score`, `_of_score` — they are still used downstream.

```python
# ── Zone 1: Signal Strip data ─────────────────────────────────────────
_macro_s   = int(_rc.get("macro_score") or 50)
_leading_s = int(_rc.get("leading_score") or 50)
_div_pts   = int(_rc.get("leading_divergence") or 0)
_div_label = _rc.get("leading_label") or "Aligned"

# Tactical: find signals by name, not index
_tac_sigs  = _tac.get("signals", []) if _tac else []
_spy_ma_sig = next((s for s in _tac_sigs if "SPY vs" in s["Signal"]), None)
_roc_sig    = next((s for s in _tac_sigs if "Momentum" in s["Signal"]), None)
_vix_sig    = next((s for s in _tac_sigs if "VIX Level" in s["Signal"]), None)

# Options: gamma by name; call% from separate session key
_of_sigs    = _of.get("signals", [])
_gamma_sig  = next((s for s in _of_sigs if s["Signal"] == "Gamma Zone"), None)
_ua_sent    = st.session_state.get("_unusual_activity_sentiment") or {}
_call_pct   = _ua_sent.get("call_pct")
```

- [ ] **Step 2: Build the Regime cell HTML**

```python
def _cell(label_icon, content_html, border_color="#1e293b"):
    return (
        f'<div style="background:#0d1117;border:1px solid {border_color};'
        f'border-radius:5px;padding:8px 10px;">'
        f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">{label_icon}</div>'
        f'{content_html}'
        f'</div>'
    )

# Regime cell
if _rc:
    _r_bull  = "Risk-On" in _regime_label or _regime_score > 0.3
    _r_bear  = "Risk-Off" in _regime_label or _regime_score < -0.3
    _rc_col  = "#22c55e" if _r_bull else ("#ef4444" if _r_bear else "#f59e0b")
    _rc_arr  = "▲" if _r_bull else ("▼" if _r_bear else "◆")
    _div_sign = f"+{_div_pts}" if _div_pts >= 0 else str(_div_pts)
    _div_pill_bg = (
        "#22c55e" if _div_label == "Early Risk-On Setup" else
        "#ef4444" if _div_label == "Early Risk-Off Warning" else
        "#1e293b"
    )
    _div_pill_fg = "#052e16" if _div_pill_bg == "#22c55e" else "white"
    _div_badge = (
        f'<span style="background:{_div_pill_bg};color:{_div_pill_fg};'
        f'font-weight:800;font-size:8px;padding:1px 5px;border-radius:3px;">'
        f'{_div_sign} pts · {_div_label.upper()}</span>'
    )
    _regime_cell = _cell("📡 REGIME", (
        f'<div style="color:{_rc_col};font-size:12px;font-weight:800;'
        f'font-family:\'JetBrains Mono\',Consolas,monospace;margin-bottom:5px;">'
        f'{_rc_arr} {_regime_label}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:9px;color:#475569;">Composite</span>'
        f'<span style="font-size:12px;font-weight:800;color:#f1f5f9;">{_macro_s}/100</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="font-size:9px;color:#475569;">Leading</span>'
        f'<span style="font-size:12px;font-weight:800;color:#f1f5f9;">{_leading_s}/100</span>'
        f'</div>'
        f'{_div_badge}'
    ), border_color=_rc_col + "44")
else:
    _regime_cell = _cell("📡 REGIME",
        '<div style="color:#374151;font-size:11px;">◌ Regime — run QIR</div>')
```

- [ ] **Step 3: Build the Tactical cell HTML**

```python
# Tactical cell
if _tac:
    _t_bull = _tac_score >= 65; _t_bear = _tac_score < 38
    _tc     = "#22c55e" if _t_bull else ("#ef4444" if _t_bear else "#f59e0b")
    _ta     = "▲" if _t_bull else ("▼" if _t_bear else "◆")
    _tac_lines = (
        f'<div style="color:{_tc};font-size:12px;font-weight:800;'
        f'font-family:\'JetBrains Mono\',Consolas,monospace;margin-bottom:5px;">'
        f'{_ta} {_tac.get("label","")}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:9px;color:#475569;">Score</span>'
        f'<span style="font-size:12px;font-weight:800;color:#f1f5f9;">{_tac_score}/100</span>'
        f'</div>'
    )
    if _spy_ma_sig:
        _tac_lines += (
            f'<div style="font-size:10px;color:#64748b;padding:1px 0;">'
            f'<span style="color:#94a3b8;">SPY MA</span> '
            f'<span style="color:#e2e8f0;">{_spy_ma_sig["Value"]}</span></div>'
        )
    if _roc_sig:
        _tac_lines += (
            f'<div style="font-size:10px;color:#64748b;padding:1px 0;">'
            f'<span style="color:#94a3b8;">Momo</span> '
            f'<span style="color:#e2e8f0;">{_roc_sig["Value"]}</span></div>'
        )
    # VIX note (matched by name, not index)
    if _vix_sig:
        try:
            import re as _vre2
            _vm2 = _vre2.search(r"(\d+\.?\d*)", str(_vix_sig.get("Value", "")))
            if _vm2:
                _vix2 = float(_vm2.group(1))
                if _vix2 > 28:
                    _tac_lines += f'<div style="font-size:9px;color:#f59e0b;margin-top:3px;">⚠ VIX {_vix2:.0f} — premiums elevated</div>'
                elif _vix2 < 15:
                    _tac_lines += f'<div style="font-size:9px;color:#4B9FFF;margin-top:3px;">ℹ VIX {_vix2:.0f} — options cheap</div>'
        except Exception:
            pass
    _tactical_cell = _cell("⚡ TACTICAL", _tac_lines, border_color=_tc + "44")
else:
    _tactical_cell = _cell("⚡ TACTICAL",
        '<div style="color:#374151;font-size:11px;">◌ Tactical — run QIR</div>')
```

- [ ] **Step 4: Build the Options Flow cell HTML**

```python
# Options Flow cell
if _of and not _of.get("data_unavailable"):
    _o_bull = _of_score >= 65; _o_bear = _of_score < 38
    _oc     = "#22c55e" if _o_bull else ("#ef4444" if _o_bear else "#f59e0b")
    _oa     = "▲" if _o_bull else ("▼" if _o_bear else "◆")
    _of_lines = (
        f'<div style="color:{_oc};font-size:12px;font-weight:800;'
        f'font-family:\'JetBrains Mono\',Consolas,monospace;margin-bottom:5px;">'
        f'{_oa} {_of.get("label","")}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:9px;color:#475569;">Score</span>'
        f'<span style="font-size:12px;font-weight:800;color:#f1f5f9;">{_of_score}/100</span>'
        f'</div>'
    )
    _pc = _of.get("pc_ratio")
    if _pc is not None:
        _of_lines += (
            f'<div style="font-size:10px;padding:1px 0;">'
            f'<span style="color:#94a3b8;">P/C Ratio</span> '
            f'<span style="color:#e2e8f0;">{_pc:.2f}</span></div>'
        )
    if _gamma_sig:
        _of_lines += (
            f'<div style="font-size:10px;padding:1px 0;">'
            f'<span style="color:#94a3b8;">Gamma</span> '
            f'<span style="color:#e2e8f0;">{_gamma_sig["Value"]}</span></div>'
        )
    if _call_pct is not None:
        _of_lines += (
            f'<div style="font-size:10px;padding:1px 0;">'
            f'<span style="color:#94a3b8;">Unusual Call%</span> '
            f'<span style="color:#e2e8f0;">{_call_pct:.0f}%</span></div>'
        )
    _options_cell = _cell("📊 OPTIONS FLOW", _of_lines, border_color=_oc + "44")
elif _of.get("data_unavailable"):
    _options_cell = _cell("📊 OPTIONS FLOW",
        '<div style="color:#475569;font-size:11px;">◆ Opt Flow — market closed</div>')
else:
    _options_cell = _cell("📊 OPTIONS FLOW",
        '<div style="color:#374151;font-size:11px;">◌ Opt Flow — run QIR</div>')
```

- [ ] **Step 5: Assemble `_sig_strip_html`**

```python
_sig_strip_html = (
    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px;">'
    f'{_regime_cell}{_tactical_cell}{_options_cell}'
    f'</div>'
)
```

- [ ] **Step 6: Verify syntax — run Python import check**

```bash
cd "C:/Users/16476/claude projects/narrative-investing-tool"
python -c "import modules.quick_run"
```

Expected: no output (no errors).

- [ ] **Step 7: Commit**

```bash
git add modules/quick_run.py
git commit -m "feat(qir): Zone 1 signal strip — regime/tactical/options cells"
```

---

## Chunk 2: Zone 2 + Zone 3 Reorder

### Task 2: Promote Entry Rec card + clean up verdict HTML

**Files:**
- Modify: `modules/quick_run.py` — inside `if _populated:` block

The entry rec card is currently appended to `_verdict_html` (after the conviction bar). It needs to become a separate string `_entry_rec_html` built before `_verdict_html`, so it can be placed between the signal strip and the pattern verdict in the final render call.

Also: remove the standalone `_leading_warning` block and `_vix_note` block from `_verdict_html` — those are now in the signal cells.

- [ ] **Step 1: Extract entry rec card into `_entry_rec_html`**

Find the entry rec card block currently inside `_verdict_html` (the `# ── Entry Signal card ─────` section). Cut it out of `_verdict_html` and assign it to a new variable instead.

The block currently starts after `_entry_rec = _classify_entry_recommendation(...)` and is appended to `_verdict_html`. Change it to:

```python
# Build _entry_rec_html (Zone 2 — positioned before pattern verdict)
_er_color = _entry_rec["color"]
_er_bg    = _entry_rec["bg"]
_er_icon  = _entry_rec["icon"]
_er_verb  = _entry_rec["verdict"]
_er_rsn   = _entry_rec["reasoning"]
_er_ldg   = _entry_rec["leading_score"]
_er_mac   = _entry_rec["macro_score"]
_er_dpts  = _entry_rec["divergence_pts"]
_er_dlbl  = _entry_rec["divergence_label"]
_er_dsign = f"+{_er_dpts}" if _er_dpts >= 0 else str(_er_dpts)
_er_div_color = (
    "#22c55e" if _er_dlbl == "Early Risk-On Setup" else
    "#ef4444" if _er_dlbl == "Early Risk-Off Warning" else
    "#64748b"
)
_er_div_fg = "#052e16" if _er_div_color == "#22c55e" else "white"
_er_div_badge = (
    f'<span style="background:{_er_div_color};color:{_er_div_fg};'
    f'font-weight:800;font-size:8px;padding:1px 6px;border-radius:3px;letter-spacing:0.05em;">'
    f'{_er_dsign} pts · {_er_dlbl.upper()}</span>'
)
_entry_rec_html = (
    f'<div style="background:{_er_bg};border:1px solid {_er_color}44;'
    f'border-radius:6px;padding:10px 14px;margin:0 0 10px;">'
    f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">ENTRY SIGNAL</div>'
    f'<div style="font-size:20px;font-weight:900;color:{_er_color};'
    f'letter-spacing:0.04em;margin-bottom:8px;">{_er_icon} {_er_verb}</div>'
    f'<div style="display:grid;grid-template-columns:1fr 1fr 2fr;gap:8px;margin-bottom:8px;">'
    f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">LEADING</div>'
    f'<div style="font-size:15px;font-weight:800;color:#f1f5f9;">{_er_ldg}/100</div></div>'
    f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">MACRO</div>'
    f'<div style="font-size:15px;font-weight:800;color:#f1f5f9;">{_er_mac}/100</div></div>'
    f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">DIVERGENCE</div>'
    f'<div style="margin-top:2px;">{_er_div_badge}</div></div>'
    f'</div>'
    f'<div style="background:#0d1117;border-left:3px solid {_er_color}44;'
    f'padding:7px 10px;font-size:11px;color:#94a3b8;line-height:1.6;border-radius:0 3px 3px 0;">'
    f'{_er_rsn}</div>'
    f'</div>'
)
```

For the `else` branch (not populated): `_entry_rec_html = ""`

- [ ] **Step 2: Remove `_leading_warning` block from `_verdict_html`**

Delete this block entirely (it was lines ~870–876 before edits — find by content):

```python
        # Leading divergence warning
        if _leading_warning:
            _verdict_html += (
                f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                f'padding:6px 10px;font-size:10px;color:#f59e0b;margin-bottom:8px;">'
                f'⚠ {_leading_warning}</div>'
            )
```

Note: `_leading_warning = _cls.get("leading_warning")` at the top of the `if _populated:` block — keep that variable assignment; remove only the HTML block.

- [ ] **Step 3: Remove `_vix_note` block from `_verdict_html`**

Delete the old VIX note builder and its append block:

```python
        # VIX note
        _vix_note = ""
        try:
            import re as _vre
            _vix_raw = (_tac.get("signals") or [{}])[0].get("Value", "") if _tac else ""
            _vm = _vre.search(r"(\d+\.?\d*)", str(_vix_raw))
            if _vm:
                _vix = float(_vm.group(1))
                if _vix > 28:
                    _vix_note = f"⚠ VIX {_vix:.0f} — premiums elevated. Prefer ETFs over options."
                elif _vix < 15:
                    _vix_note = f"ℹ VIX {_vix:.0f} — low vol. Options cheap: calls over ETFs for leverage efficiency."
        except Exception:
            pass
```

And delete:

```python
        if _vix_note:
            _verdict_html += (
                f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                f'padding:5px 10px;font-size:10px;color:#f59e0b;margin-bottom:8px;">{_vix_note}</div>'
            )
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "import modules.quick_run"
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add modules/quick_run.py
git commit -m "feat(qir): Zone 2 entry rec card promoted; remove floating warning bars"
```

---

## Chunk 3: Zone 4 Footer + Final Assembly

### Task 3: Build `_footer_html` and rewire `st.markdown`

**Files:**
- Modify: `modules/quick_run.py` — replace `_t2`/`_t3` column data with footer builder; update final `st.markdown` call

- [ ] **Step 1: Build `_footer_html` from macro events + earnings**

Replace the `_t2` and `_t3` building blocks with a single footer string. Place this code where `_t2` used to be built (after the signal strip data extraction):

```python
# ── Zone 4: Compact footer (macro events + earnings) ─────────────────
_footer_parts = []
try:
    from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
    for _ev_label, _ev_fn in (("FOMC", get_next_fomc), ("CPI", get_next_cpi), ("NFP", get_next_nfp)):
        try:
            _ev = _ev_fn()
            _d  = _ev.get("days_away", 99)
            _dt = _ev.get("date", "")[:6]
            if   _d == 0: _ec = "#ef4444"; _ds = "TODAY"
            elif _d == 1: _ec = "#f97316"; _ds = "TMRW"
            elif _d <= 5: _ec = "#f59e0b"; _ds = f"{_d}d"
            else:         _ec = "#475569"; _ds = f"{_d}d"
            _footer_parts.append(
                f'<span style="color:#64748b;">{_ev_label}</span>'
                f'<span style="color:{_ec};"> {_ds}</span>'
                f'<span style="color:#475569;font-size:9px;"> {_dt}</span>'
            )
        except Exception:
            _footer_parts.append(f'<span style="color:#374151;">{_ev_label} —</span>')
except Exception:
    _footer_parts.append('<span style="color:#374151;">Macro events unavailable</span>')

_earn_parts = []
if _er:
    for _e in _er[:3]:
        _ed = _e["days_away"]
        _em = _e.get("expected_move_pct")
        _em_str = f" ±{_em:.1f}%" if _em else ""
        _ec2 = "#ef4444" if _ed <= 3 else ("#f59e0b" if _ed <= 7 else "#475569")
        _eicon = "⚠" if _ed <= 7 else "📅"
        _earn_parts.append(
            f'<span style="color:{_ec2};">{_eicon} {_e["ticker"]}</span>'
            f'<span style="color:#64748b;"> {_ed}d</span>'
            f'<span style="color:{_ec2};font-size:9px;">{_em_str}</span>'
        )

_macro_inline = (
    f'<span style="color:#475569;font-size:8px;font-weight:700;'
    f'letter-spacing:0.08em;margin-right:6px;">EVENTS</span>'
    + ' · '.join(_footer_parts)
)
if _earn_parts:
    _macro_inline += (
        f'<span style="color:#1e293b;margin:0 8px;">│</span>'
        f'<span style="color:#475569;font-size:8px;font-weight:700;'
        f'letter-spacing:0.08em;margin-right:6px;">EARNINGS</span>'
        + '  '.join(_earn_parts)
    )

_footer_html = (
    f'<div style="border-top:1px solid #1e293b;margin-top:8px;padding-top:6px;'
    f'font-size:10px;font-family:\'JetBrains Mono\',Consolas,monospace;'
    f'display:flex;flex-wrap:wrap;gap:4px;align-items:center;">'
    f'{_macro_inline}'
    f'</div>'
)
```

- [ ] **Step 2: Delete dead `_t1`, `_t2`, `_t3` builder blocks**

These are now replaced by `_sig_strip_html` and `_footer_html`. Delete the following blocks entirely from `_render_qir_dashboard()`:

- The `_t1` block (starts with `_t1 = '<div style="margin-bottom:2px...TIMING STACK...'`, includes the regime/tactical/options one-liners)
- The `_t2` block (starts with `_t2 = '<div style="margin-bottom:2px...MACRO EVENTS...'`)
- The `_t3` block (starts with `_t3 = '<div style="margin-bottom:2px...EARNINGS RISK...'`)

These variables are no longer referenced anywhere after the `st.markdown` is rewired.

- [ ] **Step 3: Update the final `st.markdown` call**

Find the `st.markdown(...)` call that renders the outer card (currently line ~1041). Replace the inner template:

**Old structure:**
```python
    st.markdown(
        f'...'
        f'{_freshness_html}'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
        f'<div>{_t1}</div><div>{_t2}</div><div>{_t3}</div>'
        f'</div>'
        f'{_verdict_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
```

**New structure:**
```python
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid {_border_color};border-radius:8px;'
        f'box-shadow:{_border_glow};padding:14px 16px;margin:8px 0 12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;'
        f'text-transform:uppercase;">QIR Intelligence Dashboard</div>'
        f'</div>'
        f'{_freshness_html}'
        f'{_sig_strip_html}'
        f'{_entry_rec_html}'
        f'{_verdict_html}'
        f'{_footer_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "import modules.quick_run"
```

Expected: no output.

- [ ] **Step 5: Final verification checklist**

Run the app and confirm each item:

```bash
streamlit run app.py
```

- [ ] Before running QIR: all three signal cells show dim placeholder text (`◌ … — run QIR`)
- [ ] After running QIR: Regime cell shows composite score, leading score, and divergence badge
- [ ] Tactical cell shows score + SPY MA + momentum signal values + VIX note if triggered
- [ ] Options cell shows score + P/C ratio + gamma zone + unusual call% if available
- [ ] Entry Recommendation card appears directly below the signal strip
- [ ] Pattern label + conviction bar + buy/short panels appear below Entry Recommendation
- [ ] `_leading_warning` floating orange bar is gone — divergence shown only in Regime cell
- [ ] `_vix_note` floating bar is gone — VIX note shown only in Tactical cell
- [ ] Footer line shows FOMC · CPI · NFP and earnings tickers on one line
- [ ] Judge Judy debate section is visually below the recommendation (unchanged)

- [ ] **Step 6: Commit**

```bash
git add modules/quick_run.py
git commit -m "feat(qir): Zone 4 footer + final layout assembly — signal strip → entry rec → detail → footer"
```
