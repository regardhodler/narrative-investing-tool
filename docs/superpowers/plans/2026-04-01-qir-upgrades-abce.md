# QIR Upgrades (A, B, C, E) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the QIR module with four independent upgrades: Run History & Conviction Timeline (A), Portfolio-Aware Event Surfacing (B), Market Breadth as a 4th Signal Layer (C), and QIR Scorecard Export (E).

**Architecture:** Each upgrade is self-contained — A adds a new persistence service and mini-chart, B extends Round 5 with AI-powered event matching against open positions, C adds a new breadth service that plugs into Round 1 as a 4th parallel task, and E adds a dedicated export button to the post-run summary section. All four read/write through existing patterns (signals_cache, session_state, export_hub).

**Tech Stack:** Streamlit, Plotly, pandas, yfinance (via market_data.py), existing claude_client.py AI calls, utils/journal.py for positions, utils/theme.py for chart styling.

---

## Chunk 1 — Feature A: QIR Run History & Conviction Timeline

### Files
- **Create:** `services/qir_history.py` — load/save/query run history
- **Create:** `data/qir_run_history.json` — empty array `[]` (seed file)
- **Modify:** `modules/quick_run.py` — save run record inside button handler + render sparkline in post-run summary

---

### Task A1: Create the run history service

**File:** `services/qir_history.py`

- [ ] **Create `services/qir_history.py`** with this exact content:

```python
"""QIR run history — persists each run's key signals to data/qir_run_history.json."""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "qir_run_history.json")
_MAX_RUNS = 30  # keep last 30 runs


@dataclass
class QIRRunRecord:
    run_id: str
    timestamp: str          # ISO format
    pattern: str
    conviction: str         # BULLISH | BEARISH | MIXED | UNCERTAIN | ""
    tactical_score: int
    options_score: float
    regime_label: str
    quadrant: str
    n_ok: int
    n_total: int
    engine: str             # Freeloader / Regard / Highly Regarded


def load_qir_history() -> list[dict]:
    """Load run history from disk. Returns list of dicts, newest first."""
    path = os.path.normpath(_HISTORY_PATH)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(reversed(data)) if isinstance(data, list) else []
    except Exception:
        return []


def append_qir_run(record: QIRRunRecord) -> None:
    """Append a run record, keeping only the last _MAX_RUNS entries."""
    path = os.path.normpath(_HISTORY_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(asdict(record))
    # Trim to max
    if len(existing) > _MAX_RUNS:
        existing = existing[-_MAX_RUNS:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
```

- [ ] **Create seed file** `data/qir_run_history.json` with content `[]`

- [ ] **Verify** the file is importable:
  ```bash
  cd "C:/Users/16476/claude projects/narrative-investing-tool"
  python -c "from services.qir_history import load_qir_history, append_qir_run; print('OK', load_qir_history())"
  ```
  Expected: `OK []`

---

### Task A2: Save a run record after each QIR run

**File:** `modules/quick_run.py`

The run record should be saved inside the `if st.button(...)` block, just before `st.rerun()` (around line 1200).

- [ ] **Find the line** `# Rerun so the Intelligence Dashboard at the top reflects freshly-populated session_state` (the comment just before `st.rerun()` inside the button handler).

- [ ] **Insert the following block** immediately before that comment:

```python
        # ── Save QIR run record to history ────────────────────────────────────
        try:
            import uuid as _uuid
            from services.qir_history import QIRRunRecord, append_qir_run as _append_run
            _rc_h  = st.session_state.get("_regime_context") or {}
            _tac_h = st.session_state.get("_tactical_context") or {}
            _of_h  = st.session_state.get("_options_flow_context") or {}
            _syn_h = st.session_state.get("_macro_synopsis") or {}
            _n_ok_h = st.session_state.get("_qir_last_n_ok", 0)
            _n_tot_h = st.session_state.get("_qir_last_n_total", 0)
            _eng_h = st.session_state.get("_macro_synopsis_engine", "")
            _append_run(QIRRunRecord(
                run_id=str(_uuid.uuid4())[:8],
                timestamp=__import__("datetime").datetime.now().isoformat(),
                pattern=st.session_state.get("_qir_last_pattern", ""),
                conviction=_syn_h.get("conviction", ""),
                tactical_score=int(_tac_h.get("tactical_score", 0)),
                options_score=float(_of_h.get("options_score", 0)),
                regime_label=_rc_h.get("regime", ""),
                quadrant=_rc_h.get("quadrant", ""),
                n_ok=_n_ok_h,
                n_total=_n_tot_h,
                engine=_eng_h,
            ))
        except Exception:
            pass
```

- [ ] **Run the app** (`streamlit run app.py`), navigate to Quick Intel Run, click RUN ALL INTEL MODULES.

- [ ] **Verify** `data/qir_run_history.json` now contains one entry with the correct fields.

---

### Task A3: Render the conviction timeline sparkline in post-run summary

**File:** `modules/quick_run.py` — inside the `# ── QIR Post-Run Summary` section (outside button handler)

- [ ] **Add the following block** at the end of the `# ── QIR Post-Run Summary` section, just before the `# ── Signal Coverage Panel` comment:

```python
    # ── QIR Run History Sparkline ──────────────────────────────────────────
    try:
        from services.qir_history import load_qir_history as _load_hist
        import plotly.graph_objects as go as _go
        from utils.theme import apply_dark_layout, COLORS as _C
        _hist = _load_hist()
        if len(_hist) >= 2:
            _conv_map = {"BULLISH": 1, "MIXED": 0, "UNCERTAIN": 0, "BEARISH": -1}
            _dates    = [h["timestamp"][:10] for h in _hist[:14]][::-1]
            _convs    = [_conv_map.get(h.get("conviction", ""), 0) for h in _hist[:14]][::-1]
            _tacs     = [h.get("tactical_score", 50) for h in _hist[:14]][::-1]
            _opts     = [h.get("options_score", 50)  for h in _hist[:14]][::-1]
            _conv_colors = [
                "#22c55e" if v == 1 else ("#ef4444" if v == -1 else "#f59e0b")
                for v in _convs
            ]
            _fig = _go.Figure()
            _fig.add_trace(_go.Bar(
                x=_dates, y=_convs,
                marker_color=_conv_colors,
                name="Conviction",
                opacity=0.7,
            ))
            _fig.add_trace(_go.Scatter(
                x=_dates, y=[t / 100 for t in _tacs],
                mode="lines+markers", name="Tactical",
                line={"color": "#38bdf8", "width": 2},
                marker={"size": 5},
                yaxis="y2",
            ))
            _fig.add_trace(_go.Scatter(
                x=_dates, y=[o / 100 for o in _opts],
                mode="lines+markers", name="Opt Flow",
                line={"color": "#a78bfa", "width": 2, "dash": "dot"},
                marker={"size": 5},
                yaxis="y2",
            ))
            _fig.update_layout(
                height=140,
                margin={"t": 4, "b": 4, "l": 4, "r": 4},
                yaxis={"range": [-1.5, 1.5], "tickvals": [-1, 0, 1],
                       "ticktext": ["BEAR", "MIX", "BULL"], "showgrid": False},
                yaxis2={"range": [0, 1], "overlaying": "y", "side": "right",
                        "showgrid": False, "tickformat": ".0%"},
                legend={"orientation": "h", "y": -0.15, "font": {"size": 9}},
                barmode="relative",
            )
            apply_dark_layout(_fig)
            st.markdown(
                '<div style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
                'color:#475569;margin-top:12px;margin-bottom:2px;">QIR RUN HISTORY</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass
```

- [ ] **Run the app**, do 2+ QIR runs, verify the sparkline appears in the post-run summary showing BULL/BEAR/MIX bars + tactical and options flow lines.

- [ ] **Commit:**
  ```bash
  git add services/qir_history.py data/qir_run_history.json modules/quick_run.py
  git commit -m "feat(qir): add run history service and conviction timeline sparkline"
  ```

---

## Chunk 2 — Feature B: Portfolio-Aware Event Surfacing

### Files
- **Modify:** `modules/quick_run.py` — extend Round 5 to match events against open positions, store `_qir_portfolio_events` in session state, render in post-run summary

---

### Task B1: Match events to open positions (inside button handler, Round 5 area)

**File:** `modules/quick_run.py`

This runs after Round 1 (digest + doom are available) and after Round 5 (earnings risk loaded). Add it to the end of Round 5, alongside the earnings scan (around line 1140 in the button handler).

- [ ] **Find** the line `except Exception as _er_e:` (the one that catches earnings risk failure, inside the button handler). Add the following block **after** it (still inside `if st.button(...)`):

```python
        # ── Portfolio Event Matching (cross-reference events against open positions) ──
        try:
            from utils.journal import load_journal as _lj_b
            from services.claude_client import generate_portfolio_event_alerts as _gen_pea
            _open_b = [t for t in _lj_b() if t.get("status") == "open"]
            _tickers_b = [t["ticker"] for t in _open_b if t.get("ticker")]
            if _tickers_b:
                _digest_b = st.session_state.get("_current_events_digest", "")
                _doom_b   = st.session_state.get("_doom_briefing", "")
                _swans_b  = st.session_state.get("_custom_swans") or {}
                _events_text = "\n\n".join(filter(None, [
                    f"NEWS DIGEST:\n{_digest_b[:600]}" if _digest_b else "",
                    f"DOOM BRIEFING:\n{_doom_b[:400]}" if _doom_b else "",
                    "BLACK SWANS:\n" + "; ".join(
                        f"{k} ({v.get('probability_pct',0):.1f}%)"
                        for k, v in list(_swans_b.items())[:4]
                    ) if _swans_b else "",
                ]))
                if _events_text.strip():
                    _pea = _gen_pea(
                        tickers=_tickers_b,
                        events_text=_events_text,
                        use_claude=_use_claude,
                        model=_cl_model,
                    )
                    import datetime as _pea_dt
                    st.session_state["_qir_portfolio_events"]    = _pea
                    st.session_state["_qir_portfolio_events_ts"] = _pea_dt.datetime.now()
                    st.success(f"✅ Portfolio Event Alerts — {len(_pea)} position(s) flagged")
        except Exception as _pea_e:
            st.warning(f"⚠ Portfolio Event Alerts skipped: {_pea_e}")
```

- [ ] **Add `_qir_portfolio_events` and `_qir_portfolio_events_ts` to `services/signals_cache.py`** `_SIGNAL_KEYS` list (find the list and append at the bottom):
  ```python
  "_qir_portfolio_events",
  "_qir_portfolio_events_ts",
  ```

---

### Task B2: Add `generate_portfolio_event_alerts` to claude_client.py

**File:** `services/claude_client.py`

- [ ] **Add the following function** at the end of the file, before the last line:

```python
def generate_portfolio_event_alerts(
    tickers: list[str],
    events_text: str,
    use_claude: bool = False,
    model: str | None = None,
) -> list[dict]:
    """Cross-reference macro events against held positions.

    Returns list of dicts:
      {ticker, event_summary, severity: "HIGH"|"MEDIUM"|"LOW", action: str}
    Only includes tickers that have meaningful event exposure.
    """
    _ticker_list = ", ".join(tickers)
    prompt = (
        f"You are a portfolio risk analyst. Review the macro events below and identify "
        f"which of these held positions are directly or materially affected.\n\n"
        f"HELD POSITIONS: {_ticker_list}\n\n"
        f"{events_text}\n\n"
        f"Return ONLY valid JSON array (no markdown fences). Each element:\n"
        f'{{"ticker": "AAPL", "event_summary": "one-sentence reason", '
        f'"severity": "HIGH|MEDIUM|LOW", "action": "Monitor|Reduce|Hedge|Exit"}}\n'
        f"Rules:\n"
        f"- Only include tickers with clear, direct exposure to the events above\n"
        f"- Skip tickers with no meaningful connection to these events\n"
        f"- Be specific: cite the event and explain the mechanism of impact\n"
        f"- HIGH = direct material impact, MEDIUM = indirect, LOW = minor mention\n"
        f"- Return [] if no positions are meaningfully exposed"
    )
    _cl_model = model or "grok-4-1-fast-reasoning"
    import json as _json, re as _re

    def _parse(raw: str) -> list[dict]:
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"\s*```$", "", raw, flags=_re.MULTILINE).strip()
        result = _json.loads(raw)
        return result if isinstance(result, list) else []

    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            raw = _call_xai([{"role": "user", "content": prompt}], _cl_model, 800, 0.2)
            return _parse(raw)
        except Exception:
            pass
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model, max_tokens=800, temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse(msg.content[0].text.strip())
        except Exception:
            pass

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800, "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return _parse(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception:
        return []
```

---

### Task B3: Display portfolio event alerts in post-run summary

**File:** `modules/quick_run.py` — in the `# ── QIR Post-Run Summary` section, after the `_qir_earnings_risk` section of the QIR Intelligence Dashboard.

- [ ] **Add the following block** just before the `# ── Signal Coverage Panel` comment (or after the run history sparkline from Task A3):

```python
    # ── Portfolio Event Alerts ─────────────────────────────────────────────
    _pea = st.session_state.get("_qir_portfolio_events") or []
    if _pea:
        _sev_colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#64748b"}
        _sev_bg     = {"HIGH": "#1a0000", "MEDIUM": "#1a1200", "LOW": "#0d1117"}
        _pea_html = ""
        for _pa in _pea:
            _sc = _sev_colors.get(_pa.get("severity", "LOW"), "#64748b")
            _sb = _sev_bg.get(_pa.get("severity", "LOW"), "#0d1117")
            _pea_html += (
                f'<div style="background:{_sb};border-left:3px solid {_sc};'
                f'padding:6px 10px;margin-bottom:4px;border-radius:0 4px 4px 0;">'
                f'<span style="color:{_sc};font-weight:700;font-size:11px;">'
                f'{_pa.get("ticker","?")} · {_pa.get("severity","?")}</span>'
                f'<span style="color:#94a3b8;font-size:10px;margin-left:8px;">'
                f'{_pa.get("action","")}</span>'
                f'<div style="color:#cbd5e1;font-size:11px;margin-top:2px;">'
                f'{_pa.get("event_summary","")}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="margin-top:12px;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
            f'color:#475569;margin-bottom:6px;">PORTFOLIO EVENT ALERTS</div>'
            f'{_pea_html}</div>',
            unsafe_allow_html=True,
        )
```

- [ ] **Run the app**, run QIR with open positions, verify event alerts appear in the post-run summary.

- [ ] **Commit:**
  ```bash
  git add modules/quick_run.py services/claude_client.py services/signals_cache.py
  git commit -m "feat(qir): add portfolio-aware event surfacing (Feature B)"
  ```

---

## Chunk 3 — Feature C: Market Breadth as 4th Signal Layer

### Files
- **Create:** `services/breadth_client.py` — fetches sector ETF breadth, computes 0-100 score
- **Modify:** `modules/quick_run.py` — add `run_quick_breadth()` to Round 1 parallel pool, store `_breadth_context`, display 4th row in Timing Stack
- **Modify:** `modules/quick_run.py` — update `_classify_signals()` to accept optional breadth input for confidence modulation

---

### Task C1: Create breadth service

**File:** `services/breadth_client.py`

The breadth score uses the 11 SPDR sector ETFs. For each ETF, compute whether its current price is above its 50-day MA. The % above = the breadth score (0-100). Also compute z-score of RSP/SPY ratio (equal-weight participation). Blend into a single 0-100 breadth score.

- [ ] **Create `services/breadth_client.py`** with this content:

```python
"""Market breadth service for QIR — uses sector ETF participation as breadth proxy.

Fetches the 11 SPDR sector ETFs + RSP (equal-weight S&P 500).
Computes % of sector ETFs above their 50-day MA → breadth score 0-100.
Background-safe: no st.* calls.
"""
import numpy as np

SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Cons. Disc.", "XLP": "Cons. Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLC": "Comm. Services",
}
_BREADTH_TICKERS = list(SECTOR_ETFS.keys()) + ["RSP", "SPY", "IWM"]


def run_quick_breadth() -> dict | None:
    """Compute market breadth score (0-100). Returns dict or None on failure."""
    try:
        from services.market_data import fetch_batch_safe
        snaps = fetch_batch_safe({t: t for t in _BREADTH_TICKERS})

        above_50d = []
        sector_signals = []
        for ticker, label in SECTOR_ETFS.items():
            snap = snaps.get(ticker)
            if snap and snap.series is not None and len(snap.series) >= 50:
                price = float(snap.series.iloc[-1])
                ma50  = float(snap.series.iloc[-50:].mean())
                is_above = price > ma50
                above_50d.append(is_above)
                pct_above_ma = (price - ma50) / ma50 * 100
                direction = "Above 50d MA" if is_above else "Below 50d MA"
                sector_signals.append({
                    "Signal": label,
                    "Value": f"{price:.2f}",
                    "Score": round(pct_above_ma / 10, 2),  # normalize
                    "Direction": direction,
                })

        if not above_50d:
            return None

        pct_above = sum(above_50d) / len(above_50d)  # 0.0 – 1.0
        breadth_score = int(round(pct_above * 100))

        # RSP/SPY participation bonus (equal-weight > cap-weight = broad participation)
        rsp = snaps.get("RSP")
        spy = snaps.get("SPY")
        if rsp and spy and rsp.series is not None and spy.series is not None:
            rsp_30 = float(rsp.series.iloc[-30:].pct_change().sum())
            spy_30 = float(spy.series.iloc[-30:].pct_change().sum())
            rsp_lead = rsp_30 - spy_30
            # Adjust score: RSP outperforming → +5, underperforming → -5
            breadth_score = int(min(100, max(0, breadth_score + (5 if rsp_lead > 0 else -5))))

        # IWM relative strength vs SPY
        iwm = snaps.get("IWM")
        if iwm and spy and iwm.series is not None and spy.series is not None:
            iwm_30 = float(iwm.series.iloc[-30:].pct_change().sum())
            spy_30b = float(spy.series.iloc[-30:].pct_change().sum())
            iwm_lead = iwm_30 - spy_30b
            sector_signals.append({
                "Signal": "IWM vs SPY (30d)",
                "Value": f"{iwm_lead*100:+.1f}%",
                "Score": round(iwm_lead * 10, 2),
                "Direction": "Small-cap leading" if iwm_lead > 0 else "Small-cap lagging",
            })

        # Labels
        if breadth_score >= 70:
            label = "Broad Participation"
            action_bias = "Wide breadth — confirms trend strength"
        elif breadth_score >= 50:
            label = "Moderate Breadth"
            action_bias = "Mixed participation — moderate conviction"
        elif breadth_score >= 30:
            label = "Narrow Leadership"
            action_bias = "Only a few sectors leading — caution on new longs"
        else:
            label = "Breadth Collapse"
            action_bias = "Most sectors below 50d MA — risk-off positioning"

        return {
            "breadth_score": breadth_score,
            "label": label,
            "action_bias": action_bias,
            "pct_above_50d": round(pct_above * 100, 1),
            "sectors_above": sum(above_50d),
            "sectors_total": len(above_50d),
            "signals": sector_signals,
        }
    except Exception:
        return None
```

- [ ] **Verify import works:**
  ```bash
  python -c "from services.breadth_client import run_quick_breadth; print('OK')"
  ```
  Expected: `OK`

---

### Task C2: Add breadth to Round 1 parallel pool

**File:** `modules/quick_run.py`

- [ ] **Find** the Round 1 imports block (around `from services.stocktwits_client import run_quick_stocktwits`). Add:
  ```python
  from services.breadth_client import run_quick_breadth
  ```

- [ ] **Find** the `ThreadPoolExecutor(max_workers=7)` block. Change `max_workers=7` to `max_workers=8` and add a new future:
  ```python
  _fut_breadth  = _pool.submit(run_quick_breadth)
  ```

- [ ] **In the `for _fut, _key in (...)` loop**, add the breadth entry:
  ```python
  (_fut_breadth,  "breadth"),
  ```

- [ ] **In the result handling block** (the `elif _key == "social" and _val:` section), add:
  ```python
  elif _key == "breadth" and _val:
      import datetime as _bdt
      st.session_state["_breadth_context"]    = _val
      st.session_state["_breadth_context_ts"] = _bdt.datetime.now()
  ```

- [ ] **Find** the success/warning display section after Round 1. Add:
  ```python
  if _results.get("breadth"):
      _br = st.session_state.get("_breadth_context") or {}
      st.success(f"✅ Market Breadth — {_br.get('sectors_above','?')}/{_br.get('sectors_total','?')} sectors above 50d MA ({_br.get('breadth_score','?')}/100)")
  else:
      st.warning("⚠ Breadth: sector data unavailable")
  ```

- [ ] **Add `_breadth_context` and `_breadth_context_ts` to `services/signals_cache.py`** `_SIGNAL_KEYS`.

---

### Task C3: Display breadth in the QIR Intelligence Dashboard (4th Timing Stack row)

**File:** `modules/quick_run.py` — in `_render_qir_dashboard()` function

- [ ] **Find** the `# ── Timing Stack column` section in `_render_qir_dashboard()`. After the Options Flow `elif`/`else` block, add:

```python
    _br = st.session_state.get("_breadth_context") or {}
    if _br:
        _br_score = _br.get("breadth_score", 50)
        _br_bull = _br_score >= 65; _br_bear = _br_score < 35
        _bc = "#22c55e" if _br_bull else ("#ef4444" if _br_bear else "#f59e0b")
        _ba = "▲" if _br_bull else ("▼" if _br_bear else "◆")
        _t1 += (f'<div style="color:{_bc};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_ba} 🌊 Breadth: <span style="color:#e2e8f0;">{_br.get("label","")}</span>'
                f'<span style="color:{_bc};font-size:10px;"> ({_br_score}/100)</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ 🌊 Breadth — run QIR</div>'
```

- [ ] **Add `_breadth_context` to the `_sig_checks` list** in `_render_qir_dashboard()`:
  ```python
  ("Breadth",   "_breadth_context"),
  ```
  (This adds it to the freshness dot indicator.)

---

### Task C4: Add breadth expander to post-run summary

**File:** `modules/quick_run.py` — in the `# ── QIR Post-Run Summary` section

- [ ] **Add after the Options Flow Preview expander** (`# ── Options Flow Preview` section):

```python
    # ── Market Breadth Preview ─────────────────────────────────────────────
    _br_ctx = st.session_state.get("_breadth_context") or {}
    if _br_ctx:
        _br_val   = _br_ctx.get("breadth_score", 50)
        _br_label = _br_ctx.get("label", "")
        _br_bias  = _br_ctx.get("action_bias", "")
        _br_color = "#22c55e" if _br_val >= 65 else ("#f59e0b" if _br_val >= 35 else "#ef4444")
        with st.expander(f"🌊 Market Breadth — {_br_label} ({_br_val}/100)", expanded=False):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="background:{_br_color};color:black;font-weight:800;font-size:11px;'
                f'padding:3px 10px;border-radius:4px;">{_br_label.upper()}</span>'
                f'<span style="color:{_br_color};font-size:12px;font-weight:700;">{_br_val}/100</span>'
                f'<span style="color:#94a3b8;font-size:11px;">{_br_bias}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _br_sigs = _br_ctx.get("signals", [])
            if _br_sigs:
                _br_html = ""
                for _s in _br_sigs:
                    _sc2 = "#22c55e" if _s["Score"] > 0 else "#ef4444"
                    _br_html += (
                        f'<div style="color:{_sc2};font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:11px;padding:1px 0;">'
                        f'{"▲" if _s["Score"] > 0 else "▼"} {_s["Signal"]}: '
                        f'<span style="color:#e2e8f0;">{_s["Value"]}</span>'
                        f'<span style="color:#475569;"> ({_s["Direction"]})</span></div>'
                    )
                st.markdown(f'<div>{_br_html}</div>', unsafe_allow_html=True)
```

- [ ] **Run the app**, run QIR, verify breadth appears in the Timing Stack (4th row) and in the post-run expander.

- [ ] **Commit:**
  ```bash
  git add services/breadth_client.py modules/quick_run.py services/signals_cache.py
  git commit -m "feat(qir): add market breadth as 4th signal layer (Feature C)"
  ```

---

## Chunk 4 — Feature E: QIR Scorecard Export

### Files
- **Modify:** `modules/export_hub.py` — add `_section_qir_scorecard()` function
- **Modify:** `modules/quick_run.py` — add export button in post-run summary

---

### Task E1: Add QIR scorecard section to export hub

**File:** `modules/export_hub.py`

- [ ] **Add the following function** near the other `_section_*` functions (find `def _section_executive_summary` and add this before it):

```python
def _section_qir_scorecard() -> str:
    """QIR Scorecard — pattern, conviction, all timing scores, synopsis, earnings risk."""
    import streamlit as st
    _rc  = st.session_state.get("_regime_context") or {}
    _tac = st.session_state.get("_tactical_context") or {}
    _of  = st.session_state.get("_options_flow_context") or {}
    _br  = st.session_state.get("_breadth_context") or {}
    _syn = st.session_state.get("_macro_synopsis") or {}
    _pat = st.session_state.get("_qir_last_pattern", "")
    _er  = st.session_state.get("_qir_earnings_risk") or []
    _pea = st.session_state.get("_qir_portfolio_events") or []
    _ts  = st.session_state.get("_macro_synopsis_ts")

    if not _rc and not _tac and not _of:
        return ""

    lines = [
        "## QIR SCORECARD",
        f"Generated: {_ts.strftime('%Y-%m-%d %H:%M') if _ts else 'N/A'}",
        "",
        "### Signal Stack",
        f"- Macro Regime:    {_rc.get('regime','N/A')} | Quadrant: {_rc.get('quadrant','N/A')} | Score: {_rc.get('score', 0):+.2f}",
        f"- Tactical:        {_tac.get('label','N/A')} ({_tac.get('tactical_score','N/A')}/100) — {_tac.get('action_bias','')}",
        f"- Options Flow:    {_of.get('label','N/A')} ({_of.get('options_score','N/A')}/100) — {_of.get('action_bias','')}",
    ]
    if _br:
        lines.append(f"- Breadth:         {_br.get('label','N/A')} ({_br.get('breadth_score','N/A')}/100) — {_br.get('action_bias','')}")
    if _pat:
        lines += ["", f"### QIR Pattern: {_pat.replace('_',' ')}"]
    if _syn.get("conviction"):
        lines += [
            "",
            f"### Macro Conviction: {_syn['conviction']}",
            f"{_syn.get('summary', '')}",
        ]
        for kp in _syn.get("key_points", []):
            lines.append(f"  · {kp}")
        for ct in _syn.get("contradictions", []):
            lines.append(f"  ⚠ Contradiction: {ct}")
    if _er:
        lines += ["", "### Earnings at Risk"]
        for e in _er[:6]:
            em = f" ±{e['expected_move_pct']:.1f}%" if e.get("expected_move_pct") else ""
            lines.append(f"  · {e['ticker']} — {e['days_away']}d ({e.get('date','?')}){em}")
    if _pea:
        lines += ["", "### Portfolio Event Alerts"]
        for p in _pea:
            lines.append(f"  · [{p.get('severity','?')}] {p.get('ticker','?')}: {p.get('event_summary','')}")

    return "\n".join(lines)
```

- [ ] **Add `_section_qir_scorecard()` to the markdown export section list** in `_build_markdown_export`. Find the list of sections being joined (usually a list comprehension with `filter(None, [...])`) and add `_section_qir_scorecard()` near the top of that list (after the executive summary).

---

### Task E2: Add inline export button in QIR post-run summary

**File:** `modules/quick_run.py`

- [ ] **Add the following block** at the top of the `# ── QIR Post-Run Summary` section, immediately after the completion banner block (`if _last_ok is not None and _last_total is not None:`):

```python
    # ── QIR Scorecard Export ───────────────────────────────────────────────
    _has_qir_data = bool(
        st.session_state.get("_regime_context") or
        st.session_state.get("_tactical_context") or
        st.session_state.get("_macro_synopsis")
    )
    if _has_qir_data:
        try:
            from modules.export_hub import _section_qir_scorecard
            import datetime as _exp_dt
            _scorecard_text = _section_qir_scorecard()
            if _scorecard_text:
                _exp_col1, _exp_col2 = st.columns([3, 1])
                with _exp_col2:
                    st.download_button(
                        label="📋 Export Scorecard",
                        data=_scorecard_text,
                        file_name=f"qir_scorecard_{_exp_dt.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download QIR scorecard as a text file",
                    )
        except Exception:
            pass
```

- [ ] **Run the app**, run QIR, verify "📋 Export Scorecard" button appears. Click it and verify the downloaded file contains regime, tactical, synopsis, earnings risk, and any portfolio event alerts.

- [ ] **Commit:**
  ```bash
  git add modules/export_hub.py modules/quick_run.py
  git commit -m "feat(qir): add QIR scorecard export button (Feature E)"
  ```

---

---

## Chunk 5 — Cross-Module Signal Injection (Breadth + Run History)

### Files
- **Modify:** `services/claude_client.py` — add `_fmt_breadth_ctx()` helper
- **Modify:** `modules/valuation.py` — inject breadth + run history into signals_text before AI call; add Breadth signal tile
- **Modify:** `modules/trade_journal.py` — pass breadth to `interpret_risk_matrix()`, add breadth warning flag, show regime consistency from run history

---

### Task F1: Add breadth formatter to claude_client.py

**File:** `services/claude_client.py`

- [ ] **Add after `_fmt_tactical_ctx()`** (around line 16):

```python
def _fmt_breadth_ctx(ctx: dict | None) -> str:
    """Format a _breadth_context dict as a one-line prompt injection string."""
    if not ctx:
        return ""
    score = ctx.get("breadth_score", "?")
    label = ctx.get("label", "?")
    pct   = ctx.get("pct_above_50d", "?")
    return f"Market Breadth: {score}/100 ({label}) — {pct}% of sectors above 50d MA"
```

---

### Task F2: Inject breadth + run history into Valuation AI call

**File:** `modules/valuation.py` — around line 682-685

- [ ] **Find** the block:
  ```python
  _ce_val = st.session_state.get("_current_events_digest", "")
  _tac_val = _fmt_tactical_ctx(st.session_state.get("_tactical_context"))
  result = generate_valuation(ticker, signals_text, ...)
  ```

- [ ] **Replace** with:
  ```python
  from services.claude_client import generate_valuation, _fmt_tactical_ctx, _fmt_breadth_ctx
  _ce_val      = st.session_state.get("_current_events_digest", "")
  _tac_val     = _fmt_tactical_ctx(st.session_state.get("_tactical_context"))
  _breadth_val = _fmt_breadth_ctx(st.session_state.get("_breadth_context"))

  # Append breadth + run history consistency to signals_text
  _extra_ctx = []
  if _breadth_val:
      _extra_ctx.append(_breadth_val)
  try:
      from services.qir_history import load_qir_history as _lh_v
      _hist_v = _lh_v()[:7]
      if len(_hist_v) >= 3:
          _conv_counts = {}
          for _h in _hist_v:
              _cv = _h.get("conviction", "")
              _conv_counts[_cv] = _conv_counts.get(_cv, 0) + 1
          _dominant = max(_conv_counts, key=_conv_counts.get)
          _consistency = f"QIR History (last {len(_hist_v)} runs): {_dominant} dominant " \
                         f"({_conv_counts[_dominant]}x) — " + \
                         ", ".join(f"{k} {v}x" for k, v in _conv_counts.items() if k != _dominant)
          _extra_ctx.append(_consistency)
  except Exception:
      pass
  if _extra_ctx:
      signals_text = signals_text + "\n\n" + "\n".join(_extra_ctx)

  result = generate_valuation(ticker, signals_text, use_claude=_use_claude, model=_cl_model,
                              current_events=_ce_val, tactical_context=_tac_val)
  ```

---

### Task F3: Add Breadth signal tile to Valuation signal grid

**File:** `modules/valuation.py`

The valuation module has signal tiles like `_signal_tile(r4c1, "Options Flow Sentiment", ...)`. Find the tile section and add a Breadth tile.

- [ ] **Find** the `_signal_tile(r4c2, "Crowd Sentiment", ...)` line (around line 1536). After it, add:

```python
_br_v = st.session_state.get("_breadth_context") or {}
if _br_v:
    _br_score_v = _br_v.get("breadth_score", 50)
    _br_label_v = _br_v.get("label", "")
    _br_verdict = "bullish" if _br_score_v >= 65 else ("bearish" if _br_score_v < 35 else "neutral")
    _signal_tile(
        r4c3 if "r4c3" in dir() else st,  # use next available tile slot
        "Market Breadth",
        _br_verdict,
        [f"{_br_score_v}/100 — {_br_label_v}",
         f"{_br_v.get('pct_above_50d','?')}% of sectors above 50d MA"],
    )
else:
    _signal_tile(r4c3 if "r4c3" in dir() else st, "Market Breadth", "unavailable", ["Run QIR to populate"])
```

> **Note:** Check the actual column variable names around line 1536 in valuation.py and adjust `r4c3` to the correct next column variable.

---

### Task F4: Inject breadth into Portfolio Intel risk snapshot

**File:** `modules/trade_journal.py` — in `run_quick_risk_snapshot()` (around line 239-251)

- [ ] **Find** the `interpret_risk_matrix` call:
  ```python
  interp = _irm(snapshot, regime_ctx, use_claude=use_claude, model=model,
                tactical_context=_tac_snap or None)
  ```

- [ ] **Replace** with:
  ```python
  _br_snap = st.session_state.get("_breadth_context") or None
  interp = _irm(snapshot, regime_ctx, use_claude=use_claude, model=model,
                tactical_context=_tac_snap or None,
                breadth_context=_br_snap)
  ```

- [ ] **Add breadth warning flag** to the snapshot dict (find where `_portfolio_risk_snapshot` is built, around line 588, and add):
  ```python
  "breadth_score": (_br_snap or {}).get("breadth_score"),
  "breadth_label": (_br_snap or {}).get("label", ""),
  ```

---

### Task F5: Show regime consistency in Portfolio Intel panel

**File:** `modules/trade_journal.py` — in the Portfolio Intelligence tab (around line 1059 where `_rc_pi` is read)

- [ ] **Add after the existing regime context display**:

```python
# ── QIR Regime Consistency (run history) ──────────────────────────────
try:
    from services.qir_history import load_qir_history as _lh_pi
    _hist_pi = _lh_pi()[:7]
    if len(_hist_pi) >= 3:
        _conv_pi = [h.get("conviction", "") for h in _hist_pi if h.get("conviction")]
        _tac_pi  = [h.get("tactical_score", 50) for h in _hist_pi]
        _dominant_pi = max(set(_conv_pi), key=_conv_pi.count) if _conv_pi else ""
        _tac_avg_pi  = sum(_tac_pi) / len(_tac_pi) if _tac_pi else 50
        _regime_volatile = len(set(_conv_pi)) >= 3  # 3+ different convictions = volatile
        _consistency_color = "#ef4444" if _regime_volatile else ("#22c55e" if _dominant_pi == "BULLISH" else "#f59e0b")
        _consistency_label = "⚠ Volatile — reduce conviction sizing" if _regime_volatile \
            else f"Consistent {_dominant_pi} — avg tactical {_tac_avg_pi:.0f}/100"
        st.markdown(
            f'<div style="background:#0d1117;border-left:3px solid {_consistency_color};'
            f'padding:6px 12px;font-size:11px;margin-top:6px;">'
            f'<span style="color:#64748b;font-weight:700;">REGIME CONSISTENCY (last {len(_hist_pi)} runs)</span>'
            f'<span style="color:{_consistency_color};margin-left:8px;">{_consistency_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass
```

- [ ] **Run the app**, open Portfolio Intel, verify regime consistency banner shows after QIR has been run at least 3 times.

- [ ] **Commit:**
  ```bash
  git add services/claude_client.py modules/valuation.py modules/trade_journal.py
  git commit -m "feat(qir): inject breadth + run history into Valuation and Portfolio Intel (Feature F)"
  ```

---

## Verification Checklist

After all four features are implemented:

- [ ] Run QIR twice → `data/qir_run_history.json` has 2 entries, sparkline shows both runs
- [ ] Open positions in Trade Journal → run QIR → portfolio event alerts appear (or empty list if no relevant events)
- [ ] Breadth appears as 4th row in the Timing Stack dashboard (before running QIR: greyed "◌ 🌊 Breadth — run QIR"; after run: colored score)
- [ ] Export Scorecard button visible after QIR run, downloaded file has all sections
- [ ] Navigate away from QIR and back → post-run summary (synopsis, digest, breadth, alerts) all still visible (session state persistence)
- [ ] Run on Freeloader mode → all 4 features work (no AI dependency for history, breadth, export)
- [ ] Open Module 7 Valuation → signals_text includes breadth context and run history consistency line before AI call
- [ ] Valuation signal grid shows "Market Breadth" tile (populated if QIR has run, "Run QIR to populate" otherwise)
- [ ] Portfolio Intel panel shows "REGIME CONSISTENCY" banner after 3+ QIR runs
- [ ] Portfolio risk snapshot dict includes `breadth_score` and `breadth_label` fields
