# QIR Intelligence Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-column Short/Buy panels in Quick Intel Run with a single fused dashboard that shows timing stack, macro events, earnings risk, and buy/short guidance in one glanceable panel.

**Architecture:** A pure `_classify_signals()` helper maps the three timing signals to one of 7 patterns, each with buy/short tiers and instrument lists. `_render_qir_dashboard()` renders the full panel as a greyed shell before QIR and an activated, color-bordered panel after. Earnings risk is fetched per held position in Round 5 and flows downstream into Portfolio Intelligence and Valuation.

**Tech Stack:** Streamlit, Python, `services/market_data.py` (fetch_earnings_intelligence), `services/fed_forecaster.py` (get_next_fomc/cpi/nfp), session state via `services/signals_cache.py`

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `modules/quick_run.py` | Modify | Remove lines 208–452 (Short/Buy block + helpers); add `_classify_signals()` and `_render_qir_dashboard()` above persistent section; fetch earnings in Round 5; inject macro events into `_sig_parts` |
| `services/signals_cache.py` | Modify | Add `_qir_earnings_risk`, `_qir_earnings_risk_ts` to `_SIGNAL_KEYS` |
| `modules/valuation.py` | Modify | Inject `_qir_earnings_risk` ticker match into `signals_text` |
| `modules/trade_journal.py` | Modify | Add `earnings_risk` text key to `_upstream` dict |
| `tests/test_qir_dashboard.py` | Create | Unit tests for `_classify_signals()` |

---

## Chunk 1: `_classify_signals()` helper

### Task 1: Write and test `_classify_signals()`

This is a pure function with no Streamlit dependencies — easiest to write and test first.

**Files:**
- Create: `tests/test_qir_dashboard.py`
- Modify: `modules/quick_run.py` (add helper before the `render()` function)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qir_dashboard.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the helper directly — avoids triggering Streamlit on import
# We'll add _classify_signals as a module-level function in quick_run.py
from modules.quick_run import _classify_signals


def _rc(score, regime=""):
    return {"score": score, "regime": regime, "quadrant": "Goldilocks"}

def _tac(score):
    return {"tactical_score": score, "label": "test", "action_bias": "test"}

def _of(score):
    return {"options_score": score, "label": "test", "action_bias": "test"}


class TestClassifySignals:

    def test_all_bullish_returns_bullish_confirmation(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        assert r["pattern"] == "BULLISH_CONFIRMATION"
        assert r["color"] == "#22c55e"
        assert r["buy_tier"] == "STRONG"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_all_bearish_returns_bearish_confirmation(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(30))
        assert r["pattern"] == "BEARISH_CONFIRMATION"
        assert r["color"] == "#ef4444"
        assert r["buy_tier"] == "NOT A BUYING ENV"
        assert r["short_tier"] == "STRONG"

    def test_regime_up_tac_down_of_up_is_pullback(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(30), _of(70))
        assert r["pattern"] == "PULLBACK_IN_UPTREND"
        assert r["buy_tier"] == "STRONG"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_regime_up_tac_up_of_down_is_options_divergence(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(30))
        assert r["pattern"] == "OPTIONS_FLOW_DIVERGENCE"
        assert r["buy_tier"] == "MODERATE"

    def test_regime_down_tac_up_of_up_is_bear_bounce(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(70), _of(70))
        assert r["pattern"] == "BEAR_MARKET_BOUNCE"
        assert r["buy_tier"] == "SELECTIVE"
        assert r["short_tier"] == "MODERATE"

    def test_regime_down_tac_down_of_up_is_late_cycle_squeeze(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(70))
        assert r["pattern"] == "LATE_CYCLE_SQUEEZE"
        assert r["short_tier"] == "STRONG"

    def test_all_neutral_is_genuine_uncertainty(self):
        r = _classify_signals(_rc(0.0), _tac(50), _of(50))
        assert r["pattern"] == "GENUINE_UNCERTAINTY"
        assert r["color"] == "#475569"
        assert r["buy_tier"] == "NOT A BUYING ENV"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_empty_contexts_return_uncertainty(self):
        r = _classify_signals({}, {}, {})
        assert r["pattern"] == "GENUINE_UNCERTAINTY"

    def test_regime_label_risk_on_overrides_score(self):
        # "Risk-On" in label should classify as bullish even with score=0
        r = _classify_signals(_rc(0.0, "Risk-On — Goldilocks"), _tac(70), _of(70))
        assert r["pattern"] == "BULLISH_CONFIRMATION"

    def test_result_has_all_required_keys(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        for key in ("pattern", "label", "color", "interpretation",
                    "buy_tier", "short_tier", "instruments_buy",
                    "instruments_short", "entry_buy", "entry_short"):
            assert key in r, f"Missing key: {key}"

    def test_instruments_buy_is_list(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        assert isinstance(r["instruments_buy"], list)
        assert len(r["instruments_buy"]) > 0

    def test_instruments_short_is_list(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(30))
        assert isinstance(r["instruments_short"], list)
        assert len(r["instruments_short"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_qir_dashboard.py -v 2>&1 | head -30
```

Expected: `ImportError` — `_classify_signals` not yet defined.

- [ ] **Step 3: Add `_classify_signals()` to `modules/quick_run.py`**

Insert this function near the top of `quick_run.py`, before the `render()` function (around line 18, after imports):

```python
# ── QIR Dashboard helpers ───────────────────────────────────────────────────

_PATTERNS = {
    "BULLISH_CONFIRMATION": {
        "label": "BULLISH CONFIRMATION",
        "color": "#22c55e",
        "interpretation": "All three timing layers aligned bullish — highest-conviction long entry.",
        "buy_tier": "STRONG",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("QQQ / SPY", "Broad market long — simplest expression of risk-on"),
            ("XLK", "Tech leads in risk-on / Goldilocks — amplified beta"),
            ("SPY Calls", "ATM or 5% OTM, 30–60 DTE — leveraged with defined risk"),
            ("TQQQ / UPRO", "Max leverage for highest-conviction only — short duration"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Enter on pullbacks to 20d MA or prior breakout level — not at all-time highs\n"
            "Confirm with breadth: >60% of S&P 500 above 50d MA\n"
            "Stop: close below 20d MA or -5% from entry, whichever is tighter\n"
            "Scale: 50% at entry, add 25% on first successful retest"
        ),
        "entry_short": "All layers lean bullish — avoid net short exposure.\nIf forced: only extreme overbought bounces, very tight stops.",
    },
    "BEARISH_CONFIRMATION": {
        "label": "BEARISH CONFIRMATION",
        "color": "#ef4444",
        "interpretation": "All three layers aligned bearish — highest-conviction short or cash environment.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "STRONG",
        "instruments_buy": [],
        "instruments_short": [
            ("SH", "1× inverse S&P 500 — no decay, suits multi-week holds"),
            ("SPY Puts", "ATM or 5% OTM, 30–60 DTE — defined risk, leverage"),
            ("SDS", "2× inverse S&P — short duration only (decay risk)"),
            ("SQQQ", "3× inverse Nasdaq — tech-led selloff only, size small"),
        ],
        "entry_buy": "No layers confirm bullish conditions — hold cash or hedge.\nNo new longs until at least Tactical turns.",
        "entry_short": (
            "Enter on dead-cat bounces into resistance — not into freefall\n"
            "Confirm with failed rally: rejection at 50d MA on volume\n"
            "Stop: close above last swing high or 50d MA\n"
            "Scale out: 50% at first target, trail stop on remainder"
        ),
    },
    "PULLBACK_IN_UPTREND": {
        "label": "PULLBACK IN UPTREND",
        "color": "#f59e0b",
        "interpretation": "Regime and Options Flow confirm bull trend — Tactical dip is a buy-the-dip setup.",
        "buy_tier": "STRONG",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("QQQ / SPY", "Broad market — buy the dip in the uptrend"),
            ("Sector leaders", "XLK, XLY — highest-momentum sectors in the current regime"),
            ("SPY Calls", "30–60 DTE calls capture the bounce with defined risk"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Scale in on red days — Tactical weakness is your entry window\n"
            "Stop: close below 20d MA\n"
            "Target: prior swing high / all-time high\n"
            "Wait for Tactical to turn ≥55 before adding full size"
        ),
        "entry_short": "Trend is up — pullbacks are entries, not reversals.\nAvoiding fighting the regime.",
    },
    "OPTIONS_FLOW_DIVERGENCE": {
        "label": "OPTIONS FLOW DIVERGENCE",
        "color": "#f59e0b",
        "interpretation": "Regime and Tactical bullish but options crowd is hedging — smart money buying protection.",
        "buy_tier": "MODERATE",
        "short_tier": "SELECTIVE",
        "instruments_buy": [
            ("SPY", "Broad market — reduced size until Options Flow confirms"),
            ("MSFT / GOOGL", "Defensive growth — quality names hold up if crowd is right"),
        ],
        "instruments_short": [
            ("Weak sector ETFs", "XLE, XLU — hedge only, not directional short"),
        ],
        "entry_buy": (
            "Smaller position size — wait for Options Flow to confirm ≥65\n"
            "Prefer quality (MSFT, GOOGL) over high-beta names\n"
            "Stop: tighter than normal — -3% from entry"
        ),
        "entry_short": "Treat as hedge only — not a directional short setup.\nOptions crowd may be wrong; regime and tactical are bullish.",
    },
    "BEAR_MARKET_BOUNCE": {
        "label": "BEAR MARKET BOUNCE",
        "color": "#f97316",
        "interpretation": "Short-term momentum against the macro trend — take profits quickly, don't chase.",
        "buy_tier": "SELECTIVE",
        "short_tier": "MODERATE",
        "instruments_buy": [
            ("QQQ / ARKK", "Momentum names — bounce candidates, not trend reversals"),
        ],
        "instruments_short": [
            ("SH", "1× inverse S&P — size for the eventual trend resumption"),
            ("XLE / XLF Puts", "Weak sectors underperform when the bounce fades"),
        ],
        "entry_buy": (
            "Tight stops — this is a counter-trend trade\n"
            "Take 50% profits at first resistance\n"
            "Do not add to winners — the macro trend is down"
        ),
        "entry_short": (
            "Wait for the bounce to fade — sell into strength, not freefall\n"
            "Stop: above the bounce high\n"
            "Scale in: build position in tranches as momentum rolls over"
        ),
    },
    "LATE_CYCLE_SQUEEZE": {
        "label": "LATE CYCLE SQUEEZE",
        "color": "#ef4444",
        "interpretation": "Options crowd squeezing higher against a bearish regime and tactical — high risk of reversal.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "STRONG",
        "instruments_buy": [],
        "instruments_short": [
            ("SH / SDS", "Build short in tranches — fade the squeeze"),
            ("QQQ Puts", "45–60 DTE — defined risk on the squeeze reversal"),
        ],
        "entry_buy": "Don't chase the squeeze — regime and tactical are bearish.\nWait for regime to confirm before entering longs.",
        "entry_short": (
            "Build short position in tranches — don't front-run the squeeze\n"
            "Stop: above squeeze high\n"
            "Target: regime-implied support level\n"
            "Patience required — squeezes can persist 1–3 sessions"
        ),
    },
    "GENUINE_UNCERTAINTY": {
        "label": "GENUINE UNCERTAINTY",
        "color": "#475569",
        "interpretation": "No edge — signals are conflicting with no clear majority direction.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [],
        "instruments_short": [],
        "entry_buy": "Wait for at least 2 of 3 layers to align before committing capital.",
        "entry_short": "No directional edge in either direction — stay patient.",
    },
}


def _classify_signals(regime_ctx: dict, tac_ctx: dict, of_ctx: dict) -> dict:
    """Map three timing signal contexts to one of 7 dashboard patterns.

    Uses identical direction thresholds to the existing Short/Buy panel logic.
    Returns full pattern dict with label, color, tiers, instruments, and entry rules.
    """
    regime_ctx = regime_ctx or {}
    tac_ctx    = tac_ctx    or {}
    of_ctx     = of_ctx     or {}

    score        = regime_ctx.get("score", 0.0)
    regime_label = regime_ctx.get("regime", "")
    tac_score    = tac_ctx.get("tactical_score", 50) if tac_ctx else 50
    of_score     = of_ctx.get("options_score", 50)   if of_ctx  else 50

    # Thresholds match existing lines 221–226 of quick_run.py
    r_bull = "Risk-On"  in regime_label or score >  0.3
    r_bear = "Risk-Off" in regime_label or score < -0.3
    t_bull = tac_score >= 65
    t_bear = tac_score <  38
    o_bull = of_score  >= 65
    o_bear = of_score  <  38

    if   r_bull and t_bull and o_bull: pattern = "BULLISH_CONFIRMATION"
    elif r_bear and t_bear and o_bear: pattern = "BEARISH_CONFIRMATION"
    elif r_bull and t_bear and o_bull: pattern = "PULLBACK_IN_UPTREND"
    elif r_bull and t_bull and o_bear: pattern = "OPTIONS_FLOW_DIVERGENCE"
    elif r_bear and t_bull and o_bull: pattern = "BEAR_MARKET_BOUNCE"
    elif r_bear and t_bear and o_bull: pattern = "LATE_CYCLE_SQUEEZE"
    else:                              pattern = "GENUINE_UNCERTAINTY"

    return {"pattern": pattern, **_PATTERNS[pattern]}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_qir_dashboard.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('modules/quick_run.py', encoding='utf-8-sig').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_qir_dashboard.py modules/quick_run.py
git commit -m "feat(qir-dashboard): add _classify_signals() helper with 7 signal patterns"
```

---

## Chunk 2: Earnings risk fetch + signals cache

### Task 2: Fetch earnings risk in Round 5 and persist to session state

**Files:**
- Modify: `modules/quick_run.py` (Round 5 block, ~lines 695–714)
- Modify: `services/signals_cache.py` (add to `_SIGNAL_KEYS`)

- [ ] **Step 1: Add `_qir_earnings_risk` to `_SIGNAL_KEYS` in `services/signals_cache.py`**

Find the block ending with:
```python
    # StockTwits social sentiment (computed in QIR Round 1)
    "_stocktwits_digest",
    "_stocktwits_digest_ts",
]
```

Add after it:
```python
    # QIR Earnings Risk (held positions with earnings ≤21 days, computed in Round 5)
    "_qir_earnings_risk",
    "_qir_earnings_risk_ts",
]
```

- [ ] **Step 2: Verify signals_cache syntax**

```bash
python -c "import ast; ast.parse(open('services/signals_cache.py', encoding='utf-8-sig').read()); print('OK')"
```

- [ ] **Step 3: Add earnings fetch to QIR Round 5**

In `modules/quick_run.py`, find the Round 5 block (starts with `# ── Round 5: Portfolio Risk Snapshot`). After the existing `run_quick_risk_snapshot` call and its result handling, add the earnings risk fetch. The full Round 5 block should become:

```python
        # ── Round 5: Portfolio Risk Snapshot + Earnings Risk ──────────────
        with st.spinner("📊 Round 5/5 — Portfolio Risk Snapshot + Earnings Risk..."):
            try:
                from modules.trade_journal import run_quick_risk_snapshot
                _risk_ok = run_quick_risk_snapshot(use_claude=_use_claude, model=_cl_model)
                _results["risk_snapshot"] = _risk_ok
                if _risk_ok:
                    import datetime as _rsdt
                    st.session_state["_portfolio_risk_snapshot_ts"] = _rsdt.datetime.now()
                    _snap = st.session_state.get("_portfolio_risk_snapshot") or {}
                    _interp = st.session_state.get("_risk_matrix_interpretation") or {}
                    _alert = _interp.get("alert_level", "")
                    _alert_str = f" · Risk Alert: {_alert}" if _alert else ""
                    _beta = _snap.get("beta", "?")
                    st.success(f"✅ Portfolio Risk Snapshot — Beta {_beta}{_alert_str}")
                else:
                    st.info("ℹ️ Risk Snapshot skipped — no open positions in Trade Journal")
            except Exception as e:
                _results["risk_snapshot"] = False
                st.error(f"❌ Risk Snapshot failed: {e}")

            # ── Earnings Risk: fetch per held position ────────────────────
            try:
                from utils.journal import load_journal as _lj_r5
                from services.market_data import fetch_earnings_intelligence as _fei
                import datetime as _er_dt
                _open_tks = list({t["ticker"].upper() for t in _lj_r5() if t.get("status") == "open"})
                _earn_risk = []
                for _etk in _open_tks:
                    try:
                        _ei = _fei(_etk)
                        _ne = _ei.get("next_earnings") or {}
                        _em = _ei.get("expected_move") or {}
                        _days = _ne.get("days_away")
                        if _days is not None and _days <= 21:
                            _earn_risk.append({
                                "ticker": _etk,
                                "days_away": _days,
                                "date": _ne.get("date", ""),
                                "expected_move_pct": _em.get("pct"),
                                "expected_move_dollar": _em.get("dollar"),
                            })
                    except Exception:
                        continue
                _earn_risk.sort(key=lambda x: x["days_away"])
                if _earn_risk:
                    st.session_state["_qir_earnings_risk"] = _earn_risk
                    st.session_state["_qir_earnings_risk_ts"] = _er_dt.datetime.now()
                    _er_names = ", ".join(
                        f"{e['ticker']} ({e['days_away']}d)" for e in _earn_risk[:3]
                    )
                    st.success(f"✅ Earnings Risk — {len(_earn_risk)} position(s) flagged: {_er_names}")
                else:
                    st.info("ℹ️ No held positions with earnings in the next 21 days")
            except Exception as _er_e:
                st.warning(f"⚠ Earnings Risk scan failed: {_er_e}")
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "import ast; ast.parse(open('modules/quick_run.py', encoding='utf-8-sig').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add modules/quick_run.py services/signals_cache.py
git commit -m "feat(qir-dashboard): fetch earnings risk per held position in Round 5"
```

---

## Chunk 3: `_render_qir_dashboard()` + replace Short/Buy block

### Task 3: Build the dashboard render function

**Files:**
- Modify: `modules/quick_run.py`

- [ ] **Step 1: Add `_render_qir_dashboard()` to `modules/quick_run.py`**

Insert this function immediately after `_classify_signals()` (after the `_PATTERNS` dict and before `render()`):

```python
def _render_qir_dashboard() -> None:
    """Render the QIR Intelligence Dashboard.

    Always visible — greyed shell before QIR, activated with glow border after.
    Reads from session state only — no API calls.
    """
    from utils.theme import COLORS

    _rc  = st.session_state.get("_regime_context")  or {}
    _tac = st.session_state.get("_tactical_context") or {}
    _of  = st.session_state.get("_options_flow_context") or {}
    _er  = st.session_state.get("_qir_earnings_risk") or []
    _populated = bool(_rc or _tac or _of)

    # ── Signal classification ─────────────────────────────────────────────
    _cls = _classify_signals(_rc, _tac, _of)
    _border_color = _cls["color"] if _populated else "#1e293b"
    _border_glow  = f"0 0 8px {_cls['color']}44" if _populated else "none"

    # ── Timing Stack column ───────────────────────────────────────────────
    def _timing_row(icon, label, ctx, score_key, label_key, bull_fn, bear_fn):
        if not ctx:
            return (f'<div style="color:#374151;font-size:11px;padding:2px 0;">'
                    f'◌ {icon} {label} — <span style="color:#1f2937;">run QIR</span></div>')
        score = ctx.get(score_key, 0)
        lbl   = ctx.get(label_key, "")
        bull  = bull_fn(ctx)
        bear  = bear_fn(ctx)
        c     = "#22c55e" if bull else ("#ef4444" if bear else "#f59e0b")
        arrow = "▲" if bull else ("▼" if bear else "◆")
        return (f'<div style="color:{c};font-family:\'JetBrains Mono\',Consolas,monospace;'
                f'font-size:11px;padding:2px 0;">{arrow} {icon} {label}: '
                f'<span style="color:#e2e8f0;">{lbl}</span> '
                f'<span style="color:{c};font-size:10px;">({score:.0f}{"" if score_key == "score" else "/100"})</span>'
                f'</div>')

    _regime_score = _rc.get("score", 0)
    _regime_label = _rc.get("regime", "")
    _tac_score    = _tac.get("tactical_score", 50) if _tac else 50
    _of_score     = _of.get("options_score", 50)   if _of  else 50

    _t1 = (
        f'<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">TIMING STACK</div>'
    )
    if _rc:
        _r_bull = "Risk-On"  in _regime_label or _regime_score >  0.3
        _r_bear = "Risk-Off" in _regime_label or _regime_score < -0.3
        _rc_color = "#22c55e" if _r_bull else ("#ef4444" if _r_bear else "#f59e0b")
        _rc_arrow = "▲" if _r_bull else ("▼" if _r_bear else "◆")
        _t1 += (f'<div style="color:{_rc_color};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_rc_arrow} 📡 Regime: <span style="color:#e2e8f0;">{_regime_label[:28]}</span>'
                f'<span style="color:{_rc_color};font-size:10px;"> ({_regime_score:+.2f})</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ 📡 Regime — run QIR</div>'

    if _tac:
        _t_bull = _tac_score >= 65; _t_bear = _tac_score < 38
        _tc = "#22c55e" if _t_bull else ("#ef4444" if _t_bear else "#f59e0b")
        _ta = "▲" if _t_bull else ("▼" if _t_bear else "◆")
        _t1 += (f'<div style="color:{_tc};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_ta} ⚡ Tactical: <span style="color:#e2e8f0;">{_tac.get("label","")}</span>'
                f'<span style="color:{_tc};font-size:10px;"> ({_tac_score}/100)</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ ⚡ Tactical — run QIR</div>'

    if _of:
        _o_bull = _of_score >= 65; _o_bear = _of_score < 38
        _oc = "#22c55e" if _o_bull else ("#ef4444" if _o_bear else "#f59e0b")
        _oa = "▲" if _o_bull else ("▼" if _o_bear else "◆")
        _t1 += (f'<div style="color:{_oc};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_oa} 📊 Opt Flow: <span style="color:#e2e8f0;">{_of.get("label","")}</span>'
                f'<span style="color:{_oc};font-size:10px;"> ({_of_score}/100)</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ 📊 Opt Flow — run QIR</div>'

    # ── Macro Events column ───────────────────────────────────────────────
    _t2 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">MACRO EVENTS</div>'
    try:
        from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
        for _ev_label, _ev_fn in (("FOMC", get_next_fomc), ("CPI", get_next_cpi), ("NFP", get_next_nfp)):
            try:
                _ev = _ev_fn()
                _d  = _ev.get("days_away", 99)
                _dt = _ev.get("date", "")[:6]
                if   _d == 0:   _ec = "#ef4444"; _ds = "TODAY"
                elif _d == 1:   _ec = "#f97316"; _ds = "TMRW"
                elif _d <= 5:   _ec = "#f59e0b"; _ds = f"{_d}d"
                else:           _ec = "#475569"; _ds = f"{_d}d"
                _t2 += (f'<div style="font-size:11px;padding:1px 0;'
                        f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                        f'<span style="color:#64748b;">{_ev_label}</span>'
                        f'<span style="color:{_ec};"> {_ds}</span>'
                        f'<span style="color:#475569;font-size:10px;"> {_dt}</span></div>')
            except Exception:
                _t2 += f'<div style="color:#374151;font-size:11px;padding:1px 0;">{_ev_label} —</div>'
    except Exception:
        _t2 += '<div style="color:#374151;font-size:11px;">Macro events unavailable</div>'

    # ── Earnings Risk column ──────────────────────────────────────────────
    _t3 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">EARNINGS RISK</div>'
    if _er:
        for _e in _er[:4]:
            _ed = _e["days_away"]
            _em = _e.get("expected_move_pct")
            _em_str = f" ±{_em:.1f}%" if _em else ""
            if   _ed <= 3:  _ec2 = "#ef4444"; _eicon = "⚠"
            elif _ed <= 7:  _ec2 = "#f59e0b"; _eicon = "⚠"
            else:           _ec2 = "#475569"; _eicon = "📅"
            _t3 += (f'<div style="font-size:11px;padding:1px 0;'
                    f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                    f'<span style="color:{_ec2};">{_eicon} {_e["ticker"]}</span>'
                    f'<span style="color:#64748b;"> {_ed}d</span>'
                    f'<span style="color:{_ec2};font-size:10px;">{_em_str}</span></div>')
    else:
        _t3 += '<div style="color:#374151;font-size:11px;">No earnings ≤21d</div>'

    # ── Verdict section ───────────────────────────────────────────────────
    if _populated:
        _verdict_color = _cls["color"]
        _verdict_label = _cls["label"]
        _verdict_interp = _cls["interpretation"]
        _buy_tier   = _cls["buy_tier"]
        _short_tier = _cls["short_tier"]
        _instr_buy  = _cls["instruments_buy"]
        _instr_shrt = _cls["instruments_short"]
        _entry_buy  = _cls["entry_buy"]
        _entry_shrt = _cls["entry_short"]

        # Quadrant-specific instrument override (reuses existing quadrant logic)
        _quadrant = _rc.get("quadrant", "")
        if _cls["pattern"] in ("BULLISH_CONFIRMATION", "PULLBACK_IN_UPTREND"):
            if _quadrant == "Goldilocks":
                _instr_buy = [("XLK / QQQ", "Tech leads in Goldilocks — low rates, strong growth"),
                              ("XLY", "Consumer discretionary benefits from spending confidence")] + list(_instr_buy)
            elif _quadrant == "Overheating":
                _instr_buy = [("XLE", "Energy outperforms in overheating / commodity-driven growth"),
                              ("XLB", "Materials benefit from rising input prices")] + list(_instr_buy)
            elif _quadrant == "Reflation":
                _instr_buy = [("XLE / XLB", "Commodity producers lead in reflation"),
                              ("XLF", "Financials benefit from steepening yield curve")] + list(_instr_buy)
        elif _cls["pattern"] in ("BEARISH_CONFIRMATION", "LATE_CYCLE_SQUEEZE"):
            if _quadrant == "Stagflation":
                _instr_shrt = list(_instr_shrt) + [("XLY Puts", "Consumer discretionary crushed by stagflation"),
                                                    ("QQQ Puts", "Growth multiples compress fastest under stagflation")]
            elif _quadrant in ("Deflation", "Recession"):
                _instr_shrt = list(_instr_shrt) + [("XLF Puts", "Credit losses mount in deflation/recession"),
                                                    ("XLE Puts", "Demand collapses before supply adjusts")]

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

        # Earnings footer caveat (≤14 days)
        _earn_caveats = [
            f"⚠ {e['ticker']} earnings in {e['days_away']}d — size to survive "
            f"±{e['expected_move_pct']:.1f}% gap before the print"
            for e in _er
            if e["days_away"] <= 14 and e.get("expected_move_pct")
        ]

        # Build verdict HTML
        def _tier_badge(tier):
            _tc = {"STRONG": "#22c55e", "MODERATE": "#f59e0b",
                   "SELECTIVE": "#f97316", "NOT A BUYING ENV": "#ef4444",
                   "NOT A SHORTING ENV": "#22c55e"}.get(tier, "#64748b")
            return (f'<span style="background:{_tc};color:black;font-weight:800;'
                    f'font-size:9px;padding:1px 7px;border-radius:3px;">{tier}</span>')

        def _instruments_html(instruments):
            if not instruments:
                return ""
            rows = "".join(
                f'<div style="padding:2px 0;border-bottom:1px solid #1e293b;">'
                f'<span style="color:#f1f5f9;font-weight:700;font-size:10px;">{t}</span>'
                f'<span style="color:#94a3b8;font-size:10px;"> — {d}</span></div>'
                for t, d in instruments[:4]
            )
            return (f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin:6px 0 2px;">INSTRUMENTS</div>'
                    f'<div style="margin-bottom:6px;">{rows}</div>')

        def _entry_html(entry_text, label="ENTRY / RISK RULES"):
            rows = "".join(
                f'<div style="color:#94a3b8;font-size:10px;padding:1px 0;">· {r.strip()}</div>'
                for r in entry_text.strip().split("\n") if r.strip()
            )
            return (f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin:6px 0 2px;">{label}</div>{rows}')

        _verdict_html = (
            f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
            f'<div style="font-size:13px;font-weight:800;color:{_verdict_color};'
            f'letter-spacing:0.04em;margin-bottom:4px;">{_verdict_label}</div>'
            f'<div style="color:#94a3b8;font-size:11px;margin-bottom:10px;">{_verdict_interp}</div>'
        )
        if _vix_note:
            _verdict_html += (
                f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                f'padding:5px 10px;font-size:10px;color:#f59e0b;margin-bottom:8px;">{_vix_note}</div>'
            )

        # Buy + Short side by side
        _buy_html = (
            f'<div style="padding:8px;background:#0a1628;border:1px solid {_cls["color"]}22;border-radius:5px;">'
            f'<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">BUY SETUP</div>'
            f'{_tier_badge(_buy_tier)}'
            f'{_instruments_html(_instr_buy)}'
            f'{_entry_html(_entry_buy)}'
            f'</div>'
        )
        _short_html = (
            f'<div style="padding:8px;background:#160a0a;border:1px solid {_cls["color"]}22;border-radius:5px;">'
            f'<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">SHORT SETUP</div>'
            f'{_tier_badge(_short_tier)}'
            f'{_instruments_html(_instr_shrt)}'
            f'{_entry_html(_entry_shrt)}'
            f'</div>'
        )

        _verdict_html += (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">'
            f'{_buy_html}{_short_html}</div>'
        )

        if _earn_caveats:
            _verdict_html += "".join(
                f'<div style="background:#1a0a00;border-left:3px solid #ef4444;'
                f'padding:5px 10px;font-size:10px;color:#ef4444;margin-top:4px;">{c}</div>'
                for c in _earn_caveats
            )
    else:
        _verdict_html = (
            f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
            f'<div style="color:#374151;font-size:12px;text-align:center;padding:12px 0;">'
            f'Run QIR to activate the intelligence dashboard</div>'
        )

    # ── Render the full dashboard ─────────────────────────────────────────
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid {_border_color};border-radius:8px;'
        f'box-shadow:{_border_glow};padding:14px 16px;margin:8px 0 12px;">'
        f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;'
        f'text-transform:uppercase;margin-bottom:10px;">QIR Intelligence Dashboard</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
        f'<div>{_t1}</div><div>{_t2}</div><div>{_t3}</div>'
        f'</div>'
        f'{_verdict_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('modules/quick_run.py', encoding='utf-8-sig').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add modules/quick_run.py
git commit -m "feat(qir-dashboard): add _render_qir_dashboard() function"
```

---

## Chunk 4: Wire dashboard into QIR + remove old Short/Buy block

### Task 4: Replace Short/Buy block with dashboard call

**Files:**
- Modify: `modules/quick_run.py`

- [ ] **Step 1: Remove old Short/Buy block and wire in dashboard**

Find and delete the entire block from:
```python
    # ── Buy / Short Market Timing Panels (side by side) ────────────────────────
    _sh_rc  = st.session_state.get("_regime_context") or {}
```
to the end of the `with _col_buy:` block (just before `# ── Run button`).

Replace with a single call:
```python
    # ── QIR Intelligence Dashboard ─────────────────────────────────────────────
    _render_qir_dashboard()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('modules/quick_run.py', encoding='utf-8-sig').read()); print('OK')"
```

- [ ] **Step 3: Run existing tests to confirm nothing broken**

```bash
python -m pytest tests/test_qir_dashboard.py -v
```

- [ ] **Step 4: Commit**

```bash
git add modules/quick_run.py
git commit -m "feat(qir-dashboard): replace Short/Buy panels with fused dashboard"
```

---

## Chunk 5: Macro events in `_sig_parts` + downstream injections

### Task 5: Inject macro events into AI prompt + downstream modules

**Files:**
- Modify: `modules/quick_run.py` (`_sig_parts` block, ~line 623)
- Modify: `modules/valuation.py` (signals_text block)
- Modify: `modules/trade_journal.py` (`_upstream` dict)

- [ ] **Step 1: Add macro events to `_sig_parts` in `quick_run.py`**

Find the `_sig_parts` assembly block (look for `_sig_parts = []` and the block that builds it). Add macro event injection after the existing `_dq_ctx` block:

```python
        # ── Macro calendar context ────────────────────────────────────────
        try:
            from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
            _fomc = get_next_fomc(); _cpi = get_next_cpi(); _nfp = get_next_nfp()
            _sig_parts.append(
                f"MACRO CALENDAR: FOMC in {_fomc.get('days_away','?')}d ({_fomc.get('date','')})"
                f" | CPI in {_cpi.get('days_away','?')}d ({_cpi.get('date','')})"
                f" | NFP in {_nfp.get('days_away','?')}d ({_nfp.get('date','')})"
            )
        except Exception:
            pass

        # ── Earnings risk context ─────────────────────────────────────────
        _er_sig = st.session_state.get("_qir_earnings_risk") or []
        if _er_sig:
            _er_parts = [
                f"{e['ticker']} in {e['days_away']}d"
                + (f" (±{e['expected_move_pct']:.1f}%)" if e.get('expected_move_pct') else "")
                for e in _er_sig[:5]
            ]
            _sig_parts.append(f"EARNINGS RISK: {', '.join(_er_parts)}")
```

- [ ] **Step 2: Add earnings risk to Valuation `signals_text`**

In `modules/valuation.py`, find the StockTwits injection block (the one we added previously that starts with `# Inject StockTwits crowd sentiment`). Add immediately before it:

```python
    # Inject QIR Earnings Risk for current ticker
    _qir_er = st.session_state.get("_qir_earnings_risk") or []
    _ticker_er = next((e for e in _qir_er if e.get("ticker", "").upper() == ticker.upper()), None)
    if _ticker_er:
        _em_str = f", options pricing ±{_ticker_er['expected_move_pct']:.1f}% move (${_ticker_er.get('expected_move_dollar','?')})" if _ticker_er.get('expected_move_pct') else ""
        signals_text += f"\nEarnings Risk: {ticker} reports in {_ticker_er['days_away']}d{_em_str}"

```

- [ ] **Step 3: Add earnings risk to Portfolio Intelligence `_upstream`**

In `modules/trade_journal.py`, find the `_upstream` dict (look for `"social_sentiment":` which we added previously). Add after it:

```python
                    "earnings_risk": "; ".join(
                        f"{e['ticker']} in {e['days_away']}d"
                        + (f" ±{e['expected_move_pct']:.1f}%" if e.get('expected_move_pct') else "")
                        for e in (st.session_state.get("_qir_earnings_risk") or [])
                    ) or "",
```

- [ ] **Step 4: Verify all syntax**

```bash
python -c "
import ast
for f in ['modules/quick_run.py','modules/valuation.py','modules/trade_journal.py']:
    ast.parse(open(f, encoding='utf-8-sig').read())
    print(f'OK  {f}')
"
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_qir_dashboard.py -v
```

- [ ] **Step 6: Final commit**

```bash
git add modules/quick_run.py modules/valuation.py modules/trade_journal.py
git commit -m "feat(qir-dashboard): macro events in AI prompt + earnings risk downstream injections"
```

---

## Final Verification

- [ ] App starts without error: `streamlit run app.py` — check for import errors in terminal
- [ ] Dashboard shell visible before QIR (greyed, "Run QIR to activate")
- [ ] After QIR: border glows, correct pattern label and verdict visible
- [ ] Buy/Short tiers and instruments render in two-column layout
- [ ] Earnings Risk column shows held positions ≤21 days
- [ ] Footer warning fires for positions ≤14 days
- [ ] Macro Events column shows FOMC/CPI/NFP countdowns
- [ ] Signal Coverage bar shows 15/15 after full QIR run
