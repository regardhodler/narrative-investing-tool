"""Quick Intel Run — one button runs Regime + Rate-Path Plays + Current Events + Doom Briefing."""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.theme import COLORS
from utils.ai_tier import TIER_OPTS, TIER_MAP, MODEL_HINT_HTML


# ── QIR Dashboard helpers ────────────────────────────────────────────────────

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


def _render_qir_dashboard() -> None:
    """Render the QIR Intelligence Dashboard.

    Always visible — greyed shell before QIR, activated with glow border after.
    Reads from session state only — no API calls.
    """
    _rc  = st.session_state.get("_regime_context")       or {}
    _tac = st.session_state.get("_tactical_context")     or {}
    _of  = st.session_state.get("_options_flow_context") or {}
    _er  = st.session_state.get("_qir_earnings_risk")    or []
    _populated = bool(_rc or _tac or _of)

    # ── Signal classification ─────────────────────────────────────────────
    _cls = _classify_signals(_rc, _tac, _of)
    _border_color = _cls["color"] if _populated else "#1e293b"
    _border_glow  = f"0 0 8px {_cls['color']}44" if _populated else "none"

    # ── Timing Stack column ───────────────────────────────────────────────
    _regime_score = _rc.get("score", 0)
    _regime_label = _rc.get("regime", "")
    _tac_score    = _tac.get("tactical_score", 50) if _tac else 50
    _of_score     = _of.get("options_score", 50)   if _of  else 50

    _t1 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">TIMING STACK</div>'
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
                if   _d == 0:  _ec = "#ef4444"; _ds = "TODAY"
                elif _d == 1:  _ec = "#f97316"; _ds = "TMRW"
                elif _d <= 5:  _ec = "#f59e0b"; _ds = f"{_d}d"
                else:          _ec = "#475569"; _ds = f"{_d}d"
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
            if   _ed <= 3: _ec2 = "#ef4444"; _eicon = "⚠"
            elif _ed <= 7: _ec2 = "#f59e0b"; _eicon = "⚠"
            else:          _ec2 = "#475569"; _eicon = "📅"
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
        _instr_buy  = list(_cls["instruments_buy"])
        _instr_shrt = list(_cls["instruments_short"])
        _entry_buy  = _cls["entry_buy"]
        _entry_shrt = _cls["entry_short"]

        # Quadrant-specific instrument override
        _quadrant = _rc.get("quadrant", "")
        if _cls["pattern"] in ("BULLISH_CONFIRMATION", "PULLBACK_IN_UPTREND"):
            if _quadrant == "Goldilocks":
                _instr_buy = [("XLK / QQQ", "Tech leads in Goldilocks — low rates, strong growth"),
                              ("XLY", "Consumer discretionary benefits from spending confidence")] + _instr_buy
            elif _quadrant == "Overheating":
                _instr_buy = [("XLE", "Energy outperforms in overheating / commodity-driven growth"),
                              ("XLB", "Materials benefit from rising input prices")] + _instr_buy
            elif _quadrant == "Reflation":
                _instr_buy = [("XLE / XLB", "Commodity producers lead in reflation"),
                              ("XLF", "Financials benefit from steepening yield curve")] + _instr_buy
        elif _cls["pattern"] in ("BEARISH_CONFIRMATION", "LATE_CYCLE_SQUEEZE"):
            if _quadrant == "Stagflation":
                _instr_shrt = _instr_shrt + [("XLY Puts", "Consumer discretionary crushed by stagflation"),
                                              ("QQQ Puts", "Growth multiples compress fastest under stagflation")]
            elif _quadrant in ("Deflation", "Recession"):
                _instr_shrt = _instr_shrt + [("XLF Puts", "Credit losses mount in deflation/recession"),
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

        # Build verdict HTML helpers
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


def render():
    _oc = COLORS["bloomberg_orange"]

    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:2px;">⚡ QUICK INTEL RUN</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Runs Risk Regime + Fed Rate Path + Policy Transmission + Current Events + Doom Briefing + Whale Activity + Black Swans + Macro Synopsis + Portfolio Risk Snapshot in sequence. "
        "Navigate to Portfolio Intelligence when done."
    )

    # ── VIX live ticker ────────────────────────────────────────────────────────
    try:
        import yfinance as _yf
        @st.cache_data(ttl=120)
        def _fetch_vix_change():
            _v = _yf.Ticker("^VIX")
            _h = _v.history(period="2d", interval="1d")
            if _h is None or len(_h) < 2:
                return None, None, None
            _prev = float(_h["Close"].iloc[-2])
            _cur  = float(_h["Close"].iloc[-1])
            _chg  = _cur - _prev
            _pct  = (_chg / _prev) * 100
            return round(_cur, 2), round(_chg, 2), round(_pct, 2)

        _vix_cur, _vix_chg, _vix_pct = _fetch_vix_change()
        if _vix_cur is not None:
            if _vix_pct >= 10:
                _vc = "#ef4444"; _vglow = "0 0 10px #ef444488"; _vbg = "#2d0a0a"; _varrow = "▲"
            elif _vix_pct >= 3:
                _vc = "#f97316"; _vglow = "0 0 8px #f9731666"; _vbg = "#1f1000"; _varrow = "▲"
            elif _vix_pct > 0:
                _vc = "#f59e0b"; _vglow = "none"; _vbg = "#1a1200"; _varrow = "▲"
            elif _vix_pct <= -10:
                _vc = "#22c55e"; _vglow = "0 0 10px #22c55e88"; _vbg = "#052e16"; _varrow = "▼"
            elif _vix_pct <= -3:
                _vc = "#22c55e"; _vglow = "0 0 8px #22c55e55"; _vbg = "#0c1a0c"; _varrow = "▼"
            elif _vix_pct < 0:
                _vc = "#4ade80"; _vglow = "none"; _vbg = "#0c1a0c"; _varrow = "▼"
            else:
                _vc = "#475569"; _vglow = "none"; _vbg = "#0d1117"; _varrow = "◆"
            _vsign = "+" if _vix_chg >= 0 else ""
            st.markdown(
                f'<div style="background:{_vbg};border:1px solid {_vc}55;border-radius:6px;'
                f'box-shadow:{_vglow};padding:7px 14px;margin-bottom:8px;'
                f'display:inline-flex;align-items:center;gap:10px;">'
                f'<span style="color:#475569;font-size:10px;font-weight:700;letter-spacing:0.1em;">VIX</span>'
                f'<span style="color:{_vc};font-size:18px;font-weight:800;font-family:monospace;'
                f'text-shadow:{_vglow};">{_vix_cur}</span>'
                f'<span style="color:{_vc};font-size:12px;font-weight:700;">'
                f'{_varrow} {_vsign}{_vix_chg} ({_vsign}{_vix_pct}%)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Highly Regarded gate: check if conditions warrant Sonnet ──────────────
    def _hr_gate_check() -> tuple[bool, str]:
        """Returns (unlocked, reason). Unlocked when macro stress or event warrants Sonnet."""
        reasons = []
        ctx = st.session_state.get("_regime_context", {})
        regime = ctx.get("regime", "")
        score = ctx.get("score", 0)
        quadrant = ctx.get("quadrant", "")

        if "Risk-Off" in regime:
            reasons.append("Risk-Off regime")
        if score < -0.3:
            reasons.append(f"bearish score ({score:+.2f})")
        if quadrant in ("Stagflation", "Deflation"):
            reasons.append(f"{quadrant} quadrant")

        try:
            from datetime import date, timedelta
            from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
            import datetime as _dt
            today = date.today()
            window = today + timedelta(days=1)
            for fn, label in [(get_next_fomc, "FOMC"), (get_next_cpi, "CPI"), (get_next_nfp, "NFP")]:
                ev = fn()
                ev_date = ev.get("date")
                if ev_date:
                    if isinstance(ev_date, str):
                        ev_date = _dt.datetime.strptime(ev_date[:10], "%Y-%m-%d").date()
                    if today <= ev_date <= window:
                        reasons.append(f"{label} day")
        except Exception:
            pass

        doom = st.session_state.get("_doom_briefing", "")
        if doom and any(w in doom.upper() for w in ["CRITICAL", "SEVERE", "HIGH STRESS", "EXTREME"]):
            reasons.append("elevated stress signals")

        return bool(reasons), " · ".join(reasons)

    # ── Engine selector ────────────────────────────────────────────────────────
    import os
    _has_xai      = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    _hr_unlocked, _hr_reason = _hr_gate_check()

    _rec_map = {
        "⚡ Freeloader Mode":      "Daily routine — all 8 modules in ~90s, completely free.",
        "🧠 Regard Mode":            "Active day — Grok 4.1 reasoning for deeper synthesis + live X feed in Current Events.",
        "👑 Highly Regarded Mode":   "High conviction — Sonnet on all 8 modules before running Portfolio.",
    }
    _sel = st.radio("Engine", TIER_OPTS, horizontal=True, key="qr_engine")
    st.markdown(MODEL_HINT_HTML, unsafe_allow_html=True)
    st.caption(f"💡 {_rec_map.get(_sel, '')}")

    # Show gate status
    if _has_xai and _has_anthropic and not _hr_unlocked:
        st.markdown(
            f'<div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:#f59e0b;margin-bottom:4px;">'
            f'🔒 <b>Highly Regarded Mode locked</b> — unlocks on Risk-Off regime, '
            f'Stagflation/Deflation quadrant, bearish macro score, or FOMC/CPI/NFP day</div>',
            unsafe_allow_html=True,
        )
    elif _hr_unlocked and _has_anthropic:
        st.markdown(
            f'<div style="background:#1a0d00;border:1px solid #FF881166;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:{_oc};margin-bottom:4px;">'
            f'🔓 <b>Highly Regarded Mode unlocked</b> — {_hr_reason}</div>',
            unsafe_allow_html=True,
        )
    elif not _has_xai and not _has_anthropic:
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:#8b949e;margin-bottom:4px;">'
            f'ℹ️ Add <b>XAI_API_KEY</b> to unlock Regard Mode (Grok 4.1) · '
            f'Add <b>ANTHROPIC_API_KEY</b> for Highly Regarded (Sonnet)</div>',
            unsafe_allow_html=True,
        )

    # Confirmation gate when Highly Regarded is selected
    _use_claude, _cl_model = TIER_MAP[_sel]
    if _sel == "👑 Highly Regarded Mode":
        st.warning("👑 Highly Regarded uses Claude Sonnet — reserve for elevated volatility or high-conviction sessions.")
        _confirmed = st.checkbox("I confirm this is a high-conviction session", key="qr_hr_confirm")
        if not _confirmed:
            _use_claude, _cl_model = True, "grok-4-1-fast-reasoning"
            st.caption("*Running in Regard Mode until confirmed.*")

    # ── Signal readiness ───────────────────────────────────────────────────────
    _signal_keys = ["_regime_context", "_tactical_context", "_options_flow_context", "_dominant_rate_path", "_rp_plays_result", "_fed_plays_result", "_current_events_digest", "_doom_briefing", "_chain_narration", "_custom_swans", "_whale_summary", "_macro_synopsis", "_portfolio_risk_snapshot", "_stocktwits_digest"]
    _signal_labels = ["Regime", "Tactical", "Opt Flow", "Fed Rate Path", "Rate-Path Plays", "Fed Plays", "News Digest", "Doom Briefing", "Policy Trans.", "Black Swans", "Whale Activity", "Macro Synopsis", "Risk Snapshot", "Social Sentiment"]
    _populated = [(k, l) for k, l in zip(_signal_keys, _signal_labels) if st.session_state.get(k)]

    if _populated:
        _badges = " &nbsp;".join(
            f'<span style="background:#052e16;color:#22c55e;border:1px solid #22c55e44;'
            f'border-radius:3px;padding:1px 7px;font-size:10px;">{l}</span>'
            for _, l in _populated
        )
        st.markdown(
            f'<div style="margin:6px 0 6px 0;">{_badges}</div>',
            unsafe_allow_html=True,
        )
        _dq_persistent = st.session_state.get("_data_quality") or {}
        if _dq_persistent:
            _dqp_col = "#22c55e" if _dq_persistent["score"] >= 80 else ("#f59e0b" if _dq_persistent["score"] >= 60 else "#ef4444")
            _dqp_bg  = "#052e16" if _dq_persistent["score"] >= 80 else ("#1a1200" if _dq_persistent["score"] >= 60 else "#2d0a0a")
            _dqp_label = _dq_persistent.get("label", "")
            _dqp_stale = _dq_persistent.get("stale_market", []) + _dq_persistent.get("stale_fred", [])
            _dqp_detail = f" · Stale: {', '.join(_dqp_stale[:4])}" if _dqp_stale else ""
            st.markdown(
                f'<div style="background:{_dqp_bg};border:1px solid {_dqp_col}66;border-radius:5px;'
                f'padding:7px 14px;margin-bottom:6px;display:flex;align-items:center;gap:12px;">'
                f'<span style="color:{_dqp_col};font-weight:800;font-size:15px;font-family:monospace;">'
                f'{_dq_persistent["score"]}/100</span>'
                f'<span style="color:{_dqp_col};font-size:11px;font-weight:700;letter-spacing:0.06em;">DATA QUALITY</span>'
                f'<span style="color:#94a3b8;font-size:11px;">{_dqp_label}{_dqp_detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _of_persistent = st.session_state.get("_options_flow_context") or {}
        if _of_persistent:
            _ofp_score = _of_persistent.get("options_score", 50)
            _ofp_col = "#22c55e" if _ofp_score >= 65 else ("#f59e0b" if _ofp_score >= 38 else "#ef4444")
            _ofp_bg  = "#052e16" if _ofp_score >= 65 else ("#1a1200" if _ofp_score >= 38 else "#2d0a0a")
            st.markdown(
                f'<div style="background:{_ofp_bg};border:1px solid {_ofp_col}66;border-radius:5px;'
                f'padding:7px 14px;margin-bottom:10px;display:flex;align-items:center;gap:12px;">'
                f'<span style="color:{_ofp_col};font-weight:800;font-size:15px;font-family:monospace;">'
                f'{_ofp_score}/100</span>'
                f'<span style="color:{_ofp_col};font-size:11px;font-weight:700;letter-spacing:0.06em;">OPTIONS FLOW</span>'
                f'<span style="color:#94a3b8;font-size:11px;">{_of_persistent.get("label","")} — {_of_persistent.get("action_bias","")[:60]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Score Interpretation Reference ─────────────────────────────────────────
    with st.expander("📖 How to read Tactical + Options Flow scores", expanded=False):
        st.markdown("""
**Both scores run 0–100. Higher = more bullish conditions.**

---

### Tactical Score (days/weeks layer)
Measures near-term market internals: momentum, breadth, credit spreads, bond yields, volatility.

| Score | Label | What to do |
|-------|-------|------------|
| ≥65 | Favorable Entry | Conditions support adding risk |
| 52–64 | Neutral | Hold existing positions, wait for clearer signal |
| 38–51 | Caution | Reduce new buys, tighten stops |
| <38 | Risk-Off | Defensive posture — consider exits or hedges |

---

### Options Flow Score (hours/days layer)
Measures what SPY options participants are doing *right now*: put/call ratio, gamma positioning, IV skew, and unusual flow.

| Score | Label | What it means |
|-------|-------|---------------|
| ≥65 | Call-Skewed Flow | Crowd positioned bullish — calls dominant, low fear skew |
| 52–64 | Neutral Flow | Mixed positioning, no strong lean |
| 38–51 | Put-Skewed Flow | Elevated hedging, cautious tone |
| <38 | Bearish Hedging | Heavy put buying, fear premium elevated |

**Sub-signals:**
- **P/C Ratio** (weight 3×) — Put vol ÷ call vol. Below 0.9 = bullish crowd. Above 1.2 = hedging.
- **Gamma Zone** (1.5×) — Positive gamma stabilizes prices (dealers fade moves). Negative gamma amplifies them (dealers chase).
- **IV Skew** (2×) — OTM put IV ÷ OTM call IV. Skew > 1.4 = expensive tail hedges = fear.
- **Unusual Activity Bias** (1.5×) — Vol/OI > 2× = unusual. Call-heavy unusual flow = bullish. Put-heavy = defensive.

---

### Reading them together

| Tactical | Opt Flow | Interpretation |
|----------|----------|----------------|
| ≥65 | ≥65 | **Strong alignment — highest conviction long entry** |
| ≥65 | 38–64 | Tactical setup good, crowd cautious — enter smaller |
| 38–64 | ≥65 | Options bullish but macro not yet confirmed — wait |
| <38 | <38 | **Avoid new longs — both layers warn** |
| <38 | ≥65 | Crowd complacent in a weak tape — possible contrarian short setup |

> **Rule of thumb:** Tactical tells you *when* to enter. Options Flow tells you *what the crowd is doing today*. Alignment between both = higher conviction. Divergence = reduce size or wait.
        """)

    # ── QIR Intelligence Dashboard ─────────────────────────────────────────────
    _render_qir_dashboard()

    # ── Run button ─────────────────────────────────────────────────────────────
    if st.button("⚡ RUN ALL INTEL MODULES", type="primary", key="qr_run_all", use_container_width=True):

        # Clear ALL cached data so every module fetches fresh on this run
        st.cache_data.clear()

        _results = {}
        _macro_ctx, _fred_data = {}, {}

        # ── Round 1 (parallel): Regime + Current Events + Whale + Options Flow + StockTwits ──
        # These five are fully independent — run concurrently to save time.
        from modules.risk_regime import run_quick_regime
        from modules.current_events import run_quick_digest
        from modules.whale_buyers import run_quick_whale
        from modules.options_activity import run_quick_options_flow
        from services.stocktwits_client import run_quick_stocktwits

        with st.spinner("📡 Round 1/4 — Regime · Current Events · Whale · Options Flow · Social (parallel)..."):
            _r1_errors = {}
            _macro_ctx, _fred_data = None, None
            import datetime as _dt_qir
            with ThreadPoolExecutor(max_workers=5) as _pool:
                _fut_regime = _pool.submit(run_quick_regime, _use_claude, _cl_model)
                _fut_digest = _pool.submit(run_quick_digest, _use_claude, _cl_model)
                _fut_whale  = _pool.submit(run_quick_whale,  _use_claude, _cl_model)
                _fut_opts   = _pool.submit(run_quick_options_flow, _use_claude, _cl_model)
                _fut_stwit  = _pool.submit(run_quick_stocktwits)
                for _fut, _key in (
                    (_fut_regime, "regime"),
                    (_fut_digest, "digest"),
                    (_fut_whale,  "whale"),
                    (_fut_opts,   "opts"),
                    (_fut_stwit,  "social"),
                ):
                    try:
                        _val = _fut.result()
                        if _key == "regime" and _val:
                            _macro_ctx, _fred_data, _tac_data, _tac_text, _regime_ctx, _plays, _tier, _dq = _val
                            # Write ALL regime data from main thread
                            st.session_state["_regime_context"] = _regime_ctx
                            st.session_state["_regime_context_ts"] = _dt_qir.datetime.now()
                            st.session_state["_rp_plays_result"] = _plays
                            st.session_state["_rp_plays_last_tier"] = _tier
                            if _tac_data:
                                st.session_state["_tactical_context"] = _tac_data
                                st.session_state["_tactical_context_ts"] = _dt_qir.datetime.now()
                            if _tac_text:
                                st.session_state["_tactical_analysis"] = _tac_text
                                st.session_state["_tactical_analysis_ts"] = _dt_qir.datetime.now()
                            if _dq:
                                st.session_state["_data_quality"] = _dq
                                st.session_state["_data_quality_ts"] = _dt_qir.datetime.now()
                                # Persist to alerts_config so background worker and alert checks can read it
                                try:
                                    from utils.alerts_config import load_config as _lc, save_config as _sc
                                    _ac = _lc()
                                    _ac["current_data_quality_score"] = _dq["score"]
                                    _ac["last_tactical_score"] = _ac.get("current_tactical_score")
                                    if _tac_data:
                                        _ac["current_tactical_score"] = _tac_data["tactical_score"]
                                    _sc(_ac)
                                except Exception:
                                    pass
                        elif _key == "digest" and _val:
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        elif _key == "whale" and _val:
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        elif _key == "opts" and _val:
                            st.session_state["_options_flow_context"] = _val
                            st.session_state["_options_flow_context_ts"] = _dt_qir.datetime.now()
                        elif _key == "social" and _val:
                            st.session_state["_stocktwits_digest"] = _val
                            st.session_state["_stocktwits_digest_ts"] = _dt_qir.datetime.now()
                        _results[_key] = bool(_val)
                    except Exception as _e:
                        _results[_key] = False
                        _r1_errors[_key] = str(_e)

        _regime_ok = _results.get("regime", False)
        if _regime_ok:
            st.success("✅ Risk Regime + Rate-Path Plays — done")
        else:
            st.error(f"❌ Regime failed: {_r1_errors.get('regime', '?')}")
        if _results.get("digest"):
            st.success("✅ Current Events Digest — done")
        else:
            st.warning(f"⚠ Digest skipped — {_r1_errors.get('digest', 'no content available')}")
        if _results.get("whale"):
            st.success("✅ Whale Activity — done")
        else:
            st.warning(f"⚠ Whale scan: {_r1_errors.get('whale', 'no data returned')}")
        if _results.get("opts"):
            _of_r1 = st.session_state.get("_options_flow_context") or {}
            st.success(f"✅ Options Flow — {_of_r1.get('label','?')} ({_of_r1.get('options_score','?')}/100)")
        else:
            st.warning(f"⚠ Options Flow: {_r1_errors.get('opts', 'SPY chain unavailable')}")
        if _results.get("social"):
            _st_r1 = st.session_state.get("_stocktwits_digest") or {}
            _st_mood = _st_r1.get("market_mood", "?")
            _st_bull = _st_r1.get("overall_bull_pct", "?")
            _st_top = ", ".join(_st_r1.get("top_bullish", [])[:3])
            st.success(f"✅ Social Sentiment — {_st_mood} ({_st_bull}% bull) · trending: {_st_top}")
        else:
            st.warning(f"⚠ Social Sentiment: {_r1_errors.get('social', 'StockTwits unavailable')}")

        # ── Round 2 (parallel): Fed + Doom + Black Swans ──────────────────────
        # Fed uses regime output from Round 1. Doom reads digest from session_state.
        # Black Swans are fully independent.
        from modules.fed_forecaster import run_quick_fed, run_quick_swans
        from modules.stress_signals import run_quick_doom

        with st.spinner("📈 Round 2/4 — Fed Rate Path · Doom Briefing · Black Swans (parallel)..."):
            _r2_errors = {}
            with ThreadPoolExecutor(max_workers=3) as _pool2:
                _fut_fed   = _pool2.submit(run_quick_fed, _macro_ctx, _fred_data, _use_claude, _cl_model)
                _fut_doom  = _pool2.submit(run_quick_doom, _use_claude, _cl_model)
                _fut_swans = _pool2.submit(run_quick_swans, _use_claude, _cl_model)
                for _fut, _key in ((_fut_fed, "fed"), (_fut_doom, "doom"), (_fut_swans, "swans")):
                    try:
                        _val = _fut.result()
                        if _val and isinstance(_val, dict):
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        _results[_key] = bool(_val)
                    except Exception as _e:
                        _results[_key] = False
                        _r2_errors[_key] = str(_e)

        if _results.get("fed"):
            st.success("✅ Fed Rate Path — done")
        else:
            st.error(f"❌ Fed Rate Path failed: {_r2_errors.get('fed', '?')}")
        if _results.get("doom"):
            st.success("✅ Doom Briefing — done")
        else:
            st.error(f"❌ Doom Briefing failed: {_r2_errors.get('doom', '?')}")
        if _results.get("swans"):
            _bs_count = len(st.session_state.get("_custom_swans", {}))
            st.success(f"✅ Black Swans — {_bs_count} scenarios ready")
        else:
            st.warning(f"⚠ Black Swans: {_r2_errors.get('swans', 'no results')}")

        # ── Round 3: Policy Transmission (needs Fed output from Round 2) ───────
        with st.spinner("🔗 Round 3/4 — Policy transmission path..."):
            try:
                from modules.fed_forecaster import run_quick_chain
                ok = run_quick_chain(use_claude=_use_claude, model=_cl_model)
                _results["chain"] = ok
                if ok:
                    import datetime as _cdt
                    st.session_state["_chain_narration_ts"] = _cdt.datetime.now()
                    st.success("✅ Policy Transmission — done")
                else:
                    st.warning("⚠ Policy Transmission skipped — rate path not available")
            except Exception as e:
                _results["chain"] = False
                st.error(f"❌ Policy Transmission failed: {e}")

        # ── Round 4: Macro Conviction Synopsis (cross-signal coherence check) ───
        with st.spinner("🧠 Round 4/4 — Macro Conviction Synopsis..."):
            try:
                import datetime as _syndt
                from services.claude_client import generate_macro_synopsis as _gen_synopsis

                # Build signal summary from session state
                _rc = st.session_state.get("_regime_context") or {}
                _dp = st.session_state.get("_dominant_rate_path") or {}
                _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
                _sig_parts = []
                _dq_ctx = st.session_state.get("_data_quality") or {}
                if _dq_ctx:
                    _sig_parts.append(f"DATA QUALITY: {_dq_ctx['score']}/100 — {_dq_ctx['label']}")

                # ── Tactical Regime first — actionable timeframe, highest weight ──
                _tac = st.session_state.get("_tactical_context")
                if _tac:
                    _tac_sigs = _tac.get("signals", [])
                    _tac_sig_str = "  |  ".join(
                        f"{s['Signal'].split('(')[0].strip()}: {s['Value']} ({s['Direction']})"
                        for s in _tac_sigs
                    ) if _tac_sigs else ""
                    _tac_block = (
                        f"TACTICAL REGIME: {_tac['tactical_score']}/100 ({_tac['label']}) — {_tac['action_bias']}"
                    )
                    if _tac_sig_str:
                        _tac_block += f"\n  Signals: {_tac_sig_str}"
                    _tac_ai_text = st.session_state.get("_tactical_analysis", "")
                    if _tac_ai_text:
                        _tac_block += f"\n  AI Analysis: {_tac_ai_text[:500]}"
                    _sig_parts.append(_tac_block)

                # ── Options Flow (hours/days layer) ──────────────────────────────
                _of = st.session_state.get("_options_flow_context")
                if _of:
                    _of_block = f"OPTIONS FLOW: {_of['options_score']}/100 ({_of['label']}) — {_of['action_bias']}"
                    _of_sigs = "  |  ".join(
                        f"{s['Signal']}: {s['Value']} ({s['Direction']})"
                        for s in _of.get("signals", [])
                    )
                    if _of_sigs:
                        _of_block += f"\n  Signals: {_of_sigs}"
                    _sig_parts.append(_of_block)

                # ── Macro Regime ──────────────────────────────────────────────────
                if _rc.get("regime"):
                    _sig_parts.append(f"MACRO REGIME: {_rc['regime']} (score {_rc.get('score',0):+.2f}) | Quadrant: {_rc.get('quadrant','')}")
                if _dp.get("scenario"):
                    _sig_parts.append(f"FED RATE PATH: {_dp_labels.get(_dp['scenario'], _dp['scenario'])} ({_dp.get('prob_pct',0):.0f}% probability)")
                _doom = st.session_state.get("_doom_briefing", "")
                if _doom:
                    _sig_parts.append(f"DOOM BRIEFING: {_doom[:400]}")
                _digest = st.session_state.get("_current_events_digest", "")
                if _digest:
                    _sig_parts.append(f"NEWS DIGEST: {_digest[:400]}")
                _chain = st.session_state.get("_chain_narration", "")
                if _chain:
                    _sig_parts.append(f"POLICY TRANSMISSION: {_chain[:300]}")
                _whale = st.session_state.get("_whale_summary", "")
                if _whale:
                    _sig_parts.append(f"WHALE ACTIVITY: {_whale[:300]}")
                _bs = st.session_state.get("_custom_swans", {})
                if _bs:
                    _bs_summary = "; ".join(
                        f"{k} ({v.get('probability_pct', 0):.1f}% annual)"
                        for k, v in list(_bs.items())[:3]
                    )
                    _sig_parts.append(f"BLACK SWANS: {_bs_summary}")

                # ── Macro calendar context ────────────────────────────────
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

                # ── Earnings risk context ─────────────────────────────────
                _er_sig = st.session_state.get("_qir_earnings_risk") or []
                if _er_sig:
                    _er_parts = [
                        f"{e['ticker']} in {e['days_away']}d"
                        + (f" (±{e['expected_move_pct']:.1f}%)" if e.get('expected_move_pct') else "")
                        for e in _er_sig[:5]
                    ]
                    _sig_parts.append(f"EARNINGS RISK: {', '.join(_er_parts)}")

                _synopsis = _gen_synopsis("\n\n".join(_sig_parts), use_claude=_use_claude, model=_cl_model)
                _syn_tier = "👑 Highly Regarded Mode" if (_use_claude and _cl_model == "claude-sonnet-4-6") \
                    else ("🧠 Regard Mode" if _use_claude else "⚡ Freeloader Mode")
                st.session_state["_macro_synopsis"] = _synopsis
                st.session_state["_macro_synopsis_ts"] = _syndt.datetime.now()
                st.session_state["_macro_synopsis_engine"] = _syn_tier
                _results["synopsis"] = True
                st.success(f"✅ Macro Conviction Synopsis — {_synopsis.get('conviction', '?')}")
            except Exception as e:
                _results["synopsis"] = False
                st.error(f"❌ Synopsis failed: {e}")

        # ── Round 5: Portfolio Risk Snapshot (headless risk matrix) ──────────
        with st.spinner("📊 Round 5/5 — Portfolio Risk Snapshot + AI Interpretation..."):
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

        # ── Completion summary ─────────────────────────────────────────────────
        _n_ok = sum(1 for v in _results.values() if v)
        if _n_ok == 9:
            st.markdown(
                f'<div style="background:#052e16;border:1px solid #22c55e;border-radius:6px;'
                f'padding:12px 16px;margin-top:10px;">'
                f'<div style="color:#22c55e;font-weight:700;font-size:13px;">✅ All intel modules ready</div>'
                f'<div style="color:#86efac;font-size:11px;margin-top:4px;">'
                f'Navigate to <b>Portfolio Intelligence</b> to run your final analysis.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            try:
                from services.telegram_client import send_alert as _tg_qir
                _rc_f  = st.session_state.get("_regime_context") or {}
                _tac_f = st.session_state.get("_tactical_context") or {}
                _dq_f  = st.session_state.get("_data_quality") or {}
                _of_f = st.session_state.get("_options_flow_context") or {}
                _tg_qir(
                    f"⚡ <b>Quick Intel Run Complete</b>\n"
                    f"Regime: {_rc_f.get('regime','?')} | {_rc_f.get('quadrant','?')}\n"
                    f"Tactical: {_tac_f.get('tactical_score','?')}/100 — {_tac_f.get('label','?')}\n"
                    f"Options Flow: {_of_f.get('options_score','?')}/100 — {_of_f.get('label','')}\n"
                    f"Data Quality: {_dq_f.get('score','?')}/100 — {_dq_f.get('label','')}"
                )
            except Exception:
                pass
        else:
            st.warning(f"{_n_ok}/9 modules completed — check errors above.")

        # ── Signal Coverage Panel ──────────────────────────────────────────────
        from utils.components import render_signal_coverage
        render_signal_coverage()
        _dq_disp = st.session_state.get("_data_quality") or {}
        if _dq_disp:
            _dq_col = "#22c55e" if _dq_disp["score"] >= 80 else ("#f59e0b" if _dq_disp["score"] >= 60 else "#ef4444")
            _dq_stale = ""
            if _dq_disp.get("stale_market") or _dq_disp.get("stale_fred"):
                _all_stale = _dq_disp.get("stale_market", []) + _dq_disp.get("stale_fred", [])
                _dq_stale = f' · Stale: {", ".join(_all_stale[:4])}'
            st.markdown(
                f'<div style="font-size:10px;color:{_dq_col};font-weight:600;margin-top:4px;">'
                f'DATA QUALITY: {_dq_disp["score"]}/100 — {_dq_disp["label"]}{_dq_stale}</div>',
                unsafe_allow_html=True,
            )

        # ── Previews ───────────────────────────────────────────────────────────
        _digest = st.session_state.get("_current_events_digest", "")
        _doom = st.session_state.get("_doom_briefing", "")
        _plays = st.session_state.get("_rp_plays_result") or {}

        _synopsis = st.session_state.get("_macro_synopsis") or {}
        if _synopsis.get("conviction"):
            _conv = _synopsis["conviction"]
            _conv_color = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "MIXED": "#f59e0b", "UNCERTAIN": "#94a3b8"}.get(_conv, "#94a3b8")
            _conv_bg   = {"BULLISH": "#052e16", "BEARISH": "#1a0000", "MIXED": "#1a1200", "UNCERTAIN": "#0d1117"}.get(_conv, "#0d1117")
            _kp_html = "".join(
                f'<div style="color:#94a3b8;font-size:11px;padding:2px 0;"> · {kp}</div>'
                for kp in _synopsis.get("key_points", [])
            )
            _ct_html = ""
            for ct in _synopsis.get("contradictions", []):
                _ct_html += (
                    f'<div style="color:#f59e0b;font-size:10px;padding:2px 0;">'
                    f'⚠ {ct}</div>'
                )
            _syn_eng = st.session_state.get("_macro_synopsis_engine", "")
            _eng_span = f'<span style="font-size:10px;color:#555;margin-left:auto;">{_syn_eng}</span>' if _syn_eng else ""
            st.markdown(
                f'<div style="background:{_conv_bg};border:2px solid {_conv_color};border-radius:8px;'
                f'padding:14px 18px;margin-top:14px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="background:{_conv_color};color:black;font-weight:800;font-size:12px;'
                f'padding:3px 12px;border-radius:4px;letter-spacing:0.08em;">{_conv}</span>'
                f'<span style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.08em;">MACRO CONVICTION</span>'
                f'{_eng_span}'
                f'</div>'
                f'<div style="color:#e2e8f0;font-size:12px;line-height:1.6;margin-bottom:6px;">{_synopsis.get("summary","")}</div>'
                f'{_kp_html}'
                f'{_ct_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Tactical Regime Preview ────────────────────────────────────────────
        _tac_ctx = st.session_state.get("_tactical_context", {})
        _tac_ai_prev = st.session_state.get("_tactical_analysis", "")
        if _tac_ctx:
            _ts_val  = _tac_ctx.get("tactical_score", 50)
            _tlabel  = _tac_ctx.get("label", "")
            _tbias   = _tac_ctx.get("action_bias", "")
            _tac_color = "#22c55e" if _ts_val >= 65 else ("#f59e0b" if _ts_val >= 38 else "#ef4444")
            _tac_bg    = "#0c1a0c" if _ts_val >= 65 else ("#1a1200" if _ts_val >= 38 else "#1a0000")
            with st.expander(f"⚡ Tactical Regime — {_tlabel} ({_ts_val}/100)", expanded=True):
                # Score + label badge
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                    f'<span style="background:{_tac_color};color:black;font-weight:800;font-size:11px;'
                    f'padding:3px 10px;border-radius:4px;letter-spacing:0.06em;">{_tlabel.upper()}</span>'
                    f'<span style="color:{_tac_color};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;">{_ts_val}/100</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:11px;">{_tbias}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Signal breakdown
                _tac_sigs_prev = _tac_ctx.get("signals", [])
                if _tac_sigs_prev:
                    _sigs_html = ""
                    for _sr in _tac_sigs_prev:
                        _sc = "#22c55e" if _sr["Score"] > 0.2 else ("#ef4444" if _sr["Score"] < -0.2 else "#94a3b8")
                        _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
                        _sigs_html += (
                            f'<div style="color:{_sc};font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;padding:1px 0;">{_arrow} {_sr["Signal"]}: '
                            f'<span style="color:{COLORS["text"]}">{_sr["Value"]}</span>'
                            f'<span style="color:#475569;"> ({_sr["Direction"]})</span></div>'
                        )
                    st.markdown(f'<div style="margin-bottom:8px;">{_sigs_html}</div>', unsafe_allow_html=True)
                # AI narrative
                if _tac_ai_prev:
                    st.markdown(
                        f'<div style="background:{_tac_bg};border-left:3px solid {_tac_color};'
                        f'padding:10px 14px;font-size:12px;color:{COLORS["text"]};'
                        f'line-height:1.8;white-space:pre-line;">{_tac_ai_prev}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Options Flow Preview ───────────────────────────────────────────────
        _of_ctx = st.session_state.get("_options_flow_context") or {}
        if _of_ctx:
            _os_val   = _of_ctx.get("options_score", 50)
            _of_label = _of_ctx.get("label", "")
            _of_bias  = _of_ctx.get("action_bias", "")
            _of_color = "#22c55e" if _os_val >= 65 else ("#f59e0b" if _os_val >= 38 else "#ef4444")
            _of_bg    = "#0c1a0c" if _os_val >= 65 else ("#1a1200" if _os_val >= 38 else "#1a0000")
            with st.expander(f"📊 Options Flow — {_of_label} ({_os_val}/100)", expanded=True):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                    f'<span style="background:{_of_color};color:black;font-weight:800;font-size:11px;'
                    f'padding:3px 10px;border-radius:4px;letter-spacing:0.06em;">{_of_label.upper()}</span>'
                    f'<span style="color:{_of_color};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;">{_os_val}/100</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:11px;">{_of_bias}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _of_sigs_prev = _of_ctx.get("signals", [])
                if _of_sigs_prev:
                    _of_sigs_html = ""
                    for _sr in _of_sigs_prev:
                        _sc = "#22c55e" if _sr["Score"] > 0.2 else ("#ef4444" if _sr["Score"] < -0.2 else "#94a3b8")
                        _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
                        _of_sigs_html += (
                            f'<div style="color:{_sc};font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;padding:1px 0;">{_arrow} {_sr["Signal"]}: '
                            f'<span style="color:{COLORS["text"]}">{_sr["Value"]}</span>'
                            f'<span style="color:#475569;"> ({_sr["Direction"]})</span></div>'
                        )
                    st.markdown(f'<div style="margin-bottom:8px;">{_of_sigs_html}</div>', unsafe_allow_html=True)

        if _digest or _doom or _plays or st.session_state.get("_chain_narration") or st.session_state.get("_whale_summary"):
            st.markdown(
                f'<div style="border-top:1px solid {COLORS["border"]};margin:14px 0 10px 0;"></div>',
                unsafe_allow_html=True,
            )

        _dp = st.session_state.get("_dominant_rate_path") or {}
        if _dp:
            _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
            _dp_label = _dp_labels.get(_dp.get("scenario", ""), _dp.get("scenario", ""))
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid {COLORS["border"]};border-radius:4px;'
                f'padding:8px 12px;font-size:11px;color:{COLORS["text_dim"]};margin-bottom:6px;">'
                f'📈 <b style="color:{COLORS["bloomberg_orange"]}">Fed Rate Path</b> — '
                f'Dominant: <b style="color:{COLORS["text"]}">{_dp_label}</b> '
                f'({_dp.get("prob_pct", 0):.0f}% probability)</div>',
                unsafe_allow_html=True,
            )

        if _plays:
            _regime = st.session_state.get("_regime_context", {})
            _regime_label = _regime.get("regime", "")
            _quad = _regime.get("quadrant", "")
            with st.expander(f"📡 Regime: {_regime_label} · {_quad}", expanded=True):
                _sectors = _plays.get("sectors", [])
                _stocks = _plays.get("stocks", [])
                if _sectors:
                    st.markdown("**Sectors:** " + " · ".join(
                        f"{s.get('name','')} ({'★'*s.get('conviction',1)})" for s in _sectors[:4]
                    ))
                if _stocks:
                    st.markdown("**Stocks:** " + " · ".join(
                        f"{s.get('ticker','')} ({'★'*s.get('conviction',1)})" for s in _stocks[:4]
                    ))

        if _digest:
            with st.expander("🗞 News Digest", expanded=True):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_digest}</div>',
                    unsafe_allow_html=True,
                )

        if _doom:
            with st.expander("💀 Doom Briefing", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_doom}</div>',
                    unsafe_allow_html=True,
                )

        _chain = st.session_state.get("_chain_narration", "")
        if _chain:
            with st.expander("🔗 Policy Transmission", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_chain}</div>',
                    unsafe_allow_html=True,
                )

        _whale = st.session_state.get("_whale_summary", "")
        if _whale:
            with st.expander("🐋 Whale Activity", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_whale}</div>',
                    unsafe_allow_html=True,
                )

        _bs = st.session_state.get("_custom_swans", {})
        if _bs:
            _bs_names = " · ".join(list(_bs.keys())[:3])
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #4B5EAA44;border-radius:4px;'
                f'padding:8px 12px;font-size:11px;color:{COLORS["text_dim"]};margin-top:6px;">'
                f'🦢 <b style="color:#8899CC">Black Swans</b> — '
                f'{len(_bs)} scenarios: <span style="color:{COLORS["text"]}">{_bs_names}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Data Flow Legend ───────────────────────────────────────────────────────
    with st.expander("📊 Data Flow", expanded=False):
        _oc = COLORS["bloomberg_orange"]
        st.markdown(
            f"""
<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;line-height:1.9;">

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">DATA SOURCES</div>
<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">
  {''.join(f'<span style="background:#1e293b;border:1px solid #334155;border-radius:3px;padding:2px 8px;color:#94a3b8;">{s}</span>' for s in ['yfinance','FRED API','RSS feeds','Polymarket Gist','📱 Telegram inbox','SEC EDGAR'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">SIGNAL LAYER <span style="font-weight:400;color:#475569;">(Quick Intel Run generates these)</span></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Regime + Quadrant','Tactical Regime (days-to-weeks)','Rate Path Plays','Fed Funds Rate','Current Events Digest','Doom Briefing','Whale Activity','Black Swans','Policy Transmission','Trending Narratives','Auto-Trending Groups','Portfolio Risk Snapshot'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">CONSUMERS</div>
<div style="display:flex;flex-direction:column;gap:4px;">
  <div style="background:#0c1a0c;border:1px solid #22c55e44;border-radius:4px;padding:6px 10px;">
    <span style="color:#22c55e;font-weight:700;">Portfolio Intelligence</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">uses ALL signals — regime · rate path · news digest · doom · whales · swans · current events · trending narratives · auto-trending groups · <b style="color:#22c55e88;">portfolio risk snapshot</b></span>
  </div>
  <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;padding:6px 10px;">
    <span style="color:#f59e0b;font-weight:700;">Discovery</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · rate path · fed plays · regime plays · doom · whales · swans · current events · trending narratives · auto-trending groups (feeds Cross-Signal Plays)</span>
  </div>
  <div style="background:#0d1117;border:1px solid #3b82f644;border-radius:4px;padding:6px 10px;">
    <span style="color:#3b82f6;font-weight:700;">Valuation</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · rate path · fed plays · regime plays · doom · whales · swans · current events · trending narratives · auto-trending groups · DCF + Elliott Wave + Wyckoff · <b style="color:#3b82f688;">portfolio risk snapshot</b></span>
  </div>
</div>

<div style="margin-top:12px;padding-top:8px;border-top:1px solid #1e293b;color:#334155;font-size:10px;">
  📱 Telegram inbox → Current Events Digest → Portfolio Intelligence + Doom Briefing
</div>

</div>""",
            unsafe_allow_html=True,
        )
