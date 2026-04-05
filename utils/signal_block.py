"""Centralized signal block builder.

Every prompt in the chain should call build_macro_block() and/or build_ticker_block()
and inject the result directly — this ensures raw numbers from Python math reach the AI
instead of relying on prior AI outputs cascading through the chain.

Also provides get_signal_fingerprint() / get_ticker_fingerprint() for verdict caching:
if the underlying numeric signals haven't changed, we can return the cached AI verdict
instead of burning tokens on an identical call.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date


def get_signal_fingerprint() -> str:
    """Hash current macro signal values into a short key.

    Returns a 12-char hex string. Identical fingerprint → signals haven't changed
    enough to warrant a new AI call. Rounds floats to 1 decimal to absorb noise.

    If regime signals haven't been populated yet (e.g. FRED is down), returns a
    sentinel "NO_REGIME_DATA" fingerprint so all callers treat this as a cache miss
    rather than colliding all tickers under the same empty-dict hash.
    """
    import streamlit as st
    raw = st.session_state.get("_regime_raw_signals") or {}
    tac = st.session_state.get("_tactical_context") or {}

    # Guard: if regime engine never ran, do NOT hash an empty dict — that would
    # make every ticker map to the same fingerprint, causing cache collisions.
    if not raw:
        return "NO_REGIME_DATA"

    key_vals: dict = {}
    for k, v in raw.items():
        key_vals[k] = round(v, 1) if isinstance(v, float) else v
    key_vals["tac_score"] = tac.get("tactical_score")
    key_vals["tac_label"] = tac.get("label")
    return hashlib.md5(json.dumps(key_vals, sort_keys=True).encode()).hexdigest()[:12]


def get_ticker_fingerprint(ticker: str) -> str:
    """Hash ticker-specific signal values for per-ticker verdict caching.

    Incorporates the macro fingerprint so the ticker cache also busts on regime change.
    """
    import streamlit as st
    vals: dict = {"ticker": ticker.upper(), "macro_fp": get_signal_fingerprint()}

    opts = st.session_state.get("_options_sentiment") or {}
    if opts.get("ticker", "").upper() == ticker.upper():
        vals["pc"] = round(opts.get("pc_ratio", 0), 1)
        vals["opts_sent"] = opts.get("sentiment")

    ins = st.session_state.get("_insider_net_flow") or {}
    if ins.get("ticker", "").upper() == ticker.upper():
        vals["insider"] = ins.get("bias")

    inst = st.session_state.get("_institutional_bias") or {}
    if inst.get("ticker", "").upper() == ticker.upper():
        vals["inst"] = inst.get("bias")

    return hashlib.md5(json.dumps(vals, sort_keys=True).encode()).hexdigest()[:12]


def build_macro_block() -> str:
    """Assemble a compact, structured raw-number macro context string.

    Reads from session_state (populated by risk_regime.py and fed_forecaster.py).
    Returns a plain-text block suitable for injection at the TOP of any prompt.

    This is the 'grounding header' that prevents the telephone game — every prompt
    gets the same ground truth regardless of what prior AI outputs said.
    """
    import streamlit as st

    lines: list[str] = [
        f"=== MACRO GROUND TRUTH ({date.today().isoformat()}) ===",
    ]

    # ── Risk Regime ───────────────────────────────────────────────────────────
    rc = st.session_state.get("_regime_context") or {}
    if rc:
        lines.append(
            f"Regime: {rc.get('regime','?')} | Quadrant: {rc.get('quadrant','?')} | "
            f"Score: {rc.get('score', 0):+.2f} (-1=risk-off, +1=risk-on)"
        )
        # Leading sub-score: fast-reacting signals only (VIX, credit spreads, LEI, etc.)
        # Divergence from composite = early warning of regime change before lagging data confirms.
        _ll = rc.get("leading_label", "Aligned")
        _ld = rc.get("leading_divergence", 0) or 0
        _ls = rc.get("leading_score")
        _5d = rc.get("score_5d_trend")
        if _ls is not None:
            _early_warn = (
                f" ⚠ EARLY WARNING: {_ll} — leading signals are {abs(_ld)} pts {'ahead of' if _ld > 0 else 'below'} composite"
                if _ll != "Aligned" else ""
            )
            lines.append(
                f"Leading sub-score (fast signals only): {_ls}/100 (divergence from composite: {_ld:+d} pts){_early_warn}"
            )
        _l5d = rc.get("leading_5d_trend")
        if _5d is not None or _l5d is not None:
            _c_str = f"{_5d:+d} pts" if _5d is not None else "—"
            _l_str = f"{_l5d:+d} pts" if _l5d is not None else "—"
            _conf = ""
            if _5d is not None and _l5d is not None:
                if _5d > 0 and _l5d > 0:
                    _conf = " — BOTH RISING (high conviction)"
                elif _5d < 0 and _l5d < 0:
                    _conf = " — BOTH FALLING (high conviction)"
                elif (_5d > 0) != (_l5d > 0):
                    _conf = " — DIVERGING (composite and leading disagree)"
            lines.append(f"5-session trend — composite: {_c_str} | leading: {_l_str}{_conf}")

    # ── Raw macro z-scores from regime engine ─────────────────────────────────
    raw = st.session_state.get("_regime_raw_signals") or {}
    _skip = {"macro_score_norm", "macro_regime", "quadrant", "fear_greed",
             "fear_greed_label", "tactical_score",
             "leading_label", "leading_score", "leading_divergence", "score_5d_trend"}
    z_lines = [
        f"  {k.replace('_', ' ').title()}: {v:+.3f}"
        for k, v in raw.items()
        if k not in _skip and isinstance(v, (int, float)) and v is not None
    ]
    if z_lines:
        lines.append("Raw macro signal z-scores (from regime engine):")
        lines.extend(z_lines)

    # ── Fed rate path ─────────────────────────────────────────────────────────
    ff = st.session_state.get("_fed_funds_rate")
    if ff is not None:
        lines.append(f"Fed Funds Rate: {ff:.2f}%")

    dp = st.session_state.get("_dominant_rate_path") or {}
    if dp:
        lines.append(
            f"Dominant rate path: {dp.get('scenario','?').replace('_',' ')} "
            f"({dp.get('prob_pct', 0):.1f}% probability)"
        )

    rate_probs = st.session_state.get("_rate_path_probs") or []
    if rate_probs:
        prob_parts = []
        for r in sorted(rate_probs, key=lambda x: x.get("prob", 0), reverse=True):
            prob_parts.append(
                f"{r.get('scenario','?').replace('_',' ')} {r.get('prob',0)*100:.1f}%"
            )
        lines.append("Rate path probs: " + " | ".join(prob_parts))

    # ── Tactical regime ────────────────────────────────────────────────────────
    tac = st.session_state.get("_tactical_context") or {}
    if tac:
        lines.append(
            f"Tactical regime: {tac.get('label','?')} | score {tac.get('tactical_score','?')}/100"
        )

    # ── VIX curve ─────────────────────────────────────────────────────────────
    vc = st.session_state.get("_vix_curve") or {}
    if vc:
        lines.append(
            f"VIX term structure: {vc.get('structure','?')} | "
            f"9D={vc.get('vix9d','?')} / VIX={vc.get('vix','?')} / "
            f"3M={vc.get('vix3m','?')} / 6M={vc.get('vix6m','?')}"
        )

    # ── Crowd sentiment ────────────────────────────────────────────────────────
    fg = st.session_state.get("_fear_greed") or {}
    if fg:
        lines.append(f"Fear & Greed: {fg.get('score','?')}/100 ({fg.get('label','')})")

    aaii = st.session_state.get("_aaii_sentiment") or {}
    if aaii:
        lines.append(
            f"AAII: Bull {aaii.get('bull_pct','?')}% / Bear {aaii.get('bear_pct','?')}% "
            f"(spread {aaii.get('bull_bear_spread', 0):+}%)"
        )

    # ── Macro calendar — upcoming catalysts ───────────────────────────────────
    _cal_import_ok = False
    try:
        from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
        _cal_import_ok = True
    except Exception:
        pass

    if _cal_import_ok:
        _cal_lines = []
        for _label, _fn in (("FOMC", get_next_fomc), ("CPI", get_next_cpi), ("NFP", get_next_nfp)):
            try:
                _ev = _fn()
                if _ev:
                    _d = _ev.get("days_away", "?")
                    _dt_str = _ev.get("date", "?")
                    _urgency = " ⚠ IMMINENT" if isinstance(_d, int) and _d <= 5 else (
                               " 📅 THIS WEEK" if isinstance(_d, int) and _d <= 7 else "")
                    _cal_lines.append(f"  {_label}: {_dt_str} ({_d}d){_urgency}")
            except Exception as _ce:
                _cal_lines.append(f"  {_label}: unavailable ({type(_ce).__name__})")
        if _cal_lines:
            lines.append("Upcoming macro events:")
            lines.extend(_cal_lines)
    else:
        lines.append("Upcoming macro events: unavailable (fed_forecaster import failed)")

    # ── QIR macro debate verdict (Judge Judy ruling) ──────────────────────────
    # Downstream debates (valuation, discovery) inherit this ruling so they don't
    # re-litigate macro — they argue their narrower question on top of it.
    # CONFLICT DETECTION: if the math direction contradicts Judge Judy's ruling,
    # the math wins — the verdict is flagged as advisory only, not settled.
    _dbt = st.session_state.get("_adversarial_debate") or {}
    if _dbt.get("verdict"):
        _dbt_verdict    = _dbt.get("verdict", "CONTESTED")
        _dbt_conf       = _dbt.get("confidence", 5)
        _dbt_bull_best  = _dbt.get("bull_strongest", "")
        _dbt_bear_best  = _dbt.get("bear_strongest", "")
        _dbt_asym       = _dbt.get("asymmetry", "")

        # Derive math direction from composite score
        _math_score = rc.get("macro_score", 50) if rc else 50
        _lead_div   = rc.get("leading_divergence", 0) or 0 if rc else 0
        if _math_score > 55 or (_math_score > 50 and _lead_div > 5):
            _math_dir = "BULL"
        elif _math_score < 45 or (_math_score < 50 and _lead_div < -5):
            _math_dir = "BEAR"
        else:
            _math_dir = "NEUTRAL"

        _judy_dir = (
            "BULL" if "BULL" in _dbt_verdict.upper()
            else "BEAR" if "BEAR" in _dbt_verdict.upper()
            else "CONTESTED"
        )

        _conflict = (
            _math_dir != "NEUTRAL"
            and _judy_dir != "CONTESTED"
            and _math_dir != _judy_dir
        )

        lines.append(
            f"Judge Judy macro verdict (QIR debate): {_dbt_verdict} "
            f"(confidence {_dbt_conf}/10)"
        )
        if _dbt_bull_best:
            lines.append(f"  Strongest bull argument: {_dbt_bull_best}")
        if _dbt_bear_best:
            lines.append(f"  Strongest bear argument: {_dbt_bear_best}")
        if _dbt_asym:
            lines.append(f"  Asymmetry: {_dbt_asym}")

        if _conflict:
            lines.append(
                f"  ⚠ MATH vs AI CONFLICT: Composite score ({_math_score}/100) and leading "
                f"divergence ({_lead_div:+d}) signal {_math_dir} but Judge Judy ruled {_judy_dir}. "
                f"THE MATH IS OBJECTIVE — treat debate verdict as advisory only. "
                f"Do not treat as settled. Weight the numeric signals over the AI interpretation."
            )
        else:
            lines.append(
                "  Math and debate verdict are aligned — macro direction SETTLED. "
                "Build your case on top of this, do not re-argue macro."
            )

    # ── Quantified signal scores ──────────────────────────────────────────────
    _fc = st.session_state.get("_fear_composite") or {}
    if _fc:
        lines.append(
            f"Fear Composite Index: {_fc.get('score','?')}/100 — {_fc.get('label','?')} "
            f"(Stress {_fc.get('components',{}).get('Stress','?')} · "
            f"Macro {_fc.get('components',{}).get('Macro','?')} · "
            f"Canary {_fc.get('components',{}).get('Canary','?')} · "
            f"Whale {_fc.get('components',{}).get('Whale','?')} · "
            f"Events {_fc.get('components',{}).get('Events','?')})"
        )
    _sq_stress = st.session_state.get("_stress_zscore") or {}
    if _sq_stress:
        lines.append(f"Stress z-score: {_sq_stress.get('z','?'):+} ({_sq_stress.get('pct','?')}th pct vs 1yr history)")
    _sq_whale = st.session_state.get("_whale_flow_score") or {}
    if _sq_whale:
        lines.append(
            f"Whale 13F flow: {_sq_whale.get('bull_pct','?')}% bull · "
            f"net {_sq_whale.get('net_flow_bn','?'):+.1f}B · "
            f"rotation {_sq_whale.get('rotation','?'):+.2f} · "
            f"{_sq_whale.get('label','?')} (NOTE: 13F is 45-day lagged — structural bias, not timing)"
        )
    _sq_events = st.session_state.get("_events_sentiment_score") or {}
    if _sq_events:
        _src = _sq_events.get("source", "keyword")
        lines.append(
            f"Events sentiment [{_src}]: {_sq_events.get('sentiment','?'):+.2f} · "
            f"uncertainty {_sq_events.get('uncertainty','?'):.2f} · "
            f"{_sq_events.get('label','?')}"
            + (f" · theme: {_sq_events['dominant_theme']}" if _sq_events.get("dominant_theme") else "")
        )
    _sq_canary = st.session_state.get("_canary_score") or {}
    if _sq_canary:
        lines.append(
            f"Canary breadth: {_sq_canary.get('composite','?')}/100 · "
            f"breadth {_sq_canary.get('breadth_pct','?')}% · "
            f"1m avg {_sq_canary.get('momentum_avg','?'):+.2f}% · "
            f"vol surge {_sq_canary.get('vol_surge','?')}x"
        )

    lines.append("=== END MACRO GROUND TRUTH ===")
    return "\n".join(lines)


def build_ticker_block(ticker: str) -> str:
    """Assemble raw numeric signals for a specific ticker.

    Reads from session_state (populated by options, insider, valuation modules)
    and fetches fundamentals from market_data if not already cached.
    Returns a plain-text block for prompt injection.
    """
    import streamlit as st
    from services.market_data import fetch_ticker_fundamentals

    lines: list[str] = [
        f"=== {ticker.upper()} RAW SIGNALS ({date.today().isoformat()}) ===",
    ]

    # ── Price momentum ─────────────────────────────────────────────────────────
    pm = st.session_state.get("_price_momentum") or {}
    if pm.get("ticker", "").upper() == ticker.upper():
        rsi = pm.get("rsi")
        if rsi is not None:
            lines.append(f"RSI(14): {rsi:.1f}")
        ma = pm.get("ma_signals") or {}
        for k, v in ma.items():
            lines.append(
                f"{k.upper()}: ${v.get('value',0):.2f} ({'above' if v.get('above') else 'below'})"
            )
        lines.append(f"Volume vs avg: {pm.get('vol_ratio', 1.0):.2f}x")

    # ── Options ────────────────────────────────────────────────────────────────
    opts = st.session_state.get("_options_sentiment") or {}
    if opts.get("ticker", "").upper() == ticker.upper():
        lines.append(
            f"Options: P/C ratio={opts.get('pc_ratio',0):.2f} | "
            f"sentiment={opts.get('sentiment','?')} | "
            f"call vol={opts.get('call_vol',0):,} / put vol={opts.get('put_vol',0):,}"
        )

    # ── GEX / Dealer Gamma ────────────────────────────────────────────────────
    gex = st.session_state.get("_gex_profile") or {}
    if gex.get("ticker", "").upper() == ticker.upper() and gex.get("gamma_flip"):
        spot = gex.get("spot", 0)
        flip = gex.get("gamma_flip", 0)
        flip_dist = ((flip - spot) / spot * 100) if spot > 0 else 0
        lines.append(
            f"GEX: {gex.get('zone','?')} | Gamma Flip ${flip:.2f} ({flip_dist:+.1f}% from spot) | "
            f"Call Wall ${gex.get('call_wall',0):.2f} | Put Wall ${gex.get('put_wall',0):.2f} | "
            f"Total GEX ${gex.get('total_gex',0):+.1f}M | Dealer Delta {gex.get('dealer_net_delta',0):+.3f}"
        )

    # ── Insider flows ──────────────────────────────────────────────────────────
    ins = st.session_state.get("_insider_net_flow") or {}
    if ins.get("ticker", "").upper() == ticker.upper():
        lines.append(
            f"Insider flows: {ins.get('bias','?')} | "
            f"buy_pct={ins.get('buy_pct',50):.0f}% | "
            f"net_buy=${ins.get('buy_value',0)-ins.get('sell_value',0):,.0f} | "
            f"n_trades={ins.get('n_trades',0)}"
        )

    # ── Congress ───────────────────────────────────────────────────────────────
    cong = st.session_state.get("_congress_bias") or {}
    if cong.get("ticker", "").upper() == ticker.upper():
        lines.append(
            f"Congress trading: {cong.get('bias','?')} | "
            f"buy_pct={cong.get('buy_pct',50):.0f}%"
        )

    # ── Institutional ──────────────────────────────────────────────────────────
    inst = st.session_state.get("_institutional_bias") or {}
    if inst.get("ticker", "").upper() == ticker.upper():
        lines.append(
            f"Institutional: {inst.get('bias','?')} | "
            f"weighted_chg={inst.get('weighted_pct',0):+.1f}%"
        )

    # ── Fundamentals (yfinance) ────────────────────────────────────────────────
    try:
        fund = fetch_ticker_fundamentals(ticker)
        if fund:
            f_lines = []
            if fund.get("pe_trailing") is not None:
                f_lines.append(f"P/E (trailing): {fund['pe_trailing']:.1f}x")
            if fund.get("pe_forward") is not None:
                f_lines.append(f"P/E (forward): {fund['pe_forward']:.1f}x")
            if fund.get("peg") is not None:
                f_lines.append(f"PEG: {fund['peg']:.2f}")
            if fund.get("ps_ratio") is not None:
                f_lines.append(f"P/S: {fund['ps_ratio']:.1f}x")
            if fund.get("pb_ratio") is not None:
                f_lines.append(f"P/B: {fund['pb_ratio']:.1f}x")
            if fund.get("ev_ebitda") is not None:
                f_lines.append(f"EV/EBITDA: {fund['ev_ebitda']:.1f}x")
            if fund.get("div_yield") is not None:
                f_lines.append(f"Div yield: {fund['div_yield']*100:.2f}%")
            if fund.get("roe") is not None:
                f_lines.append(f"ROE: {fund['roe']*100:.1f}%")
            if fund.get("debt_to_equity") is not None:
                f_lines.append(f"Debt/Equity: {fund['debt_to_equity']:.2f}x")
            if fund.get("current_ratio") is not None:
                f_lines.append(f"Current ratio: {fund['current_ratio']:.2f}")
            if fund.get("fcf_yield") is not None:
                f_lines.append(f"FCF yield: {fund['fcf_yield']*100:.2f}%")
            if fund.get("revenue_growth_yoy") is not None:
                f_lines.append(f"Revenue growth YoY: {fund['revenue_growth_yoy']*100:+.1f}%")
            if fund.get("earnings_growth_yoy") is not None:
                f_lines.append(f"Earnings growth YoY: {fund['earnings_growth_yoy']*100:+.1f}%")
            if fund.get("short_pct_float") is not None:
                f_lines.append(f"Short % of float: {fund['short_pct_float']*100:.1f}%")
            if fund.get("short_ratio") is not None:
                f_lines.append(f"Short ratio (days to cover): {fund['short_ratio']:.1f}")
            if fund.get("analyst_score") is not None:
                f_lines.append(
                    f"Analyst score: {fund['analyst_score']:.1f}/5 "
                    f"({fund.get('analyst_count',0)} analysts) | "
                    f"target ${fund.get('target_mean',0):.2f}"
                )
            if fund.get("revision_score") is not None:
                sign = "+" if fund["revision_score"] >= 0 else ""
                f_lines.append(f"Analyst revisions (30d): {sign}{fund['revision_score']:+.0f} net upgrades")
            if f_lines:
                lines.append("Fundamentals:")
                lines.extend(f"  {l}" for l in f_lines)
    except Exception:
        pass

    # ── Earnings catalyst ─────────────────────────────────────────────────────
    try:
        from services.market_data import fetch_earnings_date as _fed
        _earn = _fed(ticker)
        if _earn:
            _d = _earn.get("days_away", "?")
            _urgency = " ⚠ EARNINGS IMMINENT — elevated volatility risk" if isinstance(_d, int) and _d <= 7 else (
                       " 📅 earnings approaching — factor into targets" if isinstance(_d, int) and _d <= 21 else "")
            lines.append(f"Next earnings: {_earn.get('date','?')} ({_d}d){_urgency}")
    except Exception:
        pass

    # ── Credit / Fixed Income ─────────────────────────────────────────────────
    try:
        from services.market_data import fetch_credit_metrics as _fcm
        _cm = _fcm(ticker)
        if _cm:
            _cr_lines = []
            if _cm.get("interest_coverage") is not None:
                _cov = _cm["interest_coverage"]
                _cov_flag = ""
                if _cov < 1.5:
                    _cov_flag = " ⚠ CRITICAL"
                elif _cov < 3.0:
                    _cov_flag = " ⚠ WARNING"
                _cr_lines.append(f"Interest coverage: {_cov:.1f}x{_cov_flag}")
            if _cm.get("debt_to_ebitda") is not None:
                _lev = _cm["debt_to_ebitda"]
                _lev_flag = " ⚠ HIGH" if _lev > 4 else (" 🔴 DANGEROUS" if _lev > 6 else "")
                _cr_lines.append(f"Debt/EBITDA: {_lev:.1f}x{_lev_flag}")
            if _cm.get("net_debt_B") is not None:
                _cr_lines.append(f"Net debt: ${_cm['net_debt_B']:.1f}B")
            if _cm.get("current_debt_ratio") is not None:
                _pct = _cm["current_debt_ratio"] * 100
                _mflag = " ⚠ REFINANCING RISK" if _pct > 20 else ""
                _cr_lines.append(f"Near-term debt maturity: {_pct:.0f}% of total{_mflag}")
            if _cm.get("fcf_debt_coverage") is not None:
                _cr_lines.append(f"FCF/Debt coverage: {_cm['fcf_debt_coverage']:.2f}x")
            if _cr_lines:
                lines.append("Credit / Fixed Income:")
                lines.extend(f"  {l}" for l in _cr_lines)
    except Exception:
        pass

    lines.append(f"=== END {ticker.upper()} SIGNALS ===")
    return "\n".join(lines)
