"""Export Hub — Download all macro intelligence as Markdown or JSON for AI chat."""

import json
from datetime import datetime

import streamlit as st

from utils.journal import load_journal
from utils.theme import COLORS
from services.signals_cache import _serialize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_str(ts) -> str:
    """Return human-readable age string from a datetime or None."""
    if ts is None:
        return ""
    now = datetime.now()
    delta = now - ts
    mins = int(delta.total_seconds() / 60)
    if mins < 2:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        rem = mins % 60
        return f"{hours}h {rem}m ago" if rem else f"{hours}h ago"
    return f"{delta.days}d ago"


def _sectors_str(sectors) -> str:
    """Flatten sectors list (dicts or strings) to a comma-separated string."""
    if not sectors:
        return ""
    parts = []
    for s in sectors:
        parts.append(s.get("name", "") if isinstance(s, dict) else str(s))
    return ", ".join(p for p in parts if p)


def _stocks_str(stocks) -> str:
    """Flatten stocks list (dicts or strings) to a comma-separated string."""
    if not stocks:
        return ""
    parts = []
    for s in stocks:
        if isinstance(s, dict):
            tk = s.get("ticker", "")
            name = s.get("name", "")
            parts.append(f"{tk} ({name})" if name and name != tk else tk)
        else:
            parts.append(str(s))
    return ", ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Markdown section builders (each returns "" when data is missing)
# ---------------------------------------------------------------------------

def _section_executive_summary(open_trades: list) -> str:
    ctx = st.session_state.get("_regime_context") or {}
    if not ctx:
        return ""
    regime = ctx.get("regime", "—")
    score = ctx.get("score", 0)
    quadrant = ctx.get("quadrant", "—")
    dominant = st.session_state.get("_dominant_rate_path") or {}
    rate_path_line = ""
    if dominant:
        if isinstance(dominant, dict):
            rate_path_line = f"\n- **Rate Path:** {dominant.get('scenario','—').replace('_',' ').title()} ({dominant.get('prob_pct',0):.0f}% market-implied)"
        else:
            rate_path_line = f"\n- **Rate Path:** {dominant}"
    pa = st.session_state.get("_portfolio_analysis") or {}
    risk_score_line = f"\n- **Portfolio Risk Score:** {pa['risk_score']}/10" if pa.get("risk_score") else ""
    actions = pa.get("priority_actions", [])
    actions_block = ""
    if actions:
        action_items = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions[:3]))
        actions_block = f"\n- **Top Actions:**\n{action_items}"
    pos_line = f"\n- **Open Positions:** {len(open_trades)}" if open_trades else ""
    return (
        "## EXECUTIVE SUMMARY\n"
        f"- **Regime:** {regime} (score {score:+.2f}) | **Quadrant:** {quadrant}"
        f"{rate_path_line}"
        f"{risk_score_line}"
        f"{pos_line}"
        f"{actions_block}\n"
    )


def _section_current_events() -> str:
    digest = st.session_state.get("_current_events_digest", "")
    engine = st.session_state.get("_current_events_engine", "")
    ts = st.session_state.get("_current_events_digest_ts")
    age = ""
    if ts:
        from datetime import datetime as _dt2
        mins = int((_dt2.now() - ts).total_seconds() / 60)
        age = f"{mins}m ago"
    engine_tag = f" *(via {engine})*" if engine else ""
    age_tag = f" · {age}" if age else ""

    from services.news_feed import load_news_inbox
    inbox = load_news_inbox()

    if not digest and not inbox:
        return ""
    lines = [f"## CURRENT EVENTS{engine_tag}{age_tag}", ""]
    if digest:
        lines.append(f"**News Digest:** {digest}")
    if inbox:
        lines.append("\n### Inbox Items")
        for item in inbox[-5:]:
            ts_str = item.get("ts", "")[:16].replace("T", " ")
            lines.append(f"- [{ts_str}] {item.get('text','')}")
    return "\n".join(lines) + "\n"


def _section_regime() -> str:
    ctx = st.session_state.get("_regime_context")
    if not ctx:
        return ""
    regime = ctx.get("regime", "—")
    score = ctx.get("score", 0)
    quadrant = ctx.get("quadrant", "")
    sig = ctx.get("signal_summary", "")
    fed_rate = st.session_state.get("_fed_funds_rate")
    rate_line = f"\n- **Fed Funds Rate:** {fed_rate:.2f}%" if fed_rate else ""
    quad_line = f" | Quadrant: {quadrant}" if quadrant else ""
    sig_line = f"\n- **Signal Summary:** {sig}" if sig else ""
    return (
        "## MACRO REGIME\n"
        f"- **Regime:** {regime} (score {score:+.2f}){quad_line}"
        f"{rate_line}"
        f"{sig_line}\n"
    )


def _section_rate_path() -> str:
    probs = st.session_state.get("_rate_path_probs")
    dominant = st.session_state.get("_dominant_rate_path")
    if not probs and not dominant:
        return ""
    lines = ["## FED RATE PATH"]
    # _dominant_rate_path is stored as dict {"scenario": str, "prob_pct": float}
    if dominant:
        if isinstance(dominant, dict):
            dom_label = dominant.get("scenario", "—")
            dom_prob_pct = dominant.get("prob_pct", 0)
            lines.append(f"- **Dominant:** {dom_label} ({dom_prob_pct:.0f}% probability)")
        else:
            lines.append(f"- **Dominant:** {dominant}")
    if probs:
        scenarios = " | ".join(
            f"{r.get('scenario', '?')} {int(r.get('prob', 0) * 100)}%"
            for r in probs
        )
        lines.append(f"- **Scenarios:** {scenarios}")
    return "\n".join(lines) + "\n"


def _section_policy_transmission() -> str:
    narration = st.session_state.get("_chain_narration")
    if not narration:
        return ""
    engine = st.session_state.get("_chain_narration_engine", "")
    engine_tag = f" *(via {engine})*" if engine else ""
    return f"## POLICY TRANSMISSION CHAIN{engine_tag}\n{narration}\n"


def _section_regime_plays() -> str:
    plays = st.session_state.get("_rp_plays_result")
    if not plays:
        return ""
    sectors = _sectors_str(plays.get("sectors", []))
    stocks = _stocks_str(plays.get("stocks", []))
    bonds = _stocks_str(plays.get("bonds", []))
    rationale = plays.get("rationale", "")
    lines = ["## AI REGIME PLAYS"]
    if sectors:
        lines.append(f"- **Favored Sectors:** {sectors}")
    if stocks:
        lines.append(f"- **Favored Stocks:** {stocks}")
    if bonds:
        lines.append(f"- **Bonds/Macro:** {bonds}")
    if rationale:
        lines.append(f"\n**Rationale:** {rationale}")
    return "\n".join(lines) + "\n"


def _section_fed_plays() -> str:
    plays = st.session_state.get("_fed_plays_result")
    if not plays:
        return ""
    sectors = _sectors_str(plays.get("sectors", []))
    stocks = _stocks_str(plays.get("stocks", []))
    bonds = _stocks_str(plays.get("bonds", []))
    rationale = plays.get("rationale", "")
    lines = ["## RATE-PATH PLAYS"]
    if sectors:
        lines.append(f"- **Favored Sectors:** {sectors}")
    if stocks:
        lines.append(f"- **Favored Stocks:** {stocks}")
    if bonds:
        lines.append(f"- **Bonds/Macro:** {bonds}")
    if rationale:
        lines.append(f"\n**Rationale:** {rationale}")
    return "\n".join(lines) + "\n"


def _section_black_swans() -> str:
    swans = st.session_state.get("_custom_swans")
    if not swans:
        return ""
    lines = ["## BLACK SWAN TAIL RISKS"]
    items = swans.items() if isinstance(swans, dict) else enumerate(swans)
    for label, data in items:
        if isinstance(data, dict):
            prob = data.get("probability_pct", "?")
            impacts = data.get("asset_impacts", {})
            eq = impacts.get("equities", "")
            bd = impacts.get("bonds", "")
            lines.append(f"\n### {label} ({prob}% probability)")
            if eq:
                lines.append(f"- **Equities:** {eq}")
            if bd:
                lines.append(f"- **Bonds:** {bd}")
            # Any other impact keys
            for k, v in impacts.items():
                if k not in ("equities", "bonds") and v:
                    lines.append(f"- **{k.title()}:** {v}")
    return "\n".join(lines) + "\n"


def _section_doom() -> str:
    briefing = st.session_state.get("_doom_briefing")
    if not briefing:
        return ""
    engine = st.session_state.get("_doom_briefing_engine", "")
    engine_tag = f" *(via {engine})*" if engine else ""
    return f"## RISK INTELLIGENCE BRIEFING (DOOM){engine_tag}\n{briefing}\n"


def _section_whale() -> str:
    summary = st.session_state.get("_whale_summary")
    if not summary:
        return ""
    engine = st.session_state.get("_whale_summary_engine", "")
    engine_tag = f" *(via {engine})*" if engine else ""
    return f"## INSTITUTIONAL WHALE ACTIVITY{engine_tag}\n{summary}\n"


def _section_discovery_plays() -> str:
    plays = st.session_state.get("_plays_result")
    if not plays:
        return ""
    engine = st.session_state.get("_plays_engine", "")
    sectors = _sectors_str(plays.get("sectors", []))
    stocks = _stocks_str(plays.get("stocks", []))
    rationale = plays.get("rationale", "")
    engine_tag = f" *(via {engine})*" if engine else ""
    lines = [f"## CROSS-SIGNAL DISCOVERY PLAYS{engine_tag}"]
    if sectors:
        lines.append(f"- **Sectors:** {sectors}")
    if stocks:
        lines.append(f"- **Stocks:** {stocks}")
    if rationale:
        lines.append(f"\n**Rationale:** {rationale}")
    return "\n".join(lines) + "\n"


def _section_portfolio(open_trades) -> str:
    if not open_trades:
        return ""
    lines = ["## OPEN PORTFOLIO POSITIONS", ""]
    lines.append("| Ticker | Direction | Entry | Qty | Entry Date | Regime at Entry |")
    lines.append("|--------|-----------|-------|-----|------------|-----------------|")
    for t in open_trades:
        lines.append(
            f"| {t['ticker']} | {t['direction'].upper()} | ${t['entry_price']:.2f}"
            f" | {t.get('position_size', '—')}"
            f" | {t.get('entry_date', '—')}"
            f" | {t.get('regime_at_entry', '—')} |"
        )
    return "\n".join(lines) + "\n"


def _section_factor_exposure(open_trades) -> str:
    """Aggregate factor exposure section for the briefing."""
    if not open_trades:
        return ""
    try:
        from services.portfolio_sizing import aggregate_factor_exposure as _agg, score_portfolio as _sp
        from services.market_data import fetch_prices_batch
        _rc = st.session_state.get("_regime_context") or {}
        _live = {}
        try:
            _tks = [t["ticker"] for t in open_trades if t.get("status") == "open"]
            if _tks:
                import yfinance as yf
                _raw = yf.download(_tks, period="2d", interval="1d", progress=False, auto_adjust=True)
                if hasattr(_raw.columns, "levels"):
                    _closes = _raw["Close"].iloc[-1].to_dict()
                    _live = {str(k).upper(): float(v) for k, v in _closes.items() if v == v}
        except Exception:
            pass
        _total_pv = sum(
            (_live.get(t["ticker"].upper(), float(t.get("entry_price") or 0)) * int(t.get("position_size") or 0))
            for t in open_trades if t.get("status") == "open"
        )
        if _total_pv <= 0:
            return ""
        _scored = _sp(open_trades, _rc, _total_pv, _live)
        _fe = _agg(_scored["positions"])
        _fvals = _fe.get("factors", {})
        if not _fvals:
            return ""
        lines = ["## PORTFOLIO FACTOR EXPOSURE", ""]
        lines.append("| Factor | Exposure | Reading |")
        lines.append("|--------|----------|---------|")
        for f in ["growth", "inflation", "liquidity", "credit"]:
            v = _fvals.get(f, 0)
            reading = "Overweight" if v > 0.5 else ("Underweight" if v < -0.5 else "Neutral")
            lines.append(f"| {f.capitalize()} | {v:+.2f}x | {reading} |")
        if _fe.get("warnings"):
            lines.append("")
            for w in _fe["warnings"]:
                lines.append(f"⚠ {w}")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _section_portfolio_analysis(open_trades) -> str:
    pa = st.session_state.get("_portfolio_analysis")
    if not pa:
        return (
            "## PORTFOLIO AI ANALYSIS\n"
            "*(Run Portfolio Analysis in My Regarded Portfolio to populate)*\n"
        )
    engine = st.session_state.get("_portfolio_analysis_engine", "")
    verdict = pa.get("verdict", "—")
    risk_score = pa.get("risk_score", "—")
    narrative = pa.get("narrative", "")
    positions = {p["ticker"].upper(): p for p in pa.get("positions", [])}
    priority = pa.get("priority_actions", [])

    engine_tag = f" *(via {engine})*" if engine else ""
    lines = [f"## PORTFOLIO AI ANALYSIS{engine_tag}", ""]
    lines.append(f"**Verdict:** {verdict} | **Risk Score:** {risk_score}/10")
    if narrative:
        lines.append(f"\n**Narrative:** {narrative}")

    if positions and open_trades:
        lines.append("\n### Per-Position Verdicts")
        for t in open_trades:
            tk = t["ticker"].upper()
            pos = positions.get(tk, {})
            action = pos.get("action", "—")
            rationale = pos.get("rationale", "")
            lines.append(f"- **{tk}:** {action} — {rationale}")

    if priority:
        lines.append("\n### Priority Actions")
        for i, action in enumerate(priority, 1):
            lines.append(f"{i}. {action}")

    return "\n".join(lines) + "\n"


def _section_factor_analysis() -> str:
    fa = st.session_state.get("_factor_analysis")
    if not fa or "_error" in fa:
        return ""
    engine = st.session_state.get("_factor_analysis_engine", "")
    engine_tag = f" *(via {engine})*" if engine else ""
    lines = [f"## FACTOR AI ANALYSIS{engine_tag}", ""]
    if fa.get("headline"):
        lines.append(f"**Headline:** {fa['headline']}")
    verdicts = fa.get("factor_verdicts", [])
    if verdicts:
        lines.append("\n### Factor Verdicts")
        if isinstance(verdicts, list):
            for v in verdicts:
                factor = v.get("factor", "").capitalize()
                verdict = v.get("verdict", "")
                regime_fit = v.get("regime_fit", "")
                comment = v.get("comment", "")
                lines.append(f"- **{factor}:** {verdict} ({regime_fit}) — {comment}")
        else:
            for factor, text in verdicts.items():
                lines.append(f"- **{factor.capitalize()}:** {text}")
    if fa.get("top_risk"):
        lines.append(f"\n**Top Risk:** {fa['top_risk']}")
    suggestions = fa.get("suggestions", [])
    if suggestions:
        lines.append("\n### Suggestions")
        for s in suggestions:
            lines.append(f"- {s}")
    return "\n".join(lines) + "\n"


def _section_sim_verdict() -> str:
    sv = st.session_state.get("_sim_verdict")
    if not sv or "_error" in sv:
        return ""
    engine = st.session_state.get("_sim_verdict_engine", "")
    ticker = st.session_state.get("_sim_ticker", "?")
    amount = st.session_state.get("_sim_amount", 0)
    engine_tag = f" *(via {engine})*" if engine else ""
    lines = [f"## PRE-TRADE SIMULATOR — {ticker}{engine_tag}", ""]
    lines.append(f"**Proposed:** ${amount:,.0f} in {ticker}")
    lines.append(f"**Verdict:** {sv.get('verdict', '—')} | **Thesis:** {sv.get('thesis_check', '—')}")
    if sv.get("verdict_reason"):
        lines.append(f"\n**Reason:** {sv['verdict_reason']}")
    if sv.get("regime_fit_comment"):
        lines.append(f"**Regime Fit:** {sv['regime_fit_comment']}")
    if sv.get("overlap_warning"):
        lines.append(f"**Overlap Warning:** {sv['overlap_warning']}")
    if sv.get("sizing_suggestion"):
        lines.append(f"**Sizing:** {sv['sizing_suggestion']}")
    return "\n".join(lines) + "\n"


def _section_gex() -> str:
    gex = st.session_state.get("_gex_dealer_context") or {}
    if not gex or not gex.get("zone"):
        return ""
    lines = ["## GEX DEALER POSITIONING"]
    lines.append(f"- **Zone:** {gex.get('zone','–')} | Composite: {gex.get('composite',0):+.3f}")
    lines.append(f"- **Delta:** {gex.get('dealer_net_delta',0):+.3f} (call/put OI ratio)")
    lines.append(f"- **Gamma Flip:** ${gex.get('gamma_flip',0):,.0f}")
    lines.append(f"- **Walls:** Put ${gex.get('put_wall',0):,.0f} — Call ${gex.get('call_wall',0):,.0f}")
    return "\n".join(lines) + "\n"


def _section_hmm() -> str:
    hmm = st.session_state.get("_hmm_state") or {}
    if not hmm or not isinstance(hmm, dict):
        return ""
    label = hmm.get("state_label", "–")
    conf = hmm.get("confidence", "–")
    persist = hmm.get("persistence", "–")
    lines = ["## HMM BRAIN STATE"]
    lines.append(f"- **State:** {label} | Confidence: {conf}")
    lines.append(f"- **Persistence:** {persist} days")
    mult_map = {"Bull": "×1.10", "Neutral": "×1.00", "Early Stress": "×0.90",
                "Stress": "×0.85", "Late Cycle": "×0.75", "Crisis": "×0.60"}
    lines.append(f"- **Kelly Multiplier:** {mult_map.get(label, '–')}")
    return "\n".join(lines) + "\n"


def _section_lean_tracker() -> str:
    import os as _os
    _lt_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "lean_tracker.json")
    if not _os.path.exists(_lt_path):
        return ""
    try:
        with open(_lt_path, encoding="utf-8") as _f:
            lt = json.load(_f)
    except Exception:
        return ""
    if not lt:
        return ""
    last = lt[-1]
    filled = [e for e in lt if e.get("fwd_5d_spy_return") is not None]
    lines = ["## LEAN TRACKER"]
    lines.append(f"- **Latest:** {last.get('lean','–')} on {last.get('date','–')} | Domain avg: {last.get('domain_avg','–')}")
    lines.append(f"- **Entries:** {len(lt)} total, {len(filled)} with forward returns")
    if len(filled) >= 3:
        bull = [e for e in filled if e.get("lean") == "BULLISH"]
        bear = [e for e in filled if e.get("lean") == "BEARISH"]
        if bull:
            avg5 = sum(e["fwd_5d_spy_return"] for e in bull) / len(bull)
            lines.append(f"- **Bullish lean avg 5d:** {avg5:+.2f}% (n={len(bull)})")
        if bear:
            avg5 = sum(e["fwd_5d_spy_return"] for e in bear) / len(bear)
            lines.append(f"- **Bearish lean avg 5d:** {avg5:+.2f}% (n={len(bear)})")
    return "\n".join(lines) + "\n"


def _section_suggested_prompts() -> str:
    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "")
    is_risk_off = "Risk-Off" in regime

    prompts = [
        "Given this macro regime and my portfolio positions, what are the biggest risks I should be aware of?",
        "Which of my positions are most exposed to the black swan events listed?",
        "Given the current rate path, should I rotate any positions or adjust my sector exposure?",
    ]
    if is_risk_off:
        prompts.append(
            "The regime is Risk-Off — which positions should I consider hedging or reducing first?"
        )
    else:
        prompts.append(
            "The regime is Risk-On — which sectors or stocks look most attractive for new entries?"
        )
    prompts.append(
        "Summarize the top 3 macro themes from this briefing and suggest a trade idea for each."
    )
    prompts.append(
        "Run the probability calibration again with updated CME FedWatch data. "
        "Use today's ZQ futures implied rate, latest Core PCE, unemployment, VIX, and oil price. "
        "Show the full Bayesian table (prior → likelihood multiplier → posterior) and flag any scenario "
        "where the posterior has shifted >5pp from the previous briefing."
    )

    lines = ["## SUGGESTED AI CONVERSATION STARTERS", ""]
    for p in prompts:
        lines.append(f'> "{p}"')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level export builders
# ---------------------------------------------------------------------------

def _build_markdown_export(open_trades: list) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        _section_executive_summary(open_trades),
        _section_current_events(),
        _section_regime(),
        _section_rate_path(),
        _section_policy_transmission(),
        _section_regime_plays(),
        _section_fed_plays(),
        _section_black_swans(),
        _section_doom(),
        _section_whale(),
        _section_discovery_plays(),
        _section_portfolio(open_trades),
        _section_factor_exposure(open_trades),
        _section_factor_analysis(),
        _section_portfolio_analysis(open_trades),
        _section_sim_verdict(),
        _section_gex(),
        _section_hmm(),
        _section_lean_tracker(),
    ]
    populated = sum(1 for s in sections if s.strip())
    total = len(sections)

    header = (
        f"# MACRO INTELLIGENCE BRIEFING\n"
        f"Generated: {now_str} | Signals: {populated}/{total} populated\n"
    )

    body_parts = [header]
    for section in sections:
        if section.strip():
            body_parts.append("---\n")
            body_parts.append(section)

    body_parts.append("---\n")
    body_parts.append(_section_suggested_prompts())

    return "\n".join(body_parts)


def _build_regard_export() -> str:
    """Full technical specification: data sources, math formulas, decision engine, AI debate.
    Every number, every formula, every weight — with live values substituted in.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Pull live state ───────────────────────────────────────────────────────
    rc  = st.session_state.get("_regime_context") or {}
    tac = st.session_state.get("_tactical_context") or {}
    of  = st.session_state.get("_options_flow_context") or {}
    sz  = st.session_state.get("_stress_zscore") or {}
    wf  = st.session_state.get("_whale_flow_score") or {}
    ev  = st.session_state.get("_events_sentiment_score") or {}
    ca  = st.session_state.get("_canary_score") or {}
    fc  = st.session_state.get("_fear_composite") or {}
    rp  = st.session_state.get("_dominant_rate_path") or {}
    cv  = st.session_state.get("_qir_conviction_score")
    gex = st.session_state.get("_gex_dealer_context") or {}

    def v(val, fmt=None, fallback="not run"):
        if val is None or val == "" or val == "–":
            return fallback
        try:
            return format(val, fmt) if fmt else str(val)
        except Exception:
            return str(val)

    score      = rc.get("score", 0) or 0
    macro_s    = rc.get("macro_score", 50) or 50
    leading_s  = rc.get("leading_score", 50) or 50
    div_pts    = rc.get("leading_divergence", 0) or 0
    tac_score  = tac.get("tactical_score", 50) or 50
    of_score   = of.get("options_score", 50) or 50
    bull_pct   = wf.get("bull_pct", 50) or 50
    fear_score = fc.get("score", 50) or 50

    # Rate path scale
    _RATE_SCALE = {"CUT_75":1.0,"CUT_50":0.75,"CUT_25":0.5,"HOLD":0.0,
                   "HIKE_25":-0.5,"HIKE_50":-0.75,"HIKE_100":-1.0}
    rate_path_str = rp.get("scenario","HOLD") if isinstance(rp, dict) else str(rp)
    rate_val   = _RATE_SCALE.get(str(rate_path_str).upper(), 0.0)

    lines = [
        "# REGARD EXPORTS — NARRATIVE INVESTING TOOL",
        f"Generated: {now_str}",
        "Full technical specification — every data source, formula, weight, and AI flow.",
        "Hand this to any quant or engineer; they will understand exactly how the machine works.",
        "",
        "═" * 70,
        "",

        # ── SECTION 1: DATA COLLECTION PIPELINE ──────────────────────────────
        "## SECTION 1 — DATA COLLECTION PIPELINE",
        "",
        "### 1A. FRED (Federal Reserve Economic Data)",
        "Base URL:  https://api.stlouisfed.org/fred/series/observations",
        "Auth:      FRED_API_KEY env var",
        "TTL:       6 hours (stress_client.py _fetch_fred_series)",
        "Rate:      no explicit limit; sequential with timeout=10s",
        "",
        "Series fetched:",
        "  DGS10        10-Year Treasury Yield                (daily)",
        "  DGS2         2-Year Treasury Yield                 (daily)",
        "  T10Y2Y       10Y-2Y Yield Curve Spread             (daily)",
        "  T10Y3M       10Y-3M Yield Curve Spread             (daily)",
        "  DFII10       10-Year Real TIPS Yield               (daily)",
        "  BAMLH0A0HYM2 ICE BofA HY OAS (credit spread)      (daily)",
        "  BAMLC0A0CM   ICE BofA IG OAS (credit spread)      (daily)",
        "  DPCCRV1Y     1Y Consumer Inflation Expectations    (monthly)",
        "  UMCSENT      U of Michigan Consumer Sentiment      (monthly)",
        "  PERMIT       Building Permits                      (monthly)",
        "  PAYEMS       Nonfarm Payrolls                      (monthly)",
        "  ICSA         Initial Jobless Claims                (weekly)",
        "  INDPRO       Industrial Production Index           (monthly)",
        "  BUSLOANS     Commercial & Industrial Loans         (monthly)",
        "  M2SL         M2 Money Supply                      (monthly)",
        "  PCEPI        PCE Price Index (core inflation)      (monthly)",
        "  MEHOINUSA672N Median Household Income              (annual)",
        "",
        "### 1B. yfinance (Market Data)",
        "Library:   yfinance — wraps Yahoo Finance",
        "TTL:       1 hour for prices (market_data.py @st.cache_data ttl=3600)",
        "           4 hours for options chain",
        "           24 hours for factor sensitivity (portfolio_sizing.py)",
        "Batch:     yf.download(ticker_list, threads=True) — parallel fetch",
        "",
        "Tickers used across modules:",
        "  SPY, QQQ, DIA, IWM         — equity benchmarks",
        "  RSP                        — equal-weight S&P (breadth signal)",
        "  HYG, JNK                   — high yield ETFs (credit proxy)",
        "  TLT, IEI, AGG              — treasury ETFs",
        "  TIP                        — TIPS ETF (inflation expectations)",
        "  GLD, SLV                   — commodities",
        "  USO, XLE                   — oil/energy",
        "  COPX, FCX                  — copper (growth proxy)",
        "  UUP                        — US Dollar ETF",
        "  EWG, FXI                   — global manufacturing proxy",
        "  UVXY, VXX                  — VIX ETPs",
        "  USDCAD=X                   — FX rate for CAD portfolio normalization",
        "  Canary watchlist (20+ ETFs) — breadth/stress signals",
        "",
        "### 1C. SEC EDGAR",
        "Base URL:  https://data.sec.gov / https://efts.sec.gov",
        "Auth:      User-Agent header (NarrativeInvestingTool jud_rabs@yahoo.com)",
        "Rate:      10 req/sec max via _rate_limit() — time.sleep enforced",
        "TTL:       24 hours (@st.cache_data ttl=86400)",
        "Parallel:  ThreadPoolExecutor(max_workers=5)",
        "",
        "Endpoints used:",
        "  /submissions/{CIK}.json    — company filing history",
        "  /api/xbrl/companyfacts/    — XBRL financial facts",
        "  company_tickers.json       — CIK-to-ticker mapping (~12,000 companies)",
        "  efts full-text search      — filing keyword search",
        "  13F-HR filings             — institutional holdings (quarterly)",
        "  Form 4 filings             — insider transactions",
        "  SC 13D/G filings           — activist positions",
        "",
        "### 1D. AI APIs",
        "Groq:      api.groq.com/openai/v1/chat/completions",
        "           Auth: GROQ_API_KEY env var",
        "           Models: llama-3.3-70b-versatile (default), llama-3.1-8b-instant (fast)",
        "           Retry: exponential backoff on 429/5xx (2s → 4s, max 2 attempts)",
        "",
        "Claude:    api.anthropic.com (via anthropic SDK)",
        "           Auth: ANTHROPIC_API_KEY env var",
        "           Model: claude-sonnet-4-6",
        "           Used for: valuation deep analysis, adversarial debate (Highly Regarded Mode)",
        "",
        "xAI/Grok:  api.x.ai/v1/chat/completions",
        "           Auth: XAI_API_KEY env var",
        "           Model: grok-4-1-fast-reasoning (Regard Mode)",
        "",
        "### 1E. Other Sources",
        "NewsAPI:   newsapi.org — current events headlines (NEWSAPI_KEY)",
        "GitHub Gist: signals cache, regime history, inbox — persists across Streamlit redeploys",
        "Telegram:  Outbound alerts via TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID",
        "",
        "═" * 70,
        "",

        # ── SECTION 2: REGIME COMPOSITE MATH ─────────────────────────────────
        "## SECTION 2 — MACRO REGIME COMPOSITE MATH",
        "(modules/risk_regime.py — _build_macro_dashboard)",
        "",
        "### Step 1: Per-signal z-score normalization",
        "For each signal series s with 252-day rolling window:",
        "  z = (s_current - mean_252d) / std_252d",
        "  z is clamped to [-3, +3] to prevent outlier dominance",
        "  Confidence = 100 × (1 - std_252d / abs(mean_252d + ε))",
        "    → high confidence when signal is stable relative to its level",
        "    → low confidence when signal is noisy or near zero",
        "",
        "### Step 2: Signal tier weights",
        "TIER 1 (weight 2.0×) — highest predictive lead time:",
        "  Credit Spreads HY OAS    z-score, inverted (tighter = risk-on)",
        "  Yield Curve 10Y-2Y       z-score + shape modifier",
        "  VIX                      z-score, inverted",
        "  Financial Conditions     z-score, inverted",
        "  Leading Economic Index   z-score",
        "  Real Yields (TIPS 10Y)   z-score, inverted",
        "  Credit Impulse (loans)   z-score of quarterly YoY change",
        "  Yield Curve 3M-10Y       z-score + shape modifier",
        "",
        "TIER 2 (weight 1.5×) — strong confirming signals:",
        "  Equity Trend (% vs 120d MA)    SPY/QQQ/DIA blended",
        "  Unemployment (Sahm Rule)       z-score, inverted",
        "  Net Liquidity (M2 - Drains)    z-composite (M2 - RealYield - USD - HY)",
        "  Initial Jobless Claims         z-score, inverted",
        "  Manufacturing Employment       z-score of YoY change",
        "  Market Breadth (RSP/SPY)       z-score of ratio",
        "  Credit Spreads IG OAS          z-score, inverted",
        "  HY-IG Quality Spread           z-score, inverted",
        "  Rate Expectations (2Y vs FFR)  z-score, inverted",
        "",
        "TIER 3 (weight 1.0×) — standard signals:",
        "  Commodity Trend      50/30/20% blend of 1d/5d/30d returns",
        "  US Dollar (UUP)      blended return",
        "  Industrial Production z-score of YoY change",
        "  Core Inflation (PCE) z-score of YoY change",
        "  Term Premium         z-score",
        "  Gamma Exposure       SPY dealer gamma at spot / 10000",
        "  Copper/Gold Ratio    z-score (growth vs safety proxy)",
        "  Consumer Sentiment   z-score",
        "  Global Manufacturing EWG+FXI 1-month blended return",
        "",
        "TIER 4 (weight 0.5×) — slow-moving / noisy:",
        "  S&P P/E (CAPE proxy) clamped score",
        "  Corporate CAPEX      CAPEX YoY minus M2 YoY",
        "  Building Permits     z-score",
        "",
        "### Step 3: Category-level aggregation",
        "Signals grouped by asset class (Rates, Credit, Equities, Growth, Volatility, etc.)",
        "Within each category:",
        "  category_score = Σ(signal_z × tier_weight × confidence/100)",
        "                   ─────────────────────────────────────────────",
        "                   Σ(tier_weight × confidence/100)",
        "",
        "Across categories:",
        "  composite = Σ(category_score × best_tier_weight × avg_confidence)",
        "              ────────────────────────────────────────────────────────",
        "              Σ(best_tier_weight × avg_confidence)",
        "",
        "### Step 4: Scale to 0-100",
        "  macro_score = int(round((composite + 1.0) × 50))",
        "  ≥60 = Risk-On  |  ≤40 = Risk-Off  |  40-60 = Neutral",
        "",
        f"LIVE: composite z = {v(score, '+.3f')}  →  macro_score = {v(macro_s)}/100",
        "",
        "### Step 5: Leading sub-score (12 fast signals)",
        "Subset: Credit Impulse, LEI, Yield Curves ×2, Credit Spreads ×2,",
        "        HY-IG Quality, Real Yields, VIX, Building Permits, Jobless Claims,",
        "        Rate Expectations",
        "Formula: confidence-weighted average only (no tier suppression)",
        "  leading_score = int(round((leading_agg + 1.0) × 50))",
        "",
        "Leading Divergence = leading_score − macro_score",
        "  > +7 pts  → 'Early Risk-On Setup'  (fast signals running ahead bullish)",
        "  < −7 pts  → 'Early Risk-Off Warning' (fast signals running ahead bearish)",
        "  else      → 'Aligned'",
        "",
        f"LIVE: leading_score = {v(leading_s)}/100  |  divergence = {v(div_pts, '+d')} pts",
        "",
        "═" * 70,
        "",

        # ── SECTION 3: TACTICAL LAYER MATH ───────────────────────────────────
        "## SECTION 3 — TACTICAL LAYER MATH",
        "(modules/risk_regime.py — _build_tactical_dashboard)",
        "",
        "9 signals, each scored 0-100, weighted and averaged:",
        "",
        "  Signal                         Weight   Method",
        "  ─────────────────────────────────────────────────────────────────",
        "  VIX level + 5d trend           2.0×     z-score, inverted; trend ±5pts",
        "  VIX term structure (VIX/VIX3M) 2.0×     contango=bullish, backwardation=bearish",
        "  SPY vs 20d/50d MA + slope      1.5×     % above/below MA, slope direction",
        "  SPY momentum (5d vs 20d ROC)   1.5×     acceleration, not just level",
        "  Market breadth (RSP/SPY 5d Δ)  1.0×     equal-weight vs cap-weight divergence",
        "  VIX full curve (9D/1M/3M/6M)   1.5×     shape of vol term structure",
        "  CBOE SKEW (tail risk)          1.0×     high skew = hedging demand = bearish",
        "  Fear & Greed Index             1.0×     contrarian — extreme fear = bullish",
        "  AAII Sentiment spread          0.8×     contrarian — bears > bulls = bullish",
        "",
        "  tactical_score = Σ(signal × weight) / Σ(weight)   →  0-100",
        "  ≥65 = bullish  |  <38 = bearish  |  38-64 = neutral",
        "",
        f"LIVE: tactical_score = {v(tac_score)}/100 — {tac.get('label','not run')}",
        "",
        "═" * 70,
        "",

        # ── SECTION 4: QUANTIFIED SIGNAL SCORES ──────────────────────────────
        "## SECTION 4 — QUANTIFIED SIGNAL SCORES",
        "(services/signal_quantifier.py)",
        "",
        "### 4A. Stress Z-Score",
        "Inputs: 5 FRED series, z-scored vs 52-week history",
        "  HY OAS    (BAMLH0A0HYM2)  — high yield credit stress",
        "  IG OAS    (BAMLC0A0CM)    — investment grade credit stress",
        "  T10Y2Y                    — yield curve inversion",
        "  T10Y3M                    — Fed's preferred recession indicator",
        "  DFII10                    — real yields (liquidity tightness)",
        "",
        "For each series: z = (current - mean_52wk) / std_52wk",
        "Composite stress_z = mean of all available z-scores",
        "stress_pct = percentile rank vs 52-week history",
        "High stress_pct (>80th) = elevated systemic risk",
        "",
        f"LIVE: stress_z = {v(sz.get('z'), '+.2f')}  |  {v(sz.get('pct'))}th percentile",
        "",
        "### 4B. Whale Flow Score",
        "(from SEC 13F quarterly filings via screen_whale_buyers())",
        "  bull_rows = status in ['new', 'increased']  →  bull_flow = Σ value_change",
        "  bear_rows = status in ['decreased','sold']  →  bear_flow = Σ |value_change|",
        "  bull_pct  = bull_flow / (bull_flow + bear_flow) × 100",
        "  net_flow_bn = (bull_flow - bear_flow) / 1e9",
        "  conviction  = new_positions / (new + sold + 1)",
        "  rotation    = (risk_on_flow - defensive_flow) / (|risk_on| + |defensive| + 1)",
        "",
        "13D Activism cross-reference:",
        "  Activist target names fuzzy-matched against 13F bull position issuers",
        "  Each aligned activist: conviction += 0.15 (cap +0.40)",
        "  If aligned_bull > 0 and aligned_bear == 0: leading_signal = 'BULLISH'",
        "",
        f"LIVE: bull_pct = {v(bull_pct, '.1f')}%  |  {wf.get('label','not run')}",
        f"      activism_aligned = {v(wf.get('activism_aligned'))}  |  leading = {wf.get('leading_signal','–')}",
        "",
        "### 4C. Canary Score",
        "Canary watchlist: 20+ cross-asset ETFs (EEM, HYG, JNK, TLT, GLD, etc.)",
        "  breadth_pct = % of canaries above 20d MA",
        "  momentum_avg = average 20d return across canaries",
        "  drawdown_pct = average drawdown from 52w high",
        "  vol_surge = % of canaries with 10d vol > 30d vol",
        "  composite = 0.40×breadth + 0.30×momentum + 0.20×drawdown + 0.10×vol_surge",
        "  All sub-scores normalized 0-100, composite 0-100",
        "",
        f"LIVE: canary_composite = {v(ca.get('composite'))}  |  breadth = {v(ca.get('breadth_pct'))}%",
        "",
        "### 4D. Events Sentiment Score",
        "Source: AI-extracted SENTIMENT JSON from current events digest (Groq)",
        "Format: {sentiment: float [-1,+1], uncertainty: float [0,1], bull_hits: int, bear_hits: int}",
        "Sentiment = net directional lean of macro news headlines",
        "Fallback: keyword counting if AI score not yet populated",
        "",
        f"LIVE: sentiment = {v(ev.get('sentiment'), '+.3f')}  |  uncertainty = {v(ev.get('uncertainty'), '.2f')}",
        "",
        "### 4E. Fear Composite Index",
        "(services/signal_quantifier.py — compute_fear_composite)",
        "Combines 5 inputs, all normalized to 0-100 (higher = more fearful):",
        "",
        "  Input              Weight   Normalization",
        "  ──────────────────────────────────────────────────────────",
        "  Stress z-score     25%      pct_rank → 0-100 (high pct = high fear)",
        "  Macro regime score 20%      inverted: (100 - macro_score)",
        "  Canary breadth     20%      inverted: (100 - breadth_pct)",
        "  Whale bull_pct     20%      inverted: (100 - bull_pct)",
        "  Events sentiment   15%      inverted: (1 - sentiment) × 50 + 50",
        "",
        "  fear_score = Σ(input_normalized × weight)",
        "  0-20: Euphoria  |  20-35: Low Fear  |  35-55: Neutral",
        "  55-70: Elevated  |  70-85: High Fear  |  85-100: Extreme Fear",
        "",
        f"LIVE: fear_score = {v(fear_score, '.0f')}/100 — {fc.get('label','not run')}",
        "",
        "═" * 70,
        "",

        # ── SECTION 5: PATTERN CLASSIFICATION ────────────────────────────────
        "## SECTION 5 — PATTERN CLASSIFICATION",
        "(modules/quick_run.py — _classify_signals)",
        "",
        "Three binary thresholds determine which pattern fires:",
        "  r_bull = 'Risk-On' in regime_label  OR  score > +0.30",
        "  r_bear = 'Risk-Off' in regime_label  OR  score < -0.30",
        "  t_bull = tactical_score >= 65",
        "  t_bear = tactical_score <  38",
        "  o_bull = options_score  >= 65",
        "  o_bear = options_score  <  38",
        "",
        "  Pattern                  Condition",
        "  ──────────────────────────────────────────────────────────",
        "  BULLISH_CONFIRMATION     r_bull AND t_bull AND o_bull",
        "  BEARISH_CONFIRMATION     r_bear AND t_bear AND o_bear",
        "  PULLBACK_IN_UPTREND      r_bull AND t_bear AND o_bull",
        "  OPTIONS_FLOW_DIVERGENCE  r_bull AND t_bull AND o_bear",
        "  BEAR_MARKET_BOUNCE       r_bear AND t_bull AND o_bull",
        "  LATE_CYCLE_SQUEEZE       r_bear AND t_bear AND o_bull",
        "  GENUINE_UNCERTAINTY      else (no 2/3 alignment)",
        "",
        f"LIVE: score={v(score,'+.2f')}  tac={v(tac_score)}  opts={v(of_score)}",
        f"  r_bull={'YES' if 'Risk-On' in rc.get('regime','') or score > 0.3 else 'NO'}  "
        f"r_bear={'YES' if 'Risk-Off' in rc.get('regime','') or score < -0.3 else 'NO'}  "
        f"t_bull={'YES' if tac_score >= 65 else 'NO'}  t_bear={'YES' if tac_score < 38 else 'NO'}  "
        f"o_bull={'YES' if of_score >= 65 else 'NO'}  o_bear={'YES' if of_score < 38 else 'NO'}",
        "",
        "═" * 70,
        "",

        # ── SECTION 6: CONVICTION SCORING FORMULA ────────────────────────────
        "## SECTION 6 — CONVICTION SCORING FORMULA",
        "(modules/quick_run.py — _classify_signals, post-pattern)",
        "",
        "Only fires for the 6 concrete patterns (not GENUINE_UNCERTAINTY).",
        "All components normalized to 0→1 before applying max-point weight.",
        "",
        "  Component          Formula                          Max pts",
        "  ─────────────────────────────────────────────────────────────",
        "  Regime strength    abs(score) × 40                    40",
        "    age-weighted:    × max(0.75, 1 - age_hours/48)      (tapers if FRED stale)",
        "  Rate path          rate_dir × abs(rate_val) × 8      ±8",
        "    rate_val scale:  CUT_75=1.0  CUT_50=0.75  CUT_25=0.5",
        "                     HOLD=0.0",
        "                     HIKE_25=-0.5  HIKE_50=-0.75  HIKE_100=-1.0",
        "    rate_dir:        +1 if rate aligns with pattern, -1 if contradicts",
        "  Tactical strength  abs(tac-50)/50 × 30                30",
        "  Options strength   abs(of-50)/50 × 20                 20",
        "  Whale flow         whale_dir × abs((bull_pct-50)/50) × 10  ±10",
        "    whale_dir:       +1 if whale direction matches pattern, -1 if not",
        "  Leading divergence dir_match × min(abs(div)/20, 1.0) × 10  ±10",
        "    dir_match:       +1 if fast signals confirm pattern, -1 if contradict",
        "  Regime velocity    vel_dir × abs(vel_norm) × 8        ±8",
        "    velocity:        current_macro_score - score_5d_ago (from history)",
        "    vel_norm:        clamp(velocity / 25, -1, 1)",
        "    vel_dir:         +1 if accelerating in pattern direction",
        "",
        "  raw_conviction = sum of all components above",
        "  fear_mult      = 1.0 - abs(fear_score - 50)/50 × 0.30   (0.70..1.00)",
        "                 → extremes (panic or euphoria) degrade certainty by up to 30%",
        "  conviction     = int(max(0, min(100, round(raw_conviction × fear_mult))))",
        "",
        "  Conviction → Position size:",
        "    ≥75 → 50% SIZE   |   55-74 → 40% SIZE",
        "    40-54 → 30% SIZE  |   <40  → 20% SIZE",
        "",
    ]

    # Live conviction calculation
    if cv is not None:
        r_pts = abs(score) * 40
        t_pts = abs(tac_score - 50) / 50.0 * 30
        o_pts = abs(of_score - 50) / 50.0 * 20
        w_norm = (bull_pct - 50) / 50.0
        f_ext = abs(fear_score - 50) / 50.0
        f_mult = 1.0 - f_ext * 0.30
        r_val_n = _RATE_SCALE.get(str(rate_path_str).upper(), 0.0)
        lines += [
            "LIVE CONVICTION BREAKDOWN:",
            f"  regime_pts    = abs({v(score,'+.3f')}) × 40         = {r_pts:.1f}",
            f"  rate_pts      = rate_dir × abs({r_val_n:.2f}) × 8  = ±{abs(r_val_n)*8:.1f} (direction depends on pattern)",
            f"  tac_pts       = abs({v(tac_score)}-50)/50 × 30      = {t_pts:.1f}",
            f"  opts_pts      = abs({v(of_score)}-50)/50 × 20       = {o_pts:.1f}",
            f"  whale_pts     = whale_dir × abs({w_norm:+.2f}) × 10  = ±{abs(w_norm)*10:.1f}",
            f"  fear_mult     = 1.0 - {f_ext:.2f}×0.30              = {f_mult:.2f}",
            f"  FINAL: {cv}/100",
        ]
    else:
        lines.append("LIVE: conviction not yet computed — run QIR first")

    lines += [
        "",
        "═" * 70,
        "",

        # ── SECTION 7: KELLY CRITERION ────────────────────────────────────────
        "## SECTION 7 — KELLY CRITERION (QIR POSITION SIZING)",
        "(services/portfolio_sizing.py — compute_qir_kelly + compute_triple_kelly)",
        "",
        "### p — Win Probability (Bayesian Shrinkage)",
        "Prevents sparse trade history from dominating — anchors to 50% prior:",
        "  shrink_weight  = 5.0  (pseudo-observations)",
        "  p_hist_shrunk  = (n_wins + 5.0×0.5) / (n_closed + 5.0)",
        "  history_weight = min(n_closed / 20.0, 0.6)    ← max 60% weight at 20 trades",
        "  p_conviction   = conviction_score / 100",
        "  p = p_hist_shrunk × history_weight + p_conviction × (1 - history_weight)",
        "",
        "Behaviour:",
        "  0 closed trades  → p = pure conviction score / 100",
        "  1 closed trade   → history_weight = 0.05  (5% history, 95% conviction)",
        "  5 closed trades  → history_weight = 0.25",
        "  20+ closed trades→ history_weight = 0.60  (60% history, 40% conviction)",
        "",
        "### b — Win/Loss Ratio",
        "  Preferred: avg_win_pct / abs(avg_loss_pct) from closed trades",
        "    Requires: ≥1 win AND ≥1 loss in trade journal",
        "  Fallback — regime-implied LONG b (_REGIME_B_IMPLIED):",
        "    Goldilocks  → b = 1.8   (trends extend, winners run)",
        "    Reflation   → b = 1.4",
        "    Neutral     → b = 1.2   (default when quadrant unrecognised)",
        "    Stagflation → b = 0.8   (chop, even wins are small)",
        "    Deflation   → b = 0.6",
        "  Bearish verdict (p < 0.45) uses SHORT b (_SHORT_B_IMPLIED):",
        "    Stagflation → b = 1.8   (best short env — mirror of Goldilocks long)",
        "    Deflation   → b = 1.4",
        "    Reflation   → b = 0.8",
        "    Goldilocks  → b = 0.6   (worst short env)",
        "",
        "### Kelly Formula",
        "  q            = 1 - p",
        "  kelly_full   = (b×p - q) / b",
        "  kelly_half   = kelly_full × 0.5              (conservative half-Kelly)",
        "",
        "### Stress Discount",
        "  stress_discount = (fear_score / 100) × 0.30  (up to 30% reduction)",
        "  kelly_half_base = kelly_half × (1 - stress_discount)   ← saved as reference",
        "  Cap:            kelly_half = min(kelly_half, 0.15)  (15% max position)",
        "",
        "### Signal-Aligned Kelly (4-tier multiplier stack)",
        "After stress discount, two additional multipliers are applied in sequence:",
        "",
        "  1. Cross-timeframe Alignment Multiplier (align_multiplier):",
        "     Compares each signal's direction vs the verdict direction (bull/bear/neutral):",
        "       Options Flow → Fast signal",
        "       Tactical     → Medium signal",
        "       Regime       → Slow signal",
        "       Conviction   → Very Slow signal",
        "     Count how many of the 4 agree with verdict direction:",
        "       4/4 agree → ×1.00   (full size — no arbitrage conflict)",
        "       3/4 agree → ×0.90",
        "       2/4 agree → ×0.75",
        "       1/4 agree → ×0.50",
        "       0/4 agree → ×0.25   (minimum — signals fighting each other)",
        "",
        "  2. HMM Regime Multiplier (hmm_multiplier) — applied after alignment:",
        "     HMM is a GATE, not a signal. It scales the already-aligned Kelly:",
        "       Bull        → ×1.10",
        "       Neutral     → ×1.00",
        "       Early Stress→ ×0.90",
        "       Stress      → ×0.85",
        "       Late Cycle  → ×0.75",
        "       Crisis      → ×0.60",
        "",
        "  Final: kelly_half = kelly_half_base × align_multiplier × hmm_multiplier",
        "         (re-capped at 15% after multipliers)",
        "",
        "### HALF-KELLY REFERENCE TABLE",
        "Displayed in the Kelly card — shows all alignment × HMM combinations:",
        "  Rows: 4/4, 3/4, 2/4, 1/4, 0/4 signal agreement",
        "  Cols: Bull, Neutral, Stress, Late Cycle, Crisis HMM states",
        "  Each cell: base_pct × align_mult × hmm_mult (capped 15%)",
        "",
        "### FORECAST STREAKS (kelly card)",
        "Two independent streak counters displayed above signal alignment:",
        "  Price streak  — consecutive correct/incorrect outcomes for SPY price trades",
        "                  (valuation and squeeze-type entries, price-resolved)",
        "  Signal streak — consecutive correct/incorrect macro verdict outcomes",
        "                  (all other signal types, macro-resolved)",
        "  Divergence flag: price streak correct + signal streak incorrect",
        "                   → 'execution ahead of model' warning",
        "  Streak break:    last 2 outcomes in a category flipped direction",
        "                   → zero-cross alert",
        "",
        "═" * 70,
        "",

        # ── SECTION 7B: BIMODAL SIZING — TRIPLE KELLY ────────────────────────
        "## SECTION 7B — BIMODAL SIZING: TRIPLE KELLY (GENUINE_UNCERTAINTY only)",
        "(services/portfolio_sizing.py — compute_triple_kelly)",
        "",
        "Fires ONLY when QIR pattern = GENUINE_UNCERTAINTY.",
        "Separates the market into three concurrent position frameworks with different",
        "time horizons, p-sources, and b-values. Displayed as collapsible rows in the",
        "Kelly card area above the standard single-Kelly card.",
        "",
        "### 1. Structural Long (weeks/months)",
        "  p = macro_score / 100           (regime confidence, long-duration)",
        "  b = _REGIME_B_IMPLIED[quadrant]  (always long-side — structural is a regime trade)",
        "  Direction is set by macro/HMM, NOT the short-term forced lean.",
        "  The lean is derived from noisy short-duration inputs (sentiment, options, event risk).",
        "  Letting the lean flip the structural b would introduce short-term noise into a",
        "  position meant to hold for weeks/months — by design, structural stays long-biased.",
        "  HMM multiplier applied           (this IS the regime trade)",
        "  Cap: 15%",
        "  Note: b defaults to 1.2 when quadrant not in the map (e.g. 'Transition')",
        "  ⚠ p is signal-derived, not historically calibrated vs actual trade outcomes",
        "",
        "### 2. Tactical Short (days/weeks)",
        "  p = lean_pct / 100              (forced lean confidence from domain avg, 51–75%)",
        "  b = _SHORT_B_IMPLIED[quadrant]   (short-side regime edge)",
        "  Fear boost: × (1 + (fear_score/100) × 0.20)  — high fear amplifies short edge",
        "  No HMM multiplier               (short-term hedge, not a regime trade)",
        "  Cap: 15%",
        "  ⚠ lean_pct is a sentiment-derived confidence measure, NOT a historical win rate.",
        "    Use this as a directional sizing guide, not a calibrated Kelly.",
        "",
        "### 3. Tactical Long / Scalp (1-3 days)",
        "  uncertainty_penalty = 1 - (uncertainty_score / 200)  → range 0.50–1.00",
        "  p = (leading_score / 100) × uncertainty_penalty",
        "  b = 1.1  (tight — momentum scalps have narrow edge)",
        "  half_kelly × uncertainty_penalty again (double-penalised for uncertainty)",
        "  Cap: 8%  (scalps never go large)",
        "  If p < 0.5 (negative EV) → outputs 0.0% correctly",
        "",
        "### Tactical Short Kelly badge (Short Setup card)",
        "When GENUINE_UNCERTAINTY + BEARISH lean, the Tactical Short half-Kelly %",
        "is injected as a sizing badge at the top of the Short Setup card.",
        "",
        "### Tactical Long / Scalp Kelly badge (Buy Setup card)",
        "Same logic — appears in Buy Setup card. Labeled 'TACTICAL LONG KELLY'",
        "for bullish lean, 'SCALP KELLY' for bearish lean (smaller, uncertainty-penalised).",
        "",
        "1.5R Reward-to-Risk: stop×2 ATR, target×3 ATR → breakeven win rate = 40%",
        "",
        "═" * 70,
        "",

        # ── SECTION 8: SPY SIGNAL LOG & AUTO-EVALUATION ───────────────────────
        "## SECTION 8 — SPY SIGNAL LOG & AUTO-EVALUATION",
        "(services/forecast_tracker.py)",
        "",
        "Purpose: accumulate signal-tagged SPY trades to eventually derive empirical",
        "         signal weights via OLS regression (Y=alpha vs SPY, X=signal vector)",
        "",
        "### ATR Computation (Weekly)",
        "  Period:  2 years of weekly OHLC bars (interval='1wk')",
        "  ATR(14): rolling 14-week True Range mean",
        "  Stop:    entry ± 2 × weekly_ATR   (weekly ATR ≈ 5× daily ATR)",
        "  Target:  entry ∓ 3 × weekly_ATR",
        "  R:R      = 1.5:1  →  breakeven win rate = 40%",
        "  Horizon: 60 days (macro signals need time to play out)",
        "",
        "### Auto-Evaluation (evaluate_pending)",
        "Runs automatically on every QIR run (no manual trigger needed).",
        "For each open SPY trade:",
        "  Walk daily price history since log date",
        "  Short trades: trail low watermark − 2×ATR as stop, target = entry − 3×ATR",
        "  Long trades:  trail high watermark + 2×ATR as stop, target = entry + 3×ATR",
        "  Outcome: 'correct' (target hit), 'incorrect' (stop hit), 'pending' (horizon not elapsed)",
        "  Records: return_pct, spy_return_pct, alpha_pct (return vs SPY)",
        "",
        "### Signal Vector Captured Per Trade",
        "Full market_context logged at trade entry time:",
        "  regime, quadrant, regime_score, macro_score, leading_score, leading_divergence",
        "  tactical_score, options_score",
        "  whale_bull_pct, whale_leading, whale_conviction",
        "  fear_composite_score, fear_composite_label",
        "  stress_z, stress_z_pct",
        "  conviction_score",
        "  fear_greed_score, vix_spot, vix_structure, fed_path",
        "",
        "Future: once 20-30 SPY trades close, run OLS:",
        "  Y = alpha_pct",
        "  X = [conviction, tactical, options, whale_bull_pct, fear_composite,",
        "       stress_z, leading_divergence, ...]",
        "  Beta coefficients = empirical weights to replace hardcoded points",
        "",
        "═" * 70,
        "",

        # ── SECTION 9: AI DEBATE ARCHITECTURE ────────────────────────────────
        "## SECTION 9 — AI DEBATE ARCHITECTURE",
        "(services/claude_client.py — generate_adversarial_debate)",
        "",
        "3 sequential LLM calls using the same signal block as input:",
        "",
        "### Signal Block (input to all 3 calls)",
        "Built by utils/signal_block.py — build_macro_block()",
        "Contains: regime score, quadrant, leading divergence, tactical signals,",
        "          options flow, stress z-scores, whale flow, current events digest,",
        "          Fed rate path, doom briefing, plays + tickers from narrative discovery",
        "",
        "### Call 1: Dr. Doomburger (Bear)",
        "Persona: legendary permabear macro analyst",
        "Task:    strongest possible BEARISH case using ONLY the signal data",
        "Rules:   must cite specific numbers and signal names — no vague doom",
        "         not allowed to be balanced — maximum bear conviction required",
        "Output:  bear_argument (3-5 sentences, data-cited)",
        "",
        "### Call 2: Sir Fukyerputs (Bull)",
        "Persona: relentless bull maximalist",
        "Task:    strongest possible BULLISH case using ONLY the signal data",
        "Rules:   same as bear — cite numbers, maximum conviction, no hedging",
        "Output:  bull_argument (3-5 sentences, data-cited)",
        "",
        "### Call 3: Commander Wincyl (Verdict)",
        "Persona: no-nonsense macro risk arbiter",
        "Input:   both arguments + original signal block + court record (last 5 verdicts)",
        "Task:    structured verdict — who won and why",
        "Output:  {",
        "  verdict: 'BULL WINS' | 'BEAR WINS' | 'CONTESTED'",
        "  confidence: 1-10 (how decisive the evidence was)",
        "  bear_strongest: Judge's pick of the best bear argument point",
        "  bull_strongest: Judge's pick of the best bull argument point",
        "  key_risk: the single most important risk to monitor",
        "  contested_bias: 'BULL' | 'BEAR' | None (when CONTESTED)",
        "  contested_bias_reason: explanation",
        "}",
        "",
        "### Court Record",
        "Last 5 debate verdicts pulled from data/debate_record.db",
        "Included in Judge's prompt for consistency — prevents flip-flopping",
        "Each verdict auto-resolved 5 trading days later vs SPX price change",
        "outcome: 'correct' (SPX moved in verdict direction) | 'wrong'",
        "",
        "### Confidence Penalty",
        "apply_confidence_penalty(confidence):",
        "  if debate verdict contradicts current pattern: confidence × 0.8",
        "  prevents over-confidence when AI and systematic signals disagree",
        "",
        "═" * 70,
        "",

        # ── SECTION 10: DYNAMIC FACTOR SENSITIVITY ────────────────────────────
        "## SECTION 10 — DYNAMIC FACTOR SENSITIVITY",
        "(services/portfolio_sizing.py — compute_dynamic_sensitivity)",
        "",
        "4-factor model: Growth, Inflation, Liquidity, Credit",
        "Factor proxies: SPY (growth), TIP (inflation), IEI (liquidity), HYG (credit)",
        "",
        "OLS regression per ticker (daily log-returns, 126-day lookback, cached 24h):",
        "  log_ret_i = Σ(β_f × log_ret_factor_f) + ε",
        "  betas = (X'X)^-1 X'y  (least squares)",
        "  Normalize: sensitivity_f = tanh(β_f / 2)   → clamps to [-1, +1]",
        "",
        "Usage in conviction scoring:",
        "  regime_fit_score = dot(ticker_sensitivities, quadrant_vector) scaled 0-100",
        "  quadrant vectors: Goldilocks=[+0.9,-0.2,+0.6,+0.4]",
        "                    Stagflation=[-0.5,+0.9,-0.4,-0.5]",
        "                    Reflation=[+0.7,+0.5,+0.3,+0.2]",
        "                    Deflation=[-0.7,-0.5,+0.6,-0.6]",
        "",
        "Fallback: static _SENSITIVITY dict if <63 days of data",
        "",
        "═" * 70,
        "",

        # ── SECTION 11: LIVE SIGNAL STATE SNAPSHOT ───────────────────────────
        "## SECTION 11 — LIVE SIGNAL STATE SNAPSHOT",
        f"(as of {now_str})",
        "",
        f"  Regime:             {rc.get('regime','not run')} | Quadrant: {rc.get('quadrant','–')}",
        f"  Regime score:       {v(score,'+.3f')} (-1=risk-off, +1=risk-on)",
        f"  Macro score:        {v(macro_s)}/100",
        f"  Leading score:      {v(leading_s)}/100",
        f"  Leading divergence: {v(div_pts,'+d')} pts — {rc.get('leading_label','–')}",
        f"  Tactical score:     {v(tac_score)}/100 — {tac.get('label','–')}",
        f"  Options score:      {v(of_score)}/100 — {of.get('label','–')}",
        f"  GEX dealer:         {gex.get('zone','not run')} | composite {v(gex.get('composite'),'+.3f')} | delta {v(gex.get('dealer_net_delta'),'+.3f')} | flip ${v(gex.get('gamma_flip'),',.0f')} | walls ${v(gex.get('put_wall'),',.0f')}–${v(gex.get('call_wall'),',.0f')}",
        f"  Whale bull_pct:     {v(bull_pct,'.1f')}% — {wf.get('label','–')}",
        f"  Whale leading:      {wf.get('leading_signal','–')}",
        f"  Fear composite:     {v(fear_score,'.0f')}/100 — {fc.get('label','–')}",
        f"  Stress z:           {v(sz.get('z'),'+.2f')} ({v(sz.get('pct'))}th pct)",
        f"  Events sentiment:   {v(ev.get('sentiment'),'+.3f')} | uncertainty {v(ev.get('uncertainty'),'.2f')}",
        f"  Canary composite:   {v(ca.get('composite'))}/100 | breadth {v(ca.get('breadth_pct'))}%",
        f"  Rate path:          {rate_path_str} | val={v(rate_val,'+.2f')}",
        f"  QIR conviction:     {v(cv) if cv is not None else 'not run'}/100",
        "",
        "═" * 70,
        "",

        # ── SECTION 12: CALIBRATION ROADMAP ──────────────────────────────────
        "## SECTION 12 — SIGNAL CALIBRATION ROADMAP",
        "",
        "Current state: hardcoded weights derived from economic first principles",
        "Target state:  OLS-derived weights from historical SPY trade outcomes",
        "",
        "What's being collected now (every QIR SPY trade log):",
        "  - Full 20-variable signal vector at time of decision",
        "  - Outcome: alpha vs SPY at stop/target/horizon",
        "  - Weekly ATR-based stops give consistent R:R across market regimes",
        "",
        "When to run calibration:",
        "  - Minimum: 20 closed SPY trades (coefficients unstable below this)",
        "  - Ideal:   30-50 trades across different regime quadrants",
        "  - Method:  OLS regression Y=alpha, X=signal_vector",
        "  - Walk-forward: retrain every 20 new trades, decay old signals",
        "",
        "Signals most likely to show high empirical weight (prior belief):",
        "  leading_divergence  — fast signals lead by design",
        "  fear_composite      — extremes historically mark turning points",
        "  whale_bull_pct      — institutional flow precedes retail price action",
        "",
        "Signals that may prove zero-weight (prior skepticism):",
        "  events_sentiment    — news is often already priced",
        "  options_score       — dealer positioning can reverse quickly",
        "",
        "═" * 70,
        "",

        # ── SECTION 13: HMM BRAIN STATE ───────────────────────────────────────
        "## SECTION 13 — HMM BRAIN STATE (HIDDEN MARKOV MODEL REGIME DETECTOR)",
        "(services/hmm_regime.py — train_hmm, load_current_hmm_state, load_hmm_brain)",
        "",
        "### Purpose",
        "Independent regime detector using unsupervised learning on macro features.",
        "NOT a primary signal — acts as a MULTIPLIER (gate) on Kelly sizing.",
        "Detects latent market states from price/vol/spread behaviour.",
        "",
        "### Model Architecture",
        "  Model:     GaussianHMM (hmmlearn) — 6 hidden states",
        "  States:    Bull, Neutral, Early Stress, Stress, Late Cycle, Crisis",
        "  Features:  SPY return, VIX level, VIX change, HY spread change,",
        "             yield curve (10Y-2Y), TIPS breakeven, SPY 20d vol",
        "  Training:  15 years of daily data, 200 iterations (EM algorithm)",
        "  Retrain:   manual trigger in dashboard; RETRAIN DUE badge after 90 days",
        "",
        "### Initialization — Critical Detail",
        "  init_params='smc' is set explicitly on the GaussianHMM constructor.",
        "  The 's' (startprob), 'm' (means), 'c' (covariance) are re-initialized.",
        "  The 't' (transmat) is intentionally EXCLUDED — preserving the seeded",
        "  diagonal prior instead of letting hmmlearn overwrite it with uniform values.",
        "",
        "### Transition Matrix Prior (Anti-Ping-Pong)",
        "Before fitting, transmat_ is seeded with a diagonal prior to enforce regime",
        "persistence and prevent degenerate state-switching every bar:",
        "  diagonal_prior = 0.70   (each state tends to persist)",
        "  off_diagonal   = (1 - 0.70) / (n_states - 1)",
        "  Laplace smoothing: += 1e-6 per cell, then renormalized",
        "  Result: states are sticky — transitions require sustained signal shift",
        "",
        "### State Assignment",
        "After training, states are labelled by matching learned means to known",
        "macro fingerprints (e.g. high VIX + negative SPY return → Crisis/Stress).",
        "Persistence: consecutive days in the current state (from Viterbi path).",
        "Confidence:  probability of the most likely state from forward algorithm.",
        "",
        "### Transition Matrix Display",
        "Shows next-step probabilities from the current state (one row of transmat_).",
        "Values < 1% displayed as '<1%' to avoid rounding-to-zero confusion.",
        "",
        "### HMM as Kelly Multiplier",
        "  Bull        → ×1.10   (amplify in confirmed uptrend)",
        "  Neutral     → ×1.00   (no adjustment)",
        "  Early Stress→ ×0.90",
        "  Stress      → ×0.85",
        "  Late Cycle  → ×0.75",
        "  Crisis      → ×0.60   (cut size aggressively)",
        "Applied AFTER the cross-timeframe alignment multiplier.",
        "HMM does NOT appear in the Forced Directional Lean / Entry Signal —",
        "it only affects position sizing, not direction.",
        "",
        "### QIR Dashboard Layout (Zone 2)",
        "Top row:    Conviction score (120px) | Kelly + Triple Kelly (flex width)",
        "Bottom row: HMM Brain State (full width)",
        "HMM is placed below Kelly because it is supporting context,",
        "not the primary sizing decision.",
        "",
        "═" * 70,
        "",

        # ── SECTION 14: GEX DEALER POSITIONING ───────────────────────────────
        "## SECTION 14 — GEX DEALER POSITIONING (STRUCTURAL GAMMA EXPOSURE)",
        "(services/market_data.py — fetch_gex_profile)",
        "",
        "### Purpose",
        "Measures aggregate dealer gamma exposure to detect structural support/resistance",
        "zones and likely vol regimes. Dealers hedge options by buying/selling the underlying;",
        "positive gamma = dealers dampen moves (low vol), negative = dealers amplify (high vol).",
        "",
        "### Data Source",
        "  yfinance options chain for SPY (3 nearest expirations)",
        "  TTL: 5 minutes (@st.cache_data ttl=300)",
        "  Thread pool note: @st.cache_data must be unwrapped via __wrapped__ for ThreadPoolExecutor",
        "",
        "### Key Calculations",
        "  Net GEX per strike = call_OI × call_gamma - put_OI × put_gamma  (×100 ×spot)",
        "  Zone:  determined by net GEX at the spot strike",
        "         positive → Positive Gamma Zone (dealers dampen), negative → Negative Gamma Zone (amplify)",
        "  dealer_net_delta = (total_call_OI - total_put_OI) / (total_call_OI + total_put_OI)",
        "         Normalized call/put OI ratio: +1 = all calls, -1 = all puts",
        "  Gamma Flip:  strike where cumulative GEX crosses zero",
        "  Put Wall:    strike with highest put OI × gamma (dealer buying floor)",
        "  Call Wall:   strike with highest call OI × gamma (dealer selling ceiling)",
        "",
        "### 4-Factor Composite Score [-1, +1]",
        "  zone_score  = tanh(total_gex / 2000)              weight 35%",
        "  flip_score  = tanh((spot - flip_strike) / 10)     weight 25%",
        "  delta_score = dealer_net_delta (already normalized) weight 25%",
        "  width_score = tanh((call_wall - put_wall) / 20)   weight 15%",
        "  composite   = 0.35×zone + 0.25×flip + 0.25×delta + 0.15×width",
        "",
        "### Integration",
        "  QIR dashboard:   GEX card with key levels (SPOT, FLIP, DELTA, PUT WALL, CALL WALL)",
        "  Regime Risks:    Full GEX Dealer Positioning section with factor bars",
        "  Valuation:       GEX zone + composite injected into AI valuation prompt",
        "  Export Hub:       Live values in signal snapshot + Co-Engineer brief",
        "  signals_cache:   _gex_dealer_context, _gex_dealer_context_ts",
        "",
        "═" * 70,
        "",

        # ── SECTION 15: LEAN TRACKER ─────────────────────────────────────────
        "## SECTION 15 — DAILY LEAN TRACKER (AUTOMATIC SIGNAL ACCURACY)",
        "(modules/quick_run.py — lean logging after _classify_genuine_uncertainty)",
        "",
        "### Purpose",
        "Logs each QIR's forced directional lean (BULLISH/BEARISH) with full domain scores,",
        "then backfills what SPY actually did 5 and 20 trading days later.",
        "Builds a dataset to answer: which lean + regime + signal combos actually predict returns?",
        "",
        "### Entry Schema (data/lean_tracker.json)",
        "  date, lean, lean_pct, domain_avg",
        "  macro_score, tech_score, opts_score, sent_score, event_score",
        "  regime, hmm_state, hmm_entropy, gex_composite, conviction_score, pattern",
        "  fwd_5d_spy_return (null → backfilled), fwd_20d_spy_return (null → backfilled)",
        "",
        "### Logging",
        "  One entry per calendar day — skips if today already exists",
        "  Runs automatically after every QIR (no manual action needed)",
        "",
        "### Forward-Return Backfill",
        "  Same yfinance pattern as HMM backfill:",
        "  5d return:  look back ~6 entries, fetch SPY close on date vs 5 business days later",
        "  20d return: look back ~22 entries, fetch SPY close on date vs 20 business days later",
        "  Stamps percentage return and saves",
        "",
        "### QIR Dashboard — LEAN ACCURACY Card",
        "  Per-direction avg 5d/20d returns, count, current streak",
        "  Regime × Lean breakdown when 3+ entries have data",
        "  Shows 'accumulating data...' if fewer than 5 entries with forward returns",
        "",
        "═" * 70,
        "",

        # ── SECTION 16: TRADE SPY POPOVER + ATR ─────────────────────────────
        "## SECTION 16 — TRADE SPY POPOVER & ATR TRAILING STOP",
        "(modules/quick_run.py — Trade SPY popover; services/forecast_tracker.py — ATR)",
        "",
        "### SPY Trade Popover",
        "Replaced one-click Log Signal with st.popover containing:",
        "  - Auto-detected key levels: classic pivot points (PP, R1, R2, S1, S2),",
        "    5-day high/low, GEX walls (put wall, call wall, gamma flip)",
        "  - Entry price (default: latest SPY close)",
        "  - Setup level dropdown (user selects which level they're playing)",
        "  - Notes field (free text for thesis, resistance levels, etc.)",
        "  - Confirm button → logs to Forecast Tracker AND Trade Journal",
        "",
        "### Pivot Point Formulas",
        "  PP = (High + Low + Close) / 3",
        "  R1 = 2×PP - Low,  R2 = PP + (High - Low)",
        "  S1 = 2×PP - High, S2 = PP - (High - Low)",
        "  Uses 5-day OHLC from yfinance",
        "",
        "### ATR Trailing Stop Parameters",
        "  Period:  14-week ATR (2 years of weekly bars)",
        "  Stop:    2 × weekly ATR  (filters noise, ~2σ weekly move)",
        "  Target:  3 × weekly ATR  (1.5:1 R:R, breakeven at 40% win rate)",
        "  Horizon: 60 calendar days",
        "",
        "### Auto-Evaluation",
        "  Runs on every QIR (forecast_tracker.evaluate_pending)",
        "  Walks daily price since entry, trails watermark ± ATR stop",
        "  Outcomes: correct (target hit), incorrect (stop hit), pending",
        "",
        "═" * 70,
    ]

    return "\n".join(lines)


def _build_coengine_export() -> str:
    """Build a rich AI co-engineer brief with live signal state + full architecture."""
    from services.signals_cache import _SIGNAL_KEYS
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Live signal state ---
    rc = st.session_state.get("_regime_context") or {}
    tac = st.session_state.get("_tactical_context") or {}
    of = st.session_state.get("_options_flow_context") or {}
    rp = st.session_state.get("_dominant_rate_path") or {}

    regime = rc.get("regime", "not run")
    quadrant = rc.get("quadrant", "unknown")
    score = rc.get("score", 0)
    macro_score = rc.get("macro_score", "–")
    leading_score = rc.get("leading_score", "–")
    div_pts = rc.get("leading_divergence", 0)
    div_label = rc.get("leading_label", "Aligned")
    score_5d = rc.get("score_5d_trend", "–")

    tac_score = tac.get("tactical_score", "–")
    tac_label = tac.get("label", "–")
    action_bias = tac.get("action_bias", "–")

    of_score = of.get("options_score", "–")
    of_label = of.get("label", "–")
    pc_ratio = of.get("pc_ratio", "–")

    _gex_co = st.session_state.get("_gex_dealer_context") or {}
    gex_zone = _gex_co.get("zone", "–")
    gex_comp = _gex_co.get("composite", "–")
    gex_delta = _gex_co.get("dealer_net_delta", "–")
    gex_flip = _gex_co.get("gamma_flip", "–")

    fed_rate = st.session_state.get("_fed_funds_rate", "–")
    scenario = rp.get("scenario", "–")
    prob_pct = rp.get("prob_pct", "–")

    qir_ts = st.session_state.get("_regime_context_ts")
    qir_run = qir_ts.strftime("%Y-%m-%d %H:%M") if isinstance(qir_ts, datetime) else "not run this session"

    # Entry verdict
    entry_verdict = "N/A — run QIR first"
    if rc and tac:
        try:
            from modules.quick_run import _classify_entry_recommendation
            _ls = rc.get("leading_score", 50)
            _ms = rc.get("macro_score", 50)
            _ts = tac.get("tactical_score", 50)
            _os = of.get("options_score", 50) if of else 50
            _dl = rc.get("leading_label", "Aligned")
            _dp = int(rc.get("leading_divergence", 0))
            _er = _classify_entry_recommendation(_ls, _ms, _ts, _os, _dl, _dp)
            entry_verdict = _er.get("verdict", "–")
        except Exception:
            pass

    # Signals loaded count
    loaded = sum(1 for k in _SIGNAL_KEYS if st.session_state.get(k) is not None)
    total = len(_SIGNAL_KEYS)
    missing_keys = [k for k in _SIGNAL_KEYS if st.session_state.get(k) is None]
    missing_str = ", ".join(missing_keys[:8]) + ("…" if len(missing_keys) > 8 else "") if missing_keys else "all loaded"

    lines = [
        "# NARRATIVE INVESTING TOOL — AI CO-ENGINEER BRIEF",
        f"Generated: {now_str}",
        "",
        "## PURPOSE",
        "Paste this into Grok / ChatGPT / Gemini to discuss architecture,",
        "recommend upgrades, and help engineer improvements.",
        "",
        "---",
        "",
        "## 1. LIVE SIGNAL STATE",
        f"QIR last run: {qir_run}",
        f"Signals loaded: {loaded}/{total} keys",
        "",
        "MACRO REGIME",
        f"  Status:     {regime} | Quadrant: {quadrant}",
        f"  Score:      {score:+.2f}  (−1=risk-off, +1=risk-on)" if isinstance(score, (int, float)) else f"  Score:      {score}",
        f"  Composite:  {macro_score}/100",
        f"  Leading:    {leading_score}/100",
        f"  Divergence: {div_pts:+d} pts — {div_label}" if isinstance(div_pts, (int, float)) else f"  Divergence: {div_pts} pts — {div_label}",
        f"  Entry Signal: {entry_verdict}  (BUY THE DIP / WAIT / HOLD / SELL THE RIP)",
        f"  5d Trend:   {score_5d}",
        "",
        "TACTICAL",
        f"  Score:  {tac_score}/100 — {tac_label}",
        f"  Action: {action_bias}",
        "",
        "OPTIONS FLOW",
        f"  Score:  {of_score}/100 — {of_label}",
        f"  P/C Ratio: {pc_ratio}",
        "",
        "GEX DEALER POSITIONING",
        f"  Zone:      {gex_zone}",
        f"  Composite: {gex_comp}",
        f"  Delta:     {gex_delta} (call/put OI ratio)",
        f"  Flip:      {gex_flip}",
        "",
        "FED / RATE PATH",
        f"  Current Fed Funds: {fed_rate}%",
        f"  Dominant scenario: {scenario} ({prob_pct}%)",
    ]

    # HMM state
    _hmm = st.session_state.get("_hmm_state") or {}
    _hmm_label = _hmm.get("state_label", "–") if isinstance(_hmm, dict) else "–"
    _hmm_conf = _hmm.get("confidence", "–") if isinstance(_hmm, dict) else "–"
    lines += [
        "",
        "HMM BRAIN STATE",
        f"  State:      {_hmm_label}",
        f"  Confidence: {_hmm_conf}",
    ]

    # Lean tracker latest
    try:
        import os as _os
        _lt_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "lean_tracker.json")
        if _os.path.exists(_lt_path):
            with open(_lt_path, encoding="utf-8") as _f:
                _lt = json.load(_f)
            if _lt:
                _last = _lt[-1]
                _filled = sum(1 for e in _lt if e.get("fwd_5d_spy_return") is not None)
                lines += [
                    "",
                    "LEAN TRACKER",
                    f"  Latest lean:  {_last.get('lean','–')} ({_last.get('date','–')})",
                    f"  Domain avg:   {_last.get('domain_avg','–')}",
                    f"  Entries:      {len(_lt)} total, {_filled} with forward returns",
                ]
    except Exception:
        pass

    # Quantified scores block (populated after QIR runs signal_quantifier)
    _sq_stress  = st.session_state.get("_stress_zscore") or {}
    _sq_whale   = st.session_state.get("_whale_flow_score") or {}
    _sq_events  = st.session_state.get("_events_sentiment_score") or {}
    _sq_canary  = st.session_state.get("_canary_score") or {}

    def _fmt(v, fmt=None):
        if v in ("–", None, ""):
            return "–"
        try:
            return format(v, fmt) if fmt else str(v)
        except Exception:
            return str(v)

    lines += [
        "",
        "QUANTIFIED SCORES (z-scored vs 1yr history where applicable)",
        f"  Stress z-score:  {_fmt(_sq_stress.get('z'), '+.2f')}  ({_fmt(_sq_stress.get('pct'))}th pct)"
        + (("  [" + "  ".join(f"{k} {v:+.1f}σ" for k, v in _sq_stress.get('components', {}).items()) + "]")
           if _sq_stress.get("components") else ""),
        f"  Whale flow:      {_fmt(_sq_whale.get('bull_pct'), '.1f')}% bull"
        f"   net {_fmt(_sq_whale.get('net_flow_bn'), '+.2f')}B"
        f"   rotation {_fmt(_sq_whale.get('rotation'), '+.2f')}"
        f"   conviction {_fmt(_sq_whale.get('conviction'), '.2f')}"
        f"   — {_sq_whale.get('label', '–')}",
        f"  Events tone:     {_fmt(_sq_events.get('sentiment'), '+.3f')}"
        f"   uncertainty {_fmt(_sq_events.get('uncertainty'), '.2f')}"
        f"   bull {_sq_events.get('bull_hits', '–')} / bear {_sq_events.get('bear_hits', '–')}"
        f"   — {_sq_events.get('label', '–')}",
        f"  Canary breadth:  {_fmt(_sq_canary.get('composite'), '.1f')}/100"
        f"   breadth {_fmt(_sq_canary.get('breadth_pct'), '.1f')}%"
        f"   1m avg {_fmt(_sq_canary.get('momentum_avg'), '+.2f')}%"
        f"   vol surge {_fmt(_sq_canary.get('vol_surge'), '.2f')}x",
        "",
        f"Missing signals: {missing_str}",
        "",
        "---",
        "",
        "## 2. ARCHITECTURE OVERVIEW",
        "",
        "Entry point: app.py — Streamlit sidebar with 8 modules, password auth via APP_PASSWORD env var.",
        "State bus: Streamlit session_state (~134 keys, persisted to GitHub Gist + data/signals_cache.json).",
        "",
        "MODULE MAP",
        "  Module 0: Risk Regime         modules/risk_regime.py  (z-score regime + tactical + GEX dealer positioning)",
        "  QIR:      Quick Intel Run     modules/quick_run.py    (orchestrator: regime + debate + lean tracker + HMM)",
        "  Module 1: Narrative Discovery modules/narrative_discovery.py  (AI-grouped ticker clusters)",
        "  Module 2: Narrative Pulse     modules/narrative_pulse.py      (sentiment tracking)",
        "  Module 3: EDGAR Scanner       modules/edgar_scanner.py        (SEC filing search)",
        "  Module 4: Institutional       modules/institutional.py        (13F holdings)",
        "  Module 5: Insider / Congress  modules/insider_congress.py     (Form 4 + Congress trades)",
        "  Module 6: Options Activity    modules/options_activity.py     (flow analysis + unusual activity)",
        "  Module 7: Valuation           modules/valuation.py            (AI rating + DCF + Kelly sizing)",
        "  –:        Macro Scorecard     modules/macro_scorecard.py      (fear composite + signal scorecard)",
        "  –:        Forecast Accuracy   modules/forecast_accuracy.py    (SPY trade tracking + ATR evaluation)",
        "  –:        Trade Journal       modules/trade_journal.py        (trade log + P&L + debate verdicts)",
        "  –:        Portfolio Intel     modules/tail_risk_studio.py     (risk snapshot + factor analysis)",
        "  –:        Export Hub          modules/export_hub.py           (REGARD + Co-Engineer + JSON exports)",
        "",
        "SERVICES",
        "  market_data.py      — yfinance + FRED batch fetch, caching, AssetSnapshot, GEX profile",
        "  claude_client.py    — Claude AI + Groq LLM integration + adversarial debate",
        "  scoring.py          — 0-100 per-ticker composite score (6 dimensions)",
        "  sec_client.py       — SEC EDGAR rate-limited API (10 req/sec)",
        "  fed_forecaster.py   — FOMC scenario forecasting, rate path probabilities",
        "  signals_cache.py    — _SIGNAL_KEYS registry (~140 keys), GitHub Gist persistence",
        "  hmm_regime.py       — GaussianHMM regime detector (6 states, Kelly multiplier)",
        "  forecast_tracker.py — SPY signal log, ATR trailing stop, auto-evaluation",
        "  portfolio_sizing.py — Kelly criterion, 4-factor OLS sensitivity, position sizing",
        "  signal_quantifier.py— z-scored stress, whale, events, canary composites",
        "",
        "UTILS",
        "  signal_block.py     — build_macro_block() + build_ticker_block() ground-truth injectors",
        "  debate_record.py    — Commander Wincyl court record persistence (SQLite)",
        "  options_history.py  — options flow history tracking",
        "",
        "---",
        "",
        "## 3. DATA FLOW: QIR → EVERYTHING",
        "",
        "QIR (quick_run.py) is the data loader. It runs concurrent rounds of tasks:",
        "",
        "  Round 1 (parallel, ThreadPoolExecutor):",
        "    → run_quick_regime()        writes: _regime_context, _regime_raw_signals, _tactical_context",
        "    → run_quick_options_flow()  writes: _options_flow_context, _unusual_activity_sentiment",
        "    → run_quick_rate_path()     writes: _dominant_rate_path, _rate_path_probs, _fed_funds_rate",
        "    → run_fed_plays()           writes: _fed_plays_result",
        "    → run_current_events()      writes: _current_events_digest",
        "    → run_doom_briefing()       writes: _doom_briefing",
        "",
        "  Round 2 (after regime resolves):",
        "    → run_quick_sector_regime() writes: _sector_regime_digest",
        "",
        "  Round 3 (ad-hoc):",
        "    → generate_macro_synopsis() writes: _macro_synopsis",
        "    → adversarial_debate()      writes: _adversarial_debate",
        "",
        "  Post-QIR (automatic):",
        "    → lean_tracker logging      writes: data/lean_tracker.json (daily lean + domain scores)",
        "    → lean_tracker backfill     stamps: fwd_5d_spy_return, fwd_20d_spy_return on old entries",
        "    → HMM backfill              stamps: fwd_5d_spy_return on old HMM state history entries",
        "    → forecast_tracker.evaluate_pending()  resolves open SPY trades vs ATR stops",
        "",
        "  Downstream readers:",
        "    narrative_discovery.py  reads: _regime_context, _tactical_context, _dominant_rate_path,",
        "                                   _sector_regime_digest, _fed_funds_rate",
        "    valuation.py            reads: ALL of the above + _options_flow_context, _doom_briefing,",
        "                                   _whale_summary, _fed_plays_result, _rp_plays_result,",
        "                                   _options_sentiment, _unusual_activity_sentiment,",
        "                                   _institutional_bias, _insider_net_flow, _congress_bias,",
        "                                   _fear_greed, _aaii_sentiment, _vix_curve, _portfolio_risk_snapshot",
        "    portfolio intel         reads: _regime_context, _dominant_rate_path, _sector_regime_digest,",
        "                                   _portfolio_risk_snapshot, _macro_synopsis",
        "",
        "AI GROUNDING PATTERN (signal_block.py):",
        "  Every AI call injects: build_macro_block() + build_ticker_block(ticker)",
        "  These pull raw z-scores and numeric values — not AI interpretations — to prevent",
        "  \"telephone game\" compounding errors across multi-step AI chains.",
        "",
        "---",
        "",
        "## 4. HOW NARRATIVE DISCOVERY WORKS (modules/narrative_discovery.py)",
        "",
        "Input:  User enters a ticker or narrative theme (or runs auto-trending mode)",
        "Output: AI-grouped ticker clusters with macro-aligned play ideas",
        "",
        "Step-by-step:",
        "  1. _get_macro_context_for_plays() — reads _regime_context + _tactical_context + _dominant_rate_path",
        "     → builds a macro summary string injected into every AI call in this module",
        "  2. Auto mode: fetches trending tickers from StockTwits + unusual options activity",
        "  3. claude_client.group_tickers_by_narrative(tickers) — AI groups tickers by theme",
        "  4. For each group: generates regime-aligned play (bull/bear/neutral) using macro context",
        "  5. Conviction stars (1–5) derived from alignment of: regime direction × tactical score × options sentiment",
        "  6. Rate path overlay: if _dominant_rate_path loaded, overlays rate sensitivity on each play",
        "",
        "Key session keys written:",
        "  _plays_result, _trending_narratives, _auto_trending_groups, _sector_regime_digest (consumed from QIR)",
        "",
        "---",
        "",
        "## 5. HOW VALUATION WORKS (modules/valuation.py)",
        "",
        "Input:  Single ticker + AI model selection",
        "Output: Buy/Hold/Sell rating + DCF + Kelly sizing + signal transparency",
        "",
        "Step-by-step:",
        "  1. _collect_signals(ticker) — gathers 15+ signal dimensions from session_state + live fetch:",
        "       Technicals (SMA/RSI/momentum), Fundamentals (P/E, PEG, growth, margins),",
        "       Insider flow (SEC Form 4), Institutional (13F changes), Congress trades,",
        "       Options sentiment (P/C ratio, gamma zone), Short interest,",
        "       Macro regime score, Rate path sensitivity, Sector rotation signal",
        "  2. build_macro_block() + build_ticker_block(ticker) — raw-number ground truth injected into prompt",
        "  3. Claude/Groq generates rating + thesis + entry/exit levels + risk factors",
        "  4. _compute_dcf(ticker) — DCF engine with 3 scenarios (base/bull/bear):",
        "       Fetches: revenue growth, margins, WACC, terminal growth from fundamentals",
        "       Produces: intrinsic value range, upside/downside %, scenario sensitivity table",
        "  5. _render_kelly() — Kelly criterion position sizing:",
        "       Inputs: AI win probability, expected gain/loss, account size",
        "       Output: full Kelly %, half Kelly %, suggested position size in $",
        "  6. Signal fingerprinting (signal_block.get_ticker_fingerprint) — MD5 hash of all inputs",
        "       If fingerprint unchanged since last run → reuse cached verdict (avoids redundant AI calls)",
        "",
        "Key session keys written: _val_result_{ticker}, _dcf_result_{ticker}",
        "",
        "---",
        "",
        "## 6. HOW PORTFOLIO INTELLIGENCE WORKS",
        "",
        "Input:  Open positions from trade journal + all QIR macro signals",
        "Output: Portfolio-level risk snapshot, factor exposures, regime-aligned action items",
        "",
        "Step-by-step:",
        "  1. Reads open positions from data/plays_log.json",
        "  2. Fetches live prices for all held tickers (market_data.fetch_batch_safe)",
        "  3. Correlates each position against regime quadrant:",
        "       Goldilocks → tech/growth overweight OK",
        "       Stagflation → commodities/defensives, reduce growth exposure",
        "  4. _portfolio_risk_snapshot: VaR, beta-adjusted exposure, correlation concentration",
        "  5. Factor analysis: maps positions against macro factors (duration, inflation, credit, growth)",
        "  6. AI action items: regime-specific position adjustments via Claude prompt",
        "     Prompt includes: build_macro_block() + full position table + factor exposures",
        "",
        "Key session keys read/written: _portfolio_risk_snapshot, _portfolio_analysis, _factor_analysis",
        "",
        "---",
        "",
        "## 7. KEY ENGINEERING PATTERNS",
        "",
        "Caching:      @st.cache_data(ttl=3600) for live data; ttl=86400 for historical FRED",
        "Concurrency:  ThreadPoolExecutor(max_workers=5) for SEC + yfinance batch fetches",
        "Rate limits:  SEC EDGAR: 10 req/sec via _rate_limit() in sec_client.py",
        "AI models:    Groq (fast/cheap for synthesis) + Claude (deep reasoning for valuation/debate)",
        "Persistence:  GitHub Gist (survives Streamlit Cloud redeploys) + local JSON fallback",
        "Fingerprinting: MD5 hash of signal inputs → skip AI if inputs unchanged (signals_cache.py)",
        "Adversarial:  3-agent debate: Dr. Doomburger 🐻 vs Sir Fukyerputs 🐂, judged by Commander Wincyl ⚖️",
        "              Verdict + confidence stored in _adversarial_debate, logged to data/debate_record.db",
        "",
        "---",
        "",
        "## 8. KNOWN GAPS & HONEST CONSTRAINTS",
        "",
        "- Session state coupling: all modules share one global dict — no isolation between runs",
        "- No unit tests: all logic is tested manually via the UI",
        "- FRED data latency: some series update monthly; stale data silently lowers confidence scores",
        "- Options data: market-hours only; after-hours runs get data_unavailable flag",
        "- AI output contracts: no JSON schema validation on LLM responses — rely on try/except fallbacks",
        "- Streamlit rerun overhead: heavy QIR runs can feel slow due to sequential st.session_state writes",
        "- GitHub Gist as DB: single-file JSON is fine for now but will hit size limits with more signal keys",
        "",
        "---",
        "",
        "## 9. STARTER PROMPTS FOR AI",
        "",
        "Use one of these to start your conversation:",
        "",
        "ARCHITECTURE REVIEW:",
        "\"You are a principal engineer reviewing a Streamlit investment intelligence app.",
        "Here is the full architecture brief. Identify the top 5 architectural risks and",
        "suggest concrete improvements with code-level guidance.\"",
        "",
        "FEATURE ENGINEERING:",
        "\"Based on this app brief, suggest 3 high-impact features I could add to improve",
        "the Discovery → Valuation → Portfolio flow. For each: describe the feature,",
        "what data it needs, which files to modify, and rough implementation complexity.\"",
        "",
        "UPGRADE ROADMAP:",
        "\"Review this app and produce a prioritized 2-week upgrade roadmap. Focus on:",
        "signal quality, UI clarity, AI prompt improvements, and engineering robustness.",
        "Be specific — name files, functions, and patterns to change.\"",
        "",
        "CODE REVIEW:",
        "\"Given this architecture, what are the most likely sources of silent bugs or",
        "data quality issues? Focus on the session state data flow and AI grounding patterns.\"",
        "",
        "---",
        "",
        f"Brief generated by Regarded Terminals | {now_str}",
    ]
    return "\n".join(lines)


def _build_json_export(open_trades: list) -> str:
    from services.signals_cache import _SIGNAL_KEYS
    payload = {}
    for key in _SIGNAL_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            payload[key] = _serialize(val)
    # Add open positions
    payload["open_positions"] = open_trades
    payload["_export_ts"] = datetime.now().isoformat()
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Readiness panel data
# ---------------------------------------------------------------------------

_READINESS_ITEMS = [
    ("Regime + Rate Path",    "_regime_context_ts",        "Run Risk Regime"),
    ("AI Regime Plays",       "_rp_plays_last_tier",       "Run Regime Plays in Risk Regime"),
    ("Fed Rate-Path Plays",   "_fed_plays_result_ts",      "Run Fed Forecaster Plays"),
    ("Policy Transmission",   None,                        "Run Policy Transmission in Risk Regime"),
    ("Doom Briefing",         "_doom_briefing_ts",         "Run Stress Signals"),
    ("Whale Summary",         "_whale_summary_ts",         "Run Whale Movement"),
    ("Black Swans",           "_custom_swans_ts",          "Run Black Swans in Risk Regime"),
    ("Discovery Plays",       None,                        "Run Discovery Plays in Narrative Discovery"),
    ("Portfolio Positions",   None,                        "Add trades in My Regarded Portfolio"),
    ("Current Events",        "_current_events_digest_ts", "Generate News Digest in Risk Regime"),
    ("Factor AI Analysis",    "_factor_analysis_ts",       "Run Analyze Factors in My Regarded Portfolio"),
    ("Pre-Trade Verdict",     None,                        "Run Pre-Trade Simulator + Get AI Verdict"),
]

_STALE_THRESHOLD_H = 6
_WARN_THRESHOLD_H = 2


def _check_item(label: str, ts_key, run_hint: str, open_trades: list):
    """Return (icon, label, age_text, hint) for a readiness item."""
    # Special cases with non-timestamp presence checks
    if label == "Regime + Rate Path":
        has = bool(st.session_state.get("_regime_context"))
        ts = st.session_state.get("_regime_context_ts")
    elif label == "AI Regime Plays":
        has = bool(st.session_state.get("_rp_plays_result"))
        ts = None  # no dedicated ts key
    elif label == "Policy Transmission":
        has = bool(st.session_state.get("_chain_narration"))
        ts = None
    elif label == "Discovery Plays":
        has = bool(st.session_state.get("_plays_result"))
        ts = None
    elif label == "Portfolio Positions":
        has = bool(open_trades)
        ts = None
    elif label == "Pre-Trade Verdict":
        has = bool(st.session_state.get("_sim_verdict"))
        ts = None
    else:
        val = st.session_state.get(ts_key) if ts_key else None
        # ts_key may store datetime OR just a non-None value (e.g., tier string)
        if isinstance(val, datetime):
            has = True
            ts = val
        else:
            # For non-ts keys (like _rp_plays_last_tier), check the actual result key
            result_key_map = {
                "_rp_plays_last_tier": "_rp_plays_result",
                "_fed_plays_result_ts": "_fed_plays_result",
                "_doom_briefing_ts": "_doom_briefing",
                "_whale_summary_ts": "_whale_summary",
                "_custom_swans_ts": "_custom_swans",
                "_factor_analysis_ts": "_factor_analysis",
                "_current_events_digest_ts": "_current_events_digest",
            }
            rk = result_key_map.get(ts_key)
            has = bool(st.session_state.get(rk)) if rk else bool(val)
            ts = val if isinstance(val, datetime) else None

    if not has:
        return "✗", label, "missing", run_hint

    age_text = _age_str(ts) if ts else "populated"
    if ts:
        delta_h = (datetime.now() - ts).total_seconds() / 3600
        if delta_h > _STALE_THRESHOLD_H:
            return "⚠", label, f"{age_text} — consider refreshing", run_hint
        if delta_h > _WARN_THRESHOLD_H:
            return "⚠", label, age_text, run_hint

    return "✅", label, age_text, run_hint


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:20px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.1em;margin-bottom:4px;">EXPORT HUB</div>'
        f'<div style="height:2px;background:linear-gradient(90deg,'
        f'{COLORS["bloomberg_orange"]},{COLORS["bloomberg_orange"]}44,transparent);'
        f'border-radius:1px;margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )
    st.caption("Download your macro intelligence as a briefing document to paste into Claude.ai, ChatGPT, or Gemini.")

    # Load positions
    all_trades = load_journal()
    open_trades = [t for t in all_trades if t.get("status") == "open"]

    # --- Section A: Readiness Panel ---
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">PRE-EXPORT CHECKLIST</div>',
        unsafe_allow_html=True,
    )

    statuses = []
    for label, ts_key, hint in _READINESS_ITEMS:
        statuses.append(_check_item(label, ts_key, hint, open_trades))

    ready_count = sum(1 for icon, *_ in statuses if icon == "✅")
    warn_count = sum(1 for icon, *_ in statuses if icon == "⚠")
    total_count = len(statuses)

    score_color = COLORS["positive"] if ready_count == total_count else (
        "#f59e0b" if ready_count >= total_count // 2 else COLORS["negative"]
    )

    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;padding:14px 18px;margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
        f'<span style="font-size:12px;color:#888;letter-spacing:0.06em;">SIGNAL READINESS</span>'
        f'<span style="font-size:16px;font-weight:700;color:{score_color};">'
        f'{ready_count + warn_count} / {total_count} signals available</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for icon, label, age_text, hint in statuses:
        icon_color = COLORS["positive"] if icon == "✅" else ("#f59e0b" if icon == "⚠" else "#ef4444")
        age_color = "#888" if icon != "✗" else "#ef4444"
        hint_html = (
            f' <span style="color:#555;font-size:10px;">→ {hint}</span>'
            if icon == "✗" else ""
        )
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;'
            f'border-bottom:1px solid {COLORS["border"]}22;">'
            f'<span style="color:{icon_color};width:18px;flex-shrink:0;">{icon}</span>'
            f'<span style="flex:1;font-size:12px;color:#ccc;">{label}</span>'
            f'<span style="font-size:11px;color:{age_color};">{age_text}{hint_html}</span>'
            f'</div>'
        )

    st.markdown(
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )

    if ready_count < total_count:
        st.caption("💡 For best results, populate missing signals before exporting. Run them in the order shown above.")

    # ── helper: render download / telegram / stats row ──────────────────────
    def _export_action_row(data: str, filename: str, mime: str, dl_key: str, tg_key: str, tg_caption: str):
        _a1, _a2, _a3 = st.columns([1, 1, 2])
        with _a1:
            st.download_button(
                f"💾 Download {filename.rsplit('.', 1)[-1].upper()}",
                data,
                file_name=filename,
                mime=mime,
                key=dl_key,
                use_container_width=True,
            )
        with _a2:
            try:
                from services.telegram_client import is_configured as _tg_chk, send_document as _tg_snd
                if _tg_chk():
                    if st.button("📲 Send to Telegram", key=tg_key, use_container_width=True):
                        with st.spinner("Sending…"):
                            _sent = _tg_snd(filename, data, caption=tg_caption)
                        st.success("✅ Sent") if _sent else st.error("❌ Failed — check bot token")
                else:
                    st.caption("📲 Telegram not configured")
            except ImportError:
                st.caption("📲 Telegram not configured")
        with _a3:
            st.markdown(
                f'<div style="padding:7px 0 0 4px;font-size:11px;color:#475569;font-family:\'JetBrains Mono\',monospace;">'
                f'{data.count(chr(10)):,} lines &nbsp;·&nbsp; {len(data):,} chars &nbsp;·&nbsp; ~{len(data)//4:,} tokens</div>',
                unsafe_allow_html=True,
            )

    def _export_card_header(label: str, subtitle: str):
        st.markdown(
            f'<div style="margin:18px 0 10px 0;">'
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.1em;margin-bottom:3px;">{label}</div>'
            f'<div style="font-size:12px;color:#64748b;">{subtitle}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _export_card_tip(text: str):
        st.markdown(
            f'<div style="margin:10px 0 4px 0;font-size:11px;color:#334155;'
            f'border-left:2px solid #1e293b;padding-left:8px;">{text}</div>',
            unsafe_allow_html=True,
        )

    # ── Section B: Macro Briefing ─────────────────────────────────────────────
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 0 0;"></div>', unsafe_allow_html=True)

    _export_card_header(
        "📊 MACRO BRIEFING EXPORT",
        "Live signals, regime, rate path &amp; portfolio. Use when making trade decisions.",
    )

    _fmt = st.radio(
        "Format",
        ["📝 Markdown (AI chat)", "🔧 JSON (raw data)"],
        horizontal=True,
        key="export_fmt",
        label_visibility="collapsed",
    )

    if st.button("⬇ Generate Macro Briefing", type="primary", key="export_generate", use_container_width=True):
        with st.spinner("Building…"):
            if "Markdown" in _fmt:
                content = _build_markdown_export(open_trades)
                filename = f"macro_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                mime = "text/markdown"
            else:
                content = _build_json_export(open_trades)
                filename = f"macro_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                mime = "application/json"
        st.session_state["_export_content"] = content
        st.session_state["_export_filename"] = filename
        st.session_state["_export_mime"] = mime

    content = st.session_state.get("_export_content")
    if content:
        filename = st.session_state.get("_export_filename", "export.md")
        mime = st.session_state.get("_export_mime", "text/markdown")
        _export_action_row(
            content, filename, mime,
            dl_key="export_download",
            tg_key="export_telegram",
            tg_caption=f"📊 Macro Briefing — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        with st.expander("Preview", expanded=False):
            st.code("\n".join(content.split("\n")[:60]), language="markdown")

    _export_card_tip("Trade decisions — ask: <i>\"What should I do with my portfolio right now?\"</i>")

    # ── Section D: AI Co-Engineer Brief ──────────────────────────────────────
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 0 0;"></div>', unsafe_allow_html=True)

    _export_card_header(
        "🤖 AI CO-ENGINEER BRIEF",
        "Architecture, data flow &amp; module internals. Use when improving the tool.",
    )

    if st.button("⬇ Generate Co-Engineer Brief", type="primary", key="export_coengine_gen", use_container_width=True):
        with st.spinner("Building…"):
            _coengine_content = _build_coengine_export()
        st.session_state["_coengine_content"] = _coengine_content

    _coengine = st.session_state.get("_coengine_content")
    if _coengine:
        _coengine_filename = f"coengine_brief_{datetime.now().strftime('%Y%m%d')}.md"
        _export_action_row(
            _coengine, _coengine_filename, "text/markdown",
            dl_key="export_coengine_dl",
            tg_key="export_coengine_tg",
            tg_caption=f"🤖 Co-Engineer Brief — {datetime.now().strftime('%Y-%m-%d')}",
        )
        with st.expander("Preview", expanded=False):
            st.code(_coengine, language="markdown")

    _export_card_tip("Tool improvements — ask: <i>\"What should I add, fix, or refactor?\"</i>")

    # ── Section E: Regard Export ──────────────────────────────────────────────
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 0 0;"></div>', unsafe_allow_html=True)

    _export_card_header(
        "📐 REGARD EXPORT",
        "Full technical spec — every formula, every weight, every data source, with live values.",
    )
    st.caption("Hand this to any quant or engineer. 13 sections covering data collection, signal math, conviction formula, Kelly criterion (signal-aligned + Triple Kelly / Bimodal Sizing), HMM Brain State, AI debate architecture, and calibration roadmap.")

    if st.button("⬇ Generate Regard Export", type="primary", key="export_regard_gen", use_container_width=True):
        with st.spinner("Building full technical spec…"):
            _regard_content = _build_regard_export()
        st.session_state["_regard_content"] = _regard_content

    _regard = st.session_state.get("_regard_content")
    if _regard:
        _regard_filename = f"regard_export_{datetime.now().strftime('%Y%m%d')}.md"
        _export_action_row(
            _regard, _regard_filename, "text/markdown",
            dl_key="export_regard_dl",
            tg_key="export_regard_tg",
            tg_caption=f"📐 Regard Export — {datetime.now().strftime('%Y-%m-%d')}",
        )
        with st.expander("Preview", expanded=False):
            st.code(_regard, language="markdown")

    _export_card_tip("Signal calibration — ask: <i>\"Based on this spec, which weights are most likely wrong and why?\"</i>")

    # --- Section F: Suggested prompts ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">READY-TO-USE AI PROMPTS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Copy any of these and paste alongside your exported briefing:")

    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "")
    is_risk_off = "Risk-Off" in regime
    has_portfolio = bool(open_trades)
    has_doom = bool(st.session_state.get("_doom_briefing"))
    has_swans = bool(st.session_state.get("_custom_swans"))

    prompts = [
        ("Portfolio Risk", "Given this macro regime and my portfolio positions, what are the biggest risks I should be aware of and what actions should I take?"),
        ("Black Swan Exposure", "Which of my portfolio positions are most exposed to the black swan tail risks listed? Rank them by vulnerability."),
        ("Rate Path Trade", "Given the Fed rate path probabilities, which sectors and assets should I overweight or underweight?"),
        ("Regime Rotation", "Based on the macro regime and signal summary, suggest a sector rotation strategy with specific tickers."),
        ("Full Briefing", "Here is my current macro intelligence briefing. Please summarize the top 3 themes, highlight the most actionable signals, and give me a prioritized action plan for my portfolio."),
    ]

    if is_risk_off:
        prompts.insert(0, ("Risk-Off Hedge", "The macro regime is Risk-Off. Which of my long positions should I hedge or reduce first, and what instruments would you use?"))
    if has_doom and has_swans:
        prompts.append(("Stress Test", "Cross-reference the doom briefing with the black swan tail risks. Which scenario poses the greatest threat to my specific positions?"))
    for title, prompt in prompts:
        with st.expander(f"📋 {title}", expanded=False):
            st.code(prompt, language=None)
