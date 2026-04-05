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
        "FED / RATE PATH",
        f"  Current Fed Funds: {fed_rate}%",
        f"  Dominant scenario: {scenario} ({prob_pct}%)",
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
        "  Module 0: Risk Regime         modules/risk_regime.py",
        "  QIR:      Quick Intel Run     modules/quick_run.py",
        "  Module 1: Narrative Discovery modules/narrative_discovery.py",
        "  Module 6: Options Activity    modules/options_activity.py",
        "  Module 7: Valuation           modules/valuation.py",
        "  –:        Portfolio Intel     modules/tail_risk_studio.py (+ related)",
        "  –:        Export Hub          modules/export_hub.py",
        "",
        "SERVICES",
        "  market_data.py      — yfinance + FRED batch fetch, caching, AssetSnapshot dataclass",
        "  claude_client.py    — Claude AI + Groq LLM integration",
        "  scoring.py          — 0-100 per-ticker composite score (6 dimensions)",
        "  sec_client.py       — SEC EDGAR rate-limited API (10 req/sec)",
        "  fed_forecaster.py   — FOMC scenario forecasting, rate path probabilities",
        "  signals_cache.py    — _SIGNAL_KEYS registry, GitHub Gist persistence",
        "",
        "UTILS",
        "  signal_block.py     — build_macro_block() + build_ticker_block() ground-truth injectors",
        "  debate_record.py    — Judge Judy court record persistence (SQLite)",
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
        "Adversarial:  3-agent debate: Sir Doomburger 🐻 vs Sir Fukyerputs 🐂, judged by Judge Judy ⚖️",
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

    # --- Section E: Suggested prompts ---
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
