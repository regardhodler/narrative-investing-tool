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

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

    # --- Section B: Format selector + Generate ---
    _fmt = st.radio(
        "Format",
        ["📝 Text / Markdown (for AI chat)", "🔧 JSON (raw data)"],
        horizontal=True,
        key="export_fmt",
    )
    st.caption("💡 Markdown recommended — paste directly into Claude.ai, ChatGPT, or Gemini for instant macro analysis.")

    if st.button("⬇ Generate Export", type="primary", key="export_generate"):
        with st.spinner("Building export document..."):
            if "Markdown" in _fmt:
                content = _build_markdown_export(open_trades)
                filename = f"macro_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                mime = "text/plain"
            else:
                content = _build_json_export(open_trades)
                filename = f"macro_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                mime = "application/json"

        st.session_state["_export_content"] = content
        st.session_state["_export_filename"] = filename
        st.session_state["_export_mime"] = mime

    # --- Section C: Download + Preview ---
    content = st.session_state.get("_export_content")
    if content:
        filename = st.session_state.get("_export_filename", "export.txt")
        mime = st.session_state.get("_export_mime", "text/markdown")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.download_button(
                label=f"💾 Download {filename.split('.')[-1].upper()}",
                data=content,
                file_name=filename,
                mime=mime,
                type="primary",
                key="export_download",
            )
        with col2:
            char_count = len(content)
            line_count = content.count("\n")
            st.markdown(
                f'<div style="padding:8px 0;font-size:12px;color:#888;">'
                f'{line_count} lines · {char_count:,} characters · {char_count // 4:,} est. tokens</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Preview (first 60 lines)", expanded=False):
            preview_lines = content.split("\n")[:60]
            st.code("\n".join(preview_lines), language="markdown")

        st.markdown(
            f'<div style="background:{COLORS["surface"]};border-left:3px solid {COLORS["bloomberg_orange"]};'
            f'padding:10px 14px;border-radius:0 4px 4px 0;margin-top:8px;font-size:12px;color:#bbb;">'
            f'💡 <b>Tip:</b> Paste this into Claude.ai and ask:<br>'
            f'<i>"Analyze my portfolio given this macro environment and suggest my top 3 priority actions."</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Section D: Suggested prompts ---
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
