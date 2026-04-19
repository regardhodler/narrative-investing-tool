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

def _section_qir_snapshot() -> str:
    """Export current QIR pattern, conviction, Kelly, and Bottom Watch status."""
    pattern = st.session_state.get("_qir_pattern")
    conviction = st.session_state.get("_conviction_score")
    kly = st.session_state.get("_kly") or {}
    kelly_pct = kly.get("kelly_half_pct") if kly else st.session_state.get("_kelly_half_pct")
    bottom_signals = st.session_state.get("_bottom_watch_signals")
    hmm_ctx = st.session_state.get("_hmm_context")
    ll_crisis = st.session_state.get("_ll_anchored_crisis") or {}
    shadow = st.session_state.get("_shadow_state_obj")

    if all(v is None for v in [pattern, conviction, kelly_pct, bottom_signals, hmm_ctx, ll_crisis, shadow]):
        return ""

    lines = ["## QIR SNAPSHOT"]

    if pattern is not None:
        lines.append(f"- **Signal Pattern:** {pattern}")

    if conviction is not None:
        if conviction >= 75:
            conv_label = "HIGH CONVICTION — all signal domains agree"
        elif conviction >= 55:
            conv_label = "MODERATE CONVICTION — majority aligned"
        elif conviction >= 40:
            conv_label = "LOW CONVICTION — signals mixed"
        else:
            conv_label = "VERY LOW CONVICTION — signals conflict or absent"
        lines.append(f"- **Conviction × Uncertainty:** {conviction}/100 ({conv_label})")

    if kelly_pct is not None:
        viable = kelly_pct >= 0.1
        lines.append(f"- **Long Term Kelly:** {kelly_pct:.2f}% {'(viable)' if viable else '(below threshold — negative expectancy)'}")

    if bottom_signals is not None:
        fired = [k for k, v in bottom_signals.items() if v] if isinstance(bottom_signals, dict) else []
        total_sigs = len(bottom_signals) if isinstance(bottom_signals, dict) else 4
        lines.append(f"- **Market Bottom Watch:** {len(fired)}/{total_sigs} signals live ({', '.join(fired) if fired else 'none'})")
    else:
        lines.append("- **Market Bottom Watch:** dormant (CI% < 22% or not yet computed)")

    if hmm_ctx is not None and isinstance(hmm_ctx, dict):
        state = hmm_ctx.get("state_label", "—")
        conf = hmm_ctx.get("confidence", 0)
        ll_z = hmm_ctx.get("ll_zscore", 0)
        ci_pct = abs(ll_z) / 0.467 * 100 if ll_z else 0
        lines.append(f"- **HMM State:** {state} (conf={conf:.2f}, ll_z={ll_z:.3f}, CI%={ci_pct:.1f}%)")

    # Prefer LL-anchored crisis block + shadow brain when available (new QIR wiring)
    if ll_crisis:
        _hmm_lbl = ll_crisis.get("hmm_state", "—")
        _hmm_llz = float(ll_crisis.get("ll_zscore", 0.0) or 0.0)
        _hmm_ci = ll_crisis.get("ci_pct")
        if _hmm_ci is None:
            _hmm_ci = abs(_hmm_llz) / 0.467 * 100 if _hmm_llz < 0 else 0.0
        lines.append(f"- **HMM Learn Brain:** {_hmm_lbl} (ll_z={_hmm_llz:+.2f}, CI%={float(_hmm_ci):.1f}%)")

    if shadow is not None:
        _s_lbl = getattr(shadow, "state_label", "—")
        _s_llz = float(getattr(shadow, "ll_zscore", 0.0) or 0.0)
        _s_ci = float(getattr(shadow, "ci_pct", 0.0) or 0.0)
        _s_cr = float(getattr(shadow, "crash_prob_10pct", 0.0) or 0.0)
        lines.append(
            f"- **HMM Shadow Brain:** {_s_lbl} (ll_z={_s_llz:+.2f}, CI%={_s_ci:.1f}%, crash>10%={_s_cr*100:.0f}%)"
        )

    _dbt_payload = st.session_state.get("_qir_debate_signals_text")
    if _dbt_payload:
        lines.append(f"- **QIR Debate Payload:** loaded ({len(_dbt_payload):,} chars)")

    lines.append("")
    return "\n".join(lines)


def _section_qir_debate_payload() -> str:
    """Export the full QIR debate payload assembled in Quick Run (includes HMM + shadow lines)."""
    payload = st.session_state.get("_qir_debate_signals_text")
    if not payload:
        return ""
    return (
        "## QIR DEBATE PAYLOAD\n"
        "(This is the exact text fed to Adversarial Debate when available.)\n\n"
        f"{payload}\n"
    )


def _build_markdown_export(open_trades: list) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        _section_executive_summary(open_trades),
        _section_current_events(),
        _section_regime(),
        _section_qir_snapshot(),
        _section_qir_debate_payload(),
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


def _build_pipeline_export() -> str:
    """Build a deep, runtime-aware pipeline map of the app for AI review."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "unknown")
    quadrant = ctx.get("quadrant", "unknown")
    score_mode = st.session_state.get("_rr_score_mode", "normal")

    # Live HMM state
    try:
        from services.hmm_regime import load_current_hmm_state, load_hmm_brain
        _hmm_live = load_current_hmm_state()
        _hmm_brain = load_hmm_brain()
        hmm_state_str = (
            f"{_hmm_live.state_label} (conf={_hmm_live.confidence:.2f}, "
            f"ll_z={getattr(_hmm_live,'ll_zscore',0):.3f})"
        ) if _hmm_live else "unavailable"
        hmm_brain_str = (
            f"trained={_hmm_brain.trained_at[:10]}, "
            f"lookback={_hmm_brain.lookback_years}yr, "
            f"n_states={_hmm_brain.n_states}, "
            f"ll_baseline_mean={_hmm_brain.ll_baseline_mean:.4f}, "
            f"ll_baseline_std={_hmm_brain.ll_baseline_std:.4f}"
        ) if _hmm_brain else "no brain found"
    except Exception:
        hmm_state_str = "unavailable"
        hmm_brain_str = "unavailable"

    data_sources = [
        "yfinance - prices, OHLCV, options chain snapshots",
        "FRED - 9 series: HY spreads, IG spreads, yield curve (10y2y, 10y3m), DGS10, DGS2, DFII10, NFCI, ICSA",
        "SEC EDGAR - filings, insider Form 4, 13F holdings",
        "Google Trends - narrative momentum",
        "Congress trading feed - disclosed congressional transactions",
    ]

    service_layer = [
        "services/market_data.py - batch fetch, FRED wrappers, AssetSnapshot dataclass, z-scores",
        "services/hmm_regime.py - GaussianHMM (6-state) trained on FRED+VIX. Brain in data/hmm_brain.json. lookback_years stored in brain and must match live scoring exactly",
        "services/sec_client.py - EDGAR requests, CIK/ticker mapping, rate limiting (10 req/s)",
        "services/claude_client.py - all LLM synthesis, debates, valuation, plays",
        "services/backtest_engine.py - backtest orchestration",
        "services/signals_cache.py - persist/reload session signals",
        "services/forecast_tracker.py - call logging and evaluation",
        "services/telegram_client.py - outbound alert and report delivery",
        "services/wyckoff_engine.py - Wyckoff phase detection (Accumulation/Distribution/etc)",
        "services/elliott_wave_engine.py - Elliott Wave pattern detection",
        "services/scoring.py - conviction score computation",
        "services/indicators.py - technical indicators",
        "services/portfolio_sizing.py - Kelly criterion position sizing",
        "services/sector_rotation.py - sector rotation signals",
        "services/turning_point.py - market turning point detection",
        "services/whale_screener.py - large buyer/seller detection",
        "services/qir_history.py - QIR run history persistence",
        "services/tactical_history.py - tactical score history",
    ]

    module_layer = [
        ("modules/quick_run.py", "QIR master dashboard: HMM Brain State card, LL-anchored CI% crisis detection, Market Bottom Watch (always visible; dormant CI<22 / live CI≥22), Conviction×Uncertainty card, Long Term Kelly card, Buy/Short Setup cards (LONG KELLY/SHORT KELLY badges), Lean Accuracy block, GEX + signal breakdown. 27-pattern classifier (_classify_signals) maps regime×tactical×options to named signal patterns with label/color/instruments/entry rules/tips. Options instruments removed from Buy/Short setup tiers (flow still affects Kelly via alignment multiplier)."),
        ("modules/risk_regime.py", "cross-asset macro classifier + quadrant + tactical overlay (17+ z-scored signals)"),
        ("modules/backtesting.py", "historical snapshot viewer with CI% LL-anchored block, Wyckoff pill, regime overlays"),
        ("modules/fed_forecaster.py", "Fed path probabilities and implications"),
        ("modules/current_events.py", "headline/inbox context feed"),
        ("modules/stress_signals.py", "doom and stress framing"),
        ("modules/whale_buyers.py", "whale movement intelligence"),
        ("modules/narrative_discovery.py", "cross-signal discovery and narrative mapping"),
        ("modules/options_activity.py", "options flow and unusual activity signal"),
        ("modules/valuation.py", "AI valuation and DCF overlays"),
        ("modules/trade_journal.py", "portfolio intelligence, factor lens, debate display"),
        ("modules/signal_audit.py", "signal quality/consistency audit"),
        ("modules/forecast_accuracy.py", "signal outcome scoring + ATR outcomes"),
        ("modules/export_hub.py", "briefing and pipeline exports — this file"),
        ("modules/elliott_wave.py", "Elliott Wave analysis UI"),
        ("modules/wyckoff.py", "Wyckoff phase analysis UI"),
        ("modules/macro_scorecard.py", "macro signal scorecard"),
        ("modules/tail_risk_studio.py", "tail risk analysis"),
        ("modules/performance.py", "portfolio performance"),
        ("modules/signal_scorecard.py", "signal performance scorecard"),
    ]

    qir_rounds = [
        "Round 1: Risk Regime + Fed path context",
        "Round 2: Policy/chain and event context",
        "Round 3: Stress, whale, and swan context",
        "Round 4: Macro synthesis and recommendation",
        "Round 5: Portfolio risk snapshot",
        "Debate: manual trigger only (not auto-run inside QIR)",
    ]

    debate_contract = [
        "Inputs: signal block + optional topic + model tier",
        "Agents: Sir Doomburger (bear), Sir Fukyerputs (bull), Judge Judy (arbiter)",
        "Outputs: bear_argument, bull_argument, verdict, confidence, asymmetry, key_disagreement",
        "Tie handling: contested_bias + contested_bias_reason when verdict=CONTESTED",
    ]

    signal_keys = [
        "_regime_context", "_rate_path_probs", "_dominant_rate_path", "_rp_plays_result",
        "_fed_plays_result", "_current_events_digest", "_doom_briefing", "_whale_summary",
        "_plays_result", "_portfolio_analysis", "_factor_analysis", "_sim_verdict",
        "_adversarial_debate", "_portfolio_risk_snapshot",
    ]
    loaded = [k for k in signal_keys if st.session_state.get(k) is not None]
    missing = [k for k in signal_keys if st.session_state.get(k) is None]

    lines = [
        "# REGARDED TERMINALS - DEEP PIPELINE MAP",
        f"Generated: {now_str}",
        f"Runtime Regime: {regime} | Quadrant: {quadrant} | Score Mode: {score_mode}",
        "Purpose: exact end-to-end architecture, contracts, execution order, and live runtime coverage.",
        "",
        "---",
        "## 1) SYSTEM TOPOLOGY",
        "",
        "### External Sources",
    ]

    lines.extend([f"- {x}" for x in data_sources])
    lines.extend(["", "### Services Layer"])
    lines.extend([f"- {x}" for x in service_layer])
    lines.extend(["", "### Module Layer"])
    lines.extend([f"- {path}: {desc}" for path, desc in module_layer])

    lines.extend([
        "",
        "---",
        "## 2) LL-ANCHORED CRISIS DETECTION SYSTEM (CI%)",
        "",
        "This is the core crisis signal — validated against 3,408 trading days (2013–2026).",
        "",
        "### Mathematics",
        "- Feature matrix: 9 FRED series + VIX, 20-day EWMA smoothed, 5yr rolling z-scored, ±3σ capped",
        "- Model: GaussianHMM (BIC-selected, 2–6 states), full covariance, empirical transmat with Laplace smoothing",
        "- LL scoring: model.score(X_full) / len(X) = log-likelihood per observation",
        "- Z-score: (ll_per_obs - brain.ll_baseline_mean) / brain.ll_baseline_std",
        "- CI%: abs(ll_zscore) / 0.448 * 100  [uncapped — COVID in-sample = 100%]",
        "- CRITICAL: lookback_years stored in brain JSON — live scoring must use brain.lookback_years exactly",
        "",
        "### CI% Zones",
        "- Zone 1 (CI < 22%):  NORMAL — conviction signals suppressed (they fire every day alone = 0% precision)",
        "- Zone 2 (CI 22–67%): MODEL STRESS — signals visible as context (unvalidated individually)",
        "- Zone 3 (CI ≥ 67%):  CRISIS GATE OPEN — 9.25% crash probability (3x baseline)",
        "- Zone 4 (CI > 100%): BEYOND TRAINING RANGE — purple, model scoring post-training extremes",
        "",
        "### Backtested Event Anchors",
        "- Tariff/Rate shocks (2022, 2025): CI 12–21% — correctly ignored (known regimes)",
        "- Volmageddon (Feb 2018): CI 76% — detected (z=-0.341)",
        "- Fed Panic (Dec 2018): CI 96% — detected 84 days early (z=-0.428)",
        "- COVID Start (Feb 2020): CI 94% — detected 36 days early (z=-0.421)",
        "- COVID Bottom (Mar 2020): CI 100% — peak (z=-0.448, the anchor)",
        "- Gate threshold: z < -0.30 = 67% CI",
        "",
        "### Post-Retrain Workflow",
        "1. Retrain brain (lookback_years=15 stored automatically in brain JSON)",
        "2. Run: python ll_gate_backtest_live_brain.py",
        "3. Check new COVID peak z-score — if changed from -0.448, update anchor in:",
        "   - modules/quick_run.py: abs(_tb_ll_z) / 0.448 * 100",
        "   - modules/backtesting.py: same formula",
        "",
        "### Market Bottom Watch (linked to CI%)",
        "- Always visible in QIR; dormant (opacity 0.55, grey) when CI% < 22, live when CI% ≥ 22",
        "- Live state: 4 bottom signals (LL recovery, VIX-was-elevated, HY compress, VVIX compress)",
        "- 4-tile legend: 1/4 slate → 2/4 blue → 3/4 amber → 4/4 emerald",
        "",
        "### Current Live State",
        f"- HMM state: {hmm_state_str}",
        f"- Brain: {hmm_brain_str}",
        "",
        "---",
        "## 3) ORCHESTRATION FLOWS",
        "",
        "### QIR Signal Pattern System",
        "- 27 named signal patterns covering all 3³ regime×tactical×options combinations",
        "- Each pattern has: label, color, bullish flag, interpretation, buy/short instrument tiers,",
        "  entry/risk rules, 2 tip directions (up/down)",
        "- Pattern classifier: _classify_signals(regime_dir, tactical_dir, options_dir) → pattern name",
        "- GENUINE_UNCERTAINTY is residual/fallback (unreachable in practice — all combos named)",
        "- Options instruments excluded from Buy/Short setup tiers; options flow affects Kelly via",
        "  alignment multiplier (0/4=×0.25 … 4/4=×1.00)",
        "",
        "### Quick Intel Run (QIR) Card Order",
        "1. HMM Brain State card",
        "2. LL-Anchored Crisis Detection (CI%)",
        "3. Market Bottom Watch card (always visible)",
        "4. Conviction × Uncertainty card",
        "5. Long Term Kelly card (renamed from 'KELLY SIZING'; tooltip: 'your core portfolio allocation')",
        "6. Buy/Short Setup cards (LONG KELLY / SHORT KELLY badges)",
        "7. Lean Accuracy block",
        "8. GEX + signal breakdown",
        "",
        "### Quick Intel Run (QIR) Rounds",
    ])
    lines.extend([f"- {step}" for step in qir_rounds])

    lines.extend([
        "",
        "### Discovery -> Valuation -> Portfolio Chain",
        "- Discovery builds narratives/themes/ticker shortlist and optional plays.",
        "- Valuation consumes ticker + macro + tactical + risk context for AI verdict + DCF.",
        "- Portfolio Intelligence aggregates positions + macro + factor + stress views.",
        "",
        "### Risk Regime Scoring Modes",
        "- Normal: rolling z-score baseline.",
        "- Coke Mode: EWMA fast-react on selected fast signals (credit, VIX, breadth).",
        "",
        "---",
        "## 4) DATA CONTRACTS",
        "",
        "### Debate Contract",
    ])
    lines.extend([f"- {item}" for item in debate_contract])

    lines.extend([
        "",
        "### Core Session Contracts (selected)",
        "- _regime_context: regime, score, quadrant, signal summary",
        "- _dominant_rate_path: scenario + probability",
        "- _portfolio_analysis: verdict, risk_score, positions, actions",
        "- _factor_analysis: factor verdicts, top risk, suggestions",
        "- _sim_verdict: GO/CAUTION/PASS style pre-trade verdict",
        "- _adversarial_debate: arguments, verdict, confidence, contested lean",
        "",
        "---",
        "## 5) LIVE RUNTIME SNAPSHOT",
        "",
        f"- Loaded signal keys: {len(loaded)}/{len(signal_keys)}",
        f"- Loaded: {', '.join(loaded) if loaded else '(none)'}",
        f"- Missing: {', '.join(missing) if missing else '(none)'}",
        f"- Current selected ticker: {st.session_state.get('ticker', 'unknown')}",
        f"- Current selected narrative: {st.session_state.get('narrative', 'unknown')}",
        "",
        "---",
        "## 6) CACHING, LIMITS, AND INTEGRITY",
        "",
        "- Streamlit @st.cache_data used across heavy fetch/synthesis paths (ttl=3600 or 86400).",
        "- SEC request pacing and bounded concurrency (ThreadPoolExecutor max_workers=5).",
        "- Session signals persisted through signals_cache for cross-module continuity.",
        "- HMM brain lookback_years stored in JSON — must equal value used in live scoring.",
        "- LL backtest (ll_gate_backtest_live_brain.py) must be re-run after every brain retrain.",
        "",
        "---",
        "## 7) GAP-ANALYSIS PROMPTS",
        "",
        "Use these prompts with this pipeline map:",
        "- Which contracts are weakly defined and should become typed schemas?",
        "- Which modules rely on stale assumptions or duplicated logic?",
        "- What monitoring and tests are missing for high-impact failure points?",
        "- Where should the pipeline add probabilistic uncertainty and calibration?",
        "- How should the 27-pattern classifier be backtested for signal accuracy?",
        "- What additional FRED features would improve HMM regime discrimination?",
        "",
        "---",
        f"Deep Pipeline Map generated by Regarded Terminals | {now_str}",
    ])
    return "\n".join(lines)


def _build_cross_sectional_export() -> str:
    """Build a cross-sectional AI context snapshot from all module context dicts."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# REGARDED TERMINALS — CROSS-SECTIONAL AI CONTEXT",
        f"Generated: {now_str}",
        "Source: QIR run — all module context blocks captured in one export",
        "",
        "---",
        "",
    ]

    _MODULES = [
        ("QIR / Signal State",      "_qir_ai_context"),
        ("QIR Debate Payload",      "_qir_debate_signals_text"),
        ("Risk Regime",             "_risk_regime_ai_context"),
        ("Options Flow",            "_options_flow_ai_context"),
        ("Wyckoff / Tactical",      "_wyckoff_ai_context"),
        ("Stress Signals / CI%",    "_stress_signals_ai_context"),
        ("Tail Risk Studio",        "_tail_risk_ai_context"),
        ("Portfolio / Journal",     "_trade_journal_ai_context"),
        ("Whale Movement",          "_whale_ai_context"),
        ("Narrative Discovery",     "_discovery_ai_context"),
    ]

    any_data = False
    for label, key in _MODULES:
        ctx = st.session_state.get(key) or {}
        if not ctx:
            continue
        any_data = True
        lines.append(f"## {label}")
        for k, v in ctx.items():
            if isinstance(v, list):
                lines.append(f"**{k}:**")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"**{k}:** {v}")
        lines.append("")
        lines.append("---")
        lines.append("")

    if not any_data:
        lines += [
            "⚠ No AI context data found.",
            "",
            "Run QIR first (⚡ Quick Intel Run → RUN ALL INTEL MODULES).",
            "After the run completes, the 🤖 AI CONTEXT LIVE badge will appear on the QIR page.",
            "Then return here and generate this export.",
        ]

    lines += [
        "## HOW TO USE THIS EXPORT",
        "",
        "Paste this document into Claude, ChatGPT, Gemini, or any AI and ask:",
        '- "Given this cross-sectional market state, what is my highest-conviction trade?"',
        '- "Where is my portfolio most vulnerable right now?"',
        '- "Which signals are confirming each other and which are diverging?"',
        '- "Given CI% and HMM state, should I be adding or reducing exposure?"',
        "",
        "All signal definitions are embedded above. No additional context needed.",
    ]

    return "\n".join(lines)


def _build_pipeline_graph_json() -> str:
    """Build machine-readable dependency graph JSON for LLM/tool analysis."""
    now_iso = datetime.now().isoformat()
    score_mode = st.session_state.get("_rr_score_mode", "normal")

    modules = [
        "modules/risk_regime.py",
        "modules/quick_run.py",
        "modules/fed_forecaster.py",
        "modules/current_events.py",
        "modules/stress_signals.py",
        "modules/whale_buyers.py",
        "modules/narrative_discovery.py",
        "modules/options_activity.py",
        "modules/valuation.py",
        "modules/trade_journal.py",
        "modules/signal_audit.py",
        "modules/forecast_accuracy.py",
        "modules/backtesting.py",
        "modules/export_hub.py",
    ]

    services = [
        "services/market_data.py",
        "services/sec_client.py",
        "services/claude_client.py",
        "services/signals_cache.py",
        "services/forecast_tracker.py",
        "services/backtest_engine.py",
        "services/news_feed.py",
        "services/telegram_client.py",
    ]

    data_sources = [
        "source:yfinance",
        "source:FRED",
        "source:SEC_EDGAR",
        "source:Google_Trends",
        "source:Congress_API",
        "source:Telegram_API",
    ]

    signal_keys = [
        "_regime_context", "_rate_path_probs", "_dominant_rate_path", "_rp_plays_result",
        "_fed_plays_result", "_current_events_digest", "_doom_briefing", "_whale_summary",
        "_plays_result", "_portfolio_analysis", "_factor_analysis", "_sim_verdict",
        "_adversarial_debate", "_portfolio_risk_snapshot", "_qir_debate_signals_text",
    ]

    nodes = []
    for m in modules:
        nodes.append({"id": m, "kind": "module"})
    for s in services:
        nodes.append({"id": s, "kind": "service"})
    for src in data_sources:
        nodes.append({"id": src, "kind": "source"})
    for key in signal_keys:
        nodes.append({
            "id": f"state:{key}",
            "kind": "state_key",
            "populated": st.session_state.get(key) is not None,
        })

    # Weights: 1=light, 3=medium, 5=heavy dependency.
    edges = [
        # Module -> Service
        {"source": "modules/risk_regime.py", "target": "services/market_data.py", "type": "uses", "weight": 5},
        {"source": "modules/risk_regime.py", "target": "services/claude_client.py", "type": "uses", "weight": 2},
        {"source": "modules/risk_regime.py", "target": "services/signals_cache.py", "type": "uses", "weight": 2},

        {"source": "modules/quick_run.py", "target": "services/claude_client.py", "type": "uses", "weight": 5},
        {"source": "modules/quick_run.py", "target": "services/market_data.py", "type": "uses", "weight": 3},
        {"source": "modules/quick_run.py", "target": "services/forecast_tracker.py", "type": "uses", "weight": 4},

        {"source": "modules/fed_forecaster.py", "target": "services/claude_client.py", "type": "uses", "weight": 4},
        {"source": "modules/fed_forecaster.py", "target": "services/market_data.py", "type": "uses", "weight": 4},

        {"source": "modules/current_events.py", "target": "services/news_feed.py", "type": "uses", "weight": 4},
        {"source": "modules/current_events.py", "target": "services/claude_client.py", "type": "uses", "weight": 3},

        {"source": "modules/stress_signals.py", "target": "services/claude_client.py", "type": "uses", "weight": 4},
        {"source": "modules/whale_buyers.py", "target": "services/sec_client.py", "type": "uses", "weight": 4},
        {"source": "modules/whale_buyers.py", "target": "services/claude_client.py", "type": "uses", "weight": 2},

        {"source": "modules/narrative_discovery.py", "target": "services/claude_client.py", "type": "uses", "weight": 4},
        {"source": "modules/narrative_discovery.py", "target": "services/market_data.py", "type": "uses", "weight": 3},
        {"source": "modules/narrative_discovery.py", "target": "services/sec_client.py", "type": "uses", "weight": 2},

        {"source": "modules/options_activity.py", "target": "services/market_data.py", "type": "uses", "weight": 5},
        {"source": "modules/options_activity.py", "target": "services/claude_client.py", "type": "uses", "weight": 2},

        {"source": "modules/valuation.py", "target": "services/claude_client.py", "type": "uses", "weight": 5},
        {"source": "modules/valuation.py", "target": "services/market_data.py", "type": "uses", "weight": 3},

        {"source": "modules/trade_journal.py", "target": "services/claude_client.py", "type": "uses", "weight": 4},
        {"source": "modules/trade_journal.py", "target": "services/market_data.py", "type": "uses", "weight": 3},

        {"source": "modules/signal_audit.py", "target": "services/signals_cache.py", "type": "uses", "weight": 4},
        {"source": "modules/forecast_accuracy.py", "target": "services/forecast_tracker.py", "type": "uses", "weight": 5},
        {"source": "modules/backtesting.py", "target": "services/backtest_engine.py", "type": "uses", "weight": 5},

        {"source": "modules/export_hub.py", "target": "services/signals_cache.py", "type": "uses", "weight": 4},
        {"source": "modules/export_hub.py", "target": "services/telegram_client.py", "type": "uses", "weight": 2},

        # Service -> Data source
        {"source": "services/market_data.py", "target": "source:yfinance", "type": "reads", "weight": 5},
        {"source": "services/market_data.py", "target": "source:FRED", "type": "reads", "weight": 5},
        {"source": "services/sec_client.py", "target": "source:SEC_EDGAR", "type": "reads", "weight": 5},
        {"source": "services/news_feed.py", "target": "source:Google_Trends", "type": "reads", "weight": 2},
        {"source": "services/telegram_client.py", "target": "source:Telegram_API", "type": "writes", "weight": 3},

        # Module -> session state contracts (high fan-out keys marked heavy)
        {"source": "modules/risk_regime.py", "target": "state:_regime_context", "type": "produces", "weight": 5},
        {"source": "modules/fed_forecaster.py", "target": "state:_rate_path_probs", "type": "produces", "weight": 4},
        {"source": "modules/fed_forecaster.py", "target": "state:_dominant_rate_path", "type": "produces", "weight": 4},
        {"source": "modules/current_events.py", "target": "state:_current_events_digest", "type": "produces", "weight": 3},
        {"source": "modules/stress_signals.py", "target": "state:_doom_briefing", "type": "produces", "weight": 3},
        {"source": "modules/whale_buyers.py", "target": "state:_whale_summary", "type": "produces", "weight": 3},
        {"source": "modules/narrative_discovery.py", "target": "state:_plays_result", "type": "produces", "weight": 3},
        {"source": "modules/trade_journal.py", "target": "state:_portfolio_analysis", "type": "produces", "weight": 4},
        {"source": "modules/trade_journal.py", "target": "state:_factor_analysis", "type": "produces", "weight": 3},
        {"source": "modules/trade_journal.py", "target": "state:_sim_verdict", "type": "produces", "weight": 3},
        {"source": "modules/quick_run.py", "target": "state:_adversarial_debate", "type": "produces", "weight": 3},
        {"source": "modules/quick_run.py", "target": "state:_portfolio_risk_snapshot", "type": "produces", "weight": 3},

        {"source": "modules/quick_run.py", "target": "state:_regime_context", "type": "consumes", "weight": 5},
        {"source": "modules/valuation.py", "target": "state:_regime_context", "type": "consumes", "weight": 5},
        {"source": "modules/valuation.py", "target": "state:_dominant_rate_path", "type": "consumes", "weight": 4},
        {"source": "modules/trade_journal.py", "target": "state:_regime_context", "type": "consumes", "weight": 5},
        {"source": "modules/trade_journal.py", "target": "state:_portfolio_risk_snapshot", "type": "consumes", "weight": 4},
        {"source": "modules/narrative_discovery.py", "target": "state:_regime_context", "type": "consumes", "weight": 4},
        {"source": "modules/export_hub.py", "target": "state:_regime_context", "type": "consumes", "weight": 5},
    ]

    payload = {
        "schema_version": "1.0",
        "generated_at": now_iso,
        "runtime": {
            "score_mode": score_mode,
            "regime": (st.session_state.get("_regime_context") or {}).get("regime"),
            "quadrant": (st.session_state.get("_regime_context") or {}).get("quadrant"),
        },
        "weight_scale": {
            "1": "light",
            "3": "medium",
            "5": "heavy",
        },
        "nodes": nodes,
        "edges": edges,
    }
    return json.dumps(payload, indent=2)


def _build_pipeline_upgrade_brief() -> str:
    """Build a concise upgrade-ready brief for external AI consultations."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    rc = st.session_state.get("_regime_context") or {}
    regime = rc.get("regime", "unknown")
    quadrant = rc.get("quadrant", "unknown")
    score_mode = st.session_state.get("_rr_score_mode", "normal")

    # Live HMM state
    try:
        from services.hmm_regime import load_current_hmm_state, load_hmm_brain
        _hmm_live = load_current_hmm_state()
        _hmm_brain = load_hmm_brain()
        hmm_str = (
            f"{_hmm_live.state_label}, conf={_hmm_live.confidence:.2f}, "
            f"ll_z={getattr(_hmm_live,'ll_zscore',0):.3f}, "
            f"CI%={abs(getattr(_hmm_live,'ll_zscore',0))/0.448*100:.1f}%"
        ) if _hmm_live else "unavailable"
        brain_str = (
            f"trained {_hmm_brain.trained_at[:10]}, "
            f"{_hmm_brain.n_states} states, lookback={_hmm_brain.lookback_years}yr"
        ) if _hmm_brain else "no brain"
    except Exception:
        hmm_str = "unavailable"
        brain_str = "unavailable"

    key_signals = [
        "_regime_context", "_dominant_rate_path", "_current_events_digest", "_doom_briefing",
        "_whale_summary", "_plays_result", "_portfolio_analysis", "_factor_analysis",
        "_sim_verdict", "_adversarial_debate", "_portfolio_risk_snapshot",
    ]
    loaded = [k for k in key_signals if st.session_state.get(k) is not None]
    missing = [k for k in key_signals if st.session_state.get(k) is None]

    lines = [
        "# BRAIN DUMP (FOR EXTERNAL AI)",
        f"Generated: {now_str}",
        f"Current runtime: regime={regime}, quadrant={quadrant}, score_mode={score_mode}",
        "Goal: get practical upgrade recommendations with implementation priority.",
        "",
        "## 1) SYSTEM STRUCTURE",
        "- App: Streamlit, entry point app.py, sidebar module routing, APP_PASSWORD auth gate",
        "- Modules (25): quick_run, risk_regime, backtesting, fed_forecaster, current_events,",
        "  stress_signals, whale_buyers, narrative_discovery, options_activity, valuation,",
        "  trade_journal, signal_audit, forecast_accuracy, export_hub, elliott_wave,",
        "  wyckoff, macro_scorecard, tail_risk_studio, performance, signal_scorecard, and more",
        "- Services (30+): hmm_regime, market_data, sec_client, claude_client, backtest_engine,",
        "  signals_cache, forecast_tracker, telegram_client, wyckoff_engine, elliott_wave_engine,",
        "  scoring, indicators, portfolio_sizing, sector_rotation, turning_point, and more",
        "- Utils (15): session, theme, components, signal_block, styles, auth, state_keys, etc.",
        "- State bus: Streamlit session_state with cross-module signal keys",
        "",
        "## 2) CORE CRISIS DETECTION ENGINE (CI%)",
        "This is the most mathematically rigorous component — validated on 3,408 days:",
        "",
        "- Model: GaussianHMM (BIC-selected up to 6 states, full covariance)",
        "- Features: 9 FRED series (HY/IG spreads, yield curve, DGS10/2, DFII10, NFCI, ICSA) + VIX",
        "  Pipeline: raw → 20d EWMA → 5yr rolling z-score → ±3σ cap",
        "- LL scoring: model.score(X_full) / len(X) = log-likelihood per observation (expanding window)",
        "- CI% formula: abs(ll_zscore) / 0.448 * 100 (uncapped, COVID in-sample = 100%)",
        "- Zones: Normal (<22%) | Stress (22–67%) | Crisis Confirmed (≥67%) | Beyond Training (>100%)",
        "- Gate z=-0.30: 9.25% crash prob (3x baseline). Volmageddon 76%, Fed Panic 96%, COVID 100%",
        "- CRITICAL BUG fixed: lookback_years now stored in brain JSON. Old bug: 14yr vs 15yr lookback",
        "  produced z=-1.272 (false crisis) vs correct z=-0.000 (normal market)",
        "- Market Bottom Watch: always-visible QIR card. Dormant (CI%<22) → live (CI%≥22) with",
        "  4 bottom signals: LL recovery, VIX-was-elevated, HY compress, VVIX compress",
        f"- Current live: {hmm_str}",
        f"- Brain: {brain_str}",
        "",
        "## 3) CRITICAL FLOWS",
        "- Macro flow: Risk Regime → Fed path → QIR synthesis → Portfolio/Valuation/Discovery",
        "- Crisis flow: FRED data → HMM score → CI% zone → conviction gate → QIR display",
        "- Decision flow: Discovery → Valuation → Portfolio AI actions → tracking/backtest",
        "- Debate flow: 3-agent adversarial (bear/bull/arbiter) with contested tie-bias lean",
        "- QIR pattern flow: regime_dir × tactical_dir × options_dir → _classify_signals → 1 of 27",
        "  named patterns → label/color/instruments/entry rules/tips rendered in Buy/Short Setup cards",
        "- Kelly flow: p = Bayesian blend(history_win_rate, conviction_score, history_weight=min(n/20,0.6))",
        "  b = avg_win%/avg_loss% from trade_journal.json; fallback: regime-implied (Goldilocks=1.6, etc.)",
        "  multipliers: alignment(0–4 signals)×HMM_state×stress_discount → half-Kelly capped at 15%",
        "  card: 'LONG TERM KELLY' (structural weeks/months). WHY 0% box shown when expectancy negative.",
        "",
        "## 4) DEPENDENCY HOTSPOTS",
        "- services/hmm_regime.py: lookback_years must match training. Re-run backtest after retrain.",
        "- services/claude_client.py: central dependency for synthesis and recommendations",
        "- services/market_data.py: central dependency for market and macro inputs",
        "- _regime_context: high blast-radius state key consumed by many modules",
        "- CI% anchor 0.448: hardcoded COVID peak z-score — must update after brain retrain if changed",
        "",
        "## 5) CURRENT COVERAGE SNAPSHOT",
        f"- Loaded keys ({len(loaded)}/{len(key_signals)}): {', '.join(loaded) if loaded else '(none)'}",
        f"- Missing keys: {', '.join(missing) if missing else '(none)'}",
        "",
        "## 6) KNOWN CONSTRAINTS",
        "- Multi-source API variability and intermittent data freshness",
        "- Session-state coupling across modules",
        "- Mixed output contracts across AI features",
        "- Need to preserve Streamlit responsiveness under heavier analysis",
        "- HMM only detects systemic regime shifts — not sudden flash crashes",
        "",
        "## 7) WHAT I WANT FROM YOU (OTHER AI)",
        "Please answer with:",
        "1. Top 10 upgrade ideas ranked by impact x effort",
        "2. 3 highest-risk dependencies and hardening steps",
        "3. How to extend the CI% system to detect market BOTTOMS (not just tops)",
        "4. What FRED features would improve HMM regime discrimination",
        "5. Contract/schema improvements for session-state and AI outputs",
        "6. Testing strategy to prevent regressions in QIR, Valuation, and Portfolio",
        "7. Performance optimizations (cache strategy, parallelization, fallback behavior)",
        "8. A 2-week implementation roadmap with milestones",
        "",
        "## 8) COPY-PASTE PROMPT",
        "Use this prompt with the brief:",
        '\"You are a principal engineer and quant researcher reviewing an investment intelligence',
        "Streamlit app called Regarded Terminals. Use this pipeline brief to produce a prioritized",
        "upgrade plan with concrete code-level recommendations, mathematical improvements to the",
        'CI% crisis detection system, and measurable success criteria.\"',
        "",
        f"Brain Dump generated by Regarded Terminals | {now_str}",
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
    ("QIR Debate Payload",    None,                        "Run QIR once to build debate payload"),
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
    elif label == "QIR Debate Payload":
        has = bool(st.session_state.get("_qir_debate_signals_text"))
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

    # --- Section B: Generate ---
    st.caption("💡 Paste directly into Claude.ai, ChatGPT, or Gemini for instant macro analysis.")

    if st.button("⬇ Generate Export", type="primary", key="export_generate"):
        with st.spinner("Building export document..."):
            content = _build_markdown_export(open_trades)
            filename = f"macro_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            mime = "text/plain"

        st.session_state["_export_content"] = content
        st.session_state["_export_filename"] = filename
        st.session_state["_export_mime"] = mime

    # --- Section C: Download + Preview ---
    content = st.session_state.get("_export_content")
    if content:
        filename = st.session_state.get("_export_filename", "export.txt")
        mime = st.session_state.get("_export_mime", "text/markdown")

        col1, col2, col3 = st.columns([1, 1, 2])
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
            from services.telegram_client import is_configured as _tg_ok, send_document as _tg_send_doc
            if _tg_ok():
                if st.button("📲 Send to Telegram", key="export_telegram"):
                    with st.spinner("Sending to Telegram..."):
                        _caption = f"📊 Macro Briefing — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        _ok = _tg_send_doc(filename, content, caption=_caption)
                    if _ok:
                        st.success("✅ Sent to Telegram")
                    else:
                        st.error("❌ Telegram send failed — check bot token & chat ID")
            else:
                st.caption("📲 Telegram not configured")
        with col3:
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

    # --- Section D: Architecture Exports ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">ARCHITECTURE EXPORTS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Export app architecture for AI-assisted upgrade planning and dependency analysis.")
    st.markdown(
        f'<div style="background:{COLORS["surface"]};border-left:3px solid #475569;'
        f'padding:8px 12px;border-radius:0 4px 4px 0;margin:6px 0 10px 0;font-size:11px;color:#94a3b8;">'
        f'<b style="color:#cbd5e1;">Tips:</b> '
        f'1) Export <b>Macro Briefing</b> for daily market analysis. '
        f'2) Export <b>Dependency Graph JSON</b> + <b>Brain Dump</b> together for architecture work. '
        f'3) Ask AI to rank upgrades by <i>impact × effort</i> and identify single points of failure. '
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("🕸 Generate Dependency Graph JSON", key="export_pipeline_graph_gen"):
        with st.spinner("Building dependency graph JSON..."):
            _graph_content = _build_pipeline_graph_json()
        st.session_state["_pipeline_graph_content"] = _graph_content

    if st.button("🧠 Generate Brain Dump", key="export_pipeline_upgrade_gen"):
        with st.spinner("Building brain dump..."):
            _upgrade_content = _build_pipeline_upgrade_brief()
        st.session_state["_pipeline_upgrade_content"] = _upgrade_content

    st.markdown(
        '<div style="font-size:10px;color:#64748b;margin-top:4px;line-height:1.6;">'
        '🕸 <b style="color:#94a3b8;">Dependency Graph JSON</b> — machine-readable weighted dependencies<br>'
        '🧠 <b style="color:#94a3b8;">Brain Dump</b> — full quant + architecture brief for external AI'
        '</div>',
        unsafe_allow_html=True,
    )

    _graph = st.session_state.get("_pipeline_graph_content")
    if _graph:
        _graph_filename = f"regarded_terminals_dependency_graph_{datetime.now().strftime('%Y%m%d')}.json"
        _g1, _g2, _g3 = st.columns([1, 1, 2])
        with _g1:
            st.download_button(
                label="💾 Download Dependency JSON",
                data=_graph,
                file_name=_graph_filename,
                mime="application/json",
                key="export_pipeline_graph_dl",
            )
        with _g2:
            try:
                from services.telegram_client import is_configured as _tg_ok3, send_document as _tg_send_doc3
                if _tg_ok3():
                    if st.button("📲 Send Dependency JSON", key="export_pipeline_graph_tg"):
                        with st.spinner("Sending to Telegram..."):
                            _ok3 = _tg_send_doc3(
                                _graph_filename,
                                _graph,
                                caption=f"🕸 Dependency Graph JSON — {datetime.now().strftime('%Y-%m-%d')}",
                            )
                        if _ok3:
                            st.success("✅ Sent to Telegram")
                        else:
                            st.error("❌ Telegram send failed")
                else:
                    st.caption("📲 Telegram not configured")
            except ImportError:
                st.caption("📲 Telegram not configured")
        with _g3:
            st.markdown(
                f'<div style="padding:8px 0;font-size:12px;color:#888;">'
                f'{len(_graph.splitlines())} lines · {len(_graph):,} chars · ~{len(_graph)//4:,} tokens</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Preview Dependency Graph JSON", expanded=False):
            st.code(_graph[:3000] + ("\n...[truncated]" if len(_graph) > 3000 else ""), language="json")

        st.markdown(
            f'<div style="background:{COLORS["surface"]};border-left:3px solid {COLORS["bloomberg_orange"]};'
            f'padding:10px 14px;border-radius:0 4px 4px 0;margin-top:8px;font-size:12px;color:#bbb;">'
            f'💡 <b>Suggested prompt:</b> <i>"Use this graph JSON to identify the top heavy dependencies '
            f'(weight=5), single points of failure, and highest blast-radius session keys. '
            f'Propose hardening steps in priority order."</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _upgrade = st.session_state.get("_pipeline_upgrade_content")
    if _upgrade:
        _upgrade_filename = f"regarded_terminals_brain_dump_{datetime.now().strftime('%Y%m%d')}.txt"
        _u1, _u2, _u3 = st.columns([1, 1, 2])
        with _u1:
            st.download_button(
                label="💾 Download Brain Dump",
                data=_upgrade,
                file_name=_upgrade_filename,
                mime="text/plain",
                key="export_pipeline_upgrade_dl",
            )
        with _u2:
            try:
                from services.telegram_client import is_configured as _tg_ok4, send_document as _tg_send_doc4
                if _tg_ok4():
                    if st.button("📲 Send Brain Dump", key="export_pipeline_upgrade_tg"):
                        with st.spinner("Sending to Telegram..."):
                            _ok4 = _tg_send_doc4(
                                _upgrade_filename,
                                _upgrade,
                                caption=f"🧠 Brain Dump — {datetime.now().strftime('%Y-%m-%d')}",
                            )
                        if _ok4:
                            st.success("✅ Sent to Telegram")
                        else:
                            st.error("❌ Telegram send failed")
                else:
                    st.caption("📲 Telegram not configured")
            except ImportError:
                st.caption("📲 Telegram not configured")
        with _u3:
            st.markdown(
                f'<div style="padding:8px 0;font-size:12px;color:#888;">'
                f'{len(_upgrade.splitlines())} lines · {len(_upgrade):,} chars · ~{len(_upgrade)//4:,} tokens</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Preview Brain Dump", expanded=False):
            st.code(_upgrade[:3000] + ("\n...[truncated]" if len(_upgrade) > 3000 else ""), language="markdown")

    # --- Section E: Cross-Sectional AI Context Export ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;letter-spacing:0.08em;">CROSS-SECTIONAL AI CONTEXT</span>'
        + (
            f'<span style="font-size:8px;font-weight:700;letter-spacing:0.1em;'
            f'background:#052e16;color:#4ade80;border:1px solid #166534;'
            f'padding:2px 7px;border-radius:3px;">🤖 CONTEXT LIVE</span>'
            if st.session_state.get("_qir_ai_context") else
            f'<span style="font-size:8px;font-weight:700;letter-spacing:0.1em;'
            f'background:#1c1917;color:#57534e;border:1px solid #292524;'
            f'padding:2px 7px;border-radius:3px;">🤖 RUN QIR FIRST</span>'
        )
        + '</div>',
        unsafe_allow_html=True,
    )
    st.caption("All 9 module context blocks in one document — paste into any AI for full cross-sectional analysis.")

    if st.button("🤖 Generate Cross-Sectional Export", key="export_cross_sectional_gen"):
        with st.spinner("Packaging all module context blocks..."):
            _xs_content = _build_cross_sectional_export()
        st.session_state["_cross_sectional_content"] = _xs_content

    _xs = st.session_state.get("_cross_sectional_content")
    if _xs:
        _xs_filename = f"regarded_terminals_ai_context_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        _xs1, _xs2, _xs3 = st.columns([1, 1, 2])
        with _xs1:
            st.download_button(
                label="💾 Download AI Context",
                data=_xs,
                file_name=_xs_filename,
                mime="text/markdown",
                key="export_cross_sectional_dl",
            )
        with _xs2:
            try:
                from services.telegram_client import is_configured as _tg_ok_xs, send_document as _tg_send_xs
                if _tg_ok_xs():
                    if st.button("📲 Send to Telegram", key="export_cross_sectional_tg"):
                        with st.spinner("Sending to Telegram..."):
                            _ok_xs = _tg_send_xs(
                                _xs_filename, _xs,
                                caption=f"🤖 Cross-Sectional AI Context — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            )
                        if _ok_xs:
                            st.success("✅ Sent to Telegram")
                        else:
                            st.error("❌ Telegram send failed")
                else:
                    st.caption("📲 Telegram not configured")
            except ImportError:
                st.caption("📲 Telegram not configured")
        with _xs3:
            st.markdown(
                f'<div style="padding:8px 0;font-size:12px;color:#888;">'
                f'{len(_xs.splitlines())} lines · {len(_xs):,} chars · ~{len(_xs)//4:,} tokens</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Preview Cross-Sectional Context", expanded=False):
            st.code(_xs[:3000] + ("\n...[truncated]" if len(_xs) > 3000 else ""), language="markdown")

        st.markdown(
            f'<div style="background:{COLORS["surface"]};border-left:3px solid #4285f4;'
            f'padding:10px 14px;border-radius:0 4px 4px 0;margin-top:8px;font-size:12px;color:#bbb;">'
            f'💡 <b>Suggested prompt:</b> <i>"Here is my full cross-sectional market state across all modules. '
            f'What is my highest-conviction trade right now, where is my biggest risk, '
            f'and which signals are diverging that I should watch?"</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Section F: Suggested prompts ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">READY-TO-USE AI PROMPTS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Copy any prompt and paste it alongside your exported Macro Briefing, Dependency Graph, or Brain Dump:")

    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "")
    is_risk_off = "Risk-Off" in regime
    has_portfolio = bool(open_trades)
    has_doom = bool(st.session_state.get("_doom_briefing"))
    has_swans = bool(st.session_state.get("_custom_swans"))

    prompts = [
        ("📋 Full Briefing Analysis", "Here is my current macro intelligence briefing. Summarize the top 3 themes, highlight the most actionable signals, and give me a prioritized action plan for my portfolio."),
        ("🤖 Cross-Sectional Analysis", "Here is my full cross-sectional market state across all modules (QIR, Risk Regime, Options Flow, Wyckoff, Stress Signals, Portfolio, Whale, Tail Risk, Discovery). What is my highest-conviction trade right now, where is my biggest risk, and which signals are diverging that I should watch?"),
        ("📊 Portfolio Risk Check", "Given this macro regime and my portfolio positions, what are the biggest risks I should be aware of and what actions should I take?"),
        ("🦢 Black Swan Exposure", "Which of my portfolio positions are most exposed to the black swan tail risks listed? Rank them by vulnerability."),
        ("🏦 Rate Path Trade", "Given the Fed rate path probabilities, which sectors and assets should I overweight or underweight?"),
        ("🔄 Regime Rotation", "Based on the macro regime and signal summary, suggest a sector rotation strategy with specific tickers."),
        ("🕸 Dependency Hardening", "Use this dependency graph JSON to identify the top heavy dependencies (weight=5), single points of failure, and highest blast-radius session keys. Propose hardening steps in priority order."),
        ("🧭 Upgrade Planning", "Given this upgrade brief of my investment tool, rank the top 10 upgrade ideas by impact × effort. List the top 3 quick wins I can implement this week."),
    ]

    if is_risk_off:
        prompts.insert(0, ("🛡 Risk-Off Hedge", "The macro regime is Risk-Off. Which of my long positions should I hedge or reduce first, and what instruments would you use?"))
    if has_doom and has_swans:
        prompts.append(("💥 Stress Test", "Cross-reference the doom briefing with the black swan tail risks. Which scenario poses the greatest threat to my specific positions?"))
    for title, prompt in prompts:
        with st.expander(title, expanded=False):
            st.code(prompt, language=None)
