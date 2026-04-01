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


def _build_pipeline_export() -> str:
    """Build a structured pipeline map of the entire app for AI review."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "unknown")
    quadrant = ctx.get("quadrant", "unknown")

    lines = [
        "# REGARDED TERMINALS — PIPELINE MAP",
        f"Generated: {now_str}  |  Current Regime: {regime}  |  Quadrant: {quadrant}",
        "Purpose: Full app pipeline description for AI gap-analysis.",
        "Paste into Claude.ai/ChatGPT and ask: 'What signals, tools, or analytical layers is this pipeline missing?'",
        "",
        "---",
        "## ARCHITECTURE OVERVIEW",
        "",
        "```",
        "External Data Sources",
        "  ├─ yfinance          → Market prices, OHLC, options chain",
        "  ├─ SEC EDGAR API     → 13F filings, Form 4 insider trades, full-text search",
        "  ├─ Google Trends     → Search interest (narrative momentum proxy)",
        "  ├─ FRED / market     → VIX, yield curve, credit spreads, macro indicators",
        "  └─ Congress API      → Congressional trade disclosures",
        "",
        "Services Layer  (services/)",
        "  ├─ market_data.py    → Batch yfinance fetch, z-scores, AssetSnapshot dataclass",
        "  ├─ sec_client.py     → SEC EDGAR rate-limited client (10 req/s), CIK mapping",
        "  ├─ claude_client.py  → Claude AI synthesis (Anthropic API)",
        "  ├─ congress_client.py→ Congressional trades feed",
        "  ├─ trends_client.py  → Google Trends pytrends wrapper",
        "  ├─ ibkr_client.py    → Interactive Brokers live connection (ib_insync)",
        "  ├─ forecast_tracker.py → Signal logging, ATR evaluation, accuracy stats",
        "  ├─ backtest_engine.py→ Strategy backtesting (ATR trailing stop exits)",
        "  └─ signals_cache.py  → Session-state signal persistence",
        "",
        "Modules  (modules/)",
        "  ├─ risk_regime.py        → Module 0: Cross-asset regime indicator",
        "  ├─ narrative_discovery.py→ Module 1: AI-grouped narrative/ticker discovery",
        "  ├─ narrative_pulse.py    → Module 2: Narrative sentiment tracking",
        "  ├─ edgar_scanner.py      → Module 3: SEC EDGAR full-text filing search",
        "  ├─ institutional.py      → Module 4: 13F institutional holdings analysis",
        "  ├─ insider_congress.py   → Module 5: Form 4 insider + Congress trades",
        "  ├─ options_activity.py   → Module 6: Options flow (unusual activity verdict)",
        "  ├─ valuation.py          → Module 7: AI valuation + recommendation",
        "  ├─ quick_run.py          → QIR Intelligence Dashboard (AI synthesis of all signals)",
        "  ├─ forecast_accuracy.py  → Forecast Accuracy Tracker",
        "  ├─ backtesting.py        → Strategy Backtest + Walk-Forward + ATR Replay",
        "  ├─ stress_signals.py     → Doom/stress scenario briefing",
        "  ├─ whale_buyers.py       → Institutional whale movement detection",
        "  ├─ fed_forecaster.py     → Fed rate-path probability forecasting",
        "  ├─ trade_journal.py      → Portfolio trade journal",
        "  ├─ performance.py        → Portfolio performance + P&L",
        "  ├─ macro_scorecard.py    → Cross-asset macro scorecard",
        "  ├─ signal_audit.py       → Signal quality audit trail",
        "  ├─ wyckoff.py            → Wyckoff price cycle analysis",
        "  ├─ elliott_wave.py       → Elliott Wave pattern detection",
        "  ├─ export_hub.py         → This export system",
        "  └─ alerts_settings.py    → Alert thresholds + Telegram config",
        "```",
        "",
        "---",
        "## MODULE 0 — RISK REGIME INDICATOR",
        "",
        "**Purpose:** Classify the current macro environment so all other signals are regime-aware.",
        "",
        "**How it works:**",
        "- Computes z-scores for 17+ cross-asset signals vs. their rolling 252-day history",
        "- Signals: SPY momentum, VIX level, credit spreads (HYG/LQD), yield curve (2s10s),",
        "  gold (GLD), USD (DXY), TLT (duration demand), small-cap vs large-cap (IWM/SPY),",
        "  copper (Dr. Copper), oil (USO), semiconductor index (SOXX), emerging markets (EEM),",
        "  high-beta vs low-vol ratio, put/call ratio",
        "- Composite z-score → Risk-On / Neutral / Risk-Off regime label",
        "- **Quadrant system:** Growth vs Inflation axis (4 quadrants: Goldilocks, Stagflation,",
        "  Reflation, Deflation) derived from equity/bond/commodity/currency z-scores",
        "- **Daily history:** Regime is saved with timestamp so signals can be evaluated",
        "  against the regime that existed when they were logged",
        "- **VIX buckets:** <15 calm, 15-20 normal, 20-30 elevated, >30 stress",
        "",
        "**Outputs stored in session state:**",
        "- `_regime_context` → {regime, score, quadrant, signals dict, history}",
        "- `_regime_context_ts` → timestamp of last run",
        "- `_dominant_rate_path` → dominant Fed scenario from rate-path overlay",
        "",
        "---",
        "## MODULE 1 — NARRATIVE DISCOVERY",
        "",
        "**Purpose:** Surface investment narratives (themes) and group tickers by them using AI.",
        "",
        "**How it works:**",
        "- User enters a ticker or macro theme",
        "- Google Trends data pulled for search momentum",
        "- Claude AI groups related tickers by narrative theme (e.g. 'AI infrastructure', 'reshoring')",
        "- Each narrative gets a momentum score, trend direction, and related ticker list",
        "",
        "---",
        "## MODULE 2 — NARRATIVE PULSE",
        "",
        "**Purpose:** Track sentiment and momentum for a specific narrative over time.",
        "",
        "**How it works:**",
        "- Pulls Google Trends weekly interest for a keyword",
        "- Compares current vs. 4-week and 13-week averages",
        "- Claude AI generates narrative sentiment summary (bullish/bearish/neutral)",
        "- Stores trend momentum for cross-module use",
        "",
        "---",
        "## MODULE 3 — EDGAR SCANNER",
        "",
        "**Purpose:** Search SEC EDGAR full-text search for specific keywords in filings.",
        "",
        "**How it works:**",
        "- SEC EDGAR full-text search API (efts.sec.gov)",
        "- Filters by form type (10-K, 10-Q, 8-K, etc.), date range, ticker",
        "- Rate-limited at 10 req/s with retry logic",
        "- Results include filing URL, description snippet, date filed",
        "",
        "---",
        "## MODULE 4 — INSTITUTIONAL HOLDINGS (13F)",
        "",
        "**Purpose:** Track what institutional investors (hedge funds, pensions) are buying/selling.",
        "",
        "**How it works:**",
        "- Pulls 13F filings from SEC EDGAR for any institution by CIK",
        "- Parses holdings XML (primary_doc.xml in filing index)",
        "- Diffs consecutive quarters to find new positions, increased stakes, exits",
        "- CIK-to-ticker mapping built from EDGAR company_tickers.json",
        "- Concurrent fetching with ThreadPoolExecutor (max 5 workers) to respect rate limits",
        "",
        "---",
        "## MODULE 5 — INSIDER + CONGRESS TRADES",
        "",
        "**Purpose:** Track insider Form 4 filings and Congressional stock trades.",
        "",
        "**How it works:**",
        "- **Insiders:** SEC EDGAR Form 4 filings parsed per ticker; extracts transaction type",
        "  (P=Purchase, S=Sale), shares, price, date; filters for open-market buys only",
        "- **Congress:** Congressional trade disclosure API (quiverquant-style); filters by",
        "  party, chamber, recency; flags when multiple members trade same ticker",
        "- Cluster detection: 3+ insider buys within 30-day window = cluster signal",
        "",
        "---",
        "## MODULE 6 — OPTIONS FLOW",
        "",
        "**Purpose:** Detect unusual options activity as a directional signal.",
        "",
        "**How it works:**",
        "- Pulls live options chain via yfinance for a given ticker",
        "- Computes open interest, volume, implied volatility per strike",
        "- Identifies unusual activity: volume/OI ratio > 3, far-OTM large premium flows",
        "- Aggregates call vs put bias → sentiment verdict (Bullish / Bearish / Neutral)",
        "- Stores verdict in session state for QIR synthesis and ATR trade logging",
        "",
        "---",
        "## MODULE 7 — AI VALUATION",
        "",
        "**Purpose:** Generate an AI-powered valuation and trade recommendation for a ticker.",
        "",
        "**How it works:**",
        "- Fetches fundamental data (P/E, EPS, revenue growth, debt/equity) via yfinance",
        "- Pulls regime context, options verdict, insider data from session state",
        "- Sends all data to Claude AI which outputs:",
        "  - Fair value estimate (DCF / comparable)",
        "  - Bull/base/bear case scenarios",
        "  - Conviction score (1-10)",
        "  - Buy / Hold / Sell recommendation",
        "- Logging: user can log the recommendation to Forecast Accuracy Tracker with one click",
        "",
        "---",
        "## QIR — QUICK INTELLIGENCE REPORT (Synthesis Layer)",
        "",
        "**Purpose:** Single-screen synthesis of all active signals into a macro verdict.",
        "",
        "**How it works:**",
        "- Reads all signal data from session state (regime, options, insider, institutional,",
        "  Fed rate path, narrative sentiment, whale movement, black swans)",
        "- Sends unified context to Claude AI",
        "- Outputs: Macro verdict label (e.g. BULLISH CONFIRMATION, BEARISH CONFIRMATION,",
        "  PULLBACK IN UPTREND, LATE CYCLE SQUEEZE, GENUINE UNCERTAINTY, etc.)",
        "- **Auto-logging:** Verdict is auto-logged to Forecast Tracker as a macro signal",
        "- **SPY Trade button:** User can log a SPY Buy/Sell ATR trade alongside the macro verdict",
        "- Verdict → SPY direction mapping:",
        "  - BULLISH CONFIRMATION / PULLBACK IN UPTREND / OPTIONS FLOW DIVERGENCE /",
        "    BEAR MARKET BOUNCE → Buy SPY",
        "  - BEARISH CONFIRMATION / LATE CYCLE SQUEEZE → Sell SPY",
        "  - GENUINE UNCERTAINTY → disabled",
        "",
        "---",
        "## FORECAST ACCURACY TRACKER",
        "",
        "**Purpose:** Measure whether the AI's signals actually worked (honest signal quality).",
        "",
        "**Signal types:**",
        "- `valuation` — ticker-based calls from Module 7 or QIR SPY trades → ATR exit",
        "- `squeeze` — squeeze/momentum signals → ATR exit",
        "- `regime` — macro regime direction calls → calendar horizon (no price to trail)",
        "- `fed` — Fed rate-path scenario calls → calendar horizon",
        "- `manual` — user-entered custom calls → calendar horizon",
        "",
        "**ATR Exit Engine:**",
        "- ATR period: 14 days",
        "- Trailing stop: 2.0 × ATR below high watermark (long) / above low watermark (short)",
        "- Profit target: 3.0 × ATR from entry",
        "- R:R ratio: 1.5:1 (target ÷ stop)",
        "- No forced calendar close for ticker signals — stays pending until ATR fires",
        "- Rationale: calendar exits corrupt accuracy stats (a great early call looks wrong)",
        "",
        "**Accuracy split:**",
        "- Macro accuracy: regime + fed calls (correct direction %)",
        "- Price accuracy: valuation + squeeze calls (target hit % vs stop hit %)",
        "- SPY alpha: (ticker return) - (SPY return) over same hold period",
        "",
        "**Regime win-rate breakdown:**",
        "- Groups price calls by macro quadrant + VIX bucket at log time",
        "- Shows which regime conditions produce the best signal quality",
        "",
        "---",
        "## BACKTESTING ENGINE",
        "",
        "**Strategies:**",
        "1. **SMA Crossover** — Golden cross (short SMA > long SMA) → ATR trailing exit",
        "2. **VIX Spike** — Contrarian buy SPY when VIX crosses threshold → ATR exit",
        "3. **Regime Flip** — Buy SPY on Risk-Off → Risk-On transition → ATR exit",
        "4. **Insider Cluster** — Buy on 3+ insider purchases in rolling window → ATR exit",
        "",
        "**ATR Exit in backtests:**",
        "- Same engine as live tracker: trailing stop on watermark, target fires on intraday H/L",
        "- Overlapping trade prevention: skips new signals if existing trade is still open",
        "- Exit reason tracked: profit_target / trailing_stop / data_end",
        "",
        "**Validation:**",
        "- Walk-forward: N sliding windows (train → test), OOS win-rate vs IS win-rate",
        "- OOS confidence: HIGH (≥55% OOS win rate + positive return), MODERATE, LOW",
        "- ATR Replay tab: simulate a single trade on any ticker/date to verify engine behavior",
        "",
        "---",
        "## DATA FRESHNESS + CACHING",
        "",
        "- `@st.cache_data(ttl=3600)` — most market data (1-hour cache)",
        "- `@st.cache_data(ttl=86400)` — SEC CIK maps, slow-moving reference data (24-hour)",
        "- SEC rate limit: 10 req/s enforced by `_rate_limit()` sleep in sec_client.py",
        "- ThreadPoolExecutor(max_workers=5) for concurrent SEC fetches",
        "- yfinance batch download: `droplevel('Ticker', axis=1)` handles MultiIndex columns",
        "",
        "---",
        "## ALERT + NOTIFICATION SYSTEM",
        "",
        "- Telegram bot integration: send_message(), send_document() via telegram_client.py",
        "- Alert thresholds configured in alerts_settings.py (VIX spike, regime change, etc.)",
        "- ATR fire notification: banner shown in Forecast Tracker when a trade resolves",
        "- Export Hub: any briefing document can be sent to Telegram as a file",
        "",
        "---",
        "## AUTHENTICATION",
        "",
        "- Password gate via APP_PASSWORD environment variable",
        "- Set in .env file; loaded via python-dotenv at app startup",
        "",
        "---",
        "## WHAT THIS PIPELINE DOES NOT CURRENTLY HAVE",
        "(Fill this in by asking the AI to review the above)",
        "",
        "Suggested prompt: 'Given this full pipeline map, what analytical capabilities, data sources,",
        "signal types, or risk measures would you add to make this a more complete investment",
        "intelligence system? List gaps by priority.'",
        "",
        "---",
        f"Pipeline Map generated by Regarded Terminals | {now_str}",
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

    # --- Section D: Pipeline Map Export ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">PIPELINE MAP EXPORT</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Exports a full structured description of every module, signal, data source, and "
        "engine in this app. Paste into any AI to find gaps, missing signals, or architecture improvements."
    )

    if st.button("🗺 Generate Pipeline Map", key="export_pipeline_gen"):
        with st.spinner("Building pipeline map..."):
            _pipeline_content = _build_pipeline_export()
        st.session_state["_pipeline_content"] = _pipeline_content

    _pipe = st.session_state.get("_pipeline_content")
    if _pipe:
        _pipe_filename = f"regarded_terminals_pipeline_{datetime.now().strftime('%Y%m%d')}.txt"
        col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
        with col_p1:
            st.download_button(
                label="💾 Download Pipeline Map",
                data=_pipe,
                file_name=_pipe_filename,
                mime="text/plain",
                key="export_pipeline_dl",
            )
        with col_p2:
            try:
                from services.telegram_client import is_configured as _tg_ok2, send_document as _tg_send_doc2
                if _tg_ok2():
                    if st.button("📲 Send Pipeline to Telegram", key="export_pipeline_tg"):
                        with st.spinner("Sending to Telegram..."):
                            _ok2 = _tg_send_doc2(_pipe_filename, _pipe,
                                                  caption=f"🗺 Regarded Terminals Pipeline Map — {datetime.now().strftime('%Y-%m-%d')}")
                        if _ok2:
                            st.success("✅ Sent to Telegram")
                        else:
                            st.error("❌ Telegram send failed")
                else:
                    st.caption("📲 Telegram not configured")
            except ImportError:
                st.caption("📲 Telegram not configured")
        with col_p3:
            st.markdown(
                f'<div style="padding:8px 0;font-size:12px;color:#888;">'
                f'{len(_pipe.splitlines())} lines · {len(_pipe):,} chars · ~{len(_pipe)//4:,} tokens</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Preview Pipeline Map", expanded=False):
            st.code(_pipe[:3000] + ("\n…[truncated]" if len(_pipe) > 3000 else ""), language="markdown")

        st.markdown(
            f'<div style="background:{COLORS["surface"]};border-left:3px solid {COLORS["bloomberg_orange"]};'
            f'padding:10px 14px;border-radius:0 4px 4px 0;margin-top:8px;font-size:12px;color:#bbb;">'
            f'💡 <b>Suggested prompt:</b> <i>"Given this full pipeline map of my investment tool, '
            f'what analytical capabilities, data sources, or signal types are missing? '
            f'List gaps by priority with a brief explanation for each."</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
