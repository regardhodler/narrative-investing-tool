import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    import os
    _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    _tier_options = ["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
    _tier_map = {
        "⚡ Standard": (False, None),
        "🧠 Regard Mode": (True, "claude-haiku-4-5-20251001"),
        "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
    }
    _tier_badges = {
        "⚡ Standard": '<span style="font-size:11px;background:#2A3040;color:#888;padding:2px 7px;border-radius:3px;">⚡ Groq</span>',
        "🧠 Regard Mode": '<span style="font-size:11px;background:#FF8811;color:#000;padding:2px 7px;border-radius:3px;font-weight:700;">🧠 Haiku</span>',
        "👑 Highly Regarded Mode": '<span style="font-size:11px;background:linear-gradient(90deg,#c89b3c,#f0d060);color:#000;padding:2px 7px;border-radius:3px;font-weight:700;">👑 Sonnet</span>',
    }

    # ── Context Readiness ──────────────────────────────────────────────────────
    _regime_tier = st.session_state.get("_regime_plays_tier")
    _disc_tier   = st.session_state.get("_discovery_tier")

    def _tier_badge_html(tier):
        if tier is None:
            return '<span style="font-size:11px;background:#3B1F1F;color:#ef4444;padding:2px 8px;border-radius:4px;">⚠️ Not loaded</span>'
        icon = tier.split()[0]
        bg    = {"⚡": "#334155", "🧠": "#4C1D95", "👑": "#1a1000"}.get(icon, "#334155")
        color = {"⚡": "#94a3b8", "🧠": "white",   "👑": "#FF8811"}.get(icon, "white")
        return f'<span style="font-size:11px;background:{bg};color:{color};padding:2px 8px;border-radius:4px;font-weight:600;">{tier}</span>'

    _both_loaded = _regime_tier is not None and _disc_tier is not None
    _both_sonnet = (
        _regime_tier == "👑 Highly Regarded Mode" and _disc_tier == "👑 Highly Regarded Mode"
    )
    _cr_suffix = " ✅" if _both_sonnet else " ⚠️" if not _both_loaded else " 🟡"
    with st.expander(f"📡 Context Readiness{_cr_suffix}", expanded=not _both_loaded):
        # ── Engine selectors ──────────────────────────────────────────────────
        _cr_tier_opts = ["⚡ Groq", "🧠 Regard Mode", "👑 Highly Regarded Mode"] if _has_anthropic else ["⚡ Groq"]
        _cr_model_map = {
            "⚡ Groq": (False, None),
            "🧠 Regard Mode": (True, "claude-haiku-4-5-20251001"),
            "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
        }
        _sel_r_idx = _cr_tier_opts.index(_regime_tier) if _regime_tier in _cr_tier_opts else 0
        _sel_d_idx = _cr_tier_opts.index(_disc_tier)   if _disc_tier   in _cr_tier_opts else 0

        _cre1, _cre2 = st.columns(2)
        with _cre1:
            _regime_tier_sel = st.radio("Risk Regime Engine", _cr_tier_opts, horizontal=True,
                                        key="cr_regime_engine", index=_sel_r_idx)
        with _cre2:
            _disc_tier_sel   = st.radio("Discovery Engine",   _cr_tier_opts, horizontal=True,
                                        key="cr_disc_engine",  index=_sel_d_idx)

        # ── Status cards with orange glow ─────────────────────────────────────
        def _glow_card(title, tier_sel, status_badge, caption_text):
            is_gold  = tier_sel == "👑 Highly Regarded Mode"
            border   = "2px solid #FF8811" if is_gold else "1px solid #334155"
            shadow   = "0 0 14px #FF881155" if is_gold else "none"
            return (
                f'<div style="border:{border};border-radius:8px;padding:12px 16px;'
                f'box-shadow:{shadow};margin:6px 0;">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:0.08em;'
                f'color:#94a3b8;text-transform:uppercase;margin-bottom:6px;">{title}</div>'
                f'{status_badge}'
                f'<div style="font-size:11px;color:#64748b;margin-top:6px;">{caption_text}</div>'
                f'</div>'
            )

        _gc1, _gc2 = st.columns(2)
        with _gc1:
            st.markdown(_glow_card("Risk Regime AI", _regime_tier_sel,
                _tier_badge_html(_regime_tier), "Risk Regime → Fed Forecaster → Rate-Path Plays"),
                unsafe_allow_html=True)
        with _gc2:
            st.markdown(_glow_card("Discovery AI", _disc_tier_sel,
                _tier_badge_html(_disc_tier), "Discovery → Cross-Signal Macro Plays"),
                unsafe_allow_html=True)

        # ── Workflow step guide ───────────────────────────────────────────────
        st.markdown("---")
        _regime_ctx       = st.session_state.get("_regime_context")
        _has_rate_path_v  = bool(st.session_state.get("_dominant_rate_path"))
        _has_black_swans_v = bool(st.session_state.get("_custom_swans"))
        _bs_count_v       = len(st.session_state.get("_custom_swans", {}))
        _step1_done  = bool(_regime_ctx) and _has_rate_path_v
        _step3_done  = _both_loaded   # Step 3 = run plays
        _step4_ready = _step1_done and _step3_done

        def _step_badge(done, active):
            if done:   return '<span style="color:#22c55e;font-weight:700;">✅</span>'
            if active: return '<span style="color:#FF8811;font-weight:700;">▶</span>'
            return '<span style="color:#475569;font-weight:700;">○</span>'

        def _val_age(ts_key: str) -> str:
            from datetime import datetime as _dt
            _ts = st.session_state.get(ts_key)
            if not _ts:
                return ""
            _mins = int((_dt.now() - _ts).total_seconds() / 60)
            if _mins < 1:
                return ' <span style="color:#22c55e;font-size:10px;">— just now</span>'
            if _mins < 60:
                return f' <span style="color:#64748b;font-size:10px;">— {_mins}m ago</span>'
            return f' <span style="color:#64748b;font-size:10px;">— {_mins // 60}h ago</span>'

        _s1 = _step_badge(_step1_done, not _step1_done)
        # Step 2 (optional) — just shows ✅ or ○, never blocks
        _s2_opt = '✅' if _has_black_swans_v else '○'
        _s2_color = '#22c55e' if _has_black_swans_v else '#475569'
        _bs_step_hint = f"{_bs_count_v} events analyzed ✓" if _has_black_swans_v else "go to Risk Regime → Fed Forecaster → Black Swan"
        _s3 = _step_badge(_step3_done, _step1_done and not _step3_done)
        _s4 = _step_badge(_step4_ready, _step3_done and not _step4_ready)

        st.markdown(
            f'<div style="font-size:11px;color:#94a3b8;line-height:2.1;">'
            f'{_s1} <b>Step 1</b> — <b>Risk Regime</b> → <b>Fed Forecaster</b> (regime + rate path)'
            f'{"" if _step1_done else " ← do this first"}'
            f'{_val_age("_regime_context_ts") if _step1_done else ""}<br>'
            f'<span style="color:{_s2_color};font-weight:700;">{_s2_opt}</span>'
            f' <b>Step 2</b> <span style="color:#64748b;">(optional)</span>'
            f' — <b>Black Swan</b> events — <span style="color:#64748b;">{_bs_step_hint}</span>'
            f'{_val_age("_custom_swans_ts") if _has_black_swans_v else ""}<br>'
            f'{_s3} <b>Step 3</b> — Click <b>Run Regime Plays</b> &amp; <b>Run Discovery Plays</b> below'
            f'{"" if _step3_done else (" ← do this now" if _step1_done else " (needs Step 1)")}'
            f'{_val_age("_fed_plays_result_ts") if _step3_done else ""}<br>'
            f'{_s4} <b>Step 4</b> — Select engine + toggle Undervaluation Spotlight → <b>Analyze Ticker</b>'
            f'{"&nbsp;&nbsp;<span style=\'color:#22c55e;\'>← ready!</span>" if _step4_ready else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── Action buttons ────────────────────────────────────────────────────
        _ab1, _ab2 = st.columns(2)
        with _ab1:
            _r_use_cl, _r_model = _cr_model_map[_regime_tier_sel]
            _r_icon = _regime_tier_sel.split()[0]
            _r_btn_label = f"{'▶ ' if _step1_done and not _regime_tier else ''}Run {_r_icon} Regime Plays"
            if st.button(_r_btn_label, key="cr_run_regime",
                         disabled=not bool(_regime_ctx),
                         help="Re-runs Regime Plays with the selected engine" if _regime_ctx else "Load Risk Regime → Fed Forecaster first"):
                from services.claude_client import suggest_regime_plays
                with st.spinner(f"Running {_regime_tier_sel}..."):
                    suggest_regime_plays(
                        _regime_ctx["regime"], _regime_ctx["score"], _regime_ctx["signal_summary"],
                        use_claude=_r_use_cl, model=_r_model,
                    )
                st.session_state["_regime_plays_tier"] = _regime_tier_sel
                st.rerun()
            if not _regime_ctx:
                st.caption("⚠ Load Risk Regime → Fed Forecaster first")
        with _ab2:
            _d_use_cl, _d_model = _cr_model_map[_disc_tier_sel]
            _d_icon = _disc_tier_sel.split()[0]
            if st.button(f"Run {_d_icon} Discovery Plays", key="cr_run_disc"):
                from services.claude_client import suggest_regime_plays as _srp_disc
                from modules.narrative_discovery import _get_macro_context_for_plays
                with st.spinner(f"Running {_disc_tier_sel}..."):
                    _mctx = _get_macro_context_for_plays()
                    if _mctx:
                        _srp_disc(
                            _mctx["regime"], _mctx["score"], str(_mctx),
                            use_claude=_d_use_cl, model=_d_model,
                        )
                st.session_state["_discovery_tier"] = _disc_tier_sel
                st.rerun()

    # ── Undervaluation Spotlight toggle ───────────────────────────────────────
    st.toggle(
        "🎯 Undervaluation Spotlight",
        key="underval_spotlight",
        help="Show a prominent banner with DCF discount % above the analysis",
    )
    # ──────────────────────────────────────────────────────────────────────────

    _prev_tier = st.session_state.get("_val_tier_prev")
    _selected_val_tier = st.radio(
        "Engine", _tier_options, horizontal=True, key="val_engine_radio",
        help="Standard = Groq (fast/free) · Regard Mode = Claude Haiku · Highly Regarded = Claude Sonnet"
    )
    _use_claude, _cl_model = _tier_map[_selected_val_tier]
    if not _has_anthropic and _use_claude:
        st.caption("⚠ ANTHROPIC_API_KEY not set — falling back to Groq.")
        _use_claude, _cl_model = False, None

    st.session_state["_val_tier_prev"] = _selected_val_tier

    _badge_html = f' {_tier_badges[_selected_val_tier]}'
    st.markdown(
        f'<h2 style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:20px;'
        f'font-weight:700;margin-bottom:4px;">AI VALUATION &amp; RECOMMENDATION{_badge_html}</h2>',
        unsafe_allow_html=True,
    )

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view AI valuation.")
        return

    with st.spinner("Collecting market signals..."):
        signals = _collect_signals(ticker)

    if not signals:
        st.warning("Could not collect enough data for valuation.")
        return

    from datetime import datetime

    # ── Ticker info bar ────────────────────────────────────────────────────────
    _meta = signals.get("meta", {})
    _price_data = signals.get("price") or {}
    _company_name = _meta.get("name", ticker)
    _sector = _meta.get("sector", "")
    _current_price = _price_data.get("current")
    _1y_return = _price_data.get("period_return_pct")
    _price_str = f"${_current_price:,.2f}" if _current_price else "—"
    _ret_color = "#22c55e" if (_1y_return or 0) >= 0 else "#ef4444"
    _ret_str = f"{_1y_return:+.1f}% (1Y)" if _1y_return is not None else ""
    st.markdown(
        f'<div style="border:1px solid #1e293b;border-radius:8px;padding:10px 18px;'
        f'margin-bottom:10px;background:#0f172a;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:18px;'
        f'font-weight:700;color:#f1f5f9;">{_company_name}</span>'
        f'<span style="font-size:12px;color:#64748b;font-weight:600;letter-spacing:0.06em;">{ticker.upper()}</span>'
        f'<span style="color:#334155;">·</span>'
        f'<span style="font-size:16px;font-weight:700;color:#e2e8f0;">{_price_str}</span>'
        + (f'<span style="font-size:12px;font-weight:600;color:{_ret_color};">{_ret_str}</span>' if _ret_str else "")
        + (f'<span style="font-size:11px;color:#475569;">{_sector}</span>' if _sector else "")
        + f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 24H")

    signals_text = _format_signals_text(ticker, signals)

    # Inject rate path probabilities from Fed Forecaster into signals_text
    _rp = st.session_state.get("_dominant_rate_path")
    _rp_all = st.session_state.get("_rate_path_probs", [])
    if _rp and _rp.get("scenario"):
        _scenario_labels = {
            "cut_25": "25bp cut", "cut_50": "50bp cut",
            "hold": "Hold", "hike_25": "25bp hike",
        }
        _rp_label = _scenario_labels.get(_rp["scenario"], _rp["scenario"])
        _rp_line = f"Fed Rate Path (market-implied): {_rp_label} {_rp['prob_pct']:.0f}% probability"
        if _rp_all:
            _others = [
                f"{_scenario_labels.get(r['scenario'], r['scenario'])} {round(r.get('prob', 0) * 100)}%"
                for r in sorted(_rp_all, key=lambda r: r.get("prob", 0), reverse=True)
                if r.get("scenario") != _rp["scenario"]
            ]
            if _others:
                _rp_line += f" | Alt scenarios: {', '.join(_others[:2])}"
        signals_text += f"\n{_rp_line}"

    # Inject Black Swan tail risks into signals_text
    _val_swans = st.session_state.get("_custom_swans", {})
    if _val_swans:
        _swan_lines = []
        for _slabel, _sdata in list(_val_swans.items())[:3]:
            _sprob = _sdata.get("probability_pct", 0)
            _simpacts = _sdata.get("asset_impacts", {})
            _simpact_str = ", ".join(
                f"{k}={v}" for k, v in list(_simpacts.items())[:4]
            )
            _swan_lines.append(f"{_slabel} ({_sprob:.0f}% prob): {_simpact_str}")
        signals_text += "\nBlack Swan Tail Risks: " + "; ".join(_swan_lines)

    # Inject NewsAPI sentiment (silent if NEWSAPI_KEY absent)
    from services.claude_client import fetch_news_sentiment as _fetch_news
    _company_name = signals.get("meta", {}).get("name", ticker)
    _news_sent = _fetch_news(ticker, _company_name)
    if _news_sent:
        _sent_icon = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(_news_sent.get("overall", ""), "")
        signals_text += f"\n\n## Recent News Sentiment\n"
        signals_text += (
            f"Overall: {_sent_icon} {_news_sent.get('overall','').capitalize()} "
            f"(score: {_news_sent.get('score', 0):+.2f})\n"
        )
        for _h in _news_sent.get("headlines", [])[:5]:
            signals_text += f"- [{_h.get('sentiment','')}] {_h.get('title','')}\n"

    # Full signal transparency expander
    with st.expander("📊 Full Signal Transparency (AI Inputs)", expanded=False):
        _render_signal_transparency(signals)

    with st.spinner("Generating AI valuation..."):
        from services.claude_client import generate_valuation
        result = generate_valuation(ticker, signals_text, use_claude=_use_claude, model=_cl_model)

    if not result:
        has_key = bool(os.getenv("GROQ_API_KEY", ""))
        if not has_key:
            st.error("GROQ_API_KEY is not set. Add it to your .env file or Streamlit Cloud secrets.")
        else:
            st.error("AI valuation failed — the LLM returned an unparseable response. Try again.")
            with st.expander("Debug: Signal data sent to LLM"):
                st.code(signals_text)
        return

    _grad_color = "#c89b3c" if _selected_val_tier == "👑 Highly Regarded Mode" else "#FF8811"
    if _use_claude:
        st.markdown(
            f'<div style="height:2px;background:linear-gradient(90deg,{_grad_color},{_grad_color}33,transparent);'
            'margin:6px 0 12px 0;border-radius:1px;"></div>',
            unsafe_allow_html=True,
        )
    _render_rating_banner(result)
    _render_signal_scorecard(signals)
    _render_analysis(result)

    # DCF Valuation
    st.markdown("---")
    _dcf_scenarios = _render_dcf(ticker)

    # Kelly Position Sizing
    if _dcf_scenarios and result:
        st.markdown("---")
        _render_kelly(result, _dcf_scenarios)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This valuation is for informational purposes only "
        "and does not constitute financial advice. The DCF model uses estimates and assumptions "
        "that may not reflect actual future performance. Always do your own research."
    )


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_signals(ticker: str) -> dict | None:
    """Gather signals from yfinance and existing services into a structured dict."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        return None

    signals = {}

    # Meta — company name and sector for display
    signals["meta"] = {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
    }

    # 1. Price & Technicals
    try:
        hist = stock.history(period="1y")
        if not hist.empty:
            close = hist["Close"]
            current = close.iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]

            # RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            rsi = 100 - (100 / (1 + rs))

            signals["price"] = {
                "current": round(current, 2),
                "52w_high": round(close.max(), 2),
                "52w_low": round(close.min(), 2),
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "above_sma20": current > sma20,
                "above_sma50": current > sma50,
                "above_sma200": current > sma200,
                "rsi14": round(rsi, 1),
                "period_return_pct": round((current / close.iloc[0] - 1) * 100, 1),
            }
    except Exception:
        signals["price"] = None

    # 2. Fundamentals
    try:
        signals["fundamentals"] = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        signals["fundamentals"] = None

    # 3. Institutional
    try:
        signals["institutional"] = {
            "inst_pct": info.get("heldPercentInstitutions"),
            "insider_pct": info.get("heldPercentInsiders"),
            "num_institutions": info.get("institutionCount"),
        }
    except Exception:
        signals["institutional"] = None

    # 4. Insider Activity
    try:
        from services.sec_client import get_insider_trades
        trades = get_insider_trades(ticker)
        if trades is not None and not trades.empty:
            buys = trades[trades["type"].str.contains("Purchase|Buy|Acquisition", case=False, na=False)]
            sells = trades[trades["type"].str.contains("Sale|Sell|Disposition", case=False, na=False)]
            signals["insider"] = {
                "buy_count": len(buys),
                "sell_count": len(sells),
                "buy_value": buys["value"].sum() if "value" in buys.columns else 0,
                "sell_value": sells["value"].sum() if "value" in sells.columns else 0,
                "net_sentiment": "bullish" if len(buys) > len(sells) else "bearish" if len(sells) > len(buys) else "neutral",
            }
        else:
            signals["insider"] = None
    except Exception:
        signals["insider"] = None

    # 5. Options Sentiment
    try:
        expirations = stock.options
        if expirations:
            chain = stock.option_chain(expirations[0])
            call_vol = chain.calls["volume"].sum()
            put_vol = chain.puts["volume"].sum()
            pc_ratio = put_vol / call_vol if call_vol > 0 else 0
            signals["options"] = {
                "put_call_ratio": round(pc_ratio, 2),
                "call_volume": int(call_vol),
                "put_volume": int(put_vol),
                "sentiment": "bearish" if pc_ratio > 1.0 else "bullish" if pc_ratio < 0.7 else "neutral",
            }
        else:
            signals["options"] = None
    except Exception:
        signals["options"] = None

    # 6. Company Profile
    try:
        from services.claude_client import describe_company
        name = info.get("longName", ticker)
        sic = info.get("industry", "")
        profile = describe_company(name, ticker, sic)
        signals["profile"] = profile
    except Exception:
        signals["profile"] = None

    # 7. Cross-module: institutional 13F flow + stress signals
    signals["whale"] = _collect_whale_signals(ticker)
    signals["stress"] = _collect_stress_signals()

    return signals


@st.cache_data(ttl=3600)
def _collect_whale_signals(ticker: str) -> dict:
    """Institutional 13F position-change bias from yfinance."""
    result = {}
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is not None and "pctChange" in df.columns and "Value" in df.columns:
            df = df.dropna(subset=["pctChange", "Value"])
            total_val = df["Value"].sum()
            if total_val > 0:
                weighted = (df["pctChange"] * df["Value"]).sum() / total_val
                result["institutional_bias"] = (
                    "BULLISH" if weighted > 0.005 else
                    "BEARISH" if weighted < -0.005 else "NEUTRAL"
                )
                result["institutional_score"] = round(weighted * 100, 2)
                result["institution_count"] = len(df)
            else:
                result["institutional_bias"] = "NEUTRAL"
        else:
            result["institutional_bias"] = "UNKNOWN"
    except Exception:
        result["institutional_bias"] = "UNKNOWN"

    # Insider buy % by dollar value (from SEC Form 4)
    try:
        from services.sec_client import get_insider_trades
        trades = get_insider_trades(ticker)
        if trades is not None and not getattr(trades, "empty", True):
            buys_df = trades[trades["type"].str.contains("Purchase|Buy|Acquisition", case=False, na=False)]
            sells_df = trades[trades["type"].str.contains("Sale|Sell|Disposition", case=False, na=False)]
            buys_val = buys_df["value"].sum() if "value" in buys_df.columns else 0
            sells_val = sells_df["value"].sum() if "value" in sells_df.columns else 0
            total = buys_val + sells_val
            pct = buys_val / total if total > 0 else 0.5
            result["insider_buy_pct"] = round(pct * 100, 1)
            result["insider_sentiment"] = (
                "BULLISH" if pct > 0.55 else "BEARISH" if pct < 0.45 else "NEUTRAL"
            )
        else:
            result["insider_sentiment"] = "UNKNOWN"
    except Exception:
        result["insider_sentiment"] = "UNKNOWN"

    return result


@st.cache_data(ttl=3600)
def _collect_stress_signals() -> dict:
    """Macro stress indicators from FRED and VIX (reuses cached market_data layer)."""
    from services.market_data import fetch_fred_series_safe, fetch_batch_safe

    result = {}
    try:
        hy = fetch_fred_series_safe("BAMLH0A0HYM2")
        result["hy_spread"] = round(float(hy.iloc[-1]), 1) if hy is not None and len(hy) else None
    except Exception:
        result["hy_spread"] = None

    try:
        yc = fetch_fred_series_safe("T10Y2Y")
        result["yield_curve"] = round(float(yc.iloc[-1]), 2) if yc is not None and len(yc) else None
    except Exception:
        result["yield_curve"] = None

    try:
        snaps = fetch_batch_safe({"^VIX": "VIX"}, "5d", "1d")
        result["vix"] = snaps["^VIX"].latest_price
    except Exception:
        result["vix"] = None

    hy_v = result.get("hy_spread") or 0
    vix_v = result.get("vix") or 0
    result["stress_label"] = (
        "HIGH" if (hy_v > 500 or vix_v > 35) else
        "ELEVATED" if (hy_v > 300 or vix_v > 25) else "CALM"
    )
    return result


def _format_signals_text(ticker: str, signals: dict) -> str:
    """Convert signals dict into a readable text block for the LLM prompt."""
    lines = [f"=== Signal Snapshot for {ticker} ===\n"]

    if signals.get("price"):
        p = signals["price"]
        lines.append("## Price & Technicals")
        lines.append(f"Current: ${p['current']} | 52W High: ${p['52w_high']} | 52W Low: ${p['52w_low']}")
        lines.append(f"SMA20: ${p['sma20']} ({'above' if p['above_sma20'] else 'below'}) | "
                      f"SMA50: ${p['sma50']} ({'above' if p['above_sma50'] else 'below'}) | "
                      f"SMA200: ${p['sma200']} ({'above' if p['above_sma200'] else 'below'})")
        lines.append(f"RSI(14): {p['rsi14']} | 1Y Return: {p['period_return_pct']}%\n")

    if signals.get("fundamentals"):
        f = signals["fundamentals"]
        lines.append("## Fundamentals")
        lines.append(f"P/E: {f['pe_ratio']} | Fwd P/E: {f['forward_pe']} | P/S: {f['ps_ratio']} | P/B: {f['pb_ratio']}")
        mc = f['market_cap']
        cap_str = f"${mc / 1e9:.1f}B" if mc and mc > 1e9 else f"${mc / 1e6:.0f}M" if mc else "N/A"
        lines.append(f"Market Cap: {cap_str} | Sector: {f['sector']} | Industry: {f['industry']}")
        pm = f"{f['profit_margin'] * 100:.1f}%" if f['profit_margin'] else "N/A"
        eg = f"{f['earnings_growth'] * 100:.1f}%" if f['earnings_growth'] else "N/A"
        rg = f"{f['revenue_growth'] * 100:.1f}%" if f['revenue_growth'] else "N/A"
        dy = f"{f['dividend_yield'] * 100:.2f}%" if f['dividend_yield'] else "N/A"
        lines.append(f"Profit Margin: {pm} | Earnings Growth: {eg} | Revenue Growth: {rg} | Div Yield: {dy}\n")

    if signals.get("institutional"):
        i = signals["institutional"]
        inst = f"{i['inst_pct'] * 100:.1f}%" if i['inst_pct'] else "N/A"
        ins = f"{i['insider_pct'] * 100:.1f}%" if i['insider_pct'] else "N/A"
        lines.append("## Institutional Ownership")
        lines.append(f"Institutional: {inst} | Insider: {ins} | # Institutions: {i['num_institutions']}\n")

    if signals.get("insider"):
        ins = signals["insider"]
        lines.append("## Insider Activity (Recent)")
        lines.append(f"Buys: {ins['buy_count']} (${ins['buy_value']:,.0f}) | "
                      f"Sells: {ins['sell_count']} (${ins['sell_value']:,.0f}) | "
                      f"Net: {ins['net_sentiment']}\n")

    if signals.get("options"):
        o = signals["options"]
        lines.append("## Options Sentiment")
        lines.append(f"P/C Ratio: {o['put_call_ratio']} | Call Vol: {o['call_volume']:,} | "
                      f"Put Vol: {o['put_volume']:,} | Sentiment: {o['sentiment']}\n")

    if signals.get("profile"):
        pr = signals["profile"]
        lines.append("## Company Profile")
        if pr.get("description"):
            lines.append(pr["description"])
        if pr.get("narrative"):
            lines.append(f"Narrative: {pr['narrative']}\n")

    if signals.get("stress"):
        s = signals["stress"]
        lines.append("## Macro & Stress Context")
        hy = f"{s['hy_spread']}bps" if s.get("hy_spread") is not None else "N/A"
        vix = f"{s['vix']:.1f}" if s.get("vix") is not None else "N/A"
        yc_v = s.get("yield_curve")
        yc = f"{yc_v:+.2f}%" if yc_v is not None else "N/A"
        lines.append(f"System Stress: {s['stress_label']} | VIX: {vix} | HY Spread: {hy} | Yield Curve (10Y-2Y): {yc}")
        if yc_v is not None:
            yc_label = "inverted (recession risk)" if yc_v < 0 else "normal" if yc_v > 0.5 else "flat (caution)"
            lines.append(f"Yield curve is {yc_label}\n")
        else:
            lines.append("")

    if signals.get("whale"):
        w = signals["whale"]
        lines.append("## Smart Money Flow")
        if w.get("institutional_bias") and w["institutional_bias"] != "UNKNOWN":
            score_str = f"{w['institutional_score']:+.2f}%" if w.get("institutional_score") is not None else ""
            cnt = w.get("institution_count", "?")
            lines.append(f"Institutional 13F: {w['institutional_bias']} (weighted avg position change {score_str}, {cnt} funds)")
        if w.get("insider_sentiment") and w["insider_sentiment"] != "UNKNOWN":
            lines.append(f"Insider Activity (6M): {w['insider_sentiment']} ({w.get('insider_buy_pct', '?')}% buys by dollar value)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

RATING_COLORS = {
    "Strong Buy": "#00C853",
    "Buy": "#69F0AE",
    "Hold": "#FFD600",
    "Sell": "#FF5252",
    "Strong Sell": "#D50000",
}


def _signal_tile(col, label: str, sentiment: str, lines: list[str]):
    """Render a single Bloomberg-style signal tile inside a column."""
    bg = {"bullish": "#1B3D2F", "bearish": "#3D1B1B"}.get(sentiment, "#2A2A2A")
    icon = {"bullish": "🟢", "bearish": "🔴", "unavailable": "⚫"}.get(sentiment, "🟡")
    rows_html = "".join(
        f'<div style="font-size:11px;color:#aaa;line-height:1.5;">{ln}</div>' for ln in lines
    )
    with col:
        st.markdown(
            f'<div style="background:{bg};padding:10px 12px;border-radius:8px;margin-bottom:6px;">'
            f'<div style="font-size:10px;color:#666;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
            f'<div style="font-size:14px;margin:2px 0;">{icon}</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )


def _render_signal_transparency(signals: dict):
    """Full Bloomberg-style 8-tile grid showing every raw signal the AI uses."""
    def _fmt_pct(v, decimals=1):
        return f"{v * 100:.{decimals}f}%" if v is not None else "N/A"

    def _fmt_val(v, prefix="", suffix="", decimals=2):
        return f"{prefix}{v:.{decimals}f}{suffix}" if v is not None else "N/A"

    # ── Row 1 ──
    r1c1, r1c2, r1c3 = st.columns(3)

    # Tile 1: Price & Technicals
    p = signals.get("price") or {}
    if p:
        sma_str = (
            f"SMA20 ${p['sma20']} {'✓' if p['above_sma20'] else '✗'}  "
            f"SMA50 ${p['sma50']} {'✓' if p['above_sma50'] else '✗'}  "
            f"SMA200 ${p['sma200']} {'✓' if p['above_sma200'] else '✗'}"
        )
        _p_sentiment, _ = _score_price(p)
        _signal_tile(r1c1, "Price & Technicals", _p_sentiment, [
            f"${p['current']}  ·  RSI(14): {p['rsi14']}",
            f"52W: ${p['52w_low']} – ${p['52w_high']}",
            sma_str,
            f"1Y Return: {p['period_return_pct']:+.1f}%",
        ])
    else:
        _signal_tile(r1c1, "Price & Technicals", "unavailable", ["No price data"])

    # Tile 2: Fundamentals
    f = signals.get("fundamentals") or {}
    if f:
        _f_sentiment, _ = _score_fundamentals(f)
        pe_str = f"P/E {f['pe_ratio']:.1f}" if f.get('pe_ratio') else "P/E N/A"
        fpe_str = f"Fwd P/E {f['forward_pe']:.1f}" if f.get('forward_pe') else "Fwd P/E N/A"
        ps_str = f"P/S {f['ps_ratio']:.1f}" if f.get('ps_ratio') else "P/S N/A"
        pb_str = f"P/B {f['pb_ratio']:.1f}" if f.get('pb_ratio') else "P/B N/A"
        _signal_tile(r1c2, "Fundamentals", _f_sentiment, [
            f"{pe_str}  ·  {fpe_str}",
            f"{ps_str}  ·  {pb_str}",
            f"Margin: {_fmt_pct(f.get('profit_margin'))}  ·  Div: {_fmt_pct(f.get('dividend_yield'))}",
            f"Rev Growth: {_fmt_pct(f.get('revenue_growth'))}  ·  EPS Growth: {_fmt_pct(f.get('earnings_growth'))}",
        ])
    else:
        _signal_tile(r1c2, "Fundamentals", "unavailable", ["No fundamental data"])

    # Tile 3: Institutional Ownership
    i = signals.get("institutional") or {}
    if i:
        _i_sentiment, _ = _score_institutional(i)
        inst_pct = _fmt_pct(i.get('inst_pct'))
        ins_pct = _fmt_pct(i.get('insider_pct'))
        n_inst = i.get('num_institutions') or "N/A"
        _signal_tile(r1c3, "Institutional Ownership", _i_sentiment, [
            f"Institutional: {inst_pct}",
            f"Insider held: {ins_pct}",
            f"# Institutions: {n_inst}",
        ])
    else:
        _signal_tile(r1c3, "Institutional Ownership", "unavailable", ["No ownership data"])

    # ── Row 2 ──
    r2c1, r2c2, r2c3 = st.columns(3)

    # Tile 4: Insider Activity
    ins = signals.get("insider") or {}
    if ins:
        _ins_sentiment, _ = _score_insider(ins)
        bv = f"${ins['buy_value']:,.0f}" if ins.get('buy_value') else "$0"
        sv = f"${ins['sell_value']:,.0f}" if ins.get('sell_value') else "$0"
        _signal_tile(r2c1, "Insider Activity", _ins_sentiment, [
            f"Buys: {ins['buy_count']} ({bv})",
            f"Sells: {ins['sell_count']} ({sv})",
            f"Net: {ins['net_sentiment'].title()}",
        ])
    else:
        _signal_tile(r2c1, "Insider Activity", "unavailable", ["No insider data"])

    # Tile 5: Options Flow
    o = signals.get("options") or {}
    if o:
        _o_sentiment, _ = _score_options(o)
        _signal_tile(r2c2, "Options Flow", _o_sentiment, [
            f"P/C Ratio: {o['put_call_ratio']}  →  {o['sentiment'].title()}",
            f"Call Vol: {o['call_volume']:,}",
            f"Put Vol: {o['put_volume']:,}",
        ])
    else:
        _signal_tile(r2c2, "Options Flow", "unavailable", ["No options data"])

    # Tile 6: Company Profile
    pr = signals.get("profile") or {}
    if pr and pr.get("narrative"):
        narrative = pr["narrative"][:90] + "…" if len(pr.get("narrative", "")) > 90 else pr.get("narrative", "")
        _signal_tile(r2c3, "Company Profile", "neutral", [
            f"Narrative: {narrative}",
        ])
    else:
        _signal_tile(r2c3, "Company Profile", "unavailable", ["No AI profile"])

    # ── Row 3 ──
    r3c1, r3c2, r3c3 = st.columns(3)

    # Tile 7: Macro & Stress
    s = signals.get("stress") or {}
    if s:
        _s_sentiment, _ = _score_stress(s)
        vix_v = s.get('vix')
        yc_v = s.get('yield_curve')
        _signal_tile(r3c1, "Macro & Stress", _s_sentiment, [
            f"Stress: {s.get('stress_label', 'N/A')}",
            f"VIX: {f'{vix_v:.1f}' if vix_v else 'N/A'}  ·  HY Spread: {s.get('hy_spread', 'N/A')} bps",
            f"Yield Curve (10Y-2Y): {f'{yc_v:+.2f}%' if yc_v is not None else 'N/A'}",
        ])
    else:
        _signal_tile(r3c1, "Macro & Stress", "unavailable", ["No macro data"])

    # Tile 8: Smart Money / 13F
    w = signals.get("whale") or {}
    if w:
        _w_sentiment, _ = _score_whale(w)
        ib = w.get('institutional_bias', 'UNKNOWN')
        ic = w.get('institution_count', '?')
        score_str = f" ({w['institutional_score']:+.2f}%)" if w.get('institutional_score') is not None else ""
        ins_sent = w.get('insider_sentiment', 'UNKNOWN')
        ins_pct = w.get('insider_buy_pct', '?')
        _signal_tile(r3c2, "Smart Money / 13F", _w_sentiment, [
            f"13F Bias: {ib}{score_str}",
            f"  ({ic} funds reporting)",
            f"Insider Flow: {ins_sent} ({ins_pct}% buys by $)",
        ])
    else:
        _signal_tile(r3c2, "Smart Money / 13F", "unavailable", ["No 13F data"])

    # Spacer column (r3c3 intentionally empty)


def _render_rating_banner(result: dict):
    """Large colored rating badge + confidence meter."""
    rating = result.get("rating", "Hold")
    confidence = result.get("confidence", 50)
    color = RATING_COLORS.get(rating, COLORS["yellow"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f'<div style="background:{color}; color:#000; padding:24px; '
            f'border-radius:12px; text-align:center;">'
            f'<div style="font-size:14px; font-weight:600; opacity:0.7;">AI RATING</div>'
            f'<div style="font-size:36px; font-weight:800;">{rating.upper()}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**Confidence**")
        st.progress(confidence / 100)
        st.caption(f"{confidence}% confidence based on available signals")
        if result.get("summary"):
            st.markdown(f"*{result['summary']}*")


def _render_signal_scorecard(signals: dict):
    """2x3 grid showing each data category with mini sentiment indicator."""
    st.markdown("### Signal Scorecard")

    categories = [
        ("Price & Technicals", _score_price(signals.get("price"))),
        ("Fundamentals", _score_fundamentals(signals.get("fundamentals"))),
        ("Institutional", _score_institutional(signals.get("institutional"))),
        ("Insider Activity", _score_insider(signals.get("insider"))),
        ("Options Flow", _score_options(signals.get("options"))),
        ("Company Profile", ("neutral", "—") if signals.get("profile") else ("unavailable", "No data")),
        ("Macro Stress", _score_stress(signals.get("stress"))),
        ("13F Flow", _score_whale(signals.get("whale"))),
    ]

    rows = [categories[:3], categories[3:6], categories[6:]]
    for row in rows:
        cols = st.columns(3)
        for col, (label, (sentiment, detail)) in zip(cols, row):
            with col:
                if sentiment == "bullish":
                    icon, bg = "🟢", "#1B3D2F"
                elif sentiment == "bearish":
                    icon, bg = "🔴", "#3D1B1B"
                elif sentiment == "unavailable":
                    icon, bg = "⚫", "#2A2A2A"
                else:
                    icon, bg = "🟡", "#3D3A1B"

                st.markdown(
                    f'<div style="background:{bg}; padding:12px; border-radius:8px; margin-bottom:8px;">'
                    f'<div style="font-size:12px; color:#888;">{label}</div>'
                    f'<div style="font-size:18px;">{icon} {sentiment.title()}</div>'
                    f'<div style="font-size:11px; color:#aaa;">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _render_analysis(result: dict):
    """Bull/bear factors, key levels, recommendation."""
    # Bull vs Bear columns
    st.markdown("### Bull vs Bear Case")
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown(
            f'<div style="background:#1B3D2F; padding:16px; border-radius:8px; '
            f'border-left:4px solid #00C853;">'
            f'<div style="color:#69F0AE; font-weight:700; margin-bottom:8px;">BULLISH FACTORS</div>',
            unsafe_allow_html=True,
        )
        for factor in result.get("bullish_factors", []):
            st.markdown(f"- {factor}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bear:
        st.markdown(
            f'<div style="background:#3D1B1B; padding:16px; border-radius:8px; '
            f'border-left:4px solid #FF5252;">'
            f'<div style="color:#FF5252; font-weight:700; margin-bottom:8px;">BEARISH FACTORS</div>',
            unsafe_allow_html=True,
        )
        for factor in result.get("bearish_factors", []):
            st.markdown(f"- {factor}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Key Levels
    key_levels = result.get("key_levels", {})
    if key_levels.get("support") or key_levels.get("resistance"):
        st.markdown("### Key Levels")
        lc, rc = st.columns(2)
        with lc:
            sup = key_levels.get("support")
            st.metric("Support", f"${sup}" if sup else "N/A")
        with rc:
            res = key_levels.get("resistance")
            st.metric("Resistance", f"${res}" if res else "N/A")

    # Recommendation
    if result.get("recommendation"):
        st.markdown("### Recommendation")
        st.info(result["recommendation"])


# ---------------------------------------------------------------------------
# Signal scoring helpers
# ---------------------------------------------------------------------------

def _score_price(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    rsi = data["rsi14"]
    above_count = sum([data["above_sma20"], data["above_sma50"], data["above_sma200"]])
    if above_count >= 2 and rsi < 70:
        return ("bullish", f"RSI {rsi} · Above {above_count}/3 SMAs")
    elif above_count <= 1 or rsi > 70:
        return ("bearish", f"RSI {rsi} · Above {above_count}/3 SMAs")
    return ("neutral", f"RSI {rsi} · Above {above_count}/3 SMAs")


def _score_fundamentals(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    pe = data.get("pe_ratio")
    growth = data.get("earnings_growth")
    if pe and growth:
        if pe < 25 and growth and growth > 0:
            return ("bullish", f"P/E {pe:.1f} · Growth {growth * 100:.0f}%")
        elif pe > 40 or (growth and growth < 0):
            return ("bearish", f"P/E {pe:.1f} · Growth {growth * 100:.0f}%")
    detail = f"P/E {pe:.1f}" if pe else "Limited data"
    return ("neutral", detail)


def _score_institutional(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    pct = data.get("inst_pct")
    if pct and pct > 0.7:
        return ("bullish", f"{pct * 100:.0f}% institutional")
    elif pct and pct < 0.3:
        return ("bearish", f"{pct * 100:.0f}% institutional")
    detail = f"{pct * 100:.0f}% inst" if pct else "N/A"
    return ("neutral", detail)


def _score_insider(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    s = data["net_sentiment"]
    detail = f"{data['buy_count']} buys / {data['sell_count']} sells"
    return (s, detail)


def _score_options(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    s = data["sentiment"]
    return (s, f"P/C ratio: {data['put_call_ratio']}")


def _score_stress(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    label = data.get("stress_label", "CALM")
    vix = data.get("vix")
    hy = data.get("hy_spread")
    detail = f"VIX {vix:.0f} · HY {hy}bps" if vix and hy else label
    if label == "HIGH":
        return ("bearish", detail)
    elif label == "ELEVATED":
        return ("neutral", detail)
    return ("bullish", detail)


def _score_whale(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    ib = data.get("institutional_bias", "UNKNOWN")
    ins = data.get("insider_sentiment", "UNKNOWN")
    if ib == "UNKNOWN" and ins == "UNKNOWN":
        return ("unavailable", "No data")
    # Combine: both bullish = bullish, both bearish = bearish, else neutral
    signals = [x for x in [ib, ins] if x not in ("UNKNOWN", None)]
    bullish_count = sum(1 for x in signals if x == "BULLISH")
    bearish_count = sum(1 for x in signals if x == "BEARISH")
    sentiment = "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "neutral"
    detail_parts = []
    if ib not in ("UNKNOWN", None):
        detail_parts.append(f"13F: {ib.title()}")
    if ins not in ("UNKNOWN", None):
        pct = data.get("insider_buy_pct", "?")
        detail_parts.append(f"Insider: {pct}% buys")
    return (sentiment, " · ".join(detail_parts) or "N/A")


# ---------------------------------------------------------------------------
# 2-Stage Discounted Cash Flow (Simply Wall St methodology)
# ---------------------------------------------------------------------------
# Levered DCF: discounts FCF to equity at Cost of Equity (CAPM)
# Stage 1 (years 1-10): analyst growth tapering to terminal rate
# Stage 2: Gordon Growth Model terminal value
# Risk-free rate & terminal growth = 5Y avg of 10Y Treasury yield
# Beta: industry unlevered beta re-levered, clamped [0.8, 2.0]
# ERP: Damodaran US equity risk premium (default 5.5%)
# ---------------------------------------------------------------------------

# Sector → approximate unlevered beta (Damodaran Jan 2024 US data)
_SECTOR_UNLEVERED_BETA = {
    "Technology": 1.12,
    "Communication Services": 0.82,
    "Consumer Cyclical": 0.95,
    "Consumer Defensive": 0.58,
    "Healthcare": 0.85,
    "Financial Services": 0.55,
    "Industrials": 0.85,
    "Energy": 0.90,
    "Basic Materials": 0.85,
    "Real Estate": 0.55,
    "Utilities": 0.35,
}

_DEFAULT_ERP = 0.055  # Damodaran US ERP

# Sector profiles ported from Stock Ticker Checker / build_model.py
# Each profile sets per-sector growth clamps, terminal growth, and optional WACC premium
_SECTOR_PROFILES = {
    "High-Growth Tech": {
        "growth_yr1_clamp": (0.03, 0.55),
        "growth_yr610_clamp": (0.02, 0.35),
        "terminal_growth": 0.04,
        "wacc_premium": 0.0,
    },
    "Consumer Cyclical": {
        "growth_yr1_clamp": (0.02, 0.40),
        "growth_yr610_clamp": (0.02, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.0,
    },
    "Consumer Defensive": {
        "growth_yr1_clamp": (0.01, 0.20),
        "growth_yr610_clamp": (0.01, 0.12),
        "terminal_growth": 0.03,
        "wacc_premium": 0.0,
    },
    "Healthcare": {
        "growth_yr1_clamp": (0.02, 0.45),
        "growth_yr610_clamp": (0.02, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.0,
    },
    "Energy/Materials/Industrials": {
        "growth_yr1_clamp": (0.00, 0.35),
        "growth_yr610_clamp": (0.01, 0.15),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
    },
    "Utilities": {
        "growth_yr1_clamp": (0.01, 0.15),
        "growth_yr610_clamp": (0.01, 0.08),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
    },
    "Growth-Platform": {
        "growth_yr1_clamp": (0.15, 0.45),
        "growth_yr610_clamp": (0.08, 0.20),
        "terminal_growth": 0.035,
        "wacc_premium": 0.01,
    },
    "High-SBC-Tech": {
        "growth_yr1_clamp": (0.15, 0.50),
        "growth_yr610_clamp": (0.08, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.015,
    },
    "Cyclical-Commodity": {
        "growth_yr1_clamp": (0.00, 0.20),
        "growth_yr610_clamp": (0.01, 0.10),
        "terminal_growth": 0.02,
        "wacc_premium": 0.0,
    },
    "Mature-Stable": {
        "growth_yr1_clamp": (0.03, 0.08),
        "growth_yr610_clamp": (0.02, 0.05),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
    },
    "Early-Stage-PreProfit": {
        "growth_yr1_clamp": (0.20, 0.60),
        "growth_yr610_clamp": (0.10, 0.30),
        "terminal_growth": 0.03,
        "wacc_premium": 0.03,
    },
}

_SECTOR_ARCHETYPE_MAP = {
    "Technology": "High-Growth Tech",
    "Communication Services": "High-Growth Tech",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Defensive": "Consumer Defensive",
    "Healthcare": "Healthcare",
    "Energy": "Energy/Materials/Industrials",
    "Basic Materials": "Energy/Materials/Industrials",
    "Industrials": "Energy/Materials/Industrials",
    "Utilities": "Utilities",
    "Financial Services": "Energy/Materials/Industrials",
    "Real Estate": "Energy/Materials/Industrials",
}


def _detect_sector_profile(info: dict) -> str:
    """Auto-detect the best-fit sector profile from company fundamentals."""
    sector = info.get("sector", "")
    revenue = info.get("totalRevenue", 0) or 0
    ebit = info.get("ebit", 0) or 0

    # 1. Early-Stage-PreProfit: negative operating income and small revenue
    operating_income = info.get("operatingIncome", ebit) or 0
    if operating_income < 0 and revenue < 2e9:
        return "Early-Stage-PreProfit"

    # 2. High-SBC-Tech: SBC > 10% of revenue in tech/comms
    # Note: yfinance doesn't always expose SBC in info; skip if unavailable
    # (handled by falling through to archetype)

    # 3. Cyclical-Commodity: energy/materials with high price volatility signal
    industry = (info.get("industry", "") or "").lower()
    commodity_keywords = ["oil", "gas", "mining", "steel", "aluminum", "copper",
                          "shipping", "coal", "lumber"]
    if sector in {"Energy", "Basic Materials"} and any(k in industry for k in commodity_keywords):
        return "Cyclical-Commodity"

    # 4. Mature-Stable: low-growth defensives
    fwd_pe = info.get("forwardPE") or 0
    earnings_growth = info.get("earningsGrowth") or 0
    if (sector in {"Consumer Defensive", "Healthcare", "Industrials"}
            and 0 < earnings_growth < 0.08 and fwd_pe > 10):
        return "Mature-Stable"

    # 5. Growth-Platform: subscription/SaaS/DTC keywords
    description = (info.get("longBusinessSummary", "") or "").lower()
    platform_keywords = ["subscription", "saas", "recurring", "platform",
                         "direct-to-consumer", "membership", "telehealth"]
    kw_hits = sum(1 for kw in platform_keywords if kw in description)
    if kw_hits >= 2 and earnings_growth > 0.15:
        return "Growth-Platform"

    # 6. Default: sector archetype
    return _SECTOR_ARCHETYPE_MAP.get(sector, "Consumer Defensive")


def _get_risk_free_rate() -> float:
    """5-year average of 10-year US Treasury yield."""
    try:
        tnx = yf.download("^TNX", period="5y", interval="1mo", progress=False, auto_adjust=True)
        if tnx is not None and not tnx.empty:
            close = tnx["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) > 12:
                avg_yield = close.mean() / 100  # ^TNX is in percent
                return max(0.01, min(0.08, float(avg_yield)))
    except Exception:
        pass
    return 0.04  # fallback


def _compute_dcf(ticker: str, growth_adj: float = 0.0) -> dict | None:
    """
    2-stage levered DCF following Simply Wall St methodology.
    Returns dict with all intermediate values for display.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        return None

    # ── Gather inputs ──
    shares = info.get("sharesOutstanding")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt", 0) or 0
    total_equity = market_cap or (shares * current_price if shares and current_price else None)
    sector = info.get("sector", "")
    company_name = info.get("shortName") or info.get("longName") or ticker
    tax_rate = info.get("effectiveTaxRate") or 0.21
    if isinstance(tax_rate, (int, float)) and tax_rate > 1:
        tax_rate = tax_rate / 100

    if not shares or not current_price:
        return None

    # ── Levered Free Cash Flow ──
    # Try to get FCF from cash flow statement
    try:
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            # yfinance cashflow: rows are items, columns are dates
            fcf_row = None
            for label in ["Free Cash Flow", "FreeCashFlow"]:
                if label in cf.index:
                    fcf_row = cf.loc[label]
                    break
            if fcf_row is None:
                # Compute from operating cash flow - capex
                op_cf = None
                capex = None
                for label in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                    if label in cf.index:
                        op_cf = cf.loc[label]
                        break
                for label in ["Capital Expenditure", "Capital Expenditures"]:
                    if label in cf.index:
                        capex = cf.loc[label]
                        break
                if op_cf is not None and capex is not None:
                    fcf_row = op_cf + capex  # capex is typically negative
                elif op_cf is not None:
                    fcf_row = op_cf

            if fcf_row is not None:
                fcf_values = fcf_row.dropna().sort_index(ascending=False)
                if len(fcf_values) > 0:
                    latest_fcf = float(fcf_values.iloc[0])
                else:
                    return None
            else:
                return None
        else:
            return None
    except Exception:
        return None

    if latest_fcf <= 0:
        # Can't do DCF on negative FCF
        return {"error": "negative_fcf", "latest_fcf": latest_fcf, "company_name": company_name}

    # ── Historical FCF growth (for fallback) ──
    hist_fcf_growth = None
    if fcf_values is not None and len(fcf_values) >= 2:
        oldest = float(fcf_values.iloc[-1])
        newest = float(fcf_values.iloc[0])
        n_years = len(fcf_values) - 1
        if oldest > 0 and n_years > 0:
            hist_fcf_growth = (newest / oldest) ** (1 / n_years) - 1

    # ── Growth rate estimates ──
    # Analyst growth estimate (forward)
    analyst_growth = info.get("earningsGrowth")  # next year
    revenue_growth = info.get("revenueGrowth")

    # Pick best available growth estimate for initial rate
    if analyst_growth and abs(analyst_growth) < 1:
        initial_growth = float(analyst_growth)
    elif revenue_growth and abs(revenue_growth) < 1:
        initial_growth = float(revenue_growth)
    elif hist_fcf_growth and abs(hist_fcf_growth) < 1:
        initial_growth = float(hist_fcf_growth)
    else:
        initial_growth = 0.08  # fallback 8%

    # ── Sector profile (sector-specific growth clamps + terminal growth) ──
    profile_name = _detect_sector_profile(info)
    profile = _SECTOR_PROFILES[profile_name]

    # Clamp initial growth using sector-specific bounds
    _min_g, _max_g = profile["growth_yr1_clamp"]
    initial_growth = max(_min_g, min(_max_g, initial_growth))

    # Apply scenario growth adjustment (bear = negative, bull = positive)
    if growth_adj != 0.0:
        initial_growth = initial_growth * (1 + growth_adj)
        initial_growth = max(min(initial_growth, 0.35), -0.10)

    # ── Risk-free rate (5Y avg of 10Y Treasury) ──
    rf = _get_risk_free_rate()

    # Terminal growth = sector-specific rate, floored at (rf-1%) but capped at 2.5% (long-run GDP)
    # Prevents overvaluation when rates are high (e.g. rf=5% would otherwise force terminal_growth=5%)
    terminal_growth = max(profile["terminal_growth"], min(rf - 0.01, 0.025))

    # ── Cost of Equity (CAPM) + sector WACC premium ──
    unlevered_beta = _SECTOR_UNLEVERED_BETA.get(sector, 0.85)

    # Re-lever beta for company's capital structure
    de_ratio = total_debt / total_equity if total_equity and total_equity > 0 else 0
    levered_beta = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
    levered_beta = max(0.8, min(2.0, levered_beta))  # clamp

    cost_of_equity = rf + levered_beta * _DEFAULT_ERP
    discount_rate = cost_of_equity + profile.get("wacc_premium", 0.0)

    if discount_rate <= terminal_growth:
        # Model breaks down
        return None

    # ── Stage 1: Project 10 years of FCF (2-segment taper) ──
    # Yr 1-5: initial_growth → mid_growth (yr6-10 clamp)
    # Yr 6-10: mid_growth → terminal_growth
    _max_mid = profile["growth_yr610_clamp"][1]
    mid_growth = min(initial_growth, _max_mid)

    projected_fcf = []
    growth_rates = []
    pv_fcf = []
    fcf = latest_fcf

    for year in range(1, 11):
        if year <= 5:
            weight = (year - 1) / 4  # 0 at yr1, 1 at yr5
            g = initial_growth + weight * (mid_growth - initial_growth)
        else:
            weight = (year - 6) / 4  # 0 at yr6, 1 at yr10
            g = mid_growth + weight * (terminal_growth - mid_growth)
        growth_rates.append(g)
        fcf = fcf * (1 + g)
        projected_fcf.append(fcf)
        pv = fcf / (1 + discount_rate) ** year
        pv_fcf.append(pv)

    pv_stage1 = sum(pv_fcf)

    # ── Stage 2: Terminal Value (Gordon Growth Model) ──
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** 10

    # ── Intrinsic Value ──
    total_equity_value = pv_stage1 + pv_terminal
    intrinsic_per_share = total_equity_value / shares

    # ── Discount/Premium ──
    discount_pct = (intrinsic_per_share / current_price - 1) * 100

    # Growth source labels
    growth_source = "Analyst Est." if analyst_growth else ("Hist. FCF" if hist_fcf_growth else "Default")

    return {
        "company_name": company_name,
        "ticker": ticker,
        "current_price": current_price,
        "intrinsic_value": intrinsic_per_share,
        "discount_pct": discount_pct,
        "latest_fcf": latest_fcf,
        "shares": shares,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "total_equity_value": total_equity_value,
        "projected_fcf": projected_fcf,
        "growth_rates": growth_rates,
        "pv_fcf": pv_fcf,
        "terminal_value": terminal_value,
        "discount_rate": discount_rate,
        "risk_free_rate": rf,
        "terminal_growth": terminal_growth,
        "levered_beta": levered_beta,
        "unlevered_beta": unlevered_beta,
        "de_ratio": de_ratio,
        "initial_growth": initial_growth,
        "growth_source": growth_source,
        "sector": sector,
        "erp": _DEFAULT_ERP,
        "profile_name": profile_name,
        "mid_growth": mid_growth,
    }


def _compute_sws_variant(base_dcf: dict) -> dict:
    """
    Recompute intrinsic value using SWS-style smooth linear growth decline.
    Our model uses a 2-segment taper; SWS uses a smooth curve from initial→terminal over 10 years.
    All other inputs (FCF, beta, WACC, terminal growth) are identical to base_dcf.
    """
    initial_g  = base_dcf["initial_growth"]
    terminal_g = base_dcf["terminal_growth"]
    dr         = base_dcf["discount_rate"]
    base_fcf   = base_dcf["latest_fcf"]
    shares     = base_dcf["shares"]
    price      = base_dcf["current_price"]

    if shares <= 0 or dr <= terminal_g:
        return {}

    # Smooth linear decline: Year i growth interpolates from initial_g (yr1) to terminal_g (yr10)
    fcf = base_fcf
    pv_stage1 = 0.0
    for i in range(1, 11):
        # i=1 → g = initial_g, i=10 → g = terminal_g (linear interpolation)
        g = initial_g + (terminal_g - initial_g) * (i - 1) / 9
        fcf = fcf * (1 + g)
        pv_stage1 += fcf / (1 + dr) ** i

    tv    = fcf * (1 + terminal_g) / (dr - terminal_g)
    pv_tv = tv / (1 + dr) ** 10

    iv   = (pv_stage1 + pv_tv) / shares
    disc = (iv / price - 1) * 100

    return {
        "intrinsic_value": iv,
        "discount_pct": disc,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_tv,
    }


def _fmt_big(val: float) -> str:
    """Format large numbers: 1.2T, 345.6B, 12.3M, etc."""
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


def _make_dcf_waterfall(dcf: dict) -> go.Figure:
    """Waterfall chart showing PV of Stage 1, Terminal, Total → per share."""
    pv1 = dcf["pv_stage1"]
    pvt = dcf["pv_terminal"]
    total = dcf["total_equity_value"]

    fig = go.Figure(go.Waterfall(
        x=["PV of 10Y Cash Flows", "PV of Terminal Value", "Total Equity Value"],
        y=[pv1, pvt, total],
        measure=["relative", "relative", "total"],
        text=[_fmt_big(pv1), _fmt_big(pvt), _fmt_big(total)],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text"]),
        connector=dict(line=dict(color=COLORS["grid"])),
        increasing=dict(marker=dict(color=COLORS["green"])),
        totals=dict(marker=dict(color=COLORS["blue"])),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Equity Value Breakdown")
    return fig


def _make_dcf_value_comparison(dcf: dict) -> go.Figure:
    """Bar chart comparing current price vs intrinsic value."""
    price = dcf["current_price"]
    iv = dcf["intrinsic_value"]
    disc = dcf["discount_pct"]

    colors = [COLORS["text_dim"], COLORS["green"] if disc > 0 else COLORS["red"]]

    fig = go.Figure(go.Bar(
        x=["Current Price", "Intrinsic Value (DCF)"],
        y=[price, iv],
        marker_color=colors,
        text=[f"${price:.2f}", f"${iv:.2f}"],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text"]),
        width=0.5,
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(title="Price ($)", gridcolor=COLORS["grid"]),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Price vs Intrinsic Value")
    return fig


def _dcf_verdict(disc: float) -> tuple[str, str, str]:
    """Return (label, color, bg) for a discount_pct value."""
    if disc >= 40:
        return "Significantly Undervalued", "#00C853", "rgba(0,200,83,0.10)"
    elif disc >= 20:
        return "Moderately Undervalued", "#69F0AE", "rgba(105,240,174,0.08)"
    elif disc >= 0:
        return "Slightly Undervalued", COLORS["yellow"], "rgba(255,215,0,0.08)"
    elif disc >= -20:
        return "Slightly Overvalued", "#FF8A65", "rgba(255,138,101,0.08)"
    elif disc >= -40:
        return "Moderately Overvalued", "#FF5252", "rgba(255,82,82,0.08)"
    else:
        return "Significantly Overvalued", "#D50000", "rgba(213,0,0,0.10)"


def _render_dcf(ticker: str) -> dict | None:
    """Render the 2-stage DCF valuation section. Returns {bear, base, bull} dicts or None."""
    st.markdown("### 2-Stage DCF Valuation")
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};margin-top:-8px;font-size:13px;'>"
        "Levered DCF model (Simply Wall St methodology) &mdash; discounts free cash flow to equity at cost of equity</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing DCF scenarios..."):
        dcf_bear = _compute_dcf(ticker, growth_adj=-0.35)
        dcf_base = _compute_dcf(ticker, growth_adj=0.0)
        dcf_bull = _compute_dcf(ticker, growth_adj=+0.35)

    if dcf_base is None:
        st.warning("Could not compute DCF — insufficient financial data available for this ticker.")
        return None

    if dcf_base.get("error") == "negative_fcf":
        st.warning(
            f"DCF not applicable — {dcf_base['company_name']} has negative free cash flow "
            f"({_fmt_big(dcf_base['latest_fcf'])}). The model requires positive FCF."
        )
        return None

    # Use base for all detailed views
    dcf = dcf_base
    price = dcf["current_price"]
    iv = dcf["intrinsic_value"]
    disc = dcf["discount_pct"]

    # ── Undervaluation Spotlight banner (base scenario) ──
    if st.session_state.get("underval_spotlight"):
        _sp_label, _sp_color, _ = _dcf_verdict(disc)
        st.markdown(
            f'<div style="border:2px solid {_sp_color};border-radius:10px;'
            f'padding:20px 28px;margin:0 0 18px 0;background:{_sp_color}18;'
            f'box-shadow:0 0 20px {_sp_color}44;">'
            f'<div style="font-size:42px;font-weight:900;color:{_sp_color};line-height:1;">'
            f'{disc:+.1f}%</div>'
            f'<div style="font-size:16px;font-weight:700;margin-top:6px;color:{_sp_color};">'
            f'{_sp_label}</div>'
            f'<div style="font-size:12px;color:#94a3b8;margin-top:8px;">'
            f'DCF Intrinsic Value <b style="color:#e2e8f0">${iv:.2f}</b> '
            f'vs Current <b style="color:#e2e8f0">${price:.2f}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Bear / Base / Bull scenario columns ──
    _sc1, _sc2, _sc3 = st.columns(3)
    for _col, _dcf_s, _label, _border_color, _adj_pct in [
        (_sc1, dcf_bear, "🐻 Bear", "#ef4444", "−35% growth"),
        (_sc2, dcf_base, "📊 Base", "#f59e0b", "Base growth"),
        (_sc3, dcf_bull, "🐂 Bull", "#22c55e", "+35% growth"),
    ]:
        with _col:
            if _dcf_s and not _dcf_s.get("error"):
                _s_iv   = _dcf_s["intrinsic_value"]
                _s_disc = _dcf_s["discount_pct"]
                _s_verdict, _s_color, _s_bg = _dcf_verdict(_s_disc)
                st.markdown(
                    f'<div style="border:1px solid {_border_color};border-radius:10px;'
                    f'padding:14px 16px;background:{_s_bg};text-align:center;">'
                    f'<div style="font-size:11px;font-weight:700;color:{_border_color};'
                    f'letter-spacing:0.08em;text-transform:uppercase;">{_label}</div>'
                    f'<div style="font-size:11px;color:#64748b;margin-bottom:6px;">{_adj_pct}</div>'
                    f'<div style="font-size:26px;font-weight:700;color:{_s_color};">${_s_iv:.2f}</div>'
                    f'<div style="font-size:18px;font-weight:600;color:{_s_color};">{_s_disc:+.1f}%</div>'
                    f'<div style="font-size:11px;color:#94a3b8;margin-top:4px;">{_s_verdict}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="border:1px solid #334155;border-radius:10px;'
                    f'padding:14px 16px;text-align:center;color:#64748b;">'
                    f'<div style="font-weight:700;">{_label}</div><div>—</div></div>',
                    unsafe_allow_html=True,
                )
    st.caption(f"Current price: **${price:.2f}** &nbsp;·&nbsp; Charts and projections reflect Base scenario.")

    # Sector profile badge
    _wacc_note = f" · +{profile['wacc_premium']*100:.1f}% risk premium" if (profile := _SECTOR_PROFILES.get(dcf.get("profile_name", ""), {})).get("wacc_premium") else ""
    st.caption(f"Sector Profile: **{dcf.get('profile_name', 'Default')}** · Growth cap yr1-5: {_SECTOR_PROFILES.get(dcf.get('profile_name',''), {}).get('growth_yr1_clamp', (0, 0.4))[1]*100:.0f}% · Terminal growth: {dcf['terminal_growth']*100:.1f}%{_wacc_note}")

    # ── Charts ──
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_make_dcf_value_comparison(dcf), use_container_width=True)
    with c2:
        st.plotly_chart(_make_dcf_waterfall(dcf), use_container_width=True)

    # ── Projection Table ──
    with st.expander("10-Year FCF Projection Table", expanded=True):
        table_data = []
        for i in range(10):
            yr = i + 1
            if i < 5:
                src = dcf["growth_source"] if i == 0 else f"Taper → {dcf['mid_growth']*100:.1f}%"
            else:
                src = f"Taper → {dcf['terminal_growth']*100:.1f}%"
            table_data.append({
                "Year": yr,
                "Growth Rate": f"{dcf['growth_rates'][i]*100:.2f}%",
                "Source": src,
                "FCF": _fmt_big(dcf["projected_fcf"][i]),
                "PV of FCF": _fmt_big(dcf["pv_fcf"][i]),
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Terminal value row
        st.markdown(
            f"**Terminal Value (Year 10+):** {_fmt_big(dcf['terminal_value'])} "
            f"&rarr; PV: {_fmt_big(dcf['pv_terminal'])}"
        )

    # ── Assumptions / CAPM ──
    with st.expander("Model Assumptions (CAPM & Inputs)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Risk-Free Rate", f"{dcf['risk_free_rate']*100:.2f}%")
            st.caption("5Y avg of 10Y Treasury")
            st.metric("Equity Risk Premium", f"{dcf['erp']*100:.1f}%")
            st.caption("Damodaran US ERP")
        with a2:
            st.metric("Levered Beta", f"{dcf['levered_beta']:.2f}")
            st.caption(f"Unlevered: {dcf['unlevered_beta']:.2f} ({dcf['sector'] or 'Default'})")
            st.metric("D/E Ratio", f"{dcf['de_ratio']:.2f}")
        with a3:
            st.metric("Cost of Equity (Discount Rate)", f"{dcf['discount_rate']*100:.2f}%")
            st.caption(f"Rf + β × ERP = {dcf['risk_free_rate']*100:.1f}% + {dcf['levered_beta']:.2f} × {dcf['erp']*100:.1f}%")
            st.metric("Terminal Growth", f"{dcf['terminal_growth']*100:.2f}%")
            st.caption(f"Sector profile: {dcf.get('profile_name', 'Default')}")

        st.markdown(
            f"**Latest Reported FCF:** {_fmt_big(dcf['latest_fcf'])} | "
            f"**Initial Growth:** {dcf['initial_growth']*100:.1f}% ({dcf['growth_source']}) | "
            f"**Shares Outstanding:** {dcf['shares']/1e9:.3f}B"
        )

    with st.expander("How This DCF Works"):
        st.markdown(f"""
This is a **2-stage Levered Discounted Cash Flow** model:

**Stage 1 (Years 1-10):** 2-segment taper — years 1-5 taper from the initial rate
({dcf['initial_growth']*100:.1f}%) to a mid-cycle rate ({dcf['mid_growth']*100:.1f}%),
then years 6-10 taper further to the terminal rate ({dcf['terminal_growth']*100:.2f}%).
This reflects the sector-specific reality: a utility should never project at tech-level growth.

**Stage 2 (Year 10+):** Terminal value via Gordon Growth Model — all cash flows from
year 11 to infinity growing at the terminal rate.

**Discount Rate:** Cost of Equity via CAPM (not WACC), since we're discounting levered
free cash flow. High-risk profiles (early-stage, high-SBC tech) carry an additional
risk premium on top of CAPM.

**Sector profile: {dcf.get('profile_name', 'Default')}**
- Growth cap yr1-5: {_SECTOR_PROFILES.get(dcf.get('profile_name',''), {}).get('growth_yr1_clamp', (0, 0.4))[1]*100:.0f}% (prevents inflated growth assumptions)
- Risk-free rate = 5-year average of 10Y Treasury (avoids short-term rate volatility)
- Beta = industry unlevered beta re-levered for company's D/E ratio, clamped to [0.8, 2.0]
- Equity risk premium = {dcf['erp']*100:.1f}% (Damodaran US estimate)

**Limitations:** DCF models are highly sensitive to growth assumptions and discount rates.
Small changes in inputs can produce large swings in intrinsic value. This works best for
companies with positive, relatively stable free cash flows.
        """)

    with st.expander("📐 Simply Wall St Methodology Cross-Reference"):
        st.markdown("""
**Our Model vs Simply Wall St — 2-Stage Levered FCF DCF**

| Parameter | Our Model | Simply Wall St |
|---|---|---|
| Cash flow type | Levered FCF | Levered FCF ✓ |
| Forecast horizon | 10 years | 10 years ✓ |
| Growth estimate | Analyst est → revenue growth → hist FCF CAGR | Analyst consensus, weighted by recency |
| Discount rate | CAPM — Cost of Equity | CAPM — Cost of Equity ✓ |
| Beta | Damodaran unlevered, re-levered, clamped [0.8, 2.0] | Same method ✓ |
| Terminal growth | max(rf rate, sector floor) | 10Y gov bond yield (5Y avg) ≈ same ✓ |
| Sector growth caps | ✅ 11 per-sector profiles | ❌ Not applied |
| Margin of safety | ❌ Raw discount % shown | ✅ 20% threshold for "undervalued" |
| Model variants | 2-stage FCF only | FCF · DDM · Excess Returns (banks) · AFFO (REITs) |
| Stock-based comp | ❌ Not adjusted | Deducted from FCF |

**Where we match SWS:** The core formula is identical — levered FCF discounted at CAPM cost of equity, \
Damodaran unlevered beta re-levered for each company's D/E ratio, Gordon Growth Model terminal value, \
risk-free rate from the 5Y average 10Y Treasury yield.

**Where we go further:** Our 11 sector profiles apply realistic per-sector growth ceilings \
(e.g. Utilities capped at 15% yr1-5, Tech at 55%) — something SWS does not publish in their \
open-source model. This prevents inflated DCF valuations for slow-growth sectors.

**Where SWS goes further:** (1) A 20% margin-of-safety discount is applied before calling a stock \
"undervalued" — we show the raw discount so you can apply your own safety buffer. \
(2) SWS uses alternative models for banks (Excess Returns method) and REITs (AFFO-based), \
which are more accurate than FCF for capital-intensive or asset-heavy structures. \
(3) SWS adjusts FCF for stock-based compensation; we do not.

[📎 Simply Wall St Company Analysis Model — GitHub](https://github.com/SimplyWallSt/Company-Analysis-Model)
        """)

        # ── Live SWS variant comparison ───────────────────────────────────────
        st.markdown("#### Live Valuation Comparison")
        st.caption("Same inputs (FCF, beta, WACC) — different growth curve shape")
        _sws = _compute_sws_variant(dcf)
        if _sws:
            _sws_iv   = _sws["intrinsic_value"]
            _sws_disc = _sws["discount_pct"]
            _our_iv   = dcf["intrinsic_value"]
            _our_disc = dcf["discount_pct"]

            def _verdict(d):
                if d >= 40: return "Significantly Undervalued"
                if d >= 20: return "Moderately Undervalued"
                if d >= 0:  return "Slightly Undervalued"
                if d >= -20: return "Slightly Overvalued"
                if d >= -40: return "Moderately Overvalued"
                return "Significantly Overvalued"

            def _disc_color(d):
                return "#00C853" if d >= 20 else "#69F0AE" if d >= 0 else "#FF5252"

            _comp_c1, _comp_c2 = st.columns(2)
            with _comp_c1:
                _c = _disc_color(_our_disc)
                st.markdown(
                    f'<div style="border:1px solid {_c};border-radius:8px;padding:14px;text-align:center;">'
                    f'<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:6px;">Our Model (2-Stage Taper)</div>'
                    f'<div style="font-size:26px;font-weight:800;color:{_c};">${_our_iv:.2f}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{_c};">{_our_disc:+.1f}%</div>'
                    f'<div style="font-size:12px;color:#94a3b8;margin-top:4px;">{_verdict(_our_disc)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _comp_c2:
                _c2 = _disc_color(_sws_disc)
                st.markdown(
                    f'<div style="border:1px solid {_c2};border-radius:8px;padding:14px;text-align:center;">'
                    f'<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:6px;">SWS-Style (Smooth Decline)</div>'
                    f'<div style="font-size:26px;font-weight:800;color:{_c2};">${_sws_iv:.2f}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{_c2};">{_sws_disc:+.1f}%</div>'
                    f'<div style="font-size:12px;color:#94a3b8;margin-top:4px;">{_verdict(_sws_disc)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Convergence verdict
            _agree = (_our_disc >= 0) == (_sws_disc >= 0)
            _delta = abs(_our_disc - _sws_disc)
            if _agree and _delta < 15:
                _conv_color, _conv_label = "#22c55e", "✅ High Conviction — both models agree"
            elif _agree:
                _conv_color, _conv_label = "#f59e0b", "🟡 Moderate Agreement — same direction, different magnitude"
            else:
                _conv_color, _conv_label = "#ef4444", "⚠️ Model Sensitivity Warning — models disagree on direction"
            st.markdown(
                f'<div style="border:1px solid {_conv_color};border-radius:6px;padding:10px 14px;'
                f'margin-top:10px;background:{_conv_color}18;">'
                f'<span style="font-weight:700;color:{_conv_color};">{_conv_label}</span>'
                f'<span style="font-size:12px;color:#94a3b8;"> · Δ {_delta:.1f}pp between models</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("SWS variant unavailable — insufficient DCF data.")

    return {"bear": dcf_bear, "base": dcf_base, "bull": dcf_bull}


def _render_kelly(ai_result: dict, dcf_scenarios: dict) -> None:
    """Render half-Kelly position sizing card below DCF."""
    dcf_bear = dcf_scenarios.get("bear") or {}
    dcf_bull = dcf_scenarios.get("bull") or {}
    dcf_base = dcf_scenarios.get("base") or {}

    confidence = (ai_result.get("confidence") or 50) / 100.0
    upside   = max(dcf_bull.get("discount_pct", 0), 0) / 100      # bull upside as fraction
    downside = abs(min(dcf_bear.get("discount_pct", 0), 0)) / 100  # bear downside as fraction
    beta     = (dcf_base.get("levered_beta") or 1.0)

    st.markdown("### 📐 Position Sizing (Half-Kelly)")

    if downside < 0.01:
        st.caption("Kelly unavailable — insufficient downside data in bear scenario.")
        return

    b = upside / downside          # win/loss ratio
    p = confidence
    q = 1.0 - p
    kelly_full = (b * p - q) / b
    kelly_half = max(kelly_full * 0.5, 0.0)  # half-Kelly, floor at 0

    capped = False
    if beta > 1.5:
        kelly_half = min(kelly_half, 0.05)
        capped = True

    pct = round(kelly_half * 100, 1)
    _k_color = "#22c55e" if pct >= 5 else "#f59e0b" if pct >= 2 else "#ef4444"

    st.markdown(
        f'<div style="border:1px solid #1e293b;border-radius:10px;padding:16px 20px;'
        f'margin-bottom:12px;background:#0f172a;">'
        f'<div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">'
        f'<div>'
        f'<div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;">Suggested Position</div>'
        f'<div style="font-size:36px;font-weight:900;color:{_k_color};">{pct}%</div>'
        f'<div style="font-size:11px;color:#64748b;">of portfolio</div>'
        f'</div>'
        f'<div style="font-size:12px;color:#94a3b8;line-height:1.8;">'
        f'<b style="color:#cbd5e1;">b</b> = {b:.2f} (upside/downside) &nbsp;|&nbsp; '
        f'<b style="color:#cbd5e1;">p</b> = {p*100:.0f}% (AI confidence) &nbsp;|&nbsp; '
        f'β = {beta:.2f}'
        f'{"<br><span style=\'color:#f59e0b;font-size:11px;\'>⚠ High-beta cap applied (β > 1.5)</span>" if capped else ""}'
        f'<br><span style="color:#475569;">Half-Kelly · conservative sizing · not financial advice</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
