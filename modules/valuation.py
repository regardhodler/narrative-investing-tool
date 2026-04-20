import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf  # kept for: options volume, institutional_holders, _collect_signals retry loop

from services.market_data import fetch_ohlcv_single, get_yf_info_safe
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout
from utils.components import render_rr_score_mode_toggle, render_intel_health_bar, apply_confidence_penalty


def render():
    import os
    _has_xai = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    render_rr_score_mode_toggle(key="valuation_rr_score_mode_ui", compact=True)
    render_intel_health_bar(compact=True)

    from utils.ai_tier import TIER_OPTS as _VAL_MAIN_OPTS, TIER_MAP as _VAL_MAIN_MAP
    _tier_badges = {
        "⚡ Freeloader Mode": '<span style="font-size:11px;background:#2A3040;color:#888;padding:2px 7px;border-radius:3px;">⚡ Freeloader Mode</span>',
        "🧠 Regard Mode": '<span style="font-size:11px;background:#FF8811;color:#000;padding:2px 7px;border-radius:3px;font-weight:700;">🧠 Grok 4.1</span>',
        "👑 Highly Regarded Mode": '<span style="font-size:11px;background:linear-gradient(90deg,#c89b3c,#f0d060);color:#000;padding:2px 7px;border-radius:3px;font-weight:700;">👑 Sonnet</span>',
    }

    # ── Resolve ticker + signals first so DCF can show at the top ─────────────
    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view AI valuation.")
        return

    with st.spinner("Collecting market signals..."):
        try:
            signals = _collect_signals(ticker)
        except Exception as _e:
            st.warning(f"Could not collect data for **{ticker}** — {_e}. yfinance may be rate-limiting; try again in a moment.")
            if st.button("🔄 Retry", key="valuation_retry"):
                st.rerun()
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

    # ── Signal Coverage ────────────────────────────────────────────────────────
    from utils.components import render_signal_coverage as _render_sig_cov_val
    _render_sig_cov_val()

    # Always-on spotlight so valuation opens with the DCF discount banner.
    st.session_state["underval_spotlight"] = True

    _prev_tier = st.session_state.get("_val_tier_prev")
    from utils.ai_tier import render_ai_tier_selector as _val_main_tier
    _use_claude, _cl_model = _val_main_tier(
        key="val_engine_radio",
        label="Engine",
        recommendation="🧠 Regard recommended for valuation · 👑 Highly Regarded for high-conviction positions",
    )
    _selected_val_tier = st.session_state.get("val_engine_radio", "⚡ Freeloader Mode")
    st.session_state["_val_tier_prev"] = _selected_val_tier

    _badge_html = f' {_tier_badges.get(_selected_val_tier, "")}'
    st.markdown(
        f'<h2 style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:20px;'
        f'font-weight:700;margin-bottom:4px;">AI VALUATION &amp; RECOMMENDATION{_badge_html}</h2>',
        unsafe_allow_html=True,
    )

    # ── Data Quality Warning ───────────────────────────────────────────────────
    _dq_missing, _dq_stale = [], []
    _dq_crit = {
        "_regime_context":    "Risk Regime",
        "_dominant_rate_path": "Fed Rate Path",
        "_rate_path_probs":   "Rate Probs",
    }
    _dq_ts = {
        "_regime_context_ts":   ("Risk Regime",  4),
        "_rate_path_probs_ts":  ("Fed Rate Path", 8),
        "_fed_plays_result_ts": ("Fed Plays",    8),
        "_doom_briefing_ts":    ("Doom Briefing", 12),
    }
    for _k, _lbl in _dq_crit.items():
        if not st.session_state.get(_k):
            _dq_missing.append(_lbl)
    for _tsk, (_lbl, _max_h) in _dq_ts.items():
        _ts = st.session_state.get(_tsk)
        if _ts:
            _age_h = (datetime.now() - _ts).total_seconds() / 3600
            if _age_h > _max_h:
                _dq_stale.append(f"{_lbl} ({_age_h:.0f}h old)")
    if _dq_missing or _dq_stale:
        _dq_parts = []
        if _dq_missing:
            _dq_parts.append(f"<b>Missing:</b> {', '.join(_dq_missing)}")
        if _dq_stale:
            _dq_parts.append(f"<b>Stale:</b> {', '.join(_dq_stale)}")
        st.markdown(
            f'<div style="background:#1a0d00;border:1px solid #f59e0b55;border-radius:6px;'
            f'padding:8px 14px;margin-bottom:8px;font-size:11px;">'
            f'<span style="color:#f59e0b;font-weight:700;">⚠ Data Quality</span>'
            f'<span style="color:#94a3b8;margin-left:8px;">{" &nbsp;·&nbsp; ".join(_dq_parts)}</span>'
            f'<span style="color:#64748b;margin-left:8px;">— run ⚡ Quick Intel Run to refresh</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Earnings Intelligence panel ─────────────────────────────────────────
    from services.market_data import fetch_earnings_intelligence as _fetch_ei
    _ei = _fetch_ei(ticker)
    if _ei:
        with st.expander("📅 EARNINGS INTELLIGENCE", expanded=True):
            _ei_cols = st.columns([1.2, 1.2, 1.2, 1.4])

            # Next earnings countdown
            _ne = _ei.get("next_earnings") or {}
            _ne_days = _ne.get("days_away")
            if _ne_days is not None:
                _ne_color = "#ef4444" if _ne_days <= 7 else ("#f59e0b" if _ne_days <= 21 else "#94a3b8")
                _ei_cols[0].markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                    f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">NEXT EARNINGS</div>'
                    f'<div style="font-size:20px;font-weight:700;color:{_ne_color};">{_ne_days}d</div>'
                    f'<div style="font-size:11px;color:#64748b;">{_ne.get("date","—")}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                _ei_cols[0].markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                    f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">NEXT EARNINGS</div>'
                    f'<div style="font-size:16px;font-weight:600;color:#475569;">—</div>'
                    f'</div>', unsafe_allow_html=True
                )

            # Analyst consensus
            _an = _ei.get("analyst") or {}
            _an_buy = _an.get("buy", 0); _an_hold = _an.get("hold", 0); _an_sell = _an.get("sell", 0)
            _an_total = _an_buy + _an_hold + _an_sell
            _an_target = _an.get("mean_target"); _an_upside = _an.get("upside_pct")
            _an_upside_color = "#22c55e" if (_an_upside or 0) >= 0 else "#ef4444"
            _an_target_str = f"${_an_target:,.2f}" if _an_target else "—"
            _an_upside_str = f"{_an_upside:+.1f}%" if _an_upside is not None else ""
            _ei_cols[1].markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">ANALYST CONSENSUS</div>'
                f'<div style="font-size:13px;font-weight:700;color:#f1f5f9;">{_an_buy}B / {_an_hold}H / {_an_sell}S'
                + (f' <span style="color:#94a3b8;font-weight:400;font-size:11px;">({_an_total} analysts)</span>' if _an_total else "")
                + f'</div>'
                f'<div style="font-size:12px;color:#94a3b8;">Target: {_an_target_str}'
                + (f' <span style="color:{_an_upside_color};font-weight:600;">{_an_upside_str}</span>' if _an_upside_str else "")
                + f'</div>'
                f'</div>', unsafe_allow_html=True
            )

            # Expected move from IV
            _em = _ei.get("expected_move") or {}
            _em_pct = _em.get("pct"); _em_dollar = _em.get("dollar"); _em_iv = _em.get("iv"); _em_dte = _em.get("dte")
            _em_pct_str = f"±{_em_pct:.1f}%" if _em_pct else "—"
            _em_dollar_str = f"±${_em_dollar:.2f}" if _em_dollar else ""
            _em_sub = f"IV {_em_iv*100:.0f}% · {_em_dte}d expiry" if (_em_iv and _em_dte) else ""
            _ei_cols[2].markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">EXPECTED MOVE (IV)</div>'
                f'<div style="font-size:20px;font-weight:700;color:#a78bfa;">{_em_pct_str}</div>'
                + (f'<div style="font-size:11px;color:#64748b;">{_em_dollar_str}  {_em_sub}</div>' if (_em_dollar_str or _em_sub) else "")
                + f'</div>', unsafe_allow_html=True
            )

            # EPS history
            _eps_hist = _ei.get("eps_history") or []
            if _eps_hist:
                _beat_str = " ".join(
                    ("✅" if q.get("beat") else "❌") for q in _eps_hist
                )
                _last_surp = _eps_hist[0].get("surprise_pct") if _eps_hist else None
                _surp_color = "#22c55e" if (_last_surp or 0) >= 0 else "#ef4444"
                _surp_str = f"{_last_surp:+.1f}%" if _last_surp is not None else "—"
                _ei_cols[3].markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                    f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">EPS HISTORY (4Q)</div>'
                    f'<div style="font-size:16px;letter-spacing:4px;">{_beat_str}</div>'
                    f'<div style="font-size:12px;color:#94a3b8;">Last surprise: <span style="color:{_surp_color};font-weight:700;">{_surp_str}</span></div>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                _ei_cols[3].markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:10px 14px;">'
                    f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">EPS HISTORY (4Q)</div>'
                    f'<div style="font-size:16px;color:#475569;">—</div>'
                    f'</div>', unsafe_allow_html=True
                )

            # Quarterly EPS table
            if _eps_hist:
                st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
                _q_rows = ""
                for _q in _eps_hist:
                    _qsurp = _q.get("surprise_pct")
                    _qcolor = "#22c55e" if (_qsurp or 0) >= 0 else "#ef4444"
                    _qbeat_badge = (
                        '<span style="background:#14532d;color:#86efac;padding:1px 6px;border-radius:3px;font-size:10px;">BEAT</span>'
                        if _q.get("beat")
                        else '<span style="background:#7f1d1d;color:#fca5a5;padding:1px 6px;border-radius:3px;font-size:10px;">MISS</span>'
                    )
                    _qest = f"${_q.get('estimate', 0):.2f}" if _q.get('estimate') is not None else "—"
                    _qact = f"${_q.get('actual', 0):.2f}" if _q.get('actual') is not None else "—"
                    _qsurp_str = f"{_qsurp:+.1f}%" if _qsurp is not None else "—"
                    _q_rows += (
                        f'<tr style="border-bottom:1px solid #1e293b;">'
                        f'<td style="padding:5px 10px;color:#94a3b8;font-size:12px;">{_q.get("period","—")}</td>'
                        f'<td style="padding:5px 10px;color:#e2e8f0;font-size:12px;">{_qest}</td>'
                        f'<td style="padding:5px 10px;color:#e2e8f0;font-size:12px;">{_qact}</td>'
                        f'<td style="padding:5px 10px;font-size:12px;color:{_qcolor};">{_qsurp_str}</td>'
                        f'<td style="padding:5px 10px;">{_qbeat_badge}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:6px;overflow:hidden;">'
                    f'<thead><tr style="background:#1e293b;">'
                    f'<th style="padding:6px 10px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">PERIOD</th>'
                    f'<th style="padding:6px 10px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">ESTIMATE</th>'
                    f'<th style="padding:6px 10px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">ACTUAL</th>'
                    f'<th style="padding:6px 10px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">SURPRISE</th>'
                    f'<th style="padding:6px 10px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;"></th>'
                    f'</tr></thead><tbody>{_q_rows}</tbody></table>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

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

    # Inject AI Regime Plays (sector/stock picks from Risk Regime module)
    _rp_result = st.session_state.get("_rp_plays_result")
    if _rp_result:
        _rp_sectors = ", ".join(s.get("name", "") for s in _rp_result.get("sectors", [])[:3])
        _rp_stocks  = ", ".join(s.get("ticker", "") for s in _rp_result.get("stocks", [])[:4])
        signals_text += f"\nAI Regime Plays: Sectors={_rp_sectors} | Stocks={_rp_stocks} | {_rp_result.get('rationale', '')}"

    # Inject Policy Transmission narration
    _narration = st.session_state.get("_chain_narration")
    if _narration:
        signals_text += f"\nPolicy Transmission: {_narration}"

    # Inject Cross-Signal Discovery Plays (from Narrative Discovery module)
    _disc_plays = st.session_state.get("_plays_result")
    if _disc_plays:
        _dp_sectors = ", ".join(s.get("name", "") for s in _disc_plays.get("sectors", [])[:3])
        _dp_stocks  = ", ".join(s.get("ticker", "") for s in _disc_plays.get("stocks", [])[:4])
        signals_text += f"\nCross-Signal Discovery Plays: Sectors={_dp_sectors} | Stocks={_dp_stocks} | {_disc_plays.get('rationale', '')}"

    # Inject Doom Briefing risk assessment (cap at 600 chars to avoid token bloat)
    _doom = st.session_state.get("_doom_briefing")
    if _doom:
        signals_text += f"\nRisk Intelligence Briefing: {_doom[:600]}"

    # Inject Whale Activity Summary (institutional flow context)
    _whale_sum = st.session_state.get("_whale_summary")
    if _whale_sum:
        signals_text += f"\nInstitutional Whale Activity: {_whale_sum[:500]}"
    _activism_dig = st.session_state.get("_activism_digest")
    if _activism_dig:
        signals_text += f"\nActivism Campaigns (13D): {_activism_dig[:400]}"

    # Inject Sector×Regime digest (sector momentum vs macro regime confirmation)
    _srd_val = st.session_state.get("_sector_regime_digest")
    if _srd_val:
        signals_text += f"\nSector×Regime: {_srd_val[:350]}"

    # Inject Trending Narratives (market attention signals from Google Trends + news)
    _tn_val = st.session_state.get("_trending_narratives")
    if _tn_val:
        _tn_lines = [
            f"- {n['narrative']} ({n.get('conviction','')}) — tickers: {', '.join(n.get('tickers', []))}"
            for n in _tn_val[:3]
        ]
        signals_text += "\n\nTRENDING NARRATIVES (market attention signals):\n" + "\n".join(_tn_lines)

    # Inject Auto-Trending ticker groups (Yahoo Finance price movers)
    _atg_val = st.session_state.get("_auto_trending_groups")
    if _atg_val:
        _atg_lines = [
            f"- {g['narrative']} ({g.get('conviction','')}, {g.get('regime_alignment','')}) — {', '.join(g.get('tickers', []))}"
            for g in _atg_val[:3]
        ]
        signals_text += "\n\nTRENDING PRICE MOVERS (Yahoo Finance themes):\n" + "\n".join(_atg_lines)

    # Inject ticker-level smart money signals (only if they match the current ticker)
    _opt_s = st.session_state.get("_options_sentiment") or {}
    if _opt_s.get("ticker", "").upper() == ticker.upper():
        signals_text += (
            f"\nOptions Sentiment (P/C {_opt_s.get('pc_ratio', 0):.2f}): {_opt_s.get('sentiment', '')}"
            f" | Call Vol {_opt_s.get('call_vol', 0):,} vs Put Vol {_opt_s.get('put_vol', 0):,}"
        )

    _ua_s = st.session_state.get("_unusual_activity_sentiment") or {}
    if _ua_s.get("ticker", "").upper() == ticker.upper():
        signals_text += (
            f"\nUnusual Options Activity: {_ua_s.get('sentiment', '')} "
            f"({_ua_s.get('call_pct', 0):.0f}% calls / {_ua_s.get('put_pct', 0):.0f}% puts, "
            f"{_ua_s.get('flagged_contracts', 0)} flagged contracts)"
        )

    # Inject macro options flow (SPY-level market-wide P/C, gamma, put wall)
    _of_ctx_v = st.session_state.get("_options_flow_context") or {}
    if _of_ctx_v:
        signals_text += (
            f"\nMacro Options Flow (SPY): {_of_ctx_v.get('label', '')} "
            f"(score {_of_ctx_v.get('options_score', 50)}/100) | "
            f"{_of_ctx_v.get('action_bias', '')[:150]}"
        )

    # Inject Fear & Greed Index (contrarian sentiment)
    _fg_v = st.session_state.get("_fear_greed") or {}
    if _fg_v:
        _fg_chg = _fg_v.get("change_7d", 0)
        _fg_chg_str = f" ({_fg_chg:+d} vs last week)" if _fg_chg else ""
        signals_text += (
            f"\nFear & Greed Index: {_fg_v.get('score','?')}/100 — {_fg_v.get('label','?')}{_fg_chg_str}"
            f" | Contrarian implication: {'Oversold — potential bounce' if _fg_v.get('score',50) <= 25 else ('Overbought — potential reversal' if _fg_v.get('score',50) >= 75 else 'Neutral crowd sentiment')}"
        )

    # Inject AAII Sentiment (contrarian weekly survey)
    _aaii_v = st.session_state.get("_aaii_sentiment") or {}
    if _aaii_v:
        signals_text += (
            f"\nAAII Investor Sentiment (weekly): Bull {_aaii_v.get('bull_pct','?')}% / Bear {_aaii_v.get('bear_pct','?')}%"
            f" | Spread {_aaii_v.get('bull_bear_spread','?'):+}% ({_aaii_v.get('label','?')})"
        )

    # Inject VIX term structure (volatility curve shape)
    _vc_v = st.session_state.get("_vix_curve") or {}
    if _vc_v:
        signals_text += (
            f"\nVIX Term Structure: {_vc_v.get('structure','?')}"
            f" | VIX9D {_vc_v.get('vix9d','?')} / VIX {_vc_v.get('vix','?')} / VIX3M {_vc_v.get('vix3m','?')} / VIX6M {_vc_v.get('vix6m','?')}"
        )

    _inst_b = st.session_state.get("_institutional_bias") or {}
    if _inst_b.get("ticker", "").upper() == ticker.upper():
        signals_text += (
            f"\nInstitutional Bias: {_inst_b.get('bias', '')} "
            f"(weighted position change {_inst_b.get('weighted_pct', 0):+.1f}%)"
        )

    _ins_f = st.session_state.get("_insider_net_flow") or {}
    if _ins_f.get("ticker", "").upper() == ticker.upper():
        signals_text += (
            f"\nInsider Net Flow: {_ins_f.get('bias', '')} "
            f"({_ins_f.get('buy_pct', 50):.0f}% buys by $ value, {_ins_f.get('n_trades', 0)} trades)"
        )

    _cong_b = st.session_state.get("_congress_bias") or {}
    if _cong_b.get("ticker", "").upper() == ticker.upper():
        signals_text += (
            f"\nCongress Trading Bias: {_cong_b.get('bias', '')} "
            f"({_cong_b.get('buy_pct', 50):.0f}% cumulative buy bias)"
        )

    # Inject QIR Earnings Risk for current ticker
    _qir_er = st.session_state.get("_qir_earnings_risk") or []
    _ticker_er = next((e for e in _qir_er if e.get("ticker", "").upper() == ticker.upper()), None)
    if _ticker_er:
        _em_str = (
            f", options pricing ±{_ticker_er['expected_move_pct']:.1f}% move"
            f" (${_ticker_er.get('expected_move_dollar','?')})"
            if _ticker_er.get('expected_move_pct') else ""
        )
        signals_text += f"\nEarnings Risk: {ticker} reports in {_ticker_er['days_away']}d{_em_str}"

    # Inject StockTwits crowd sentiment
    _st_val = st.session_state.get("_stocktwits_digest") or {}
    if _st_val:
        _st_ticker_item = next(
            (t for t in _st_val.get("trending_tickers", [])
             if t.get("symbol", "").upper() == ticker.upper()),
            None
        )
        if _st_ticker_item:
            _st_why = (_st_ticker_item.get("trending_summary") or "")[:120]
            signals_text += (
                f"\nStockTwits Crowd ({ticker}): {_st_ticker_item.get('sentiment_label','Neutral')} "
                f"({_st_ticker_item.get('sentiment_score', 50)}/100 sentiment score)"
                + (f" — {_st_why}" if _st_why else "")
            )
        else:
            signals_text += (
                f"\nStockTwits Market Mood: {_st_val.get('avg_sentiment_score', _st_val.get('overall_bull_pct','?'))}/100 "
                f"({_st_val.get('market_mood','?')}) — {ticker} not in top social trending"
            )

    # Inject Price Momentum signal (from Narrative Pulse)
    _pm = st.session_state.get("_price_momentum") or {}
    if _pm.get("ticker", "").upper() == ticker.upper():
        _ma_parts = []
        for _k, _v in (_pm.get("ma_signals") or {}).items():
            _period = _k.replace("sma_", "SMA")
            _dir = "above" if _v["above"] else "below"
            _ma_parts.append(f"{_period} ${_v['value']:.2f} ({_dir})")
        signals_text += (
            f"\nPrice Momentum: RSI {_pm.get('rsi', 0):.1f} ({_pm.get('rsi_label', '')})"
            f" | MA Trend: {_pm.get('ma_trend', '')}"
            + (f" | {', '.join(_ma_parts)}" if _ma_parts else "")
            + f" | Vol ratio vs avg: {_pm.get('vol_ratio', 1.0):.2f}x"
        )

    # Inject Filing Digest (from EDGAR Scanner)
    _fd = st.session_state.get("_filing_digest") or {}
    if _fd.get("ticker", "").upper() == ticker.upper() and _fd.get("summary"):
        signals_text += (
            f"\nRecent SEC Filing ({_fd.get('form_type', '')} {_fd.get('date', '')}): "
            f"{str(_fd['summary'])[:600]}"
        )

    # Inject Sector Rotation context (live momentum + regime alignment for this ticker's sector)
    _sr_ctx = signals.get("sector_rotation")
    if _sr_ctx:
        signals_text += f"\n{_sr_ctx}"

    # Inject Macro Regime context (the most critical signal — regime determines sector rotation)
    _regime_ctx_val = st.session_state.get("_regime_context")
    if _regime_ctx_val:
        _r_regime = _regime_ctx_val.get("regime", "")
        _r_score = _regime_ctx_val.get("score", 0.0)
        _r_quad = _regime_ctx_val.get("quadrant", "")
        _r_sig = (_regime_ctx_val.get("signal_summary") or "")[:500]
        signals_text += (
            f"\nMacro Regime: {_r_regime} (score {_r_score:+.2f})"
            f" | Quadrant: {_r_quad}"
            f"\nRegime Signal Breakdown: {_r_sig}"
        )

    # Inject Fed Funds Rate
    _ff_rate_val = st.session_state.get("_fed_funds_rate")
    if _ff_rate_val is not None:
        signals_text += f"\nCurrent Fed Funds Rate: {_ff_rate_val:.2f}%"

    # Inject Earnings Intelligence into signals_text
    if _ei:
        _ei_lines = []
        _ne2 = _ei.get("next_earnings") or {}
        if _ne2.get("days_away") is not None:
            _ei_lines.append(f"Next earnings: {_ne2.get('date','—')} ({_ne2['days_away']}d away)")
        _an2 = _ei.get("analyst") or {}
        if _an2.get("mean_target"):
            _ei_lines.append(
                f"Analyst consensus: {_an2.get('buy',0)}B/{_an2.get('hold',0)}H/{_an2.get('sell',0)}S"
                f" | Mean target ${_an2['mean_target']:.2f}"
                + (f" ({_an2['upside_pct']:+.1f}% upside)" if _an2.get('upside_pct') is not None else "")
            )
        _eps2 = _ei.get("eps_history") or []
        if _eps2:
            _beat_count = sum(1 for q in _eps2 if q.get("beat"))
            _last_surp2 = _eps2[0].get("surprise_pct")
            _ei_lines.append(
                f"EPS beat rate: {_beat_count}/{len(_eps2)} last quarters"
                + (f" | Most recent surprise: {_last_surp2:+.1f}%" if _last_surp2 is not None else "")
            )
        _em2 = _ei.get("expected_move") or {}
        if _em2.get("pct"):
            _ei_lines.append(
                f"Options-implied expected move: ±{_em2['pct']:.1f}%"
                + (f" (IV {_em2['iv']*100:.0f}%, {_em2['dte']}d expiry)" if _em2.get('iv') and _em2.get('dte') else "")
            )
        if _ei_lines:
            signals_text += "\n\n## Earnings Intelligence\n" + "\n".join(f"- {l}" for l in _ei_lines)

    # Inject Elliott Wave + Wyckoff price action signals
    _pa = signals.get("price_action") or {}
    if _pa:
        _pa_lines = []
        if _pa.get("wyckoff_phase"):
            _wy_conf = f" ({_pa['wyckoff_confidence']*100:.0f}% conf)" if _pa.get("wyckoff_confidence") else ""
            _wy_sub = f" sub-{_pa['wyckoff_sub_phase']}" if _pa.get("wyckoff_sub_phase") else ""
            _wy_target = f" — cause target ${_pa['wyckoff_cause_target']:,.2f}" if _pa.get("wyckoff_cause_target") else ""
            _pa_lines.append(f"Wyckoff: {_pa['wyckoff_phase']}{_wy_sub}{_wy_conf}{_wy_target}")
        if _pa.get("ew_wave"):
            _ew_dir = f" ({_pa['ew_direction']})" if _pa.get("ew_direction") else ""
            _ew_conf = f" {_pa['ew_confidence']*100:.0f}% conf" if _pa.get("ew_confidence") else ""
            _ew_target = f" — primary target ${_pa['ew_primary_target']:,.2f} ({_pa.get('ew_primary_prob',0)*100:.0f}%)" if _pa.get("ew_primary_target") else ""
            _ew_inv = f" | invalidation ${_pa['ew_invalidation']:,.2f}" if _pa.get("ew_invalidation") else ""
            _pa_lines.append(f"Elliott Wave: Wave {_pa['ew_wave']}{_ew_dir}{_ew_conf}{_ew_target}{_ew_inv}")
        if _pa_lines:
            signals_text += "\n\n## Price Action (Elliott Wave & Wyckoff)\n" + "\n".join(f"- {l}" for l in _pa_lines)

    # Inject Fed Rate-Path Plays (sectors/bonds to own given dominant rate path)
    _fed_plays_val = st.session_state.get("_fed_plays_result")
    if _fed_plays_val:
        _fp_sectors = ", ".join(s.get("name", "") for s in _fed_plays_val.get("sectors", [])[:3])
        _fp_bonds = ", ".join(s.get("ticker", "") for s in _fed_plays_val.get("bonds", [])[:3])
        signals_text += (
            f"\nRate-Path Plays: Favored Sectors={_fp_sectors}"
            f" | Bonds={_fp_bonds}"
            f" | {_fed_plays_val.get('rationale', '')[:200]}"
        )

    # Inject Portfolio Risk Snapshot (from Quick Intel Run or Trade Journal risk matrix)
    _port_risk = st.session_state.get("_portfolio_risk_snapshot") or {}
    if _port_risk:
        _pr_parts = []
        if _port_risk.get("beta") is not None:
            _pr_parts.append(f"Portfolio Beta {_port_risk['beta']} | VaR95 {_port_risk.get('var_95_pct')}% | CVaR95 {_port_risk.get('cvar_95_pct')}%")
        _sw = _port_risk.get("sector_weights") or {}
        if _sw:
            _pr_parts.append("Sector weights: " + ", ".join(f"{s} {w}%" for s, w in sorted(_sw.items(), key=lambda x: -x[1])[:4]))
        _rf = _port_risk.get("risk_flags") or []
        if _rf:
            _pr_parts.append("Portfolio risk flags: " + "; ".join(f.replace("⚠ ", "") for f in _rf))
        _stress = _port_risk.get("stress_scenarios") or []
        if _stress:
            _worst = min(_stress, key=lambda s: s["port_impact_pct"])
            _pr_parts.append(f"Worst stress scenario: {_worst['scenario']} {_worst['port_impact_pct']:+.1f}%")
        if _pr_parts:
            signals_text += "\n\nPORTFOLIO RISK CONTEXT:\n" + "\n".join(f"- {l}" for l in _pr_parts)

    # Full signal transparency expander
    with st.expander("📊 Full Signal Transparency (AI Inputs)", expanded=False):
        _render_signal_transparency(signals)

    # ── Cross-module injected signals (visible preview) ────────────────────────
    _inj_rows = []
    _rc_v = st.session_state.get("_regime_context") or {}
    if _rc_v.get("regime"):
        _val_leading_lbl = _rc_v.get("leading_label", "Aligned")
        _val_leading_div = _rc_v.get("leading_divergence", 0) or 0
        _val_leading_sfx = f" · {_val_leading_lbl} ({_val_leading_div:+d})" if _val_leading_lbl != "Aligned" else ""
        _inj_rows.append(("Macro Regime", f"{_rc_v['regime']} · {_rc_v.get('quadrant','')} · score {_rc_v.get('score',0):+.2f}{_val_leading_sfx}"))
    _tac_v = st.session_state.get("_tactical_context") or {}
    if _tac_v:
        _inj_rows.append(("Tactical Regime", f"{_tac_v.get('tactical_score','?')}/100 · {_tac_v.get('label','')} — {_tac_v.get('action_bias','')[:80]}"))
    _doom_v = st.session_state.get("_doom_briefing", "")
    if _doom_v:
        _inj_rows.append(("Doom Briefing", _doom_v[:120] + "…"))
    _whale_v = st.session_state.get("_whale_summary", "")
    if _whale_v:
        _inj_rows.append(("Whale Activity", _whale_v[:120] + "…"))
    _act_v = st.session_state.get("_activism_digest", "")
    if _act_v:
        _inj_rows.append(("Activism (13D)", _act_v[:120] + "…"))
    _srd_v2 = st.session_state.get("_sector_regime_digest", "")
    if _srd_v2:
        _inj_rows.append(("Sector×Regime", _srd_v2[:120] + "…"))
    _of_v = st.session_state.get("_options_flow_context") or {}
    if _of_v:
        _inj_rows.append(("Macro Options Flow", f"{_of_v.get('label','')} · score {_of_v.get('options_score','?')}/100 — {_of_v.get('action_bias','')[:80]}"))
    _fg_v2 = st.session_state.get("_fear_greed") or {}
    if _fg_v2:
        _inj_rows.append(("Fear & Greed", f"{_fg_v2.get('score','?')}/100 — {_fg_v2.get('label','')}"))
    _aaii_v2 = st.session_state.get("_aaii_sentiment") or {}
    if _aaii_v2:
        _inj_rows.append(("AAII Sentiment", f"Bull {_aaii_v2.get('bull_pct','?')}% / Bear {_aaii_v2.get('bear_pct','?')}% (spread {_aaii_v2.get('bull_bear_spread','?'):+}%)"))
    _vc_v2 = st.session_state.get("_vix_curve") or {}
    if _vc_v2:
        _inj_rows.append(("VIX Term Structure", f"{_vc_v2.get('structure','?')} — 9D:{_vc_v2.get('vix9d','?')} VIX:{_vc_v2.get('vix','?')} 3M:{_vc_v2.get('vix3m','?')} 6M:{_vc_v2.get('vix6m','?')}"))
    _ce_v2 = st.session_state.get("_current_events_digest", "")
    if _ce_v2:
        _inj_rows.append(("Current Events", _ce_v2[:120] + "…"))

    if _inj_rows:
        _inj_html = "".join(
            f'<tr>'
            f'<td style="padding:3px 12px 3px 0;white-space:nowrap;color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">{lbl}</td>'
            f'<td style="padding:3px 0;color:#94a3b8;font-size:11px;font-family:\'JetBrains Mono\',monospace;">{val}</td>'
            f'</tr>'
            for lbl, val in _inj_rows
        )
        with st.expander(f"📡 Injected AI Context ({len(_inj_rows)} signals)", expanded=False):
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;">{_inj_html}</table>',
                unsafe_allow_html=True,
            )

    # ── Run AI Valuation button ────────────────────────────────────────────────
    _cached_val = st.session_state.get("_last_valuation_result") or {}
    _has_cached = _cached_val.get("ticker", "").upper() == ticker.upper()
    _btn_label = "🔄 Refresh Valuation" if _has_cached else "🔍 Run AI Valuation"

    _val_btn_col, _val_status_col = st.columns([1, 3])
    with _val_btn_col:
        _run_val_btn = st.button(_btn_label, key=f"run_val_btn_{ticker}", type="primary", use_container_width=True)
    with _val_status_col:
        _opt_now = st.session_state.get("_options_sentiment") or {}
        if _opt_now.get("ticker", "").upper() == ticker.upper():
            st.caption(f"✓ Options loaded · P/C {_opt_now.get('pc_ratio', '—')} · {_opt_now.get('sentiment', '—')}")
        else:
            st.caption("⚠️ Options not loaded for this ticker — will auto-fetch on Run")

    result = None

    if _run_val_btn:
        # 1. Fetch fresh ticker options + unusual activity
        with st.spinner(f"Fetching live options flow for {ticker}..."):
            _fresh_opts = _fetch_ticker_options_live(ticker)
        if _fresh_opts.get("options_sentiment"):
            st.session_state["_options_sentiment"] = _fresh_opts["options_sentiment"]
            _ots = _fresh_opts["options_sentiment"]
            signals_text += (
                f"\nOptions Sentiment LIVE (P/C {_ots['pc_ratio']:.2f}): {_ots['sentiment']}"
                f" | Call Vol {_ots['call_vol']:,} vs Put Vol {_ots['put_vol']:,}"
            )
        if _fresh_opts.get("unusual_activity"):
            st.session_state["_unusual_activity_sentiment"] = _fresh_opts["unusual_activity"]
            _uas = _fresh_opts["unusual_activity"]
            signals_text += (
                f"\nUnusual Options Activity LIVE: {_uas['sentiment']} "
                f"({_uas['call_pct']:.0f}% calls / {_uas['put_pct']:.0f}% puts, "
                f"{_uas['flagged_contracts']} flagged contracts)"
            )

        # 2. Clear signals cache so 13F + insider data is refetched fresh
        _collect_signals.clear()

        # 3. Run AI
        with st.spinner("Generating AI valuation..."):
            from services.claude_client import generate_valuation, _fmt_tactical_ctx
            _ce_val = st.session_state.get("_current_events_digest", "")
            _tac_val = _fmt_tactical_ctx(st.session_state.get("_tactical_context"))
            result = generate_valuation(ticker, signals_text, use_claude=_use_claude, model=_cl_model, current_events=_ce_val, tactical_context=_tac_val)

        if not result:
            has_key = bool(os.getenv("GROQ_API_KEY", ""))
            if not has_key:
                st.error("GROQ_API_KEY is not set. Add it to your .env file or Streamlit Cloud secrets.")
            else:
                st.error("AI valuation failed — the LLM returned an unparseable response. Try again.")
                with st.expander("Debug: Signal data sent to LLM"):
                    st.code(signals_text)
            return

    elif _has_cached:
        result = _cached_val
    else:
        st.markdown(
            '<div style="padding:24px; background:#1A1A2E; border-radius:8px; text-align:center; '
            'border:1px dashed #334155; margin:16px 0;">'
            '<div style="color:#64748b; font-size:14px;">Click <strong>🔍 Run AI Valuation</strong> above to generate analysis with fresh data</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _render_signal_scorecard(signals)
        return

    # Persist result so Forecast Tracker can quick-capture it
    st.session_state["_last_valuation_result"] = {
        "rating":            result.get("rating", "Hold"),
        "confidence":        result.get("confidence", 50),
        "summary":           result.get("summary", ""),
        "engine":            _cl_model or "llama-3.3-70b-versatile",
        "ticker":            ticker,
        "key_levels":        result.get("key_levels", {}),
        "time_horizon":      result.get("time_horizon", ""),
        "conviction_drivers": result.get("conviction_drivers", []),
        "signal_conflicts":  result.get("signal_conflicts", []),
        "scenarios":         result.get("scenarios", {}),
        "catalysts":         result.get("catalysts", []),
    }

    _grad_color = "#c89b3c" if _selected_val_tier == "👑 Highly Regarded Mode" else "#FF8811"
    if _use_claude:
        st.markdown(
            f'<div style="height:2px;background:linear-gradient(90deg,{_grad_color},{_grad_color}33,transparent);'
            'margin:6px 0 12px 0;border-radius:1px;"></div>',
            unsafe_allow_html=True,
        )
    _render_rating_banner(result)

    # ── Inline forecast log button ─────────────────────────────────────────────
    _v = st.session_state.get("_last_valuation_result") or {}
    if _v.get("rating"):
        from modules.forecast_accuracy import render_log_button
        _cl, _cr = st.columns([4, 1])
        with _cr:
            render_log_button(
                signal_type="valuation",
                prediction=_v["rating"],
                confidence=_v.get("confidence", 65),
                summary=_v.get("summary", ""),
                model=_v.get("engine", ""),
                ticker=ticker,
                target_price=_v.get("key_levels", {}).get("resistance"),
                horizon_days=60,  # fundamental thesis needs time
                key=f"val_log_{ticker}_{_v['rating']}",
            )

    _render_signal_scorecard(signals)
    _render_analysis(result)

    # ── DCF Valuation — shown below AI rating ─────────────────────────────────
    st.markdown("---")
    _dcf_scenarios = _render_dcf(ticker)

    # Kelly Position Sizing
    if _dcf_scenarios and result:
        st.markdown("---")
        _render_kelly(result, _dcf_scenarios)

    # ── On-demand adversarial debate ──────────────────────────────────────────
    st.markdown("---")
    try:
        from utils.ai_tier import TIER_OPTS as _vd_tier_opts, TIER_MAP as _vd_tier_map
    except ImportError:
        _vd_tier_opts = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
        _vd_tier_map  = {
            "⚡ Freeloader Mode":      (False, None),
            "🧠 Regard Mode":          (True, "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
        }
    _vd_col1, _vd_col2, _vd_col3 = st.columns([2, 1.5, 1])
    with _vd_col1:
        st.markdown(
            f'<span style="color:#64748b;font-size:11px;">⚔️ Debate this valuation — '
            f'Sir Doomburger 🐻 vs Sir Fukyerputs 🐂, judged by Judge Judy ⚖️. '
            f'<span style="color:#475569;">3 LLM calls</span></span>'
            f'<div style="color:#475569;font-size:10px;margin-top:3px;">💡 Best results: run the AI Valuation above first — debate uses the rating &amp; confidence as evidence</div>',
            unsafe_allow_html=True,
        )
    with _vd_col2:
        _vd_tier = st.selectbox("", _vd_tier_opts, key=f"val_debate_tier_{ticker}", label_visibility="collapsed")
    with _vd_col3:
        _run_val_debate = st.button("⚔️ Debate This", key=f"btn_val_debate_{ticker}", use_container_width=True)
    _vd_use_claude, _vd_model = _vd_tier_map.get(_vd_tier, (False, None))
    _vd_engine = _vd_tier

    _val_debate_key = f"_val_debate_{ticker}"
    _val_debate = st.session_state.get(_val_debate_key) or {}

    if _run_val_debate:
        st.session_state[_val_debate_key] = {}
        with st.spinner("⚔️ Debate — Sir Doomburger 🐻 vs Sir Fukyerputs 🐂..."):
            try:
                from services.claude_client import generate_adversarial_debate as _gen_vd
                from utils.signal_block import build_ticker_block as _build_vtb, build_macro_block as _build_vmb
                _val_signals = ""
                try:
                    _val_signals = _build_vmb() + "\n\n" + _build_vtb(ticker)
                except Exception:
                    _val_signals = signals_text
                _val_context = (
                    f"TICKER: {ticker}\n"
                    f"AI RATING: {(result or {}).get('rating','?')} | Confidence: {(result or {}).get('confidence','?')}%\n"
                    f"SUMMARY: {(result or {}).get('summary','')}\n\n"
                    f"{_val_signals}"
                )
                _val_debate = _gen_vd(
                    _val_context,
                    use_claude=_vd_use_claude,
                    model=_vd_model,
                    ticker=ticker,
                    topic=f"Is {ticker} a BUY or a SELL at current price given this valuation, macro regime, and AI rating? Argue the specific stock — not the macro.",
                )
                _val_debate["engine"] = _vd_engine
                st.session_state[_val_debate_key] = _val_debate
                from utils.debate_record import log_verdict as _log_vdv
                _rc_vd = st.session_state.get("_regime_context") or {}
                _log_vdv(
                    verdict=_val_debate.get("verdict", "CONTESTED"),
                    confidence=_val_debate.get("confidence", 5),
                    regime=_rc_vd.get("regime", ""),
                    quadrant=_rc_vd.get("quadrant", ""),
                    regime_score=float(_rc_vd.get("score", 0.0)),
                )
                st.success(
                    f"⚔️ Debate complete [{_vd_engine}] — "
                    f"{_val_debate.get('verdict', 'CONTESTED')} "
                    f"(confidence {apply_confidence_penalty(_val_debate.get('confidence', 5))}/10)"
                )
            except Exception as _vde:
                st.error(f"Debate failed: {_vde}")

    if _val_debate.get("bear_argument") or _val_debate.get("bull_argument"):
        from modules.trade_journal import _render_debate_panel as _rdp
        _rdp(_val_debate)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This valuation is for informational purposes only "
        "and does not constitute financial advice. The DCF model uses estimates and assumptions "
        "that may not reflect actual future performance. Always do your own research."
    )


# ---------------------------------------------------------------------------
# Live options flow fetch (fresh per-ticker, 15-min cache)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def _fetch_ticker_options_live(ticker: str) -> dict:
    """Fetch live options chain for ticker: P/C ratio, volumes, unusual activity.
    Returns dict with 'options_sentiment' and 'unusual_activity' sub-dicts.
    """
    result = {}
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options[:3]
        if not exps:
            return result
        call_vol = put_vol = 0.0
        unusual_call = unusual_put = 0
        for exp in exps:
            try:
                chain = tk.option_chain(exp)
                calls, puts = chain.calls, chain.puts
                call_vol += float(calls["volume"].fillna(0).sum()) if "volume" in calls.columns else 0
                put_vol += float(puts["volume"].fillna(0).sum()) if "volume" in puts.columns else 0
                if "volume" in calls.columns and "openInterest" in calls.columns:
                    c2 = calls[calls["openInterest"] > 100]
                    unusual_call += int((c2["volume"].fillna(0) / c2["openInterest"].replace(0, 1) > 2.0).sum())
                if "volume" in puts.columns and "openInterest" in puts.columns:
                    p2 = puts[puts["openInterest"] > 100]
                    unusual_put += int((p2["volume"].fillna(0) / p2["openInterest"].replace(0, 1) > 2.0).sum())
            except Exception:
                continue
        pc_ratio = put_vol / call_vol if call_vol > 0 else 1.0
        sentiment = "BULLISH" if pc_ratio < 0.8 else "BEARISH" if pc_ratio > 1.2 else "NEUTRAL"
        total_unusual = unusual_call + unusual_put
        call_pct = unusual_call / total_unusual * 100 if total_unusual > 0 else 50.0
        put_pct = unusual_put / total_unusual * 100 if total_unusual > 0 else 50.0
        ua_sentiment = "BULLISH" if call_pct >= 65 else "BEARISH" if put_pct >= 65 else "MIXED"
        result["options_sentiment"] = {
            "ticker": ticker,
            "sentiment": sentiment,
            "pc_ratio": round(pc_ratio, 3),
            "call_vol": int(call_vol),
            "put_vol": int(put_vol),
        }
        result["unusual_activity"] = {
            "ticker": ticker,
            "sentiment": ua_sentiment,
            "call_pct": round(call_pct, 1),
            "put_pct": round(put_pct, 1),
            "flagged_contracts": total_unusual,
        }
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _collect_signals(ticker: str) -> dict:
    """Gather signals from yfinance and existing services into a structured dict.
    Raises RuntimeError on failure so Streamlit does not cache the bad result.
    Retries once with backoff on rate-limit (429 / Too Many Requests).
    """
    import time

    info = {}
    stock = None
    for _attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"):
                break  # good data — proceed
            # Empty info = rate-limited or bad ticker; wait and retry
            if _attempt < 2:
                time.sleep(2 ** _attempt)  # 1s, 2s
        except Exception:
            if _attempt < 2:
                time.sleep(2 ** _attempt)
            continue

    if not info.get("currentPrice") and not info.get("regularMarketPrice") and not info.get("previousClose"):
        raise RuntimeError(f"Too Many Requests — Yahoo Finance is rate-limiting. Wait 30s and retry.")

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
            volume = hist["Volume"] if "Volume" in hist.columns else None
            current = close.iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]

            # RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            _g, _l = gain.iloc[-1], loss.iloc[-1]
            if np.isnan(_g) or np.isnan(_l):
                rsi = None
            else:
                rs = _g / _l if _l != 0 else 100
                rsi = round(100 - (100 / (1 + rs)), 1)

            # Volume analysis
            vol_20d_avg = None
            vol_ratio = None
            unusual_volume = False
            if volume is not None and len(volume) >= 21:
                vol_20d_avg = round(float(volume.iloc[-21:-1].mean()), 0)
                today_vol = float(volume.iloc[-1])
                if vol_20d_avg and vol_20d_avg > 0:
                    vol_ratio = round(today_vol / vol_20d_avg, 2)
                    unusual_volume = vol_ratio >= 2.0

            # Relative strength vs SPY (1-month)
            rs_vs_spy = None
            try:
                spy_hist = fetch_ohlcv_single("SPY", period="1mo", interval="1d")
                if spy_hist is not None and not spy_hist.empty:
                    spy_close = spy_hist["Close"]
                    if isinstance(spy_close, pd.DataFrame):
                        spy_close = spy_close.iloc[:, 0]
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
                    stock_1m = close.iloc[-1] / close.iloc[-22] - 1 if len(close) >= 22 else None
                    if stock_1m is not None:
                        rs_vs_spy = round(float(stock_1m) * 100 - spy_ret, 1)
            except Exception:
                pass

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
                "rsi14": rsi,
                "period_return_pct": round((current / close.iloc[0] - 1) * 100, 1),
                "vol_20d_avg": vol_20d_avg,
                "vol_ratio": vol_ratio,
                "unusual_volume": unusual_volume,
                "rs_vs_spy_1m": rs_vs_spy,
            }
    except Exception:
        signals["price"] = None

    # 1b. Earnings Intelligence (last 4Q surprise trend + analyst consensus)
    try:
        from services.market_data import fetch_earnings_intelligence
        ei = fetch_earnings_intelligence(ticker)
        eps_hist = ei.get("eps_history", [])
        analyst = ei.get("analyst", {})
        next_earn = ei.get("next_earnings")
        # Compute beat streak and surprise acceleration
        beats = [q["beat"] for q in eps_hist if q.get("beat") is not None]
        surprises = [q["surprise_pct"] for q in eps_hist if q.get("surprise_pct") is not None]
        beat_streak = sum(1 for b in beats if b) if beats else 0
        surprise_trend = None
        if len(surprises) >= 2:
            surprise_trend = "accelerating" if surprises[0] > surprises[-1] else "decelerating"
        signals["earnings"] = {
            "eps_history": eps_hist,
            "beat_streak": beat_streak,
            "total_quarters": len(beats),
            "surprise_trend": surprise_trend,
            "analyst_buy": analyst.get("strong_buy", 0) + analyst.get("buy", 0),
            "analyst_hold": analyst.get("hold", 0),
            "analyst_sell": analyst.get("sell", 0) + analyst.get("strong_sell", 0),
            "mean_target": analyst.get("mean_target"),
            "upside_pct": analyst.get("upside_pct"),
            "next_earnings": next_earn,
        }
    except Exception:
        signals["earnings"] = None

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

    # 5. Short Interest + trend
    try:
        short_pct = info.get("shortPercentOfFloat") or 0.0
        days_to_cover = info.get("shortRatio") or 0.0
        shares_short = info.get("sharesShort") or 0
        shares_short_prior = info.get("sharesShortPriorMonth") or 0
        short_chg_pct = None
        short_trend = "unknown"
        if shares_short_prior and shares_short_prior > 0:
            short_chg_pct = round((shares_short - shares_short_prior) / shares_short_prior * 100, 1)
            short_trend = "increasing" if short_chg_pct > 5 else ("decreasing" if short_chg_pct < -5 else "stable")
        signals["short_interest"] = {
            "short_pct_float": round(short_pct * 100, 1),
            "days_to_cover": round(days_to_cover, 1),
            "short_chg_pct": short_chg_pct,
            "short_trend": short_trend,
            "squeeze_potential": (
                "extreme" if short_pct >= 0.30 else
                "high" if short_pct >= 0.20 else
                "elevated" if short_pct >= 0.10 else
                "moderate" if short_pct >= 0.05 else "low"
            ),
        }
    except Exception:
        signals["short_interest"] = None

    # 5b. Analyst revisions (recent upgrades/downgrades) + buyback yield
    try:
        upgrades_df = stock.upgrades_downgrades
        upgrades = downgrades = 0
        if upgrades_df is not None and not upgrades_df.empty:
            # Last 90 days
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            if upgrades_df.index.tz is not None:
                cutoff = cutoff.tz_localize(upgrades_df.index.tz)
            recent = upgrades_df[upgrades_df.index >= cutoff]
            if "ToGrade" in recent.columns and "FromGrade" in recent.columns:
                _pos = {"Buy", "Strong Buy", "Outperform", "Overweight", "Market Outperform"}
                _neg = {"Sell", "Strong Sell", "Underperform", "Underweight", "Market Underperform"}
                for _, row in recent.iterrows():
                    to_g = str(row.get("ToGrade", ""))
                    fr_g = str(row.get("FromGrade", ""))
                    if to_g in _pos and fr_g not in _pos:
                        upgrades += 1
                    elif to_g in _neg and fr_g not in _neg:
                        downgrades += 1
        revision_bias = "positive" if upgrades > downgrades else ("negative" if downgrades > upgrades else "neutral")

        # Buyback yield: (repurchase of capital stock / market cap)
        buyback_yield = None
        try:
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                repurchase_row = next((r for r in cf.index if "Repurchase" in str(r) or "Buyback" in str(r)), None)
                if repurchase_row is not None:
                    repurchase = abs(float(cf.loc[repurchase_row].iloc[0]))
                    mkt_cap = info.get("marketCap")
                    if mkt_cap and mkt_cap > 0:
                        buyback_yield = round(repurchase / mkt_cap * 100, 2)
        except Exception:
            pass

        signals["revisions"] = {
            "upgrades_90d": upgrades,
            "downgrades_90d": downgrades,
            "revision_bias": revision_bias,
            "buyback_yield_pct": buyback_yield,
        }
    except Exception:
        signals["revisions"] = None

    # 6. Options Sentiment
    try:
        expirations = stock.options
        if expirations:
            chain = stock.option_chain(expirations[0])
            call_vol = int(chain.calls["volume"].sum())
            put_vol = int(chain.puts["volume"].sum())
            if call_vol == 0 and put_vol == 0:
                pc_ratio = None
                pc_sentiment = "N/A (no options data)"
            else:
                pc_ratio = round(put_vol / call_vol, 2) if call_vol > 0 else None
                pc_sentiment = "bearish" if pc_ratio and pc_ratio > 1.0 else ("bullish" if pc_ratio and pc_ratio < 0.7 else "neutral")
            signals["options"] = {
                "put_call_ratio": pc_ratio,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "sentiment": pc_sentiment,
            }
        else:
            signals["options"] = None
    except Exception:
        signals["options"] = None

    # 7. Company Profile
    try:
        from services.claude_client import describe_company
        name = info.get("longName", ticker)
        sic = info.get("industry", "")
        profile = describe_company(name, ticker, sic)
        signals["profile"] = profile
    except Exception:
        signals["profile"] = None

    # 8. MD&A Tone (from session cache — populated by EDGAR Scanner)
    _cached_mda = st.session_state.get("_mda_sentiment")
    if _cached_mda and _cached_mda.get("ticker", "").upper() == ticker.upper():
        signals["mda_tone"] = {
            "tone": _cached_mda.get("tone"),
            "tone_score": _cached_mda.get("tone_score"),
            "forward_outlook": _cached_mda.get("forward_outlook"),
            "summary": _cached_mda.get("summary", ""),
            "date": _cached_mda.get("date", ""),
        }
    else:
        signals["mda_tone"] = None

    # 9. Cross-module: institutional 13F flow + stress signals
    signals["whale"] = _collect_whale_signals(ticker)
    signals["stress"] = _collect_stress_signals()

    # 10. Price action: Elliott Wave + Wyckoff (lightweight, cached)
    signals["price_action"] = _get_price_action_signal(ticker)

    # 11. Sector rotation context (live momentum + regime alignment)
    try:
        from services.sector_rotation import get_sector_context_str
        _quadrant_v = st.session_state.get("_regime_context", {}).get("quadrant", "")
        _ticker_sector_v = signals.get("fundamentals", {}).get("sector", "") or ""
        if _quadrant_v:
            signals["sector_rotation"] = get_sector_context_str(_ticker_sector_v, _quadrant_v)
        else:
            signals["sector_rotation"] = None
    except Exception:
        signals["sector_rotation"] = None

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
def _get_price_action_signal(ticker: str) -> dict:
    """Run Elliott Wave + Wyckoff on daily data and return a compact signal dict.
    Cached 1hr — these are compute-heavy engines."""
    result = {}
    try:
        from services.market_data import fetch_ohlcv_single
        df = fetch_ohlcv_single(ticker, period="2y", interval="1d")
        if df is None or df.empty or len(df) < 60:
            return result

        # ── Wyckoff ──────────────────────────────────────────────────────────
        try:
            from services.wyckoff_engine import analyze_wyckoff
            wa = analyze_wyckoff(df["Close"], df["High"], df["Low"], df["Volume"])
            if wa and wa.current_phase:
                cp = wa.current_phase
                result["wyckoff_phase"]      = cp.phase
                result["wyckoff_sub_phase"]  = cp.sub_phase or ""
                result["wyckoff_confidence"] = round(float(cp.confidence), 2) if cp.confidence else None
                result["wyckoff_cause_target"] = round(float(cp.cause_target), 2) if cp.cause_target else None
        except Exception:
            pass

        # ── Elliott Wave ──────────────────────────────────────────────────────
        try:
            from services.elliott_wave_engine import get_best_wave_count, build_wave_forecast
            bc = get_best_wave_count(df["Close"])
            if bc:
                result["ew_wave"]        = bc.current_wave_label
                result["ew_confidence"]  = round(float(bc.confidence), 2) if bc.confidence else None
                result["ew_invalidation"] = round(float(bc.invalidation_level), 2) if bc.invalidation_level else None
                current_price = float(df["Close"].iloc[-1])
                fc = build_wave_forecast(bc, current_price)
                if fc:
                    result["ew_direction"]         = fc.direction
                    result["ew_primary_target"]    = round(float(fc.primary_target), 2) if fc.primary_target else None
                    result["ew_primary_prob"]      = round(float(fc.primary_probability), 2) if fc.primary_probability else None
                    result["ew_alternate_target"]  = round(float(fc.alternate_target), 2) if fc.alternate_target else None
        except Exception:
            pass

    except Exception:
        pass
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
        _rsi = p['rsi14']
        rsi_str = f"{_rsi}" if _rsi is not None and not (isinstance(_rsi, float) and np.isnan(_rsi)) else "N/A"
        vol_str = (f"Vol Ratio vs 20d avg: {p['vol_ratio']}x" +
                   (" ⚠️ UNUSUAL VOLUME" if p.get("unusual_volume") else "")
                   ) if p.get("vol_ratio") is not None else ""
        rs_str = f"RS vs SPY (1M): {p['rs_vs_spy_1m']:+.1f}%" if p.get("rs_vs_spy_1m") is not None else ""
        lines.append(f"RSI(14): {rsi_str} | 1Y Return: {p['period_return_pct']}%")
        if vol_str or rs_str:
            lines.append(" | ".join(x for x in [vol_str, rs_str] if x) + "\n")
        else:
            lines.append("")

    if signals.get("fundamentals"):
        f = signals["fundamentals"]
        lines.append("## Fundamentals")
        pe  = f['pe_ratio']   if f['pe_ratio']   is not None else "N/A"
        fpe = f['forward_pe'] if f['forward_pe'] is not None else "N/A"
        ps  = f['ps_ratio']   if f['ps_ratio']   is not None else "N/A"
        pb  = f['pb_ratio']   if f['pb_ratio']   is not None else "N/A"
        lines.append(f"P/E: {pe} | Fwd P/E: {fpe} | P/S: {ps} | P/B: {pb}")
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
        n_inst = i['num_institutions'] if i['num_institutions'] is not None else "N/A"
        lines.append(f"Institutional: {inst} | Insider: {ins} | # Institutions: {n_inst}\n")

    if signals.get("insider"):
        ins = signals["insider"]
        lines.append("## Insider Activity (Recent)")
        lines.append(f"Buys: {ins['buy_count']} (${ins['buy_value']:,.0f}) | "
                      f"Sells: {ins['sell_count']} (${ins['sell_value']:,.0f}) | "
                      f"Net: {ins['net_sentiment']}\n")

    if signals.get("short_interest"):
        si = signals["short_interest"]
        chg_str = f" | MoM change: {si['short_chg_pct']:+.1f}% ({si['short_trend']})" if si.get("short_chg_pct") is not None else ""
        lines.append("## Short Interest")
        lines.append(f"Short % Float: {si['short_pct_float']}% | Days-to-Cover: {si['days_to_cover']}d | Squeeze potential: {si['squeeze_potential']}{chg_str}\n")

    if signals.get("revisions"):
        rv = signals["revisions"]
        by_str = f" | Buyback yield: {rv['buyback_yield_pct']:.2f}%" if rv.get("buyback_yield_pct") is not None else ""
        lines.append("## Analyst Revisions (90d)")
        lines.append(f"Upgrades: {rv['upgrades_90d']} | Downgrades: {rv['downgrades_90d']} | Revision bias: {rv['revision_bias']}{by_str}\n")

    if signals.get("earnings"):
        e = signals["earnings"]
        lines.append("## Earnings & Analyst")
        if e.get("eps_history"):
            hist_parts = []
            for q in e["eps_history"]:
                beat_str = "BEAT" if q.get("beat") else ("MISS" if q.get("beat") is False else "?")
                surp = f" ({q['surprise_pct']:+.1f}%)" if q.get("surprise_pct") is not None else ""
                hist_parts.append(f"{q['period']}: {beat_str}{surp}")
            lines.append("EPS last 4Q: " + " | ".join(hist_parts))
        streak = e.get("beat_streak", 0)
        total = e.get("total_quarters", 0)
        trend = e.get("surprise_trend")
        streak_str = f"Beat streak: {streak}/{total}" if total else "N/A"
        trend_str = f" — surprise trend {trend}" if trend else ""
        lines.append(f"{streak_str}{trend_str}")
        ab = e.get("analyst_buy", 0)
        ah = e.get("analyst_hold", 0)
        as_ = e.get("analyst_sell", 0)
        tgt = f"${e['mean_target']:.2f}" if e.get("mean_target") else "N/A"
        upside = f" ({e['upside_pct']:+.1f}% upside)" if e.get("upside_pct") is not None else ""
        lines.append(f"Analyst: {ab} Buy / {ah} Hold / {as_} Sell | Target: {tgt}{upside}")
        if e.get("next_earnings"):
            ne = e["next_earnings"]
            lines.append(f"Next earnings: {ne['date']} ({ne['days_away']}d away)\n")
        else:
            lines.append("")

    if signals.get("options"):
        o = signals["options"]
        lines.append("## Options Sentiment")
        pc_str = f"{o['put_call_ratio']}" if o['put_call_ratio'] is not None else "N/A"
        lines.append(f"P/C Ratio: {pc_str} | Call Vol: {o['call_volume']:,} | "
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

    # Tile 9: Tactical Timing
    _tac_ctx = st.session_state.get("_tactical_context") or {}
    if _tac_ctx:
        _ts = _tac_ctx.get("tactical_score", 50)
        _tlabel = _tac_ctx.get("label", "")
        _tbias = _tac_ctx.get("action_bias", "")
        _tac_sentiment = "bullish" if _ts >= 65 else ("bearish" if _ts < 38 else "neutral")
        _tac_sigs = _tac_ctx.get("signals", [])
        _sig_lines = []
        for _sr in _tac_sigs:
            _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
            _sig_lines.append(f"{_arrow} {_sr['Signal'].split('(')[0].strip()}: {_sr['Value']}")
        _signal_tile(r3c3, f"Tactical Timing · {_ts}/100", _tac_sentiment,
                     [_tlabel, _tbias[:60] + ("…" if len(_tbias) > 60 else "")] + _sig_lines[:3])
    else:
        _signal_tile(r3c3, "Tactical Timing", "unavailable", ["Run QIR to populate"])

    # ── Row 4 ──
    r4c1, r4c2, r4c3 = st.columns(3)

    # Tile 10: Options Flow Sentiment
    _of_ctx = st.session_state.get("_options_flow_context") or {}
    if _of_ctx:
        _os = _of_ctx.get("options_score", 50)
        _of_sentiment = "bullish" if _os >= 65 else ("bearish" if _os < 38 else "neutral")
        _of_sig_lines = []
        for _sr in _of_ctx.get("signals", [])[:3]:
            _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
            _of_sig_lines.append(f"{_arrow} {_sr['Signal']}: {_sr['Value']}")
        _of_bias = _of_ctx.get("action_bias", "")
        _signal_tile(r4c1, f"Options Flow Sentiment · {_os}/100", _of_sentiment,
                     [_of_ctx["label"], _of_bias[:60] + ("…" if len(_of_bias) > 60 else "")]
                     + _of_sig_lines)
    else:
        _signal_tile(r4c1, "Options Flow Sentiment", "unavailable", ["Run QIR to populate"])

    # Tile 11: StockTwits Crowd Sentiment
    _st_ctx = st.session_state.get("_stocktwits_digest") or {}
    if _st_ctx:
        _st_avg = _st_ctx.get("avg_sentiment_score", _st_ctx.get("overall_bull_pct", 50))
        _st_mood = _st_ctx.get("market_mood", "mixed")
        _st_sentiment = "bullish" if _st_mood == "bullish" else ("bearish" if _st_mood == "bearish" else "neutral")
        # Check if the current ticker is in the trending list
        _st_ticker_data = next(
            (t for t in _st_ctx.get("trending_tickers", [])
             if t.get("symbol", "").upper() == ticker.upper()),
            None
        )
        if _st_ticker_data:
            _t_score = _st_ticker_data.get("sentiment_score", 50)
            _t_label = _st_ticker_data.get("sentiment_label", "Neutral")
            _t_why = (_st_ticker_data.get("trending_summary") or "")[:80]
            _t_sentiment = "bullish" if _t_score >= 60 else ("bearish" if _t_score <= 40 else "neutral")
            _signal_tile(r4c2, f"Crowd Sentiment · {ticker} · {_t_score}/100", _t_sentiment, [
                f"📊 {_t_label} (StockTwits score)",
                _t_why if _t_why else f"Market-wide: {_st_avg}/100 ({_st_mood})",
                f"Top trending: {', '.join(_st_ctx.get('top_bullish', [])[:3])}",
            ])
        else:
            _signal_tile(r4c2, f"Crowd Sentiment · {_st_avg}/100 (market)", _st_sentiment, [
                f"{ticker} not in top social trending",
                f"Market mood: {_st_mood}",
                f"Top bullish crowd: {', '.join(_st_ctx.get('top_bullish', [])[:3])}",
            ])
    else:
        _signal_tile(r4c2, "Crowd Sentiment", "unavailable", ["Run QIR to populate"])


def _render_rating_banner(result: dict):
    """Large colored rating badge + confidence meter + time horizon + conviction drivers."""
    rating = result.get("rating", "Hold")
    confidence = result.get("confidence", 50)
    color = RATING_COLORS.get(rating, COLORS["yellow"])
    time_horizon = result.get("time_horizon", "")
    conviction_drivers = result.get("conviction_drivers", [])

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
        if time_horizon:
            st.markdown(
                f'<div style="margin-top:8px; padding:10px; background:#1E1E2E; border-radius:8px; '
                f'font-size:11px; color:#aaa; line-height:1.6;">'
                f'<div style="color:#888; font-size:10px; font-weight:600; margin-bottom:4px;">TIME HORIZON</div>'
                f'{time_horizon}</div>',
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown("**Confidence**")
        st.progress(confidence / 100)
        st.caption(f"{confidence}% confidence based on available signals")
        if result.get("summary"):
            st.markdown(f"*{result['summary']}*")
        if conviction_drivers:
            drivers_html = "".join(
                f'<div style="display:inline-block; background:#1E3A5F; color:#7EC8E3; '
                f'padding:3px 10px; border-radius:12px; font-size:11px; margin:2px;">⚡ {d}</div>'
                for d in conviction_drivers
            )
            st.markdown(
                f'<div style="margin-top:8px;">'
                f'<div style="font-size:11px; color:#888; margin-bottom:4px;">CONVICTION DRIVERS</div>'
                f'{drivers_html}</div>',
                unsafe_allow_html=True,
            )


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
    """Bull/bear factors, scenarios, signal conflicts, key levels, catalysts, recommendation."""
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

    # Signal Conflicts
    conflicts = result.get("signal_conflicts", [])
    if conflicts:
        st.markdown("### ⚡ Signal Conflicts")
        conflicts_html = "".join(
            f'<div style="padding:8px 12px; margin:4px 0; background:#2A2020; border-left:3px solid #F59E0B; '
            f'border-radius:4px; font-size:13px; color:#F59E0B;">⚠️ {c}</div>'
            for c in conflicts
        )
        st.markdown(conflicts_html, unsafe_allow_html=True)

    # Scenarios — Bull / Base / Bear
    scenarios = result.get("scenarios", {})
    if scenarios:
        st.markdown("### Scenario Analysis")
        sc_cols = st.columns(3)
        scenario_cfg = [
            ("bull",  "🐂 Bull Case",  "#1B3D2F", "#69F0AE"),
            ("base",  "📊 Base Case",  "#1E2A3A", "#7EC8E3"),
            ("bear",  "🐻 Bear Case",  "#3D1B1B", "#FF5252"),
        ]
        for col, (key, label, bg, accent) in zip(sc_cols, scenario_cfg):
            sc = scenarios.get(key, {})
            target = sc.get("target")
            prob = sc.get("probability", "?")
            thesis = sc.get("thesis", "")
            with col:
                target_str = f"${target:,.2f}" if isinstance(target, (int, float)) and target else "—"
                st.markdown(
                    f'<div style="background:{bg}; padding:14px; border-radius:8px; border-top:3px solid {accent};">'
                    f'<div style="color:{accent}; font-weight:700; font-size:13px;">{label}</div>'
                    f'<div style="font-size:22px; font-weight:800; margin:6px 0;">{target_str}</div>'
                    f'<div style="font-size:11px; color:#888; margin-bottom:6px;">Probability: {prob}%</div>'
                    f'<div style="font-size:12px; color:#ccc;">{thesis}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Key Levels — expanded
    key_levels = result.get("key_levels", {})
    level_fields = [
        ("stop_loss",   "Stop Loss",   "#FF5252"),
        ("support",     "Support",     "#F59E0B"),
        ("resistance",  "Resistance",  "#7EC8E3"),
        ("target_1",    "Target 1",    "#69F0AE"),
        ("target_2",    "Target 2",    "#00C853"),
    ]
    filled = [(label, key_levels[k], color) for k, label, color in level_fields if key_levels.get(k)]
    if filled:
        st.markdown("### Key Price Levels")
        level_cols = st.columns(len(filled))
        for col, (label, val, color) in zip(level_cols, filled):
            with col:
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else f"${val}"
                st.markdown(
                    f'<div style="background:#1A1A2E; padding:12px; border-radius:8px; '
                    f'border-top:3px solid {color}; text-align:center;">'
                    f'<div style="font-size:11px; color:#888;">{label}</div>'
                    f'<div style="font-size:18px; font-weight:700; color:{color};">{val_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Catalysts
    catalysts = result.get("catalysts", [])
    if catalysts:
        st.markdown("### 📅 Upcoming Catalysts")
        impact_color = {"high": "#FF5252", "medium": "#F59E0B", "low": "#888"}
        dir_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
        cat_rows = ""
        for c in catalysts:
            evt = c.get("event", "")
            date = c.get("date", "TBD")
            impact = c.get("impact", "medium").lower()
            direction = c.get("direction", "neutral").lower()
            ic = impact_color.get(impact, "#888")
            di = dir_icon.get(direction, "🟡")
            cat_rows += (
                f'<div style="display:flex; align-items:center; padding:8px 12px; margin:3px 0; '
                f'background:#1A1A2E; border-radius:6px; gap:12px;">'
                f'<div style="font-size:16px;">{di}</div>'
                f'<div style="flex:1;"><div style="font-size:13px; color:#eee;">{evt}</div>'
                f'<div style="font-size:11px; color:#666;">{date}</div></div>'
                f'<div style="font-size:11px; color:{ic}; font-weight:600;">{impact.upper()}</div>'
                f'</div>'
            )
        st.markdown(cat_rows, unsafe_allow_html=True)

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
    try:
        pe = float(pe) if pe is not None else None
    except (TypeError, ValueError):
        pe = None
    try:
        growth = float(growth) if growth is not None else None
    except (TypeError, ValueError):
        growth = None
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

_DEFAULT_ERP = 0.055  # Damodaran US ERP — base rate, adjusted by regime below


def _get_regime_erp() -> tuple[float, str]:
    """Return (ERP, note) adjusted ±50bps for current macro regime.
    Risk-Off → higher premium (fear); Risk-On → lower premium (greed)."""
    ctx = st.session_state.get("_regime_context") or {}
    regime = ctx.get("regime", "")
    if "Risk-Off" in regime:
        return 0.060, "6.0% (+50bps Risk-Off)"
    if "Risk-On" in regime:
        return 0.050, "5.0% (−50bps Risk-On)"
    return _DEFAULT_ERP, "5.5% (Neutral — Damodaran base)"

# Sector profiles ported from Stock Ticker Checker / build_model.py
# Each profile sets per-sector growth clamps, terminal growth, and optional WACC premium
_SECTOR_PROFILES = {
    # default_growth_yr1: Damodaran Jan 2026 sector revenue CAGR (histgr.html)
    # default_fcf_margin: Damodaran Jan 2026 net margin proxy for FCF (margin.html)
    "High-Growth Tech": {
        "growth_yr1_clamp": (0.03, 0.55),
        "growth_yr610_clamp": (0.02, 0.35),
        "terminal_growth": 0.04,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.19,   # Software Sys/App 19.56%
        "default_fcf_margin": 0.20,   # Software net margin 25.49% → FCF ~20%
    },
    "Consumer Cyclical": {
        "growth_yr1_clamp": (0.02, 0.40),
        "growth_yr610_clamp": (0.02, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.10,   # Retail General 9.92%
        "default_fcf_margin": 0.06,   # Retail net margin 5.61%
    },
    "Consumer Defensive": {
        "growth_yr1_clamp": (0.01, 0.20),
        "growth_yr610_clamp": (0.01, 0.12),
        "terminal_growth": 0.03,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.06,   # Food/Beverage/Household avg ~7%
        "default_fcf_margin": 0.09,   # Household Products 11.68%, Food 2.82% → avg ~9%
    },
    "Healthcare": {
        "growth_yr1_clamp": (0.02, 0.45),
        "growth_yr610_clamp": (0.02, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.14,   # Healthcare Products 18.41% + Support 13.12% blend
        "default_fcf_margin": 0.10,   # Pharma 18.54%, Support Services 1.25% → blended ~10%
    },
    "Energy/Materials/Industrials": {
        "growth_yr1_clamp": (0.00, 0.35),
        "growth_yr610_clamp": (0.01, 0.15),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.09,   # Oil/Gas 4.62% + Machinery 10.37% + Metals 8.67% avg
        "default_fcf_margin": 0.09,   # Oil/Gas Integrated 8.30%, Metals 10.52% avg
    },
    "Utilities": {
        "growth_yr1_clamp": (0.01, 0.15),
        "growth_yr610_clamp": (0.01, 0.08),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.06,   # Utility General 5.33%
        "default_fcf_margin": 0.12,   # Utility General net margin 14.18% (regulated)
    },
    "Growth-Platform": {
        "growth_yr1_clamp": (0.15, 0.45),
        "growth_yr610_clamp": (0.08, 0.20),
        "terminal_growth": 0.035,
        "wacc_premium": 0.01,
        "default_growth_yr1": 0.25,   # Software Internet 29.18%; near-profitability platforms
        "default_fcf_margin": 0.08,   # Approaching profitability — thin but positive FCF
    },
    "High-SBC-Tech": {
        "growth_yr1_clamp": (0.15, 0.50),
        "growth_yr610_clamp": (0.08, 0.25),
        "terminal_growth": 0.035,
        "wacc_premium": 0.015,
        "default_growth_yr1": 0.16,   # Semiconductor 11.18% + Software blend
        "default_fcf_margin": 0.12,   # Semiconductor net margin 30.45% → FCF ~12% after SBC
    },
    "Cyclical-Commodity": {
        "growth_yr1_clamp": (0.00, 0.20),
        "growth_yr610_clamp": (0.01, 0.10),
        "terminal_growth": 0.02,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.08,   # Steel 11.37% + Metals 8.67% through-cycle avg
        "default_fcf_margin": 0.06,   # Compressed through-cycle margins; Steel 1.93%
    },
    "Mature-Stable": {
        "growth_yr1_clamp": (0.03, 0.08),
        "growth_yr610_clamp": (0.02, 0.05),
        "terminal_growth": 0.025,
        "wacc_premium": 0.0,
        "default_growth_yr1": 0.05,   # Telecom Wireless 3.70% + Railroads 4.91% avg
        "default_fcf_margin": 0.11,   # Railroads 24.73%, Telecom 12.24% → blended ~11%
    },
    "Early-Stage-PreProfit": {
        "growth_yr1_clamp": (0.20, 0.60),
        "growth_yr610_clamp": (0.10, 0.30),
        "terminal_growth": 0.03,
        "wacc_premium": 0.03,
        "default_growth_yr1": 0.25,   # Biotech 28.93% + Internet 29.18% avg
        "default_fcf_margin": -0.05,  # Cash-burning — negative FCF expected pre-profit
    },
}

# Regime multipliers applied to sector default growth and margin at DCF compute time
# Source: Damodaran base rates × macro regime adjustment
_REGIME_GROWTH_MULT: dict[str, float] = {
    "Goldilocks":  1.00,   # base rates — expansion with controlled inflation
    "Reflation":   1.10,   # +10% — inflationary growth lifts nominal revenue
    "Stagflation": 0.75,   # -25% — demand destruction, hardest environment
    "Deflation":   0.80,   # -20% — recession reduces volume and pricing power
    "Transition":  0.95,   # slight caution — no dominant regime
}
_REGIME_MARGIN_MULT: dict[str, float] = {
    "Goldilocks":  1.00,
    "Reflation":   0.95,   # input cost inflation compresses margins slightly
    "Stagflation": 0.85,   # worst — costs sticky, revenue weak
    "Deflation":   0.90,   # deflation helps input costs but hurts pricing
    "Transition":  0.97,
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
        tnx = fetch_ohlcv_single("^TNX", period="5y", interval="1mo")
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


@st.cache_data(ttl=3600)
def _compute_dcf(ticker: str, growth_adj: float = 0.0, wacc_adj: float = 0.0, tg_adj: float = 0.0) -> dict | None:
    """
    2-stage levered DCF following Simply Wall St methodology.
    Returns dict with all intermediate values for display.
    """
    try:
        info = get_yf_info_safe(ticker) or {}
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
        # Try revenue × sector FCF margin fallback before giving up
        _revenue = info.get("totalRevenue") or 0
        if _revenue:
            _quadrant = (st.session_state.get("_regime_context") or {}).get("quadrant", "Transition")
            _m_mult = _REGIME_MARGIN_MULT.get(_quadrant, 1.0)
            _prof = _SECTOR_PROFILES[_detect_sector_profile(info)]
            _fcf_margin = _prof.get("default_fcf_margin", 0.08) * _m_mult
            latest_fcf = _revenue * _fcf_margin
        if latest_fcf <= 0:
            return {"error": "negative_fcf", "latest_fcf": latest_fcf, "company_name": company_name}

    # ── Sector profile (sector-specific growth clamps + terminal growth) ──
    profile_name = _detect_sector_profile(info)
    profile = _SECTOR_PROFILES[profile_name]

    # ── Historical FCF growth (for fallback) ──
    hist_fcf_growth = None
    if fcf_values is not None and len(fcf_values) >= 2:
        oldest = float(fcf_values.iloc[-1])
        newest = float(fcf_values.iloc[0])
        n_years = len(fcf_values) - 1
        if oldest > 0 and n_years > 0:
            hist_fcf_growth = (newest / oldest) ** (1 / n_years) - 1

    # ── Growth rate estimates ──
    # Priority (SWS-aligned, most forward-looking first):
    # 1. Analyst 5-yr EPS CAGR consensus — exact input SWS uses (Yahoo "+5y" row)
    # 2. Revenue growth (trailing YOY — stable, appropriate for FCF projection)
    # 3. Historical FCF CAGR (backward-looking sanity anchor)
    # 4. Damodaran sector default × regime multiplier
    # NOTE: earningsGrowth (trailing YOY EPS) intentionally excluded — in high-growth
    #       years (e.g. GOOGL 2024: +34%) it inflates the DCF by 1.5–2× vs SWS.
    revenue_growth = info.get("revenueGrowth")

    # 1. Analyst 5-yr EPS CAGR — matches SWS methodology exactly
    growth_source = "Sector Default"
    initial_growth = None
    try:
        _ge = stock.growth_estimates
        if _ge is not None and not _ge.empty:
            # yfinance growth_estimates: rows = periods (0q,1q,0y,1y,+5y,-5y), cols vary
            for _period in ["+5y", "5y", "nextFiveYears"]:
                if _period in _ge.index:
                    _row = _ge.loc[_period]
                    # value may be in first numeric column (ticker name or "Growth")
                    _val = None
                    for _v in _row:
                        if isinstance(_v, (int, float)) and not pd.isna(_v):
                            _val = float(_v)
                            break
                    if _val is not None and 0 < abs(_val) < 1.0:
                        initial_growth = _val
                        growth_source = "5yr EPS CAGR"
                        break
    except Exception:
        pass

    if initial_growth is None and revenue_growth and abs(revenue_growth) < 1:
        initial_growth = float(revenue_growth)
        growth_source = "Revenue Growth"
    if initial_growth is None and hist_fcf_growth and abs(hist_fcf_growth) < 1:
        initial_growth = float(hist_fcf_growth)
        growth_source = "Hist. FCF CAGR"
    if initial_growth is None:
        # Damodaran sector default, adjusted for current macro regime
        _quadrant = (st.session_state.get("_regime_context") or {}).get("quadrant", "Transition")
        _g_mult = _REGIME_GROWTH_MULT.get(_quadrant, 1.0)
        initial_growth = profile.get("default_growth_yr1", 0.08) * _g_mult
        growth_source = "Sector Default"

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
    # Beta: prefer live market beta from yfinance (already levered, refreshed daily)
    # Fall back to Damodaran sector unlevered beta + re-levering if unavailable
    market_beta = info.get("beta")
    de_ratio = total_debt / total_equity if total_equity and total_equity > 0 else 0
    if market_beta and isinstance(market_beta, (int, float)) and 0.3 <= float(market_beta) <= 3.5:
        levered_beta = max(0.8, min(2.0, float(market_beta)))
        unlevered_beta = levered_beta / (1 + (1 - tax_rate) * de_ratio) if de_ratio > 0 else levered_beta
        beta_source = "market"
    else:
        unlevered_beta = _SECTOR_UNLEVERED_BETA.get(sector, 0.85)
        levered_beta = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
        levered_beta = max(0.8, min(2.0, levered_beta))
        beta_source = "sector"  # Damodaran Jan 2024 — may be stale

    # ERP: regime-adjusted (Risk-Off +50bps, Risk-On -50bps, Neutral base 5.5%)
    erp, erp_note = _get_regime_erp()

    cost_of_equity = rf + levered_beta * erp
    discount_rate = cost_of_equity + profile.get("wacc_premium", 0.0) + wacc_adj
    terminal_growth = max(0.005, terminal_growth + tg_adj)

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

    # ── Net cash/debt adjustment (SWS methodology) ──
    # Enterprise Value (PV of FCFs) ± net cash → Equity Value
    # Net cash positive (cash-rich): adds to equity; net debt negative: reduces it
    total_cash = info.get("totalCash") or info.get("cash") or 0
    net_cash = float(total_cash) - float(total_debt)  # positive = net cash, negative = net debt

    # ── Intrinsic Value ──
    total_equity_value = pv_stage1 + pv_terminal + net_cash
    intrinsic_per_share = total_equity_value / shares

    # ── Discount/Premium ──
    discount_pct = (intrinsic_per_share / current_price - 1) * 100

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
        "erp": erp,
        "erp_note": erp_note,
        "beta_source": beta_source,
        "profile_name": profile_name,
        "mid_growth": mid_growth,
        "net_cash": net_cash,
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


def _render_sensitivity_table(ticker: str, base_dcf: dict) -> None:
    """Render a 3×3 WACC × Terminal Growth sensitivity table, color-coded vs current price."""
    price = base_dcf["current_price"]
    wacc_deltas = [-0.01, 0.0, 0.01]
    tg_deltas   = [-0.005, 0.0, 0.005]

    # Pre-compute 9 cells
    grid = {}
    for wd in wacc_deltas:
        for td in tg_deltas:
            r = _compute_dcf(ticker, wacc_adj=wd, tg_adj=td)
            grid[(wd, td)] = r.get("intrinsic_value") if (r and not r.get("error")) else None

    base_wacc = base_dcf["discount_rate"]
    base_tg   = base_dcf["terminal_growth"]

    # Column headers: terminal growth labels
    tg_labels = [f"{(base_tg + td)*100:.2f}%" for td in tg_deltas]
    tg_descs  = ["−0.5%", "Base", "+0.5%"]

    # Build HTML table
    _hdr_style = "padding:6px 12px;font-size:10px;color:#64748b;font-weight:700;text-align:center;letter-spacing:0.06em;"
    _lbl_style = "padding:6px 10px;font-size:10px;color:#94a3b8;font-weight:600;white-space:nowrap;"

    rows_html = ""
    for i, wd in enumerate(wacc_deltas):
        wacc_label = f"{(base_wacc + wd)*100:.2f}%"
        wacc_desc  = ["−1%", "Base", "+1%"][i]
        cells = ""
        for td in tg_deltas:
            iv = grid.get((wd, td))
            if iv is None:
                cell_val = "N/A"
                bg = "#1e293b"
                color = "#475569"
            else:
                disc = (iv / price - 1) * 100
                cell_val = f"${iv:,.0f}<br><span style='font-size:10px;'>{disc:+.1f}%</span>"
                if disc >= 20:
                    bg, color = "#052e16", "#22c55e"
                elif disc >= 0:
                    bg, color = "#1a1a00", "#a3e635"
                elif disc >= -20:
                    bg, color = "#1a0d00", "#f59e0b"
                else:
                    bg, color = "#1a0000", "#ef4444"
            _is_base = (wd == 0.0 and td == 0.0)
            border = "2px solid #FF8811" if _is_base else "1px solid #1e293b"
            cells += (
                f'<td style="padding:8px 12px;text-align:center;background:{bg};'
                f'border:{border};font-size:12px;font-weight:700;color:{color};">'
                f'{cell_val}</td>'
            )
        rows_html += (
            f'<tr>'
            f'<td style="{_lbl_style}">'
            f'<span style="color:#64748b;font-size:10px;">{wacc_desc}</span><br>'
            f'<b>{wacc_label}</b></td>'
            f'{cells}</tr>'
        )

    col_headers = "".join(
        f'<th style="{_hdr_style}"><span style="color:#64748b;">{d}</span><br>{l}</th>'
        for l, d in zip(tg_labels, tg_descs)
    )

    st.markdown(
        f'<div style="margin:4px 0 0 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">'
        f'<span style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">WACC vs Terminal Growth — Intrinsic Value per Share</span>'
        f'<span style="font-size:10px;color:#475569;">Current price: <b style="color:#e2e8f0;">${price:,.2f}</b> &nbsp;·&nbsp; Base cell outlined</span>'
        f'</div>'
        f'<table style="width:100%;border-collapse:collapse;font-family:monospace;">'
        f'<thead><tr>'
        f'<th style="{_hdr_style};text-align:left;">WACC \\ TG</th>'
        f'{col_headers}'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
            st.caption(dcf.get("erp_note", "Damodaran US ERP"))
        with a2:
            _beta_src = dcf.get("beta_source", "sector")
            _beta_src_label = "Live market beta (yfinance)" if _beta_src == "market" else "⚠ Sector avg (Damodaran Jan 2024)"
            st.metric("Levered Beta", f"{dcf['levered_beta']:.2f}")
            st.caption(f"{_beta_src_label} · Unlevered: {dcf['unlevered_beta']:.2f}")
            st.metric("D/E Ratio", f"{dcf['de_ratio']:.2f}")
        with a3:
            st.metric("Cost of Equity (Discount Rate)", f"{dcf['discount_rate']*100:.2f}%")
            st.caption(f"Rf + β × ERP = {dcf['risk_free_rate']*100:.1f}% + {dcf['levered_beta']:.2f} × {dcf['erp']*100:.1f}%")
            st.metric("Terminal Growth", f"{dcf['terminal_growth']*100:.2f}%")
            st.caption(f"Sector profile: {dcf.get('profile_name', 'Default')}")

        _net_cash = dcf.get("net_cash", 0)
        _net_cash_str = (
            f"Net Cash +{_fmt_big(_net_cash)}" if _net_cash > 0
            else f"Net Debt −{_fmt_big(abs(_net_cash))}" if _net_cash < 0
            else "Net Cash $0"
        )
        st.markdown(
            f"**Latest Reported FCF:** {_fmt_big(dcf['latest_fcf'])} | "
            f"**Initial Growth:** {dcf['initial_growth']*100:.1f}% ({dcf['growth_source']}) | "
            f"**{_net_cash_str}** | "
            f"**Shares Outstanding:** {dcf['shares']/1e9:.3f}B"
        )

    with st.expander("📊 WACC × Terminal Growth Sensitivity"):
        _render_sensitivity_table(ticker, dcf)

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
    if b <= 0:
        st.caption("Kelly unavailable — win/loss ratio is zero (no expected upside).")
        return
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
