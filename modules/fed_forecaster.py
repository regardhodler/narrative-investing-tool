"""
Module: Fed Forecaster

Fed policy probability engine — standalone sidebar module.

Includes:
- FOMC context strip (date, rate, regime)
- Fed communications tone tracker
- Bayesian probability calibration (market-implied + structural ensemble)
- Scenario probability bars
- Rate path → sector rotation & AI regime plays
- Asset impact matrix (18 assets × 4 scenarios × 3 horizons)
- Medium-term fan charts (3–12 months)
- Long-term quarterly outlook (2-year)
- Black swan risk panel + custom event input
- Causal chain — policy transmission narration

Data layer: services/fed_forecaster.py
"""

import json
import os
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from services.market_data import (
    fetch_fred_series_safe, warm_fred_cache,
)
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric
from utils.ai_tier import render_ai_tier_selector as _ff_ai_tier


# ── Shared helpers ────────────────────────────────────────────────────────────

def _section_header(title: str, badge: str = ""):
    """Render a Bloomberg-styled section header with optional engine badge."""
    badge_html = (
        f' <span style="font-size:10px;background:{COLORS["bloomberg_orange"]};'
        f'color:#000;padding:1px 6px;border-radius:3px;vertical-align:middle;'
        f'font-weight:700;letter-spacing:0.04em;">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="border-left:3px solid {COLORS["bloomberg_orange"]};'
        f'background:{COLORS["surface"]};padding:8px 14px;margin:20px 0 10px 0;'
        f'font-family:\'JetBrains Mono\',Consolas,monospace;font-size:14px;'
        f'font-weight:600;color:{COLORS["bloomberg_orange"]};letter-spacing:0.08em;'
        f'text-transform:uppercase;">{title}{badge_html}</div>',
        unsafe_allow_html=True,
    )


# ── Fed Forecaster rendering functions ───────────────────────────────────────

def _render_fed_forecaster(macro: dict, fred_data: dict):
    """Main Fed Forecaster render — FOMC strip, tone, probabilities."""
    from services.fed_forecaster import (
        fetch_zq_probabilities, fetch_fed_communications, score_fed_tone,
        adjust_probabilities, get_next_fomc, SCENARIO_KEYS, SCENARIO_LABELS,
    )
    from datetime import datetime as _dt

    # ── Section 1: FOMC Context Strip ────────────────────────────────────────
    fomc = get_next_fomc()
    fedfunds_series = fred_data.get("fedfunds")
    current_rate_str = "N/A"
    if fedfunds_series is not None and not fedfunds_series.empty:
        current_rate_str = f"{fedfunds_series.dropna().iloc[-1]:.2f}%"

    regime_label = macro.get("macro_regime") or macro.get("regime", "Unknown")
    quadrant = macro.get("quadrant", "")
    regime_color = COLORS["red"] if "Risk-Off" in regime_label else (
        COLORS["green"] if "Risk-On" in regime_label else COLORS["yellow"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🗓 Next FOMC", fomc["date"], f"{fomc['days_away']} days away")
    c2.metric("🏦 Fed Funds Rate", current_rate_str)
    c3.markdown(
        f'<div style="padding:8px 0;">'
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};font-family:\'JetBrains Mono\',monospace;'
        f'text-transform:uppercase;letter-spacing:0.06em;">Regime</div>'
        f'<div style="font-size:18px;font-weight:700;color:{regime_color};">'
        f'{regime_label} · {quadrant}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 2: Fed Communications Tracker ────────────────────────────────
    _section_header("Fed Communications")

    comms = fetch_fed_communications(max_items=5)
    comms_updated = _dt.now().strftime("%H:%M")

    _neutral_tone = {"aggregate_bias": "neutral",
                     "prob_adjustments": {k: 0.0 for k in SCENARIO_KEYS}}

    if not comms:
        st.markdown(
            f'<div style="color:{COLORS["text_dim"]};font-size:13px;">'
            f'Fed communications unavailable — tone adjustment skipped</div>',
            unsafe_allow_html=True,
        )
        tone_result = _neutral_tone
    else:
        # Show communication headlines first (no AI needed)
        for _c in comms[:3]:
            st.markdown(
                f'<div style="font-size:11px;color:#94a3b8;margin:2px 0;">'
                f'<span style="color:#64748b;">{_c.get("date","")[:10]}</span>'
                f' · {_c.get("title","")[:90]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            "[View Latest Fed Communications →](https://www.federalreserve.gov/newsevents.htm)"
        )

        comm_key = hashlib.md5(
            str([(c["title"], c["date"]) for c in comms]).encode()
        ).hexdigest()
        _tone_ss_key = f"_fed_tone_{comm_key}"

        _tone_col, _score_col = st.columns([4, 1])
        if _score_col.button("Score Tone", key="score_tone_btn", help="Groq scores Fed communication sentiment (hawkish/dovish/neutral)"):
            with st.spinner("Scoring Fed tone…"):
                tone_result = score_fed_tone(comm_key, comms)
            st.session_state[_tone_ss_key] = tone_result

        tone_result = st.session_state.get(_tone_ss_key, _neutral_tone)

        if tone_result is not _neutral_tone:
            tone = tone_result.get("aggregate_bias", tone_result.get("tone", "neutral"))
            tone_color_map = {
                "hawkish": COLORS.get("red", "#ef4444"),
                "dovish":  COLORS.get("green", "#22c55e"),
                "neutral": COLORS.get("text_dim", "#94a3b8"),
            }
            badge_color = tone_color_map.get(tone, COLORS.get("text_dim", "#94a3b8"))
            with _tone_col:
                st.markdown(
                    f'<span style="background:{badge_color};color:white;padding:4px 14px;'
                    f'border-radius:12px;font-weight:bold;font-size:14px;">'
                    f'{tone.upper()}</span>',
                    unsafe_allow_html=True,
                )
        else:
            with _tone_col:
                st.caption("Tone: neutral (not scored) · Click Score Tone to run AI analysis")

    st.caption(f"Comms as of {comms_updated}")
    st.markdown("---")

    _render_fed_probability_bars(macro, fred_data, tone_result)


def _render_fed_sector_rotation_panel(macro: dict, adj_probs: list[dict]):
    """Rate Path → Sector Rotation + Dalio Quadrant Cross-Check + AI Regime Plays."""
    from services.fed_forecaster import SCENARIO_KEYS, SCENARIO_LABELS

    _section_header("Rate Path → Sector Rotation & Regime Plays")

    dominant_key = max(SCENARIO_KEYS, key=lambda k: next(
        (r["prob"] for r in adj_probs if r["scenario"] == k), 0.0
    ))
    dominant_prob = next((r["prob"] for r in adj_probs if r["scenario"] == dominant_key), 0.25)
    dominant_label = SCENARIO_LABELS[dominant_key]

    rate_dir = (
        "cuts" if dominant_key in ("cut_25", "cut_50")
        else "hikes" if dominant_key == "hike_25"
        else "hold"
    )

    quadrant    = macro.get("quadrant", "Goldilocks")
    regime      = macro.get("macro_regime") or macro.get("regime", "Neutral")
    growth_dir  = macro.get("growth_dir", "")
    infl_dir    = macro.get("inflation_dir", "")
    macro_score = macro.get("macro_score", 50)

    _xcheck = {
        ("cuts",  "Deflation"):   ("CONFIRMED", COLORS["green"],  "Cuts match falling growth + deflation — classic easing"),
        ("cuts",  "Goldilocks"):  ("CONFIRMED", COLORS["green"],  "Dovish pivot sustains soft landing — growth continues"),
        ("cuts",  "Stagflation"): ("CAUTION",   COLORS["yellow"], "Cutting into stagflation — unusual; watch gold & real assets"),
        ("cuts",  "Reflation"):   ("CONFLICT",  COLORS["red"],    "Cutting while inflation rising — risk of re-acceleration"),
        ("hikes", "Reflation"):   ("CONFIRMED", COLORS["green"],  "Tightening to fight inflation — textbook response"),
        ("hikes", "Stagflation"): ("CONFIRMED", COLORS["yellow"], "Hiking despite weak growth — painful but necessary"),
        ("hikes", "Goldilocks"):  ("CAUTION",   COLORS["yellow"], "Preemptive hike in soft landing — watch for policy error"),
        ("hikes", "Deflation"):   ("CONFLICT",  COLORS["red"],    "Hiking into deflation — policy error risk; avoid risk assets"),
        ("hold",  "Goldilocks"):  ("CONFIRMED", COLORS["green"],  "Hold reflects balanced growth + stable inflation"),
        ("hold",  "Reflation"):   ("CAUTION",   COLORS["yellow"], "Holding while inflation rises — falling behind the curve"),
        ("hold",  "Deflation"):   ("CAUTION",   COLORS["yellow"], "Holding while growth falls — watch for forced cuts"),
        ("hold",  "Stagflation"): ("CAUTION",   COLORS["yellow"], "Trapped: can't cut (inflation) or hike (weak growth)"),
    }
    verdict, v_color, v_msg = _xcheck.get(
        (rate_dir, quadrant), ("NEUTRAL", COLORS["yellow"], "Rate path and quadrant are ambiguous")
    )
    v_icon = {"CONFIRMED": "✅", "CONFLICT": "⚡", "CAUTION": "⚠", "NEUTRAL": "—"}.get(verdict, "—")

    _rotation = {
        "cuts": {
            "favor": [
                ("REITs",        "Lower cap rates → higher property valuations"),
                ("Utilities",    "Bond proxy re-rates higher as yields fall"),
                ("Growth Tech",  "Long-duration assets benefit from lower discount rate"),
                ("Small Cap",    "Rate-sensitive balance sheets get relief"),
                ("EM",           "Dollar weakening opens EM carry trade"),
                ("HY Bonds",     "Risk-on: spreads compress with easier conditions"),
            ],
            "avoid": [
                ("USD",          "Dollar weakens on rate differential compression"),
                ("Financials",   "NIM compression on deep cuts"),
                ("Short Bills",  "Yield cliff as rates reset lower"),
            ],
        },
        "hikes": {
            "favor": [
                ("Financials",   "NIM expansion: borrow short, lend long"),
                ("Energy",       "Inflation proxy — hike cycles often coincide"),
                ("USD",          "Rate differential attracts capital flows"),
                ("Short Bills",  "T-bills/CDs yield rises with Fed Funds"),
                ("Value/Div.",   "Duration risk penalizes growth; value outperforms"),
            ],
            "avoid": [
                ("REITs",        "Cap rate expansion compresses valuations"),
                ("Utilities",    "Bond proxy sells off as rate alternative improves"),
                ("Growth Tech",  "High-multiple stocks penalized by higher discount rate"),
                ("EM",           "Dollar strength + capital outflows"),
                ("Long Bonds",   "Duration risk: prices fall as yields rise"),
            ],
        },
        "hold": {
            "favor": [
                ("Quality GARP", "Growth at reasonable price — stable rate environment"),
                ("Div. Growth",  "Compounders shine when rates are stable"),
                ("Sector Neutral","Market-cap weighted: no directional rate bet"),
            ],
            "avoid": [
                ("High Leverage","No rate relief; refinancing risk persists"),
                ("Rate Plays",   "Avoid pure rate-direction bets when Fed is on hold"),
            ],
        },
    }
    rotation = _rotation.get(rate_dir, _rotation["hold"])
    _pill_color = (
        COLORS["green"] if rate_dir == "cuts"
        else COLORS["red"] if rate_dir == "hikes"
        else COLORS["yellow"]
    )

    _h1, _h2 = st.columns(2)
    with _h1:
        st.markdown(
            f'<div style="padding:10px 14px;background:{_pill_color}22;'
            f'border:1px solid {_pill_color}55;border-radius:6px;height:90px;">'
            f'<div style="font-size:10px;color:{COLORS["text_dim"]};font-weight:700;'
            f'letter-spacing:0.08em;text-transform:uppercase;">Dominant Rate Path</div>'
            f'<div style="font-size:20px;font-weight:700;color:{_pill_color};margin:4px 0;">'
            f'{dominant_label}</div>'
            f'<div style="font-size:12px;color:{COLORS["text_dim"]};">'
            f'{int(round(dominant_prob * 100))}% market probability</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _h2:
        st.markdown(
            f'<div style="padding:10px 14px;background:{v_color}22;'
            f'border:1px solid {v_color}55;border-radius:6px;height:90px;">'
            f'<div style="font-size:10px;color:{COLORS["text_dim"]};font-weight:700;'
            f'letter-spacing:0.08em;text-transform:uppercase;">Dalio Quadrant: {quadrant}</div>'
            f'<div style="font-size:17px;font-weight:700;color:{v_color};margin:4px 0;">'
            f'{v_icon} {verdict}</div>'
            f'<div style="font-size:11px;color:{COLORS["text_dim"]};line-height:1.4;">{v_msg}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    _r1, _r2 = st.columns(2)
    with _r1:
        st.markdown(
            f'<div style="font-size:10px;font-weight:700;color:{COLORS["green"]};'
            f'letter-spacing:0.08em;margin-bottom:6px;text-transform:uppercase;">Favor</div>',
            unsafe_allow_html=True,
        )
        for name, reason in rotation["favor"]:
            st.markdown(
                f'<div style="padding:5px 0;border-bottom:1px solid {COLORS["grid"]};">'
                f'<span style="font-size:13px;font-weight:600;color:{COLORS["text"]};">{name}</span>'
                f'<span style="font-size:11px;color:{COLORS["text_dim"]};"> — {reason}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with _r2:
        st.markdown(
            f'<div style="font-size:10px;font-weight:700;color:{COLORS["red"]};'
            f'letter-spacing:0.08em;margin-bottom:6px;text-transform:uppercase;">Avoid / Underweight</div>',
            unsafe_allow_html=True,
        )
        for name, reason in rotation["avoid"]:
            st.markdown(
                f'<div style="padding:5px 0;border-bottom:1px solid {COLORS["grid"]};">'
                f'<span style="font-size:13px;font-weight:600;color:{COLORS["text"]};">{name}</span>'
                f'<span style="font-size:11px;color:{COLORS["text_dim"]};"> — {reason}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ── AI Regime Plays ────────────────────────────────────────────────────
    _has_xai = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    _prev_fp_tier = st.session_state.get("_fed_plays_tier_prev")
    _use_cl, _cl_model = _ff_ai_tier(
        key="fed_plays_engine_radio",
        label="Engine",
        recommendation="🧠 Regard Mode recommended for rate path analysis — Grok 4.1 reasoning handles Fed dot plots well",
    )
    _sel_tier = st.session_state.get("fed_plays_engine_radio", "⚡ Freeloader Mode")

    st.session_state["_fed_plays_tier_prev"] = _sel_tier

    _gen = st.button("Generate Rate-Path Plays", type="primary", key="gen_fed_plays_btn")

    if _gen or st.session_state.get("_fed_plays_result"):
        if _gen:
            _sig = (
                f"Dalio Quadrant: {quadrant}, Regime: {regime}, "
                f"Rate Path: {dominant_label} ({int(round(dominant_prob*100))}% probability), "
                f"Rate Direction: {rate_dir}, Quadrant Signal: {verdict} — {v_msg}, "
                f"Growth: {growth_dir}, Inflation: {infl_dir}"
            )
            _norm_score = (macro_score - 50) / 50
            from services.claude_client import suggest_regime_plays
            with st.spinner("Generating rate-path plays..."):
                _plays = suggest_regime_plays(
                    regime, _norm_score, _sig,
                    use_claude=_use_cl, model=_cl_model,
                )
            st.session_state["_fed_plays_result"] = _plays
            st.session_state["_fed_plays_result_ts"] = __import__("datetime").datetime.now()
            st.session_state["_fed_plays_engine"] = _sel_tier
            st.session_state["_fed_plays_tier"] = _sel_tier
            from services.play_log import append_play as _append_play
            _append_play("Rate-Path Plays", _sel_tier, _plays,
                         meta={"regime": regime})
            st.session_state["_regime_plays_tier"] = _sel_tier
            st.session_state["_regime_context"] = {
                "regime": regime,
                "score": _norm_score,
                "signal_summary": _sig,
                "quadrant": quadrant,
            }
            st.session_state["_regime_context_ts"] = __import__("datetime").datetime.now()
        else:
            _plays = st.session_state["_fed_plays_result"]

        _eng_label = st.session_state.get("_fed_plays_engine", "⚡ Freeloader Mode")
        st.caption(f"*{_eng_label} · Rate Path: {dominant_label} · Quadrant: {quadrant}*")

        if _plays and (_plays.get("sectors") or _plays.get("stocks") or _plays.get("bonds")):
            _oc = COLORS["orange"]
            _sc, _stc, _bc = st.columns(3)

            def _play_items(col, header, items, show_reason=False):
                with col:
                    st.markdown(
                        f'<div style="font-size:10px;font-weight:700;color:{COLORS["text_dim"]};'
                        f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">{header}</div>',
                        unsafe_allow_html=True,
                    )
                    for item in (items or []):
                        _stars = "★" * item.get("conviction", 1) + "☆" * (3 - item.get("conviction", 1))
                        _name = item.get("name") or item.get("ticker", "")
                        _reason = f'<br><span style="font-size:11px;color:{COLORS["text_dim"]};">{item.get("reason","")}</span>' if show_reason else ""
                        st.markdown(
                            f'<div style="padding:4px 0;border-bottom:1px solid {COLORS["grid"]};">'
                            f'<span style="font-size:13px;font-weight:600;">{_name}</span> '
                            f'<span style="font-size:11px;color:{_oc};">{_stars}</span>{_reason}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            _play_items(_sc,  "Sectors",     _plays.get("sectors", []))
            _play_items(_stc, "Stocks",      _plays.get("stocks",  []), show_reason=True)
            _play_items(_bc,  "Bonds/Macro", _plays.get("bonds",   []), show_reason=True)

            if _plays.get("rationale"):
                st.markdown(
                    f'<div style="margin-top:10px;padding:10px 14px;'
                    f'border-left:3px solid {COLORS["orange"]};background:{COLORS["surface"]}88;'
                    f'border-radius:0 4px 4px 0;">'
                    f'<div style="font-size:10px;font-weight:700;color:{COLORS["text_dim"]};'
                    f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">Rationale</div>'
                    f'<div style="font-size:13px;color:{COLORS["text"]};line-height:1.5;">'
                    f'{_plays["rationale"]}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")


def _render_macro_event_strip():
    """Compact upcoming macro events bar: FOMC, CPI, NFP countdowns."""
    from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
    fomc = get_next_fomc()
    cpi  = get_next_cpi()
    nfp  = get_next_nfp()

    def _pill(label: str, ev: dict) -> str:
        days = ev["days_away"]
        color = "#ef4444" if days <= 7 else ("#f59e0b" if days <= 21 else "#64748b")
        bg = "#2d1010" if days <= 7 else ("#2d2010" if days <= 21 else "#1e293b")
        return (
            f'<span style="background:{bg};border:1px solid {color};border-radius:4px;'
            f'padding:3px 10px;margin-right:6px;font-size:11px;color:{color};font-weight:600;">'
            f'{label} {ev["date"]} <span style="opacity:0.7;">({days}d)</span></span>'
        )

    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-right:8px;">UPCOMING</span>'
        f'{_pill("FOMC", fomc)}{_pill("CPI", cpi)}{_pill("NFP", nfp)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_calibration_table(calib: dict):
    """Render the Bayesian probability calibration table."""
    posteriors = calib.get("posteriors", [])
    signals    = calib.get("signals_used", {})
    if not posteriors:
        return

    dom_scenario = max(posteriors, key=lambda r: r["posterior_pct"])["scenario"]
    _scenario_colors = {
        "hold":    "#f0c040",
        "cut_25":  "#40c080",
        "cut_50":  "#22c55e",
        "hike_25": "#e05050",
    }

    chips = ""
    for k, v in signals.items():
        if v is None:
            continue
        chips += (
            f'<span style="background:#1e293b;border:1px solid #334155;border-radius:3px;'
            f'padding:2px 8px;margin-right:4px;font-size:10px;color:#94a3b8;">'
            f'{k.replace("_"," ").upper()}: {v}</span>'
        )

    rows = ""
    for row in posteriors:
        sc   = row["scenario"]
        col  = _scenario_colors.get(sc, "#888")
        is_dom = sc == dom_scenario
        post_style = f'color:{col};font-weight:700;font-size:13px;' if is_dom else f'color:{col};'
        rows += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:6px 10px;font-size:12px;color:#ccc;white-space:nowrap;">{row["label"]}</td>'
            f'<td style="padding:6px 10px;font-size:12px;color:#888;text-align:right;">{row["market_pct"]:.0f}%</td>'
            f'<td style="padding:6px 10px;font-size:12px;color:#888;text-align:right;">{row["structural_pct"]:.0f}%</td>'
            f'<td style="padding:6px 10px;text-align:right;{post_style}">{row["posterior_pct"]:.0f}%</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:#555;text-align:right;">±{row["band_pct"]:.0f}%</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:#64748b;max-width:260px;">{row["rationale"]}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:6px;'
        f'padding:14px 16px;margin-bottom:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="font-size:12px;font-weight:700;color:#60a5fa;letter-spacing:0.08em;">'
        f'PROBABILITY CALIBRATION LAYER</span>'
        f'<span style="font-size:10px;color:#475569;">✦ Bayesian + Market-Implied Ensemble</span>'
        f'</div>'
        f'<div style="margin-bottom:10px;">{chips}</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="border-bottom:1px solid #334155;">'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:left;font-weight:600;">SCENARIO</th>'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:right;font-weight:600;">MARKET-IMPLIED</th>'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:right;font-weight:600;">STRUCTURAL</th>'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:right;font-weight:600;">POSTERIOR</th>'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:right;font-weight:600;">BAND</th>'
        f'<th style="padding:4px 10px;font-size:10px;color:#475569;text-align:left;font-weight:600;">SIGNAL RATIONALE</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_bayesian_bar_chart(calib: dict):
    """Grouped bar chart: Market-Implied → Structural → Posterior per scenario."""
    posteriors = calib.get("posteriors", [])
    if not posteriors:
        return
    from services.fed_forecaster import SCENARIO_KEYS, SCENARIO_LABELS
    _sc_colors = {
        "hold":    "#f0c040",
        "cut_25":  "#40c080",
        "cut_50":  "#22c55e",
        "hike_25": "#e05050",
    }
    labels   = [SCENARIO_LABELS.get(p["scenario"], p["scenario"]) for p in posteriors]
    market   = [p.get("market_pct", 0) for p in posteriors]
    struct   = [p.get("structural_pct", 0) for p in posteriors]
    posterior = [p.get("posterior_pct", 0) for p in posteriors]
    bar_colors = [_sc_colors.get(p["scenario"], "#888") for p in posteriors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Market-Implied", x=labels, y=market,
        marker_color="#3b82f6", opacity=0.7,
        text=[f"{v:.0f}%" for v in market], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Structural", x=labels, y=struct,
        marker_color="#64748b", opacity=0.7,
        text=[f"{v:.0f}%" for v in struct], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Posterior", x=labels, y=posterior,
        marker_color=bar_colors, opacity=1.0,
        text=[f"{v:.0f}%" for v in posterior], textposition="outside",
    ))
    apply_dark_layout(fig, title="Bayesian Calibration — Prior → Posterior")
    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 105], ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Market-Implied = ZQ futures signal · Structural = macro signals (PCE, unemployment, VIX, spreads) · Posterior = Bayesian blend")


def _render_fed_probability_bars(macro: dict, fred_data: dict, tone_result: dict):
    """Sections 3–6: probability bars, asset matrix, causal chain, fan charts."""
    from services.fed_forecaster import (
        fetch_zq_probabilities, adjust_probabilities, calibrate_probabilities,
        build_fed_context, SCENARIO_KEYS, SCENARIO_LABELS,
    )
    from datetime import datetime as _dt

    _render_macro_event_strip()

    _section_header("Scenario Probabilities (Fed Funds Futures)")

    base_probs = fetch_zq_probabilities()
    futures_updated = _dt.now().strftime("%H:%M")
    # Store raw market-implied probs (pre-Bayesian) for transparency display
    st.session_state["_rate_path_raw_market"] = base_probs

    if any(r.get("data_unavailable") for r in base_probs):
        st.warning("⚠ Futures data unavailable — showing equal-weight 25% per scenario")

    adj_probs = adjust_probabilities(base_probs, tone_result, macro=macro)

    try:
        _fed_ctx = build_fed_context(macro, fred_data)
        _calib = calibrate_probabilities(base_probs, adj_probs, _fed_ctx)
        _render_calibration_table(_calib)
        _render_bayesian_bar_chart(_calib)
        _post_map = {p["scenario"]: p["posterior"] for p in _calib["posteriors"]}
        _final_probs = [
            {**r, "prob": _post_map.get(r["scenario"], r["prob"])}
            for r in adj_probs
        ]
        st.session_state["_calibrated_rate_probs"] = _calib["posteriors"]
    except Exception:
        _final_probs = adj_probs

    if _final_probs:
        _dp = max(_final_probs, key=lambda r: r.get("prob", 0))
        st.session_state["_rate_path_probs"] = _final_probs
        st.session_state["_rate_path_probs_ts"] = __import__("datetime").datetime.now()
        st.session_state["_dominant_rate_path"] = {
            "scenario": _dp.get("scenario", ""),
            "prob_pct": round(_dp.get("prob", 0) * 100, 1),
        }
        _ff_series = fred_data.get("fedfunds")
        _ff_rate = float(_ff_series.dropna().iloc[-1]) if (_ff_series is not None and not _ff_series.empty) else None
        st.session_state["_fed_funds_rate"] = _ff_rate
        from services.play_log import append_play as _append_play
        _append_play("Fed Forecaster", st.session_state.get("fed_engine_radio", "⚡ Freeloader Mode"),
                     {"rate_path_probs": _final_probs, "dominant": st.session_state["_dominant_rate_path"]},
                     meta={"fed_funds_rate": _ff_rate})

    source = (base_probs[0].get("source", "fallback") if base_probs else "fallback")
    source_label = {
        "CME FedWatch": "Source: CME FedWatch",
        "yfinance": "Source: ZQ Futures (yfinance)",
        "fallback": "⚠ Source: equal-weight fallback",
    }.get(source, f"Source: {source}")
    st.caption(f"{source_label}  |  As of {futures_updated}  |  ✦ Bayesian calibration applied")

    scenario_colors = {
        "hold":    COLORS.get("yellow", "#f0c040"),
        "cut_25":  COLORS.get("green",  "#40c080"),
        "cut_50":  COLORS.get("green",  "#40c080"),
        "hike_25": COLORS.get("red",    "#e05050"),
    }

    labels = [SCENARIO_LABELS[k] for k in SCENARIO_KEYS]
    probs  = [next((r["prob"] for r in _final_probs if r["scenario"] == k), 0.25) for k in SCENARIO_KEYS]
    deltas = [next((r.get("delta", 0.0) for r in _final_probs if r["scenario"] == k), 0.0) for k in SCENARIO_KEYS]
    colors = [scenario_colors[k] for k in SCENARIO_KEYS]

    text_labels = []
    for p, d in zip(probs, deltas):
        pct = int(round(p * 100))
        if abs(d) > 0.005:
            sign = "▲" if d > 0 else "▼"
            pp = int(round(abs(d) * 100))
            text_labels.append(f"{pct}%  {sign}{pp}pp")
        else:
            text_labels.append(f"{pct}%")

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        text=text_labels,
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
    ))
    apply_dark_layout(fig)
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw vs Calibrated comparison
    with st.expander("🔬 Market-Implied vs Calibrated", expanded=False):
        st.caption("Market-implied = pure futures signal. Calibrated = Bayesian blend (40% futures + 40% macro signals + 20% Fed tone).")
        _raw_map = {r["scenario"]: r.get("prob", 0.25) for r in base_probs}
        _cal_map = {r["scenario"]: r.get("prob", 0.25) for r in _final_probs}
        _cmp_rows = []
        for k in SCENARIO_KEYS:
            raw_pct = int(round(_raw_map.get(k, 0.25) * 100))
            cal_pct = int(round(_cal_map.get(k, 0.25) * 100))
            diff = cal_pct - raw_pct
            diff_str = f"+{diff}pp" if diff > 0 else (f"{diff}pp" if diff < 0 else "—")
            _cmp_rows.append({
                "Scenario": SCENARIO_LABELS[k],
                f"Market ({source})": f"{raw_pct}%",
                "Calibrated": f"{cal_pct}%",
                "Δ": diff_str,
            })
        import pandas as _pd_fc
        st.dataframe(_pd_fc.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

    if st.button("🔄 Refresh Forecaster", key="refresh_forecaster"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    _render_dot_plot_vs_futures(_final_probs, fred_data)
    _render_data_sensitivity(_final_probs, locals().get("_fed_ctx", {}))

    st.markdown("---")

    _render_fed_sector_rotation_panel(macro, _final_probs)
    _render_fed_asset_matrix(macro, fred_data, _final_probs)


def _render_dot_plot_vs_futures(final_probs: list[dict], fred_data: dict):
    """Section: Fed Dot Plot vs Futures — where the market diverges from the Fed's own projections."""
    from services.fed_forecaster import SCENARIO_KEYS
    _SCENARIO_DELTAS = {"cut_50": -0.50, "cut_25": -0.25, "hold": 0.00, "hike_25": +0.25}

    _section_header("Dot Plot vs Futures Gap")
    st.caption(
        "Fed's own median rate projection (SEP) vs market-implied terminal rate (ZQ futures). "
        "Gap = where trades live — divergence signals re-pricing risk."
    )

    # Current rate
    _ff_series = fred_data.get("fedfunds")
    current_rate = float(_ff_series.dropna().iloc[-1]) if (_ff_series is not None and not _ff_series.empty) else st.session_state.get("_fed_funds_rate") or 4.50

    # Futures-implied terminal rate: probability-weighted delta from current
    _weighted_delta = sum(
        r.get("prob", 0.25) * _SCENARIO_DELTAS.get(r["scenario"], 0.0)
        for r in final_probs
    )
    futures_implied = current_rate + _weighted_delta

    # Latest SEP median projections (March 2026 FOMC — updated each SEP meeting)
    # Sources: FOMC Summary of Economic Projections, March 2026
    _SEP = {
        "2026 Year-End": 3.875,   # median dot for end-2026
        "2027 Year-End": 3.375,   # median dot for end-2027
        "Longer Run":    2.875,   # longer-run neutral rate median
    }
    _SEP_DATE = "Mar 2026 SEP"

    # Compute implied cuts for each data point (relative to current rate)
    def _cuts_label(rate: float) -> str:
        delta_bp = int(round((rate - current_rate) * 100))
        if delta_bp == 0:
            return "no change"
        elif delta_bp < 0:
            return f"{abs(delta_bp)}bp of cuts"
        else:
            return f"{delta_bp}bp of hikes"

    _futures_color = COLORS.get("green", "#22c55e") if futures_implied <= _SEP["2026 Year-End"] else COLORS.get("red", "#ef4444")

    # Chart: rate levels with implied-cut annotation in bar text
    _chart_labels = [
        _SEP_DATE + " · Longer Run",
        _SEP_DATE + " · 2027",
        _SEP_DATE + " · 2026",
        "Futures Implied",
        "Current Rate",
    ]
    _chart_vals = [_SEP["Longer Run"], _SEP["2027 Year-End"], _SEP["2026 Year-End"], futures_implied, current_rate]
    _chart_colors = [
        COLORS.get("bloomberg_orange", "#FF8811"),
        COLORS.get("bloomberg_orange", "#FF8811"),
        COLORS.get("bloomberg_orange", "#FF8811"),
        _futures_color,
        COLORS.get("text_dim", "#94a3b8"),
    ]
    # Bar text: "3.88% (62bp cuts)" — rate level + what that implies in cut terms
    _chart_text = [
        f"{v:.2f}%  ({_cuts_label(v)})" for v in _chart_vals
    ]

    fig = go.Figure(go.Bar(
        x=_chart_vals,
        y=_chart_labels,
        orientation="h",
        marker_color=_chart_colors,
        marker_line_width=0,
        text=_chart_text,
        textposition="outside",
    ))
    apply_dark_layout(fig)
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=160, t=10, b=10),
        xaxis=dict(range=[0, max(_chart_vals) * 1.35], ticksuffix="%", showgrid=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    # Reference line at current rate for visual clarity
    fig.add_vline(
        x=current_rate,
        line_dash="dot",
        line_color=COLORS.get("text_dim", "#64748b"),
        line_width=1,
        annotation_text="Today",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=COLORS.get("text_dim", "#64748b"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Gap analysis
    gap_26 = futures_implied - _SEP["2026 Year-End"]
    gap_bp = int(round(gap_26 * 100))
    abs_bp = abs(gap_bp)

    if gap_bp < -10:
        signal_color = COLORS.get("green", "#22c55e")
        signal_label = f"Futures pricing {abs_bp}bp MORE CUTS than Fed dots → dovish repricing risk"
        signal_sub   = "If Fed validates: bonds rally, growth stocks outperform. If pushback: rate shock."
    elif gap_bp > 10:
        signal_color = COLORS.get("red", "#ef4444")
        signal_label = f"Futures pricing {abs_bp}bp FEWER CUTS than Fed dots → hawkish drift risk"
        signal_sub   = "Rates staying higher longer. Value/financials benefit, duration gets punished."
    else:
        signal_color = COLORS.get("yellow", "#f0c040")
        signal_label = f"Futures aligned with Fed dots (gap: {gap_bp:+d}bp) → low re-pricing risk"
        signal_sub   = "Market and Fed broadly agree. Watch incoming data for the next catalyst."

    st.markdown(
        f'<div style="border-left:4px solid {signal_color};background:{COLORS.get("surface","#1e293b")};'
        f'border-radius:0 6px 6px 0;padding:10px 16px;margin:8px 0;">'
        f'<div style="font-size:13px;font-weight:700;color:{signal_color};">{signal_label}</div>'
        f'<div style="font-size:12px;color:{COLORS.get("text_dim","#94a3b8")};margin-top:4px;">{signal_sub}</div>'
        f'<div style="font-size:11px;color:{COLORS.get("text_dim","#64748b")};margin-top:6px;">'
        f'Current: {current_rate:.2f}%  ·  Futures implied: {futures_implied:.2f}%  ·  '
        f'Fed dot 2026: {_SEP["2026 Year-End"]:.3f}%  ·  Neutral: {_SEP["Longer Run"]:.3f}%  ·  '
        f'Source: {_SEP_DATE} (hardcoded — update each SEP meeting)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_data_sensitivity(final_probs: list[dict], context: dict):
    """Section: Data Sensitivity — what macro print would flip the dominant scenario?"""
    from services.fed_forecaster import (
        SCENARIO_KEYS, SCENARIO_LABELS, calibrate_probabilities, adjust_probabilities,
    )

    with st.expander("📊 Data Sensitivity — What Would Flip the Dominant Scenario?", expanded=False):
        st.caption(
            "Runs Bayesian re-calibration with perturbed macro inputs. "
            "Shows how much each signal would need to change to shift the dominant scenario probability by ≥10pp."
        )

        if not context:
            st.info("Context unavailable — generate forecast first.")
            return

        dominant = max(final_probs, key=lambda r: r.get("prob", 0))
        dom_key   = dominant.get("scenario", "hold")
        dom_prob  = dominant.get("prob", 0.25)
        dom_label = SCENARIO_LABELS.get(dom_key, dom_key)

        st.markdown(
            f'<div style="font-size:12px;color:{COLORS.get("text_dim","#94a3b8")};margin-bottom:10px;">'
            f'Dominant scenario: <b style="color:{COLORS.get("bloomberg_orange","#FF8811")};">'
            f'{dom_label} ({int(round(dom_prob*100))}%)</b> — showing what shifts it by ≥10pp</div>',
            unsafe_allow_html=True,
        )

        def _perturb_prob(signal: str, delta: float) -> float:
            """Return dominant scenario probability after perturbing one signal."""
            ctx2 = dict(context)
            if signal == "core_pce":
                ctx2["core_pce"] = (ctx2.get("core_pce") or 2.5) + delta
            elif signal == "unemployment":
                ctx2["unemployment"] = (ctx2.get("unemployment") or 4.0) + delta
            elif signal == "vix_z":
                ctx2["vix_z"] = (ctx2.get("vix_z") or 0.0) + delta
            elif signal == "credit_z":
                ctx2["credit_z"] = (ctx2.get("credit_z") or 0.0) + delta
            try:
                _c = calibrate_probabilities(final_probs, final_probs, ctx2)
                _pm = {p["scenario"]: p["posterior"] for p in _c["posteriors"]}
                return _pm.get(dom_key, dom_prob)
            except Exception:
                return dom_prob

        _rows = []
        _scenarios = [
            ("core_pce",    +0.30, "CPI/PCE +0.3pp",         "Hot inflation print"),
            ("core_pce",    -0.30, "CPI/PCE -0.3pp",         "Cool inflation print"),
            ("unemployment",+0.30, "Unemployment +0.3pp",     "Softer jobs report"),
            ("unemployment",-0.30, "Unemployment -0.3pp",     "Strong jobs report"),
            ("vix_z",       +1.00, "VIX shock (+1σ)",         "Sudden market stress"),
            ("vix_z",       -0.50, "VIX calms (-0.5σ)",       "Stress relief rally"),
            ("credit_z",    +0.80, "Credit spreads widen",    "Risk-off / HY selloff"),
        ]

        for sig, delta, label, desc in _scenarios:
            new_prob = _perturb_prob(sig, delta)
            shift_pp = int(round((new_prob - dom_prob) * 100))
            if shift_pp > 0:
                shift_str = f'+{shift_pp}pp'
                shift_color = COLORS.get("green", "#22c55e")
                direction = "▲ Reinforces"
            elif shift_pp < 0:
                shift_str = f'{shift_pp}pp'
                shift_color = COLORS.get("red", "#ef4444")
                direction = "▼ Weakens"
            else:
                shift_str = "—"
                shift_color = COLORS.get("text_dim", "#64748b")
                direction = "Neutral"
            _rows.append({
                "Signal": label,
                "Description": desc,
                f"{dom_label} Δ": shift_str,
                "Direction": direction,
                "_color": shift_color,
                "_abs": abs(shift_pp),
            })

        # Sort by impact magnitude
        _rows.sort(key=lambda r: r["_abs"], reverse=True)

        # Render table
        _header_html = (
            f'<div style="display:grid;grid-template-columns:1fr 1.4fr 90px 120px;'
            f'gap:0;padding:6px 12px;background:{COLORS.get("bg","#0f172a")};'
            f'font-size:10px;font-weight:700;color:{COLORS.get("text_dim","#94a3b8")};'
            f'text-transform:uppercase;letter-spacing:0.07em;border-radius:6px 6px 0 0;">'
            f'<span>Signal</span><span>What it means</span>'
            f'<span style="text-align:right;">Δ {dom_label[:8]}</span>'
            f'<span style="text-align:center;">Effect</span></div>'
        )
        _rows_html = ""
        for i, row in enumerate(_rows):
            bg = COLORS.get("surface", "#1e293b") if i % 2 == 0 else COLORS.get("bg", "#0f172a")
            _rows_html += (
                f'<div style="display:grid;grid-template-columns:1fr 1.4fr 90px 120px;'
                f'gap:0;padding:7px 12px;background:{bg};font-size:12px;">'
                f'<span style="font-weight:600;">{row["Signal"]}</span>'
                f'<span style="color:{COLORS.get("text_dim","#94a3b8")};">{row["Description"]}</span>'
                f'<span style="text-align:right;font-weight:700;color:{row["_color"]};">{row[f"{dom_label} Δ"]}</span>'
                f'<span style="text-align:center;color:{row["_color"]};font-size:11px;">{row["Direction"]}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="border:1px solid {COLORS.get("border","#334155")};border-radius:6px;'
            f'overflow:hidden;margin-top:8px;">'
            f'{_header_html}{_rows_html}</div>',
            unsafe_allow_html=True,
        )

        # Flip threshold indicator
        _flip_candidates = [r for r in _rows if r["_abs"] >= 10]
        if _flip_candidates:
            _biggest = _flip_candidates[0]
            st.markdown(
                f'<div style="margin-top:10px;font-size:12px;color:{COLORS.get("text_dim","#94a3b8")};">'
                f'⚠ Largest flip risk: <b>{_biggest["Signal"]}</b> would shift '
                f'<b style="color:{_biggest["_color"]};">{_biggest[f"{dom_label} Δ"]}</b> '
                f'on {dom_label}</div>',
                unsafe_allow_html=True,
            )


def _render_fed_asset_matrix(macro: dict, fred_data: dict, adj_probs: list[dict]):
    """Section 4: grouped 18-asset near-term impact matrix."""
    from services.fed_forecaster import (
        build_fed_context, generate_expanded_forecast,
        SCENARIO_KEYS, SCENARIO_LABELS, ASSET_LABELS as SVC_ASSET_LABELS,
    )
    import json as _json

    context = build_fed_context(macro, fred_data)
    context_json   = _json.dumps(context)
    scenarios_json = _json.dumps(adj_probs)

    _has_xai = bool(os.getenv("XAI_API_KEY"))
    _has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    _tier_options = ["⚡ Freeloader Mode"] + (["🧠 Regard Mode"] if _has_xai else []) + (["👑 Highly Regarded Mode"] if _has_claude else [])
    _tier_hints   = {"⚡ Freeloader Mode": "", "🧠 Regard Mode": "Grok 4.1", "👑 Highly Regarded Mode": "Sonnet"}
    _tier_map     = {"⚡ Freeloader Mode": "groq", "🧠 Regard Mode": "grok", "👑 Highly Regarded Mode": "sonnet"}

    if _has_xai or _has_claude:
        _prev_tier = st.session_state.get("_fed_tier", "⚡ Freeloader Mode")
        _tier_cols = st.columns([4, 1])
        with _tier_cols[0]:
            _selected_tier = st.radio(
                "Fed Forecaster Engine",
                _tier_options,
                index=_tier_options.index(_prev_tier) if _prev_tier in _tier_options else 0,
                horizontal=True,
                key="fed_engine_radio",
                help="Groq = free/fast. Regard Mode = Grok 4.1 (~$0.03). Highly Regarded Mode = Claude Sonnet (~$0.12, most accurate).",
            )
            st.markdown(
                '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
                'margin-top:-10px;margin-bottom:2px;">'
                '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
                '</div>',
                unsafe_allow_html=True,
            )
        _hint = _tier_hints[_selected_tier]
        if _hint:
            st.caption(f"*{_hint}*")
        st.caption("💡 🧠 Grok 4.1 sufficient here — probability weighting is math-heavy; LLM formats results")
        if _selected_tier != _prev_tier:
            generate_expanded_forecast.clear()
            st.session_state["_fed_tier"] = _selected_tier
            st.rerun()
        _model_tier = _tier_map[_selected_tier]
    else:
        _selected_tier = "⚡ Freeloader Mode"
        _model_tier = "groq"

    _use_claude = _model_tier != "groq"

    _forecast_key = f"_fed_forecast_run_{_model_tier}"
    _gen_col, _refresh_col = st.columns([5, 1])
    if _refresh_col.button("🔄 Refresh", key="refresh_forecast", help="Re-fetch all forecast data"):
        generate_expanded_forecast.clear()
        st.session_state.pop(_forecast_key, None)
        st.rerun()

    if _forecast_key not in st.session_state:
        st.markdown(
            f'<div style="background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;'
            f'padding:20px 24px;text-align:center;margin:12px 0;">'
            f'<div style="font-size:13px;color:#64748b;margin-bottom:12px;">'
            f'Asset matrix, causal chain &amp; fan charts are generated on demand to keep the page fast.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⚡ Generate AI Forecast", key="gen_forecast_btn", type="primary", use_container_width=False):
            st.session_state[_forecast_key] = True
            st.rerun()
        return

    with st.spinner("Generating AI forecast…"):
        full_expanded = generate_expanded_forecast(context_json, scenarios_json, model_tier=_model_tier)

    expanded = full_expanded  # expanded_forecast is a superset of matrix_forecast

    status = full_expanded.get("_call_status", {})
    status_parts = []
    for call_name, msg in status.items():
        if msg == "ok":
            status_parts.append(f"✓ {call_name}")
        else:
            status_parts.append(f"✗ {call_name}: {msg}")
    _engine_badge_map = {"groq": "⚡ Freeloader Mode", "grok": "🧠 Regard Mode", "sonnet": "👑 Highly Regarded Mode"}
    _engine_badge = _engine_badge_map.get(full_expanded.get("_core_engine", "groq"), "⚡ Freeloader Mode")
    if status_parts:
        _gen_col.caption(f"{_engine_badge}  |  " + "  |  ".join(status_parts))

    medium = expanded.get("medium_term", {})

    _medium_has_data = any(bool(assets) for assets in medium.values())
    if not _medium_has_data:
        st.warning("⚠ Medium-term forecast data unavailable — check Groq API status above.")
        st.markdown("---")
        _render_fed_fan_charts(expanded.get("medium_term", {}), adj_probs, full_expanded, use_claude=_use_claude, engine=_selected_tier)
        return

    _h_col, _t_col = st.columns([3, 1])
    _h_col.markdown(
        f'<div style="font-size:16px;font-weight:700;letter-spacing:0.04em;'
        f'padding:4px 0;">Asset Impact Matrix</div>',
        unsafe_allow_html=True,
    )
    horizon = _t_col.radio(
        "Horizon",
        options=["3M", "6M", "1Y"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="asset_matrix_horizon",
    )
    _horizon_index = {"3M": 2, "6M": 5, "1Y": 11}[horizon]
    _horizon_label = {"3M": "3-Month", "6M": "6-Month", "1Y": "1-Year"}[horizon]
    st.caption(f"{_horizon_label} cumulative % change per scenario")

    GROUP_ORDER = [
        ("🇺🇸 US Equities",    ["spy", "qqq", "iwm", "dji"]),
        ("🏦 Bonds",            ["bonds_long", "bonds_short"]),
        ("🛢 Commodities",      ["oil", "natgas", "gold", "silver"]),
        ("🌏 International",    ["china", "india", "japan", "europe"]),
        ("💵 Dollar",           ["usd"]),
    ]

    header_cols = st.columns([2] + [1] * 4)
    header_cols[0].markdown(
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};'
        f'font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;'
        f'letter-spacing:0.06em;">Asset</div>', unsafe_allow_html=True
    )
    for i, key in enumerate(SCENARIO_KEYS):
        prob = next((r["prob"] for r in adj_probs if r["scenario"] == key), 0.25)
        header_cols[i+1].markdown(
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
            f'font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;'
            f'letter-spacing:0.06em;">{SCENARIO_LABELS[key]}<br>'
            f'<span style="color:{COLORS["text"]};">{int(round(prob*100))}%</span></div>',
            unsafe_allow_html=True
        )

    for group_name, assets in GROUP_ORDER:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{COLORS["text_dim"]};'
            f'margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.08em;">'
            f'{group_name}</div>',
            unsafe_allow_html=True,
        )
        for asset in assets:
            row_cols = st.columns([2] + [1] * 4)
            row_cols[0].markdown(
                f'<div style="font-size:13px;padding:6px 0;">'
                f'{SVC_ASSET_LABELS.get(asset, asset)}</div>',
                unsafe_allow_html=True,
            )
            for i, scenario_key in enumerate(SCENARIO_KEYS):
                vals = medium.get(scenario_key, {}).get(asset, [])
                cell_val = vals[_horizon_index] if _horizon_index < len(vals) else None
                is_fallback = False
                if cell_val is None:
                    near_vals = expanded.get("near_term", {}).get(scenario_key, {}).get(asset, [])
                    if near_vals:
                        cell_val = near_vals[0]
                        is_fallback = True
                if cell_val is None:
                    row_cols[i+1].markdown(
                        f'<div style="font-size:13px;color:{COLORS["text_dim"]};padding:6px 0;">—</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if cell_val > 2:
                        color = COLORS.get("green", "#22c55e")
                    elif cell_val > 0:
                        color = "#86efac"
                    elif cell_val > -2:
                        color = "#fca5a5"
                    else:
                        color = COLORS.get("red", "#ef4444")
                    prefix = "~" if is_fallback else ""
                    row_cols[i+1].markdown(
                        f'<div style="font-size:13px;font-weight:600;color:{color};padding:6px 0;">'
                        f'{prefix}{cell_val:+.1f}%</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    _render_fed_fan_charts(expanded.get("medium_term", {}), adj_probs, full_expanded, use_claude=_use_claude, engine=_selected_tier)


def _render_fed_causal_chain(chains: dict, adj_probs: list[dict], medium: dict, expanded: dict, use_claude: bool = False, engine: str = ""):
    """Section: full causal chain with cumulative confidence decay."""
    from services.fed_forecaster import SCENARIO_KEYS, SCENARIO_LABELS

    _badge = engine or ("🧠 Regard Mode" if use_claude else "⚡ Standard")
    _section_header("Causal Chain — Policy Transmission Path", badge=_badge)
    st.caption("How each Fed scenario propagates through the economy — confidence decays with each link")

    dominant_key = max(SCENARIO_KEYS, key=lambda k: next(
        (r["prob"] for r in adj_probs if r["scenario"] == k), 0.0
    ))
    ordered_keys = [dominant_key] + [k for k in SCENARIO_KEYS if k != dominant_key]

    for scenario_key in ordered_keys:
        chain_steps = chains.get(scenario_key, [])
        prob = next((r["prob"] for r in adj_probs if r["scenario"] == scenario_key), 0.25)
        label = f"{SCENARIO_LABELS[scenario_key]} [{int(round(prob*100))}%]"
        is_dominant = scenario_key == dominant_key

        with st.expander(("⭐ " if is_dominant else "") + label, expanded=is_dominant):
            if not chain_steps:
                st.caption("Chain data unavailable.")
                continue

            start_conf = 95
            end_conf = max(50, start_conf - (len(chain_steps) - 1) * 5)
            st.caption(f"{len(chain_steps)} steps · confidence {start_conf}% → {end_conf}%")

            for idx, step_text in enumerate(chain_steps):
                conf_pct = max(50, 95 - idx * 5)
                color = (
                    COLORS.get("green",    "#40c080") if conf_pct >= 70 else
                    COLORS.get("yellow",   "#f0c040") if conf_pct >= 55 else
                    COLORS.get("text_dim", "#888888")
                )
                arrow = "→ " if idx > 0 else "● "
                st.markdown(
                    f'<div style="padding:4px 0 4px 16px;border-left:3px solid {color};margin-bottom:2px;">'
                    f'<span style="font-size:11px;color:{COLORS.get("text_dim", "#888")};font-weight:600;">'
                    f'Step {idx+1}</span>&nbsp;&nbsp;'
                    f'<span style="font-size:13px;">{arrow}{step_text}</span>'
                    f'<span style="float:right;font-size:11px;color:{color};font-weight:600;">'
                    f'{conf_pct}%</span></div>',
                    unsafe_allow_html=True,
                )


def _render_fed_fan_charts(medium: dict, adj_probs: list[dict], expanded: dict, use_claude: bool = False, engine: str = ""):
    """Section 6: probability-weighted medium-term fan charts in tabbed layout."""
    from services.fed_forecaster import (
        SCENARIO_KEYS, SCENARIO_LABELS, ASSET_LABELS as SVC_ASSET_LABELS,
    )

    _badge = engine or ("🧠 Regard Mode" if use_claude else "⚡ Standard")
    _section_header("Medium-Term Outlook (3–12 months)", badge=_badge)

    prob_map = {r["scenario"]: r["prob"] for r in adj_probs}

    st.caption("Market-implied forecast — weighted by Fed Funds Futures probabilities across all FOMC scenarios.  "
               "🟢 Green area = positive return expected  ·  🔴 Red area = negative return expected  ·  "
               "y-axis = cumulative % change")

    _ALL_MEDIUM_ASSETS = ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short",
                          "oil", "natgas", "gold", "silver", "china", "india", "japan", "europe", "usd"]
    _total_w = sum(prob_map.get(sk, 0.25) for sk in SCENARIO_KEYS)
    _asset_returns = {}
    for _asset in _ALL_MEDIUM_ASSETS:
        _w_vals = []
        for _m in range(12):
            _wv = sum(
                prob_map.get(sk, 0.25) * (
                    medium.get(sk, {}).get(_asset, [])[_m]
                    if _m < len(medium.get(sk, {}).get(_asset, []))
                    else 0.0
                )
                for sk in SCENARIO_KEYS
            )
            _w_vals.append(_wv / _total_w if _total_w > 0 else 0.0)
        _final = _w_vals[-1] if _w_vals else 0.0
        if _final != 0.0:
            _asset_returns[_asset] = _final

    if _asset_returns:
        _sorted = sorted(_asset_returns.items(), key=lambda x: x[1], reverse=True)
        _gainers = [(k, v) for k, v in _sorted if v > 0][:3]
        _losers = [(k, v) for k, v in _sorted if v < 0][-3:]
        _losers.reverse()

        _gain_str = "  ".join(
            f'<span style="color:#22c55e;font-weight:600;">{SVC_ASSET_LABELS.get(k, k)} {v:+.1f}%</span>'
            for k, v in _gainers
        ) if _gainers else '<span style="color:#888;">none</span>'
        _loss_str = "  ".join(
            f'<span style="color:#ef4444;font-weight:600;">{SVC_ASSET_LABELS.get(k, k)} {v:+.1f}%</span>'
            for k, v in _losers
        ) if _losers else '<span style="color:#888;">none</span>'

        st.markdown(
            f'<div style="background:{COLORS.get("surface", "#1e293b")};border:1px solid {COLORS.get("border", "#334155")};'
            f'border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:13px;">'
            f'<b>12-Month Outlook:</b>&nbsp;&nbsp;'
            f'▲ {_gain_str}&nbsp;&nbsp;&nbsp;▼ {_loss_str}'
            f'</div>',
            unsafe_allow_html=True,
        )

    SCENARIO_COLORS = {
        "hold":    COLORS.get("yellow", "#f0c040"),
        "cut_25":  "#22c55e",
        "cut_50":  "#16a34a",
        "hike_25": COLORS.get("red", "#ef4444"),
    }

    def _draw_fan_chart(asset_key: str, col_or_container=None):
        months = list(range(1, 13))
        target = col_or_container or st
        total_w = sum(prob_map.get(sk, 0.25) for sk in SCENARIO_KEYS)
        weighted = []
        for m in range(12):
            w_val = sum(
                prob_map.get(sk, 0.25) * (
                    medium.get(sk, {}).get(asset_key, [])[m]
                    if m < len(medium.get(sk, {}).get(asset_key, []))
                    else 0.0
                )
                for sk in SCENARIO_KEYS
            )
            weighted.append(w_val / total_w if total_w > 0 else 0.0)

        if all(v == 0.0 for v in weighted):
            target.caption(f"_{SVC_ASSET_LABELS.get(asset_key, asset_key)}: forecast unavailable_")
            return

        pos_y = [v if v >= 0 else 0.0 for v in weighted]
        neg_y = [v if v < 0 else 0.0 for v in weighted]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=pos_y,
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.25)",
            line=dict(color="rgba(34,197,94,0.9)", width=2),
            name="Positive",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=months, y=neg_y,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.25)",
            line=dict(color="rgba(239,68,68,0.9)", width=2),
            name="Negative",
            showlegend=False,
        ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color=COLORS.get("border", "#444"), line_width=1)
        apply_dark_layout(fig)
        fig.update_layout(
            title=dict(text=SVC_ASSET_LABELS.get(asset_key, asset_key), font_size=13),
            height=220,
            margin=dict(l=0, r=10, t=30, b=20),
            xaxis=dict(title="Month", tickmode="linear", dtick=3),
            yaxis=dict(title="% cum."),
        )
        target.plotly_chart(fig, use_container_width=True)

    def _draw_near_term_bar(asset_key: str, col_or_container=None):
        near = expanded.get("near_term", {})
        days = [f"D{i+1}" for i in range(7)]
        target = col_or_container or st

        fig = go.Figure()
        for sk in SCENARIO_KEYS:
            vals = near.get(sk, {}).get(asset_key, [])
            if not vals or len(vals) < 7:
                continue
            prob = prob_map.get(sk, 0.25)
            label = f"{SCENARIO_LABELS[sk]} ({int(round(prob*100))}%)"
            color = SCENARIO_COLORS.get(sk, "#888")
            fig.add_trace(go.Bar(
                x=days,
                y=list(vals[:7]),
                name=label,
                marker_color=color,
                opacity=max(0.3, prob),
                showlegend=True,
            ))

        apply_dark_layout(fig)
        fig.update_layout(
            title=dict(text=SVC_ASSET_LABELS.get(asset_key, asset_key), font_size=13),
            height=220,
            margin=dict(l=0, r=10, t=30, b=20),
            barmode="group",
            xaxis_title="Day",
            yaxis_title="% change",
            legend=dict(font_size=9, orientation="h", y=-0.25),
        )
        target.plotly_chart(fig, use_container_width=True)

    if not medium:
        st.info("Medium-term forecast unavailable.")
        return

    tabs = st.tabs(["🇺🇸 US Equities", "🏦 Bonds", "🛢 Commodities", "🌏 International", "💵 Dollar"])

    with tabs[0]:
        cols = st.columns(2)
        for i, asset in enumerate(["spy", "qqq", "iwm", "dji"]):
            _draw_fan_chart(asset, cols[i % 2])

    with tabs[1]:
        cols = st.columns(2)
        for i, asset in enumerate(["bonds_long", "bonds_short"]):
            _draw_fan_chart(asset, cols[i])

    with tabs[2]:
        cols = st.columns(2)
        for i, asset in enumerate(["oil", "natgas", "gold", "silver"]):
            _draw_fan_chart(asset, cols[i % 2])

    with tabs[3]:
        cols = st.columns(2)
        for i, asset in enumerate(["china", "india", "japan", "europe"]):
            _draw_fan_chart(asset, cols[i % 2])

    with tabs[4]:
        _draw_fan_chart("usd")

    st.markdown("---")
    _render_fed_causal_chain(
        expanded.get("causal_chains", {}), adj_probs,
        expanded.get("medium_term", {}), expanded,
        use_claude=use_claude, engine=engine,
    )


def run_quick_fed(macro: dict, fred_data: dict, use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Computes Fed rate path probabilities and sector plays.
    Stores _dominant_rate_path, _rate_path_probs, _fed_funds_rate, _fed_plays_result to session_state.
    """
    import streamlit as st
    import datetime as _dt
    from services.fed_forecaster import (
        fetch_zq_probabilities, adjust_probabilities,
        calibrate_probabilities, build_fed_context, _neutral_tone_fallback,
    )
    from services.claude_client import suggest_regime_plays

    macro = macro or {}
    fred_data = fred_data or {}

    base_probs = fetch_zq_probabilities()
    tone_result = _neutral_tone_fallback()  # neutral tone — no Groq call for speed
    adj_probs = adjust_probabilities(base_probs, tone_result, macro=macro)

    try:
        fed_ctx = build_fed_context(macro, fred_data)
        _calib = calibrate_probabilities(base_probs, adj_probs, fed_ctx)
        _post_map = {p["scenario"]: p["posterior"] for p in _calib["posteriors"]}
        final_probs = [
            {**r, "prob": _post_map.get(r["scenario"], r["prob"])}
            for r in adj_probs
        ]
    except Exception:
        final_probs = adj_probs

    if not final_probs:
        return False

    dominant = max(final_probs, key=lambda r: r.get("prob", 0))
    _dominant_rp = {
        "scenario": dominant.get("scenario", "hold"),
        "prob_pct": round(dominant.get("prob", 0) * 100, 1),
    }

    _fed_funds_rate = None
    ff_series = fred_data.get("fedfunds")
    if ff_series is not None and not ff_series.empty:
        _fed_funds_rate = float(ff_series.dropna().iloc[-1])

    # Fed Rate-Path Plays using dominant scenario as context
    _labels = {"cut_25": "25bp cut", "cut_50": "50bp cut", "hold": "Hold", "hike_25": "25bp hike"}
    _scenario_label = _labels.get(dominant.get("scenario", "hold"), "Hold")
    _prob_pct = round(dominant.get("prob", 0) * 100)
    regime = macro.get("macro_regime", "Neutral")
    norm_score = (macro.get("macro_score", 50) - 50) / 50
    sig = (
        f"Dominant rate path: {_scenario_label} ({_prob_pct}% probability)\n"
        f"Regime: {regime} | Quadrant: {macro.get('quadrant', 'Unknown')}"
    )
    _plays = suggest_regime_plays(regime, norm_score, sig, use_claude=use_claude, model=model)
    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") else ("🧠 Regard Mode" if use_claude else "⚡ Freeloader Mode")
    result = {
        "_rate_path_probs": final_probs,
        "_rate_path_probs_ts": _dt.datetime.now(),
        "_dominant_rate_path": _dominant_rp,
        "_fed_plays_result": _plays,
        "_fed_plays_result_ts": _dt.datetime.now(),
        "_fed_plays_engine": _tier,
    }
    if _fed_funds_rate is not None:
        result["_fed_funds_rate"] = _fed_funds_rate
    return result


def run_quick_chain(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Generates policy transmission narration from stored _rate_path_probs.
    Stores _chain_narration + _chain_narration_engine to session_state.
    """
    import json as _json
    from services.claude_client import narrate_policy_transmission

    probs = st.session_state.get("_rate_path_probs", [])
    if not probs:
        return False

    probs_json = _json.dumps([{"scenario": r.get("scenario"), "prob": r.get("prob")} for r in probs])

    _rc = st.session_state.get("_regime_context") or {}
    chains = {
        "hold": ["Fed holds → short rates stable → mortgage market flat → credit spreads contained"],
        "cut_25": ["Fed cuts 25bp → front-end rally → credit spreads tighten → housing activity improves"],
        "cut_50": ["Fed cuts 50bp → equities surge → USD weakens → EM bid → commodities lift"],
        "hike_25": ["Fed hikes → front-end selloff → growth/tech pressure → USD strengthens"],
    }
    if _rc.get("quadrant"):
        chains["_regime"] = [f"Current quadrant: {_rc['quadrant']} — {_rc.get('regime', '')}"]

    chains_json = _json.dumps(chains)
    narration = narrate_policy_transmission(chains_json, probs_json, use_claude=use_claude, model=model)
    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") \
        else ("🧠 Regard Mode" if use_claude else "⚡ Freeloader Mode")
    st.session_state["_chain_narration"] = narration
    st.session_state["_chain_narration_engine"] = _tier
    return True


def run_quick_swans(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Auto-generates 3 regime-relevant black swan scenarios.
    Skips if _custom_swans already populated and fresh (< 24h).
    Stores _custom_swans + _custom_swans_ts to session_state.
    """
    import json as _json
    import datetime as _dt

    _existing = st.session_state.get("_custom_swans", {})
    _ts = st.session_state.get("_custom_swans_ts")
    if _existing and _ts:
        _age_h = (_dt.datetime.now() - _ts).total_seconds() / 3600
        if _age_h < 24:
            return True  # Already fresh, skip

    _rc = st.session_state.get("_regime_context") or {}
    _dp = st.session_state.get("_dominant_rate_path") or {}
    _base_ctx = _json.dumps({
        "regime": _rc.get("regime", ""),
        "quadrant": _rc.get("quadrant", ""),
        "score": _rc.get("score", 0),
        "dominant_rate_path": _dp.get("scenario", ""),
    })

    quadrant = _rc.get("quadrant", "")
    if quadrant in ("Stagflation", "Overheating"):
        scenarios = ["Inflation Spiral", "Fed Policy Error", "Dollar Collapse"]
    elif quadrant in ("Deflation", "Recession"):
        scenarios = ["Credit Market Freeze", "US Recession Deepens", "Fed Emergency Rate Cut"]
    else:
        scenarios = ["Geopolitical Black Swan", "US Credit Downgrade", "Oil Supply Shock"]

    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") \
        else ("🧠 Regard Mode" if use_claude else "⚡ Freeloader Mode")
    _new_swans = {}
    for label in scenarios:
        try:
            if use_claude:
                result = _call_claude_custom_event_forecast(label, _base_ctx, model or "grok-4-1-fast-reasoning")
            else:
                result = _call_groq_custom_event_forecast(label, _base_ctx)
            if result:
                result["_engine_tier"] = _tier
                _new_swans[label] = result
        except Exception:
            pass

    if _new_swans:
        return {
            "_custom_swans": _new_swans,
            "_custom_swans_ts": _dt.datetime.now(),
        }
    return None


# ── Public entry point ────────────────────────────────────────────────────────

def render():
    """Entry point for the Fed Forecaster sidebar module."""
    _oc = COLORS["bloomberg_orange"]
    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">FED FORECASTER</div>',
        unsafe_allow_html=True,
    )
    st.caption("Fed policy probability engine · Bayesian calibration · Asset impact matrix")

    _FRED_SERIES_IDS = [
        "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
        "PCEPILFE", "PNFI", "THREEFYTP10",
        "INDPRO", "NFCI", "DGS10", "ICSA", "USSLIND",
        "UMCSENT", "PERMIT", "FEDFUNDS",
    ]
    fred_ids = {
        "yield_curve":   "T10Y2Y",
        "credit_spread": "BAMLH0A0HYM2",
        "m2":            "M2SL",
        "sahm":          "SAHMREALTIME",
        "unrate":        "UNRATE",
        "core_pce":      "PCEPILFE",
        "capex":         "PNFI",
        "icsa":          "ICSA",
        "lei":           "USSLIND",
        "term_premium":  "THREEFYTP10",
        "ism":           "INDPRO",
        "fci":           "NFCI",
        "dgs10":         "DGS10",
        "umcsent":       "UMCSENT",
        "permit":        "PERMIT",
        "fedfunds":      "FEDFUNDS",
    }

    _loaded_key = "_fed_forecaster_loaded"

    _hdr_col, _btn_col = st.columns([5, 1])
    with _btn_col:
        if st.button("🔄 Refresh", key="fed_refresh_top", help="Clear cache and reload all FRED + Fed data"):
            st.cache_data.clear()
            st.session_state.pop(_loaded_key, None)
            st.rerun()

    if _loaded_key not in st.session_state:
        st.markdown(
            f'<div style="background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;'
            f'padding:32px 24px;text-align:center;margin:24px 0;">'
            f'<div style="font-size:14px;color:#C8D8E8;margin-bottom:8px;font-weight:600;">'
            f'Fed Forecaster — Ready to Load</div>'
            f'<div style="font-size:12px;color:#64748b;margin-bottom:16px;">'
            f'Fetches FRED macro data, Fed communications, and CME futures pricing.<br>'
            f'First load ~10s · Cached 1–4h thereafter.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⚡ Load Fed Forecaster", key="fed_load_btn", type="primary"):
            st.session_state[_loaded_key] = True
            st.rerun()
        return

    with st.spinner("Loading macro data…"):
        warm_fred_cache(_FRED_SERIES_IDS)
        fred_data = {k: fetch_fred_series_safe(v) for k, v in fred_ids.items()}

    # Use regime context from session_state if Risk Regime already ran,
    # otherwise build a minimal version from FRED data
    macro = st.session_state.get("_regime_context") or {}
    if not macro:
        _pce   = fred_data.get("core_pce")
        _unrate = fred_data.get("unrate")
        macro = {
            "macro_regime": "Unknown — run Risk Regime for full context",
            "quadrant":     "Unknown",
            "macro_score":  50,
            "core_pce":     float(_pce.dropna().iloc[-1]) if _pce is not None and not _pce.empty else 0.0,
            "unemployment": float(_unrate.dropna().iloc[-1]) if _unrate is not None and not _unrate.empty else 0.0,
            "vix_z":        0.0,
            "credit_z":     0.0,
        }

    _render_fed_forecaster(macro, fred_data)
