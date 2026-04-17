"""Tail Risk Studio — standalone black swan scenario analysis module."""

import json

import streamlit as st

from utils.theme import COLORS, apply_dark_layout
from utils.ai_tier import render_ai_tier_selector, MODEL_HINT_HTML

# Asset display labels (mirrors services/fed_forecaster.py ASSET_LABELS)
_ASSET_LABELS = {
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "bonds_long": "TLT",
    "bonds_short": "SHY",
    "gold": "GLD",
    "oil": "USO",
    "usd": "DXY",
}

_IMPACT_ORDER = ["strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"]

_IMPACT_COLOR = {
    "strongly bullish": "#00C853",
    "bullish": "#00D4AA",
    "neutral": "#64748b",
    "bearish": "#FF8C00",
    "strongly bearish": "#FF4B4B",
}

_IMPACT_SHORT = {
    "strongly bullish": "++",
    "bullish": "+",
    "neutral": "~",
    "bearish": "−",
    "strongly bearish": "−−",
}


def _section_header(title: str, badge: str = ""):
    badge_html = (
        f'<span style="font-size:10px;background:#1a2040;color:{COLORS["bloomberg_orange"]};'
        f'padding:2px 8px;border-radius:3px;margin-left:8px;font-weight:600;">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:11px;font-weight:700;letter-spacing:0.08em;color:{COLORS["bloomberg_orange"]};'
        f'text-transform:uppercase;margin:18px 0 8px 0;">'
        f'{title}{badge_html}</div>'
        f'<div style="height:1px;background:{COLORS["border"]};margin-bottom:14px;"></div>',
        unsafe_allow_html=True,
    )


def _build_context_json() -> str:
    """Assemble macro context from session state — no fed forecast required."""
    ctx: dict = {}

    rc = st.session_state.get("_regime_context", {})
    if rc:
        ctx["regime"] = rc.get("regime", "")
        ctx["quadrant"] = rc.get("quadrant", "")
        ctx["macro_score"] = rc.get("score", 0)
        if rc.get("signal_summary"):
            ctx["regime_ai_context"] = rc["signal_summary"]

    rate_probs = st.session_state.get("_rate_path_probs", [])
    if rate_probs:
        dp = max(rate_probs, key=lambda r: r.get("prob", 0))
        ctx["dominant_rate_path"] = dp.get("scenario", "")
        ctx["dominant_rate_prob_pct"] = round(dp.get("prob", 0) * 100, 1)
        ctx["rate_path_scenarios"] = [
            {"scenario": r.get("scenario"), "prob_pct": round(r.get("prob", 0) * 100, 1)}
            for r in sorted(rate_probs, key=lambda r: r.get("prob", 0), reverse=True)
        ]

    return json.dumps(ctx)


def _pill(asset_key: str, impact_val: str) -> str:
    v = str(impact_val).lower()
    if "strongly bullish" in v:
        c = _IMPACT_COLOR["strongly bullish"]
    elif "bullish" in v:
        c = _IMPACT_COLOR["bullish"]
    elif "strongly bearish" in v:
        c = _IMPACT_COLOR["strongly bearish"]
    elif "bearish" in v:
        c = _IMPACT_COLOR["bearish"]
    else:
        c = _IMPACT_COLOR["neutral"]
    label = _ASSET_LABELS.get(asset_key, asset_key)
    short = _IMPACT_SHORT.get(v.strip(), v[:2])
    return (
        f'<span style="background:{COLORS["surface"]};border:1px solid {c};color:{c};'
        f'padding:2px 7px;border-radius:4px;font-size:0.73em;margin:2px;display:inline-block;">'
        f'{label} {short}</span>'
    )


def _render_event_card(label: str, data: dict, badge: str = "", col=None):
    prob = data.get("probability_pct", 0)
    narrative = data.get("narrative", "")
    impacts = data.get("asset_impacts", {})

    if prob > 10:
        prob_color, sev_icon = COLORS["red"], "🔴"
    elif prob > 3:
        prob_color, sev_icon = "#f59e0b", "🟡"
    else:
        prob_color, sev_icon = COLORS["green"], "🟢"

    pills = "".join(_pill(k, v) for k, v in impacts.items())

    badge_html = (
        f'<span style="font-size:10px;background:#2A3565;color:#8899CC;'
        f'padding:1px 6px;border-radius:3px;margin-left:6px;">{badge}</span>'
        if badge else ""
    )

    html = (
        f'<div style="border:1px solid {COLORS["border"]};border-radius:8px;'
        f'padding:14px;margin-bottom:10px;background:{COLORS["surface"]};">'
        f'<div style="font-weight:700;font-size:14px;margin-bottom:6px;">'
        f'{sev_icon} {label}{badge_html}</div>'
        f'<span style="background:{prob_color};color:white;padding:2px 10px;'
        f'border-radius:10px;font-size:0.8em;font-weight:600;">'
        f'{prob:.1f}% annual prob</span>'
        f'<p style="margin:10px 0 8px 0;font-size:0.85em;color:{COLORS["text_dim"]};">'
        f'{narrative}</p>'
        f'<div style="margin-top:6px;">{pills}</div>'
        f'</div>'
    )
    target = col if col is not None else st
    target.markdown(html, unsafe_allow_html=True)


def _render_impact_heatmap(all_events: dict):
    """Render a compact heatmap table: events as rows, assets as columns."""
    if not all_events:
        return

    _section_header("Impact Heatmap — All Scenarios")

    assets = list(_ASSET_LABELS.keys())
    headers = ["Event"] + [_ASSET_LABELS[a] for a in assets]

    header_row = "".join(
        f'<th style="padding:6px 10px;text-align:center;font-size:11px;'
        f'color:{COLORS["bloomberg_orange"]};letter-spacing:0.05em;">{h}</th>'
        for h in headers
    )

    rows_html = ""
    for event_label, data in all_events.items():
        impacts = data.get("asset_impacts", {})
        cells = f'<td style="padding:6px 10px;font-size:12px;white-space:nowrap;' \
                f'color:{COLORS["text"]};">{event_label[:28]}</td>'
        for a in assets:
            v = str(impacts.get(a, "")).lower().strip()
            c = _IMPACT_COLOR.get(v, COLORS["text_dim"])
            short = _IMPACT_SHORT.get(v, "?")
            cells += (
                f'<td style="padding:6px 10px;text-align:center;font-weight:700;'
                f'font-size:13px;color:{c};">{short}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    legend = " &nbsp;".join(
        f'<span style="color:{c};font-weight:700;">{s}</span> {lbl}'
        for lbl, s, c in [
            ("Strong Bull", "++", _IMPACT_COLOR["strongly bullish"]),
            ("Bull", "+", _IMPACT_COLOR["bullish"]),
            ("Neutral", "~", _IMPACT_COLOR["neutral"]),
            ("Bear", "−", _IMPACT_COLOR["bearish"]),
            ("Strong Bear", "−−", _IMPACT_COLOR["strongly bearish"]),
        ]
    )

    st.markdown(
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;'
        f'background:{COLORS["surface"]};border-radius:8px;">'
        f'<thead><tr style="border-bottom:1px solid {COLORS["border"]};">{header_row}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};margin-top:6px;">{legend}</div>',
        unsafe_allow_html=True,
    )


def render():
    _ai_ctx = st.session_state.get("_tail_risk_ai_context") or {}
    if _ai_ctx:
        _ctx_lines = "\n".join(f"  {k}: {v}" for k, v in _ai_ctx.items())
        st.markdown(
            f'<div id="rt-tail-risk-ai-context" style="display:none;position:absolute;width:0;height:0;overflow:hidden;">'
            f'<pre>{_ctx_lines}</pre></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:20px;font-weight:700;letter-spacing:0.06em;margin-bottom:2px;">'
        f'<span style="color:{COLORS["bloomberg_orange"]}">◈</span> '
        f'TAIL RISK STUDIO</div>'
        f'<div style="font-size:12px;color:{COLORS["text_dim"]};margin-bottom:18px;">'
        f'Black swan scenario analysis — runs standalone, no fed forecast required</div>',
        unsafe_allow_html=True,
    )

    # ── Macro context strip ───────────────────────────────────────────────────
    rc = st.session_state.get("_regime_context", {})
    rate_probs = st.session_state.get("_rate_path_probs", [])
    _has_regime = bool(rc.get("regime"))
    _has_rates = bool(rate_probs)

    if _has_regime or _has_rates:
        _ctx_parts = []
        if _has_regime:
            _ctx_parts.append(
                f'<span style="color:{COLORS["text_dim"]};">Regime:</span> '
                f'<span style="color:{COLORS["accent"]};font-weight:600;">'
                f'{rc.get("regime","")} / {rc.get("quadrant","")}</span>'
            )
        if _has_rates:
            dp = max(rate_probs, key=lambda r: r.get("prob", 0))
            _ctx_parts.append(
                f'<span style="color:{COLORS["text_dim"]};">Dominant rate path:</span> '
                f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:600;">'
                f'{dp.get("scenario","").replace("_"," ").title()} '
                f'({round(dp.get("prob",0)*100,1)}%)</span>'
            )
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;padding:8px 14px;font-size:12px;margin-bottom:16px;">'
            + " &nbsp;·&nbsp; ".join(_ctx_parts) +
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("💡 Run **Quick Intel Run** first to inject macro context — analysis will be more accurate.")

    # ── AI Engine ─────────────────────────────────────────────────────────────
    use_claude, model = render_ai_tier_selector(
        key="tail_risk_engine",
        label="Analysis Engine",
        recommendation="🧠 Regard Mode recommended — tail risk reasoning benefits from Grok 4.1",
        default=1,
    )
    _tier_label = st.session_state.get("tail_risk_engine", "🧠 Regard Mode")

    ctx_json = _build_context_json()

    # ── Section 1: Standard Black Swan Events ────────────────────────────────
    _section_header("Standard Tail Risk Events")

    _stored_swans = st.session_state.get("_trs_standard_swans", {})
    _stored_tier = st.session_state.get("_trs_standard_tier", "")

    _tier_changed = _stored_tier and _stored_tier != _tier_label
    if _tier_changed:
        st.caption(f"⚠ Engine changed from {_stored_tier} → {_tier_label}. Regenerate to update.")

    _gen_col, _info_col = st.columns([2, 5])
    if _gen_col.button(
        "⚡ Analyze All Events" if not _stored_swans else "🔄 Re-analyze",
        key="trs_gen_std_btn",
        type="primary" if not _stored_swans else "secondary",
    ):
        from services.fed_forecaster import (
            _call_groq_black_swan_forecast,
            _call_claude_black_swan_forecast,
        )
        with st.spinner("Analyzing tail risk events…"):
            try:
                if use_claude and model:
                    swans = _call_claude_black_swan_forecast(ctx_json, model=model)
                else:
                    swans = _call_groq_black_swan_forecast(ctx_json)
                from services.fed_forecaster import BLACK_SWAN_EVENTS as _BSE
                st.session_state["_trs_standard_swans"] = {
                    _BSE[k]: swans[k] for k in _BSE if k in swans
                }
                st.session_state["_trs_standard_tier"] = _tier_label
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if _stored_swans:
        from services.fed_forecaster import BLACK_SWAN_EVENTS as _BSE

        avg_prob = sum(
            d.get("probability_pct", 0) for d in _stored_swans.values()
        ) / max(len(_stored_swans), 1)

        if avg_prob > 8:
            agg_color, agg_label = COLORS["red"], "ELEVATED"
        elif avg_prob > 3:
            agg_color, agg_label = "#f59e0b", "MODERATE"
        else:
            agg_color, agg_label = COLORS["green"], "LOW"

        _info_col.markdown(
            f'<div style="padding:6px 0;font-size:13px;">'
            f'Aggregate tail risk: '
            f'<span style="color:{agg_color};font-weight:700;">{agg_label}</span>'
            f' &nbsp;·&nbsp; avg {avg_prob:.1f}% annual probability'
            f' &nbsp;·&nbsp; '
            f'<span style="font-size:11px;color:{COLORS["text_dim"]};">{_stored_tier}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(2)
        for i, (label, data) in enumerate(_stored_swans.items()):
            _render_event_card(label, data, col=cols[i % 2])

    # ── Section 2: Custom Event Analysis ─────────────────────────────────────
    _section_header("Custom Scenario Analysis")
    st.caption("Type any tail-risk event — AI estimates probability and directional asset impact")

    with st.form("trs_custom_form", clear_on_submit=True):
        _custom_label = st.text_input(
            "Describe the scenario",
            placeholder="e.g. Reverse yen carry trade unwind, US debt ceiling breach, China invades Taiwan",
            label_visibility="collapsed",
        )
        _sub = st.form_submit_button("Analyze Scenario", type="primary")

    if _sub and _custom_label.strip():
        from services.fed_forecaster import (
            _call_groq_custom_event_forecast,
            _call_claude_custom_event_forecast,
        )
        with st.spinner(f"Analyzing: {_custom_label}…"):
            try:
                if use_claude and model:
                    result = _call_claude_custom_event_forecast(
                        _custom_label.strip(), ctx_json, model
                    )
                else:
                    result = _call_groq_custom_event_forecast(
                        _custom_label.strip(), ctx_json
                    )
                result["_engine_tier"] = _tier_label
                stored = st.session_state.get("_trs_custom_swans", {})
                stored[_custom_label.strip()] = result
                st.session_state["_trs_custom_swans"] = stored
                try:
                    from services.play_log import append_play as _ap
                    _ap("Tail Risk Studio", _tier_label,
                        {_custom_label.strip(): result},
                        meta={"event": _custom_label.strip()})
                except Exception:
                    pass
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    custom_swans = st.session_state.get("_trs_custom_swans", {})
    if custom_swans:
        ccols = st.columns(2)
        for i, (label, data) in enumerate(custom_swans.items()):
            _render_event_card(label, data, badge="CUSTOM", col=ccols[i % 2])

        if st.button("Clear Custom Scenarios", key="trs_clear_custom"):
            st.session_state["_trs_custom_swans"] = {}
            st.rerun()

    # ── Section 3: Combined Impact Heatmap ───────────────────────────────────
    all_events = {**st.session_state.get("_trs_standard_swans", {}),
                  **st.session_state.get("_trs_custom_swans", {})}
    if all_events:
        st.markdown("---")
        _render_impact_heatmap(all_events)
