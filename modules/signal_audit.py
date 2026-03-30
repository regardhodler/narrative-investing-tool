"""Signal Audit Dashboard — live view of all cross-module AI signals in session_state."""

import streamlit as st
from datetime import datetime
from utils.theme import COLORS


# ── Signal Registry ────────────────────────────────────────────────────────────
# Each entry defines one row in the audit matrix.
# key: session_state key for the data
# ts_key: session_state key for the timestamp (None if no ts)
# sub_key: if set, check value[sub_key] instead of value directly
# valuation/discovery/portfolio: whether this module reads the signal
# preview_fn: callable(value) -> str for the Data Preview column

def _prev_regime_base(v):
    return f"regime={v.get('regime','')} | score={v.get('score',0):+.2f} | quadrant={v.get('quadrant','')}"

def _prev_regime_summary(v):
    s = v.get("signal_summary", "") or ""
    return s[:120] + ("…" if len(s) > 120 else "")

def _prev_regime_plays(v):
    sectors = ", ".join(s.get("name", "") for s in v.get("sectors", [])[:3])
    stocks = ", ".join(s.get("ticker", "") for s in v.get("stocks", [])[:3])
    return f"Sectors: {sectors} | Stocks: {stocks}"

def _prev_fed_rate(v):
    return f"{v:.2f}%" if v is not None else ""

def _prev_dominant_rp(v):
    labels = {"cut_25": "25bp cut", "cut_50": "50bp cut", "hold": "Hold", "hike_25": "25bp hike"}
    sc = labels.get(v.get("scenario", ""), v.get("scenario", ""))
    return f"{sc} ({v.get('prob_pct', 0):.0f}%)"

def _prev_all_rp(v):
    labels = {"cut_25": "25bp cut", "cut_50": "50bp cut", "hold": "Hold", "hike_25": "25bp hike"}
    if not v:
        return ""
    top = sorted(v, key=lambda r: r.get("prob", 0), reverse=True)[:3]
    return " | ".join(f"{labels.get(r['scenario'], r['scenario'])} {round(r.get('prob',0)*100)}%" for r in top)

def _prev_fed_plays(v):
    sectors = ", ".join(s.get("name", "") for s in v.get("sectors", [])[:3])
    bonds = ", ".join(s.get("ticker", "") for s in v.get("bonds", [])[:2])
    return f"Sectors: {sectors} | Bonds: {bonds}"

def _prev_text(v):
    s = str(v)
    return s[:120] + ("…" if len(s) > 120 else "")

def _prev_swans(v):
    names = list(v.keys())[:4]
    return f"{len(v)} event(s): {', '.join(names)}"

def _prev_discovery(v):
    sectors = ", ".join(s.get("name", "") for s in v.get("sectors", [])[:3])
    stocks = ", ".join(s.get("ticker", "") for s in v.get("stocks", [])[:3])
    return f"Sectors: {sectors} | Stocks: {stocks}"

def _prev_macro_fit(v):
    return " | ".join(f"{tk}: {d.get('verdict','')} ({'★'*d.get('fit_stars',0)})" for tk, d in list(v.items())[:3])

def _prev_current_events(v):
    s = str(v)
    return s[:120] + ("…" if len(s) > 120 else "")


def _prev_macro_synopsis(v):
    if isinstance(v, dict):
        s = v.get("summary", "") or v.get("text", "") or str(v)
    else:
        s = str(v)
    return s[:120] + ("…" if len(s) > 120 else "")

def _prev_risk_snapshot(v):
    beta = v.get("portfolio_beta", "?")
    var = v.get("var_95_pct", "?")
    flags = len(v.get("risk_flags", []))
    return f"Beta: {beta} | VaR 95%: {var} | Risk flags: {flags}"

def _prev_tactical(v):
    if not isinstance(v, dict):
        return str(v)[:120]
    score = v.get("tactical_score", "?")
    label = v.get("label", "")
    bias  = v.get("action_bias", "")
    return f"Score: {score}/100 ({label}) — {bias}"


def _prev_forecast_log(v):
    if not isinstance(v, list):
        return ""
    total = len(v)
    resolved = [e for e in v if e.get("outcome") in ("correct", "incorrect")]
    correct = sum(1 for e in resolved if e.get("outcome") == "correct")
    acc = round(correct / len(resolved) * 100, 1) if resolved else 0.0
    return f"{total} logged · {len(resolved)} resolved · {acc}% accuracy"


_SIGNAL_REGISTRY = [
    {
        "label": "Regime label / score / quadrant",
        "key": "_regime_context",
        "ts_key": "_regime_context_ts",
        "sub_check": "regime",  # must have this sub-key to count as present
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_regime_base,
        "run_hint": "Risk Regime → Generate Rate-Path Plays",
    },
    {
        "label": "Regime signal_summary (17-signal)",
        "key": "_regime_context",
        "ts_key": "_regime_context_ts",
        "sub_check": "signal_summary",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_regime_summary,
        "run_hint": "Risk Regime → Generate Rate-Path Plays",
    },
    {
        "label": "AI Regime Plays",
        "key": "_rp_plays_result",
        "ts_key": None,
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_regime_plays,
        "run_hint": "Risk Regime → Generate Regime Plays",
    },
    {
        "label": "Tactical Regime",
        "key": "_tactical_context",
        "ts_key": "_tactical_context_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_tactical,
        "run_hint": "Risk Regime → Tactical Regime (Quick Intel Run)",
    },
    {
        "label": "Fed Funds Rate %",
        "key": "_fed_funds_rate",
        "ts_key": None,
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_fed_rate,
        "run_hint": "Risk Regime → Fed Forecaster (auto-populates on run)",
    },
    {
        "label": "Rate Path — dominant scenario",
        "key": "_dominant_rate_path",
        "ts_key": "_rate_path_probs_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_dominant_rp,
        "run_hint": "Risk Regime → Fed Forecaster",
    },
    {
        "label": "Rate Path — all scenarios",
        "key": "_rate_path_probs",
        "ts_key": "_rate_path_probs_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_all_rp,
        "run_hint": "Risk Regime → Fed Forecaster",
    },
    {
        "label": "Fed Rate-Path Plays",
        "key": "_fed_plays_result",
        "ts_key": "_fed_plays_result_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_fed_plays,
        "run_hint": "Risk Regime → Generate Rate-Path Plays",
    },
    {
        "label": "Policy Transmission narration",
        "key": "_chain_narration",
        "ts_key": "_chain_narration_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_text,
        "run_hint": "Risk Regime → Transmission Path",
    },
    {
        "label": "Black Swans",
        "key": "_custom_swans",
        "ts_key": "_custom_swans_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_swans,
        "run_hint": "Risk Regime → Black Swan events",
    },
    {
        "label": "Doom Briefing",
        "key": "_doom_briefing",
        "ts_key": "_doom_briefing_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_text,
        "run_hint": "Stress Signals → Generate Doom Briefing",
    },
    {
        "label": "Whale Activity Summary",
        "key": "_whale_summary",
        "ts_key": "_whale_summary_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_text,
        "run_hint": "Whale Movement → Generate Whale Summary",
    },
    {
        "label": "Cross-Signal Discovery Plays",
        "key": "_plays_result",
        "ts_key": None,
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_discovery,
        "run_hint": "Discovery → Generate Plays",
    },
    {
        "label": "Macro Fit (per-ticker dict)",
        "key": "_macro_fit_results",
        "ts_key": None,
        "valuation": False, "discovery": True, "portfolio": True,
        "preview_fn": _prev_macro_fit,
        "run_hint": "Discovery → Macro Fit → Assess a ticker",
    },
    {
        "label": "Current Events Digest",
        "key": "_current_events_digest",
        "ts_key": "_current_events_digest_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_current_events,
        "run_hint": "Current Events → Generate News Digest",
    },
    {
        "label": "Macro Synopsis (Quick Intel Round 4)",
        "key": "_macro_synopsis",
        "ts_key": "_macro_synopsis_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_macro_synopsis,
        "run_hint": "Quick Intel Run → Round 4: Macro Conviction Synopsis",
    },
    {
        "label": "Portfolio Risk Snapshot",
        "key": "_portfolio_risk_snapshot",
        "ts_key": "_portfolio_risk_snapshot_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_risk_snapshot,
        "run_hint": "My Regarded Portfolio → Risk Matrix tab (or Quick Intel Run Round 5)",
    },
    {
        "label": "Forecast Accuracy Log",
        "key": "_forecast_log",
        "ts_key": "_forecast_log_ts",
        "valuation": False, "discovery": False, "portfolio": True,
        "preview_fn": _prev_forecast_log,
        "run_hint": "Forecast Tracker → Log Forecast tab",
    },
    {
        "label": "Activism Analysis (13D digest)",
        "key": "_activism_digest",
        "ts_key": "_activism_digest_ts",
        "valuation": True, "discovery": True, "portfolio": True,
        "preview_fn": _prev_text,
        "run_hint": "Whale Movement → Activism tab → Generate Activism Analysis",
    },
]

_CRITICAL_KEYS = {"_regime_context", "_dominant_rate_path", "_rate_path_probs"}


# ── Status helpers ─────────────────────────────────────────────────────────────

def _is_present(row: dict) -> bool:
    """Check if the signal has usable data in session_state."""
    v = st.session_state.get(row["key"])
    if v is None:
        return False
    # For dicts/lists, check they're non-empty
    if isinstance(v, (dict, list)) and not v:
        return False
    # Check sub_key if specified
    sub = row.get("sub_check")
    if sub and isinstance(v, dict) and not v.get(sub):
        return False
    return True


def _age_info(row: dict, now: datetime) -> tuple:
    """Returns (age_str, status) where status in 'fresh'|'warn'|'stale'|'no_ts'|'missing'."""
    if not _is_present(row):
        return "—", "missing"
    ts_key = row.get("ts_key")
    if not ts_key:
        return "no ts", "no_ts"
    ts = st.session_state.get(ts_key)
    if ts is None:
        return "no ts", "no_ts"
    age_min = (now - ts).total_seconds() / 60
    if age_min < 120:
        return f"{int(age_min)}m ago", "fresh"
    elif age_min < 360:
        return f"{int(age_min // 60)}h ago", "warn"
    else:
        return f"{int(age_min // 60)}h ago", "stale"


def _get_preview(row: dict) -> str:
    v = st.session_state.get(row["key"])
    if not _is_present(row) or v is None:
        return ""
    try:
        return row["preview_fn"](v)
    except Exception:
        return str(v)[:80]


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _status_badge(age_str: str, status: str) -> str:
    cfg = {
        "fresh":   ("#22c55e", "#052e16", "✅"),
        "warn":    ("#f59e0b", "#1c1400", "⚠"),
        "stale":   ("#ef4444", "#1a0a0a", "✗"),
        "no_ts":   ("#22c55e", "#052e16", "✅"),
        "missing": ("#ef4444", "#1a0a0a", "✗"),
    }
    color, bg, icon = cfg.get(status, ("#888", "#111", "?"))
    label = "Missing" if status == "missing" else age_str
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color}33;'
        f'border-radius:3px;padding:1px 7px;font-size:11px;font-weight:600;'
        f'white-space:nowrap;">{icon} {label}</span>'
    )


def _wire_badge(wired: bool) -> str:
    if wired:
        return '<span style="color:#22c55e;font-size:13px;">✅</span>'
    return '<span style="color:#444;font-size:12px;">—</span>'


# ── Main render ────────────────────────────────────────────────────────────────

def render():
    _oc = COLORS["bloomberg_orange"]
    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">SIGNAL AUDIT DASHBOARD</div>',
        unsafe_allow_html=True,
    )
    st.caption("Live view of all cross-module AI signals — shows what's populated, how fresh, and which AI module reads it.")

    now = datetime.now()

    # ── Section A: Health metrics ──────────────────────────────────────────────
    total = len(_SIGNAL_REGISTRY)
    populated = sum(1 for r in _SIGNAL_REGISTRY if _is_present(r))
    stale_count = sum(
        1 for r in _SIGNAL_REGISTRY
        if _is_present(r) and _age_info(r, now)[1] in ("warn", "stale")
    )
    missing_count = total - populated
    critical_missing = [
        r for r in _SIGNAL_REGISTRY
        if r["key"] in _CRITICAL_KEYS and not _is_present(r)
    ]

    health_pct = populated / total
    if health_pct >= 0.75:
        _health_label, _health_color = "Healthy", "#22c55e"
    elif health_pct >= 0.5:
        _health_label, _health_color = "Partial", "#f59e0b"
    else:
        _health_label, _health_color = "Incomplete", "#ef4444"

    bar_filled = int(health_pct * 20)
    bar_str = "█" * bar_filled + "░" * (20 - bar_filled)

    st.markdown(
        f'<div style="background:#111;border:1px solid {COLORS["border"]};border-radius:6px;'
        f'padding:12px 18px;margin:8px 0 14px 0;display:flex;align-items:center;gap:24px;">'
        f'<span style="font-size:22px;font-weight:700;color:{_health_color};">'
        f'{populated}<span style="color:#888;font-size:14px;">/{total}</span></span>'
        f'<span>'
        f'<span style="color:#888;font-size:11px;font-family:monospace;">{bar_str}</span><br>'
        f'<span style="color:{_health_color};font-weight:700;font-size:12px;">{_health_label}</span>'
        f'<span style="color:#666;font-size:11px;"> — signals populated</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Signals", total)
    m2.metric("Populated", populated, delta=f"{populated - missing_count} vs missing")
    m3.metric("Stale (>2h)", stale_count)
    m4.metric("Missing", missing_count)

    # ── Section C: "What to run" guidance (shown before table for visibility) ──
    if critical_missing:
        _missing_hints = list({r["run_hint"] for r in critical_missing})
        _hint_html = "".join(f'<li style="color:#fca5a5;margin:2px 0;">{h}</li>' for h in _missing_hints)
        st.markdown(
            f'<div style="background:#1a0a0a;border:1px solid #ef444466;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:10px;">'
            f'<div style="color:#ef4444;font-weight:700;font-size:12px;margin-bottom:4px;">'
            f'⚠ Critical signals missing — run these first:</div>'
            f'<ul style="margin:0;padding-left:18px;font-size:12px;">{_hint_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif stale_count > 0:
        _stale_rows = [r for r in _SIGNAL_REGISTRY if _is_present(r) and _age_info(r, now)[1] in ("warn", "stale")]
        _stale_hints = list({r["run_hint"] for r in _stale_rows})
        _hint_html = "".join(f'<li style="color:#fcd34d;margin:2px 0;">{h}</li>' for h in _stale_hints[:4])
        st.markdown(
            f'<div style="background:#1a1200;border:1px solid #f59e0b66;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:10px;">'
            f'<div style="color:#f59e0b;font-weight:700;font-size:12px;margin-bottom:4px;">'
            f'⚠ {stale_count} signal(s) stale — consider refreshing:</div>'
            f'<ul style="margin:0;padding-left:18px;font-size:12px;">{_hint_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#052e16;border:1px solid #22c55e66;border-radius:6px;'
            'padding:8px 14px;margin-bottom:10px;color:#22c55e;font-size:12px;font-weight:600;">'
            '✅ All signals populated and fresh — AI modules are reading the full 360° macro picture.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:10px 0 14px 0;"></div>', unsafe_allow_html=True)

    # ── Section B: Signal Matrix Table ────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">SIGNAL MATRIX</div>',
        unsafe_allow_html=True,
    )

    # Table header
    header_style = (
        "background:#1a1a1a;color:#888;font-size:10px;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;padding:6px 10px;"
        "border-bottom:1px solid #333;"
    )
    row_style_base = (
        "border-bottom:1px solid #1e1e1e;padding:7px 10px;font-size:12px;"
        "vertical-align:middle;"
    )

    table_html = f"""
<table style="width:100%;border-collapse:collapse;font-family:monospace;">
<thead>
<tr>
  <th style="{header_style}width:22%">Signal</th>
  <th style="{header_style}width:14%">Source Key</th>
  <th style="{header_style}width:10%">Status / Age</th>
  <th style="{header_style}width:6%;text-align:center">Val</th>
  <th style="{header_style}width:6%;text-align:center">Disc</th>
  <th style="{header_style}width:6%;text-align:center">Port</th>
  <th style="{header_style}">Data Preview</th>
</tr>
</thead>
<tbody>
"""

    for i, row in enumerate(_SIGNAL_REGISTRY):
        age_str, status = _age_info(row, now)
        badge = _status_badge(age_str, status)
        preview = _get_preview(row)
        # Truncate preview for display
        if len(preview) > 100:
            preview = preview[:97] + "…"
        # Escape HTML chars in preview
        preview = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        row_bg = "#0d0d0d" if i % 2 == 0 else "#111"
        crit_border = "border-left:3px solid #ef4444;" if (row["key"] in _CRITICAL_KEYS and status == "missing") else "border-left:3px solid transparent;"
        table_html += f"""
<tr style="background:{row_bg};{crit_border}">
  <td style="{row_style_base}color:#e2e8f0;font-weight:600;">{row['label']}</td>
  <td style="{row_style_base}color:#64748b;font-size:10px;font-family:monospace;">{row['key']}</td>
  <td style="{row_style_base}">{badge}</td>
  <td style="{row_style_base}text-align:center">{_wire_badge(row['valuation'])}</td>
  <td style="{row_style_base}text-align:center">{_wire_badge(row['discovery'])}</td>
  <td style="{row_style_base}text-align:center">{_wire_badge(row['portfolio'])}</td>
  <td style="{row_style_base}color:#666;font-size:11px;max-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{preview}</td>
</tr>
"""

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:10px;color:#555;margin-top:4px;">'
        'Val = Valuation · Disc = Discovery · Port = Portfolio · ✅ = wired · — = not applicable'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Section D: Raw Session State Inspector ────────────────────────────────
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
    with st.expander("🔍 Raw Session State Inspector", expanded=False):
        _signal_keys = [r["key"] for r in _SIGNAL_REGISTRY] + [r["ts_key"] for r in _SIGNAL_REGISTRY if r.get("ts_key")]
        _all_ss_keys = sorted({k for k in st.session_state if k.startswith("_")})
        rows = []
        for k in _all_ss_keys:
            v = st.session_state[k]
            t = type(v).__name__
            if isinstance(v, dict):
                size = f"dict({len(v)} keys)"
                preview_raw = str(list(v.keys())[:4])
            elif isinstance(v, list):
                size = f"list({len(v)} items)"
                preview_raw = str(v[:3])
            elif isinstance(v, str):
                size = f"str({len(v)} chars)"
                preview_raw = v[:80]
            elif isinstance(v, datetime):
                size = "datetime"
                preview_raw = v.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(v, (int, float)):
                size = t
                preview_raw = str(v)
            else:
                size = t
                preview_raw = str(v)[:80]
            is_signal = "✅" if k in _signal_keys else ""
            rows.append({"Key": k, "Type": size, "Signal?": is_signal, "Preview": preview_raw[:100]})
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No underscore-prefixed session state keys found. Run some AI modules first.")

    # Auto-refresh note
    st.caption("⟳ This page reflects the current session state. Navigate away and back to refresh, or use the Streamlit rerun button.")
