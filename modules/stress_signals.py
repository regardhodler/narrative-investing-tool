"""
Module 9: Stress Signals / Doomsday Monitor

The canary-in-the-coal-mine dashboard — tracking credit stress, institutional
exits, distress filings, and systemic risk indicators.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.theme import COLORS, apply_dark_layout

# Doom-themed color overrides
DOOM_RED = "#FF2222"
DOOM_RED_DIM = "#991111"
DOOM_BG = "#1A0A0A"
DOOM_SURFACE = "#2A1010"
DOOM_BORDER = "#661111"
DOOM_YELLOW = "#FF8800"


def _doom_container(content_html: str, border_color: str = DOOM_BORDER) -> str:
    """Wrap content in a doom-themed container."""
    return f"""
    <div style="background:{DOOM_SURFACE};border:1px solid {border_color};
                border-radius:8px;padding:16px 20px;margin-bottom:16px;">
        {content_html}
    </div>
    """


def _compute_stress_score(fred_data: dict, canary_df: pd.DataFrame) -> tuple[float, dict]:
    """Compute a 0-100 stress score from available data.

    Returns (score, components_dict).
    """
    components = {}
    scores = []
    weights = []

    # 1. HY OAS level (higher = more stress)
    hy_df = fred_data.get("BAMLH0A0HYM2", pd.DataFrame())
    if not hy_df.empty and "value" in hy_df.columns:
        hy_val = float(hy_df["value"].iloc[-1])
        # HY OAS: <300bps = low stress, 300-500 = moderate, 500-800 = high, >800 = panic
        hy_score = min(100, max(0, (hy_val - 200) / 8 * 100))
        components["HY OAS"] = {"value": f"{hy_val:.0f} bps", "score": hy_score}
        scores.append(hy_score)
        weights.append(1.5)

    # 2. VIX level (from canary data)
    if not canary_df.empty:
        vix_row = canary_df[canary_df["ticker"] == "^VIX"]
        if not vix_row.empty:
            vix_val = float(vix_row["price"].iloc[0])
            # VIX: <15 = calm, 15-25 = normal, 25-35 = elevated, >35 = panic
            vix_score = min(100, max(0, (vix_val - 12) / 40 * 100))
            components["VIX"] = {"value": f"{vix_val:.1f}", "score": vix_score}
            scores.append(vix_score)
            weights.append(1.5)

    # 3. Yield curve inversion (T10Y2Y negative = stress)
    yc_df = fred_data.get("T10Y2Y", pd.DataFrame())
    if not yc_df.empty and "value" in yc_df.columns:
        yc_val = float(yc_df["value"].iloc[-1])
        # Inverted (<0) = stress; deeply inverted (<-0.5) = high stress
        if yc_val < 0:
            yc_score = min(100, abs(yc_val) / 1.5 * 100)
        else:
            yc_score = max(0, 20 - yc_val * 10)  # positive spread = low stress
        components["Yield Curve"] = {"value": f"{yc_val:+.2f}%", "score": yc_score}
        scores.append(yc_score)
        weights.append(1.2)

    # 4. Canary drawdown average
    if not canary_df.empty and "drawdown_52w" in canary_df.columns:
        # Exclude VIX and TLT from drawdown calc (they behave inversely)
        canary_filtered = canary_df[~canary_df["ticker"].isin(["^VIX", "TLT"])]
        if not canary_filtered.empty:
            avg_dd = float(canary_filtered["drawdown_52w"].mean())
            # avg drawdown: 0 to -5% = low, -5 to -15% = moderate, >-15% = high
            dd_score = min(100, max(0, abs(avg_dd) / 30 * 100))
            components["Canary Drawdown"] = {"value": f"{avg_dd:.1f}%", "score": dd_score}
            scores.append(dd_score)
            weights.append(1.0)

    if not scores:
        return 25.0, {"No Data": {"value": "N/A", "score": 25}}

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    final_score = weighted_sum / total_weight

    return round(final_score, 1), components


def _make_stress_gauge(score: float) -> go.Figure:
    """Create a doom-themed stress gauge (0-100)."""
    if score >= 70:
        bar_color = DOOM_RED
    elif score >= 40:
        bar_color = DOOM_YELLOW
    else:
        bar_color = COLORS["green"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "STRESS-O-METER", "font": {"size": 18, "color": DOOM_RED}},
        number={"suffix": " / 100", "font": {"size": 32, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_dim"]},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": DOOM_SURFACE,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#0A2A0A"},
                {"range": [30, 60], "color": "#2A2A0A"},
                {"range": [60, 80], "color": "#2A1A0A"},
                {"range": [80, 100], "color": "#2A0A0A"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10))
    apply_dark_layout(fig)
    fig.update_layout(paper_bgcolor=DOOM_BG, plot_bgcolor=DOOM_BG)
    return fig


def _make_credit_spread_chart(fred_data: dict) -> go.Figure | None:
    """Dual-line chart of HY OAS + IG OAS over 1 year."""
    hy_df = fred_data.get("BAMLH0A0HYM2", pd.DataFrame())
    ig_df = fred_data.get("BAMLC0A0CM", pd.DataFrame())

    if hy_df.empty and ig_df.empty:
        return None

    fig = go.Figure()

    if not hy_df.empty:
        fig.add_trace(go.Scatter(
            x=hy_df["date"], y=hy_df["value"],
            mode="lines", name="HY OAS (bps)",
            line=dict(color=DOOM_RED, width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 34, 34, 0.1)",
            hovertemplate="<b>HY OAS</b><br>%{x|%Y-%m-%d}: %{y:.0f} bps<extra></extra>",
        ))

    if not ig_df.empty:
        fig.add_trace(go.Scatter(
            x=ig_df["date"], y=ig_df["value"],
            mode="lines", name="IG OAS (bps)",
            line=dict(color=DOOM_YELLOW, width=2),
            hovertemplate="<b>IG OAS</b><br>%{x|%Y-%m-%d}: %{y:.0f} bps<extra></extra>",
        ))

    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
        yaxis=dict(title="Spread (bps)"),
        legend=dict(x=0.02, y=0.98),
    )
    apply_dark_layout(fig, title="Credit Spreads — HY vs IG OAS")
    fig.update_layout(paper_bgcolor=DOOM_BG, plot_bgcolor=DOOM_BG)
    return fig


def _make_yield_curve_chart(fred_data: dict) -> go.Figure | None:
    """Yield curve (T10Y2Y) line chart with inversion shading."""
    yc_df = fred_data.get("T10Y2Y", pd.DataFrame())
    if yc_df.empty:
        return None

    fig = go.Figure()

    # Color segments: red when inverted, green when positive
    colors = [DOOM_RED if v < 0 else COLORS["green"] for v in yc_df["value"]]

    fig.add_trace(go.Scatter(
        x=yc_df["date"], y=yc_df["value"],
        mode="lines", name="10Y-2Y Spread",
        line=dict(color=COLORS["blue"], width=2),
        hovertemplate="<b>10Y-2Y</b><br>%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=DOOM_RED, line_width=2, opacity=0.8)

    # Shade inversion zones
    if not yc_df.empty:
        inverted = yc_df[yc_df["value"] < 0]
        if not inverted.empty:
            fig.add_trace(go.Scatter(
                x=yc_df["date"], y=[0] * len(yc_df),
                mode="lines", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=yc_df["date"],
                y=[min(v, 0) for v in yc_df["value"]],
                mode="lines", line=dict(width=0), showlegend=False,
                fill="tonexty",
                fillcolor="rgba(255, 34, 34, 0.2)",
                hoverinfo="skip",
            ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        yaxis=dict(title="Spread (%)"),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Yield Curve (10Y - 2Y) — Below Zero = INVERTED")
    fig.update_layout(paper_bgcolor=DOOM_BG, plot_bgcolor=DOOM_BG)
    return fig


def _make_whale_exit_chart(exits_df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart of biggest whale exits."""
    if exits_df.empty:
        return None

    # Use top 15 for readability
    df = exits_df.head(15).copy()
    df["label"] = df.apply(
        lambda r: f"{r.get('filer', 'Unknown')[:20]} / {r.get('issuer', 'Unknown')[:15]}",
        axis=1,
    )
    # Value in thousands -> millions
    df["value_change_m"] = df["value_change"] / 1000

    fig = go.Figure(go.Bar(
        x=df["value_change_m"],
        y=df["label"],
        orientation="h",
        marker_color=DOOM_RED,
        text=[f"${v:,.0f}M ({p:+.1f}%)" if "pct_change" in df.columns and pd.notna(p) else f"${v:,.0f}M" for v, p in zip(df["value_change_m"], df.get("pct_change", [0]*len(df)))],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=10),
        hovertemplate="<b>%{y}</b><br>Change: $%{x:,.0f}M<br>%{customdata}<extra></extra>",
        customdata=[f"{p:+.1f}%" if pd.notna(p) else "N/A" for p in df.get("pct_change", [None]*len(df))],
    ))

    fig.update_layout(
        height=max(350, len(df) * 28),
        margin=dict(l=20, r=80, t=40, b=20),
        xaxis=dict(title="Value Change ($M)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Biggest Whale Exits (13F)")
    fig.update_layout(paper_bgcolor=DOOM_BG, plot_bgcolor=DOOM_BG)
    return fig


def _style_canary_df(df: pd.DataFrame) -> str:
    """Return styled HTML table for canary watchlist."""
    if df.empty:
        return "<p style='color:#888;'>No canary data available.</p>"

    rows_html = ""
    for cat in CANARY_TICKERS_ORDER:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue

        rows_html += f"""
        <tr>
            <td colspan="7" style="background:#1A0505;color:{DOOM_RED};font-weight:700;
                                   padding:8px 12px;border-bottom:1px solid {DOOM_BORDER};">
                {cat}
            </td>
        </tr>
        """

        for _, row in cat_df.iterrows():
            def _color(val):
                if val > 0:
                    return COLORS["green"]
                elif val < 0:
                    return DOOM_RED
                return COLORS["text_dim"]

            dd_style = ""
            if row["drawdown_52w"] < -20:
                dd_style = f"font-weight:700;color:{DOOM_RED};"
            else:
                dd_style = f"color:{_color(row['drawdown_52w'])};"

            vol_flag = ""
            if row["volume_ratio"] > 2.0:
                vol_flag = f' <span style="color:{DOOM_YELLOW};font-weight:700;">UNUSUAL</span>'

            rows_html += f"""
            <tr style="border-bottom:1px solid #222;">
                <td style="padding:4px 8px;color:{COLORS['text']};">{row['ticker']}</td>
                <td style="padding:4px 8px;color:{COLORS['text']};">${row['price']:,.2f}</td>
                <td style="padding:4px 8px;color:{_color(row['1w_ret'])};">{row['1w_ret']:+.1f}%</td>
                <td style="padding:4px 8px;color:{_color(row['1m_ret'])};">{row['1m_ret']:+.1f}%</td>
                <td style="padding:4px 8px;color:{_color(row['3m_ret'])};">{row['3m_ret']:+.1f}%</td>
                <td style="padding:4px 8px;{dd_style}">{row['drawdown_52w']:+.1f}%</td>
                <td style="padding:4px 8px;color:{COLORS['text_dim']};">{row['volume_ratio']:.1f}x{vol_flag}</td>
            </tr>
            """

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">
        <thead>
            <tr style="border-bottom:2px solid {DOOM_BORDER};color:{COLORS['text_dim']};">
                <th style="text-align:left;padding:6px 8px;">Ticker</th>
                <th style="text-align:left;padding:6px 8px;">Price</th>
                <th style="text-align:left;padding:6px 8px;">1W</th>
                <th style="text-align:left;padding:6px 8px;">1M</th>
                <th style="text-align:left;padding:6px 8px;">3M</th>
                <th style="text-align:left;padding:6px 8px;">52W DD</th>
                <th style="text-align:left;padding:6px 8px;">Vol Ratio</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


# Order for display
CANARY_TICKERS_ORDER = [
    "Regional Banks", "Commercial Real Estate", "Private Equity Exposure",
    "Subprime / Consumer Credit", "High Yield / Distressed", "Volatility / Fear",
]


def _render_recommendations(stress_score: float, components: dict,
                            canary_df: pd.DataFrame, exits_df: pd.DataFrame):
    """Rules-based actionable recommendations keyed on stress level."""
    st.markdown(f"### <span style='color:{DOOM_RED};'>Actionable Recommendations</span>", unsafe_allow_html=True)

    recs = []
    if stress_score >= 80:
        severity_color = DOOM_RED
        recs = [
            "REDUCE GROSS EXPOSURE IMMEDIATELY",
            "MAX HEDGE RATIOS — put spreads on major indices",
            "FAVOR CASH & TREASURIES over equities",
            "Halt new risk-on positions until stress subsides",
        ]
    elif stress_score >= 60:
        severity_color = DOOM_RED
        recs = [
            "Reduce equity beta — trim high-beta positions",
            "Increase put protection on core holdings",
            "Favor quality over growth — rotate to strong balance sheets",
            "Raise cash allocation by 10-15%",
        ]
    elif stress_score >= 40:
        severity_color = DOOM_YELLOW
        recs = [
            "Monitor credit spreads daily for acceleration",
            "Tighten stop-losses on speculative positions",
            "Reduce small-cap exposure — shift to large-cap quality",
            "Consider adding Treasury duration as hedge",
        ]
    else:
        severity_color = COLORS["green"]
        recs = [
            "Maintain current positioning",
            "Selective risk-on opportunities remain viable",
            "Monitor for deterioration in leading indicators",
        ]

    # Canary-specific warnings
    if not canary_df.empty and "drawdown_52w" in canary_df.columns:
        severe_canaries = canary_df[canary_df["drawdown_52w"] < -20]
        for _, row in severe_canaries.iterrows():
            cat = row.get("category", "")
            ticker = row.get("ticker", "")
            recs.append(f"AVOID {cat} sector — {ticker} down {row['drawdown_52w']:.0f}% from 52W high")

    # Whale-exit-specific warnings
    if not exits_df.empty:
        top_exits = exits_df.head(3)
        exit_names = [str(row.get("issuer", "Unknown"))[:20] for _, row in top_exits.iterrows()]
        if exit_names:
            recs.append(f"Smart money exiting: {', '.join(exit_names)} — consider reducing exposure")

    # Render
    for rec in recs:
        if rec.startswith("REDUCE") or rec.startswith("MAX") or rec.startswith("FAVOR CASH") or rec.startswith("Halt") or rec.startswith("AVOID"):
            icon = "&#9760;"
            color = DOOM_RED
        elif rec.startswith("Smart money"):
            icon = "&#9888;"
            color = DOOM_YELLOW
        elif stress_score >= 60:
            icon = "&#9888;"
            color = DOOM_RED
        elif stress_score >= 40:
            icon = "&#9888;"
            color = DOOM_YELLOW
        else:
            icon = "&#9679;"
            color = COLORS["green"]

        st.markdown(
            f'<div style="border-left:3px solid {color};padding:6px 12px;margin:4px 0;'
            f'background:{DOOM_SURFACE};border-radius:0 4px 4px 0;">'
            f'<span style="color:{color};font-weight:700;">{icon}</span> '
            f'<span style="color:{COLORS["text"]};font-size:13px;">{rec}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def run_quick_doom(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Fetches credit + canary data, generates doom briefing.
    Stores _doom_briefing to session_state.
    """
    import streamlit as st
    import pandas as pd
    import datetime as _dt
    from services.stress_client import get_credit_spreads, get_canary_signals
    from services.claude_client import generate_doom_briefing

    fred_data = get_credit_spreads()
    canary_df = get_canary_signals()
    stress_score, _ = _compute_stress_score(fred_data, canary_df)

    stress_text_parts = [f"Composite Stress Score: {stress_score:.1f}/100"]

    for label, key in [("HY Credit Spread (OAS)", "BAMLH0A0HYM2"), ("IG Credit Spread", "BAMLC0A0CM"), ("Yield Curve 10Y-2Y", "T10Y2Y")]:
        df = fred_data.get(key, pd.DataFrame())
        if not df.empty and "value" in df.columns:
            stress_text_parts.append(f"{label}: {float(df['value'].iloc[-1]):.2f}")

    if not canary_df.empty:
        stress_text_parts.append("\nCanary Watchlist:")
        for cat in canary_df["category"].unique():
            cat_df = canary_df[canary_df["category"] == cat]
            if not cat_df.empty:
                avg_ret = float(cat_df["1m_ret"].mean())
                avg_dd = float(cat_df["drawdown_52w"].mean())
                stress_text_parts.append(f"  {cat}: avg 1M {avg_ret:+.1f}%, drawdown {avg_dd:.1f}%")

    _ce = st.session_state.get("_current_events_digest", "")
    briefing = generate_doom_briefing(
        "\n".join(stress_text_parts),
        use_claude=use_claude, model=model, current_events=_ce,
    )
    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") else ("🧠 Regard Mode" if use_claude else "⚡ Freeloader Mode")
    st.session_state["_doom_briefing"] = briefing
    st.session_state["_doom_briefing_ts"] = _dt.datetime.now()
    st.session_state["_doom_briefing_engine"] = _tier
    return True


def render():
    # Pre-extract colors for f-string safety
    _dim = COLORS["text_dim"]
    _text = COLORS["text"]
    _blue = COLORS["blue"]
    _green = COLORS["green"]

    # ── Regime Exit Alert (reads Trade Journal open positions) ──
    from utils.journal import load_journal as _load_journal
    _open_trades = [t for t in _load_journal() if t["status"] == "open"]
    _open_longs  = [t for t in _open_trades if t["direction"] == "long"]
    _cur_regime  = st.session_state.get("_regime_context", {}).get("regime", "")
    if _open_longs and "Risk-Off" in _cur_regime:
        _tickers_at_risk = ", ".join(t["ticker"] for t in _open_longs)
        st.markdown(
            f'<div style="background:#7f1d1d;border:2px solid #ef4444;border-radius:8px;'
            f'padding:14px 18px;margin-bottom:16px;">'
            f'<div style="color:#fca5a5;font-weight:700;font-size:13px;letter-spacing:0.05em;">'
            f'⚠ REGIME SHIFT ALERT — RISK-OFF DETECTED</div>'
            f'<div style="color:#fecaca;font-size:12px;margin-top:6px;">'
            f'You have <b>{len(_open_longs)} open long position(s)</b> during a Risk-Off regime: '
            f'<b>{_tickers_at_risk}</b>. Review positions for exit per your regime-shift strategy.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif _open_longs and _cur_regime:
        st.caption(f"📋 {len(_open_longs)} open long(s) in journal · Current regime: {_cur_regime}")

    # ── Header ──
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1A0000 0%, #2A0505 50%, #1A0000 100%);
                    border:1px solid {DOOM_RED};border-radius:12px;
                    padding:24px 32px;margin-bottom:24px;">
          <div style="display:flex;align-items:center;gap:16px;">
            <span style="font-size:48px;">&#9760;</span>
            <div>
              <div style="font-size:28px;font-weight:700;color:{DOOM_RED};letter-spacing:3px;">
                STRESS SIGNALS / DOOMSDAY MONITOR
              </div>
              <div style="font-size:13px;color:#c9d1d9;margin-top:4px;">
                Credit stress, institutional exits, distress filings, and systemic risk canaries.
                This module is intentionally pessimistic &mdash; it exists to find what could go wrong.
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh Stress Data"):
            st.cache_data.clear()

    # ── Fetch all data ──
    from services.stress_client import (
        get_credit_spreads,
        get_canary_signals,
        scan_distress_filings,
        get_whale_exits,
        DTCC_TOP_CDS,
        CANARY_TICKERS,
    )

    with st.spinner("Scanning credit markets for stress signals..."):
        fred_data = get_credit_spreads()

    with st.spinner("Checking canary watchlist..."):
        canary_df = get_canary_signals()

    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 12H (FRED) / 4H (CANARY)")

    # ── 1. Stress-O-Meter ──
    stress_score, stress_components = _compute_stress_score(fred_data, canary_df)

    # Stress level label
    if stress_score >= 80:
        stress_label = "EXTREME STRESS"
        stress_emoji = "&#9760;"
        label_color = DOOM_RED
    elif stress_score >= 60:
        stress_label = "HIGH STRESS"
        stress_emoji = "&#9888;"
        label_color = DOOM_RED
    elif stress_score >= 40:
        stress_label = "ELEVATED"
        stress_emoji = "&#9888;"
        label_color = DOOM_YELLOW
    elif stress_score >= 20:
        stress_label = "MODERATE"
        stress_emoji = "&#9679;"
        label_color = COLORS["yellow"]
    else:
        stress_label = "LOW STRESS"
        stress_emoji = "&#9679;"
        label_color = COLORS["green"]

    col_gauge, col_components = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(_make_stress_gauge(stress_score), use_container_width=True)

    with col_components:
        st.markdown(
            f"""
            <div style="background:{DOOM_SURFACE};border:1px solid {DOOM_BORDER};
                        border-radius:8px;padding:16px 20px;margin-top:10px;">
                <div style="font-size:20px;font-weight:700;color:{label_color};margin-bottom:12px;">
                    {stress_emoji} {stress_label}
                </div>
                <div style="font-size:12px;color:{COLORS['text_dim']};margin-bottom:12px;">
                    Composite stress score components:
                </div>
            """,
            unsafe_allow_html=True,
        )
        for name, comp in stress_components.items():
            comp_score = comp["score"]
            if comp_score >= 60:
                comp_color = DOOM_RED
            elif comp_score >= 35:
                comp_color = DOOM_YELLOW
            else:
                comp_color = COLORS["green"]
            bar_width = max(5, min(100, comp_score))
            st.markdown(
                f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;font-size:12px;">
                        <span style="color:{COLORS['text']};">{name}</span>
                        <span style="color:{comp_color};">{comp['value']} ({comp_score:.0f}/100)</span>
                    </div>
                    <div style="background:#1A1A1A;border-radius:3px;height:6px;margin-top:2px;">
                        <div style="background:{comp_color};width:{bar_width}%;height:6px;border-radius:3px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. Credit Spreads ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>Credit Spreads</span>", unsafe_allow_html=True)

    import os
    fred_key_set = bool(os.getenv("FRED_API_KEY", ""))

    if fred_key_set:
        credit_fig = _make_credit_spread_chart(fred_data)
        if credit_fig:
            st.plotly_chart(credit_fig, use_container_width=True)
        else:
            st.info("Credit spread data unavailable from FRED.")

        yc_fig = _make_yield_curve_chart(fred_data)
        if yc_fig:
            st.plotly_chart(yc_fig, use_container_width=True)

        # Show TED spread and lending standards if available
        col_ted, col_lend = st.columns(2)
        with col_ted:
            ted_df = fred_data.get("TEDRATE", pd.DataFrame())
            if not ted_df.empty:
                ted_val = float(ted_df["value"].iloc[-1])
                ted_color = DOOM_RED if ted_val > 0.5 else (DOOM_YELLOW if ted_val > 0.3 else COLORS["green"])
                st.markdown(
                    _doom_container(
                        f"<div style='font-size:11px;color:{_dim};'>TED Spread (Interbank Stress)</div>"
                        f"<div style='font-size:28px;font-weight:700;color:{ted_color};'>{ted_val:.2f}%</div>"
                        f"<div style='font-size:10px;color:{_dim};'>Above 0.50 = elevated interbank risk</div>"
                    ),
                    unsafe_allow_html=True,
                )
        with col_lend:
            ci_df = fred_data.get("DRTSCILM", pd.DataFrame())
            if not ci_df.empty:
                ci_val = float(ci_df["value"].iloc[-1])
                ci_color = DOOM_RED if ci_val > 30 else (DOOM_YELLOW if ci_val > 10 else COLORS["green"])
                st.markdown(
                    _doom_container(
                        f"<div style='font-size:11px;color:{_dim};'>Banks Tightening C&I Loans (%)</div>"
                        f"<div style='font-size:28px;font-weight:700;color:{ci_color};'>{ci_val:+.1f}%</div>"
                        f"<div style='font-size:10px;color:{_dim};'>Positive = tightening. Above 30% = credit crunch signal</div>"
                    ),
                    unsafe_allow_html=True,
                )
    else:
        st.caption("ℹ FRED API key not set — using ETF proxies for credit spread data.")
        # Show HY/distressed canary data as proxy
        if not canary_df.empty:
            hy_proxy = canary_df[canary_df["category"] == "High Yield / Distressed"]
            if not hy_proxy.empty:
                st.dataframe(hy_proxy, use_container_width=True, hide_index=True)

    # ── 3. Canary Watchlist ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>Canary Watchlist</span>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:12px;margin-top:-10px;'>"
        "Tracking vulnerable sectors and stress proxies. Red = bleeding. Bold red drawdown = >20% off 52-week high.</p>",
        unsafe_allow_html=True,
    )

    if not canary_df.empty:
        # Use category order from CANARY_TICKERS
        global CANARY_TICKERS_ORDER
        from services.stress_client import CANARY_TICKERS as _CT
        CANARY_TICKERS_ORDER = list(_CT.keys())

        # Sort by category order then display as styled dataframe
        cat_order = {cat: i for i, cat in enumerate(CANARY_TICKERS_ORDER)}
        display_df = canary_df.copy()
        display_df["_sort"] = display_df["category"].map(cat_order).fillna(99)
        display_df = display_df.sort_values("_sort").drop(columns=["_sort"])

        # Format for display
        display_df = display_df.rename(columns={
            "ticker": "Ticker", "category": "Category", "price": "Price",
            "1w_ret": "1W %", "1m_ret": "1M %", "3m_ret": "3M %",
            "drawdown_52w": "52W DD %", "volume_ratio": "Vol Ratio",
        })

        def _color_returns(val):
            if isinstance(val, (int, float)):
                if val < -20:
                    return f"color: {DOOM_RED}; font-weight: bold"
                elif val < 0:
                    return f"color: {DOOM_RED}"
                elif val > 0:
                    return f"color: {_green}"
            return ""

        def _color_vol(val):
            if isinstance(val, (int, float)) and val > 2.0:
                return f"color: {DOOM_YELLOW}; font-weight: bold"
            return ""

        ret_cols = ["1W %", "1M %", "3M %", "52W DD %"]
        styled = display_df.style.map(
            _color_returns, subset=ret_cols
        ).map(
            _color_vol, subset=["Vol Ratio"]
        ).format({
            "Price": "${:.2f}",
            "1W %": "{:+.1f}%",
            "1M %": "{:+.1f}%",
            "3M %": "{:+.1f}%",
            "52W DD %": "{:+.1f}%",
            "Vol Ratio": "{:.1f}x",
        })

        st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

        # Alert for severe drawdowns
        severe = canary_df[canary_df["drawdown_52w"] < -20]
        if not severe.empty:
            tickers_list = ", ".join(severe["ticker"].tolist())
            st.markdown(
                f"""<div style="background:#2A0000;border-left:4px solid {DOOM_RED};
                            padding:10px 16px;border-radius:4px;margin-bottom:12px;">
                    <span style="color:{DOOM_RED};font-weight:700;">&#9888; SEVERE DRAWDOWN ALERT:</span>
                    <span style="color:{COLORS['text']};">{tickers_list} — down >20% from 52-week highs</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # Alert for unusual volume
        unusual_vol = canary_df[canary_df["volume_ratio"] > 2.0]
        if not unusual_vol.empty:
            vol_tickers = ", ".join(unusual_vol["ticker"].tolist())
            st.markdown(
                f"""<div style="background:#2A1A00;border-left:4px solid {DOOM_YELLOW};
                            padding:10px 16px;border-radius:4px;margin-bottom:12px;">
                    <span style="color:{DOOM_YELLOW};font-weight:700;">&#9888; UNUSUAL VOLUME:</span>
                    <span style="color:{COLORS['text']};">{vol_tickers} — volume >2x 20-day average</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("Canary watchlist data unavailable. Check network connectivity.")

    # ── 4. Distress Filings ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>Distress Filing Scanner</span>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:12px;margin-top:-10px;'>"
        "Recent SEC filings containing stress keywords: going concern, material weakness, "
        "covenant violation, default, bankruptcy, credit facility termination, liquidity crisis.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("View Distress Filings (last 30 days)", expanded=False):
        with st.spinner("Scanning SEC EDGAR for distress filings..."):
            try:
                filings = scan_distress_filings(days_back=30)
            except Exception as e:
                filings = []
                st.warning(f"Could not scan SEC filings: {e}")

        if filings:
            filings_html = """
            <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:monospace;">
                <thead>
                    <tr style="border-bottom:2px solid #661111;color:#888;">
                        <th style="text-align:left;padding:6px;">Date</th>
                        <th style="text-align:left;padding:6px;">Company</th>
                        <th style="text-align:left;padding:6px;">Ticker</th>
                        <th style="text-align:left;padding:6px;">Form</th>
                        <th style="text-align:left;padding:6px;">Keyword</th>
                        <th style="text-align:left;padding:6px;">Link</th>
                    </tr>
                </thead>
                <tbody>
            """
            for f in filings[:50]:
                link = f"<a href='{f['url']}' target='_blank' style='color:{COLORS['blue']};'>View</a>" if f["url"] else ""
                filings_html += f"""
                <tr style="border-bottom:1px solid #222;">
                    <td style="padding:4px 6px;color:{COLORS['text_dim']};">{f['date']}</td>
                    <td style="padding:4px 6px;color:{COLORS['text']};">{f['company'][:40]}</td>
                    <td style="padding:4px 6px;color:{DOOM_YELLOW};">{f['ticker']}</td>
                    <td style="padding:4px 6px;color:{COLORS['text_dim']};">{f['form_type']}</td>
                    <td style="padding:4px 6px;color:{DOOM_RED};">{f['keyword']}</td>
                    <td style="padding:4px 6px;">{link}</td>
                </tr>
                """
            filings_html += "</tbody></table>"
            st.markdown(
                _doom_container(filings_html),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:{COLORS['text_dim']};font-size:10px;'>"
                f"Found {len(filings)} filings. Data via SEC EDGAR full-text search.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:{COLORS['text_dim']};'>No distress filings found in the last 30 days, "
                "or SEC EDGAR search is unavailable.</p>",
                unsafe_allow_html=True,
            )

    # ── 5. Whale Exits ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>Whale Exits</span>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:12px;margin-top:-10px;'>"
        "Biggest institutional position closures and decreases from quarterly 13F filings.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Scanning 13F data for whale exits..."):
        try:
            exits_df = get_whale_exits(top_n=20)
        except Exception:
            exits_df = pd.DataFrame()

    if not exits_df.empty:
        exit_chart = _make_whale_exit_chart(exits_df)
        if exit_chart:
            st.plotly_chart(exit_chart, use_container_width=True)

        # Detail table
        with st.expander("Whale Exit Details", expanded=False):
            display_cols = []
            for col in ["filing_date", "filer", "issuer", "status", "value_change", "pct_change", "whale_category"]:
                if col in exits_df.columns:
                    display_cols.append(col)

            if display_cols:
                display_df = exits_df[display_cols].copy()
                if "value_change" in display_df.columns:
                    display_df["value_change"] = display_df["value_change"].apply(
                        lambda x: f"${x / 1000:,.0f}M" if pd.notna(x) else "N/A"
                    )
                if "pct_change" in display_df.columns:
                    display_df["pct_change"] = display_df["pct_change"].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                    )
                display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            _doom_container(
                f"<div style='color:{_dim};'>"
                "&#9888; 13F bulk data not available for the most recent quarter. "
                "This data is published ~45 days after quarter end. "
                "Check back after the filing deadline.</div>"
            ),
            unsafe_allow_html=True,
        )

    # ── 6. DTCC Top CDS Names ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>DTCC Top CDS Reference Entities</span>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:12px;margin-top:-10px;'>"
        f"Data as of {DTCC_TOP_CDS['as_of']} &mdash; updated quarterly from DTCC reports. "
        "Rising net notional / QoQ increases = more protection buying = rising default fears.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("View DTCC CDS Data", expanded=False):
        tab_sov, tab_corp = st.tabs(["Sovereign", "Corporate"])

        def _render_cds_table(entities: list[dict]):
            if not entities:
                st.info("No data available.")
                return

            cds_df = pd.DataFrame(entities)
            cds_df = cds_df.rename(columns={
                "entity": "Entity",
                "net_notional_bn": "Net Notional ($B)",
                "gross_notional_bn": "Gross Notional ($B)",
                "qoq_change_pct": "QoQ Change %",
                "contracts": "Contracts",
            })

            def _color_qoq(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return f"color: {DOOM_RED}"
                    elif val < 0:
                        return f"color: {_green}"
                return ""

            styled = cds_df.style.map(_color_qoq, subset=["QoQ Change %"]).format({
                "Net Notional ($B)": "${:.1f}B",
                "Gross Notional ($B)": "${:.1f}B",
                "QoQ Change %": "{:+.1f}%",
                "Contracts": "{:,}",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

        with tab_sov:
            _render_cds_table(DTCC_TOP_CDS.get("sovereign", []))

        with tab_corp:
            _render_cds_table(DTCC_TOP_CDS.get("corporate", []))

        st.caption("Sample data — update from DTCC quarterly reports at dtcc.com/repository-otc-data")

    # ── 7. AI Doom Briefing ──
    st.markdown(f"### <span style='color:{DOOM_RED};'>AI RISK INTELLIGENCE BRIEFING</span>", unsafe_allow_html=True)

    # Compile all stress data into text for the AI
    stress_text_parts = []

    # FRED data summary
    for sid, label in [
        ("BAMLH0A0HYM2", "HY OAS"),
        ("BAMLC0A0CM", "IG OAS"),
        ("T10Y2Y", "10Y-2Y Spread"),
        ("TEDRATE", "TED Spread"),
    ]:
        df = fred_data.get(sid, pd.DataFrame())
        if not df.empty and "value" in df.columns:
            latest = float(df["value"].iloc[-1])
            stress_text_parts.append(f"{label}: {latest:.2f}")

    # Stress score
    stress_text_parts.append(f"\nComposite Stress Score: {stress_score:.1f}/100 ({stress_label})")

    # Canary summary
    if not canary_df.empty:
        stress_text_parts.append("\nCanary Watchlist Summary:")
        for cat in CANARY_TICKERS_ORDER:
            cat_df = canary_df[canary_df["category"] == cat]
            if not cat_df.empty:
                avg_ret = float(cat_df["1m_ret"].mean())
                avg_dd = float(cat_df["drawdown_52w"].mean())
                stress_text_parts.append(f"  {cat}: avg 1M return {avg_ret:+.1f}%, avg drawdown {avg_dd:.1f}%")

    # Distress filings count
    try:
        n_filings = len(filings) if filings else 0
    except NameError:
        n_filings = 0
    stress_text_parts.append(f"\nDistress filings (last 30 days): {n_filings}")

    # Whale exits summary
    if not exits_df.empty:
        total_exit_val = exits_df["value_change"].sum() / 1000  # to millions
        stress_text_parts.append(f"Whale exits: {len(exits_df)} positions, total ${total_exit_val:,.0f}M closed/reduced")

    # Inject open portfolio positions for personalised doom briefing
    _journal_pos = [t for t in _open_trades if t["status"] == "open"]
    if _journal_pos:
        _pos_lines = []
        for _t in _journal_pos[:10]:
            _entry_px = _t.get("entry_price", 0)
            _thesis = (_t.get("thesis") or _t.get("notes") or "")[:80]
            _pos_lines.append(
                f"{_t['direction'].upper()} {_t['ticker']} @ ${_entry_px:.2f}"
                + (f" — {_thesis}" if _thesis else "")
            )
        stress_text_parts.append(
            "\nPortfolio Positions (user's open trades):\n" + "\n".join(f"  {l}" for l in _pos_lines)
        )

    stress_text = "\n".join(stress_text_parts)

    import os
    from services.claude_client import generate_doom_briefing

    _doom_has_claude = bool(os.getenv("XAI_API_KEY"))


    _has_anthropic_doom_has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    _doom_tier_opts = ["⚡ Freeloader Mode"] + (["🧠 Regard Mode"] if _doom_has_claude else []) + (["👑 Highly Regarded Mode"] if _has_anthropic_doom_has_claude else [])
    _doom_tier_map = {
        "⚡ Freeloader Mode": (False, None),
        "🧠 Regard Mode": (True, "grok-4-1-fast-reasoning"),
        "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
    }
    _sel_doom_tier = st.radio(
        "Briefing Engine", _doom_tier_opts, horizontal=True, key="doom_briefing_engine",
        help="Standard = Groq · Regard Mode = Grok 4.1 · Highly Regarded = Claude Sonnet"
    )
    st.caption("💡 👑 Sonnet recommended — briefing feeds into Valuation & Discovery; quality affects downstream AI reasoning")
    _use_claude, _doom_model = _doom_tier_map[_sel_doom_tier]
    with st.spinner("Generating risk intelligence briefing..."):
        try:
            _ce_digest = st.session_state.get("_current_events_digest", "")
            briefing = generate_doom_briefing(stress_text, use_claude=_use_claude, model=_doom_model, current_events=_ce_digest)
        except Exception as e:
            briefing = f"Unable to generate briefing: {e}"
    st.session_state["_doom_briefing"] = briefing
    st.session_state["_doom_briefing_ts"] = __import__("datetime").datetime.now()
    st.session_state["_doom_briefing_engine"] = _sel_doom_tier
    from services.play_log import append_play as _append_play
    _append_play("Doom Briefing", _sel_doom_tier, {"briefing": briefing})

    _doom_border = COLORS["bloomberg_orange"] if _use_claude else DOOM_RED
    _regard_badge = (
        f' &nbsp;<span style="font-size:10px;background:#FF8811;color:#000;'
        f'padding:1px 5px;border-radius:3px;font-weight:700;">{_sel_doom_tier}</span>'
        if _use_claude else ""
    )
    st.markdown(
        f"""
        <div style="background:{DOOM_BG};border:2px solid {_doom_border};border-radius:8px;
                    padding:20px 24px;margin-bottom:16px;">
            <div style="font-size:11px;color:{_doom_border};letter-spacing:2px;margin-bottom:12px;
                        border-bottom:1px solid {DOOM_BORDER};padding-bottom:8px;">
                &#9760; CLASSIFIED &mdash; AI RISK INTELLIGENCE BRIEFING &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M')}{_regard_badge}
            </div>
            <div style="color:{COLORS['text']};font-size:11px;line-height:1.7;white-space:pre-wrap;">
{briefing}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 8. Recommendations ──
    _render_recommendations(stress_score, stress_components, canary_df, exits_df)

    # ── Footer ──
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:10px;margin-top:24px;'>"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via FRED, Yahoo Finance, SEC EDGAR, DTCC | "
        "This module is intentionally pessimistic and designed to surface risks. "
        "Not financial advice.</p>",
        unsafe_allow_html=True,
    )
