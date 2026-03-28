from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.whale_screener import (
    WHALE_FILERS,
    get_available_quarters,
    screen_whale_buyers,
)
from services.claude_client import summarize_whale_activity
from utils.theme import COLORS, apply_dark_layout, FONT_FAMILY


def run_quick_whale(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Scans top 100 whale 13F filings (1 quarter lookback), generates AI summary.
    Stores _whale_summary, _whale_summary_ts, _whale_summary_engine to session_state.
    """
    import datetime as _dt
    from services.whale_screener import screen_whale_buyers
    from services.claude_client import summarize_whale_activity

    df = screen_whale_buyers(top_n=100, whale_only=True, exclude_etfs=True, lookback_quarters=1)
    if df is None or df.empty:
        return False

    df_sorted = df.copy()
    df_sorted["_abs_change"] = df_sorted["value_change"].abs()
    top20 = df_sorted.nlargest(20, "_abs_change")
    lines = []
    for _, row in top20.iterrows():
        filer = str(row.get("filer", "Unknown"))[:30]
        issuer = str(row.get("issuer", "Unknown"))[:30]
        val_m = row["value_change"] / 1000
        pct = row.get("pct_change", 0)
        status = row.get("status", "")
        category = row.get("whale_category", "")
        lines.append(f"{filer} ({category}): {status} {issuer} — ${val_m:+,.0f}M change ({pct:+.1f}%)")

    summary = summarize_whale_activity("\n".join(lines), use_claude=use_claude, model=model)
    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") \
        else ("🧠 Regard Mode" if use_claude else "⚡ Groq")
    st.session_state["_whale_summary"] = summary
    st.session_state["_whale_summary_ts"] = _dt.datetime.now()
    st.session_state["_whale_summary_engine"] = _tier
    return True


def render():
    tab_whales, tab_activism = st.tabs(["🐋 13F Whale Movement", "🎯 Activism (13D/13G)"])
    with tab_whales:
        _render_whale_tab()
    with tab_activism:
        _render_activism_tab()


def _render_whale_tab():
    st.header("13F WHALE MOVEMENT")
    st.caption(
        "Compares each whale's two most recent 13F filings to surface new/changed positions. "
        f"Tracking **{len(WHALE_FILERS)} institutional filers** across SEC EDGAR."
    )

    # --- Filter controls ---
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])

    with fc1:
        lookback_options = {"1 Quarter": 1, "2 Quarters": 2, "1 Year": 4}
        lookback_label = st.selectbox("Lookback Period", list(lookback_options.keys()), index=0)
        lookback_q = lookback_options[lookback_label]

    with fc2:
        min_value = st.slider(
            "Min Position Value ($M)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Set to 0 to see all positions",
        )

    with fc3:
        category_options = ["all", "fundamental", "activist", "macro", "quant"]
        selected_categories = st.multiselect(
            "Filer Category",
            options=category_options,
            default=["all"],
        )

    with fc4:
        exclude_etfs = st.toggle("Exclude ETF Positions", value=True)

    # Normalize categories
    if not selected_categories or "all" in selected_categories:
        categories = None
    else:
        categories = selected_categories

    # --- Fetch data ---
    progress_bar = st.progress(0, text="Fetching 13F filings from SEC EDGAR...")

    def _update_progress(done, total):
        progress_bar.progress(done / total, text=f"Scanning whale filers... {done}/{total}")

    try:
        df = screen_whale_buyers(
            top_n=200,
            whale_only=True,
            exclude_etfs=exclude_etfs,
            min_value=min_value,
            categories=categories,
            progress_callback=_update_progress,
            lookback_quarters=lookback_q,
        )
        progress_bar.empty()
        st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 4H")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Failed to fetch 13F data: {e}")
        return

    if df.empty:
        st.warning(
            f"No 13F whale data available for the selected lookback period ({lookback_label}). "
            "SEC filings may not be published yet."
        )
        return

    # --- Summary metrics ---
    _render_metrics(df)

    # --- Main table ---
    _render_main_table(df)

    # --- Top 20 new positions bar chart ---
    _render_new_positions_chart(df)

    # --- Whale convergence ---
    _render_convergence(df)

    # --- AI summary ---
    _render_ai_summary(df)

    # --- Treemap ---
    _render_treemap(df)


def _render_metrics(df: pd.DataFrame):
    """Summary metrics row: total new positions, biggest new buy, most active whale."""
    new_positions = df[df["status"] == "NEW"]
    total_new = len(new_positions)

    # Biggest single new buy
    if not new_positions.empty:
        biggest_idx = new_positions["value_curr"].idxmax()
        biggest_row = new_positions.loc[biggest_idx]
        biggest_name = str(biggest_row.get("issuer", "Unknown"))[:25]
        biggest_filer = str(biggest_row.get("filer", ""))[:20]
        biggest_val = biggest_row["value_curr"] / 1000  # convert from $thousands to $millions
        biggest_label = f"{biggest_name} ({biggest_filer})"
        biggest_value = f"${biggest_val:,.0f}M"
    else:
        biggest_label = "N/A"
        biggest_value = "$0"

    # Most active whale (most position changes)
    if "filer" in df.columns:
        whale_counts = df["filer"].value_counts()
        if not whale_counts.empty:
            most_active = whale_counts.index[0]
            most_active_count = whale_counts.iloc[0]
        else:
            most_active = "N/A"
            most_active_count = 0
    else:
        most_active = "N/A"
        most_active_count = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("New Positions", f"{total_new:,}")
    c2.metric("Biggest New Buy", biggest_value, help=biggest_label)
    c3.metric("Most Active Whale", most_active, delta=f"{most_active_count} changes")


def _render_main_table(df: pd.DataFrame):
    """Main data table with color-coded status."""
    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:{FONT_FAMILY};">'
            "POSITION CHANGES</span></div>",
            unsafe_allow_html=True,
        )

        display = df.copy()

        # Format columns
        display["Value ($M)"] = (display["value_curr"] / 1000).apply(
            lambda v: f"${v:,.1f}" if v >= 1 else f"${v:,.2f}"
        )
        display["Change ($M)"] = (display["value_change"] / 1000).apply(
            lambda v: f"{'+' if v > 0 else ''}{v:,.1f}"
        )
        display["Change %"] = display["pct_change"].apply(
            lambda p: f"{p:+.1f}%" if abs(p) < 10000 else "NEW" if p > 0 else "CLOSED"
        )
        display["Filer"] = display["filer"].fillna("").apply(
            lambda n: str(n)[:25]
        )
        display["Issuer"] = display.get("issuer", pd.Series([""] * len(display))).fillna("").apply(
            lambda n: str(n)[:25]
        )
        display["Filed"] = display["filing_date"].fillna("")
        display["Status"] = display["status"]

        show_cols = ["Filed", "Filer", "Issuer", "Value ($M)", "Change ($M)", "Change %", "Status"]
        show = display[show_cols]

        def _color_status(val):
            colors_map = {
                "NEW": f"color: {COLORS['green']}; font-weight: bold",
                "INCREASED": f"color: {COLORS['green']}",
                "DECREASED": f"color: {COLORS['red']}",
                "CLOSED": f"color: {COLORS['red']}; font-weight: bold",
            }
            return colors_map.get(val, "")

        def _color_change(val):
            s = str(val)
            if s.startswith("+"):
                return f"color: {COLORS['green']}"
            elif s.startswith("-"):
                return f"color: {COLORS['red']}"
            return ""

        styled = show.style.map(_color_status, subset=["Status"]).map(
            _color_change, subset=["Change ($M)", "Change %"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)
        csv_data = show.to_csv(index=False)
        st.download_button("Export CSV", csv_data, "whale_positions.csv", "text/csv", key="dl_whale")


def _render_new_positions_chart(df: pd.DataFrame):
    """Horizontal bar chart of top 20 position changes — buys green, sells red."""
    # Top 10 buys + top 10 sells
    buys = df[df["value_change"] > 0].nlargest(10, "value_change")
    sells = df[df["value_change"] < 0].nsmallest(10, "value_change")
    top = pd.concat([buys, sells]).copy()
    if top.empty:
        return

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:{FONT_FAMILY};">'
            "TOP 20 POSITION CHANGES — BUYS vs SELLS</span></div>",
            unsafe_allow_html=True,
        )

        top = top.sort_values("value_change", ascending=True)
        issuer_names = top.get("issuer", pd.Series([""] * len(top))).fillna("Unknown")
        filer_names = top["filer"].fillna("")
        labels = [
            f"{str(iss)[:20]} ({str(fil)[:15]})"
            for iss, fil in zip(issuer_names, filer_names)
        ]
        values = top["value_change"] / 1000  # to $M
        pct_changes = top["pct_change"]

        bar_colors = []
        for _, row in top.iterrows():
            if row["status"] == "CLOSED":
                bar_colors.append("#FF2222")  # bright red
            elif row["status"] == "DECREASED":
                bar_colors.append("#CC4444")  # dimmer red
            elif row["status"] == "NEW":
                bar_colors.append(COLORS["green"])
            else:
                bar_colors.append(COLORS["accent"])

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"${v:+,.0f}M ({p:+.1f}%)" if abs(p) < 10000 else f"${v:+,.0f}M" for v, p in zip(values, pct_changes)],
                textposition="outside",
                customdata=pct_changes.tolist(),
                hovertemplate="<b>%{y}</b><br>Change: $%{x:+,.0f}M (%{customdata:+.1f}%)<extra></extra>",
            )
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["text_dim"], line_width=1)

        apply_dark_layout(
            fig,
            title="Top Position Changes by Value (Green = Buy, Red = Sell)",
            xaxis_title="Value Change ($M)",
            margin=dict(l=250, r=80, t=50, b=40),
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"<span style='color:{COLORS['green']}'>Green</span> = new position · "
            f"<span style='color:{COLORS['accent']}'>Teal</span> = increased · "
            f"<span style='color:#CC4444'>Light red</span> = decreased · "
            f"<span style='color:#FF2222'>Red</span> = fully closed",
            unsafe_allow_html=True,
        )


def _render_convergence(df: pd.DataFrame):
    """Whale convergence: names where multiple whales are buying or selling."""
    dim = COLORS["text_dim"]

    def _agg_convergence(data):
        return data.groupby("cusip").agg(
            whale_count=("filer", "nunique"),
            whales=("filer", lambda x: ", ".join(sorted(set(str(n)[:20] for n in x)))),
            issuer=("issuer", "first"),
            total_change=("value_change", "sum"),
            latest_date=("filing_date", "max"),
            avg_pct_change=("pct_change", "mean"),
        ).reset_index()

    def _find_convergence(data, min_whales=2):
        if data.empty:
            return pd.DataFrame()
        conv = _agg_convergence(data)
        conv = conv[conv["whale_count"] >= min_whales].sort_values("whale_count", ascending=False)
        return conv

    def _render_conv_list(conv_df, color, action_word):
        for _, row in conv_df.head(10).iterrows():
            issuer = str(row["issuer"])[:30] if pd.notna(row["issuer"]) else row["cusip"]
            val_m = abs(row["total_change"]) / 1000
            avg_pct = row.get("avg_pct_change", 0)
            filed = row.get("latest_date", "")
            date_str = f" · Filed {filed}" if filed else ""
            st.markdown(
                f'<div style="padding:6px 12px;margin:4px 0;border-left:3px solid {color};">'
                f'<span style="color:{COLORS["text"]};font-family:{FONT_FAMILY};">'
                f'<b style="color:{color}">{issuer}</b> — '
                f'{row["whale_count"]} whales {action_word}, ${val_m:,.0f}M total ({avg_pct:+.1f}% avg)'
                f'<span style="color:{dim};font-size:0.8em;">{date_str}</span><br>'
                f'<span style="color:{dim};font-size:0.85em;">{row["whales"]}</span>'
                "</span></div>",
                unsafe_allow_html=True,
            )

    # Buying convergence
    buys = df[df["value_change"] > 0].copy()
    buy_conv = _find_convergence(buys, 2)

    # Selling convergence
    sells = df[df["value_change"] < 0].copy()
    sell_conv = _find_convergence(sells, 2)

    if buy_conv.empty and sell_conv.empty:
        return

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["yellow"]};font-weight:700;font-family:{FONT_FAMILY};">'
            "WHALE CONVERGENCE — Multiple whales on the same name</span></div>",
            unsafe_allow_html=True,
        )

        tab_buy, tab_sell = st.tabs(["Buying Together", "Selling Together"])

        with tab_buy:
            if not buy_conv.empty:
                _render_conv_list(buy_conv, COLORS["green"], "buying")
            else:
                st.caption("No buying convergence detected.")

        with tab_sell:
            if not sell_conv.empty:
                _render_conv_list(sell_conv, "#FF2222", "selling")
            else:
                st.caption("No selling convergence detected.")


def _render_ai_summary(df: pd.DataFrame):
    """AI narrative summary of whale activity."""
    import os as _os
    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["blue"]};font-weight:700;font-family:{FONT_FAMILY};">'
            "AI WHALE ACTIVITY SUMMARY</span></div>",
            unsafe_allow_html=True,
        )

        # Engine selector
        _has_xai_whale = bool(_os.getenv("XAI_API_KEY"))
        _has_anthropic = bool(_os.getenv("ANTHROPIC_API_KEY"))
        _whale_tier_opts = ["⚡ Groq"] + (["🧠 Regard Mode"] if _has_xai_whale else []) + (["👑 Highly Regarded Mode"] if _has_anthropic else [])
        _whale_tier_map = {
            "⚡ Groq":                (False, None),
            "🧠 Regard Mode":         (True,  "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True,  "claude-sonnet-4-6"),
        }
        _sel_whale_tier = st.radio(
            "Engine", _whale_tier_opts, horizontal=True, key="whale_summary_engine"
        )
        st.markdown(
            '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
            'margin-top:-10px;margin-bottom:2px;">'
            '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("💡 🧠 Regard (Grok 4.1) sufficient — summarisation task")
        _use_cl_whale, _whale_model = _whale_tier_map[_sel_whale_tier]

        # Build text summary of top 20 changes for the AI
        df_sorted = df.copy()
        df_sorted["_abs_change"] = df_sorted["value_change"].abs()
        top20 = df_sorted.nlargest(20, "_abs_change")
        lines = []
        for _, row in top20.iterrows():
            filer = str(row.get("filer", "Unknown"))[:30]
            issuer = str(row.get("issuer", "Unknown"))[:30]
            val_m = row["value_change"] / 1000
            pct = row.get("pct_change", 0)
            status = row.get("status", "")
            category = row.get("whale_category", "")
            lines.append(
                f"{filer} ({category}): {status} {issuer} — ${val_m:+,.0f}M change ({pct:+.1f}%)"
            )

        activity_text = "\n".join(lines)

        if st.button("Generate Whale Summary", key="gen_whale_summary_btn", type="primary"):
            with st.spinner("Generating AI analysis..."):
                try:
                    summary = summarize_whale_activity(activity_text, use_claude=_use_cl_whale, model=_whale_model)
                    st.session_state["_whale_summary"] = summary
                    st.session_state["_whale_summary_ts"] = __import__("datetime").datetime.now()
                    st.session_state["_whale_summary_engine"] = _sel_whale_tier
                    from services.play_log import append_play as _append_play
                    _append_play("Whale Summary", _sel_whale_tier, {"summary": summary})
                except Exception as e:
                    st.warning(f"AI summary unavailable: {e}")

        _cached_summary = st.session_state.get("_whale_summary")
        if _cached_summary:
            _eng = st.session_state.get("_whale_summary_engine", "")
            _border = "2px solid #FF8811" if "👑" in _eng else ("1px solid #22c55e" if "🧠" in _eng else "1px solid #334155")
            st.markdown(
                f'<div style="border:{_border};border-radius:6px;padding:12px 16px;'
                f'background:#1A1F2E;margin-top:8px;">{_cached_summary}</div>',
                unsafe_allow_html=True,
            )


def _render_treemap(df: pd.DataFrame):
    """Treemap: block size = absolute value change, color = green (buy) / red (sell)."""
    chart_df = df[df["value_change"] != 0].copy()
    if chart_df.empty:
        return

    chart_df["abs_change"] = chart_df["value_change"].abs()
    chart_df = chart_df.nlargest(50, "abs_change")

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:{FONT_FAMILY};">'
            "POSITION CHANGES TREEMAP — BUYS vs SELLS</span></div>",
            unsafe_allow_html=True,
        )

        filer_names = chart_df["filer"].fillna("Unknown").apply(lambda n: str(n)[:25])
        issuer_names = chart_df.get("issuer", pd.Series([""] * len(chart_df))).fillna("Unknown").apply(
            lambda n: str(n)[:20]
        )

        change_m = chart_df["value_change"] / 1000  # to $M
        pct_changes_tree = chart_df["pct_change"]
        labels = [
            f"{iss}<br>({fil})<br>${v:+,.0f}M ({p:+.1f}%)" if abs(p) < 10000 else f"{iss}<br>({fil})<br>${v:+,.0f}M"
            for iss, fil, v, p in zip(issuer_names, filer_names, change_m, pct_changes_tree)
        ]
        values = (chart_df["abs_change"] / 1000).tolist()  # absolute $M for block size

        colors = [
            COLORS["green"] if v > 0 else COLORS["red"]
            for v in chart_df["value_change"]
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Treemap(
                labels=labels,
                parents=[""] * len(labels),
                values=values,
                marker=dict(
                    colors=colors,
                    line=dict(color=COLORS["bg"], width=2),
                ),
                textinfo="label",
                textfont=dict(size=10, color="white", family=FONT_FAMILY),
                hovertemplate="<b>%{label}</b><br>|Change|: $%{value:,.0f}M<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="Position Changes — Block size = $ change magnitude",
                font=dict(color=COLORS["text"]),
            ),
            paper_bgcolor=COLORS["bg"],
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
            height=600,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"<span style='color:{COLORS['green']}'>Green</span> = buys/increases · "
            f"<span style='color:{COLORS['red']}'>Red</span> = sells/decreases — "
            "Block size = absolute dollar change",
            unsafe_allow_html=True,
        )


def _render_activism_tab():
    """Schedule 13D/13G activism filings screener."""
    from services.activism_screener import get_activism_filings

    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">SCHEDULE 13D/13G ACTIVISM TRACKER</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "SC 13D = activist stake (>5% float, intent to influence). "
        "SC 13G = passive stake (>5%). Filed within 10 days of crossing threshold. "
        "Activism campaigns historically drive 10–30% price moves."
    )

    # Controls
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        days_back = st.selectbox("Lookback Window", [7, 14, 30, 60, 90], index=2,
                                  format_func=lambda d: f"Last {d} days", key="act_days")
    with ctrl2:
        include_13g = st.toggle("Include SC 13G (passive stakes)", value=False, key="act_13g")

    if st.button("FETCH ACTIVISM FILINGS", type="primary", key="act_fetch"):
        st.session_state["act_results"] = None  # clear cache to force refresh
        get_activism_filings.clear()

    with st.spinner("Querying SEC EDGAR for 13D/13G filings..."):
        filings = get_activism_filings(days_back=days_back, include_13g=include_13g)

    if not filings:
        st.warning("No SC 13D filings found for the selected window. Try extending the lookback period.")
        return

    # Summary metric
    activist_count = sum(1 for f in filings if f["form_type"] == "SC 13D")
    passive_count = sum(1 for f in filings if f["form_type"] == "SC 13G")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Filings", len(filings))
    m2.metric("SC 13D (Activist)", activist_count)
    m3.metric("SC 13G (Passive)", passive_count)

    st.markdown(f'<div style="margin:12px 0 6px 0;font-size:12px;color:{COLORS["bloomberg_orange"]};'
                f'font-weight:700;letter-spacing:0.06em;">RECENT FILINGS</div>', unsafe_allow_html=True)

    # Build table HTML
    html = (
        f'<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:12px;">'
        f'<tr style="border-bottom:2px solid {COLORS["bloomberg_orange"]};">'
    )
    for h in ["Date", "Form", "Filer (Activist / Shareholder)", "Target Company", "SEC Filing"]:
        html += f'<th style="padding:6px 10px;text-align:left;color:{COLORS["bloomberg_orange"]};">{h}</th>'
    html += "</tr>"

    for i, row in enumerate(filings):
        bg = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        form = row["form_type"]
        form_color = COLORS["negative"] if form == "SC 13D" else COLORS["text_dim"]
        link = (
            f'<a href="{row["filing_url"]}" target="_blank" style="color:{COLORS["accent"]};'
            f'text-decoration:none;">{row["accession_no"][:20] or "View"}</a>'
            if row.get("filing_url") else "—"
        )
        html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">{row["file_date"]}</td>'
            f'<td style="padding:5px 10px;color:{form_color};font-weight:700;">{form}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text"]};">{row["filer"][:45]}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["yellow"]};">{row["subject"][:35]}</td>'
            f'<td style="padding:5px 10px;">{link}</td>'
            f'</tr>'
        )
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-top:10px;font-size:11px;color:{COLORS["text_dim"]};">'
        f'SC 13D filings highlighted in red — these signal activist intent and are the highest-alpha event. '
        f'Click accession numbers to view full filings on SEC EDGAR.</div>',
        unsafe_allow_html=True,
    )
