import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.whale_screener import (
    WHALE_FILERS,
    get_available_quarters,
    screen_whale_buyers,
)
from services.claude_client import summarize_whale_activity
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("13F WHALE BUYERS")
    st.caption(
        "Compares each whale's two most recent 13F filings to surface new/changed positions. "
        f"Tracking **{len(WHALE_FILERS)} institutional filers** across SEC EDGAR."
    )

    # --- Filter controls ---
    fc1, fc2, fc3 = st.columns([1, 1, 1])

    with fc1:
        min_value = st.slider(
            "Min Position Value ($M)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Set to 0 to see all positions",
        )

    with fc2:
        category_options = ["all", "fundamental", "activist", "macro", "quant"]
        selected_categories = st.multiselect(
            "Filer Category",
            options=category_options,
            default=["all"],
        )

    with fc3:
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
        )
        progress_bar.empty()
    except Exception as e:
        progress_bar.empty()
        st.error(f"Failed to fetch 13F data: {e}")
        return

    if df.empty:
        st.warning(
            f"No 13F whale data available for Q{sel_quarter} {sel_year}. "
            "The SEC bulk file may not be published yet for this quarter."
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
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:Courier New,monospace;">'
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
            lambda n: str(n)[:30]
        )
        display["Issuer"] = display.get("issuer", pd.Series([""] * len(display))).fillna("").apply(
            lambda n: str(n)[:30]
        )
        display["CUSIP"] = display["cusip"].fillna("")
        display["Status"] = display["status"]

        show_cols = ["Filer", "Issuer", "CUSIP", "Value ($M)", "Change ($M)", "Change %", "Status"]
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


def _render_new_positions_chart(df: pd.DataFrame):
    """Horizontal bar chart of top 20 new/increased positions by value change."""
    top = df[df["value_change"] > 0].nlargest(20, "value_change").copy()
    if top.empty:
        return

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:Courier New,monospace;">'
            "TOP 20 POSITION INCREASES</span></div>",
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

        bar_colors = [
            COLORS["green"] if row["status"] == "NEW" else COLORS["accent"]
            for _, row in top.iterrows()
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=bar_colors,
                text=values.apply(lambda v: f"${v:,.0f}M"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Change: $%{x:,.0f}M<extra></extra>",
            )
        )

        apply_dark_layout(
            fig,
            title="Top 20 Position Increases by Value",
            xaxis_title="Value Change ($M)",
            margin=dict(l=250, r=80, t=50, b=40),
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Bar color: <span style='color:{COLORS['green']}'>green</span> = brand new position, "
            f"<span style='color:{COLORS['accent']}'>teal</span> = increased existing",
            unsafe_allow_html=True,
        )


def _render_convergence(df: pd.DataFrame):
    """Whale convergence: CUSIPs where 3+ whales bought."""
    buys = df[df["value_change"] > 0].copy()
    if buys.empty:
        return

    convergence = buys.groupby("cusip").agg(
        whale_count=("filer", "nunique"),
        whales=("filer", lambda x: ", ".join(sorted(set(str(n)[:20] for n in x)))),
        issuer=("issuer", "first"),
        total_change=("value_change", "sum"),
    ).reset_index()

    convergence = convergence[convergence["whale_count"] >= 3].sort_values(
        "whale_count", ascending=False
    )

    if convergence.empty:
        # Lower threshold to 2 if no 3+ convergence
        convergence = buys.groupby("cusip").agg(
            whale_count=("filer", "nunique"),
            whales=("filer", lambda x: ", ".join(sorted(set(str(n)[:20] for n in x)))),
            issuer=("issuer", "first"),
            total_change=("value_change", "sum"),
        ).reset_index()
        convergence = convergence[convergence["whale_count"] >= 2].sort_values(
            "whale_count", ascending=False
        )
        if convergence.empty:
            return
        threshold_label = "2+"
    else:
        threshold_label = "3+"

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["yellow"]};font-weight:700;font-family:Courier New,monospace;">'
            f"WHALE CONVERGENCE ({threshold_label} buyers on the same name)</span></div>",
            unsafe_allow_html=True,
        )

        for _, row in convergence.head(15).iterrows():
            issuer = str(row["issuer"])[:30] if pd.notna(row["issuer"]) else row["cusip"]
            val_m = row["total_change"] / 1000
            st.markdown(
                f'<div style="padding:6px 12px;margin:4px 0;border-left:3px solid {COLORS["yellow"]};">'
                f'<span style="color:{COLORS["text"]};font-family:Courier New,monospace;">'
                f'<b style="color:{COLORS["yellow"]}">{issuer}</b> — '
                f'{row["whale_count"]} whales, ${val_m:,.0f}M total buying<br>'
                f'<span style="color:{COLORS["text_dim"]};font-size:0.85em;">{row["whales"]}</span>'
                "</span></div>",
                unsafe_allow_html=True,
            )


def _render_ai_summary(df: pd.DataFrame):
    """AI narrative summary of whale activity."""
    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["blue"]};font-weight:700;font-family:Courier New,monospace;">'
            "AI WHALE ACTIVITY SUMMARY</span></div>",
            unsafe_allow_html=True,
        )

        # Build text summary of top 20 changes for the AI
        top20 = df.nlargest(20, df["value_change"].abs())
        lines = []
        for _, row in top20.iterrows():
            filer = str(row.get("filer", "Unknown"))[:30]
            issuer = str(row.get("issuer", "Unknown"))[:30]
            val_m = row["value_change"] / 1000
            status = row.get("status", "")
            category = row.get("whale_category", "")
            lines.append(
                f"{filer} ({category}): {status} {issuer} — ${val_m:+,.0f}M change"
            )

        activity_text = "\n".join(lines)

        with st.spinner("Generating AI analysis..."):
            try:
                summary = summarize_whale_activity(activity_text)
                st.info(summary)
            except Exception as e:
                st.warning(f"AI summary unavailable: {e}")


def _render_treemap(df: pd.DataFrame):
    """Treemap: block size = position value, color = filer category."""
    chart_df = df[df["value_curr"] > 0].copy()
    if chart_df.empty:
        return

    # Take top 50 for readability
    chart_df = chart_df.nlargest(50, "value_curr")

    with st.container(border=True):
        st.markdown(
            f'<div style="background:{COLORS["surface"]};padding:8px 14px;border-radius:6px;">'
            f'<span style="color:{COLORS["accent"]};font-weight:700;font-family:Courier New,monospace;">'
            "WHALE HOLDINGS TREEMAP</span></div>",
            unsafe_allow_html=True,
        )

        # Category -> color mapping
        cat_colors = {
            "fundamental": COLORS["green"],
            "activist": COLORS["yellow"],
            "macro": COLORS["blue"],
            "quant": COLORS["red"],
            "": COLORS["text_dim"],
        }

        filer_names = chart_df["filer"].fillna("Unknown").apply(lambda n: str(n)[:25])
        issuer_names = chart_df.get("issuer", pd.Series([""] * len(chart_df))).fillna("Unknown").apply(
            lambda n: str(n)[:20]
        )

        labels = [
            f"{iss}<br>({fil})"
            for iss, fil in zip(issuer_names, filer_names)
        ]
        parents = filer_names.tolist()
        values = (chart_df["value_curr"] / 1000).tolist()  # in $M

        colors = [
            cat_colors.get(str(c), COLORS["text_dim"])
            for c in chart_df["whale_category"]
        ]

        # Build hierarchical data: add parent entries for each filer
        unique_filers = chart_df.drop_duplicates(subset=["filer"])
        for _, row in unique_filers.iterrows():
            filer = str(row["filer"])[:25]
            cat = str(row.get("whale_category", ""))
            labels.append(filer)
            parents.append("")
            values.append(0)
            colors.append(cat_colors.get(cat, COLORS["text_dim"]))

        fig = go.Figure()
        fig.add_trace(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(
                    colors=colors,
                    line=dict(color=COLORS["bg"], width=2),
                ),
                branchvalues="total",
                textinfo="label",
                textfont=dict(size=11, color="white", family="Courier New, monospace"),
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}M<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="Whale Holdings by Filer & Category",
                font=dict(color=COLORS["text"]),
            ),
            paper_bgcolor=COLORS["bg"],
            font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
            height=600,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Color: "
            f"<span style='color:{COLORS['green']}'>fundamental</span> · "
            f"<span style='color:{COLORS['yellow']}'>activist</span> · "
            f"<span style='color:{COLORS['blue']}'>macro</span> · "
            f"<span style='color:{COLORS['red']}'>quant</span> — "
            "Block size = position value ($M)",
            unsafe_allow_html=True,
        )
