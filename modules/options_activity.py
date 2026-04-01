from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout, FONT_FAMILY
from services.market_data import fetch_options_chain_snapshot_safe


def render():
    st.header("OPTIONS ACTIVITY")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view options activity.")
        return

    with st.spinner("Fetching options data..."):
        df, expirations = _get_options_data(ticker)

    if df is None or df.empty:
        st.warning("No options data available for this ticker.")
        return

    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 5M")

    # Expiration filter
    selected_exps = st.multiselect(
        "Expirations",
        options=expirations,
        default=expirations[:3],
        key="options_exp_filter",
    )
    if selected_exps:
        df = df[df["expiration"].isin(selected_exps)]

    if df.empty:
        st.info("No data for selected expirations.")
        return

    # --- Metrics ---
    calls = df[df["right"] == "Call"]
    puts = df[df["right"] == "Put"]

    call_vol = calls["volume"].sum()
    put_vol = puts["volume"].sum()
    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()
    pc_ratio = put_vol / call_vol if call_vol > 0 else 0

    if pc_ratio < 0.7:
        sentiment, sent_color = "BULLISH", COLORS["green"]
    elif pc_ratio > 1.0:
        sentiment, sent_color = "BEARISH", COLORS["red"]
    else:
        sentiment, sent_color = "NEUTRAL", COLORS["yellow"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Call Volume", f"{call_vol:,.0f}")
    c2.metric("Put Volume", f"{put_vol:,.0f}")
    c3.metric("P/C Ratio", f"{pc_ratio:.2f}")
    c4.metric("Call OI", f"{call_oi:,.0f}")
    c5.metric("Put OI", f"{put_oi:,.0f}")

    st.markdown(
        f"<div style='text-align:center; padding:8px; border:1px solid {sent_color}; "
        f"border-radius:8px; margin:10px 0;'>"
        f"<span style='color:{sent_color}; font-size:1.1em; font-weight:bold;'>"
        f"Options Sentiment: {sentiment} (P/C {pc_ratio:.2f})</span></div>",
        unsafe_allow_html=True,
    )

    # Inline log button
    from modules.forecast_accuracy import render_log_button
    _oc1, _oc2 = st.columns([4, 1])
    with _oc2:
        _opt_conf = 70 if sentiment != "NEUTRAL" else 50
        render_log_button(
            signal_type="valuation",
            prediction="Buy" if sentiment == "BULLISH" else ("Sell" if sentiment == "BEARISH" else "Hold"),
            confidence=_opt_conf,
            summary=f"Options flow signal: {sentiment} — P/C ratio {pc_ratio:.2f} (call vol {int(call_vol):,} / put vol {int(put_vol):,})",
            ticker=ticker,
            horizon_days=7,  # options flow is very short-lived
            key=f"opt_log_{ticker}_{sentiment}",
            label="📌 Log Signal",
        )

    # Persist P/C ratio signal for downstream use (valuation, portfolio intelligence)
    st.session_state["_options_sentiment"] = {
        "ticker": ticker,
        "sentiment": sentiment,
        "pc_ratio": round(pc_ratio, 3),
        "call_vol": int(call_vol),
        "put_vol": int(put_vol),
    }

    # --- Volume by expiration ---
    _render_volume_by_expiration(df)

    # --- OI by strike ---
    _render_oi_by_strike(df, ticker)

    # --- Gamma Exposure ---
    _render_gamma_exposure(ticker)

    # --- IV smile ---
    _render_iv_smile(df, ticker)

    # --- Treemap ---
    _render_options_treemap(df, ticker)

    # --- Unusual activity ---
    _render_unusual_activity(df, ticker)

    # --- Highest OI concentration ---
    _render_oi_concentration(df)



def _render_volume_by_expiration(df: pd.DataFrame):
    """Grouped bar chart of call vs put volume by expiration."""
    vol_by_exp = df.groupby(["expiration", "right"])["volume"].sum().reset_index()

    fig = go.Figure()
    for right, color, name in [("Call", COLORS["green"], "Calls"), ("Put", COLORS["red"], "Puts")]:
        subset = vol_by_exp[vol_by_exp["right"] == right]
        fig.add_trace(
            go.Bar(x=subset["expiration"], y=subset["volume"], name=name, marker_color=color)
        )

    apply_dark_layout(
        fig, title="Volume by Expiration", barmode="group",
        xaxis_title="Expiration", yaxis_title="Volume",
    )
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def _render_oi_by_strike(df: pd.DataFrame, ticker: str):
    """Mirrored bar chart showing call OI vs put OI by strike price."""
    oi = df.groupby(["strike", "right"])["openInterest"].sum().reset_index()
    call_oi = oi[oi["right"] == "Call"].set_index("strike")["openInterest"]
    put_oi = oi[oi["right"] == "Put"].set_index("strike")["openInterest"]
    call_wall_strike = call_oi.idxmax() if not call_oi.empty else None
    put_wall_strike = put_oi.idxmax() if not put_oi.empty else None

    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_strikes,
            y=[call_oi.get(s, 0) for s in all_strikes],
            name="Call OI",
            marker_color=COLORS["green"],
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_strikes,
            y=[-put_oi.get(s, 0) for s in all_strikes],
            name="Put OI",
            marker_color=COLORS["red"],
            opacity=0.8,
        )
    )

    apply_dark_layout(
        fig, title=f"Open Interest by Strike: {ticker}",
        xaxis_title="Strike ($)", yaxis_title="Open Interest",
        barmode="relative",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(height=400)
    if call_wall_strike is not None:
        fig.add_vline(x=call_wall_strike, line_dash="dash", line_color=COLORS["green"], line_width=2,
                      annotation_text=f"CALL WALL ${call_wall_strike:.0f}", annotation_position="top right",
                      annotation_font=dict(color=COLORS["green"], size=11))
    if put_wall_strike is not None:
        fig.add_vline(x=put_wall_strike, line_dash="dash", line_color=COLORS["red"], line_width=2,
                      annotation_text=f"PUT WALL ${put_wall_strike:.0f}", annotation_position="top left",
                      annotation_font=dict(color=COLORS["red"], size=11))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above = Call OI · Below = Put OI · Dashed lines = Call/Put walls (max OI strikes)")


def _render_iv_smile(df: pd.DataFrame, ticker: str):
    """IV smile/skew chart for the nearest expiration."""
    nearest_exp = df["expiration"].min()
    exp_df = df[df["expiration"] == nearest_exp]

    fig = go.Figure()
    for right, color, name in [("Call", COLORS["green"], "Calls"), ("Put", COLORS["red"], "Puts")]:
        subset = exp_df[exp_df["right"] == right].sort_values("strike")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["strike"], y=subset["impliedVolatility"] * 100,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1f}%<extra></extra>",
            )
        )

    apply_dark_layout(
        fig, title=f"IV Smile: {ticker} ({nearest_exp})",
        xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _render_options_treemap(df: pd.DataFrame, ticker: str):
    """Treemap of options volume: size = volume, color = calls (green) / puts (red)."""
    chart_df = df[df["volume"] > 0].copy()
    if chart_df.empty:
        return

    grouped = chart_df.groupby(["strike", "expiration", "right"]).agg(
        total_vol=("volume", "sum"),
        total_oi=("openInterest", "sum"),
    ).reset_index()

    if grouped.empty:
        return

    labels = [
        f"${strike:.0f} {right}<br>{exp}<br>Vol: {vol:,.0f}"
        for strike, exp, right, vol in zip(
            grouped["strike"], grouped["expiration"], grouped["right"], grouped["total_vol"]
        )
    ]
    colors = [COLORS["green"] if r == "Call" else COLORS["red"] for r in grouped["right"]]

    fig = go.Figure()
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[""] * len(grouped),
            values=grouped["total_vol"].tolist(),
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label",
            textfont=dict(size=12, color="white", family=FONT_FAMILY),
            hovertemplate="<b>%{label}</b><br>OI: %{customdata:,.0f}<extra></extra>",
            customdata=grouped["total_oi"].tolist(),
        )
    )

    fig.update_layout(
        title=dict(text=f"Options Volume Map: {ticker}", font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["bg"],
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Block size = volume · Green = calls · Red = puts")


def _render_unusual_activity(df: pd.DataFrame, ticker: str = ""):
    """Table of contracts with unusually high volume relative to open interest,
    plus an overall sentiment verdict and visualization chart."""
    active = df[(df["volume"] > 0) & (df["openInterest"] > 0)].copy()
    if active.empty:
        st.info("No options with both volume and OI to detect unusual activity.")
        return

    active["vol_oi"] = active["volume"] / active["openInterest"]
    unusual = active[active["vol_oi"] > 2.0].sort_values("vol_oi", ascending=False).head(15)

    if unusual.empty:
        st.info("No unusual activity detected (Vol/OI > 2.0)")
        return

    st.subheader("Unusual Activity (Vol/OI > 2.0)")

    # --- Sentiment verdict ---
    u_calls = unusual[unusual["right"] == "Call"]
    u_puts = unusual[unusual["right"] == "Put"]
    call_unusual_vol = u_calls["volume"].sum()
    put_unusual_vol = u_puts["volume"].sum()
    total_unusual_vol = call_unusual_vol + put_unusual_vol

    if total_unusual_vol > 0:
        call_pct = call_unusual_vol / total_unusual_vol * 100
        put_pct = put_unusual_vol / total_unusual_vol * 100
    else:
        call_pct = put_pct = 50

    if call_pct >= 65:
        ua_sentiment, ua_color, ua_icon = "BULLISH", COLORS["green"], "🟢"
        ua_detail = "Heavy unusual call activity — smart money may be positioning for upside"
    elif put_pct >= 65:
        ua_sentiment, ua_color, ua_icon = "BEARISH", COLORS["red"], "🔴"
        ua_detail = "Heavy unusual put activity — smart money may be hedging or betting on downside"
    else:
        ua_sentiment, ua_color, ua_icon = "MIXED", COLORS["yellow"], "🟡"
        ua_detail = "Unusual activity split between calls and puts — no clear directional bias"

    if ticker:
        st.session_state["_unusual_activity_sentiment"] = {
            "ticker": ticker,
            "sentiment": ua_sentiment,
            "call_pct": round(call_pct, 1),
            "put_pct": round(put_pct, 1),
            "flagged_contracts": len(unusual),
        }

    st.markdown(
        f'<div style="background:rgba(0,0,0,0.3); border:2px solid {ua_color}; '
        f'border-radius:10px; padding:16px; margin-bottom:16px;">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<span style="font-size:28px;">{ua_icon}</span>'
        f'<div>'
        f'<div style="color:{ua_color}; font-size:20px; font-weight:800;">'
        f'Unusual Activity Sentiment: {ua_sentiment}</div>'
        f'<div style="color:#aaa; font-size:13px; margin-top:4px;">{ua_detail}</div>'
        f'</div></div>'
        f'<div style="display:flex; gap:24px; margin-top:12px; font-size:13px; color:#ccc;">'
        f'<span>Unusual Call Vol: <b style="color:{COLORS["green"]}">{call_unusual_vol:,.0f}</b> '
        f'({call_pct:.0f}%)</span>'
        f'<span>Unusual Put Vol: <b style="color:{COLORS["red"]}">{put_unusual_vol:,.0f}</b> '
        f'({put_pct:.0f}%)</span>'
        f'<span>Contracts flagged: <b>{len(unusual)}</b></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Unusual activity chart: horizontal bars by strike ---
    _render_unusual_chart(unusual)

    # --- Table ---
    def _highlight_type(val):
        if val == "Call":
            return f"color: {COLORS['green']}"
        elif val == "Put":
            return f"color: {COLORS['red']}"
        return ""

    show = unusual[["expiration", "strike", "right", "volume", "openInterest", "impliedVolatility", "vol_oi"]].copy()
    show.columns = ["Expiry", "Strike", "Type", "Volume", "OI", "IV", "Vol/OI"]
    show["Strike"] = show["Strike"].apply(lambda v: f"${v:.2f}")
    show["IV"] = show["IV"].apply(lambda v: f"{v:.1%}")
    show["Vol/OI"] = show["Vol/OI"].apply(lambda v: f"{v:.2f}")

    st.dataframe(
        show.style.map(_highlight_type, subset=["Type"]),
        use_container_width=True,
        hide_index=True,
    )
    csv = show.to_csv(index=False)
    st.download_button("Export CSV", csv, f"unusual_options.csv", "text/csv", key="dl_unusual_options")


def _render_unusual_chart(unusual: pd.DataFrame):
    """Horizontal bar chart of unusual contracts: call volume right, put volume left."""
    chart = unusual.copy()
    chart["label"] = chart.apply(
        lambda r: f"${r['strike']:.0f} {r['expiration']}", axis=1
    )

    # Aggregate by label+right in case of duplicates
    agg = chart.groupby(["label", "right"])["volume"].sum().reset_index()
    call_data = agg[agg["right"] == "Call"].set_index("label")["volume"]
    put_data = agg[agg["right"] == "Put"].set_index("label")["volume"]
    all_labels = sorted(set(call_data.index) | set(put_data.index))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=all_labels,
            x=[call_data.get(l, 0) for l in all_labels],
            name="Unusual Calls",
            orientation="h",
            marker_color=COLORS["green"],
            opacity=0.85,
            hovertemplate="%{y}<br>Call Vol: %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=all_labels,
            x=[-put_data.get(l, 0) for l in all_labels],
            name="Unusual Puts",
            orientation="h",
            marker_color=COLORS["red"],
            opacity=0.85,
            hovertemplate="%{y}<br>Put Vol: %{customdata:,.0f}<extra></extra>",
            customdata=[put_data.get(l, 0) for l in all_labels],
        )
    )

    apply_dark_layout(
        fig,
        title="Unusual Activity by Strike",
        xaxis_title="Volume (Calls → | ← Puts)",
        yaxis_title="",
        barmode="relative",
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        margin=dict(l=120, r=40, t=80, b=50),
    )
    max_vol = max(
        max((call_data.get(l, 0) for l in all_labels), default=0),
        max((put_data.get(l, 0) for l in all_labels), default=0),
    )
    fig.update_xaxes(range=[-max_vol * 1.15, max_vol * 1.15] if max_vol > 0 else None)
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_layout(height=max(350, len(all_labels) * 40 + 140))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green bars (right) = unusual call volume · Red bars (left) = unusual put volume · Sorted by strike")


def _render_oi_concentration(df: pd.DataFrame):
    """Show top 5 strikes by OI with call/put split."""
    oi_by_strike = df.groupby("strike").agg(
        call_oi=("openInterest", lambda x: x[df.loc[x.index, "right"] == "Call"].sum()),
        put_oi=("openInterest", lambda x: x[df.loc[x.index, "right"] == "Put"].sum()),
        total_oi=("openInterest", "sum"),
    ).reset_index()

    if oi_by_strike.empty:
        return

    top5 = oi_by_strike.nlargest(5, "total_oi")
    if top5.empty:
        return

    st.subheader("Highest OI Concentration")
    st.caption("Key support/resistance levels based on open interest concentration")

    for _, row in top5.iterrows():
        total = row["total_oi"]
        call_pct = row["call_oi"] / total * 100 if total > 0 else 0
        put_pct = row["put_oi"] / total * 100 if total > 0 else 0
        dominant = "CALL" if call_pct > put_pct else "PUT"
        dom_color = COLORS["green"] if dominant == "CALL" else COLORS["red"]

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:6px 12px;'
            f'border-left:3px solid {dom_color};margin:4px 0;">'
            f'<span style="font-size:16px;font-weight:700;color:{COLORS["text"]};">${row["strike"]:.0f}</span>'
            f'<span style="color:{COLORS["green"]};font-size:13px;">C: {row["call_oi"]:,.0f} ({call_pct:.0f}%)</span>'
            f'<span style="color:{COLORS["red"]};font-size:13px;">P: {row["put_oi"]:,.0f} ({put_pct:.0f}%)</span>'
            f'<span style="color:{COLORS["text_dim"]};font-size:12px;">Total: {total:,.0f}</span>'
            f'<span style="color:{dom_color};font-size:11px;font-weight:700;">{dominant} DOMINANT</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_gamma_exposure(ticker: str):
    """Gamma exposure profile using options chain snapshot."""
    snap = fetch_options_chain_snapshot_safe(ticker, max_expiries=3)
    if not snap:
        st.caption("Gamma exposure data unavailable for this ticker.")
        return

    import numpy as np

    strikes = np.array(snap["strikes"])
    net_gamma = np.array(snap["net_gamma_proxy"])
    price = snap["price"]

    total_gamma = float(net_gamma.sum())
    if total_gamma >= 0:
        gamma_label = "LONG GAMMA — STABLE / PIN RISK"
        gamma_color = COLORS["green"]
    else:
        gamma_label = "SHORT GAMMA — VOLATILE / BREAKOUT RISK"
        gamma_color = COLORS["red"]

    st.subheader("Gamma Exposure")
    st.markdown(
        f'<div style="text-align:center;padding:12px;border:2px solid {gamma_color};'
        f'border-radius:8px;margin:10px 0;">'
        f'<span style="color:{gamma_color};font-size:1.2em;font-weight:bold;">'
        f'{gamma_label}</span></div>',
        unsafe_allow_html=True,
    )

    # Gamma profile chart
    bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in net_gamma]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strikes.tolist(),
        y=net_gamma.tolist(),
        marker_color=bar_colors,
        opacity=0.7,
        hovertemplate="Strike $%{x:.0f}<br>Net Gamma: %{y:.0f}<extra></extra>",
    ))

    # Current price line
    fig.add_vline(x=price, line_color=COLORS["blue"], line_dash="dash", line_width=2,
                  annotation_text=f"SPOT ${price:.2f}", annotation_position="top right",
                  annotation_font=dict(color=COLORS["blue"], size=11))

    # Gamma flip point
    cumulative = np.cumsum(net_gamma)
    gamma_flip = None
    for i in range(1, len(cumulative)):
        if (cumulative[i - 1] <= 0 < cumulative[i]) or (cumulative[i - 1] >= 0 > cumulative[i]):
            gamma_flip = float(strikes[i])
            break

    if gamma_flip is not None:
        fig.add_vline(x=gamma_flip, line_color=COLORS["yellow"], line_dash="dot", line_width=2,
                      annotation_text=f"FLIP ${gamma_flip:.0f}", annotation_position="bottom right",
                      annotation_font=dict(color=COLORS["yellow"], size=11))

    apply_dark_layout(fig, title=f"Net Gamma by Strike: {ticker}",
                      xaxis_title="Strike ($)", yaxis_title="Net Gamma Proxy")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = positive gamma (stabilizing) · Red = negative gamma (amplifying) · Yellow = gamma flip point")


@st.cache_data(ttl=300)
def _get_options_data(ticker: str) -> tuple:
    """Fetch options chain from yfinance. Returns (DataFrame, list of expirations)."""
    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options)

        if not expirations:
            return None, []

        all_rows = []
        for exp in expirations:
            chain = stock.option_chain(exp)

            for side, label in [(chain.calls, "Call"), (chain.puts, "Put")]:
                side_df = side.copy()
                side_df["expiration"] = exp
                side_df["right"] = label
                all_rows.append(side_df)

        df = pd.concat(all_rows, ignore_index=True)

        # Keep relevant columns, fill NaN
        keep = ["expiration", "strike", "right", "lastPrice", "bid", "ask",
                "volume", "openInterest", "impliedVolatility"]
        for col in keep:
            if col not in df.columns:
                df[col] = 0
        df = df[keep]
        df["volume"] = df["volume"].fillna(0).astype(int)
        df["openInterest"] = df["openInterest"].fillna(0).astype(int)
        df["impliedVolatility"] = df["impliedVolatility"].fillna(0)

        return df, expirations
    except Exception:
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS FLOW SENTIMENT — QIR background scoring engine
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_of(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _score_to_dir(s: float) -> str:
    if s > 0.15:  return "Bullish"
    if s < -0.15: return "Bearish"
    return "Neutral"


def _build_options_flow_dashboard(spy_chain: dict, gamma_data: dict | None) -> dict:
    """4-signal Options Flow Sentiment score (0-100) for SPY — hours-to-days timeframe.

    Signals: P/C Ratio, Gamma Zone, IV Skew, Unusual Activity Bias.
    Returns same structure as _tactical_context for consistent downstream consumption.
    """
    strikes    = spy_chain.get("strikes", [])
    call_oi    = spy_chain.get("call_oi", [])
    put_oi     = spy_chain.get("put_oi", [])
    spot_price = spy_chain.get("price", 0)

    import pandas as _pd

    # Pre-fetched arrays from spy_chain (populated by run_quick_options_flow)
    _s_call_vol = spy_chain.get("call_vol", [])
    _s_put_vol  = spy_chain.get("put_vol",  [])
    _s_call_iv  = spy_chain.get("call_iv",  [])
    _s_put_iv   = spy_chain.get("put_iv",   [])
    _s_call_oi  = spy_chain.get("call_oi",  [])
    _s_put_oi   = spy_chain.get("put_oi",   [])
    _has_vol    = bool(_s_call_vol) and (sum(_s_call_vol) + sum(_s_put_vol)) > 0

    # ── Signal 1: P/C Ratio (weight 3.0) ─────────────────────────────────────
    total_call_vol = total_put_vol = 0
    if _has_vol:
        total_call_vol = sum(_s_call_vol)
        total_put_vol  = sum(_s_put_vol)
    else:
        # Fallback: use OI as volume proxy, then try fresh yfinance fetch
        if _s_call_oi and _s_put_oi:
            total_call_vol = sum(_s_call_oi)
            total_put_vol  = sum(_s_put_oi)
        else:
            try:
                import yfinance as _yf
                _spy = _yf.Ticker("SPY")
                _expiries = list(_spy.options or [])[:2]
                for _exp in _expiries:
                    _ch = _spy.option_chain(_exp)
                    total_call_vol += int(_ch.calls["volume"].fillna(0).sum())
                    total_put_vol  += int(_ch.puts["volume"].fillna(0).sum())
            except Exception:
                pass

    pc_ratio   = (total_put_vol / total_call_vol) if total_call_vol > 0 else 1.0
    sig1_score = _clamp_of((0.9 - pc_ratio) / 0.3)
    pc_display = f"{pc_ratio:.2f} ({'bullish' if pc_ratio < 0.8 else ('bearish' if pc_ratio > 1.1 else 'neutral')})"

    # ── Signal 2: Gamma Zone (weight 1.5) ─────────────────────────────────────
    sig2_score    = 0.0
    gamma_display = "N/A (neutral)"
    if gamma_data:
        zone = gamma_data.get("zone", "")
        if "Positive" in zone:
            sig2_score    = 0.4
            gamma_display = "Positive (stabilizing)"
        elif "Negative" in zone:
            sig2_score    = -0.6
            gamma_display = "Negative (amplifying)"
        gflip = gamma_data.get("gamma_flip")
        if gflip and spot_price:
            gamma_display += f" · flip ${gflip:.0f}"

    # ── Signal 3: IV Skew (weight 2.0) ────────────────────────────────────────
    sig3_score   = 0.0
    skew_display = "N/A"
    if _s_call_iv and _s_put_iv and spot_price:
        try:
            import numpy as _np3
            _strikes_arr = _np3.array(strikes)
            _call_iv_arr = _np3.array(_s_call_iv)
            _put_iv_arr  = _np3.array(_s_put_iv)
            _otm_put_idx  = int(_np3.argmin(_np3.abs(_strikes_arr - spot_price * 0.95)))
            _otm_call_idx = int(_np3.argmin(_np3.abs(_strikes_arr - spot_price * 1.05)))
            _put_iv_val  = _put_iv_arr[_otm_put_idx]
            _call_iv_val = _call_iv_arr[_otm_call_idx]
            if _call_iv_val > 0 and _put_iv_val > 0:
                _skew        = _put_iv_val / _call_iv_val
                sig3_score   = _clamp_of((1.1 - _skew) / 0.2)
                skew_display = f"{_skew:.2f} ({'flat' if _skew < 1.05 else ('steep' if _skew > 1.3 else 'moderate')})"
        except Exception:
            pass
    if skew_display == "N/A":
        try:
            import yfinance as _yf2
            _spy2 = _yf2.Ticker("SPY")
            _exp0 = list(_spy2.options or [])[0] if _spy2.options else None
            if _exp0 and spot_price:
                _chain2 = _spy2.option_chain(_exp0)
                _cdf = _chain2.calls[["strike", "impliedVolatility"]].dropna()
                _pdf = _chain2.puts[["strike",  "impliedVolatility"]].dropna()
                _pr  = _pdf.iloc[(_pdf["strike"] - spot_price * 0.95).abs().argsort()[:1]]
                _cr  = _cdf.iloc[(_cdf["strike"] - spot_price * 1.05).abs().argsort()[:1]]
                if len(_pr) and len(_cr):
                    _piv = float(_pr["impliedVolatility"].iloc[0])
                    _civ = float(_cr["impliedVolatility"].iloc[0])
                    if _civ > 0:
                        _skew = _piv / _civ
                        sig3_score   = _clamp_of((1.1 - _skew) / 0.2)
                        skew_display = f"{_skew:.2f} ({'flat' if _skew < 1.05 else ('steep' if _skew > 1.3 else 'moderate')})"
        except Exception:
            pass

    # ── Signal 4: Unusual Activity Bias (weight 1.5) ──────────────────────────
    sig4_score      = 0.0
    unusual_display = "N/A"
    if _s_call_vol and _s_call_oi and _s_put_vol and _s_put_oi:
        try:
            _c_unusual = sum(
                v for v, oi in zip(_s_call_vol, _s_call_oi) if oi > 0 and v / oi > 2.0
            )
            _p_unusual = sum(
                v for v, oi in zip(_s_put_vol, _s_put_oi) if oi > 0 and v / oi > 2.0
            )
            _total_un = _c_unusual + _p_unusual
            if _total_un > 0:
                sig4_score      = _clamp_of((_c_unusual - _p_unusual) / _total_un)
                unusual_display = (
                    f"calls {_c_unusual:,.0f} vs puts {_p_unusual:,.0f} "
                    f"({'call-heavy' if sig4_score > 0.2 else ('put-heavy' if sig4_score < -0.2 else 'mixed')})"
                )
        except Exception:
            pass
    if unusual_display == "N/A":
        try:
            import yfinance as _yf3
            _spy3 = _yf3.Ticker("SPY")
            _exp1 = list(_spy3.options or [])[0] if _spy3.options else None
            if _exp1:
                _chain3 = _spy3.option_chain(_exp1)
                _cdf3 = _chain3.calls[["volume", "openInterest"]].fillna(0)
                _pdf3 = _chain3.puts[["volume",  "openInterest"]].fillna(0)
                _cu = float(_cdf3.loc[_cdf3["openInterest"] > 0].apply(
                    lambda r: r["volume"] if r["volume"] / r["openInterest"] > 2.0 else 0, axis=1).sum())
                _pu = float(_pdf3.loc[_pdf3["openInterest"] > 0].apply(
                    lambda r: r["volume"] if r["volume"] / r["openInterest"] > 2.0 else 0, axis=1).sum())
                _tu = _cu + _pu
                if _tu > 0:
                    sig4_score      = _clamp_of((_cu - _pu) / _tu)
                    unusual_display = (
                        f"calls {_cu:,.0f} vs puts {_pu:,.0f} "
                        f"({'call-heavy' if sig4_score > 0.2 else ('put-heavy' if sig4_score < -0.2 else 'mixed')})"
                    )
        except Exception:
            pass

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scores  = [sig1_score, sig2_score, sig3_score, sig4_score]
    weights = [3.0, 1.5, 2.0, 1.5]
    agg     = float(np.average(scores, weights=weights))
    options_score = int(round((agg + 1.0) * 50))
    options_score = max(0, min(100, options_score))

    if options_score >= 65:
        label       = "Call-Skewed Flow"
        action_bias = "Options market favoring risk-on. Call flow dominant — supportive of long entries."
    elif options_score >= 52:
        label       = "Neutral Flow"
        action_bias = "Options positioning mixed. No strong directional bias in the market."
    elif options_score >= 38:
        label       = "Put-Skewed Flow"
        action_bias = "Elevated put demand. Market hedging above normal — be selective with new longs."
    else:
        label       = "Bearish Hedging"
        action_bias = "Significant defensive positioning in options. Consider reducing risk exposure."

    signal_rows = [
        {"Signal": "P/C Ratio",           "Value": pc_display,      "Score": round(sig1_score, 3), "Direction": _score_to_dir(sig1_score)},
        {"Signal": "Gamma Zone",           "Value": gamma_display,   "Score": round(sig2_score, 3), "Direction": _score_to_dir(sig2_score)},
        {"Signal": "IV Skew (put/call)",   "Value": skew_display,    "Score": round(sig3_score, 3), "Direction": _score_to_dir(sig3_score)},
        {"Signal": "Unusual Activity",     "Value": unusual_display, "Score": round(sig4_score, 3), "Direction": _score_to_dir(sig4_score)},
    ]

    return {
        "options_score": options_score,
        "label":         label,
        "action_bias":   action_bias,
        "signals":       signal_rows,
        "pc_ratio":      round(pc_ratio, 3),
        "raw_score":     round(agg, 3),
    }


def run_quick_options_flow(use_claude: bool = False, model: str | None = None) -> dict | None:
    """Background-safe QIR helper — fetches SPY options data and returns Options Flow context dict.

    No st.* calls. Caller (main thread) writes result to session_state.
    Bypasses @st.cache_data via __wrapped__ — background threads cannot use Streamlit's cache
    after cache_data.clear() is called at the start of QIR.
    Gamma is skipped (scores neutral) to avoid the double-cached _compute_spy_gamma_mode chain.
    """
    import yfinance as yf
    import pandas as pd

    _neutral_flow = {
        "options_score": 50,
        "label":         "Neutral",
        "action_bias":   "Data unavailable — options market closed or feed down",
        "signals":       [],
        "pc_ratio":      1.0,
        "raw_score":     0.0,
        "data_unavailable": True,
    }

    tk = yf.Ticker("SPY")
    hist = tk.history(period="5d", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        return _neutral_flow
    price = float(hist["Close"].iloc[-1])
    expiries = list(tk.options or [])
    if not expiries:
        return _neutral_flow

    selected = expiries[:min(2, len(expiries))]
    strike_map: dict = {}
    chain_errors = []
    for exp in selected:
        try:
            chain = tk.option_chain(exp)
        except Exception as _ce:
            chain_errors.append(f"{exp}: {_ce}")
            continue
        for side, sign in ((chain.calls, 1.0), (chain.puts, -1.0)):
            if side is None or side.empty:
                continue
            subset = side[["strike", "openInterest", "impliedVolatility", "volume"]].copy()
            subset["openInterest"]     = pd.to_numeric(subset["openInterest"],     errors="coerce").fillna(0)
            subset["impliedVolatility"] = pd.to_numeric(subset["impliedVolatility"], errors="coerce").fillna(0.2)
            subset["volume"]           = pd.to_numeric(subset["volume"],           errors="coerce").fillna(0)
            for _, row in subset.iterrows():
                s = float(row["strike"])
                if s not in strike_map:
                    strike_map[s] = {"call_oi": 0.0, "put_oi": 0.0, "call_vol": 0.0, "put_vol": 0.0,
                                     "call_iv": 0.0, "put_iv": 0.0}
                if sign > 0:
                    strike_map[s]["call_oi"]  += float(row["openInterest"])
                    strike_map[s]["call_vol"] += float(row["volume"])
                    strike_map[s]["call_iv"]   = float(row["impliedVolatility"])
                else:
                    strike_map[s]["put_oi"]  += float(row["openInterest"])
                    strike_map[s]["put_vol"] += float(row["volume"])
                    strike_map[s]["put_iv"]   = float(row["impliedVolatility"])

    if not strike_map:
        return _neutral_flow

    strikes = sorted(strike_map.keys())
    spy_chain = {
        "ticker": "SPY",
        "price": price,
        "strikes": strikes,
        "call_oi":  [strike_map[s]["call_oi"]  for s in strikes],
        "put_oi":   [strike_map[s]["put_oi"]   for s in strikes],
        "call_vol": [strike_map[s]["call_vol"] for s in strikes],
        "put_vol":  [strike_map[s]["put_vol"]  for s in strikes],
        "call_iv":  [strike_map[s]["call_iv"]  for s in strikes],
        "put_iv":   [strike_map[s]["put_iv"]   for s in strikes],
        "expiries": selected,
    }
    # Gamma skipped in background thread (doubly cached, unreliable) — scores as neutral
    return _build_options_flow_dashboard(spy_chain, gamma_data=None)
