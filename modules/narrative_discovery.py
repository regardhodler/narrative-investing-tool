import plotly.graph_objects as go
import streamlit as st

from services.trends_client import (
    get_interest_over_time,
    get_interest_over_time_multi,
    get_trending_searches,
    get_yahoo_trending_tickers,
)
from services.claude_client import classify_narrative, describe_company, group_tickers_by_narrative
from services.market_data import fetch_batch_safe
from services.sec_client import get_company_info, search_ticker_by_name
from utils.session import get_ticker, set_narrative, set_ticker
from utils.watchlist import add_to_watchlist, is_in_watchlist
from utils.theme import COLORS, apply_dark_layout

ASSET_CLASSES = {
    "Equities": None,  # sentinel — use Yahoo trending
    "Commodities": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "WTI Crude",
        "NG=F": "Natural Gas", "HG=F": "Copper",
        "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans",
    },
    "Bonds": {
        "TLT": "20Y+ Treasury", "IEF": "10Y Treasury", "SHY": "2Y Treasury",
        "LQD": "IG Corporate", "HYG": "High Yield Corp",
        "TIP": "TIPS", "AGG": "US Agg Bond",
    },
    "Currencies": {
        "UUP": "USD Bull", "FXE": "Euro", "FXY": "Yen",
        "FXB": "GBP", "FXA": "AUD", "FXC": "CAD",
    },
}

_ASSET_BADGE = {
    "Commodities": ("CMDTY", COLORS["orange"]),
    "Bonds": ("BOND", COLORS["blue"]),
    "Currencies": ("FX", COLORS["yellow"]),
}

# Lookup: ticker → asset class name (for non-equity classes)
_TICKER_TO_CLASS = {}
for _cls, _tickers in ASSET_CLASSES.items():
    if _tickers is not None:
        for _t in _tickers:
            _TICKER_TO_CLASS[_t] = _cls

_NON_FINANCIAL_KEYWORDS = {
    "nfl", "nba", "mlb", "nhl", "ufc", "wwe", "espn", "super bowl",
    "playoff", "draft pick", "touchdown", "goalkeeper", "premier league",
    "champions league", "world cup", "olympics", "quarterback",
    "kardashian", "taylor swift", "beyonce", "drake", "kanye",
    "bachelor", "bachelorette", "survivor", "big brother",
    "grammys", "oscars", "emmys", "golden globes",
    "tiktok trend", "meme", "viral video", "celebrity",
    "wordle", "fortnite", "minecraft",
}


def render():
    st.header("NARRATIVE DISCOVERY")

    mode = st.radio("Mode", ["Manual", "Auto — Trending"], horizontal=True)

    if mode == "Auto — Trending":
        _render_auto()
    else:
        _render_manual()


def _fetch_curated_assets(ticker_dict: dict[str, str], asset_class: str) -> list[dict]:
    """Fetch price data for curated non-equity tickers and return in trending-compatible shape."""
    snapshots = fetch_batch_safe(ticker_dict, period="5d")
    items = []
    for idx, (ticker, label) in enumerate(ticker_dict.items()):
        snap = snapshots.get(ticker)
        pct = snap.pct_change_1d if snap and snap.pct_change_1d is not None else 0.0
        items.append({
            "symbol": ticker,
            "name": label,
            "pct_change": pct,
            "buzz_rank": idx,
            "asset_class": asset_class,
        })
    items.sort(key=lambda x: abs(x.get("pct_change", 0)), reverse=True)
    return items


def _render_auto():
    # --- Asset class filter ---
    selected_classes = st.multiselect(
        "Asset Classes", list(ASSET_CLASSES.keys()),
        default=["Equities"], key="asset_class_filter",
    )

    if not selected_classes:
        st.info("Select at least one asset class to discover trending assets.")
        return

    all_trending = []

    # --- Equities: Yahoo Finance trending tickers ---
    if "Equities" in selected_classes:
        with st.spinner("Fetching trending tickers from Yahoo Finance..."):
            yf_trending = get_yahoo_trending_tickers()
        if yf_trending:
            for item in yf_trending:
                item["asset_class"] = "Equities"
            all_trending.extend(yf_trending)

    # --- Non-equity asset classes ---
    for cls in selected_classes:
        if cls == "Equities":
            continue
        ticker_dict = ASSET_CLASSES.get(cls)
        if ticker_dict:
            with st.spinner(f"Fetching {cls} data..."):
                all_trending.extend(_fetch_curated_assets(ticker_dict, cls))

    if not all_trending:
        st.warning("No data available for the selected asset classes.")
        return

    yf_trending = all_trending  # keep variable name for downstream compat

    if yf_trending:
        # Build a lookup from symbol → item
        ticker_lookup = {item["symbol"]: item for item in yf_trending}

        # Group tickers by narrative theme via AI
        import json as _json
        tickers_for_grouping = _json.dumps(
            [{"symbol": t["symbol"], "name": t["name"]} for t in yf_trending]
        )
        with st.spinner("Grouping tickers by narrative..."):
            narrative_groups = group_tickers_by_narrative(tickers_for_grouping)

        from datetime import datetime
        st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 1H")

        if narrative_groups:
            st.subheader(f"{len(yf_trending)} Trending Tickers · {len(narrative_groups)} Narratives")

            for g_idx, group in enumerate(narrative_groups):
                narrative_title = group.get("narrative", "Market Movers")
                description = group.get("description", "")
                group_tickers = group.get("tickers", [])

                # Narrative header
                st.markdown(
                    f'<div style="background:{COLORS["surface"]}; padding:12px 16px; '
                    f'border-radius:8px; border-left:4px solid {COLORS["accent"]}; '
                    f'margin:16px 0 8px 0;">'
                    f'<div style="color:{COLORS["accent"]}; font-size:18px; font-weight:700;">'
                    f'{narrative_title}</div>'
                    f'<div style="color:{COLORS["text_dim"]}; font-size:13px; margin-top:2px;">'
                    f'{description}</div></div>',
                    unsafe_allow_html=True,
                )

                # Ticker cards in columns
                group_items = [
                    ticker_lookup[sym] for sym in group_tickers
                    if sym in ticker_lookup
                ]
                if not group_items:
                    continue

                cols = st.columns(min(3, len(group_items)))
                for i, item in enumerate(group_items):
                    col = cols[i % len(cols)]
                    with col:
                        with st.container(border=True):
                            # Asset class badge
                            _badge = _ASSET_BADGE.get(item.get("asset_class"))
                            if _badge:
                                st.markdown(
                                    f'<span style="background:{_badge[1]}22;color:{_badge[1]};'
                                    f'font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;'
                                    f'letter-spacing:0.08em;font-family:\'JetBrains Mono\',monospace;">'
                                    f'{_badge[0]}</span>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**{item['name']}**")
                            st.code(item["symbol"], language=None)
                            # Intraday % change
                            pct = item.get("pct_change")
                            if pct is not None:
                                pct_color = COLORS.get("green", "#00d4aa") if pct >= 0 else COLORS.get("red", "#ff4d4d")
                                arrow = "▲" if pct >= 0 else "▼"
                                st.markdown(
                                    f'<span style="color:{pct_color};font-weight:700;font-size:15px;">'
                                    f'{arrow} {pct:+.2f}%</span>',
                                    unsafe_allow_html=True,
                                )
                            # Buzz star rating (top of trending list = 5 stars)
                            buzz_rank = item.get("buzz_rank", 99)
                            total = len(yf_trending)
                            stars = max(1, min(5, 5 - int((buzz_rank - 1) / max(1, total / 5))))
                            st.markdown(
                                f'<span style="color:#FFD700;font-size:16px;" title="Buzz: {stars}/5">'
                                f'{"★" * stars}{"☆" * (5 - stars)}</span>',
                                unsafe_allow_html=True,
                            )
                            if st.button("Select", key=f"yf_select_{g_idx}_{i}", type="primary"):
                                set_narrative(narrative_title)
                                set_ticker(item["symbol"])
                                st.rerun()
                            if st.button("Watch", key=f"wl_add_{g_idx}_{i}"):
                                from utils.watchlist import add_to_watchlist
                                add_to_watchlist(item["symbol"], narrative_title)
                                st.rerun()
        else:
            # Fallback: flat list if grouping fails
            st.subheader(f"{len(yf_trending)} Trending Tickers")
            cols = st.columns(3)
            for i, item in enumerate(yf_trending):
                col = cols[i % 3]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}**")
                        st.code(item["symbol"], language=None)
                        pct = item.get("pct_change")
                        if pct is not None:
                            pct_color = COLORS.get("green", "#00d4aa") if pct >= 0 else COLORS.get("red", "#ff4d4d")
                            arrow = "▲" if pct >= 0 else "▼"
                            st.markdown(
                                f'<span style="color:{pct_color};font-weight:700;font-size:15px;">'
                                f'{arrow} {pct:+.2f}%</span>',
                                unsafe_allow_html=True,
                            )
                        buzz_rank = item.get("buzz_rank", 99)
                        total = len(yf_trending)
                        stars = max(1, min(5, 5 - int((buzz_rank - 1) / max(1, total / 5))))
                        st.markdown(
                            f'<span style="color:#FFD700;font-size:16px;" title="Buzz: {stars}/5">'
                            f'{"★" * stars}{"☆" * (5 - stars)}</span>',
                            unsafe_allow_html=True,
                        )
                        if st.button("Select", key=f"yf_select_{i}", type="primary"):
                            set_narrative(item["name"])
                            set_ticker(item["symbol"])
                            st.rerun()
                        if st.button("Watch", key=f"wl_add_flat_{i}"):
                            from utils.watchlist import add_to_watchlist
                            add_to_watchlist(item["symbol"], item["name"])
                            st.rerun()

    # --- Show overview for selected ticker ---
    active = get_ticker()
    if active:
        _render_company_overview(active)

    # --- Trending search interest (auto, no click needed) ---
    if yf_trending:
        _render_trending_interest(yf_trending)

    # --- Google Trends supplement (filtered for financial topics) ---
    with st.expander("Google Trends (filtered)", expanded=False):
        try:
            with st.spinner("Fetching Google Trends..."):
                topics = get_trending_searches()
        except Exception:
            st.caption("Google Trends unavailable — service may be temporarily down.")
            topics = []

        if not topics:
            st.caption("No trending topics from Google Trends.")
        else:
            st.caption(
                f"{len(topics)} trending topics · filtering for financial relevance..."
            )
            classified = []
            for topic in topics[:20]:
                if any(kw in topic.lower() for kw in _NON_FINANCIAL_KEYWORDS):
                    continue
                result = classify_narrative(topic)
                if result.get("market_relevant"):
                    result["topic"] = topic
                    classified.append(result)

            if not classified:
                st.info("No market-relevant narratives in current Google Trends.")
            else:
                st.subheader(f"{len(classified)} Market-Relevant Narratives")
                cols = st.columns(3)
                for i, item in enumerate(classified):
                    col = cols[i % 3]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{item['topic']}**")
                            st.caption(item.get("sector", ""))
                            st.markdown(
                                item.get("thesis", ""), unsafe_allow_html=False
                            )
                            tickers = item.get("suggested_tickers", [])
                            if tickers:
                                st.code(", ".join(tickers), language=None)
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Track", key=f"gt_track_{i}"):
                                    set_narrative(item["topic"])
                                    st.rerun()
                            with col_b:
                                if tickers and st.button(
                                    tickers[0], key=f"gt_ticker_{i}"
                                ):
                                    set_narrative(item["topic"])
                                    set_ticker(tickers[0])
                                    st.rerun()


def _render_manual():
    tab_ticker, tab_narrative = st.tabs(["Ticker Symbol", "Narrative Keyword"])

    with tab_ticker:
        search_input = st.text_input(
            "Search by ticker or company name",
            placeholder="e.g. AAPL, Apple, TSLA, Tesla Inc",
        ).strip()

        if st.button("Search", type="primary", key="set_ticker_btn") and search_input:
            st.session_state["ticker_search_query"] = search_input

        query = st.session_state.get("ticker_search_query", "")
        if query:
            # Quick-set if it looks like a ticker (short alpha string)
            if query.isalpha() and len(query) <= 5:
                if st.button(f"Set **{query.upper()}** as active ticker", key="direct_set_ticker"):
                    set_ticker(query.upper())
                    st.rerun()

            # Company name search results
            results = search_ticker_by_name(query)
            if results:
                st.caption(f"Matches ({len(results)}):")
                for r in results:
                    c_name, c_btn = st.columns([3, 1])
                    c_name.markdown(f"**{r['ticker']}** — {r['name']}")
                    if c_btn.button("Select", key=f"co_sel_{r['ticker']}"):
                        set_ticker(r["ticker"])
                        st.rerun()
            elif not (query.isalpha() and len(query) <= 5):
                st.caption("No matching companies found.")

        # Show confirmed ticker info + watchlist + overview
        active = get_ticker()
        if active:
            st.success(f"Active ticker: **{active}**")
            if is_in_watchlist(active):
                st.caption(f"**{active}** is on your watchlist")
            elif st.button("+ Watch", key="watch_ticker_manual"):
                add_to_watchlist(active, st.session_state.get("active_narrative", ""))
                st.rerun()
            _render_company_overview(active)
            try:
                _render_interest_chart(active, key_suffix="ticker")
            except Exception:
                st.caption("Search interest data unavailable.")

    with tab_narrative:
        keyword = st.text_input(
            "Enter a narrative keyword",
            placeholder="e.g. nuclear energy, weight loss drugs, AI chips",
        )
        if st.button("Analyze", type="primary") and keyword:
            set_narrative(keyword)
            result = classify_narrative(keyword)
            st.session_state["narrative_result"] = result
            st.session_state["narrative_keyword"] = keyword

        # Show results persistently after analysis
        result = st.session_state.get("narrative_result")
        kw = st.session_state.get("narrative_keyword", "")
        if result:
            if result.get("market_relevant"):
                st.success(f"Narrative set: **{kw}**")
                st.markdown(f"**Sector:** {result.get('sector', 'N/A')}")
                st.markdown(f"**Thesis:** {result.get('thesis', '')}")
                tickers = result.get("suggested_tickers", [])
                if tickers:
                    st.markdown(f"**Suggested tickers:** {', '.join(tickers)}")
                    selected = st.selectbox("Set active ticker", tickers, key="narrative_ticker_select")
                    if st.button("Confirm Ticker", key="confirm_narrative_ticker") and selected:
                        set_ticker(selected)
                        st.rerun()
            else:
                st.info(
                    "Topic classified as not directly market-relevant, but narrative is set."
                )

        # Show company overview and interest chart for confirmed ticker
        active = get_ticker()
        if active and result:
            if is_in_watchlist(active):
                st.caption(f"**{active}** is on your watchlist")
            elif st.button("+ Watch", key="watch_narrative_manual"):
                add_to_watchlist(active, kw)
                st.rerun()
            _render_company_overview(active)
            try:
                _render_interest_chart(active, key_suffix="narrative")
            except Exception:
                st.caption("Search interest data unavailable.")


def _render_trending_interest(trending: list[dict]):
    """Auto-render a multi-line interest chart for the top trending tickers."""
    # Take first 5 (pytrends limit)
    top = trending[:5]
    symbols = tuple(item["symbol"] for item in top)

    st.subheader("Trending Search Interest")
    timeframe = st.select_slider(
        "Timeframe",
        options=["1M", "3M", "6M", "1Y", "YTD"],
        value="3M",
        key="interest_tf_trending",
    )

    with st.spinner("Fetching search interest for trending tickers..."):
        df = get_interest_over_time_multi(symbols, timeframe)

    if df.empty:
        st.caption("No interest data available for trending tickers.")
        return

    line_colors = [COLORS["accent"], COLORS["blue"], COLORS["yellow"], COLORS["red"], "#AB47BC"]

    fig = go.Figure()
    for i, sym in enumerate(symbols):
        if sym not in df.columns:
            continue
        color = line_colors[i % len(line_colors)]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[sym],
                mode="lines",
                name=sym,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d}}: %{{y}}<extra></extra>",
            )
        )

    apply_dark_layout(
        fig,
        title="Search Interest: Top Trending Tickers",
        yaxis_title="Relative Interest (0–100)",
        xaxis_title="",
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: Google Trends · comparing relative search interest across tickers")


def _render_company_overview(ticker: str):
    """Show company description and narrative for a ticker."""
    # Route non-equity assets to simplified overview
    if ticker in _TICKER_TO_CLASS:
        _render_asset_overview(ticker)
        return

    with st.spinner(f"Looking up {ticker}..."):
        company_info = get_company_info(ticker)

    if not company_info:
        st.warning(f"Could not fetch company info for {ticker} from SEC. The SEC API may be temporarily unavailable.")
        return

    with st.spinner("Generating company overview..."):
        overview = describe_company(
            company_info["name"],
            ticker,
            company_info["sic_description"],
        )

    with st.container(border=True):
        st.subheader(f"{company_info['name']}  ·  {ticker}")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            if overview.get("description"):
                st.markdown(overview["description"])
            else:
                st.caption(company_info["sic_description"])
        with col_b:
            st.markdown(f"**Sector:** {overview.get('sector', company_info['sic_description'])}")
            if company_info["state"]:
                st.markdown(f"**Incorporated:** {company_info['state']}")
            if company_info["exchanges"]:
                st.markdown(f"**Exchange:** {', '.join(company_info['exchanges'])}")

        if overview.get("narrative"):
            st.info(f"**Narrative:** {overview['narrative']}")


def _render_asset_overview(ticker: str):
    """Simplified overview for non-equity assets (commodities, bonds, currencies)."""
    asset_class = _TICKER_TO_CLASS.get(ticker, "Unknown")
    # Find label from ASSET_CLASSES
    label = ticker
    class_dict = ASSET_CLASSES.get(asset_class, {})
    if class_dict:
        label = class_dict.get(ticker, ticker)

    with st.spinner(f"Fetching {label} data..."):
        snapshots = fetch_batch_safe({ticker: label}, period="3mo")

    snap = snapshots.get(ticker)
    badge_info = _ASSET_BADGE.get(asset_class, (asset_class.upper(), COLORS["accent"]))

    with st.container(border=True):
        st.markdown(
            f'<span style="background:{badge_info[1]}22;color:{badge_info[1]};'
            f'font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'
            f'letter-spacing:0.08em;">{badge_info[0]}</span>',
            unsafe_allow_html=True,
        )
        st.subheader(f"{label}  ·  {ticker}")

        if snap and snap.latest_price is not None:
            col_price, col_1d, col_5d, col_30d = st.columns(4)
            with col_price:
                st.metric("Price", f"${snap.latest_price:,.2f}")
            with col_1d:
                val = snap.pct_change_1d
                st.metric("1D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)
            with col_5d:
                val = snap.pct_change_5d
                st.metric("5D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)
            with col_30d:
                val = snap.pct_change_30d
                st.metric("30D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)

            # Mini sparkline
            if snap.series is not None and not snap.series.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=snap.series.index,
                    y=snap.series.values,
                    mode="lines",
                    line=dict(color=COLORS["accent"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 170, 0.1)",
                    hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>",
                ))
                apply_dark_layout(fig, title=f"{label} — 3 Month",
                                  margin=dict(l=40, r=20, t=40, b=30))
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price data for {ticker}.")


def _render_interest_chart(keyword: str, key_suffix: str = "default"):
    """Render a Google Trends interest-over-time area chart for a keyword."""
    st.subheader("Search Interest")
    timeframe = st.select_slider(
        "Timeframe",
        options=["1M", "3M", "6M", "1Y", "YTD"],
        value="3M",
        key=f"interest_tf_{key_suffix}",
    )

    with st.spinner("Fetching search interest..."):
        df = get_interest_over_time(keyword, timeframe)

    if df.empty:
        st.caption("No interest data available for this keyword.")
        return

    peak = df["interest"].max()
    peak_date = df.loc[df["interest"].idxmax(), "date"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["interest"],
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.15)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Interest: %{y}<extra></extra>",
        )
    )

    # Peak annotation
    fig.add_annotation(
        x=peak_date,
        y=peak,
        text=f"Peak: {peak}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["accent"],
        font=dict(color=COLORS["accent"], size=11),
        ax=0,
        ay=-30,
    )

    apply_dark_layout(
        fig,
        title=f"Google Trends Interest: {keyword}",
        yaxis_title="Relative Interest (0–100)",
        xaxis_title="",
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: Google Trends · 100 = peak search interest in the selected period")
