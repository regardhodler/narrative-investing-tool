"""Current Events — standalone intelligence feed module."""

import streamlit as st

from services.news_feed import (
    fetch_financial_headlines,
    fetch_gist_intel,
    load_news_inbox,
    save_to_inbox,
    clear_inbox,
    sync_telegram_to_inbox,
    headlines_to_text,
    inbox_to_text,
    polymarket_to_text,
)
from utils.theme import COLORS


def _fetch_x_feed_via_grok(queries: list, regime_context: str = "") -> str:
    """Call xAI Responses API with x_search tool. Returns live X context string or empty string."""
    import os, requests as _req
    key = os.getenv("XAI_API_KEY", "")
    if not key:
        return ""
    search_prompt = (
        f"Search X for the most important recent posts about: {', '.join(queries)}. "
        + (f"Current macro regime: {regime_context}. " if regime_context else "")
        + "Summarize the key themes, notable views, and any breaking developments in 3-5 bullet points. "
        "Focus only on macro, financial, and geopolitical content. Strip citation links from output."
    )
    try:
        resp = _req.post(
            "https://api.x.ai/v1/responses",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "grok-4-1-fast-reasoning",
                "input": [{"role": "user", "content": search_prompt}],
                "max_output_tokens": 500,
                "tools": [{"type": "x_search"}],
            },
            timeout=45,
        )
        resp.raise_for_status()
        for item in resp.json().get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        import re
                        return re.sub(r'\[\[\d+\]\]\(https?://\S+\)', '', c["text"]).strip()
        return ""
    except Exception:
        return ""



def _detect_source_conflict(x_content: str, rss_text: str) -> str:
    """Score X feed vs RSS headlines for directional bias. Returns conflict string or ''."""
    if not x_content or not rss_text:
        return ""
    _bull = ["rally", "surge", "rise", "gain", "bull", "higher", "strong", "recover",
             "rebound", "upside", "beat", "growth", "green", "positive"]
    _bear = ["fall", "drop", "crash", "plunge", "bear", "lower", "weak", "sell",
             "decline", "loss", "miss", "recession", "red", "negative", "risk-off"]

    def _score(text: str) -> int:
        t = text.lower()
        return sum(t.count(w) for w in _bull) - sum(t.count(w) for w in _bear)

    _xs = _score(x_content)
    _rs = _score(rss_text)
    if _xs >= 3 and _rs <= -3:
        return f"⚠ Source conflict: X posts lean bullish ({_xs:+d}) but RSS headlines lean bearish ({_rs:+d}) — digest may be blending contradictory signals."
    if _xs <= -3 and _rs >= 3:
        return f"⚠ Source conflict: X posts lean bearish ({_xs:+d}) but RSS headlines lean bullish ({_rs:+d}) — digest may be blending contradictory signals."
    return ""


def run_quick_digest(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Fetches RSS + Gist + inbox, generates digest.
    Stores _current_events_digest to session_state. Does NOT call st.rerun().
    """
    import os
    import streamlit as st
    from datetime import datetime

    sync_telegram_to_inbox()
    headlines = fetch_financial_headlines()
    inbox = load_news_inbox()
    gist = fetch_gist_intel()

    parts = []
    # Inbox first — user-curated intel has highest priority
    inbox_text = inbox_to_text(inbox)
    if inbox_text:
        parts.append("USER FIELD NOTES (highest priority — curated X posts & observations):\n" + inbox_text)
    if gist and gist.get("narrative"):
        parts.append("BOT NARRATIVE ANALYSIS:\n" + gist["narrative"][:800])
    if gist and gist.get("polymarket"):
        pm_text = polymarket_to_text(gist["polymarket"])
        if pm_text:
            parts.append(pm_text)
    hl_text = headlines_to_text(headlines, max_items=15)
    if hl_text:
        parts.append("RSS HEADLINES:\n" + hl_text)

    # FedSpeak RSS — inject recent Fed speeches/testimony as a grounding signal
    try:
        from services.free_data import fetch_fedspeech_rss as _fed_rss
        _speeches = _fed_rss(max_items=5)
        if _speeches:
            _fed_lines = "\n".join(
                f"• [{s['speaker'] or 'Fed Official'}] {s['title']}"
                + (f" ({s['published'][:16]})" if s.get("published") else "")
                for s in _speeches
            )
            parts.append("FEDSPEAK (recent speeches/testimony):\n" + _fed_lines)
    except Exception:
        pass

    if not parts:
        return False

    context = "\n\n".join(parts)
    # Store raw headline text for source breakdown (populated before X feed)
    _raw_hl_text = hl_text or ""
    _raw_inbox_text = inbox_text or ""

    _rc = st.session_state.get("_regime_context") or {}
    _regime_line = ""
    if _rc.get("regime"):
        _regime_line = (
            f"CURRENT MACRO REGIME: {_rc['regime']} "
            f"(score {_rc.get('score', 0.0):+.2f}) | "
            f"Quadrant: {_rc.get('quadrant', 'Unknown')}\n"
            f"Frame your digest within this macro context — note where news confirms "
            f"or contradicts the regime.\n\n"
        )

    prompt = (
        "You are a senior macro research analyst. Based on the following current events, "
        "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
        "dominant narratives, and flags any actionable risks or opportunities. "
        "Prioritize the USER FIELD NOTES section if present — these are hand-curated signals. "
        "Be clinical, specific, and reference actual catalysts.\n\n"
        + _regime_line
        + f"{context[:5000]}"
    )

    digest = None
    _use_cl = use_claude
    _model = model
    _x_injected = False
    _x_content_out = ""

    # Inject X live feed when using Grok (xAI)
    if _use_cl and _model and _model.startswith("grok-") and os.getenv("XAI_API_KEY"):
        _x_queries = ["Federal Reserve monetary policy", "macro economy inflation",
                      "geopolitical risk markets", "financial markets stocks bonds"]
        _rc_x = st.session_state.get("_regime_context") or {}
        _regime_str = f"{_rc_x.get('regime','')} {_rc_x.get('quadrant','')}".strip()
        x_content = _fetch_x_feed_via_grok(_x_queries, _regime_str)
        if x_content:
            parts.insert(0, "LIVE X FEED (real-time macro/financial posts):\n" + x_content)
            context = "\n\n".join(parts)
            prompt = (
                "You are a senior macro research analyst. Based on the following current events, "
                "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
                "dominant narratives, and flags any actionable risks or opportunities. "
                "Prioritize the USER FIELD NOTES section if present — these are hand-curated signals. "
                "Be clinical, specific, and reference actual catalysts.\n\n"
                + _regime_line
                + f"{context[:5000]}"
            )
            _x_injected = True
            _x_content_out = x_content

    if not _use_cl:
        try:
            from groq import Groq
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                resp = Groq(api_key=groq_key).chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=300,
                )
                digest = resp.choices[0].message.content.strip()
        except Exception:
            _use_cl = True
            _model = _model or "grok-4-1-fast-reasoning"

    if _use_cl or not digest:
        _cl_model = _model or "grok-4-1-fast-reasoning"
        try:
            if _cl_model.startswith("grok-") and os.getenv("XAI_API_KEY"):
                from services.claude_client import _call_xai
                digest = _call_xai([{"role": "user", "content": prompt}], _cl_model, 300, 0.2)
            else:
                import anthropic
                msg = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "")).messages.create(
                    model=_cl_model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                digest = msg.content[0].text.strip()
        except Exception:
            return False

    if digest:
        _tier = "👑 Highly Regarded Mode" if (_use_cl and _model == "claude-sonnet-4-6") else ("🧠 Regard Mode" if _use_cl else "⚡ Freeloader Mode")
        _conflict = _detect_source_conflict(_x_content_out, _raw_hl_text)
        return {
            "_current_events_digest": digest,
            "_current_events_digest_ts": datetime.now(),
            "_current_events_engine": _tier,
            "_x_feed_injected": _x_injected,
            "_x_feed_content": _x_content_out,
            "_current_events_conflict": _conflict,
            "_current_events_sources": {
                "x_feed": _x_content_out,
                "headlines": _raw_hl_text,
                "inbox": _raw_inbox_text,
            },
        }
    return None


def _time_ago(ts_str: str) -> str:
    """Convert ISO timestamp to human-readable 'Xh ago'."""
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # Naive datetime — treat as local time, not UTC
            dt = dt.astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        diff = int((now - dt).total_seconds())
        if diff < 60:
            return "just now"
        elif diff < 3600:
            return f"{diff // 60}m ago"
        elif diff < 86400:
            return f"{diff // 3600}h ago"
        else:
            return f"{diff // 86400}d ago"
    except Exception:
        return ts_str[:16]


def render():
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:20px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.08em;margin-bottom:4px;">CURRENT EVENTS</div>'
        f'<div style="height:2px;margin-bottom:16px;'
        f'background:linear-gradient(90deg,{COLORS["bloomberg_orange"]},'
        f'{COLORS["bloomberg_orange"]}44,transparent);border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )

    # ── Telegram field notes sync ─────────────────────────────────────────────
    from services.telegram_client import is_configured as _tg_configured
    _n_new = sync_telegram_to_inbox()
    if _tg_configured():
        _tg_badge_color = "#22c55e" if _n_new > 0 else "#334155"
        _tg_badge_text = f"📱 Telegram — {_n_new} new note{'s' if _n_new != 1 else ''} synced" if _n_new else "📱 Telegram connected"
        st.markdown(
            f'<div style="font-size:10px;color:{_tg_badge_color};margin-bottom:8px;">{_tg_badge_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Bot Intel (Gist) ──────────────────────────────────────────────────────
    gist = fetch_gist_intel()
    if gist:
        updated = _time_ago(gist.get("updated_at", ""))
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
            f'font-size:12px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
            f'letter-spacing:0.06em;margin-bottom:6px;">📡 BOT INTEL'
            f'<span style="font-weight:400;color:{COLORS["text_dim"]};margin-left:8px;">'
            f'updated {updated}</span></div>',
            unsafe_allow_html=True,
        )

        # Narrative summary
        narrative = gist.get("narrative", "").strip()
        if narrative:
            with st.expander("🔍 Market Narrative Analysis", expanded=True):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;color:{COLORS["text"]};white-space:pre-wrap;">'
                    f'{narrative}</div>',
                    unsafe_allow_html=True,
                )

        # Top alerts from X
        top_alerts = gist.get("top_alerts", [])
        if top_alerts:
            with st.expander(f"🚨 High-Impact X Posts ({len(top_alerts)})", expanded=True):
                for a in top_alerts:
                    impact = a.get("impact", 0)
                    sentiment = a.get("sentiment", "Neutral")
                    sent_color = {"Bullish": COLORS.get("positive", "#00c853"), "Bearish": COLORS.get("negative", "#ff3d00")}.get(sentiment, COLORS["text_dim"])
                    summary = a.get("summary", "")
                    author = a.get("author", "")
                    url = a.get("url", "")
                    ts_raw = a.get("ts", "")
                    ts_disp = _time_ago(ts_raw) if ts_raw else ""
                    category = a.get("category", "")
                    st.markdown(
                        f'<div style="border:1px solid {COLORS["border"]};border-radius:4px;'
                        f'padding:8px 12px;margin-bottom:6px;background:{COLORS["surface"]};">'
                        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:11px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
                        f'margin-bottom:4px;">[{impact}/10] '
                        f'<span style="color:{sent_color};">{sentiment}</span>'
                        f' · <span style="color:{COLORS["text_dim"]};">{category}</span>'
                        f' · <span style="color:{COLORS["text_dim"]};">{ts_disp}</span></div>'
                        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:12px;color:{COLORS["text"]};margin-bottom:4px;">{summary}</div>'
                        f'<a href="{url}" target="_blank" style="font-family:\'JetBrains Mono\','
                        f'Consolas,monospace;font-size:11px;color:{COLORS["accent"]};'
                        f'text-decoration:none;">@{author} ↗</a>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Polymarket
        polymarket = gist.get("polymarket", [])
        if polymarket:
            with st.expander(f"🎲 Polymarket Macro Odds ({len(polymarket)} markets)"):
                for m in polymarket:
                    prob = m.get("probability")
                    prob_str = f"{prob:.0%}" if prob is not None else "N/A"
                    prob_pct = int(prob * 100) if prob is not None else 0
                    bar_color = COLORS.get("positive", "#00c853") if prob_pct >= 50 else COLORS.get("negative", "#ff3d00")
                    question = m.get("question", "")
                    url = m.get("url", "#")
                    st.markdown(
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:11px;color:{COLORS["text"]};margin-bottom:3px;">'
                        f'<a href="{url}" target="_blank" style="color:{COLORS["text"]};text-decoration:none;">{question}</a></div>'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="flex:1;height:6px;background:{COLORS["border"]};border-radius:3px;overflow:hidden;">'
                        f'<div style="width:{prob_pct}%;height:100%;background:{bar_color};border-radius:3px;"></div>'
                        f'</div>'
                        f'<span style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:12px;font-weight:700;color:{bar_color};min-width:40px;">{prob_str}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)
    else:
        st.info("Bot intel not configured. Set `NEWS_GIST_RAW_URL` in your `.env` to connect the financial-news-bot.")

    # ── RSS Headlines ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:12px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.06em;margin-bottom:6px;">📰 RSS HEADLINES</div>',
        unsafe_allow_html=True,
    )

    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("↻ Refresh", key="ce_refresh_rss"):
            fetch_financial_headlines.clear()
            st.rerun()

    headlines = fetch_financial_headlines()
    if headlines:
        for h in headlines[:20]:
            source = h.get("source", "")
            title = h.get("title", "")
            date = h.get("date", "")
            url = h.get("url", "#")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                f'padding:5px 0;border-bottom:1px solid {COLORS["border"]}22;">'
                f'<div style="flex:1;">'
                f'<span style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:10px;'
                f'color:{COLORS["bloomberg_orange"]};font-weight:700;margin-right:8px;">{source}</span>'
                f'<a href="{url}" target="_blank" style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                f'font-size:12px;color:{COLORS["text"]};text-decoration:none;">{title}</a>'
                f'</div>'
                f'<span style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:10px;'
                f'color:{COLORS["text_dim"]};white-space:nowrap;margin-left:12px;">{date}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No RSS headlines loaded — feeds may be unavailable.")

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

    # ── Manual Inbox ─────────────────────────────────────────────────────────
    inbox = load_news_inbox()
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:12px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.06em;margin-bottom:6px;">📥 INBOX'
        f'<span style="font-weight:400;color:{COLORS["text_dim"]};margin-left:8px;">'
        f'{len(inbox)} item(s) — manual pastes &amp; Telegram</span></div>',
        unsafe_allow_html=True,
    )

    if inbox:
        for item in reversed(inbox[-10:]):
            ts = _time_ago(item.get("ts", ""))
            src = item.get("source", "manual")
            text = item.get("text", "")
            st.markdown(
                f'<div style="border:1px solid {COLORS["border"]};border-radius:4px;'
                f'padding:8px 12px;margin-bottom:4px;background:{COLORS["surface"]};">'
                f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:10px;'
                f'color:{COLORS["text_dim"]};margin-bottom:4px;">{src} · {ts}</div>'
                f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
                f'color:{COLORS["text"]};">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear Inbox", key="ce_clear_inbox"):
            clear_inbox()
            st.rerun()
    else:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};">No inbox items. Paste a note below or forward to your Telegram bot.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📝 Add Note / Paste X Post"):
        note_text = st.text_area(
            "Paste text, X post, or note here",
            key="ce_note_input",
            height=100,
            label_visibility="collapsed",
            placeholder="Paste an X post, article excerpt, or any market note...",
        )
        if st.button("Save to Inbox", key="ce_save_note", type="primary"):
            if note_text.strip():
                save_to_inbox(note_text.strip(), source="manual")
                st.success("Saved.")
                st.rerun()
            else:
                st.warning("Nothing to save.")

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

    # ── AI Digest ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:12px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.06em;margin-bottom:8px;">🧠 AI NEWS DIGEST</div>',
        unsafe_allow_html=True,
    )

    # Engine selector
    engine = st.radio(
        "Engine",
        ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"],
        key="ce_digest_engine",
        horizontal=True,
        label_visibility="collapsed",
    )
    from utils.ai_tier import MODEL_HINT_HTML
    st.markdown(MODEL_HINT_HTML, unsafe_allow_html=True)
    _rec_map = {
        "⚡ Freeloader Mode": "Daily routine check — fast, free. Use when markets are calm and you just want a quick brief.",
        "🧠 Regard Mode": "Active trading day — Grok 4.1 reasoning + 🐦 live X/Twitter feed for real-time macro posts. Best for market-moving days.",
        "👑 Highly Regarded Mode": "High-conviction sessions — Sonnet reads macro nuance best. Use before running Valuation or Portfolio when volatility is elevated or a major catalyst is live.",
    }
    st.caption(f"💡 {_rec_map.get(engine, '')}")

    existing_digest = st.session_state.get("_current_events_digest", "")
    existing_ts = st.session_state.get("_current_events_digest_ts", "")
    if existing_digest:
        _rc_disp = st.session_state.get("_regime_context") or {}
        _regime_badge = ""
        if _rc_disp.get("regime"):
            _badge_color = {"Risk-On": "#22c55e", "Risk-Off": "#ef4444"}.get(_rc_disp["regime"], "#f59e0b")
            _regime_badge = (
                f'<span style="background:{_badge_color}22;color:{_badge_color};'
                f'font-size:10px;font-weight:700;padding:1px 6px;border-radius:3px;'
                f'margin-left:8px;letter-spacing:0.06em;">'
                f'{_rc_disp["regime"]} {_rc_disp.get("score", 0.0):+.2f}</span>'
            )
        _x_injected = st.session_state.get("_x_feed_injected", False)
        _x_badge = (
            '<span style="background:#1a1a2e;color:#1d9bf0;border:1px solid #1d9bf044;'
            'font-size:10px;font-weight:700;padding:1px 7px;border-radius:3px;'
            'margin-left:8px;letter-spacing:0.04em;">𝕏 Live Feed ✓</span>'
            if _x_injected else
            '<span style="background:#1a1a1a;color:#555;border:1px solid #33333344;'
            'font-size:10px;padding:1px 7px;border-radius:3px;margin-left:8px;">'
            '𝕏 RSS only</span>'
        )
        st.markdown(
            f'<div style="border:1px solid {COLORS["bloomberg_orange"]}44;border-radius:4px;'
            f'padding:10px 14px;background:{COLORS["surface"]};margin-bottom:8px;">'
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
            f'color:{COLORS["text_dim"]};margin-bottom:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
            f'Generated {_time_ago(existing_ts.isoformat() if hasattr(existing_ts, "isoformat") else existing_ts) if existing_ts else ""}'
            f'{_regime_badge}{_x_badge}</div>'
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:13px;'
            f'color:{COLORS["text"]};line-height:1.6;">{existing_digest}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Contradiction warning banner
        _conflict = st.session_state.get("_current_events_conflict", "")
        if _conflict:
            st.markdown(
                f'<div style="background:#2d1a00;border-left:4px solid #f97316;'
                f'border-radius:4px;padding:8px 14px;margin-bottom:8px;">'
                f'<span style="color:#f97316;font-size:11px;font-weight:700;">{_conflict}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Verify Sources expander — X feed vs RSS side by side
        _sources = st.session_state.get("_current_events_sources") or {}
        _src_x  = _sources.get("x_feed", "")
        _src_hl = _sources.get("headlines", "")
        _src_in = _sources.get("inbox", "")
        if _src_x or _src_hl or _src_in:
            with st.expander("🔍 Verify Sources — check digest claims against raw inputs", expanded=False):
                _sc1, _sc2 = st.columns(2)
                with _sc1:
                    if _src_x:
                        st.markdown(
                            f'<div style="font-size:10px;font-weight:700;color:#1d9bf0;'
                            f'letter-spacing:0.08em;margin-bottom:6px;">𝕏 LIVE X FEED</div>'
                            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;color:{COLORS["text_dim"]};line-height:1.7;'
                            f'white-space:pre-wrap;">{_src_x}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No X feed — ran in Freeloader or Highly Regarded Mode")
                with _sc2:
                    if _src_hl:
                        st.markdown(
                            f'<div style="font-size:10px;font-weight:700;color:#94a3b8;'
                            f'letter-spacing:0.08em;margin-bottom:6px;">📰 RSS HEADLINES</div>'
                            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;color:{COLORS["text_dim"]};line-height:1.7;'
                            f'white-space:pre-wrap;">{_src_hl[:1500]}</div>',
                            unsafe_allow_html=True,
                        )
                if _src_in:
                    st.markdown(
                        f'<div style="font-size:10px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
                        f'letter-spacing:0.08em;margin:8px 0 4px;">📂 YOUR FIELD NOTES (highest priority)</div>'
                        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                        f'font-size:11px;color:{COLORS["text_dim"]};line-height:1.7;">{_src_in}</div>',
                        unsafe_allow_html=True,
                    )

    if st.button("🗞 Generate News Digest", key="ce_gen_digest", type="primary"):
        with st.spinner("Generating digest..."):
            _run_digest(headlines, inbox, gist, engine)


def _run_digest(headlines, inbox, gist, engine: str):
    """Build context and call AI for news digest."""
    from datetime import datetime

    parts = []

    # Inbox first — user-curated intel has highest priority
    inbox_text = inbox_to_text(inbox)
    if inbox_text:
        parts.append("USER FIELD NOTES (highest priority — curated X posts & observations):\n" + inbox_text)

    if gist and gist.get("narrative"):
        parts.append("BOT NARRATIVE ANALYSIS:\n" + gist["narrative"][:800])

    if gist and gist.get("polymarket"):
        pm_text = polymarket_to_text(gist["polymarket"])
        if pm_text:
            parts.append(pm_text)

    hl_text = headlines_to_text(headlines, max_items=15)
    if hl_text:
        parts.append("RSS HEADLINES:\n" + hl_text)

    if not parts:
        st.warning("No content to digest. Refresh RSS or add inbox items first.")
        return

    context = "\n\n".join(parts)
    _raw_hl_text = hl_text or ""
    _raw_inbox_text = inbox_to_text(inbox) or ""

    _rc = st.session_state.get("_regime_context") or {}
    _regime_line = ""
    if _rc.get("regime"):
        _regime_line = (
            f"CURRENT MACRO REGIME: {_rc['regime']} "
            f"(score {_rc.get('score', 0.0):+.2f}) | "
            f"Quadrant: {_rc.get('quadrant', 'Unknown')}\n"
            f"Frame your digest within this macro context — note where news confirms "
            f"or contradicts the regime.\n\n"
        )

    prompt = (
        "You are a senior macro research analyst. Based on the following current events, "
        "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
        "dominant narratives, and flags any actionable risks or opportunities. "
        "Prioritize the USER FIELD NOTES section if present — these are hand-curated signals. "
        "Be clinical, specific, and reference actual catalysts.\n\n"
        + _regime_line
        + f"{context[:5000]}"
    )

    _tier_model = {
        "⚡ Freeloader Mode": None,
        "🧠 Regard Mode": "grok-4-1-fast-reasoning",
        "👑 Highly Regarded Mode": "claude-sonnet-4-6",
    }
    cl_model = _tier_model.get(engine)
    use_claude = cl_model is not None
    digest = None

    if not use_claude:
        try:
            import os
            from groq import Groq
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                digest = resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Groq failed ({e}), falling back to Regard Mode...")
            use_claude = True
            cl_model = "grok-4-1-fast-reasoning"

    # Inject X live feed when Regard Mode (Grok)
    import os
    if cl_model and cl_model.startswith("grok-") and os.getenv("XAI_API_KEY"):
        _x_queries = ["Federal Reserve monetary policy", "macro economy inflation",
                      "geopolitical risk markets", "financial markets stocks bonds"]
        _rc_x = st.session_state.get("_regime_context") or {}
        _regime_str = f"{_rc_x.get('regime','')} {_rc_x.get('quadrant','')}".strip()
        x_content = _fetch_x_feed_via_grok(_x_queries, _regime_str)
        if x_content:
            parts.insert(0, "LIVE X FEED (real-time macro/financial posts):\n" + x_content)
            context = "\n\n".join(parts)
            prompt = (
                "You are a senior macro research analyst. Based on the following current events, "
                "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
                "dominant narratives, and flags any actionable risks or opportunities. "
                "Prioritize the USER FIELD NOTES section if present — these are hand-curated signals. "
                "Be clinical, specific, and reference actual catalysts.\n\n"
                + _regime_line
                + f"{context[:5000]}"
            )
            st.session_state["_x_feed_injected"] = True
            st.session_state["_x_feed_content"] = x_content
        else:
            st.session_state.pop("_x_feed_injected", None)
            st.session_state.pop("_x_feed_content", None)

    if use_claude or not digest:
        try:
            if cl_model and cl_model.startswith("grok-") and os.getenv("XAI_API_KEY"):
                from services.claude_client import _call_xai
                digest = _call_xai([{"role": "user", "content": prompt}], cl_model, 300, 0.2)
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                msg = client.messages.create(
                    model=cl_model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                digest = msg.content[0].text.strip()
        except Exception as e:
            st.error(f"AI digest failed: {e}")
            return

    if digest:
        _x_out = st.session_state.get("_x_feed_content", "")
        st.session_state["_current_events_digest"] = digest
        st.session_state["_current_events_digest_ts"] = datetime.now()
        st.session_state["_current_events_engine"] = engine
        st.session_state["_current_events_conflict"] = _detect_source_conflict(_x_out, _raw_hl_text)
        st.session_state["_current_events_sources"] = {
            "x_feed": _x_out,
            "headlines": _raw_hl_text,
            "inbox": _raw_inbox_text,
        }
        st.rerun()
