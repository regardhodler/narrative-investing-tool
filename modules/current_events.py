"""Current Events — standalone intelligence feed module."""

import streamlit as st

from services.news_feed import (
    fetch_financial_headlines,
    fetch_gist_intel,
    load_news_inbox,
    save_to_inbox,
    clear_inbox,
    headlines_to_text,
    inbox_to_text,
    polymarket_to_text,
)
from utils.theme import COLORS


def run_quick_digest(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Fetches RSS + Gist + inbox, generates digest.
    Stores _current_events_digest to session_state. Does NOT call st.rerun().
    """
    import os
    import streamlit as st
    from datetime import datetime

    headlines = fetch_financial_headlines()
    inbox = load_news_inbox()
    gist = fetch_gist_intel()

    parts = []
    if gist and gist.get("narrative"):
        parts.append("BOT NARRATIVE ANALYSIS:\n" + gist["narrative"][:800])
    if gist and gist.get("polymarket"):
        pm_text = polymarket_to_text(gist["polymarket"])
        if pm_text:
            parts.append(pm_text)
    hl_text = headlines_to_text(headlines, max_items=15)
    if hl_text:
        parts.append("RSS HEADLINES:\n" + hl_text)
    inbox_text = inbox_to_text(inbox)
    if inbox_text:
        parts.append("MANUAL INBOX:\n" + inbox_text)

    if not parts:
        return False

    context = "\n\n".join(parts)
    prompt = (
        "You are a senior macro research analyst. Based on the following current events, "
        "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
        "dominant narratives, and flags any actionable risks or opportunities. "
        "Be clinical, specific, and reference actual catalysts.\n\n"
        f"{context[:3000]}"
    )

    digest = None
    _use_cl = use_claude
    _model = model

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
            _model = _model or "claude-haiku-4-5-20251001"

    if _use_cl or not digest:
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "")).messages.create(
                model=_model or "claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            digest = msg.content[0].text.strip()
        except Exception:
            return False

    if digest:
        _tier = "👑 Highly Regarded Mode" if (_use_cl and _model == "claude-sonnet-4-6") else ("🧠 Regard Mode" if _use_cl else "⚡ Groq")
        st.session_state["_current_events_digest"] = digest
        st.session_state["_current_events_digest_ts"] = datetime.now()
        st.session_state["_current_events_engine"] = _tier
        return True
    return False


def _time_ago(ts_str: str) -> str:
    """Convert ISO timestamp to human-readable 'Xh ago'."""
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
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
        ["⚡ Groq (fast)", "🧠 Regard Mode", "👑 Highly Regarded Mode"],
        key="ce_digest_engine",
        horizontal=True,
        label_visibility="collapsed",
    )
    _rec_map = {
        "⚡ Groq (fast)": "Daily routine check — fast, free. Use when markets are calm and you just want a quick brief.",
        "🧠 Regard Mode": "Active trading day — Haiku gives better synthesis than Groq. Use when you need the digest to inform Discovery or Valuation.",
        "👑 Highly Regarded Mode": "High-conviction sessions — Sonnet reads macro nuance best. Use before running Valuation or Portfolio when volatility is elevated or a major catalyst is live.",
    }
    st.caption(f"💡 {_rec_map.get(engine, '')}")

    existing_digest = st.session_state.get("_current_events_digest", "")
    existing_ts = st.session_state.get("_current_events_digest_ts", "")
    if existing_digest:
        st.markdown(
            f'<div style="border:1px solid {COLORS["bloomberg_orange"]}44;border-radius:4px;'
            f'padding:10px 14px;background:{COLORS["surface"]};margin-bottom:8px;">'
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
            f'color:{COLORS["text_dim"]};margin-bottom:6px;">Generated {_time_ago(existing_ts.isoformat() if hasattr(existing_ts, "isoformat") else existing_ts) if existing_ts else ""}</div>'
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:13px;'
            f'color:{COLORS["text"]};line-height:1.6;">{existing_digest}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗞 Generate News Digest", key="ce_gen_digest", type="primary"):
        with st.spinner("Generating digest..."):
            _run_digest(headlines, inbox, gist, engine)


def _run_digest(headlines, inbox, gist, engine: str):
    """Build context and call AI for news digest."""
    from datetime import datetime

    parts = []

    if gist and gist.get("narrative"):
        parts.append("BOT NARRATIVE ANALYSIS:\n" + gist["narrative"][:800])

    if gist and gist.get("polymarket"):
        pm_text = polymarket_to_text(gist["polymarket"])
        if pm_text:
            parts.append(pm_text)

    hl_text = headlines_to_text(headlines, max_items=15)
    if hl_text:
        parts.append("RSS HEADLINES:\n" + hl_text)

    inbox_text = inbox_to_text(inbox)
    if inbox_text:
        parts.append("MANUAL INBOX:\n" + inbox_text)

    if not parts:
        st.warning("No content to digest. Refresh RSS or add inbox items first.")
        return

    context = "\n\n".join(parts)
    prompt = (
        "You are a senior macro research analyst. Based on the following current events, "
        "generate a 3-4 sentence market digest that synthesizes the key themes, identifies "
        "dominant narratives, and flags any actionable risks or opportunities. "
        "Be clinical, specific, and reference actual catalysts.\n\n"
        f"{context[:3000]}"
    )

    _tier_model = {
        "⚡ Groq (fast)": None,
        "🧠 Regard Mode": "claude-haiku-4-5-20251001",
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
            cl_model = "claude-haiku-4-5-20251001"

    if use_claude or not digest:
        try:
            import anthropic, os
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
        st.session_state["_current_events_digest"] = digest
        st.session_state["_current_events_digest_ts"] = datetime.now()
        st.session_state["_current_events_engine"] = engine
        st.rerun()
