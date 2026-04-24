"""Shared AI tier selector component for consistent engine picker UI."""
import streamlit as st

TIER_OPTS = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
TIER_OPTS_EXT = TIER_OPTS + ["📜 DD Scholar Mode"]

TIER_MAP: dict[str, tuple[bool, str | None]] = {
    "⚡ Freeloader Mode":      (False, None),
    "🧠 Regard Mode":          (True, "grok-4-1-fast-reasoning"),
    "👑 Highly Regarded Mode": (True, "claude-haiku-4-5-20251001"),
    "📜 DD Scholar Mode":      (True, "claude-sonnet-4-6"),
}

MODEL_HINT_HTML = (
    '<div style="font-size:10px;color:#64748b;'
    "font-family:'JetBrains Mono',Consolas,monospace;"
    'margin-top:-10px;margin-bottom:4px;">'
    "⚡ Groq LLaMA 3.3 70B &nbsp;·&nbsp; "
    "🧠 Grok 4.1 &nbsp;·&nbsp; "
    "👑 Claude Haiku 4.5"
    "</div>"
)

MODEL_HINT_HTML_EXT = (
    '<div style="font-size:10px;color:#64748b;'
    "font-family:'JetBrains Mono',Consolas,monospace;"
    'margin-top:-10px;margin-bottom:4px;">'
    "⚡ Groq LLaMA 3.3 70B &nbsp;·&nbsp; "
    "🧠 Grok 4.1 &nbsp;·&nbsp; "
    "👑 Claude Haiku 4.5 &nbsp;·&nbsp; "
    "📜 Claude Sonnet 4.6"
    "</div>"
)

DEFAULT_REC = (
    "⚡ Freeloader for quick checks · "
    "🧠 Regard for active sessions · "
    "👑 Highly Regarded for high-conviction decisions"
)


def render_ai_tier_selector(
    key: str,
    label: str = "AI Engine",
    recommendation: str | None = None,
    default: int = 0,
    include_dd_scholar: bool = False,
) -> tuple[bool, str | None]:
    """Render a consistent AI engine radio selector.

    By default shows 3 tiers. Pass include_dd_scholar=True to expose the
    📜 DD Scholar tier (Claude Sonnet 4.6) for reasoning-heavy modules.
    Returns (use_claude: bool, model: str | None).
    """
    opts = TIER_OPTS_EXT if include_dd_scholar else TIER_OPTS
    hint = MODEL_HINT_HTML_EXT if include_dd_scholar else MODEL_HINT_HTML
    sel = st.radio(label, opts, horizontal=True, key=key, index=default)
    st.markdown(hint, unsafe_allow_html=True)
    rec = recommendation if recommendation is not None else DEFAULT_REC
    st.caption(f"💡 {rec}")
    return TIER_MAP[sel]
