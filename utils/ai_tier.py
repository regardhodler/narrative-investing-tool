"""Shared AI tier selector component for consistent engine picker UI."""
import streamlit as st

TIER_OPTS = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"]

TIER_MAP: dict[str, tuple[bool, str | None]] = {
    "⚡ Freeloader Mode":      (False, None),
    "🧠 Regard Mode":          (True, "grok-4-1-fast-reasoning"),
    "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
}

MODEL_HINT_HTML = (
    '<div style="font-size:10px;color:#64748b;'
    "font-family:'JetBrains Mono',Consolas,monospace;"
    'margin-top:-10px;margin-bottom:4px;">'
    "⚡ Groq LLaMA 3.3 70B &nbsp;·&nbsp; "
    "🧠 Grok 4.1 &nbsp;·&nbsp; "
    "👑 Claude Sonnet 4.6"
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
) -> tuple[bool, str | None]:
    """Render a consistent 3-tier AI engine radio selector.

    Always shows all three tiers. Returns (use_claude: bool, model: str | None).
    """
    sel = st.radio(label, TIER_OPTS, horizontal=True, key=key, index=default)
    st.markdown(MODEL_HINT_HTML, unsafe_allow_html=True)
    rec = recommendation if recommendation is not None else DEFAULT_REC
    st.caption(f"💡 {rec}")
    return TIER_MAP[sel]
