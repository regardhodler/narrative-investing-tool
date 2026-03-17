import plotly.graph_objects as go

FONT_FAMILY = "JetBrains Mono, Consolas, Courier New, monospace"

COLORS = {
    "bg": "#0E1117",
    "surface": "#1A1F2E",
    "accent": "#00D4AA",
    "accent_dim": "#00A88A",
    "red": "#FF4B4B",
    "green": "#00D4AA",
    "blue": "#4B9FFF",
    "yellow": "#FFD700",
    "text": "#E0E0E0",
    "text_dim": "#888888",
    "grid": "#2A2F3E",
    "orange": "#FF8C00",
    "surface_dark": "#0A0E14",
    "border": "#2A3040",
    "bloomberg_orange": "#FF8811",
    "header_bg": "#0A0D12",
    "sidebar_bg": "#0C1018",
    "input_bg": "#141922",
    "hover": "#1F2940",
    "positive": "#00C853",
    "negative": "#FF1744",
}


def dark_layout(**overrides) -> dict:
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="JetBrains Mono, Consolas, Courier New, monospace", color=COLORS["text"], size=12),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    layout.update(overrides)
    return layout


def apply_dark_layout(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**dark_layout(**overrides))
    return fig


def bloomberg_metric(label: str, value: str, color: str | None = None) -> str:
    """Return styled HTML for a Bloomberg-terminal-style metric block."""
    val_color = color or COLORS["text"]
    return (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'padding:10px 14px;border-radius:4px;font-family:\'JetBrains Mono\',Consolas,monospace;">'
        f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{val_color};">{value}</div>'
        f'</div>'
    )
