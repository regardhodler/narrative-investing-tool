import plotly.graph_objects as go

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
}


def dark_layout(**overrides) -> dict:
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
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
