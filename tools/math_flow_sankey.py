"""
Math Flow Sankey Diagram — Narrative Investing Tool
Shows how raw data → signal engines → composites → decisions
Run: python tools/math_flow_sankey.py
Outputs: docs/math_flow_sankey.html
"""

import os
import plotly.graph_objects as go

# ── Node definitions (label, x-position layer 0-4, y-hint) ──────────────────
NODES = [
    # Layer 0 — Raw data sources
    {"label": "FRED\n(27 series)",          "layer": 0, "color": "#4B6FA5"},
    {"label": "yfinance\n(prices / ETFs)",  "layer": 0, "color": "#4B6FA5"},
    {"label": "Options Chain\n(SPY OI/IV)", "layer": 0, "color": "#4B6FA5"},
    {"label": "SEC EDGAR\n(13F / Form 4)",  "layer": 0, "color": "#4B6FA5"},

    # Layer 1 — Signal engines
    {"label": "Z-score → tanh\n(27 indicators)",     "layer": 1, "color": "#6C8EBF"},
    {"label": "GEX\nGamma Engine",                   "layer": 1, "color": "#6C8EBF"},
    {"label": "Elliott Wave\nEngine (SPY)",           "layer": 1, "color": "#6C8EBF"},
    {"label": "Wyckoff Phase\nEngine (SPY)",          "layer": 1, "color": "#6C8EBF"},
    {"label": "Gaussian HMM\n(10 features, 15yr)",    "layer": 1, "color": "#6C8EBF"},
    {"label": "Stress Z-Score\n(5 spreads, weighted)","layer": 1, "color": "#6C8EBF"},
    {"label": "Whale Flow\nScore (13F)",              "layer": 1, "color": "#6C8EBF"},

    # Layer 2 — Composites / scored outputs
    {"label": "macro_score\n(0–100, category-averaged)", "layer": 2, "color": "#00C4B4"},
    {"label": "HMM State\n+ Confidence + Entropy",        "layer": 2, "color": "#00C4B4"},
    {"label": "Fear Composite\n(5-component, 0–100)",      "layer": 2, "color": "#00C4B4"},
    {"label": "Signal Scorecard\n(per ticker, 0–100)",     "layer": 2, "color": "#00C4B4"},

    # Layer 3 — Regime context
    {"label": "macro_regime\n(Risk-On / Neutral / Risk-Off)", "layer": 3, "color": "#FFB703"},
    {"label": "Kelly\nPosition Size",                         "layer": 3, "color": "#FFB703"},

    # Layer 4 — Decisions / UI outputs
    {"label": "AI Plays\n(Quick Run)",      "layer": 4, "color": "#FF6B6B"},
    {"label": "Sector\nRotation Recs",      "layer": 4, "color": "#FF6B6B"},
    {"label": "Valuation\nRating",          "layer": 4, "color": "#FF6B6B"},
    {"label": "Trade Journal\nSizing",      "layer": 4, "color": "#FF6B6B"},
]

N = {node["label"]: i for i, node in enumerate(NODES)}

# ── Link definitions (source_label, target_label, value, hover) ─────────────
RAW_LINKS = [
    # Data sources → engines
    ("FRED\n(27 series)",           "Z-score → tanh\n(27 indicators)",      20, "27 FRED series → z-scored + tanh normalized"),
    ("yfinance\n(prices / ETFs)",   "Z-score → tanh\n(27 indicators)",       7, "7 ETF/equity signals (SPY, QQQ, UUP, VIX, RSP, EWG, FXI)"),
    ("yfinance\n(prices / ETFs)",   "GEX\nGamma Engine",                      2, "SPY options OI × IV × spot → net gamma by strike"),
    ("Options Chain\n(SPY OI/IV)",  "GEX\nGamma Engine",                      3, "Call/put OI + IV per strike → GEX, flip, walls"),
    ("yfinance\n(prices / ETFs)",   "Elliott Wave\nEngine (SPY)",             2, "SPY daily OHLCV → pivot detection → Fibonacci scoring"),
    ("yfinance\n(prices / ETFs)",   "Wyckoff Phase\nEngine (SPY)",            2, "SPY 2yr daily OHLCV+volume → phase + sub-phase detection"),
    ("FRED\n(27 series)",           "Gaussian HMM\n(10 features, 15yr)",      9, "9 FRED series: HY OAS, IG OAS, T10Y2Y, T10Y3M, DGS10, DGS2, DFII10, NFCI, ICSA"),
    ("yfinance\n(prices / ETFs)",   "Gaussian HMM\n(10 features, 15yr)",      1, "VIX daily series"),
    ("FRED\n(27 series)",           "Stress Z-Score\n(5 spreads, weighted)",  5, "HY OAS (30%), IG OAS (20%), T10Y2Y (20%), TED (15%), Lending Stds (15%)"),
    ("SEC EDGAR\n(13F / Form 4)",   "Whale Flow\nScore (13F)",                4, "Net $ flow, conviction (new/closed ratio), category rotation, 13D activism"),

    # Engines → composites
    ("Z-score → tanh\n(27 indicators)",      "macro_score\n(0–100, category-averaged)", 27, "Two-stage: within-category conf-weighted avg → cross-category tier-weighted avg"),
    ("GEX\nGamma Engine",                     "macro_score\n(0–100, category-averaged)",  1, "Dealer gamma at spot → Positioning score (tier 3, weight 1.0)"),
    ("Elliott Wave\nEngine (SPY)",            "macro_score\n(0–100, category-averaged)",  1, "Wave label × confidence → tanh score (Market Structure, weight 0.8)"),
    ("Wyckoff Phase\nEngine (SPY)",           "macro_score\n(0–100, category-averaged)",  1, "Phase + sub-phase × confidence → tanh score (Market Structure, weight 0.8)"),
    ("Gaussian HMM\n(10 features, 15yr)",     "HMM State\n+ Confidence + Entropy",        5, "Viterbi decode → state label, confidence, entropy, 1m/2m transition risk"),
    ("Stress Z-Score\n(5 spreads, weighted)", "Fear Composite\n(5-component, 0–100)",     3, "25% weight in Fear Composite"),
    ("Whale Flow\nScore (13F)",               "Fear Composite\n(5-component, 0–100)",     2, "20% weight: bull % net flow, rotation, activism cross-ref"),
    ("SEC EDGAR\n(13F / Form 4)",             "Signal Scorecard\n(per ticker, 0–100)",    3, "Insider buys/sells → insider category score (one of 6 factors)"),

    # Composites → regime context
    ("macro_score\n(0–100, category-averaged)",      "macro_regime\n(Risk-On / Neutral / Risk-Off)", 10, "≥60 = Risk-On, ≤40 = Risk-Off, else Neutral with lean"),
    ("HMM State\n+ Confidence + Entropy",             "Kelly\nPosition Size",                          4, "Bull=×1.10, Neutral=×1.00, Stress=×0.85, Crisis=×0.60"),
    ("HMM State\n+ Confidence + Entropy",             "macro_regime\n(Risk-On / Neutral / Risk-Off)",  3, "HMM state surfaced in regime context for AI prompts"),
    ("Fear Composite\n(5-component, 0–100)",          "Kelly\nPosition Size",                          3, "Stress discount: up to 30% haircut on Kelly fraction"),
    ("Fear Composite\n(5-component, 0–100)",          "macro_regime\n(Risk-On / Neutral / Risk-Off)",  2, "Feeds _regime_context for AI signal summary"),

    # Regime context → decisions
    ("macro_regime\n(Risk-On / Neutral / Risk-Off)", "AI Plays\n(Quick Run)",      5, "Regime + quadrant + top signals → Groq/Claude play ideas"),
    ("macro_regime\n(Risk-On / Neutral / Risk-Off)", "Sector\nRotation Recs",      4, "Quadrant (Goldilocks/Stagflation/etc.) → ETF favor/avoid list"),
    ("macro_regime\n(Risk-On / Neutral / Risk-Off)", "Valuation\nRating",          3, "Regime label grounds the AI valuation prompt"),
    ("Kelly\nPosition Size",                          "Trade Journal\nSizing",      5, "Bayesian half-Kelly: win rate shrinkage + regime b + HMM × signal alignment"),
    ("Signal Scorecard\n(per ticker, 0–100)",         "Valuation\nRating",          3, "Ticker-level 6-factor score feeds valuation module"),
]

sources = [N[s] for s, _, _, _ in RAW_LINKS]
targets = [N[t] for _, t, _, _ in RAW_LINKS]
values  = [v     for _, _, v, _ in RAW_LINKS]
hovers  = [h     for _, _, _, h in RAW_LINKS]

# Build x/y positions from layer
layer_x = {0: 0.02, 1: 0.24, 2: 0.50, 3: 0.74, 4: 0.97}
layer_nodes = {0: [], 1: [], 2: [], 3: [], 4: []}
for i, node in enumerate(NODES):
    layer_nodes[node["layer"]].append(i)

node_x, node_y = [], []
for i, node in enumerate(NODES):
    layer = node["layer"]
    peers = layer_nodes[layer]
    pos = peers.index(i)
    node_x.append(layer_x[layer])
    node_y.append(0.05 + (pos / max(len(peers) - 1, 1)) * 0.90)

fig = go.Figure(go.Sankey(
    arrangement="fixed",
    node=dict(
        pad=18,
        thickness=20,
        line=dict(color="#1a1a2e", width=0.5),
        label=[n["label"] for n in NODES],
        color=[n["color"] for n in NODES],
        x=node_x,
        y=node_y,
        hovertemplate="<b>%{label}</b><extra></extra>",
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        customdata=hovers,
        hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>%{customdata}<extra></extra>",
        color="rgba(100,160,220,0.22)",
    ),
))

fig.update_layout(
    title=dict(
        text="<b>Narrative Investing Tool — Math Flow</b><br>"
             "<sub>Hover links for formula details · Width ∝ number of signal connections</sub>",
        font=dict(size=18, color="#e0e0e0"),
        x=0.5,
    ),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e0e0e0", size=11),
    height=720,
    margin=dict(l=20, r=20, t=80, b=20),
)

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "math_flow_sankey.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Saved → {out_path}")
