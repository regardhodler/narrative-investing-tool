"""Pure-math signal quantification helpers.

No Streamlit, no API calls — all inputs are pre-fetched DataFrames or strings.
Four functions, one per signal family:
  compute_stress_zscore    — FRED credit/rate series z-scored vs 252d history
  compute_whale_flow_score — net $ flow bias + category rotation from 13F DataFrame
  compute_events_sentiment — keyword ratio + uncertainty index from digest text
  compute_canary_score     — breadth + momentum composite from canary DataFrame
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Stress z-score
# ---------------------------------------------------------------------------

STRESS_WEIGHTS = {
    "BAMLH0A0HYM2": 0.30,  # HY OAS — primary credit stress gauge
    "BAMLC0A0CM":   0.20,  # IG OAS — investment-grade spread
    "T10Y2Y":       0.20,  # Yield curve (inverted = stress → sign flipped)
    "TEDRATE":      0.15,  # TED spread — interbank funding stress
    "DRTSCILM":     0.15,  # Bank C&I lending standards — lagging but confirming
}

_SERIES_LABELS = {
    "BAMLH0A0HYM2": "HY OAS",
    "BAMLC0A0CM":   "IG OAS",
    "T10Y2Y":       "Yield Curve",
    "TEDRATE":      "TED Spread",
    "DRTSCILM":     "Lending Stds",
}


def _erf(x: float) -> float:
    """Abramowitz & Stegun ERF approximation — no scipy needed."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t * np.exp(-(x * x))
    return y if x >= 0 else -y


def compute_stress_zscore(fred_data: dict) -> dict:
    """Z-score each FRED stress series vs its own 252-day history, then combine.

    Returns:
        {"z": float, "pct": int, "components": {label: z_value}}
    """
    zscores: dict[str, float] = {}
    weighted_z = 0.0
    total_w = 0.0

    for sid, weight in STRESS_WEIGHTS.items():
        df = fred_data.get(sid)
        if df is None or df.empty or "value" not in df.columns:
            continue
        series = df["value"].dropna().astype(float)
        if len(series) < 30:
            continue
        tail = series.tail(252)
        mu = float(tail.mean())
        sigma = float(tail.std())
        if sigma < 1e-9:
            continue
        z = float((series.iloc[-1] - mu) / sigma)
        # Yield curve: negative spread = stress → flip so positive z = more stress
        if sid == "T10Y2Y":
            z = -z
        label = _SERIES_LABELS.get(sid, sid)
        zscores[label] = round(z, 2)
        weighted_z += z * weight
        total_w += weight

    if total_w == 0:
        return {"z": 0.0, "pct": 50, "components": {}}

    final_z = weighted_z / total_w
    # Map to percentile via normal CDF (positive z → elevated stress → high percentile)
    pct = int(min(99, max(1, (0.5 * (1.0 + _erf(final_z / 1.41421356))) * 100)))
    return {"z": round(final_z, 2), "pct": pct, "components": zscores}


# ---------------------------------------------------------------------------
# 2. Whale flow score
# ---------------------------------------------------------------------------

_RISK_ON_CATS = {"Technology", "Consumer Cyclical", "Communication Services"}
_DEFENSIVE_CATS = {"Healthcare", "Consumer Defensive", "Utilities", "Financials"}


def compute_whale_flow_score(df) -> dict:
    """Net $ flow bias + category rotation from whale screener DataFrame.

    Expects columns: value_change, status, whale_category.
    Returns:
        {"bull_pct": float, "net_flow_bn": float, "rotation": float,
         "conviction": float, "label": str}
    """
    if df is None or df.empty:
        return {"bull_pct": 50.0, "net_flow_bn": 0.0, "rotation": 0.0,
                "conviction": 0.5, "label": "No Data"}

    bull_rows = df[df["status"].isin(["new", "increased"])]
    bear_rows = df[df["status"].isin(["decreased", "sold"])]
    bull_flow = float(bull_rows["value_change"].sum())
    bear_flow = float(bear_rows["value_change"].abs().sum())
    total = bull_flow + bear_flow
    if total < 1:
        return {"bull_pct": 50.0, "net_flow_bn": 0.0, "rotation": 0.0,
                "conviction": 0.5, "label": "No Data"}

    bull_pct = bull_flow / total * 100
    net_bn = (bull_flow - bear_flow) / 1e9

    new_c = int((df["status"] == "new").sum())
    sold_c = int((df["status"] == "sold").sum())
    conviction = new_c / (new_c + sold_c + 1)

    # Category rotation: risk-on flow vs defensive flow
    ro_flow = float(df[df["whale_category"].isin(_RISK_ON_CATS)]["value_change"].sum())
    def_flow = float(df[df["whale_category"].isin(_DEFENSIVE_CATS)]["value_change"].sum())
    rotation = (ro_flow - def_flow) / (abs(ro_flow) + abs(def_flow) + 1)

    if bull_pct > 70:
        label = "Heavy Accumulation"
    elif bull_pct > 55:
        label = "Mild Accumulation"
    elif bull_pct > 45:
        label = "Neutral"
    elif bull_pct > 30:
        label = "Distribution"
    else:
        label = "Heavy Distribution"

    return {
        "bull_pct":    round(bull_pct, 1),
        "net_flow_bn": round(net_bn, 2),
        "rotation":    round(rotation, 2),
        "conviction":  round(conviction, 2),
        "label":       label,
    }


# ---------------------------------------------------------------------------
# 3. Current events sentiment score
# ---------------------------------------------------------------------------

_BULL_W = [
    "rally", "surge", "strong", "beat", "growth", "expansion", "bullish",
    "upgrade", "hiring", "stimulus", "recovery", "record", "accelerat",
    "outperform", "resilient", "robust", "boom",
]
_BEAR_W = [
    "crash", "recession", "weak", "miss", "contraction", "layoffs", "bearish",
    "downgrade", "default", "tighten", "slowdown", "selloff", "plunge",
    "collapse", "stagflat", "decelerat", "deteriorat",
]
_UNCERTAIN_W = [
    "uncertain", "unclear", "could", "may", "risk", "concern", "volatile",
    "warning", "caution", "fragile", "unstable", "turbulence", "headwind",
    "watch", "monitor",
]


def compute_events_sentiment(text: str) -> dict:
    """Keyword ratio + uncertainty index from digest text.

    FALLBACK ONLY — used when the AI digest didn't return a SENTIMENT: block.
    The preferred path is the AI-extracted score from current_events._meta_to_sentiment_score().

    Returns:
        {"sentiment": float [-1,+1], "uncertainty": float [0,1],
         "bull_hits": int, "bear_hits": int, "label": str, "source": "keyword"}
    """
    if not text:
        return {"sentiment": 0.0, "uncertainty": 0.0,
                "bull_hits": 0, "bear_hits": 0, "label": "No Data"}
    t = text.lower()
    b = sum(t.count(w) for w in _BULL_W)
    s = sum(t.count(w) for w in _BEAR_W)
    u = sum(t.count(w) for w in _UNCERTAIN_W)
    sentiment = round((b - s) / (b + s + 1), 3)
    uncertainty = round(min(1.0, u / 20.0), 3)

    if sentiment > 0.15:
        label = "Risk-On Tone"
    elif sentiment < -0.15:
        label = "Risk-Off Tone"
    elif uncertainty > 0.5:
        label = "High Uncertainty"
    else:
        label = "Neutral"

    return {
        "sentiment":   sentiment,
        "uncertainty": uncertainty,
        "bull_hits":   b,
        "bear_hits":   s,
        "label":       label,
        "source":      "keyword",
    }


# ---------------------------------------------------------------------------
# 4. Canary breadth score
# ---------------------------------------------------------------------------


# HY ETFs are excluded from canary to avoid double-dipping with stress_z:
# HYG/JNK/BKLN/SJNK price is the mirror of BAMLH0A0HYM2 (HY OAS) already in stress_z.
# TLT is excluded because it's rate-driven (already in T10Y2Y via stress_z).
# VIX is excluded because it's inverse. Remaining canaries are genuinely orthogonal:
# Regional Banks, CRE REITs, PE firms, Consumer Credit — structural stress canaries.
_CANARY_EXCLUDE = {"^VIX", "TLT", "HYG", "JNK", "BKLN", "SJNK"}


def compute_canary_score(canary_df) -> dict:
    """Cross-sectional breadth + momentum from canary watchlist DataFrame.

    Excludes HY ETFs and rate proxies (TLT, ^VIX) to avoid double-dipping
    with stress_zscore which already z-scores HY OAS and yield curve from FRED.

    Expects columns: ticker, 1m_ret, drawdown_52w, volume_ratio.
    Returns:
        {"composite": float, "breadth_pct": float, "momentum_avg": float,
         "drawdown_pct": float, "vol_surge": float}
    """
    _empty = {"composite": 50.0, "breadth_pct": 50.0, "momentum_avg": 0.0,
               "drawdown_pct": 50.0, "vol_surge": 1.0}
    if canary_df is None or canary_df.empty:
        return _empty

    df = canary_df[~canary_df["ticker"].isin(_CANARY_EXCLUDE)].copy()
    if df.empty:
        return _empty

    n = len(df)
    breadth = len(df[df["1m_ret"] > 0]) / n * 100
    mom_avg = float(df["1m_ret"].mean())
    # Normalize momentum: -10% → 0, 0% → 50, +10% → 100
    mom_norm = min(100.0, max(0.0, (mom_avg + 10.0) / 20.0 * 100.0))
    dd_pct = len(df[df["drawdown_52w"] > -10]) / n * 100

    # Volume surge on losing tickers — elevated = stress confirmation
    stressed = df[df["1m_ret"] < 0]
    vol_surge = float(stressed["volume_ratio"].mean()) if not stressed.empty else 1.0

    composite = 0.35 * breadth + 0.35 * mom_norm + 0.30 * dd_pct

    return {
        "composite":    round(composite, 1),
        "breadth_pct":  round(breadth, 1),
        "momentum_avg": round(mom_avg, 2),
        "drawdown_pct": round(dd_pct, 1),
        "vol_surge":    round(vol_surge, 2),
    }


# ---------------------------------------------------------------------------
# 5. Fear Composite Index
# ---------------------------------------------------------------------------

def compute_fear_composite(
    stress_zscore: dict,
    whale_flow: dict,
    events_sentiment: dict,
    canary: dict,
    macro_score: float = 50.0,
) -> dict:
    """Combine all 4 quantified scores + macro regime into a single 0-100 fear index.

    100 = maximum fear / risk-off. 0 = maximum complacency / risk-on.
    Weights:
      Stress z-score percentile  25%  (credit/rate stress vs history)
      Macro regime score         20%  (inverted: low macro = high fear)
      Canary breadth             20%  (inverted: low breadth = high fear)
      Whale flow bull%           20%  (inverted: distribution = fear)
      Events sentiment           15%  (inverted: risk-off tone = fear)

    Returns:
        {"score": int, "label": str, "components": {name: contribution}}
    """
    stress_pct  = float(stress_zscore.get("pct", 50))
    whale_bull  = float(whale_flow.get("bull_pct", 50))
    sentiment   = float(events_sentiment.get("sentiment", 0.0))
    canary_comp = float(canary.get("composite", 50))
    macro       = float(macro_score)

    # Convert each to a 0-100 fear scale (high = more fear)
    c_stress  = stress_pct                        # already 0-100, high = stress
    c_macro   = 100.0 - macro                     # invert: low macro = high fear
    c_canary  = 100.0 - canary_comp               # invert: low canary = high fear
    c_whale   = 100.0 - whale_bull                # invert: distribution = fear
    c_events  = (0.5 - sentiment * 0.5) * 100.0  # -1→100, 0→50, +1→0

    score = (
        c_stress  * 0.25 +
        c_macro   * 0.20 +
        c_canary  * 0.20 +
        c_whale   * 0.20 +
        c_events  * 0.15
    )
    score = round(min(100.0, max(0.0, score)), 1)

    if score >= 75:
        label = "Extreme Fear"
    elif score >= 60:
        label = "High Fear"
    elif score >= 45:
        label = "Elevated"
    elif score >= 30:
        label = "Moderate"
    else:
        label = "Low Fear"

    return {
        "score": score,
        "label": label,
        "components": {
            "Stress":  round(c_stress, 1),
            "Macro":   round(c_macro, 1),
            "Canary":  round(c_canary, 1),
            "Whale":   round(c_whale, 1),
            "Events":  round(c_events, 1),
        },
    }
