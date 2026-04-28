"""
Tops & Bottoms Predictor — Live Decision Tool

Three-section page:
  1. TODAY'S VERDICT — composite top/bottom probability + drivers from both brains.
  2. HISTORICAL BACKTEST — every historical signal fire vs actual SPX peaks/troughs.
  3. PROBABILITY TIME SERIES — top/bottom probability over the full overlap window.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from services.hmm_regime import (
    compute_full_state_path,
    get_ci_anchor,
    load_current_hmm_state,
    load_hmm_brain,
)
from services.hmm_shadow import (
    compute_full_shadow_state_path,
    load_current_shadow_state,
    load_shadow_brain,
)
from services.hmm_top import (
    compute_full_top_state_path,
    compute_top_signal_today,
    load_top_brain,
    _BT_HITS, _BT_PEAKS, _BT_HIT_PCT, _BT_FA, _BT_AVG_LEAD,
    _LL_ROLL_THRESH, _LL_ROLL_WINDOW, _LATE_LABELS as _TOP_LATE_LABELS,
)
from services.turning_point import (
    SIGNAL_NAMES,
    compute_turning_point_probability,
    load_fingerprints,
)
from utils.ci_zone import classify_ci_zone, classify_from_ll_zscore
from utils.theme import COLORS, apply_dark_layout

# Reuse the check-engine banner already built for the regime chart
from modules.regime_chart import _brain_health, _render_status_banner

def _trigger_anchor_neg() -> float:
    return -get_ci_anchor()

# Composite-score signal thresholds
_BOTTOM_FIRE_THRESHOLD = 0.55
_TOP_FIRE_THRESHOLD = 0.55

# How wide a window counts as a "hit" when scoring against actual SPX extrema
_HIT_WINDOW_DAYS = 60

# Bear/stress states for each brain
_MAIN_BEAR_STATES = {"Stress", "Late Cycle", "Crisis", "Early Stress"}
_MAIN_TOP_RISK_STATES = {"Late Cycle", "Stress", "Early Stress"}
_SHADOW_BEAR_STATES = {"Strong Bear", "Mild Bear", "Crisis"}
_SHADOW_TOP_RISK_STATES = {"Transition", "Mild Bear"}


# ────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_spx_close() -> pd.Series:
    import yfinance as yf
    df = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def _load_main_path(_cache_key: str = "") -> Optional[pd.DataFrame]:
    return compute_full_state_path()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_shadow_path(_cache_key: str = "") -> Optional[pd.DataFrame]:
    return compute_full_shadow_state_path()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_top_path(_cache_key: str = "") -> Optional[pd.DataFrame]:
    return compute_full_top_state_path()


# ────────────────────────────────────────────────────────────────────────────
# Today's verdict computation
# ────────────────────────────────────────────────────────────────────────────

def _safe(value, default=0.0):
    """Convert None/NaN to default, else return value."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except Exception:
        pass
    return value


def _persistence_bonus(days: int, target: int = 30) -> float:
    """Soft saturating bonus: 1.0 baseline, ramps to 1.4 at `target` days, capped."""
    if days <= 0:
        return 1.0
    return 1.0 + min(0.4, 0.4 * (days / target))


def _z_reversing_up(brain_path: pd.DataFrame, lookback: int = 5) -> bool:
    """True if today's z is meaningfully above the min of the last `lookback` days."""
    if brain_path is None or brain_path.empty:
        return False
    tail = brain_path["ll_zscore"].tail(lookback + 1)
    if len(tail) < 2:
        return False
    return float(tail.iloc[-1]) > float(tail.min()) + 0.05


def _state_in(state_label: str, target_set: set) -> bool:
    return state_label in target_set


def _compute_today_verdict(main_state, shadow_state, main_path, shadow_path, shadow_brain=None) -> dict:
    """Produce the headline verdict combining both brains.

    Returns:
        {
          'verdict': 'BOTTOM PROBABLE' | 'TOP PROBABLE' | 'STRESS BUILDING' | 'NEUTRAL',
          'verdict_color': hex,
          'bottom_prob': 0-95,
          'top_prob': 0-95,
          'confidence': 'HIGH' | 'MED' | 'LOW',
          'signal_breadth': 0-100,
          'drivers_bullish': [...],
          'drivers_bearish': [...],
          'horizon': '30 days',
          'narrative': str (one-line)
        }
    """
    # ── Pull fields with safe defaults ─────────────────────────────────────
    main_z = _safe(getattr(main_state, "ll_zscore", 0.0))
    shadow_z = _safe(getattr(shadow_state, "ll_zscore", 0.0))
    main_state_label = getattr(main_state, "state_label", "Unknown") if main_state else "Unknown"
    shadow_state_label = getattr(shadow_state, "state_label", "Unknown") if shadow_state else "Unknown"
    main_persistence = int(_safe(getattr(main_state, "persistence", 0)))
    shadow_persistence = int(_safe(getattr(shadow_state, "persistence", 0)))
    main_trans_1m = _safe(getattr(main_state, "transition_risk_1m", 0.0))
    main_entropy = _safe(getattr(main_state, "entropy", 0.0))
    main_confidence = _safe(getattr(main_state, "confidence", 0.0))

    main_anchor = get_ci_anchor()
    main_zone = classify_from_ll_zscore(main_z, anchor=main_anchor)
    shadow_anchor = getattr(shadow_brain, "ci_anchor", main_anchor) if shadow_brain else main_anchor
    shadow_zone = classify_from_ll_zscore(shadow_z, anchor=shadow_anchor)

    # ── Bottom score: capitulation logic ───────────────────────────────────
    main_ci = abs(main_z) / max(main_anchor, 1e-6) if main_z < 0 else 0.0
    shadow_ci = abs(shadow_z) / max(shadow_anchor, 1e-6) if shadow_z < 0 else 0.0
    capitulation = max(main_ci, shadow_ci)
    capitulation = min(capitulation, 1.5)  # cap effective contribution

    z_reversing = _z_reversing_up(shadow_path) or _z_reversing_up(main_path)
    both_zone3 = main_zone.zone >= 3 and shadow_zone.zone >= 3

    bear_persistence = max(
        main_persistence if _state_in(main_state_label, _MAIN_BEAR_STATES) else 0,
        shadow_persistence if _state_in(shadow_state_label, _SHADOW_BEAR_STATES) else 0,
    )

    bottom_score = (
        capitulation
        * (1.0 if z_reversing else 0.55)
        * (1.3 if both_zone3 else 1.0)
        * _persistence_bonus(bear_persistence, target=20)
    ) / 1.5  # normalize

    # ── Top score: distribution / late-cycle logic ─────────────────────────
    # Use turning_point.py for the bearish_prob (top) channel
    tp = compute_turning_point_probability(
        macro_score=None,
        regime_velocity=None,
        entropy=main_entropy,
        ll_zscore=main_z,
        hmm_confidence=main_confidence,
        conviction=None,
        vix=None,
        credit_hy_z=None,
        yield_curve_z=None,
        nfci_z=None,
        transition_risk_1m=main_trans_1m,
        hmm_state_label=main_state_label,
    )
    bearish_prob = tp.get("bearish_prob", 0.0) / 100.0
    bullish_prob = tp.get("bullish_prob", 0.0) / 100.0

    top_persistence = main_persistence if _state_in(main_state_label, _MAIN_TOP_RISK_STATES) else 0
    main_late_cycle = _state_in(main_state_label, _MAIN_TOP_RISK_STATES)
    shadow_not_strong_bull = shadow_state_label != "Strong Bull"

    top_score = (
        bearish_prob
        * _persistence_bonus(top_persistence, target=60)
        * (1.2 if (main_late_cycle and shadow_not_strong_bull) else 1.0)
    )

    # Boost bottom_score with turning_point's bullish_prob if it's strong
    bottom_score = max(bottom_score, bullish_prob)

    # Clamp probabilities
    bottom_prob = max(0.0, min(0.95, bottom_score)) * 100.0
    top_prob = max(0.0, min(0.95, top_score)) * 100.0

    # ── Verdict resolution ─────────────────────────────────────────────────
    if bottom_prob >= 55.0 and bottom_prob > top_prob + 5:
        verdict = "BOTTOM PROBABLE"
        verdict_color = "#22c55e"
        narrative = (
            f"Capitulation signals firing on the {('Main' if main_ci >= shadow_ci else 'Shadow')} brain "
            f"({'both' if both_zone3 else 'one'} brain in Zone 3+). "
            f"{'Z-score reversing higher.' if z_reversing else 'Awaiting LL z-score reversal for confirmation.'}"
        )
    elif top_prob >= 55.0 and top_prob > bottom_prob + 5:
        verdict = "TOP PROBABLE"
        verdict_color = "#ef4444"
        narrative = (
            f"Distribution forming on Main brain ({main_state_label} regime, "
            f"{main_persistence}d persistence). "
            f"{'Confirmed by Shadow weakness.' if shadow_not_strong_bull else 'Shadow still constructive — partial signal.'}"
        )
    elif top_prob >= 35.0 or bottom_prob >= 35.0 or main_zone.zone >= 2:
        verdict = "STRESS BUILDING"
        verdict_color = "#f59e0b"
        narrative = (
            f"Main brain in {main_state_label}, Shadow in {shadow_state_label}. "
            f"Watch for confirmation in either direction."
        )
    else:
        verdict = "NEUTRAL"
        verdict_color = "#94a3b8"
        narrative = "Both brains in benign regimes. No actionable extreme signal."

    # ── Combo gate evaluation ──────────────────────────────────────────────
    main_ci_pct  = abs(main_z)  / max(main_anchor, 1e-6) * 100.0 if main_z  < 0 else 0.0
    shad_ci_pct  = abs(shadow_z) / max(shadow_anchor, 1e-6) * 100.0 if shadow_z < 0 else 0.0
    _cm_z2 = main_ci_pct  >= 22; _cm_z3 = main_ci_pct  >= 40
    _cs_z2 = shad_ci_pct  >= 22; _cs_z3 = shad_ci_pct  >= 40
    combo_strategies = [
        ("OR — either Zone 3",             _cm_z3 or  _cs_z3,  "7/8 (88%)", "0.5%"),
        ("AND — both Zone 3",              _cm_z3 and _cs_z3,  "5/8 (62%)", "0.0%"),
        ("OR — either Zone 2",             _cm_z2 or  _cs_z2,  "8/8 (100%)","5.4%"),
        ("AND — both Zone 2",              _cm_z2 and _cs_z2,  "7/8 (88%)", "0.0% ★"),
        ("Main Z3 OR (Main Z2 & Shad Z2)", _cm_z3 or  (_cm_z2 and _cs_z2), "7/8 (88%)", "0.0% ★"),
        ("Main Z3 OR (Shad Z3 & Main Z2)", _cm_z3 or  (_cs_z3 and _cm_z2), "7/8 (88%)", "0.0% ★"),
    ]
    n_firing = sum(1 for _, active, _, _ in combo_strategies if active)
    # Boost confidence if star strategies are firing
    star_firing = any(active and "★" in fa for _, active, _, fa in combo_strategies)

    # ── Confidence ─────────────────────────────────────────────────────────
    breadth = tp.get("signal_breadth", 0.0)
    confidence = tp.get("confidence", "LOW")
    if both_zone3 and verdict == "BOTTOM PROBABLE":
        confidence = "HIGH"
    if star_firing and n_firing >= 2:
        confidence = "HIGH"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "bottom_prob": round(bottom_prob, 1),
        "top_prob": round(top_prob, 1),
        "confidence": confidence,
        "signal_breadth": breadth,
        "drivers_bullish": tp.get("drivers_bullish", []),
        "drivers_bearish": tp.get("drivers_bearish", []),
        "horizon": "30 days",
        "narrative": narrative,
        "main_zone": main_zone,
        "shadow_zone": shadow_zone,
        "main_state": main_state_label,
        "shadow_state": shadow_state_label,
        "main_persistence": main_persistence,
        "shadow_persistence": shadow_persistence,
        "fingerprints_loaded": load_fingerprints() is not None,
        "combo_strategies": combo_strategies,
        "combo_n_firing": n_firing,
        "combo_main_ci": round(main_ci_pct, 1),
        "combo_shad_ci": round(shad_ci_pct, 1),
    }


# ────────────────────────────────────────────────────────────────────────────
# Historical extrema + signal walk
# ────────────────────────────────────────────────────────────────────────────

def _find_historical_extrema(spx: pd.Series, distance_days: int = 63,
                             prominence_pct: float = 0.10) -> dict:
    """Use scipy.signal.find_peaks to identify SPX peaks and troughs.

    `distance_days` = minimum spacing (in trading days)
    `prominence_pct` = required prominence as fraction of price level
    """
    if spx is None or spx.empty:
        return {"peaks": pd.DatetimeIndex([]), "troughs": pd.DatetimeIndex([])}

    values = spx.values
    median = float(np.median(values))
    prominence = max(median * prominence_pct, 1.0)

    peak_idx, _ = find_peaks(values, distance=distance_days, prominence=prominence)
    trough_idx, _ = find_peaks(-values, distance=distance_days, prominence=prominence)

    return {
        "peaks": spx.index[peak_idx],
        "troughs": spx.index[trough_idx],
    }


def _compute_historical_signals(brain_path: pd.DataFrame, spx: pd.Series,
                                bear_states: set, top_states: set,
                                anchor: float = 0.467) -> pd.DataFrame:
    """Walk a brain's history; fire bottom and top signals per simple composite rules.

    Bottom: ll_z < -(0.40 * anchor)  — Zone 3 gate, calibrated per brain.
            40% CI = backtest-optimal threshold (75% hit rate, 0% false alarms on main brain).
    Top: state transitioned INTO a top-risk state and persisted ≥30 days.
    Returns DataFrame with columns: date, type ('bottom'|'top'), state, ll_z.
    """
    if brain_path is None or brain_path.empty:
        return pd.DataFrame()

    z = brain_path["ll_zscore"]
    state = brain_path["state_label"]
    zone3_threshold = -0.40 * max(anchor, 1e-6)

    rows = []

    # Bottom fires: cluster of consecutive Zone 3 days, fire at the local z-min
    fire_mask = (z < zone3_threshold).values
    in_cluster = False
    cluster_start = None
    for i, is_fire in enumerate(fire_mask):
        if is_fire and not in_cluster:
            in_cluster = True
            cluster_start = i
        elif not is_fire and in_cluster:
            in_cluster = False
            window = z.iloc[cluster_start:i]
            d_min = window.idxmin()
            rows.append({
                "date": d_min,
                "type": "bottom",
                "state": str(state.loc[d_min]),
                "ll_z": float(window.min()),
            })
    if in_cluster:
        window = z.iloc[cluster_start:]
        d_min = window.idxmin()
        rows.append({
            "date": d_min,
            "type": "bottom",
            "state": str(state.loc[d_min]),
            "ll_z": float(window.min()),
        })

    # Top fires: transition into top-risk state with ≥30d persistence,
    # gated by NOT having fired a top in the previous 180 days.
    last_top_date = None
    persist = 0
    prev_state = None
    for d, s in state.items():
        if s in top_states and prev_state == s:
            persist += 1
        elif s in top_states:
            persist = 1
        else:
            persist = 0

        if (s in top_states and persist == 30):
            if last_top_date is None or (d - last_top_date).days > 180:
                rows.append({
                    "date": d,
                    "type": "top",
                    "state": str(s),
                    "ll_z": float(z.loc[d]),
                })
                last_top_date = d
        prev_state = s

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _score_signal_accuracy(signals: pd.DataFrame, peaks: pd.DatetimeIndex,
                           troughs: pd.DatetimeIndex,
                           window_days: int = _HIT_WINDOW_DAYS) -> pd.DataFrame:
    """For each signal, find nearest matching extreme and compute lead/lag in days.

    A 'bottom' signal is matched against troughs; a 'top' against peaks.
    A signal is a 'hit' if the nearest matching extreme is within ±window_days.
    """
    if signals.empty:
        return signals

    rows = []
    for _, sig in signals.iterrows():
        target = troughs if sig["type"] == "bottom" else peaks
        if len(target) == 0:
            rows.append({**sig.to_dict(), "nearest_extreme": pd.NaT,
                         "lead_days": None, "hit": False})
            continue
        diffs = (target - sig["date"]).days  # positive = extreme came AFTER signal
        # Take signed minimum-abs
        idx_min = int(np.argmin(np.abs(diffs)))
        nearest = target[idx_min]
        lead = int(diffs[idx_min])  # >0 = brain led the extreme; <0 = lagged
        hit = abs(lead) <= window_days
        rows.append({**sig.to_dict(), "nearest_extreme": nearest,
                     "lead_days": lead, "hit": hit})

    return pd.DataFrame(rows)


def _summarize_accuracy(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"bottom": None, "top": None}
    out = {}
    for typ in ("bottom", "top"):
        subset = scored[scored["type"] == typ]
        if subset.empty:
            out[typ] = None
            continue
        n = len(subset)
        hits = int(subset["hit"].sum())
        leads = subset.loc[subset["hit"], "lead_days"].dropna()
        median_lead = float(np.median(leads)) if len(leads) > 0 else None
        out[typ] = {
            "n_signals": n,
            "hits": hits,
            "hit_rate_pct": round(100.0 * hits / n, 1),
            "median_lead_days": median_lead,
        }
    return out


# ────────────────────────────────────────────────────────────────────────────
# Probability time series
# ────────────────────────────────────────────────────────────────────────────

def _compute_prob_time_series(brain_path: pd.DataFrame) -> pd.DataFrame:
    """Lightweight rolling top/bottom probability for the diagnostic panel.

    Bottom prob = clamp(|min(ll_z, 0)| / brain.ci_anchor, 0, 0.95)
    Top prob    = clamp(persistence_in_top_states / 90, 0, 0.95) * partial_z_signal
    """
    if brain_path is None or brain_path.empty:
        return pd.DataFrame()

    z = brain_path["ll_zscore"]
    state = brain_path["state_label"]

    bottom = np.clip(np.abs(np.minimum(z, 0.0)) / max(get_ci_anchor(), 1e-6), 0.0, 0.95)

    # Top: rolling count of consecutive days in top-risk regimes
    in_top = state.isin(_MAIN_TOP_RISK_STATES | {"Crisis"}).astype(int)
    persistence = in_top.groupby((in_top != in_top.shift()).cumsum()).cumsum()
    top_persist_norm = np.clip(persistence / 60.0, 0.0, 1.0)
    # Scale by an exit-pressure proxy: higher when z is deteriorating
    z_pressure = np.clip(np.abs(np.minimum(z, 0.0)) / 0.30, 0.0, 1.5)
    top = np.clip(top_persist_norm * z_pressure, 0.0, 0.95)

    return pd.DataFrame({
        "bottom_prob": bottom * 100,
        "top_prob": top * 100,
    }, index=brain_path.index)


# ────────────────────────────────────────────────────────────────────────────
# Renderers
# ────────────────────────────────────────────────────────────────────────────

def _render_verdict_card(v: dict) -> None:
    bp, tp = v["bottom_prob"], v["top_prob"]
    bigger = max(bp, tp)
    higher_color = "#22c55e" if bp >= tp else "#ef4444"

    st.markdown(
        f'<div style="background:#0a0d12;border:2px solid {v["verdict_color"]};'
        f'border-radius:8px;padding:18px 22px;margin-bottom:14px;'
        f'box-shadow:0 0 12px {v["verdict_color"]}22;">'
        f'<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:24px;align-items:center;">'

        # VERDICT column
        f'<div>'
        f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;margin-bottom:4px;">TODAY · {v["horizon"]} OUTLOOK</div>'
        f'<div style="font-size:24px;font-weight:900;color:{v["verdict_color"]};line-height:1.05;">'
        f'{v["verdict"]}</div>'
        f'<div style="font-size:11px;color:#94a3b8;margin-top:6px;line-height:1.4;">{v["narrative"]}</div>'
        f'</div>'

        # PROBABILITIES column
        f'<div>'
        f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.12em;margin-bottom:4px;">'
        f'PROBABILITY</div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<span style="color:#22c55e;font-size:11px;font-weight:700;">BOTTOM</span>'
        f'<span style="color:#22c55e;font-size:22px;font-weight:900;">{bp:.0f}%</span></div>'
        f'<div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
        f'<div style="height:100%;width:{bp}%;background:#22c55e;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:4px;">'
        f'<span style="color:#ef4444;font-size:11px;font-weight:700;">TOP</span>'
        f'<span style="color:#ef4444;font-size:22px;font-weight:900;">{tp:.0f}%</span></div>'
        f'<div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
        f'<div style="height:100%;width:{tp}%;background:#ef4444;"></div></div>'
        f'</div>'
        f'</div>'

        # CONFIDENCE column
        f'<div>'
        f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.12em;margin-bottom:4px;">'
        f'CONFIDENCE</div>'
        f'<div style="font-size:22px;font-weight:900;color:#94a3b8;">{v["confidence"]}</div>'
        f'<div style="font-size:11px;color:#64748b;margin-top:4px;">'
        f'Signal breadth: {v["signal_breadth"]:.0f}%</div>'
        f'<div style="font-size:10px;color:#475569;margin-top:8px;">'
        f'Main: <span style="color:{v["main_zone"].color};font-weight:700;">'
        f'Z{v["main_zone"].zone}</span> · {v["main_state"]} ({v["main_persistence"]}d)<br>'
        f'Shadow: <span style="color:{v["shadow_zone"].color};font-weight:700;">'
        f'Z{v["shadow_zone"].zone}</span> · {v["shadow_state"]} ({v["shadow_persistence"]}d)'
        f'</div>'
        f'</div>'

        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_drivers(drivers_bearish: list, drivers_bullish: list,
                    fingerprints_loaded: bool) -> None:
    if not fingerprints_loaded:
        st.info(
            "**Drivers panel limited**: turning-point fingerprints not calibrated. "
            "Run `services.turning_point.build_turning_point_fingerprints()` to enable "
            "per-signal driver attribution. Verdict above still works — it falls back "
            "to brain-only mode."
        )
        return

    c_bear, c_bull = st.columns(2)
    with c_bear:
        st.markdown(
            f'<div style="font-size:11px;color:#ef4444;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:6px;">BEARISH DRIVERS</div>',
            unsafe_allow_html=True,
        )
        if not drivers_bearish:
            st.markdown('<div style="color:#475569;font-size:11px;">No bearish signals firing.</div>',
                        unsafe_allow_html=True)
        else:
            for drv in drivers_bearish[:8]:
                name = SIGNAL_NAMES.get(drv.get("signal", ""), drv.get("signal", "?"))
                weight = drv.get("weight", 0.0)
                similar = drv.get("similar_to", "")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'background:#0f172a;border-left:3px solid #ef4444;'
                    f'padding:5px 10px;margin-bottom:3px;font-size:11px;color:#cbd5e1;">'
                    f'<span>{name}</span>'
                    f'<span style="color:#94a3b8;">w={weight:.2f}'
                    f'{" · " + similar if similar else ""}</span></div>',
                    unsafe_allow_html=True,
                )
    with c_bull:
        st.markdown(
            f'<div style="font-size:11px;color:#22c55e;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:6px;">BULLISH DRIVERS</div>',
            unsafe_allow_html=True,
        )
        if not drivers_bullish:
            st.markdown('<div style="color:#475569;font-size:11px;">No bullish signals firing.</div>',
                        unsafe_allow_html=True)
        else:
            for drv in drivers_bullish[:8]:
                name = SIGNAL_NAMES.get(drv.get("signal", ""), drv.get("signal", "?"))
                weight = drv.get("weight", 0.0)
                similar = drv.get("similar_to", "")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'background:#0f172a;border-left:3px solid #22c55e;'
                    f'padding:5px 10px;margin-bottom:3px;font-size:11px;color:#cbd5e1;">'
                    f'<span>{name}</span>'
                    f'<span style="color:#94a3b8;">w={weight:.2f}'
                    f'{" · " + similar if similar else ""}</span></div>',
                    unsafe_allow_html=True,
                )


def _render_signal_fire_chart(spx: pd.Series, signals: pd.DataFrame,
                              peaks: pd.DatetimeIndex, troughs: pd.DatetimeIndex,
                              brain_label: str) -> go.Figure:
    fig = go.Figure()
    chart_start = signals["date"].min() if not signals.empty else spx.index[0]
    spx_clip = spx.loc[chart_start:]
    fig.add_trace(go.Scatter(
        x=spx_clip.index, y=spx_clip.values, mode="lines",
        name="SPX (Log)",
        line=dict(color="#B0B8C5", width=1.0),
        hovertemplate="%{x|%Y-%m-%d}<br>SPX $%{y:,.2f}<extra></extra>",
    ))

    # Actual peaks/troughs (gray vertical lines)
    chart_start = spx_clip.index[0]
    chart_end = spx_clip.index[-1]
    for d in peaks:
        if chart_start <= d <= chart_end:
            fig.add_vline(x=d, line=dict(color="#475569", dash="dot", width=1), opacity=0.35)
    for d in troughs:
        if chart_start <= d <= chart_end:
            fig.add_vline(x=d, line=dict(color="#475569", dash="dot", width=1), opacity=0.35)

    if not signals.empty:
        # Bottom signals
        bot = signals[signals["type"] == "bottom"]
        if not bot.empty:
            spx_at = spx.reindex(bot["date"]).values
            fig.add_trace(go.Scatter(
                x=bot["date"], y=spx_at, mode="markers", name="Bottom Signal",
                marker=dict(symbol="triangle-up", size=11, color="#22c55e",
                            line=dict(color="#FFFFFF", width=0.6)),
                hovertemplate="%{x|%Y-%m-%d}<br>Bottom signal<br>%{customdata}<extra></extra>",
                customdata=[f"state={s}, z={z:.3f}" for s, z in zip(bot["state"], bot["ll_z"])],
            ))
        # Top signals
        top = signals[signals["type"] == "top"]
        if not top.empty:
            spx_at = spx.reindex(top["date"]).values
            fig.add_trace(go.Scatter(
                x=top["date"], y=spx_at, mode="markers", name="Top Signal",
                marker=dict(symbol="triangle-down", size=11, color="#ef4444",
                            line=dict(color="#FFFFFF", width=0.6)),
                hovertemplate="%{x|%Y-%m-%d}<br>Top signal<br>%{customdata}<extra></extra>",
                customdata=[f"state={s}, z={z:.3f}" for s, z in zip(top["state"], top["ll_z"])],
            ))

    fig.update_yaxes(type="log", title_text="SPX (Log)")
    fig.update_xaxes(title_text="Year")
    apply_dark_layout(
        fig, height=440, showlegend=True, hovermode="x unified",
        margin=dict(l=70, r=40, t=70, b=50),
        title=dict(
            text=f"<b>{brain_label} — historical signal fires vs SPX peaks/troughs</b>",
            x=0.5, xanchor="center",
            font=dict(color=COLORS["bloomberg_orange"], size=13,
                      family="JetBrains Mono, Consolas, monospace"),
        ),
        legend=dict(x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(14,17,23,0.75)", bordercolor=COLORS["border"], borderwidth=1,
                    font=dict(size=10)),
    )
    return fig


def _render_prob_time_series(brain_path: pd.DataFrame, spx: pd.Series, brain_label: str) -> go.Figure:
    probs = _compute_prob_time_series(brain_path)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Faint SPX background
    spx_clip = spx.loc[spx.index >= probs.index[0]] if not probs.empty else spx
    fig.add_trace(go.Scatter(
        x=spx_clip.index, y=spx_clip.values, mode="lines", name="SPX (Log)",
        line=dict(color="#475569", width=0.8), opacity=0.5,
        hovertemplate="%{x|%Y-%m-%d}<br>SPX $%{y:,.2f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=probs.index, y=probs["bottom_prob"], mode="lines",
        name="Bottom Probability", line=dict(color="#22c55e", width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Bottom %{y:.0f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=probs.index, y=probs["top_prob"], mode="lines",
        name="Top Probability", line=dict(color="#ef4444", width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Top %{y:.0f}%<extra></extra>",
    ), secondary_y=False)

    fig.add_hline(y=70, line=dict(color="#fbbf24", dash="dash", width=1), secondary_y=False)
    fig.add_hline(y=50, line=dict(color="#475569", dash="dot", width=1), secondary_y=False)

    fig.update_yaxes(title_text="Probability (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="SPX (Log)", type="log", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Year")

    apply_dark_layout(
        fig, height=380, showlegend=True, hovermode="x unified",
        margin=dict(l=70, r=70, t=70, b=50),
        title=dict(
            text=f"<b>{brain_label} — Top vs Bottom probability over time</b>",
            x=0.5, xanchor="center",
            font=dict(color=COLORS["bloomberg_orange"], size=13,
                      family="JetBrains Mono, Consolas, monospace"),
        ),
        legend=dict(x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(14,17,23,0.75)", bordercolor=COLORS["border"], borderwidth=1,
                    font=dict(size=10)),
    )
    return fig


def _render_brain_performance(main_state, shadow_state, top_sig: Optional[dict], verdict: Optional[dict]) -> None:
    """Compact brain performance summary — one card per brain."""
    from services.hmm_regime import get_ci_anchor
    from services.hmm_shadow import load_shadow_brain as _lsb

    _main_ll_z   = getattr(main_state,   "ll_zscore", None) if main_state   else None
    _shad_ll_z   = getattr(shadow_state, "ll_zscore", None) if shadow_state else None
    _main_anchor = get_ci_anchor()
    _shad_brain  = _lsb()
    _shad_anchor = getattr(_shad_brain, "ci_anchor", 5.0) if _shad_brain else 5.0

    main_ci    = abs(_main_ll_z)  / max(_main_anchor, 1e-6) * 100 if _main_ll_z  is not None else 0.0
    shad_ci    = abs(_shad_ll_z)  / max(_shad_anchor, 1e-6) * 100 if _shad_ll_z  is not None else 0.0
    top_firing = bool(top_sig and top_sig.get("sig_and"))
    top_roll   = (top_sig or {}).get("ll_z_roll", 0.0)
    top_days   = (top_sig or {}).get("days_in_stress", 0)
    top_regime = (top_sig or {}).get("regime_label", "—")
    combo_main = (verdict or {}).get("combo_main_ci", 0.0)
    combo_shad = (verdict or {}).get("combo_shad_ci", 0.0)
    combo_on   = combo_main >= 22.0 and combo_shad >= 22.0

    # Gate thresholds
    main_gate_z  = -0.40 * _main_anchor   # ll_z needed to open Z3
    shad_gate_z  = -0.40 * _shad_anchor

    def _ci_color(ci):
        return "#ef4444" if ci >= 40 else "#f59e0b" if ci >= 22 else "#22c55e"

    def _ci_zone(ci):
        return "Z3" if ci >= 40 else "Z2" if ci >= 22 else "Z1"

    def _card(title, purpose, hit, fa, lead, gate_label, status_html, detail_rows, border_color):
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:3px 0;border-bottom:1px solid #0f172a;">'
            f'<span style="font-size:9px;color:#475569;">{k}</span>'
            f'<span style="font-size:9px;font-weight:700;color:{vc};">{v}</span>'
            f'</div>'
            for k, v, vc in detail_rows
        )
        return f"""
<div style="background:#0a0f1a;border:1px solid #1e293b;border-top:3px solid {border_color};
            border-radius:6px;padding:10px 12px;">
  <div style="font-size:10px;color:#94a3b8;font-weight:800;letter-spacing:0.1em;
              margin-bottom:2px;">{title}</div>
  <div style="font-size:8px;color:#475569;margin-bottom:8px;">{purpose}</div>
  <div style="margin-bottom:8px;">{status_html}</div>
  {rows_html}
  <div style="font-size:7px;color:#334155;margin-top:6px;line-height:1.7;">
    Gate: {gate_label}
  </div>
</div>"""

    # ── Top Brain card ────────────────────────────────────────────────────
    top_color  = "#f59e0b" if top_firing else "#334155"
    top_status = (
        f'<span style="font-size:18px;font-weight:900;color:#f59e0b;">FIRING</span>'
        f'<span style="font-size:9px;color:#f59e0b;margin-left:6px;">{top_days}d active</span>'
        if top_firing else
        f'<span style="font-size:18px;font-weight:900;color:#334155;">QUIET</span>'
    )
    top_card = _card(
        "TOP BRAIN", "Early top detection (VIX · NFCI · BAA10Y · T10Y3M)",
        "5/8 (62%)", "4", "107d avg",
        "regime ∈ {Late Cycle, Stress} AND 40d ll_z roll < -0.20",
        top_status,
        [
            ("Hit rate",    "5/8 (62%)",          "#64748b"),
            ("False alarms","4",                   "#64748b"),
            ("Avg lead",    "107 days",            "#64748b"),
            ("40d LL roll", f"{top_roll:.3f}",     "#f59e0b" if top_firing else "#64748b"),
            ("Regime",      top_regime,            "#ef4444" if top_regime in {"Stress","Crisis","Late Cycle","Early Stress"} else "#22c55e"),
        ],
        top_color,
    )

    # ── Main Brain card ───────────────────────────────────────────────────
    main_cc    = _ci_color(main_ci)
    main_zone  = _ci_zone(main_ci)
    main_status = (
        f'<span style="font-size:18px;font-weight:900;color:{main_cc};">'
        f'{main_zone} &nbsp; {main_ci:.1f}%</span>'
    )
    main_card = _card(
        "MAIN BRAIN", "Regime classification + crisis gate (10 FRED features)",
        "7/8 (88%)", "0 ★", "coincident",
        f"CI% ≥ 40% (ll_z < {main_gate_z:.2f})",
        main_status,
        [
            ("Hit rate",    "7/8 (88%)",                    "#64748b"),
            ("False alarms","0 ★",                          "#22c55e"),
            ("CI%",         f"{main_ci:.1f}%",              main_cc),
            ("ll_z today",  f"{_main_ll_z:+.4f}" if _main_ll_z is not None else "—", "#64748b"),
            ("Gate opens",  f"ll_z < {main_gate_z:.2f}",   "#ef4444"),
        ],
        main_cc,
    )

    # ── Shadow Brain card ─────────────────────────────────────────────────
    shad_cc    = _ci_color(shad_ci)
    shad_zone  = _ci_zone(shad_ci)
    shad_status = (
        f'<span style="font-size:18px;font-weight:900;color:{shad_cc};">'
        f'{shad_zone} &nbsp; {shad_ci:.1f}%</span>'
    )
    shad_card = _card(
        "SHADOW BRAIN", "Price-action stress / bottom detector (SPX returns + VIX)",
        "7/8 (88%)", "0 ★", "coincident",
        f"CI% ≥ 22% (ll_z < {shad_gate_z:.2f})",
        shad_status,
        [
            ("Hit rate",    "7/8 (88%)",                    "#64748b"),
            ("False alarms","0 ★",                          "#22c55e"),
            ("CI%",         f"{shad_ci:.1f}%",              shad_cc),
            ("ll_z today",  f"{_shad_ll_z:+.4f}" if _shad_ll_z is not None else "—", "#64748b"),
            ("Gate opens",  f"ll_z < {shad_gate_z:.2f}",   "#ef4444"),
        ],
        shad_cc,
    )

    # ── Combo card ────────────────────────────────────────────────────────
    combo_color  = "#22c55e" if combo_on else "#334155"
    combo_status = (
        f'<span style="font-size:18px;font-weight:900;color:#22c55e;">ACTIVE</span>'
        if combo_on else
        f'<span style="font-size:18px;font-weight:900;color:#334155;">QUIET</span>'
    )
    combo_card = _card(
        "COMBO (M + S)", "Confirmed capitulation — bottom signal",
        "7/8 (88%)", "0 ★", "coincident",
        "Main CI% ≥ 22% AND Shadow CI% ≥ 22% simultaneously",
        combo_status,
        [
            ("Hit rate",      "7/8 (88%)",            "#64748b"),
            ("False alarms",  "0 ★",                  "#22c55e"),
            ("Main CI%",      f"{combo_main:.1f}%",   _ci_color(combo_main)),
            ("Shadow CI%",    f"{combo_shad:.1f}%",   _ci_color(combo_shad)),
            ("Both ≥ 22%?",   "YES" if combo_on else "NO", "#22c55e" if combo_on else "#334155"),
        ],
        combo_color,
    )

    st.markdown(
        '<div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;margin:0 0 8px 0;">BRAIN PERFORMANCE SUMMARY</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(top_card,   unsafe_allow_html=True)
    with c2: st.markdown(main_card,  unsafe_allow_html=True)
    with c3: st.markdown(shad_card,  unsafe_allow_html=True)
    with c4: st.markdown(combo_card, unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:8px;color:#334155;margin:4px 0 12px 0;">'
        '★ = zero false alarms &nbsp;·&nbsp; backtest validated 2012–2026 &nbsp;·&nbsp;'
        ' CI% anchors differ per brain — numbers not cross-comparable</div>',
        unsafe_allow_html=True,
    )


def _render_cycle_ladder(top_sig: Optional[dict], verdict: Optional[dict], main_state) -> None:
    """3-stage market cycle escalation ladder."""
    # Stage 1 — Top Brain
    s1_on = bool(top_sig and top_sig.get("sig_and"))
    s1_days = (top_sig or {}).get("days_in_stress", 0)
    s1_regime = (top_sig or {}).get("regime_label", "—")

    # Stage 2 — Main Brain Zone 3
    from services.hmm_regime import get_ci_anchor
    _main_ll_z = getattr(main_state, "ll_zscore", None) if main_state else None
    _anchor = get_ci_anchor()
    main_ci = abs(_main_ll_z) / max(_anchor, 1e-6) * 100 if _main_ll_z is not None else 0.0
    s2_on = main_ci >= 40.0

    # Stage 3 — Main + Shadow combo Zone 2 (both ≥ 22%)
    combo_main_ci = (verdict or {}).get("combo_main_ci", 0.0)
    combo_shad_ci = (verdict or {}).get("combo_shad_ci", 0.0)
    s3_on = combo_main_ci >= 22.0 and combo_shad_ci >= 22.0

    def _dot(on, color="#ef4444"):
        bg = color if on else "#1e293b"
        return (f'<div style="width:10px;height:10px;border-radius:50%;'
                f'background:{bg};flex-shrink:0;margin-top:2px;"></div>')

    def _stage(num, label, action, tip, on, color, detail=""):
        dot = _dot(on, color)
        text_col = color if on else "#334155"
        status = f'<span style="color:{color};font-weight:800;">ACTIVE</span>' if on else '<span style="color:#334155;">QUIET</span>'
        return f"""
<div style="display:flex;gap:10px;padding:10px 12px;border-radius:5px;
            background:{""+color+"11" if on else "#0a0f1a"};
            border:1px solid {""+color+"44" if on else "#1e293b"};margin-bottom:6px;">
  {dot}
  <div style="flex:1;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.08em;">
        STAGE {num} &nbsp;·&nbsp; {label}</span>
      {status}
    </div>
    <div style="font-size:13px;font-weight:800;color:{text_col};margin:2px 0;">{action}</div>
    {f'<div style="font-size:9px;color:{color};margin-bottom:2px;">{detail}</div>' if detail and on else ""}
    <div style="font-size:8px;color:#334155;line-height:1.7;">{tip}</div>
  </div>
</div>"""

    s1_detail = f"Regime: {s1_regime} · {s1_days}d active · 40d LL roll firing" if s1_on else ""
    s2_detail = f"Main CI% = {main_ci:.1f}% — crash gate open" if s2_on else ""
    s3_detail = f"Main CI={combo_main_ci:.1f}% · Shadow CI={combo_shad_ci:.1f}% — both ≥ 22%" if s3_on else ""

    html = f"""
<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;
            padding:14px 16px;margin:0 0 14px 0;">
  <div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.12em;
              text-transform:uppercase;margin-bottom:10px;">
    MARKET CYCLE LADDER &nbsp;·&nbsp;
    <span style="color:#334155;font-weight:400;font-size:9px;">
      where are we in the cycle?
    </span>
  </div>
  {_stage(1, "TOP BRAIN · macro drift",
          "Trim longs · tighten stops · raise cash",
          "Top Brain fires when macro slowly rots: credit spreads widen, conditions tighten, VIX creeps up over 40+ days. Avg 107-day lead before the actual top. Not the crash gate — gives time to act.",
          s1_on, "#f59e0b", s1_detail)}
  {_stage(2, "MAIN BRAIN · CI% Zone 3 ≥ 40%",
          "Full defense · crash is underway",
          "Crisis gate open. Main Brain sees macro deterioration beyond its worst training-day baseline. 75% historical crash detection. Reduce exposure aggressively, hold cash/hedges.",
          s2_on, "#ef4444", s2_detail)}
  {_stage(3, "MAIN + SHADOW COMBO · both Zone 2 ≥ 22%",
          "Watch for bottom · start rebuilding",
          "Both brains in stress simultaneously — historically marks capitulation. 100% crash detection, 0% false alarms. When combo fires AND price action stabilises, begin scaling back in.",
          s3_on, "#22c55e", s3_detail)}
  <div style="font-size:7px;color:#1e293b;border-top:1px solid #1e293b;
              padding-top:6px;margin-top:4px;line-height:1.8;">
    Stage 1 → trim early &nbsp;·&nbsp; Stage 2 → full defense &nbsp;·&nbsp;
    Stage 3 → rebuild &nbsp;·&nbsp; Zone 3 overrides Stage 3 — don't rebuild while crash gate is open
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_top_brain_card(sig: dict) -> None:
    """Render the Top Brain macro-drift card above the Main+Shadow combo gate."""
    firing     = sig["sig_and"]
    roll       = sig["ll_z_roll"]
    fill       = sig["roll_fill_pct"]
    regime     = sig["regime_label"]
    days_on    = sig["days_in_stress"]
    ci_anchor  = sig["ci_anchor"]
    ll_z       = sig["ll_z"]

    # Colors
    gate_color   = "#ef4444" if firing else "#22c55e"
    gate_label   = "FIRING" if firing else "QUIET"
    regime_color = "#ef4444" if regime in _TOP_LATE_LABELS else "#22c55e"

    # Meter fill color: green → amber → red
    if fill < 50:
        meter_color = "#22c55e"
    elif fill < 100:
        meter_color = "#f59e0b"
    else:
        meter_color = "#ef4444"

    meter_pct = min(fill, 100.0)

    days_html = (
        f'<div style="font-size:9px;color:#f59e0b;margin-top:3px;">'
        f'{days_on}d active</div>'
        if firing and days_on > 0 else ""
    )

    html = f"""
<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;
            padding:14px 16px;margin:12px 0 4px 0;">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.12em;
                text-transform:uppercase;">TOP BRAIN &mdash; MACRO DRIFT DETECTOR</div>
    <div style="background:{gate_color}22;border:1px solid {gate_color}55;
                border-radius:4px;padding:2px 10px;">
      <span style="font-size:11px;font-weight:800;color:{gate_color};
                   letter-spacing:0.08em;">{gate_label}</span>
    </div>
  </div>

  <!-- Three columns -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:12px;">

    <!-- Col 1: Macro Drift Meter -->
    <div>
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:4px;">
        40-DAY LL ROLL
      </div>
      <div style="background:#1e293b;border-radius:3px;height:7px;position:relative;overflow:hidden;">
        <div style="height:7px;border-radius:3px;width:{meter_pct:.1f}%;
                    background:{meter_color};transition:width 0.3s;"></div>
        <!-- Threshold marker at 100% = threshold reached -->
        <div style="position:absolute;top:0;right:0;width:2px;height:7px;
                    background:#ef444488;"></div>
      </div>
      <div style="margin-top:4px;">
        <span style="font-size:14px;font-weight:800;color:{meter_color};">{roll:.3f}</span>
        <span style="font-size:9px;color:#475569;margin-left:4px;">/ thresh {_LL_ROLL_THRESH}</span>
      </div>
      <div style="font-size:8px;color:#334155;margin-top:1px;">
        today ll_z = {ll_z:+.3f}
      </div>
      <div style="font-size:8px;color:#334155;margin-top:2px;">
        {_LL_ROLL_WINDOW}-day mean &lt; {_LL_ROLL_THRESH} to fire
      </div>
    </div>

    <!-- Col 2: Regime state -->
    <div>
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:4px;">
        REGIME
      </div>
      <div style="font-size:18px;font-weight:800;color:{regime_color};line-height:1.1;">
        {regime}
      </div>
      <div style="font-size:8px;color:#475569;margin-top:4px;">
        Gate: Late Cycle / Stress / Crisis
      </div>
      {days_html}
    </div>

    <!-- Col 3: Signal status -->
    <div>
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:4px;">
        TOP SIGNAL (sig_and)
      </div>
      <div style="font-size:18px;font-weight:800;color:{gate_color};line-height:1.1;">
        {"FIRING" if firing else "QUIET"}
      </div>
      <div style="font-size:8px;color:#475569;margin-top:4px;">
        {_BT_HITS}/{_BT_PEAKS} hits &middot; {_BT_FA} FA &middot; {_BT_AVG_LEAD}d lead avg
      </div>
      <div style="font-size:8px;color:#334155;margin-top:1px;">
        F1=0.556 (validated 2012&ndash;2026)
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div style="font-size:8px;color:#334155;border-top:1px solid #1e293b;
              padding-top:5px;line-height:1.7;">
    Features: VIX &middot; NFCI &middot; BAA10Y &middot; T10Y3M
    &nbsp;&middot;&nbsp;
    Gate: regime &isin; &#123;Late Cycle, Stress&#125; AND {_LL_ROLL_WINDOW}-day ll_z roll &lt; {_LL_ROLL_THRESH}
    &nbsp;&middot;&nbsp; ci_anchor = {ci_anchor:.3f}
  </div>

  <!-- Greyed tip -->
  <div style="margin-top:8px;padding:7px 10px;background:#0a0f1a;border-radius:4px;
              border:1px solid #1e293b;">
    <div style="font-size:8px;color:#334155;font-weight:700;letter-spacing:0.08em;
                margin-bottom:4px;">HOW TO USE</div>
    <div style="font-size:8px;color:#334155;line-height:1.8;">
      <span style="color:#3d5a80;">&#9650; Top Brain FIRING</span>
      &nbsp;&rarr;&nbsp; start trimming longs, tighten stops, raise cash &nbsp;&middot;&nbsp;
      <span style="color:#334155;">107-day avg lead gives time to act</span>
      <br>
      <span style="color:#1e4060;">&#9632; CI% Zone 3 (&ge;40%)</span>
      &nbsp;&rarr;&nbsp; crash underway &mdash; full defense mode
      <br>
      <span style="color:#2d3748;font-style:italic;">
        Top Brain fires early &mdash; don&apos;t use it as the crisis gate.
        It catches slow distribution tops; Main + Shadow catch the crash itself.
      </span>
    </div>
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_top_brain_expander(top_path: pd.DataFrame, spx: pd.Series) -> None:
    """Render the Top Brain historical sig_and fire chart + detail table."""
    from tools.top_signal_search import (
        _rising_edges, _KNOWN_PEAKS, _PEAK_LEAD_WINDOW, _PEAK_TRAIL_WINDOW,
        _detect_spx_peaks,
    )

    if top_path is None or top_path.empty:
        return

    ts, te = top_path.index[0], top_path.index[-1]

    # Build peaks (same list as the backtest)
    hardcoded = [pd.Timestamp(d) for d, _ in _KNOWN_PEAKS if ts <= pd.Timestamp(d) <= te]
    auto = _detect_spx_peaks(spx, ts)
    all_peaks: set = set(hardcoded)
    for ap in auto:
        if not any(abs((ap - hp).days) < 60 for hp in all_peaks):
            all_peaks.add(ap)
    peaks = sorted(all_peaks)
    peak_windows = [
        (p - pd.Timedelta(days=_PEAK_LEAD_WINDOW),
         p + pd.Timedelta(days=_PEAK_TRAIL_WINDOW), p)
        for p in peaks
    ]

    fires = _rising_edges(top_path["sig_and"])

    # Build detail rows
    rows = []
    for fire in fires:
        if fire not in top_path.index:
            continue
        r = top_path.loc[fire]
        nearest = min(peaks, key=lambda p: abs((p - fire).days)) if peaks else None
        lead = (nearest - fire).days if nearest else None
        in_win = nearest and any(ws <= fire <= peak for ws, we, peak in peak_windows)
        rows.append({
            "Fire date":    fire.strftime("%Y-%m-%d"),
            "Regime":       r["state_label"],
            "ll_z":         f"{r['ll_zscore']:.3f}",
            "40d roll":     f"{r['ll_z_roll']:.3f}",
            "Nearest peak": nearest.strftime("%Y-%m-%d") if nearest else "—",
            "Lead (d)":     f"{lead:+d}" if lead is not None else "—",
            "Hit":          "✓" if in_win else "✗",
        })

    detail_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Plotly chart: SPX + signal fire markers
    spx_clip = spx.loc[ts:te] if ts in spx.index or te in spx.index else spx
    spx_clip = spx_clip.loc[ts:te]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spx_clip.index, y=np.log(spx_clip.values),
        mode="lines", line=dict(color="#334155", width=1),
        name="SPX (log)", hovertemplate="%{x|%Y-%m-%d}<extra></extra>",
    ))

    if fires:
        fire_dates = [f for f in fires if ts <= f <= te]
        fire_y = [float(np.log(spx_clip.loc[f])) if f in spx_clip.index else None
                  for f in fire_dates]
        fig.add_trace(go.Scatter(
            x=fire_dates, y=fire_y,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
            name="sig_and fire",
            hovertemplate="%{x|%Y-%m-%d}<extra>TOP signal</extra>",
        ))

    for p in peaks:
        if ts <= p <= te:
            fig.add_vline(x=p, line=dict(color="#94a3b8", width=1, dash="dot"))

    apply_dark_layout(fig)
    fig.update_layout(
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        yaxis_title="log SPX",
    )

    with st.expander("📋 Top Brain — sig_and historical fires · detail", expanded=False):
        st.markdown(
            f'<div style="font-size:10px;color:#94a3b8;margin-bottom:6px;">'
            f'Red ▼ = sig_and fire &nbsp;·&nbsp; Gray dotted = known SPX peak &nbsp;·&nbsp; '
            f'Validated: {_BT_HITS}/{_BT_PEAKS} hits · {_BT_FA} FA · {_BT_AVG_LEAD}d avg lead'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        if not detail_df.empty:
            st.dataframe(detail_df, use_container_width=True, hide_index=True)


def _render_combo_gate(verdict: dict) -> None:
    """Render the Main + Shadow combo gate block."""
    strategies  = verdict.get("combo_strategies", [])
    n_firing    = verdict.get("combo_n_firing", 0)
    main_ci     = verdict.get("combo_main_ci", 0.0)
    shad_ci     = verdict.get("combo_shad_ci", 0.0)

    if not strategies:
        return

    if n_firing >= 4:
        gate_color = "#ef4444"; gate_label = "MULTIPLE GATES OPEN"
    elif n_firing >= 2:
        gate_color = "#f59e0b"; gate_label = "STRESS CONFIRMED"
    elif n_firing == 1:
        gate_color = "#f59e0b"; gate_label = "EARLY WARNING"
    else:
        gate_color = "#22c55e"; gate_label = "ALL QUIET"

    def _ci_color(ci):
        return "#ef4444" if ci >= 40 else "#f59e0b" if ci >= 22 else "#22c55e"

    def _zone_label(ci):
        return "Zone 3" if ci >= 40 else "Zone 2" if ci >= 22 else "Zone 1"

    rows = ""
    for sname, active, det, fa in strategies:
        row_bg  = "rgba(239,68,68,0.10)" if active else "transparent"
        dot_col = "#ef4444" if active else "#1e293b"
        star    = "★" in fa
        fa_clean = fa.replace(" ★", "")
        star_html = '<span style="color:#f59e0b;"> ★</span>' if star else ""
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="padding:3px 8px;color:#94a3b8;font-size:10px;">'
            f'<span style="color:{dot_col};">&#9679;</span> {sname}</td>'
            f'<td style="padding:3px 8px;text-align:center;font-size:10px;'
            f'font-weight:700;color:{"#f87171" if active else "#334155"};">'
            f'{"FIRING" if active else "—"}</td>'
            f'<td style="padding:3px 8px;text-align:center;font-size:10px;color:#475569;">{det}</td>'
            f'<td style="padding:3px 8px;text-align:center;font-size:10px;color:#334155;">'
            f'{fa_clean}{star_html}</td>'
            f'</tr>'
        )

    html = f"""
<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;
            padding:14px 16px;margin:12px 0 4px 0;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.12em;
                text-transform:uppercase;">MAIN + SHADOW COMBO GATE</div>
    <div style="background:{gate_color}22;border:1px solid {gate_color}55;
                border-radius:4px;padding:2px 10px;">
      <span style="font-size:11px;font-weight:800;color:{gate_color};
                   letter-spacing:0.08em;">{gate_label}</span>
    </div>
  </div>

  <div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-end;">
    <div style="flex:1;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:3px;">MAIN BRAIN CI%</div>
      <div style="background:#1e293b;border-radius:3px;height:7px;position:relative;overflow:hidden;">
        <div style="height:7px;border-radius:3px;width:{min(main_ci,100):.1f}%;
                    background:{_ci_color(main_ci)};"></div>
        <div style="position:absolute;top:0;left:22%;width:1px;height:7px;background:#334155;"></div>
        <div style="position:absolute;top:0;left:40%;width:1px;height:7px;background:#ef444466;"></div>
      </div>
      <div style="font-size:11px;font-weight:700;color:{_ci_color(main_ci)};margin-top:2px;">
        {main_ci:.1f}% <span style="font-size:8px;color:#475569;font-weight:400;">{_zone_label(main_ci)}</span>
      </div>
    </div>
    <div style="flex:1;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:3px;">SHADOW BRAIN CI%</div>
      <div style="background:#1e293b;border-radius:3px;height:7px;position:relative;overflow:hidden;">
        <div style="height:7px;border-radius:3px;width:{min(shad_ci,100):.1f}%;
                    background:{_ci_color(shad_ci)};"></div>
        <div style="position:absolute;top:0;left:22%;width:1px;height:7px;background:#334155;"></div>
        <div style="position:absolute;top:0;left:40%;width:1px;height:7px;background:#ef444466;"></div>
      </div>
      <div style="font-size:11px;font-weight:700;color:{_ci_color(shad_ci)};margin-top:2px;">
        {shad_ci:.1f}% <span style="font-size:8px;color:#475569;font-weight:400;">{_zone_label(shad_ci)}</span>
      </div>
    </div>
    <div style="text-align:center;min-width:64px;padding-bottom:2px;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:2px;">FIRING</div>
      <div style="font-size:28px;font-weight:800;line-height:1;color:{gate_color};">{n_firing}</div>
      <div style="font-size:8px;color:#334155;">of 6</div>
    </div>
  </div>

  <table style="width:100%;border-collapse:collapse;margin-bottom:6px;">
    <thead>
      <tr style="border-bottom:1px solid #1e293b;">
        <th style="padding:3px 8px;text-align:left;font-size:9px;color:#334155;letter-spacing:0.08em;">STRATEGY</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;color:#334155;letter-spacing:0.08em;">NOW</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;color:#334155;letter-spacing:0.08em;">DETECTION</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;color:#334155;letter-spacing:0.08em;">FALSE ALARMS</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <div style="font-size:8px;color:#334155;line-height:1.6;margin-top:4px;">
    Backtest: 8 crashes 2012–2026 · 1,098 normal-market days · ★ = best tradeoff (88% detection, 0% FA)
    &nbsp;·&nbsp; Zone 2 = CI &ge;22% · Zone 3 = CI &ge;40%
  </div>
  <div style="font-size:8px;color:#475569;line-height:1.7;margin-top:3px;
              border-top:1px solid #1e293b;padding-top:5px;">
    The one crash missed in all 88% strategies is <span style="color:#64748b;">2022-01 Rate Shock</span> — the macro brain never registered it
    (z=0, bull regime throughout), and the shadow brain didn't spike either. That's a structural blind spot:
    slow Fed rate hikes don't create LL spikes in either model.
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Main render
# ────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        f'<h2 style="color:{COLORS["bloomberg_orange"]};margin:0 0 4px 0;'
        f'font-family:JetBrains Mono,Consolas,monospace;">TOPS &amp; BOTTOMS PREDICTOR</h2>'
        f'<div style="color:{COLORS["text_dim"]};font-size:12px;margin-bottom:14px;">'
        f"Live decision tool combining Main + Shadow brain readings into a single "
        f"top/bottom verdict, with a historical accuracy backtest."
        f"</div>",
        unsafe_allow_html=True,
    )

    main_brain = load_hmm_brain()
    shadow_brain = load_shadow_brain()

    # Reuse the check-engine banner from the regime chart
    import os
    _data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    main_status = _brain_health(main_brain, os.path.join(_data_dir, "hmm_state_history.json"), "Main")
    shadow_status = _brain_health(shadow_brain, os.path.join(_data_dir, "hmm_shadow_history.json"), "Shadow")
    _render_status_banner(main_status, shadow_status)

    if main_brain is None and shadow_brain is None:
        st.warning("No trained brains found. Train both brains in the QIR module first.")
        return

    main_state = load_current_hmm_state()
    shadow_state = load_current_shadow_state()

    top_brain = load_top_brain()

    with st.spinner("Loading brain history and SPX data..."):
        main_key   = getattr(main_brain,   "trained_at", "") if main_brain   else ""
        shadow_key = getattr(shadow_brain, "trained_at", "") if shadow_brain else ""
        top_key    = getattr(top_brain,    "trained_at", "") if top_brain    else ""
        main_path   = _load_main_path(main_key)     if main_brain   else None
        shadow_path = _load_shadow_path(shadow_key) if shadow_brain else None
        top_path    = _load_top_path(top_key)       if top_brain    else None
        spx = _load_spx_close()

    # ── SECTION 1 — TODAY'S VERDICT ─────────────────────────────────────────
    if main_state is not None or shadow_state is not None:
        verdict = _compute_today_verdict(main_state, shadow_state, main_path, shadow_path, shadow_brain=shadow_brain)

        # Cycle ladder + brain performance — always first things visible
        try:
            _top_sig_for_ladder = compute_top_signal_today(top_brain) if top_brain else None
            _render_brain_performance(main_state, shadow_state, _top_sig_for_ladder, verdict)
            _render_cycle_ladder(_top_sig_for_ladder, verdict, main_state)
        except Exception as _e:
            st.warning(f"Brain summary error: {_e}")

        _render_verdict_card(verdict)
        _render_drivers(verdict["drivers_bearish"], verdict["drivers_bullish"],
                        verdict["fingerprints_loaded"])
        # Top Brain card — above the Main+Shadow combo gate
        if top_brain is not None:
            try:
                top_sig = compute_top_signal_today(top_brain)
                if top_sig is not None:
                    _render_top_brain_card(top_sig)
                else:
                    st.warning("Top Brain: signal returned None — check FRED data.")
            except Exception as _top_err:
                st.warning(f"Top Brain card error: {_top_err}")
        else:
            st.info("Top Brain not trained — run `python tools/train_top_brain.py`.")
        _render_combo_gate(verdict)
    else:
        st.info("Score both brains in the QIR module first to see today's verdict.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2 — HISTORICAL BACKTEST ─────────────────────────────────────
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:12px;'
        f'text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin:8px 0;">'
        f"HISTORICAL BACKTEST — DOES THE SYSTEM ACTUALLY WORK?"
        f"</div>",
        unsafe_allow_html=True,
    )

    extrema = _find_historical_extrema(spx)

    for path, brain, brain_label, bear_states, top_states in [
        (shadow_path, shadow_brain, "Shadow Brain", _SHADOW_BEAR_STATES, _SHADOW_TOP_RISK_STATES),
        (main_path, main_brain, "Main Brain", _MAIN_BEAR_STATES, _MAIN_TOP_RISK_STATES),
    ]:
        if path is None or path.empty or brain is None:
            continue

        brain_anchor = getattr(brain, "ci_anchor", get_ci_anchor())
        signals = _compute_historical_signals(path, spx, bear_states, top_states, anchor=brain_anchor)
        # Filter extrema to brain's date range
        brain_peaks = extrema["peaks"][(extrema["peaks"] >= path.index[0]) & (extrema["peaks"] <= path.index[-1])]
        brain_troughs = extrema["troughs"][(extrema["troughs"] >= path.index[0]) & (extrema["troughs"] <= path.index[-1])]

        scored = _score_signal_accuracy(signals, brain_peaks, brain_troughs)
        summary = _summarize_accuracy(scored)

        st.plotly_chart(
            _render_signal_fire_chart(spx, scored, brain_peaks, brain_troughs, brain_label),
            use_container_width=True,
        )

        # Summary stats
        bot = summary.get("bottom")
        top = summary.get("top")
        cols = st.columns(2)
        with cols[0]:
            if bot:
                lead_str = f"median lead {bot['median_lead_days']:+.0f}d" if bot["median_lead_days"] is not None else "—"
                st.markdown(
                    f'<div style="background:#0f172a;border-left:3px solid #22c55e;padding:8px 12px;'
                    f'font-size:11px;color:#cbd5e1;margin-bottom:6px;">'
                    f'<b style="color:#22c55e;">Bottom signals:</b> '
                    f'{bot["hits"]}/{bot["n_signals"]} hit ({bot["hit_rate_pct"]:.0f}%) · {lead_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with cols[1]:
            if top:
                lead_str = f"median lead {top['median_lead_days']:+.0f}d" if top["median_lead_days"] is not None else "—"
                st.markdown(
                    f'<div style="background:#0f172a;border-left:3px solid #ef4444;padding:8px 12px;'
                    f'font-size:11px;color:#cbd5e1;margin-bottom:6px;">'
                    f'<b style="color:#ef4444;">Top signals:</b> '
                    f'{top["hits"]}/{top["n_signals"]} hit ({top["hit_rate_pct"]:.0f}%) · {lead_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Detail table — limit to 25 most recent
        if not scored.empty:
            display = scored.tail(25).copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            display["nearest_extreme"] = display["nearest_extreme"].dt.strftime("%Y-%m-%d").fillna("—")
            display["lead_days"] = display["lead_days"].apply(
                lambda x: f"{x:+d}" if pd.notnull(x) else "—"
            )
            display["hit"] = display["hit"].map({True: "✓", False: "✗"})
            display["ll_z"] = display["ll_z"].apply(lambda x: f"{x:.3f}")
            display.columns = ["Signal date", "Type", "State at signal", "LL z",
                               "Nearest extreme", "Lead (d)", "Hit"]
            with st.expander(f"📋 {brain_label} — last 25 signals · detail", expanded=False):
                st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Top Brain expander — always shown if brain exists
    if top_path is not None and not top_path.empty:
        try:
            _render_top_brain_expander(top_path, spx)
        except Exception as _exp_err:
            with st.expander("📋 Top Brain — sig_and historical fires · detail", expanded=False):
                st.warning(f"Top Brain expander error: {_exp_err}")
    elif top_brain is not None:
        with st.expander("📋 Top Brain — sig_and historical fires · detail", expanded=False):
            st.info(f"top_path unavailable (top_path is None: {top_path is None})")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3 — PROBABILITY TIME SERIES ─────────────────────────────────
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:12px;'
        f'text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin:8px 0;">'
        f"PROBABILITY TIME SERIES — DIAGNOSTIC VIEW"
        f"</div>",
        unsafe_allow_html=True,
    )
    if shadow_path is not None and not shadow_path.empty:
        st.plotly_chart(
            _render_prob_time_series(shadow_path, spx, "Shadow Brain"),
            use_container_width=True,
        )
    if main_path is not None and not main_path.empty:
        st.plotly_chart(
            _render_prob_time_series(main_path, spx, "Main Brain"),
            use_container_width=True,
        )

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#0a0d12;border:1px solid {COLORS["border"]};border-radius:5px;'
        f'padding:10px 14px;margin-top:14px;font-size:10px;color:#64748b;line-height:1.6;">'
        f'<b style="color:#94a3b8;">Honest design constraints:</b><br>'
        f'• Bottom signals lead actual SPX troughs by 0–90 days historically (median ~30d). '
        f'Tops are flagged by a 1–6 month <i>window</i>, not a date.<br>'
        f'• Probabilities are bounded ≤95% — the model never claims certainty.<br>'
        f'• Backtest covers each brain\'s training window (~2012+ for Main, ~1960+ for Shadow). '
        f'Out-of-sample hit rate may differ.<br>'
        f'• This is a decision aid, not investment advice.'
        f'</div>',
        unsafe_allow_html=True,
    )
