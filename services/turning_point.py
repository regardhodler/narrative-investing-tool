"""
Reversal Risk — HMM backbone + signal-breadth z-score hybrid.

Uses a noisy-OR combination of:
  P_hmm   = HMM transition_risk_1m  (base rate of leaving current regime)
  P_signal = fraction of 10 signals within 1.5σ of historical peak/trough values

Formula:  P_turn = 1 - (1 - P_hmm) * (1 - P_signal * 0.7)

Fingerprints are calibrated once from 8 historical crash scenarios.
"""

from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FP_PATH = _DATA_DIR / "turning_point_fingerprints.json"

# The 10 signals we track, with human-readable names
SIGNAL_NAMES = {
    "macro_score":    "Macro Score",
    "regime_velocity":"Regime Velocity",
    "entropy":        "Entropy",
    "ll_zscore":      "LL Z-Score",
    "hmm_confidence": "HMM Confidence",
    "conviction":     "Conviction",
    "vix":            "VIX",
    "credit_hy_z":    "Credit HY Spreads",
    "yield_curve_z":  "Yield Curve (10Y-2Y)",
    "nfci_z":         "Financial Conditions",
}

# Short crash labels for "similar to" display
_CRASH_LABELS = {
    "dotcom_2000":     "Dotcom",
    "gfc_2008":        "GFC",
    "eu_debt_2011":    "EU Debt",
    "china_2015":      "China",
    "volmageddon_2018":"Volmageddon",
    "covid_2020":      "COVID",
    "rate_shock_2022": "Rate Shock",
    "carry_unwind_2024":"Carry Unwind",
}


def load_fingerprints() -> dict | None:
    """Load calibrated fingerprints from JSON. Returns None if not available."""
    if not _FP_PATH.exists():
        return None
    try:
        with open(_FP_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def build_turning_point_fingerprints() -> dict:
    """One-time calibration: reconstruct signals at all 8 peak/trough dates.

    Computes mean + std per signal across crashes, plus per-crash raw values
    for "similar crashes" lookup.  Saves to data/turning_point_fingerprints.json.
    """
    from services.backtest_engine import (
        CRASH_SCENARIOS, reconstruct_regime_at_date, reconstruct_hmm_at_date,
        _load_all_historical_data, _build_hmm_historical_inference,
    )

    data = _load_all_historical_data()
    # Build full HMM inference (needs dates, probs, state_idxs, etc.)
    hmm_data = _build_hmm_historical_inference()

    peak_records = {}   # crash_key -> {signal_name: value}
    trough_records = {}

    for key, scenario in CRASH_SCENARIOS.items():
        for label, date_key, records in [("peak", "peak", peak_records),
                                          ("trough", "trough", trough_records)]:
            date = scenario[date_key]
            # Get regime reconstruction
            regime = reconstruct_regime_at_date(date, data)
            hmm = reconstruct_hmm_at_date(date, hmm_data)

            # Extract signal values
            macro_score = regime.get("macro_score", 50)
            vix_val = regime.get("vix") or 20.0

            # Extract FRED z-scores from signal_details
            sig_z = {}
            for d in regime.get("signal_details", []):
                sig_z[d["name"]] = d.get("z_score", 0.0)

            # Velocity: need prior date's regime score
            from datetime import datetime, timedelta
            dt = datetime.strptime(date, "%Y-%m-%d")
            prior_date = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
            try:
                prior_regime = reconstruct_regime_at_date(prior_date, data)
                velocity = regime.get("regime_score", 0) - prior_regime.get("regime_score", 0)
            except Exception:
                velocity = 0.0

            # Conviction (simplified, same as backtest)
            conviction = min(100, max(0, round(abs(macro_score - 50) * 2, 0)))

            vals = {
                "macro_score":     macro_score,
                "regime_velocity": round(velocity, 4),
                "entropy":         hmm.get("entropy", 0.0) if hmm else 0.0,
                "ll_zscore":       hmm.get("ll_zscore", 0.0) if hmm else 0.0,
                "hmm_confidence":  hmm.get("confidence", 0.5) if hmm else 0.5,
                "conviction":      conviction,
                "vix":             vix_val,
                "credit_hy_z":     sig_z.get("credit_hy", 0.0),
                "yield_curve_z":   sig_z.get("yield_curve", 0.0),
                "nfci_z":          sig_z.get("fci", 0.0),
            }
            records[key] = vals

    # Compute stats per signal
    fingerprints = {"peaks": {}, "troughs": {}, "per_crash": {"peaks": peak_records, "troughs": trough_records}}
    for sig_name in SIGNAL_NAMES:
        peak_vals = [peak_records[k].get(sig_name, 0.0) for k in peak_records]
        trough_vals = [trough_records[k].get(sig_name, 0.0) for k in trough_records]

        fingerprints["peaks"][sig_name] = {
            "mean": round(float(np.mean(peak_vals)), 4),
            "std":  round(max(float(np.std(peak_vals)), 0.01), 4),
        }
        fingerprints["troughs"][sig_name] = {
            "mean": round(float(np.mean(trough_vals)), 4),
            "std":  round(max(float(np.std(trough_vals)), 0.01), 4),
        }

    # Save
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_FP_PATH, "w") as f:
        json.dump(fingerprints, f, indent=2)

    return fingerprints


def _find_similar_crashes(signal_name: str, value: float, fingerprints: dict,
                          side: str = "peaks") -> list[str]:
    """Find which historical crashes had similar readings for this signal."""
    per_crash = fingerprints.get("per_crash", {}).get(side, {})
    stats = fingerprints.get(side, {}).get(signal_name, {})
    std = stats.get("std", 1.0)
    similar = []
    for crash_key, vals in per_crash.items():
        crash_val = vals.get(signal_name, 0.0)
        if abs(value - crash_val) < 1.5 * std:
            label = _CRASH_LABELS.get(crash_key, crash_key)
            similar.append(label)
    return similar[:3]  # cap at 3


def compute_turning_point_probability(
    macro_score: float,
    regime_velocity: float,
    entropy: float,
    ll_zscore: float,
    hmm_confidence: float,
    conviction: float,
    vix: float,
    credit_hy_z: float,
    yield_curve_z: float,
    nfci_z: float,
    transition_risk_1m: float,
    hmm_state_label: str,
) -> dict:
    """Compute reversal risk probability using HMM + signal breadth hybrid.

    Returns dict with bearish/bullish probabilities, confidence, and driver attribution.
    """
    fp = load_fingerprints()

    signals = {
        "macro_score":     macro_score,
        "regime_velocity": regime_velocity,
        "entropy":         entropy,
        "ll_zscore":       ll_zscore,
        "hmm_confidence":  hmm_confidence,
        "conviction":      conviction,
        "vix":             vix,
        "credit_hy_z":     credit_hy_z,
        "yield_curve_z":   yield_curve_z,
        "nfci_z":          nfci_z,
    }

    # Count available signals (some may be None/NaN in degraded mode)
    available = {k: v for k, v in signals.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    n_available = len(available)

    if fp is None or n_available == 0:
        # Fallback: HMM-only
        p_hmm = transition_risk_1m or 0.0
        bearish_states = {"Bull", "Neutral", "Early Stress"}
        if hmm_state_label in bearish_states:
            bear_p = round(p_hmm * 100, 1)
            bull_p = 2.0
        else:
            bear_p = 2.0
            bull_p = round(p_hmm * 100, 1)
        return {
            "bearish_prob": bear_p, "bullish_prob": bull_p,
            "confidence": "LOW", "n_bearish_signals": 0, "n_bullish_signals": 0,
            "drivers_bearish": [], "drivers_bullish": [],
            "hmm_base_prob": round(p_hmm * 100, 1), "signal_breadth": 0.0,
        }

    # Measure distance from peak/trough fingerprints.
    # Use WEIGHTED breadth: signals that are far from the OPPOSITE side's
    # fingerprint get more weight (they're more discriminating).
    # A signal near BOTH peak and trough fingerprints (entropy, LL) gets
    # down-weighted because it can't distinguish direction.
    bearish_active, bullish_active = [], []

    for name, val in available.items():
        peak_stats = fp["peaks"].get(name, {})
        trough_stats = fp["troughs"].get(name, {})
        if not peak_stats or not trough_stats:
            continue

        peak_dist = abs(val - peak_stats["mean"]) / max(peak_stats["std"], 0.01)
        trough_dist = abs(val - trough_stats["mean"]) / max(trough_stats["std"], 0.01)

        # Discriminability: how much closer to one side than the other.
        # If close to peak but far from trough → strong bearish signal.
        # If close to both → ambiguous, low weight.
        if peak_dist < 1.5:
            # Weight by how far from the opposite (trough) side
            discrim = min(1.0, max(0.1, (trough_dist - peak_dist) / 2.0))
            similar = _find_similar_crashes(name, val, fp, "peaks")
            bearish_active.append({
                "signal": SIGNAL_NAMES.get(name, name),
                "value": round(val, 2),
                "distance": round(peak_dist, 2),
                "weight": round(discrim, 2),
                "similar_crashes": similar,
            })

        if trough_dist < 1.5:
            discrim = min(1.0, max(0.1, (peak_dist - trough_dist) / 2.0))
            similar = _find_similar_crashes(name, val, fp, "troughs")
            bullish_active.append({
                "signal": SIGNAL_NAMES.get(name, name),
                "value": round(val, 2),
                "distance": round(trough_dist, 2),
                "weight": round(discrim, 2),
                "similar_crashes": similar,
            })

    # Weighted signal breadth: sum of weights / max possible weight
    bear_weight_sum = sum(d["weight"] for d in bearish_active)
    bull_weight_sum = sum(d["weight"] for d in bullish_active)
    p_signal_bear = bear_weight_sum / n_available
    p_signal_bull = bull_weight_sum / n_available

    # HMM base probability: direction-aware
    p_hmm = transition_risk_1m or 0.0
    bearish_states = {"Bull", "Neutral", "Early Stress"}
    if hmm_state_label in bearish_states:
        p_hmm_bear = p_hmm
        p_hmm_bull = 0.05  # low base rate when not in stress
    else:
        p_hmm_bear = 0.05
        p_hmm_bull = p_hmm

    # ── Tiered probability with corroboration requirement ────────────────
    # Signal breadth alone can push probability to ~35% max ("conditions
    # look peak/trough-like"). To go higher, need corroborating evidence
    # from HMM or macro direction:
    #
    # Tier 1 (0-35%):  Signal breadth only — "resembles historical reversal points"
    # Tier 2 (35-60%): + HMM confirms (transition_risk > 0.5 for direction)
    # Tier 3 (60%+):   + Macro direction confirms (macro deteriorating for bearish,
    #                    macro crushed for bullish)
    #
    # This means calm bull markets cap at ~35% bearish (signal breadth matches
    # but no HMM/macro confirmation), while real peaks with multiple
    # confirming signals can reach 80%+.

    # Base: signal breadth capped at 35%
    p_bear_base = min(0.35, p_signal_bear * 0.7)
    p_bull_base = min(0.35, p_signal_bull * 0.7)

    # Tier 2 boost: HMM transition risk confirms direction
    _bear_hmm_boost = 0.0
    _bull_hmm_boost = 0.0
    if p_hmm_bear > 0.10:  # meaningful transition risk
        _bear_hmm_boost = min(0.25, (p_signal_bear * 0.5) * (p_hmm_bear / 0.50))
    if p_hmm_bull > 0.10:
        _bull_hmm_boost = min(0.25, (p_signal_bull * 0.5) * (p_hmm_bull / 0.50))

    # Tier 3 boost: macro direction confirms
    _bear_macro_boost = 0.0
    _bull_macro_boost = 0.0
    # Bearish turn confirmation: macro is elevated AND conviction is low (< 20)
    if macro_score is not None and conviction is not None and macro_score > 55 and conviction < 20:
        _bear_macro_boost = min(0.20, p_signal_bear * 0.4)
    # Also boost if macro is in the peak range (50-65) with active credit/NFCI stress
    elif macro_score is not None and macro_score > 48 and any(d.get("signal") in ("Credit HY Spreads", "Financial Conditions")
                                   and d.get("weight", 0) >= 0.5 for d in bearish_active):
        _bear_macro_boost = min(0.15, p_signal_bear * 0.3)

    # Bullish turn confirmation: macro is crushed
    if macro_score is not None and macro_score < 35:
        _bull_macro_boost = min(0.25, p_signal_bull * 0.5)
    elif macro_score is not None and macro_score < 42 and any(d.get("signal") in ("VIX", "Credit HY Spreads")
                                   and d.get("weight", 0) >= 0.5 for d in bullish_active):
        _bull_macro_boost = min(0.15, p_signal_bull * 0.3)

    p_bear = p_bear_base + _bear_hmm_boost + _bear_macro_boost
    p_bull = p_bull_base + _bull_hmm_boost + _bull_macro_boost

    # Macro-direction damper: suppress the contradictory side.
    if macro_score is not None and macro_score > 50:
        bull_damp = max(0.15, 1.0 - (macro_score - 50) / 25)
        p_bull *= bull_damp
    elif macro_score is not None and macro_score < 40:
        bear_damp = max(0.15, 1.0 - (40 - macro_score) / 25)
        p_bear *= bear_damp

    # Hard cap at 95%
    p_bear = min(0.95, p_bear)
    p_bull = min(0.95, p_bull)

    # Confidence
    n_active = max(len(bearish_active), len(bullish_active))
    conf_score = (1 - entropy) * (n_active / n_available)
    conf_label = "HIGH" if conf_score > 0.6 else ("MED" if conf_score > 0.3 else "LOW")

    # Sort drivers by distance (closest = strongest match)
    bearish_active.sort(key=lambda x: x["distance"])
    bullish_active.sort(key=lambda x: x["distance"])

    return {
        "bearish_prob": round(p_bear * 100, 1),
        "bullish_prob": round(p_bull * 100, 1),
        "confidence": conf_label,
        "n_bearish_signals": len(bearish_active),
        "n_bullish_signals": len(bullish_active),
        "drivers_bearish": bearish_active,
        "drivers_bullish": bullish_active,
        "hmm_base_prob": round(max(p_hmm_bear, p_hmm_bull) * 100, 1),
        "signal_breadth": round(max(p_signal_bear, p_signal_bull) * 100, 1),
    }
