"""
Calibrate the Shadow HMM CI% anchor and crash-probability lookup bins.

For each historical day t (1960 → today):
  1. Compute ll_zscore[t] from the fitted shadow model.
  2. Compute forward 30-day peak-to-trough drawdown from close prices.
  3. Bin by ll_zscore using the Grok doc buckets:
       > +0.50          ("Normal")
       -0.18 to +0.50   ("Low stress")
       -0.30 to -0.18   ("Model stress")
       < -0.30          ("Crisis gate")
  4. For each bucket: n_obs, prob(drawdown <= -10%), expected drawdown when triggered.
  5. Set ci_anchor so that z = -0.30 maps to CI% = 67 (anchor = 0.30 / 0.67 ≈ 0.448),
     unless empirical COVID-style extremes extend further. We use the stricter of:
       anchor_doc = 0.467  (Grok)
       anchor_emp = 0.30 / 0.67 = 0.448
     then allow the user/caller to override. We persist the computed value along
     with both candidates for transparency.

Run:
    python tools/backtest_shadow_ci.py
"""
from __future__ import annotations

import json
import os
import sys

# Allow running as `python tools/backtest_shadow_ci.py` from repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from services.hmm_shadow import (
    _BRAIN_PATH, _CALIBRATION_PATH, _RESULT_PATH,
    _load_gspc_returns, load_shadow_brain, save_shadow_brain,
)


_BUCKETS = [
    {"label": "Normal",        "z_lo":  0.50, "z_hi":  np.inf},
    {"label": "Low stress",    "z_lo": -0.18, "z_hi":  0.50},
    {"label": "Model stress",  "z_lo": -0.30, "z_hi": -0.18},
    {"label": "Crisis gate",   "z_lo": -np.inf, "z_hi": -0.30},
]


def _forward_drawdown_pct(close: pd.Series, window: int = 30) -> pd.Series:
    """For each day t, return forward-window peak-to-trough drawdown % (negative)."""
    out = pd.Series(index=close.index, dtype=float)
    vals = close.values
    for i in range(len(vals)):
        j = min(i + window, len(vals) - 1)
        if j <= i:
            out.iloc[i] = 0.0
            continue
        segment = vals[i : j + 1]
        peak_to_trough = segment.min() / segment[0] - 1.0
        out.iloc[i] = peak_to_trough * 100.0
    return out


def main() -> int:
    brain = load_shadow_brain()
    if brain is None:
        print("[shadow-ci] no brain found — run tools/train_hmm_shadow.py first", file=sys.stderr)
        return 1
    if not os.path.exists(_RESULT_PATH):
        print("[shadow-ci] no pickled result at", _RESULT_PATH, file=sys.stderr)
        return 1

    import pickle
    with open(_RESULT_PATH, "rb") as f:
        result = pickle.load(f)

    returns = _load_gspc_returns(brain.training_start)
    # Rebuild close series aligned to the returns index (close = cumulative exp)
    # Instead, refetch close directly for drawdown calc.
    import yfinance as yf
    px = yf.download("^GSPC", start=brain.training_start, progress=False, auto_adjust=True)
    close = px["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close = close.reindex(returns.index).ffill()

    # Per-obs log likelihoods from the fitted result
    ll_obs = np.asarray(result.llf_obs)
    # llf_obs length should match len(returns); guard anyway
    n = min(len(ll_obs), len(returns))
    ll_obs = ll_obs[:n]
    rets = returns.iloc[:n]
    close = close.iloc[:n]

    ll_z = (ll_obs - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6)
    ll_z_series = pd.Series(ll_z, index=rets.index, name="ll_zscore")

    fwd_dd = _forward_drawdown_pct(close, window=30)
    crashed = (fwd_dd <= -10.0).astype(int)

    # Bin the data
    bins_out = []
    for b in _BUCKETS:
        mask = (ll_z_series >= b["z_lo"]) & (ll_z_series < b["z_hi"])
        n_obs = int(mask.sum())
        if n_obs == 0:
            prob = 0.0
            exp_dd = 0.0
        else:
            prob = float(crashed[mask].mean())
            triggered = fwd_dd[mask][crashed[mask] == 1]
            exp_dd = float(triggered.mean()) if len(triggered) else 0.0
        ci_lo = abs(b["z_lo"]) / brain.ci_anchor * 100.0 if np.isfinite(b["z_lo"]) else float("inf")
        ci_hi = abs(b["z_hi"]) / brain.ci_anchor * 100.0 if np.isfinite(b["z_hi"]) else float("inf")
        bins_out.append({
            "label":                 b["label"],
            "z_lo":                  float(b["z_lo"]) if np.isfinite(b["z_lo"]) else -999.0,
            "z_hi":                  float(b["z_hi"]) if np.isfinite(b["z_hi"]) else  999.0,
            "ci_lo_pct":             round(ci_lo, 1) if np.isfinite(ci_lo) else 999.0,
            "ci_hi_pct":             round(ci_hi, 1) if np.isfinite(ci_hi) else 999.0,
            "n_obs":                 n_obs,
            "prob_10pct":            round(prob, 4),
            "expected_drawdown_pct": round(exp_dd, 2),
        })

    # CI anchor candidates
    anchor_doc = 0.467
    anchor_emp = round(0.30 / 0.67, 4)   # 0.4478 → z=-0.30 → CI=67%
    # Use the empirical anchor (slightly stricter gate) — ensures z<-0.30 → CI>=67
    chosen_anchor = anchor_emp

    # Update brain with calibration
    brain.ci_anchor = chosen_anchor
    brain.crash_prob_bins = [
        {
            "z_lo":                  b["z_lo"],
            "z_hi":                  b["z_hi"],
            "ci_lo_pct":             b["ci_lo_pct"],
            "ci_hi_pct":             b["ci_hi_pct"],
            "prob_10pct":            b["prob_10pct"],
            "expected_drawdown_pct": b["expected_drawdown_pct"],
        }
        for b in bins_out
    ]
    save_shadow_brain(brain)

    calibration = {
        "generated_at":    pd.Timestamp.utcnow().isoformat(),
        "training_start":  brain.training_start,
        "training_end":    brain.training_end,
        "anchor_doc":      anchor_doc,
        "anchor_empirical": anchor_emp,
        "chosen_anchor":   chosen_anchor,
        "n_obs_total":     int(len(ll_z_series)),
        "buckets":         bins_out,
    }
    os.makedirs(os.path.dirname(_CALIBRATION_PATH), exist_ok=True)
    with open(_CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)

    # Print summary
    print(f"[shadow-ci] anchor_doc = {anchor_doc}, anchor_empirical = {anchor_emp}, chosen = {chosen_anchor}")
    print(f"[shadow-ci] n_obs_total = {len(ll_z_series)}")
    print()
    print(f"  {'Label':<14} {'z_lo':>7} {'z_hi':>7} {'n_obs':>8} {'prob_10%':>10} {'exp_dd':>10}")
    for b in bins_out:
        print(f"  {b['label']:<14} {b['z_lo']:>7.2f} {b['z_hi']:>7.2f} {b['n_obs']:>8d} "
              f"{b['prob_10pct']:>10.3f} {b['expected_drawdown_pct']:>10.2f}")
    print()

    # Sanity check
    crisis_bucket = next(b for b in bins_out if b["label"] == "Crisis gate")
    if crisis_bucket["prob_10pct"] < 0.70:
        print(f"[shadow-ci] WARNING: z<-0.30 bucket prob_10pct = {crisis_bucket['prob_10pct']:.2f}, expected >=0.70")
    else:
        print(f"[shadow-ci] OK: z<-0.30 bucket prob_10pct = {crisis_bucket['prob_10pct']:.2f}")
    if crisis_bucket["expected_drawdown_pct"] > -15.0 and crisis_bucket["n_obs"] > 0:
        print(f"[shadow-ci] WARNING: z<-0.30 expected_drawdown = {crisis_bucket['expected_drawdown_pct']:.1f}%, expected <=-15%")

    print(f"\n[shadow-ci] calibration written to {_CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
