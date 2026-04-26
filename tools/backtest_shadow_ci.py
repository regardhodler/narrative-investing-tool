"""
Calibrate the Shadow HMM CI% anchor and crash-probability lookup bins.

Updated for GaussianHMM (SPX + VIX) — no longer uses the statsmodels pickle.

For each historical day t:
  1. Compute ll_zscore[t] using per-day emission marginals (logsumexp).
  2. Compute forward 30-day peak-to-trough drawdown from SPX close.
  3. Run a threshold sweep to find the optimal CI% gate.
  4. Update brain.ci_anchor and brain.crash_prob_bins.

Run:
    python tools/backtest_shadow_ci.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from services.hmm_shadow import (
    _BRAIN_PATH, _CALIBRATION_PATH,
    _build_shadow_features, _reconstruct_model,
    load_shadow_brain, save_shadow_brain,
)


def _forward_drawdown_pct(close: pd.Series, window: int = 30) -> pd.Series:
    out = pd.Series(index=close.index, dtype=float)
    vals = close.values
    for i in range(len(vals)):
        j = min(i + window, len(vals) - 1)
        segment = vals[i: j + 1]
        out.iloc[i] = (segment.min() / segment[0] - 1.0) * 100.0
    return out


def main() -> int:
    brain = load_shadow_brain()
    if brain is None or not brain.means:
        print("[shadow-ci] no trained GaussianHMM brain — run tools/train_hmm_shadow.py first",
              file=sys.stderr)
        return 1

    print(f"[shadow-ci] scoring {brain.n_obs} days with GaussianHMM ...")
    df, _, _ = _build_shadow_features(
        brain.training_start,
        feat_means=brain.feature_means,
        feat_stds=brain.feature_stds,
    )
    X = df.values.astype(np.float64)
    model = _reconstruct_model(brain)

    log_emit = model._compute_log_likelihood(X)
    ll_per_obs = logsumexp(log_emit, axis=1)
    ll_z = (ll_per_obs - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6)
    ll_z_series = pd.Series(ll_z, index=df.index, name="ll_zscore")

    # SPX close for drawdown calc
    import yfinance as yf
    px = yf.download("^GSPC", start=brain.training_start, progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    close = px.dropna().reindex(df.index).ffill()

    print("[shadow-ci] computing forward 30-day drawdowns ...")
    fwd_dd = _forward_drawdown_pct(close, window=30)
    crashed = (fwd_dd <= -10.0).astype(int)

    # ── Threshold sweep ────────────────────────────────────────────────────────
    anchor = brain.ci_anchor
    print(f"\n[shadow-ci] anchor = {anchor:.4f}")
    print(f"  {'gate_z':>8} {'crisis':>8} {'hits':>8} {'hitRate':>10} {'precision':>10} "
          f"{'falseAlm':>10} {'earlyWarn':>10} {'%flagged':>10}")
    print("  " + "-" * 90)

    gate_zs = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50, -0.60,
               -0.70, -0.80, -0.90, -1.00, -1.20, -1.50, -1.80, -2.00, -2.50, -3.00]

    # Identify crash episodes (forward 30d drawdown <= -10%)
    crash_dates = close.index[crashed.values == 1]
    episodes = []
    last = None
    for d in crash_dates:
        if last is None or (d - last).days > 30:
            episodes.append(d)
        last = d
    n_episodes = len(episodes)

    best_f1 = 0.0
    best_gate = -0.80

    for gz in gate_zs:
        gate_mask = ll_z_series < gz
        n_flagged = int(gate_mask.sum())
        pct_flagged = n_flagged / len(ll_z_series) * 100

        hits = 0
        false_alarms = 0
        total_lead = 0
        for ep in episodes:
            window = ll_z_series.loc[
                max(ep - pd.Timedelta(days=60), ll_z_series.index[0]):ep
            ]
            if (window < gz).any():
                hits += 1
                first_fire = window[window < gz].index[0]
                total_lead += (ep - first_fire).days

        false_alarms = max(0, n_flagged - hits)
        hit_rate = hits / n_episodes if n_episodes > 0 else 0
        precision = hits / n_flagged if n_flagged > 0 else 0
        avg_lead = total_lead / hits if hits > 0 else 0
        f1 = 2 * precision * hit_rate / (precision + hit_rate) if (precision + hit_rate) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_gate = gz

        print(f"  {gz:>8.3f} {n_flagged:>8d} {hits:>5d}/{n_episodes:<3d} "
              f"{hit_rate*100:>9.1f}% {precision*100:>9.1f}% "
              f"{false_alarms:>10d} {avg_lead:>9.1f}d {pct_flagged:>9.1f}%")

    # ── Set ci_anchor so best gate = 40% CI ───────────────────────────────────
    # Zone 3 at 40% CI is the calibrated standard. Map best_gate -> 40% CI.
    new_anchor = abs(best_gate) / 0.40
    print(f"\n[shadow-ci] BEST F1 gate: z < {best_gate:.3f}  (F1={best_f1:.3f})")
    print(f"[shadow-ci] Setting ci_anchor = {new_anchor:.4f}  (so 40% CI = z<{best_gate:.3f})")
    print(f"[shadow-ci] Old anchor: {anchor:.4f}")

    # ── Build crash_prob_bins at the new anchor ────────────────────────────────
    zone_thresholds = [
        ("Normal",       0.0,          np.inf),
        ("Zone 1",      -0.22 * new_anchor, 0.0),
        ("Zone 2",      -0.40 * new_anchor, -0.22 * new_anchor),
        ("Zone 3",      -np.inf,       -0.40 * new_anchor),
    ]
    crash_prob_bins = []
    for label, z_lo, z_hi in zone_thresholds:
        mask = (ll_z_series >= z_lo) & (ll_z_series < z_hi)
        n_obs = int(mask.sum())
        prob = float(crashed[mask].mean()) if n_obs > 0 else 0.0
        triggered = fwd_dd[mask][crashed[mask] == 1]
        exp_dd = float(triggered.mean()) if len(triggered) > 0 else 0.0
        crash_prob_bins.append({
            "label": label,
            "z_lo": float(z_lo) if np.isfinite(z_lo) else -999.0,
            "z_hi": float(z_hi) if np.isfinite(z_hi) else 999.0,
            "n_obs": n_obs,
            "prob_10pct": round(prob, 4),
            "expected_drawdown_pct": round(exp_dd, 2),
        })

    print("\n  Crash probability by zone:")
    print(f"  {'Zone':<12} {'n_obs':>8} {'prob_10%':>10} {'exp_dd':>10}")
    for b in crash_prob_bins:
        print(f"  {b['label']:<12} {b['n_obs']:>8d} {b['prob_10pct']:>10.3f} "
              f"{b['expected_drawdown_pct']:>10.2f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    brain.ci_anchor = round(new_anchor, 4)
    brain.crash_prob_bins = crash_prob_bins
    save_shadow_brain(brain)

    calibration = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "training_start": brain.training_start,
        "training_end": brain.training_end,
        "best_gate_z": best_gate,
        "best_f1": round(best_f1, 4),
        "new_ci_anchor": round(new_anchor, 4),
        "n_episodes": n_episodes,
        "crash_prob_bins": crash_prob_bins,
    }
    os.makedirs(os.path.dirname(_CALIBRATION_PATH), exist_ok=True)
    with open(_CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n[shadow-ci] calibration written to {_CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
