"""
Train the Shadow HMM brain on ^GSPC log returns + VIX (1990 -> today).

Switched from statsmodels MarkovRegression (univariate) to hmmlearn
GaussianHMM (multivariate) to support VIX as a second feature.

Run:
    python tools/train_hmm_shadow.py

Outputs:
    data/hmm_shadow_brain.json

After training, run tools/backtest_shadow_ci.py to calibrate the CI anchor
and fill in the crash-probability lookup table.
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.hmm_shadow import train_shadow_hmm


def main() -> int:
    t0 = time.time()
    print("[shadow] training 6-regime GaussianHMM on ^GSPC log returns + VIX 1990->today ...")
    brain = train_shadow_hmm()
    elapsed = time.time() - t0

    print(f"[shadow] trained in {elapsed:.1f}s")
    print(f"[shadow] features: {brain.feature_names}")
    print(f"[shadow] training window: {brain.training_start} -> {brain.training_end}")
    print(f"[shadow] ll_baseline_mean = {brain.ll_baseline_mean:.4f}  std = {brain.ll_baseline_std:.4f}")
    print(f"[shadow] ci_anchor = {brain.ci_anchor:.4f}  (auto-calibrated)")
    print()
    print("Regime summary (sorted by conditional mean SPX return):")
    print(f"  {'Label':<14} {'Mean %/day':>12} {'Variance':>12}")
    order = sorted(range(brain.n_states), key=lambda i: brain.regime_means[i])
    for i in order:
        print(f"  {brain.state_labels[i]:<14} {brain.regime_means[i]:>12.4f} {brain.regime_variances[i]:>12.6f}")
    print()
    print("Next step: python tools/backtest_shadow_ci.py  (calibrates CI anchor + crash bins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
