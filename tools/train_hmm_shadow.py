"""
Train the Shadow HMM brain on ^GSPC daily log returns (1960 -> today).

Run:
    python tools/train_hmm_shadow.py

Outputs:
    data/hmm_shadow_brain.json
    data/hmm_shadow_result.pickle

After training, run tools/backtest_shadow_ci.py to calibrate the CI anchor
and fill in the crash-probability lookup table.
"""
from __future__ import annotations

import os
import sys
import time

# Allow running as `python tools/train_hmm_shadow.py` from repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.hmm_shadow import train_shadow_hmm


def main() -> int:
    t0 = time.time()
    print("[shadow] training 6-regime MarkovRegression on ^GSPC log returns 1960->today …")
    brain = train_shadow_hmm()
    elapsed = time.time() - t0

    print(f"[shadow] trained in {elapsed:.1f}s")
    print(f"[shadow] training window: {brain.training_start} -> {brain.training_end}")
    print(f"[shadow] ll_baseline_mean = {brain.ll_baseline_mean:.4f}  std = {brain.ll_baseline_std:.4f}")
    print()
    print("Regime summary (sorted by conditional mean return):")
    print(f"  {'Label':<14} {'Mean %/day':>12} {'Variance':>12}")
    order = sorted(range(brain.k_regimes), key=lambda i: brain.regime_means[i])
    for i in order:
        print(f"  {brain.state_labels[i]:<14} {brain.regime_means[i]:>12.4f} {brain.regime_variances[i]:>12.4f}")
    print()
    print("Next step: python tools/backtest_shadow_ci.py  (calibrates CI anchor + crash bins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
