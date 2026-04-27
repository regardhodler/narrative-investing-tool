"""
tools/train_top_brain.py
=========================
One-shot script: trains the Top Brain (VIX+NFCI+BAA10Y+T10Y3M) and saves it
to data/hmm_top_brain.json.  Re-run whenever FRED data is refreshed.

Usage:
    python tools/train_top_brain.py
"""
from __future__ import annotations

import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings
warnings.filterwarnings("ignore")

from services.hmm_top import (
    train_top_brain, save_top_brain, compute_full_top_state_path,
    _LATE_LABELS, _LL_ROLL_THRESH, _LL_ROLL_WINDOW,
    _BT_HITS, _BT_PEAKS, _BT_HIT_PCT, _BT_FA, _BT_AVG_LEAD,
)


def main() -> int:
    print("=" * 70)
    print("  TOP BRAIN TRAINING  (VIX + NFCI + BAA10Y + T10Y3M)")
    print("=" * 70)

    brain = train_top_brain(lookback_years=15)
    save_top_brain(brain)

    print()
    print(f"  n_states      : {brain.n_states}")
    print(f"  training      : {brain.training_start} -> {brain.training_end}")
    print(f"  obs           : {brain.n_obs}")
    print(f"  ci_anchor     : {brain.ci_anchor:.4f}")
    print(f"  state_labels  : {brain.state_labels}")
    print(f"  ll_baseline   : mean={brain.ll_baseline_mean:.4f}  std={brain.ll_baseline_std:.4f}")
    print()
    print(f"  Signal gate:")
    print(f"    regime IN {sorted(_LATE_LABELS)}")
    print(f"    AND {_LL_ROLL_WINDOW}-day rolling ll_z mean < {_LL_ROLL_THRESH}")
    print()
    print(f"  Validated backtest (2012-2026, 8 SPX peaks):")
    print(f"    Hit rate     : {_BT_HITS}/{_BT_PEAKS} ({_BT_HIT_PCT}%)")
    print(f"    False alarms : {_BT_FA}")
    print(f"    Avg lead     : {_BT_AVG_LEAD} days")

    # Quick today's reading
    print()
    print("  Today's reading:")
    path = compute_full_top_state_path(brain)
    if path is not None and not path.empty:
        row = path.iloc[-1]
        today = path.index[-1].date()
        sig_and = bool(row["sig_and"])
        print(f"    Date         : {today}")
        print(f"    Regime       : {row['state_label']}")
        print(f"    ll_z         : {row['ll_zscore']:.3f}")
        print(f"    ll_z_roll    : {row['ll_z_roll']:.3f}  (thresh {_LL_ROLL_THRESH})")
        print(f"    sig_regime   : {bool(row['sig_regime'])}")
        print(f"    sig_ll       : {bool(row['sig_ll'])}")
        print(f"    sig_and      : {sig_and}  {'<-- TOP SIGNAL FIRING' if sig_and else ''}")

        # Streak
        days_on = 0
        for v in reversed(path["sig_and"].values):
            if v == 1:
                days_on += 1
            else:
                break
        if days_on:
            print(f"    Active for   : {days_on} consecutive days")
    else:
        print("    ERROR: could not compute path")

    print()
    print("  Done. Reload the Streamlit app to see the Top Brain card.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
