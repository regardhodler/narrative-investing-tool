#!/usr/bin/env python3
"""
LL Gate Backtest — Using the LIVE brain (same scale as production).

Scores each trading day from training_start to today using the live HMM brain,
computing LL z-scores on the same scale the user sees in the app.
Then tests the LL gate against every crash in that window.

Diagnostic / read-only — does NOT modify the brain or any source files.
The CI anchor is auto-calibrated at training time (brain.ci_anchor field),
so this script is purely informational. Thresholds below scale automatically
to the live brain's anchor so output stays meaningful across retrains.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ── Load live brain ──────────────────────────────────────────────────────────
BRAIN_PATH = os.path.join("data", "hmm_brain.json")
with open(BRAIN_PATH) as f:
    brain_raw = json.load(f)

n_states = brain_raw["n_states"]
feature_names = brain_raw["feature_names"]
ll_baseline_mean = brain_raw["ll_baseline_mean"]
ll_baseline_std = brain_raw["ll_baseline_std"]
ci_anchor = brain_raw.get("ci_anchor", 0.467)
print(f"Live brain: {n_states} states, trained {brain_raw['training_start']} to {brain_raw['training_end']}")
print(f"LL baseline: mean={ll_baseline_mean:.6f}, std={ll_baseline_std:.6f}")
print(f"CI anchor:   {ci_anchor:.4f}  (auto-calibrated at training time)")
print(f"Features: {feature_names}")
print()

# ── Reconstruct HMM model ───────────────────────────────────────────────────
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(n_components=n_states, covariance_type="full")
model.n_features = len(feature_names)
model.startprob_ = np.ones(n_states) / n_states
model.transmat_ = np.array(brain_raw["transmat"])
model.means_ = np.array(brain_raw["means"])
model.covars_ = np.array(brain_raw["covars"])

# ── Build feature matrix (same pipeline as live scoring) ─────────────────────
sys.path.insert(0, ".")
from services.hmm_regime import _build_feature_matrix

print("Building feature matrix (same pipeline as live)...")
# Use brain's exact lookback — same feature z-scores as live scoring.
# LL scale is only valid for in-sample data (post-training era).
# Pre-training dates (GCF, etc.) show Zone 0 "LL data not available" in the UI.
df = _build_feature_matrix(lookback_years=brain_raw.get("lookback_years", 15))

# Ensure columns match brain
for col in feature_names:
    if col not in df.columns:
        df[col] = 0.0
df = df.dropna()
X_full = df[feature_names].values.astype(np.float64)
dates = df.index.tolist()

print(f"Feature matrix: {len(dates)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print()

# ── Score every day using PER-DAY emission marginals ────────────────────────
# Match live scoring: ll_per_obs[t] = logsumexp(emit_log_probs[t]).
# This is the per-day emission likelihood (no transition compounding), which
# is what the brain's ci_anchor and the live UI both use.
print("Scoring all days using per-day emission marginals...")

posteriors_full = model.predict_proba(X_full)
state_seq = np.argmax(posteriors_full, axis=1)
log_emit = model._compute_log_likelihood(X_full)
ll_per_obs_full = logsumexp(log_emit, axis=1)
ll_z_full = (ll_per_obs_full - ll_baseline_mean) / max(ll_baseline_std, 1e-6)

from scipy.stats import entropy as shannon_entropy
max_entropy = float(np.log(n_states))

results = []
for i in range(len(X_full)):
    raw_entropy = float(shannon_entropy(posteriors_full[i]))
    norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    results.append({
        "date": dates[i].strftime("%Y-%m-%d"),
        "ll_zscore": round(float(ll_z_full[i]), 4),
        "ll_per_obs": round(float(ll_per_obs_full[i]), 6),
        "entropy": round(norm_entropy, 4),
        "state_idx": int(state_seq[i]),
        "state_label": brain_raw["state_labels"][int(state_seq[i])],
    })

print(f"  Done! Scored {len(results)} days total.")
print()

# ── Save results ─────────────────────────────────────────────────────────────
OUTPUT_PATH = "ll_gate_backtest_live_brain.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")

# ── Analyze LL distribution ──────────────────────────────────────────────────
ll_vals = [r["ll_zscore"] for r in results]
print()
print("LL Z-SCORE DISTRIBUTION (LIVE BRAIN SCALE):")
print(f"  Range: {min(ll_vals):.3f} to {max(ll_vals):.3f}")
print(f"  Mean: {sum(ll_vals)/len(ll_vals):.3f}")
print(f"  Days total: {len(ll_vals)}")
print()

# Thresholds scale to brain.ci_anchor — Zone 2 starts at 22% CI, Zone 3 at 67%.
# Mapping back to z: z_zone2 = -0.22 * anchor, z_zone3 = -0.67 * anchor, etc.
threshold_pct = [0.22, 0.40, 0.50, 0.67, 0.80, 1.00]
print(f"  (CI% gate thresholds scaled to brain.ci_anchor = {ci_anchor:.4f})")
for pct in threshold_pct:
    t = -pct * ci_anchor
    count = len([x for x in ll_vals if x < t])
    obs_pct = count / len(ll_vals) * 100
    zone = 1 if pct < 0.22 else (2 if pct < 0.67 else (3 if pct <= 1.0 else 4))
    print(f"  Below z={t:+.4f}  (CI={int(pct*100)}% | Zone {zone}): {count} days ({obs_pct:.2f}%)")

# Compute COVID-era worst z for sanity check vs ci_anchor
covid_window = [r for r in results if "2020-02-01" <= r["date"] <= "2020-04-30"]
covid_worst_z = min(r["ll_zscore"] for r in covid_window) if covid_window else None
if covid_worst_z:
    covid_ci = abs(covid_worst_z) / max(ci_anchor, 1e-6) * 100
    print(f"\n  COVID-era (Feb-Apr 2020) worst z: {covid_worst_z:.4f}  ->  CI% = {covid_ci:.1f}%")
    print(f"  Brain ci_anchor                  : {ci_anchor:.4f}  ->  CI% = 100% (calibration target)")
    if abs(abs(covid_worst_z) - ci_anchor) / max(ci_anchor, 1e-6) > 0.05:
        print(f"  NOTE: COVID worst z differs from ci_anchor by >5% — anchor likely set by a different extreme day.")
    else:
        print(f"  Anchor is calibrated against COVID-era extreme. OK.")
print(f"\n  ANCHOR HANDLING: brain.ci_anchor is now AUTO-CALIBRATED at training time.")
print(f"  Do NOT manually edit 0.467 anywhere — all consumers read brain.ci_anchor via")
print(f"  services.hmm_regime.get_ci_anchor(). Re-running this script after a retrain")
print(f"  is purely diagnostic — no source-code changes needed.")

# ── Test against crashes ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("LL GATE vs EVERY CRASH (LIVE BRAIN SCALE)")
print("=" * 60)

crashes = [
    ("2018-02 Volmageddon", "2018-02-05", "2017-12-01", "2018-03-15"),
    ("2018-12 Fed Panic", "2018-12-24", "2018-10-01", "2019-01-15"),
    ("2020-02 COVID Start", "2020-02-20", "2020-01-15", "2020-03-05"),
    ("2020-03 COVID Bottom", "2020-03-23", "2020-02-15", "2020-04-15"),
    ("2022-01 Rate Shock", "2022-01-04", "2021-11-15", "2022-02-15"),
    ("2022-06 Bear Mkt", "2022-06-16", "2022-04-01", "2022-07-15"),
    ("2022-10 Bear Bottom", "2022-10-12", "2022-08-15", "2022-11-15"),
    ("2025-04 Tariff Shock", "2025-04-07", "2025-03-01", "2025-04-14"),
]

# Try CI%-based gate thresholds, scaled to the brain's ci_anchor.
# 22% = Zone 2 entry, 50% = mid-stress, 67% = Zone 3 (Crisis Gate), 100% = anchor.
for ci_threshold in [0.22, 0.40, 0.50, 0.67, 0.80, 1.00]:
    threshold = -ci_threshold * ci_anchor
    print(f"\n--- THRESHOLD: z < {threshold:+.4f}  (CI={int(ci_threshold*100)}%) ---")
    detected = 0
    missed = 0
    
    for name, crash_date, window_start, window_end in crashes:
        most_neg = 0
        most_neg_date = ""
        fired_days = 0
        window_days = 0
        first_fire = None
        
        for r in results:
            if window_start <= r["date"] <= window_end:
                window_days += 1
                if r["ll_zscore"] < most_neg:
                    most_neg = r["ll_zscore"]
                    most_neg_date = r["date"]
                if r["ll_zscore"] < threshold:
                    fired_days += 1
                    if first_fire is None:
                        first_fire = r["date"]
        
        if fired_days > 0:
            detected += 1
            # Calculate lead time
            from datetime import datetime
            crash_dt = datetime.strptime(crash_date, "%Y-%m-%d")
            fire_dt = datetime.strptime(first_fire, "%Y-%m-%d")
            lead_days = (crash_dt - fire_dt).days
            print(f"  [DETECTED] {name}: LL hit {most_neg:.3f} on {most_neg_date}, gate open {fired_days}/{window_days} days, {lead_days}d lead time")
        else:
            missed += 1
            print(f"  [MISSED]   {name}: worst LL was {most_neg:.3f} on {most_neg_date} ({window_days} days checked)")
    
    total = detected + missed
    print(f"  DETECTION RATE: {detected}/{total} ({detected/total*100:.0f}%)")
    
    # False alarms in normal markets
    normal_periods = [
        ("2017 Bull", "2017-04-01", "2017-12-31"),
        ("2019 Bull", "2019-04-01", "2019-12-31"),
        ("2021 Bull", "2021-03-01", "2021-11-15"),
        ("2023 Bull", "2023-01-03", "2023-12-29"),
        ("2024 Bull", "2024-01-02", "2024-12-31"),
    ]
    
    false_fires = 0
    normal_days = 0
    for name, start, end in normal_periods:
        for r in results:
            if start <= r["date"] <= end:
                normal_days += 1
                if r["ll_zscore"] < threshold:
                    false_fires += 1
    
    fpr = false_fires / normal_days * 100 if normal_days > 0 else 0
    print(f"  FALSE ALARMS: {false_fires}/{normal_days} normal days ({fpr:.1f}%)")

print()
print("DONE! Check ll_gate_backtest_live_brain.json for full data.")
