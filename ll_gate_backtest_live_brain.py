#!/usr/bin/env python3
"""
LL Gate Backtest — Using the LIVE brain (same scale as production).

Scores each trading day from 2012–2026 using the live HMM brain,
computing LL z-scores on the same scale the user sees in the app.
Then tests the LL gate against every crash in that window.
"""
import json, os, sys
import numpy as np
import pandas as pd

# ── Load live brain ──────────────────────────────────────────────────────────
BRAIN_PATH = os.path.join("data", "hmm_brain.json")
with open(BRAIN_PATH) as f:
    brain_raw = json.load(f)

n_states = brain_raw["n_states"]
feature_names = brain_raw["feature_names"]
ll_baseline_mean = brain_raw["ll_baseline_mean"]
ll_baseline_std = brain_raw["ll_baseline_std"]
print(f"Live brain: {n_states} states, trained {brain_raw['training_start']} to {brain_raw['training_end']}")
print(f"LL baseline: mean={ll_baseline_mean:.6f}, std={ll_baseline_std:.6f}")
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
df = _build_feature_matrix(lookback_years=15)

# Ensure columns match brain
for col in feature_names:
    if col not in df.columns:
        df[col] = 0.0
df = df.dropna()
X_full = df[feature_names].values.astype(np.float64)
dates = df.index.tolist()

print(f"Feature matrix: {len(dates)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print()

# ── Score day-by-day using expanding window ──────────────────────────────────
# For each day T, score X[0:T+1] to get LL, then z-score it
# This mimics what the live system does (scores entire history up to today)
print("Scoring day-by-day (expanding window)... this takes a minute...")

MIN_WINDOW = 252  # Need at least 1 year of data before scoring
results = []

for i in range(MIN_WINDOW, len(X_full)):
    X_window = X_full[:i+1]
    date_str = dates[i].strftime("%Y-%m-%d")
    
    try:
        ll_total = float(model.score(X_window))
        ll_per_obs = ll_total / len(X_window)
        ll_zscore = (ll_per_obs - ll_baseline_mean) / max(ll_baseline_std, 1e-6)
        
        # Also get state probabilities for entropy
        posteriors = model.predict_proba(X_window)
        today_probs = posteriors[-1]
        state_idx = int(np.argmax(today_probs))
        
        from scipy.stats import entropy as shannon_entropy
        raw_entropy = float(shannon_entropy(today_probs))
        max_entropy = float(np.log(n_states))
        norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        
        results.append({
            "date": date_str,
            "ll_zscore": round(ll_zscore, 4),
            "ll_per_obs": round(ll_per_obs, 6),
            "entropy": round(norm_entropy, 4),
            "state_idx": state_idx,
            "state_label": brain_raw["state_labels"][state_idx],
        })
    except Exception as e:
        results.append({
            "date": date_str,
            "ll_zscore": 0.0,
            "ll_per_obs": 0.0,
            "entropy": 0.0,
            "state_idx": -1,
            "state_label": "Error",
        })
    
    if (i - MIN_WINDOW) % 500 == 0:
        print(f"  Scored {i - MIN_WINDOW + 1}/{len(X_full) - MIN_WINDOW} days... (current: {date_str}, LL z={ll_zscore:.3f})")

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

thresholds = [-1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
for t in thresholds:
    count = len([x for x in ll_vals if x < t])
    pct = count / len(ll_vals) * 100
    print(f"  Below z={t}: {count} days ({pct:.1f}%)")

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

# Try multiple thresholds
for threshold in [-1.5, -2.0, -2.5, -3.0]:
    print(f"\n--- THRESHOLD: z < {threshold} ---")
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
