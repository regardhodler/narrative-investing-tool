"""
Calibrate Market Top/Bottom Proximity Thresholds
=================================================
Runs build_qir_snapshot() at every known peak and trough from CRASH_SCENARIOS,
collects the actual signal values, prints a calibration table, and suggests
empirically-derived thresholds.

Run standalone (no Streamlit session needed):
    python tools/calibrate_proximity.py

Output:
    - Per-crash calibration table (console)
    - Recommended threshold updates for _compute_top_bottom_proximity()
    - data/proximity_calibration.json  (persisted for reference)
"""

import sys
import os
import json

# ── Patch streamlit before ANY app imports ────────────────────────────────────
# st.cache_data becomes a no-op passthrough so functions work without a session.
from unittest.mock import MagicMock, patch

_st_mock = MagicMock()
_st_mock.cache_data = lambda *a, **kw: (lambda f: f)  # passthrough decorator
_st_mock.cache_resource = lambda *a, **kw: (lambda f: f)
_st_mock.session_state = {}
_st_mock.spinner = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
sys.modules["streamlit"] = _st_mock
# Also patch submodules that might be imported
import types
_st_mock.runtime = types.ModuleType("streamlit.runtime")
_st_mock.runtime.exists = lambda: False
sys.modules["streamlit.runtime"] = _st_mock.runtime

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Now import app modules ─────────────────────────────────────────────────────
print("Loading historical data (this may take 30-60s)...")
from services.backtest_engine import (
    CRASH_SCENARIOS,
    _load_all_historical_data,
    _build_hmm_historical_inference,
    build_qir_snapshot,
)

# ── Fetch data once ────────────────────────────────────────────────────────────
data = _load_all_historical_data()
print(f"  SPY OHLC rows: {len(data.get('spy_ohlc', [])) if data.get('spy_ohlc') is not None else 0}")
print(f"  SPY Volume rows: {len(data.get('spy_volume', [])) if data.get('spy_volume') is not None else 0}")

print("Building HMM historical inference...")
hmm_data = _build_hmm_historical_inference()
print(f"  HMM dates covered: {len(hmm_data.get('dates', [])) if hmm_data else 0}")

# ── Run snapshots at every peak and trough ─────────────────────────────────────
records = []

for key, sc in CRASH_SCENARIOS.items():
    name  = sc["name"]
    peak  = sc["peak"]
    trough = sc["trough"]

    for role, date in [("PEAK", peak), ("TROUGH", trough)]:
        print(f"  Snapshot {name} [{role}] @ {date} ...")
        try:
            snap = build_qir_snapshot(date, data, hmm_data)
            reg   = snap["regime"]
            hmm   = snap.get("hmm") or {}
            wk    = snap.get("wyckoff") or {}

            records.append({
                "scenario":    key,
                "name":        name,
                "role":        role,
                "date":        date,
                "regime_score":   round(reg.get("regime_score", 0), 4),
                "macro_score":    round(snap["macro_score"], 1),
                "tech_score":     round(snap["tech_score"], 1),
                "conviction":     round(snap["conviction"], 1),
                "entropy":        round(hmm.get("entropy", 0), 3),
                "ll_zscore":      round(hmm.get("ll_zscore", 0), 2),
                "hmm_state":      hmm.get("state_label", "N/A"),
                "top_pct":        snap["top_bottom"]["top_pct"],
                "bottom_pct":     snap["top_bottom"]["bottom_pct"],
                "top_signals":    snap["top_bottom"]["top_signals"],
                "bottom_signals": snap["top_bottom"]["bottom_signals"],
                "wyckoff_phase":  wk.get("phase", "N/A"),
                "wyckoff_sub":    wk.get("sub_phase", ""),
                "wyckoff_conf":   wk.get("confidence", 0),
                "spy_price":      round(snap.get("spy_price") or 0, 2),
                "vix":            round(snap.get("vix") or 0, 1),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            records.append({"scenario": key, "name": name, "role": role, "date": date, "error": str(e)})

# ── Print calibration table ────────────────────────────────────────────────────
print("\n" + "="*120)
print("MARKET TOP/BOTTOM PROXIMITY — CALIBRATION TABLE")
print("="*120)
header = f"{'Name':<32} {'Role':<7} {'Date':<12} {'Regime':>8} {'Macro':>6} {'Conv':>5} {'Entropy':>7} {'LL_z':>7} {'HMM State':<14} {'TOP%':>5} {'BOT%':>5} {'Wyckoff':<22}"
print(header)
print("-"*120)

peaks   = [r for r in records if r.get("role") == "PEAK"   and "error" not in r]
troughs = [r for r in records if r.get("role") == "TROUGH" and "error" not in r]

for r in records:
    if "error" in r:
        print(f"{'  '+r['name']:<32} {r['role']:<7} {r['date']:<12}  ERROR: {r['error'][:60]}")
        continue
    wk_str = f"{r['wyckoff_phase']} {r['wyckoff_sub']} ({r['wyckoff_conf']}%)"
    print(
        f"{r['name']:<32} {r['role']:<7} {r['date']:<12} "
        f"{r['regime_score']:>8.3f} {r['macro_score']:>6.1f} {r['conviction']:>5.0f} "
        f"{r['entropy']:>7.3f} {r['ll_zscore']:>7.1f} {r['hmm_state']:<14} "
        f"{r['top_pct']:>5} {r['bottom_pct']:>5} {wk_str:<22}"
    )

# ── Summary statistics ─────────────────────────────────────────────────────────
def avg(vals): return round(sum(vals)/len(vals), 3) if vals else 0
def pct(vals, key): return [r[key] for r in vals if key in r]

print("\n" + "="*120)
print("EMPIRICAL AVERAGES AT KNOWN PEAKS (n={})".format(len(peaks)))
print(f"  regime_score : {avg(pct(peaks,'regime_score'))}   (threshold currently > 0)")
print(f"  entropy      : {avg(pct(peaks,'entropy'))}   (threshold currently > 0.70)")
print(f"  conviction   : {avg(pct(peaks,'conviction'))}   (threshold currently < 25)")
print(f"  ll_zscore    : {avg(pct(peaks,'ll_zscore'))}   (threshold currently < -0.5)")
print(f"  macro_score  : {avg(pct(peaks,'macro_score'))}")
print(f"  TOP score    : {avg(pct(peaks,'top_pct'))}%  ← should be HIGH at peaks")
print(f"  BOTTOM score : {avg(pct(peaks,'bottom_pct'))}%  ← should be LOW at peaks")

print("\nEMPIRICAL AVERAGES AT KNOWN TROUGHS (n={})".format(len(troughs)))
print(f"  regime_score : {avg(pct(troughs,'regime_score'))}   (threshold currently < -0.15)")
print(f"  entropy      : {avg(pct(troughs,'entropy'))}   ")
print(f"  conviction   : {avg(pct(troughs,'conviction'))}   (threshold currently > 20)")
print(f"  ll_zscore    : {avg(pct(troughs,'ll_zscore'))}   (threshold currently < -5)")
print(f"  macro_score  : {avg(pct(troughs,'macro_score'))}   (threshold currently < 40)")
print(f"  TOP score    : {avg(pct(troughs,'top_pct'))}%  ← should be LOW at troughs")
print(f"  BOTTOM score : {avg(pct(troughs,'bottom_pct'))}%  ← should be HIGH at troughs")

# ── Signal hit rates ───────────────────────────────────────────────────────────
print("\nSIGNAL HIT RATES AT PEAKS (% of peaks where signal fired):")
sig_names = ["Regime positive", "Velocity negative", "High entropy", "Low conviction", "LL declining", "HMM stress regime"]
for sig in sig_names:
    rate = sum(1 for r in peaks if sig in r.get("top_signals", [])) / max(1, len(peaks)) * 100
    print(f"  {sig:<30}: {rate:.0f}%")

print("\nSIGNAL HIT RATES AT TROUGHS (% of troughs where signal fired):")
sig_names_b = ["Regime deep negative", "Velocity turning positive", "Macro crushed", "Conviction building", "Extreme LL stress", "HMM crisis/late cycle"]
for sig in sig_names_b:
    rate = sum(1 for r in troughs if sig in r.get("bottom_signals", [])) / max(1, len(troughs)) * 100
    print(f"  {sig:<30}: {rate:.0f}%")

# Wyckoff hit rates
print("\nWYCKOFF PHASE DISTRIBUTION AT PEAKS:")
from collections import Counter
peak_phases = Counter(r["wyckoff_phase"] for r in peaks if "wyckoff_phase" in r)
for ph, cnt in peak_phases.most_common():
    print(f"  {ph:<20}: {cnt}/{len(peaks)} ({cnt/len(peaks)*100:.0f}%)")

print("\nWYCKOFF PHASE DISTRIBUTION AT TROUGHS:")
trough_phases = Counter(r["wyckoff_phase"] for r in troughs if "wyckoff_phase" in r)
for ph, cnt in trough_phases.most_common():
    print(f"  {ph:<20}: {cnt}/{len(troughs)} ({cnt/len(troughs)*100:.0f}%)")

# ── Recommended thresholds ─────────────────────────────────────────────────────
print("\n" + "="*120)
print("RECOMMENDED THRESHOLD UPDATES")
print("="*120)

if peaks:
    avg_regime_peak   = avg(pct(peaks, "regime_score"))
    avg_entropy_peak  = avg(pct(peaks, "entropy"))
    avg_conv_peak     = avg(pct(peaks, "conviction"))
    avg_ll_peak       = avg(pct(peaks, "ll_zscore"))

    print(f"\nTOP SIGNALS (empirical averages → suggested thresholds):")
    print(f"  regime_score > 0       → regime_score > {max(0.0, round(avg_regime_peak * 0.5, 2))}  (50% of avg peak regime)")
    print(f"  entropy > 0.70         → entropy > {round(min(0.85, avg_entropy_peak * 0.9), 2)}  (90% of avg peak entropy)")
    print(f"  conviction < 25        → conviction < {round(min(40, avg_conv_peak * 1.3), 1)}  (130% of avg peak conviction)")
    print(f"  ll_z < -0.5            → ll_z < {round(max(-3.0, avg_ll_peak * 0.5), 1)}  (50% of avg peak ll_z)")

if troughs:
    avg_regime_trough = avg(pct(troughs, "regime_score"))
    avg_macro_trough  = avg(pct(troughs, "macro_score"))
    avg_conv_trough   = avg(pct(troughs, "conviction"))
    avg_ll_trough     = avg(pct(troughs, "ll_zscore"))

    print(f"\nBOTTOM SIGNALS (empirical averages → suggested thresholds):")
    print(f"  regime_score < -0.15   → regime_score < {round(avg_regime_trough * 0.5, 2)}  (50% of avg trough regime)")
    print(f"  macro_score < 40       → macro_score < {round(min(50, avg_macro_trough * 1.1), 1)}  (110% of avg trough macro)")
    print(f"  conviction > 20        → conviction > {round(max(10, avg_conv_trough * 0.7), 1)}  (70% of avg trough conviction)")
    print(f"  ll_z < -5              → ll_z < {round(min(-2, avg_ll_trough * 0.5), 1)}  (50% of avg trough ll_z)")

# ── Persist results ────────────────────────────────────────────────────────────
out_path = os.path.join(ROOT, "data", "proximity_calibration.json")
with open(out_path, "w") as f:
    json.dump({
        "generated": str(__import__("datetime").date.today()),
        "crash_scenarios": list(CRASH_SCENARIOS.keys()),
        "records": records,
        "peak_averages": {
            "regime_score": avg(pct(peaks,"regime_score")),
            "entropy":      avg(pct(peaks,"entropy")),
            "conviction":   avg(pct(peaks,"conviction")),
            "ll_zscore":    avg(pct(peaks,"ll_zscore")),
            "macro_score":  avg(pct(peaks,"macro_score")),
            "top_pct":      avg(pct(peaks,"top_pct")),
            "bottom_pct":   avg(pct(peaks,"bottom_pct")),
        },
        "trough_averages": {
            "regime_score": avg(pct(troughs,"regime_score")),
            "entropy":      avg(pct(troughs,"entropy")),
            "conviction":   avg(pct(troughs,"conviction")),
            "ll_zscore":    avg(pct(troughs,"ll_zscore")),
            "macro_score":  avg(pct(troughs,"macro_score")),
            "top_pct":      avg(pct(troughs,"top_pct")),
            "bottom_pct":   avg(pct(troughs,"bottom_pct")),
        },
    }, f, indent=2)
print(f"\nCalibration data saved to: {out_path}")
print("Done. Use the RECOMMENDED THRESHOLD UPDATES above to update _compute_top_bottom_proximity().")
