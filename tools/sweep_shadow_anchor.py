"""
Sweep z-gate thresholds for the Shadow HMM to find the optimal
balance between hit rate and false alarm rate.

Uses the cached SPX-only pickle from the duel backtest — no refitting needed.

Run:
    python tools/sweep_shadow_anchor.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(_ROOT, "data")
_PICKLE_PATH = os.path.join(_DATA_DIR, "_duel_spx_only.pickle")
_FWD_WINDOW = 30
_CRASH_THRESHOLD = -10.0


def _load_spx():
    import yfinance as yf
    df = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    returns = (np.log(close / close.shift(1)).dropna() * 100.0)
    close = close.reindex(returns.index).ffill()
    return close, returns


def _forward_drawdown(close, window=_FWD_WINDOW):
    vals = close.values
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        j = min(i + window, len(vals) - 1)
        if j <= i:
            continue
        seg = vals[i:j+1]
        out[i] = (seg.min() / seg[0] - 1.0) * 100.0
    return pd.Series(out, index=close.index)


def _find_episodes(crashed):
    crash_dates = crashed[crashed == 1].index
    if len(crash_dates) == 0:
        return []
    episodes = []
    ep_start = crash_dates[0]
    ep_end = crash_dates[0]
    for d in crash_dates[1:]:
        if (d - ep_end).days <= 30:
            ep_end = d
        else:
            episodes.append((ep_start, ep_end))
            ep_start = d
            ep_end = d
    episodes.append((ep_start, ep_end))
    return episodes


def _eval_gate(ll_z, fwd_dd, crashed, episodes, gate_z):
    n = len(ll_z)
    crisis_calls = (ll_z < gate_z)
    n_crisis = int(crisis_calls.sum())
    if n_crisis == 0:
        return {"gate_z": gate_z, "n_crisis": 0, "hits": 0, "hit_rate": 0,
                "precision": 0, "false_alarms": 0, "false_alarm_rate": 0,
                "early_warn": 0, "pct_days_flagged": 0}

    crisis_idx = np.where(crisis_calls)[0]
    crisis_dates = fwd_dd.index[crisis_idx]

    hits = 0
    early_warnings = []
    for ep_start, ep_end in episodes:
        w_start = ep_start - pd.Timedelta(days=20)
        w_end = ep_start + pd.Timedelta(days=5)
        calls = crisis_dates[(crisis_dates >= w_start) & (crisis_dates <= w_end)]
        if len(calls) > 0:
            hits += 1
            early_warnings.append((ep_start - calls[0]).days)

    hit_rate = hits / len(episodes) if episodes else 0
    false_alarms = int((crisis_calls & (fwd_dd.values[:n] > _CRASH_THRESHOLD)).sum())
    precision = 1.0 - (false_alarms / n_crisis) if n_crisis > 0 else 0
    avg_ew = float(np.mean(early_warnings)) if early_warnings else 0

    return {
        "gate_z": round(gate_z, 3),
        "n_crisis": n_crisis,
        "hits": hits,
        "misses": len(episodes) - hits,
        "hit_rate": round(hit_rate, 4),
        "precision": round(precision, 4),
        "false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarms / n_crisis, 4) if n_crisis > 0 else 0,
        "early_warn_days": round(avg_ew, 1),
        "pct_days_flagged": round(n_crisis / n * 100, 2),
    }


def main():
    if not os.path.exists(_PICKLE_PATH):
        print("ERROR: run backtest_shadow_duel.py first to create the cached pickle")
        return 1

    with open(_PICKLE_PATH, "rb") as f:
        result = pickle.load(f)

    print("[sweep] loading SPX data ...")
    close, returns = _load_spx()

    ll_obs = np.asarray(result.llf_obs)
    n = min(len(ll_obs), len(returns))
    ll_obs = ll_obs[:n]
    close = close.iloc[:n]

    mu = np.nanmean(ll_obs)
    sd = np.nanstd(ll_obs)
    ll_z = (ll_obs - mu) / sd

    fwd_dd = _forward_drawdown(close)
    crashed = (fwd_dd <= _CRASH_THRESHOLD).astype(int)
    episodes = _find_episodes(crashed)

    print(f"[sweep] {n} obs, {len(episodes)} crash episodes")
    print()

    # Sweep z-gates from -0.1 to -3.0
    gates = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50, -0.60, -0.70,
             -0.80, -0.90, -1.00, -1.20, -1.50, -1.80, -2.00, -2.50, -3.00]

    results = []
    for g in gates:
        r = _eval_gate(ll_z, fwd_dd, crashed, episodes, g)
        results.append(r)

    print(f"  {'gate_z':>7} {'crisis':>7} {'hits':>5}/{len(episodes):<3} {'hitRate':>8} "
          f"{'precision':>10} {'falseAlm':>9} {'earlyWarn':>10} {'%flagged':>9}")
    print("  " + "-" * 82)
    for r in results:
        marker = ""
        if r["hit_rate"] >= 0.90 and r["precision"] >= 0.30:
            marker = " <-- SWEET SPOT"
        elif r["hit_rate"] >= 0.80 and r["precision"] >= 0.50:
            marker = " <-- HIGH PRECISION"
        print(f"  {r['gate_z']:>7.3f} {r['n_crisis']:>7} {r['hits']:>5}/{len(episodes):<3} "
              f"{r['hit_rate']:>8.1%} {r['precision']:>10.1%} {r['false_alarms']:>9} "
              f"{r['early_warn_days']:>10.1f}d {r['pct_days_flagged']:>8.1f}%{marker}")

    print()
    print("INTERPRETATION:")
    print("  - Current gate z<-0.30 flags ~27% of all days = too noisy")
    print("  - Look for gates where hit_rate stays >80% but precision rises above 30%+")
    print("  - The CI anchor should be set so CI%=67% maps to that z threshold")
    print()

    # Suggest anchor
    # Find the gate with best F1 score (harmonic mean of hit_rate and precision)
    best = None
    best_f1 = 0
    for r in results:
        if r["hit_rate"] == 0 or r["precision"] == 0:
            continue
        f1 = 2 * r["hit_rate"] * r["precision"] / (r["hit_rate"] + r["precision"])
        if f1 > best_f1:
            best_f1 = f1
            best = r

    if best:
        suggested_anchor = round(abs(best["gate_z"]) / 0.67, 4)
        print(f"  BEST F1 gate: z < {best['gate_z']:.3f}  (F1={best_f1:.3f})")
        print(f"    hit_rate={best['hit_rate']:.1%}, precision={best['precision']:.1%}, "
              f"false_alarms={best['false_alarms']}, early_warn={best['early_warn_days']:.1f}d")
        print(f"    Suggested CI anchor: {suggested_anchor}  (so CI%=67% at z={best['gate_z']:.3f})")
        print(f"    Current anchor: 0.4478")
        print()

        # Also show the 90%+ hit rate sweet spot
        for r in results:
            if r["hit_rate"] >= 0.90 and r["precision"] >= 0.25:
                a = round(abs(r["gate_z"]) / 0.67, 4)
                f1_r = 2 * r["hit_rate"] * r["precision"] / (r["hit_rate"] + r["precision"])
                print(f"  90%+ hit rate option: z < {r['gate_z']:.3f}  anchor={a}")
                print(f"    hit={r['hit_rate']:.1%} prec={r['precision']:.1%} "
                      f"FA={r['false_alarms']} EW={r['early_warn_days']:.1f}d F1={f1_r:.3f}")

    # Save sweep results
    out_path = os.path.join(_DATA_DIR, "shadow_anchor_sweep.json")
    with open(out_path, "w") as f:
        json.dump({"episodes": len(episodes), "n_obs": n, "sweep": results,
                    "best_f1_gate": best["gate_z"] if best else None,
                    "suggested_anchor": suggested_anchor if best else None}, f, indent=2)
    print(f"\n  Sweep saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
