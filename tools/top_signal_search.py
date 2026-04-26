"""
tools/top_signal_search.py
==========================
Two-stage combinatorial search for the optimal HMM feature set
for early market TOP detection.

SAFE: never reads or writes data/hmm_brain.json.
Results → data/top_signal_search_results.json

───────────────────────────────────────────────────────────────
SCORING LOGIC
───────────────────────────────────────────────────────────────
For each feature combination a GaussianHMM is trained (n_states
auto-selected 2→6 by BIC, 3 random restarts). The full in-sample
LL z-score path is computed. Three signals are scored:

  A. REGIME EXIT — Viterbi state leaves "Bull" and stays away ≥15 days
  B. LL STRESS   — rolling 20-day mean LL z-score drops below -0.40
  C. COMBO       — A OR B fires (union)

Each signal is scored against known SPX peaks:
  hit         = signal fires within [-90, 0] days before peak
  lead_days   = peak_date − first_signal_date  (positive = early)
  false_alarm = signal fires outside all peak windows

Summary metrics per combination:
  hit_rate    = hits / n_peaks_in_window
  precision   = hits / total_fires
  F1          = harmonic mean of hit_rate and precision
  avg_lead    = mean lead days across hits
  pct_flagged = fraction of all days flagged

───────────────────────────────────────────────────────────────
KNOWN SPX PEAKS (in-sample for main brain training start ~2012)
───────────────────────────────────────────────────────────────
  2018-09-20  Pre Q4-2018 correction   (-20%)
  2020-02-19  Pre COVID crash          (-34%)
  2021-12-27  Pre rate-shock           (-25%)
  2024-07-16  Summer 2024 correction   (-8%)
  2025-02-19  Tariff-shock top         (-~15%)

───────────────────────────────────────────────────────────────
USAGE
───────────────────────────────────────────────────────────────
  # Two-stage (fast, ~10-25 min depending on CPU):
  python tools/top_signal_search.py

  # Full sweep of all 1023 combinations (slow, hours):
  python tools/top_signal_search.py --full

  # Skip stage 1, go straight to full combo sweep:
  python tools/top_signal_search.py --full --no-ablation

  # Limit parallel workers (default = CPU count - 1):
  python tools/top_signal_search.py --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR  = os.path.join(_ROOT, "data")
_FRED_DIR  = os.path.join(_DATA_DIR, "fred_cache")
_OUT_PATH  = os.path.join(_DATA_DIR, "top_signal_search_results.json")

_ALL_FEATURES = [
    "BAA10Y",   # HY credit spread (stress proxy)
    "AAA10Y",   # IG credit spread
    "T10Y2Y",   # 10yr-2yr yield curve (inversion)
    "T10Y3M",   # 10yr-3mo inversion depth
    "DGS10",    # 10yr nominal yield
    "DGS2",     # 2yr nominal yield
    "DFII10",   # real 10yr yield
    "NFCI",     # Chicago Fed financial conditions
    "ICSA",     # weekly jobless claims
    "VIX",      # equity fear gauge
]

# Known SPX peaks — hardcoded. Auto-detection augments these.
_KNOWN_PEAKS = [
    ("2018-09-20", "Q4-2018 top"),
    ("2020-02-19", "Pre-COVID top"),
    ("2021-12-27", "Pre rate-shock top"),
    ("2024-07-16", "Summer-2024 top"),
    ("2025-02-19", "Tariff-shock top"),
]

_LOOKBACK_YEARS = 15
_ROLLING_Z_WIN  = 5 * 252      # 5-yr rolling z-score window
_EWMA_SPAN      = 10            # EWMA smoothing span
_N_RESTARTS     = 3             # random restarts per n_states
_MAX_STATES     = 6             # BIC-selected up to this

# Signal parameters
_BULL_FRAC_WINDOW     = 40      # rolling window for bull-fraction signal
_BULL_FRAC_DROP       = 0.35    # bull fraction must drop by this much over the window
_LL_TREND_WINDOW      = 40      # rolling window for LL trend signal
_LL_TREND_THRESH      = -0.20   # rolling mean LL z-score below this = persistent stress
_LL_ACCEL_WINDOW      = 20      # shorter window to detect acceleration into stress
_LATE_CYCLE_LABELS    = {"Late Cycle", "Early Stress", "Stress", "Crisis"}
_PEAK_LEAD_WINDOW     = 180     # days before peak that count as a hit (tops need more lead)
_PEAK_TRAIL_WINDOW    = 60      # days after peak still count (macro features lag price)
_CLUSTER_GAP_DAYS     = 60      # minimum gap between signal clusters


# ── Data loading (same pipeline as hmm_regime.py) ────────────────────────────

def _load_fred(series_id: str) -> pd.Series:
    path = os.path.join(_FRED_DIR, f"{series_id}.csv")
    df = pd.read_csv(path)
    date_col = next((c for c in ("observation_date", "DATE") if c in df.columns), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    val_col = [c for c in df.columns if c != date_col][0]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.name = series_id
    return s


def _load_vix() -> pd.Series:
    try:
        import yfinance as yf
        df = yf.download("^VIX", period=f"{_LOOKBACK_YEARS + 2}y",
                         interval="1d", progress=False, auto_adjust=True)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = "VIX"
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.squeeze()
    except Exception:
        return pd.Series(dtype=float, name="VIX")


def _load_spx() -> pd.Series:
    try:
        import yfinance as yf
        df = yf.download("^GSPC", period=f"{_LOOKBACK_YEARS + 2}y",
                         interval="1d", progress=False, auto_adjust=True)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.squeeze().dropna()
    except Exception:
        return pd.Series(dtype=float, name="SPX")


def _build_feature_matrix(features: list[str]) -> pd.DataFrame:
    """Build EWMA-smoothed, rolling-z-scored feature matrix for a given feature subset."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=_LOOKBACK_YEARS)
    series_list = []
    for f in features:
        if f == "VIX":
            s = _load_vix()
        else:
            try:
                s = _load_fred(f)
            except Exception:
                continue
        series_list.append(s)

    if not series_list:
        raise RuntimeError(f"No data loaded for features: {features}")

    df = pd.concat(series_list, axis=1)
    df = df.resample("B").last().ffill(limit=5)
    df = df[df.index >= cutoff].dropna()

    # EWMA smoothing
    df = df.ewm(span=_EWMA_SPAN).mean()

    # Rolling 5-yr z-score
    mu = df.rolling(window=_ROLLING_Z_WIN, min_periods=252).mean()
    sg = df.rolling(window=_ROLLING_Z_WIN, min_periods=252).std().replace(0, 1)
    df = (df - mu) / sg
    df = df.clip(-3, 3).dropna()
    return df


# ── HMM training ─────────────────────────────────────────────────────────────

def _train_combo(features: list[str], df: pd.DataFrame) -> tuple:
    """Train GaussianHMM on df, return (model, n_states, ll_per_obs, ll_std, ci_anchor, state_labels)."""
    from hmmlearn.hmm import GaussianHMM

    X = df.values.astype(np.float64)
    best_model, best_bic, best_n = None, np.inf, 3

    for n in range(2, _MAX_STATES + 1):
        for seed in range(_N_RESTARTS):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=300,
                    random_state=seed * 13 + n,
                    tol=1e-4,
                    verbose=False,
                    init_params="smc",
                )
                _off = (1 - 0.70) / max(n - 1, 1)
                model.transmat_ = np.full((n, n), _off)
                np.fill_diagonal(model.transmat_, 0.70)
                model.transmat_ += 1e-6
                model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import logging as _log
                    _log.disable(_log.CRITICAL)
                    model.fit(X)
                    _log.disable(_log.NOTSET)
                bic = model.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_n = n
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Training failed")

    # LL baseline
    ll_total = float(best_model.score(X))
    ll_per_obs = ll_total / len(X)
    win = 60
    ll_vals = [float(best_model.score(X[i - win:i])) / win
               for i in range(win, len(X))]
    ll_std = float(np.std(ll_vals)) if ll_vals else 1.0

    # CI anchor
    emit = best_model._compute_log_likelihood(X)
    live_ll = logsumexp(emit, axis=1)
    z_train = (live_ll - ll_per_obs) / max(ll_std, 1e-6)
    ci_anchor = float(max(abs(z_train.min()), 0.467))

    # State labels (Bull = lowest BAA10Y or AAA10Y mean)
    hy_idx = next(
        (i for i, f in enumerate(features) if f in ("BAA10Y", "AAA10Y")), None
    )
    n = best_n
    labels_by_order = {
        2: ["Bull", "Stress"],
        3: ["Bull", "Neutral", "Stress"],
        4: ["Bull", "Neutral", "Stress", "Late Cycle"],
        5: ["Bull", "Neutral", "Stress", "Late Cycle", "Crisis"],
        6: ["Bull", "Neutral", "Early Stress", "Stress", "Late Cycle", "Crisis"],
    }
    assigned = labels_by_order.get(n, [f"S{i}" for i in range(n)])
    if hy_idx is not None:
        order = np.argsort(best_model.means_[:, hy_idx])
        label_map = {int(state_idx): assigned[rank] for rank, state_idx in enumerate(order)}
        state_labels = [label_map[i] for i in range(n)]
    else:
        state_labels = assigned[:n]

    return best_model, best_n, ll_per_obs, ll_std, ci_anchor, state_labels


# ── Signal extraction ─────────────────────────────────────────────────────────

def _compute_signals(
    df: pd.DataFrame,
    model,
    ll_per_obs: float,
    ll_std: float,
    state_labels: list[str],
) -> pd.DataFrame:
    """Return a daily DataFrame with regime, ll_z, and signal columns."""
    X = df.values.astype(np.float64)
    decoded = model.predict(X)
    emit = model._compute_log_likelihood(X)
    live_ll = logsumexp(emit, axis=1)
    ll_z = (live_ll - ll_per_obs) / max(ll_std, 1e-6)

    out = pd.DataFrame({
        "date": df.index,
        "state": decoded,
        "state_label": [state_labels[s] for s in decoded],
        "ll_z": ll_z,
    }).set_index("date")

    # Signal A: late-cycle / stress regime entry
    # Fires on the first day the brain enters a Late Cycle, Early Stress, Stress,
    # or Crisis state (NOT Bull or Neutral). Rising-edge scoring means each
    # distinct deterioration episode produces exactly one fire date.
    is_stress = out["state_label"].isin(_LATE_CYCLE_LABELS).astype(int)
    out["sig_regime"] = is_stress

    # Signal B: persistent LL stress — rolling mean LL z-score drops below threshold.
    # Lower threshold than crisis detection (-0.20 vs -0.40) because tops show
    # mild-but-sustained stress, not sudden spikes.
    out["ll_z_roll"] = out["ll_z"].rolling(_LL_TREND_WINDOW, min_periods=10).mean()
    out["sig_ll"] = (out["ll_z_roll"] < _LL_TREND_THRESH).astype(int)

    # Signal C: OR-combo (union — higher recall, more false alarms)
    out["sig_combo"] = ((out["sig_regime"] == 1) | (out["sig_ll"] == 1)).astype(int)

    # Signal D: AND-combo (intersection — higher precision, fewer false alarms)
    out["sig_and"] = ((out["sig_regime"] == 1) & (out["sig_ll"] == 1)).astype(int)

    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class SignalScore:
    hits: int
    n_peaks: int
    total_fires: int
    false_alarms: int
    hit_rate: float
    precision: float
    f1: float
    avg_lead_days: float
    pct_flagged: float


def _rising_edge_fires(signal: pd.Series) -> list[pd.Timestamp]:
    """Return dates where signal transitions 0→1 (rising edge only).

    Each continuous stress episode produces exactly one fire date — the first day
    the signal turns on — regardless of how many days the signal stays elevated.
    This keeps false-alarm counts meaningful (one per episode, not per day).
    """
    s = signal.astype(int)
    # rising edge: current==1 AND previous==0 (or first day ==1)
    prev = s.shift(1, fill_value=0)
    edges = s.index[(s.values == 1) & (prev.values == 0)]
    return list(edges)


def _score_signal(
    signal: pd.Series,
    peaks: list[pd.Timestamp],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> SignalScore:
    """Score a binary signal series against a list of peak dates.

    Fires are de-duplicated into clusters (one fire per run of consecutive
    signal days) so false alarms count episodes, not individual days.
    """
    valid_peaks = [p for p in peaks if train_start <= p <= train_end]
    n_peaks = len(valid_peaks)
    if n_peaks == 0:
        return SignalScore(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    fire_clusters = _rising_edge_fires(signal)
    total_fires = len(fire_clusters)
    pct_flagged = float((signal == 1).sum()) / max(len(signal), 1)

    hits = 0
    lead_days_list = []
    peak_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for peak in valid_peaks:
        w_start = peak - pd.Timedelta(days=_PEAK_LEAD_WINDOW)
        w_end   = peak + pd.Timedelta(days=_PEAK_TRAIL_WINDOW)
        peak_windows.append((w_start, w_end))

        fires_in_window = [f for f in fire_clusters if w_start <= f <= peak]
        if fires_in_window:
            hits += 1
            lead_days_list.append((peak - fires_in_window[0]).days)

    # False alarm clusters: cluster fires outside all peak windows
    false_alarms = sum(
        1 for f in fire_clusters
        if not any(ws <= f <= we for ws, we in peak_windows)
    )

    hit_rate  = hits / n_peaks
    precision = hits / total_fires if total_fires > 0 else 0.0
    f1 = (2 * precision * hit_rate / (precision + hit_rate)
          if (precision + hit_rate) > 0 else 0.0)
    avg_lead = float(np.mean(lead_days_list)) if lead_days_list else 0.0

    return SignalScore(
        hits=hits,
        n_peaks=n_peaks,
        total_fires=total_fires,
        false_alarms=false_alarms,
        hit_rate=round(hit_rate, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        avg_lead_days=round(avg_lead, 1),
        pct_flagged=round(pct_flagged, 4),
    )


# ── Worker function (must be top-level for pickling) ─────────────────────────

def _worker(args: tuple) -> dict | None:
    """Train and score one feature combination. Returns result dict or None on failure."""
    features, peaks_str = args
    peaks = [pd.Timestamp(p) for p in peaks_str]

    try:
        df = _build_feature_matrix(features)
        if len(df) < 500:
            return None
        model, n_states, ll_per_obs, ll_std, ci_anchor, state_labels = _train_combo(features, df)
        sig_df = _compute_signals(df, model, ll_per_obs, ll_std, state_labels)

        train_start, train_end = df.index[0], df.index[-1]

        score_a = _score_signal(sig_df["sig_regime"], peaks, train_start, train_end)
        score_b = _score_signal(sig_df["sig_ll"],     peaks, train_start, train_end)
        score_c = _score_signal(sig_df["sig_combo"],  peaks, train_start, train_end)
        score_d = _score_signal(sig_df["sig_and"],    peaks, train_start, train_end)

        # Bull % of time (how often the brain is in bull mode)
        bull_pct = float((sig_df["state_label"] == "Bull").mean())
        state_dist = sig_df["state_label"].value_counts(normalize=True).to_dict()

        return {
            "features": features,
            "n_features": len(features),
            "n_states": n_states,
            "ci_anchor": round(ci_anchor, 4),
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "n_obs": len(df),
            "bull_pct": round(bull_pct, 4),
            "state_labels": state_labels,
            "state_distribution": {k: round(v, 4) for k, v in state_dist.items()},
            "regime_exit": asdict(score_a),
            "ll_stress": asdict(score_b),
            "combo_or": asdict(score_c),
            "combo_and": asdict(score_d),
            # Primary ranking key: best F1 across all four signals
            "combo": asdict(max([score_a, score_b, score_c, score_d], key=lambda s: s.f1)),
            "best_signal": max(
                [("regime_exit", score_a), ("ll_stress", score_b),
                 ("combo_or", score_c), ("combo_and", score_d)],
                key=lambda x: x[1].f1
            )[0],
        }
    except Exception as e:
        return {"features": features, "error": str(e)}


# ── Search stages ─────────────────────────────────────────────────────────────

def _run_parallel(combos: list[list[str]], peaks: list[pd.Timestamp],
                  workers: int, label: str) -> list[dict]:
    peaks_str = [p.strftime("%Y-%m-%d") for p in peaks]
    args = [(list(c), peaks_str) for c in combos]
    results = []
    n = len(args)
    done = 0
    t0 = time.time()

    print(f"\n  {label}: {n} combinations | {workers} workers")
    print(f"  {'Combo':>6}  {'Features':<55}  {'F1':>6}  {'Hits':>6}  {'FA':>6}  {'Lead':>7}")
    print("  " + "─" * 90)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            elapsed = time.time() - t0
            eta = elapsed / done * (n - done) if done > 0 else 0

            if res and "error" not in res:
                sc = res["combo"]
                feat_str = ",".join(res["features"])[:54]
                print(
                    f"  {done:>6}/{n}  {feat_str:<55}  "
                    f"{sc['f1']:>6.3f}  {sc['hits']:>2}/{sc['n_peaks']:<3}  "
                    f"{sc['false_alarms']:>6}  {sc['avg_lead_days']:>6.0f}d"
                    f"  ETA {eta/60:.1f}m"
                )
                results.append(res)
            elif res and "error" in res:
                print(f"  {done:>6}/{n}  ERROR {res['features']}: {res['error'][:40]}")

    return results


def _stage1_ablation(all_features: list[str], peaks: list[pd.Timestamp],
                     workers: int) -> list[str]:
    """Drop one feature at a time. Return features worth keeping."""
    print("\n" + "═" * 92)
    print("  STAGE 1 — ABLATION SWEEP (remove one feature at a time)")
    print("═" * 92)

    # Baseline: all features
    baseline_args = [(all_features, [p.strftime("%Y-%m-%d") for p in peaks])]
    baseline_results = [_worker(a) for a in baseline_args]
    baseline = baseline_results[0]
    if not baseline or "error" in baseline:
        print(f"  BASELINE FAILED: {baseline}")
        return all_features
    baseline_f1 = baseline["combo"]["f1"]
    print(f"\n  Baseline (all {len(all_features)} features): F1={baseline_f1:.3f}  "
          f"hits={baseline['combo']['hits']}/{baseline['combo']['n_peaks']}  "
          f"FA={baseline['combo']['false_alarms']}")

    # One-feature-removed combos
    ablation_combos = [
        [f for f in all_features if f != dropped]
        for dropped in all_features
    ]
    ablation_results = _run_parallel(ablation_combos, peaks, workers,
                                     "ABLATION (drop one feature)")

    print("\n  ABLATION RESULTS — ranked by combo F1:")
    print(f"  {'Dropped':<12}  {'F1':>6}  {'vs baseline':>12}  {'Hits':>8}  {'FA':>6}  {'Lead':>7}")
    print("  " + "─" * 60)

    # Map result back to dropped feature
    survivors = []
    for res in sorted(ablation_results, key=lambda r: r["combo"]["f1"], reverse=True):
        feat_set = set(res["features"])
        dropped = next((f for f in all_features if f not in feat_set), "?")
        delta = res["combo"]["f1"] - baseline_f1
        sc = res["combo"]
        indicator = "▲ IMPROVE" if delta > 0.02 else ("▼ HURT" if delta < -0.02 else "~ neutral")
        print(
            f"  {dropped:<12}  {sc['f1']:>6.3f}  {delta:>+10.3f}  {indicator:<12}  "
            f"{sc['hits']:>2}/{sc['n_peaks']:<3}  {sc['false_alarms']:>6}  {sc['avg_lead_days']:>6.0f}d"
        )
        # Keep feature if removing it HURTS (delta < 0) or is neutral
        # Drop feature if removing it IMPROVES F1 by meaningful margin
        if delta <= 0.02:
            survivors.append(dropped)

    print(f"\n  Features to KEEP for Stage 2: {survivors}")
    print(f"  Features to DROP (removing improved F1): "
          f"{[f for f in all_features if f not in survivors]}")
    return survivors


def _stage2_sweep(features: list[str], peaks: list[pd.Timestamp],
                  workers: int, min_features: int = 3) -> list[dict]:
    """Full combinatorial sweep on feature survivors."""
    n = len(features)
    combos = []
    for k in range(min_features, n + 1):
        combos.extend(list(c) for c in itertools.combinations(features, k))

    print("\n" + "═" * 92)
    print(f"  STAGE 2 — COMBINATORIAL SWEEP ({len(combos)} combinations, "
          f"{min_features}..{n} features)")
    print("═" * 92)
    return _run_parallel(combos, peaks, workers, "COMBO SWEEP")


# ── SPX peak auto-detection ───────────────────────────────────────────────────

def _detect_spx_peaks(spx: pd.Series, train_start: pd.Timestamp) -> list[pd.Timestamp]:
    """Augment hardcoded peaks with scipy find_peaks on SPX log price."""
    from scipy.signal import find_peaks
    spx_in = spx[spx.index >= train_start].dropna()
    if spx_in.empty:
        return []
    log_px = np.log(spx_in.values)
    # Detect local peaks with prominence ≥ 8% and minimum 63-day separation
    peak_idx, props = find_peaks(log_px, distance=63, prominence=np.log(1.12))
    return [spx_in.index[i] for i in peak_idx]


# ── Output formatting ─────────────────────────────────────────────────────────

def _print_leaderboard(results: list[dict], top_n: int = 20) -> None:
    ranked = sorted(
        [r for r in results if "error" not in r],
        key=lambda r: (r["combo"]["f1"], r["combo"]["avg_lead_days"]),
        reverse=True,
    )
    print("\n" + "═" * 110)
    print(f"  TOP-{top_n} FEATURE COMBINATIONS FOR TOP DETECTION")
    print("═" * 110)
    print(f"  {'Rank':>4}  {'Features':<60}  {'n':>3}  {'F1':>6}  "
          f"{'Hits':>6}  {'FA':>6}  {'Lead':>7}  {'Flagged%':>9}")
    print("  " + "─" * 106)
    for rank, res in enumerate(ranked[:top_n], 1):
        sc = res["combo"]
        feat_str = ",".join(res["features"])[:59]
        print(
            f"  {rank:>4}.  {feat_str:<60}  {res['n_features']:>3}  "
            f"{sc['f1']:>6.3f}  {sc['hits']:>2}/{sc['n_peaks']:<3}  "
            f"{sc['false_alarms']:>6}  {sc['avg_lead_days']:>6.0f}d  "
            f"{sc['pct_flagged']*100:>8.1f}%"
        )

    if ranked:
        best = ranked[0]
        print(f"\n  ★  BEST COMBO: {best['features']}")
        print(f"     n_states={best['n_states']}  ci_anchor={best['ci_anchor']}")
        sc = best["combo"]
        print(f"     Combo signal: F1={sc['f1']:.3f}  hits={sc['hits']}/{sc['n_peaks']}  "
              f"FA={sc['false_alarms']}  avg_lead={sc['avg_lead_days']:.0f}d  "
              f"flagged={sc['pct_flagged']*100:.1f}%")
        print(f"     Best signal: {best.get('best_signal', '?')}")
        sc_a = best["regime_exit"]
        print(f"     Regime-exit:  F1={sc_a['f1']:.3f}  hits={sc_a['hits']}/{sc_a['n_peaks']}  "
              f"FA={sc_a['false_alarms']}  avg_lead={sc_a['avg_lead_days']:.0f}d")
        sc_b = best["ll_stress"]
        print(f"     LL-stress:    F1={sc_b['f1']:.3f}  hits={sc_b['hits']}/{sc_b['n_peaks']}  "
              f"FA={sc_b['false_alarms']}  avg_lead={sc_b['avg_lead_days']:.0f}d")
        sc_d = best.get("combo_and", {})
        if sc_d:
            print(f"     AND-combo:    F1={sc_d['f1']:.3f}  hits={sc_d['hits']}/{sc_d['n_peaks']}  "
                  f"FA={sc_d['false_alarms']}  avg_lead={sc_d['avg_lead_days']:.0f}d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Top-signal HMM feature search")
    parser.add_argument("--full", action="store_true",
                        help="Sweep all 1023 combinations (slow)")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip stage 1 ablation (use all features in stage 2)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1),
                        help="Parallel workers (default: CPU count - 1)")
    parser.add_argument("--min-features", type=int, default=3,
                        help="Minimum features per combination in stage 2 (default: 3)")
    args = parser.parse_args()

    print("═" * 92)
    print("  TOP-SIGNAL HMM FEATURE SEARCH")
    print(f"  Mode: {'FULL SWEEP' if args.full else 'TWO-STAGE'}  |  "
          f"Workers: {args.workers}  |  "
          f"Output: {_OUT_PATH}")
    print("═" * 92)

    # Load SPX once (all workers need peaks, not the series itself)
    print("\n[1/4] Loading SPX for peak detection...")
    spx = _load_spx()

    # Determine training window start from feature matrix
    print("[2/4] Detecting SPX peaks...")
    try:
        sample_df = _build_feature_matrix(_ALL_FEATURES)
        train_start = sample_df.index[0]
        train_end   = sample_df.index[-1]
    except Exception as e:
        print(f"  ERROR building sample feature matrix: {e}")
        return 1

    print(f"  Training window: {train_start.date()} → {train_end.date()}")

    # Combine hardcoded + auto-detected peaks
    hardcoded = [pd.Timestamp(d) for d, _ in _KNOWN_PEAKS
                 if train_start <= pd.Timestamp(d) <= train_end]
    auto_peaks = _detect_spx_peaks(spx, train_start)

    all_peaks_set: set[pd.Timestamp] = set(hardcoded)
    for ap in auto_peaks:
        # Only add if not within 45 days of an existing peak
        if not any(abs((ap - hp).days) < 45 for hp in all_peaks_set):
            all_peaks_set.add(ap)

    peaks = sorted(all_peaks_set)
    print(f"  Peaks used for scoring ({len(peaks)}):")
    for p in peaks:
        label = next((l for d, l in _KNOWN_PEAKS if pd.Timestamp(d) == p), "auto-detected")
        print(f"    {p.date()}  {label}")

    print(f"\n[3/4] Running search ({args.workers} parallel workers)...")

    all_results: list[dict] = []

    if args.full:
        # Sweep all combinations
        all_combos = []
        for k in range(args.min_features, len(_ALL_FEATURES) + 1):
            all_combos.extend(list(c) for c in itertools.combinations(_ALL_FEATURES, k))
        all_results = _run_parallel(all_combos, peaks, args.workers, "FULL SWEEP")
    else:
        if not args.no_ablation:
            survivors = _stage1_ablation(_ALL_FEATURES, peaks, args.workers)
        else:
            survivors = _ALL_FEATURES

        stage2 = _stage2_sweep(survivors, peaks, args.workers, args.min_features)
        all_results.extend(stage2)

        # Always include the full-feature baseline
        print("\n  Adding full-feature baseline...")
        baseline = _worker((_ALL_FEATURES, [p.strftime("%Y-%m-%d") for p in peaks]))
        if baseline and "error" not in baseline:
            all_results.append(baseline)

    print(f"\n[4/4] Saving results ({len(all_results)} combinations)...")
    _print_leaderboard(all_results)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "mode": "full" if args.full else "two-stage",
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "peaks_used": [p.strftime("%Y-%m-%d") for p in peaks],
        "signal_params": {
            "bull_frac_window": _BULL_FRAC_WINDOW,
            "ll_stress_window": _LL_TREND_WINDOW,
            "ll_stress_thresh": _LL_TREND_THRESH,
            "peak_lead_window_days": _PEAK_LEAD_WINDOW,
            "peak_trail_window_days": _PEAK_TRAIL_WINDOW,
        },
        "n_combinations_tested": len(all_results),
        "results": sorted(
            [r for r in all_results if "error" not in r],
            key=lambda r: r["combo"]["f1"],
            reverse=True,
        ),
        "errors": [r for r in all_results if "error" in r],
    }

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results written to {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
