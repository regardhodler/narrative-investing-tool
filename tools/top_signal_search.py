"""
tools/top_signal_search.py
==========================
Two-stage combinatorial search for the optimal HMM feature set
for early market TOP detection.

SAFE: never reads or writes data/hmm_brain.json.
Results → data/top_signal_search_results.json

───────────────────────────────────────────────────────────────
DESIGN
───────────────────────────────────────────────────────────────
All FRED + VIX data is loaded ONCE in the main process and held
in a shared module-level DataFrame. Worker threads slice the
pre-built matrix by feature name — no network calls or disk I/O
inside workers. ThreadPoolExecutor avoids Windows spawn issues
while still parallelising the CPU-heavy HMM fitting (hmmlearn
releases the GIL through numpy).

───────────────────────────────────────────────────────────────
SCORING LOGIC
───────────────────────────────────────────────────────────────
For each feature combination a GaussianHMM is trained (n_states
auto-selected 2→6 by BIC, 3 random restarts). Four signals:

  A. regime_exit  — first day entering Late Cycle / Stress / Crisis
  B. ll_stress    — rolling 40-day mean LL z-score below -0.20
  C. combo_or     — A OR B (higher recall)
  D. combo_and    — A AND B (higher precision)

Fires are de-duplicated to rising-edge only (one event per
continuous stress episode). Scored against known SPX peaks
within a [-180, +60] day window (macro features lag price tops).

───────────────────────────────────────────────────────────────
KNOWN SPX PEAKS (in main brain training window 2012+)
───────────────────────────────────────────────────────────────
  2018-09-20  Pre Q4-2018 correction   (-20%)
  2020-02-19  Pre COVID crash          (-34%)
  2021-12-27  Pre rate-shock top       (-25%)
  2024-07-16  Summer 2024 correction   (-8%)
  2025-02-19  Tariff-shock top         (-~15%)

───────────────────────────────────────────────────────────────
USAGE
───────────────────────────────────────────────────────────────
  # Two-stage (fast, ~10-25 min):
  python tools/top_signal_search.py

  # Full sweep of all 1023 combinations (hours):
  python tools/top_signal_search.py --full

  # Skip ablation, go straight to full combo sweep:
  python tools/top_signal_search.py --full --no-ablation

  # Control thread count:
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(_ROOT, "data")
_FRED_DIR = os.path.join(_DATA_DIR, "fred_cache")
_OUT_PATH = os.path.join(_DATA_DIR, "top_signal_search_results.json")

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

# Hardcoded peaks — only those clearly in the training window
_KNOWN_PEAKS = [
    ("2018-09-20", "Q4-2018 top"),
    ("2020-02-19", "Pre-COVID top"),
    ("2021-12-27", "Pre rate-shock top"),
    ("2024-07-16", "Summer-2024 top"),
    ("2025-02-19", "Tariff-shock top"),
]

_LOOKBACK_YEARS     = 15
_ROLLING_Z_WIN      = 5 * 252
_EWMA_SPAN          = 10
_N_RESTARTS         = 3
_MAX_STATES         = 6

# Signal params
_LL_TREND_WINDOW    = 40
_LL_TREND_THRESH    = -0.20
_LATE_CYCLE_LABELS  = {"Late Cycle", "Early Stress", "Stress", "Crisis"}
_PEAK_LEAD_WINDOW   = 180   # days before peak that count as hit
_PEAK_TRAIL_WINDOW  = 60    # days after peak that still count

# Shared pre-built matrix (populated once in main before spawning threads)
_FULL_MATRIX: Optional[pd.DataFrame] = None


# ── Data loading ─────────────────────────────────────────────────────────────

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
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.squeeze().dropna()
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


def _build_full_matrix() -> pd.DataFrame:
    """Load all 10 features, EWMA-smooth, rolling-z-score. Called once in main."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=_LOOKBACK_YEARS)
    series_list = []
    for f in _ALL_FEATURES:
        try:
            s = _load_vix() if f == "VIX" else _load_fred(f)
            if not s.empty:
                series_list.append(s)
        except Exception as e:
            print(f"  [warn] could not load {f}: {e}")

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


def _slice_matrix(features: list[str]) -> pd.DataFrame:
    """Slice the pre-built full matrix to the requested features and drop NaN rows."""
    assert _FULL_MATRIX is not None, "Call _build_full_matrix() first"
    available = [f for f in features if f in _FULL_MATRIX.columns]
    if not available:
        raise RuntimeError(f"None of {features} found in matrix columns: {list(_FULL_MATRIX.columns)}")
    return _FULL_MATRIX[available].dropna()


# ── HMM training ─────────────────────────────────────────────────────────────

def _train_combo(features: list[str], df: pd.DataFrame):
    """Train GaussianHMM, return (model, n_states, ll_per_obs, ll_std, ci_anchor, state_labels)."""
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
                    model.fit(X)
                bic = model.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_n = n
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Training failed for all n_states")

    # LL baseline
    ll_per_obs = float(best_model.score(X)) / len(X)
    win = 60
    ll_vals = [float(best_model.score(X[i - win:i])) / win for i in range(win, len(X))]
    ll_std = float(np.std(ll_vals)) if ll_vals else 1.0

    # CI anchor
    emit  = best_model._compute_log_likelihood(X)
    z_all = (logsumexp(emit, axis=1) - ll_per_obs) / max(ll_std, 1e-6)
    ci_anchor = float(max(abs(z_all.min()), 0.467))

    # State labels: lowest BAA10Y / AAA10Y mean → Bull; highest → Crisis
    hy_idx = next(
        (i for i, f in enumerate(features) if f in ("BAA10Y", "AAA10Y")), None
    )
    n = best_n
    templates = {
        2: ["Bull", "Stress"],
        3: ["Bull", "Neutral", "Stress"],
        4: ["Bull", "Neutral", "Stress", "Late Cycle"],
        5: ["Bull", "Neutral", "Stress", "Late Cycle", "Crisis"],
        6: ["Bull", "Neutral", "Early Stress", "Stress", "Late Cycle", "Crisis"],
    }
    assigned = templates.get(n, [f"S{i}" for i in range(n)])
    if hy_idx is not None:
        order = np.argsort(best_model.means_[:, hy_idx])
        lmap = {int(s): assigned[r] for r, s in enumerate(order)}
        state_labels = [lmap[i] for i in range(n)]
    else:
        state_labels = assigned[:n]

    return best_model, best_n, ll_per_obs, ll_std, ci_anchor, state_labels


# ── Signal extraction ─────────────────────────────────────────────────────────

def _compute_signals(df: pd.DataFrame, model, ll_per_obs: float,
                     ll_std: float, state_labels: list[str]) -> pd.DataFrame:
    X = df.values.astype(np.float64)
    decoded = model.predict(X)
    emit    = model._compute_log_likelihood(X)
    ll_z    = (logsumexp(emit, axis=1) - ll_per_obs) / max(ll_std, 1e-6)

    out = pd.DataFrame({
        "state":       decoded,
        "state_label": [state_labels[s] for s in decoded],
        "ll_z":        ll_z,
    }, index=df.index)

    # Signal A: late-cycle / stress entry (rising-edge scored)
    out["sig_regime"] = out["state_label"].isin(_LATE_CYCLE_LABELS).astype(int)

    # Signal B: persistent LL stress
    out["ll_z_roll"]  = out["ll_z"].rolling(_LL_TREND_WINDOW, min_periods=10).mean()
    out["sig_ll"]     = (out["ll_z_roll"] < _LL_TREND_THRESH).astype(int)

    # Signal C / D: OR / AND combo
    out["sig_combo"]  = ((out["sig_regime"] == 1) | (out["sig_ll"] == 1)).astype(int)
    out["sig_and"]    = ((out["sig_regime"] == 1) & (out["sig_ll"] == 1)).astype(int)

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


def _rising_edges(signal: pd.Series) -> list[pd.Timestamp]:
    """One fire date per continuous stress episode (the first day the signal turns on)."""
    s = signal.astype(int)
    prev = s.shift(1, fill_value=0)
    return list(s.index[(s.values == 1) & (prev.values == 0)])


def _score_signal(signal: pd.Series, peaks: list[pd.Timestamp],
                  train_start: pd.Timestamp, train_end: pd.Timestamp) -> SignalScore:
    valid_peaks = [p for p in peaks if train_start <= p <= train_end]
    n_peaks     = len(valid_peaks)
    if n_peaks == 0:
        return SignalScore(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    fires        = _rising_edges(signal)
    total_fires  = len(fires)
    pct_flagged  = float((signal == 1).sum()) / max(len(signal), 1)

    hits, lead_days_list = 0, []
    peak_windows = []
    for peak in valid_peaks:
        ws = peak - pd.Timedelta(days=_PEAK_LEAD_WINDOW)
        we = peak + pd.Timedelta(days=_PEAK_TRAIL_WINDOW)
        peak_windows.append((ws, we))
        in_win = [f for f in fires if ws <= f <= peak]
        if in_win:
            hits += 1
            lead_days_list.append((peak - in_win[0]).days)

    false_alarms = sum(1 for f in fires
                       if not any(ws <= f <= we for ws, we in peak_windows))

    hit_rate  = hits / n_peaks
    precision = hits / total_fires if total_fires > 0 else 0.0
    f1        = (2 * precision * hit_rate / (precision + hit_rate)
                 if (precision + hit_rate) > 0 else 0.0)
    avg_lead  = float(np.mean(lead_days_list)) if lead_days_list else 0.0

    return SignalScore(
        hits=hits, n_peaks=n_peaks, total_fires=total_fires,
        false_alarms=false_alarms,
        hit_rate=round(hit_rate, 4), precision=round(precision, 4),
        f1=round(f1, 4), avg_lead_days=round(avg_lead, 1),
        pct_flagged=round(pct_flagged, 4),
    )


# ── Worker (thread-safe — uses pre-built shared matrix) ──────────────────────

def _worker(args: tuple) -> dict:
    features, peaks = args  # peaks already as Timestamps
    try:
        df = _slice_matrix(features)
        if len(df) < 500:
            return {"features": features, "error": "too few observations"}

        model, n_states, ll_per_obs, ll_std, ci_anchor, state_labels = \
            _train_combo(features, df)
        sig = _compute_signals(df, model, ll_per_obs, ll_std, state_labels)

        ts, te = df.index[0], df.index[-1]
        sa = _score_signal(sig["sig_regime"], peaks, ts, te)
        sb = _score_signal(sig["sig_ll"],     peaks, ts, te)
        sc = _score_signal(sig["sig_combo"],  peaks, ts, te)
        sd = _score_signal(sig["sig_and"],    peaks, ts, te)

        best_score = max([sa, sb, sc, sd], key=lambda s: s.f1)
        best_name  = max(
            [("regime_exit", sa), ("ll_stress", sb),
             ("combo_or", sc),    ("combo_and", sd)],
            key=lambda x: x[1].f1
        )[0]

        return {
            "features":          features,
            "n_features":        len(features),
            "n_states":          n_states,
            "ci_anchor":         round(ci_anchor, 4),
            "train_start":       ts.strftime("%Y-%m-%d"),
            "train_end":         te.strftime("%Y-%m-%d"),
            "n_obs":             len(df),
            "bull_pct":          round(float((sig["state_label"] == "Bull").mean()), 4),
            "state_labels":      state_labels,
            "state_distribution": {
                k: round(v, 4)
                for k, v in sig["state_label"].value_counts(normalize=True).items()
            },
            "regime_exit":  asdict(sa),
            "ll_stress":    asdict(sb),
            "combo_or":     asdict(sc),
            "combo_and":    asdict(sd),
            "combo":        asdict(best_score),   # best of the four — used for ranking
            "best_signal":  best_name,
        }
    except Exception as e:
        return {"features": features, "error": str(e)}


# ── Parallel runner ───────────────────────────────────────────────────────────

def _run_parallel(combos: list[list[str]], peaks: list[pd.Timestamp],
                  workers: int, label: str) -> list[dict]:
    args    = [(list(c), peaks) for c in combos]
    results = []
    n, done = len(args), 0
    t0      = time.time()

    print(f"\n  {label}: {n} combinations | {workers} threads")
    print(f"  {'Done':>8}  {'Features':<56}  {'Best':>13}  {'F1':>6}  "
          f"{'Hits':>6}  {'FA':>4}  {'Lead':>7}")
    print("  " + "─" * 105)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            done += 1
            res   = fut.result()
            elapsed = time.time() - t0
            eta   = elapsed / done * (n - done) if done > 0 else 0

            if "error" not in res:
                sc       = res["combo"]
                feat_str = ",".join(res["features"])[:55]
                best_sig = res.get("best_signal", "?")[:13]
                print(
                    f"  {done:>4}/{n:<4}  {feat_str:<56}  {best_sig:>13}  "
                    f"{sc['f1']:>6.3f}  {sc['hits']:>2}/{sc['n_peaks']:<3}  "
                    f"{sc['false_alarms']:>4}  {sc['avg_lead_days']:>6.0f}d"
                    + (f"  ETA {eta/60:.1f}m" if n > 5 else "")
                )
                results.append(res)
            else:
                print(f"  {done:>4}/{n:<4}  ERROR {res['features'][:3]}: {res['error'][:50]}")

    return results


# ── Stage 1: ablation ─────────────────────────────────────────────────────────

def _stage1_ablation(all_features: list[str], peaks: list[pd.Timestamp],
                     workers: int) -> list[str]:
    print("\n" + "═" * 108)
    print("  STAGE 1 — ABLATION (remove one feature at a time)")
    print("═" * 108)

    # Baseline
    print("  Computing baseline (all features)...")
    baseline = _worker((all_features, peaks))
    if "error" in baseline:
        print(f"  BASELINE FAILED: {baseline['error']}")
        return all_features
    baseline_f1 = baseline["combo"]["f1"]
    sc = baseline["combo"]
    print(f"  Baseline ({len(all_features)} features): "
          f"F1={baseline_f1:.3f}  hits={sc['hits']}/{sc['n_peaks']}  "
          f"FA={sc['false_alarms']}  lead={sc['avg_lead_days']:.0f}d")

    ablation_combos = [[f for f in all_features if f != dropped]
                       for dropped in all_features]
    ablation_results = _run_parallel(ablation_combos, peaks, workers,
                                     "ABLATION")

    print(f"\n  {'Dropped':<10}  {'F1':>6}  {'Delta':>8}  {'Verdict':<14}  "
          f"{'Hits':>8}  {'FA':>4}  {'Lead':>7}")
    print("  " + "─" * 70)

    survivors = []
    for res in sorted(ablation_results, key=lambda r: r["combo"]["f1"], reverse=True):
        feat_set = set(res["features"])
        dropped  = next((f for f in all_features if f not in feat_set), "?")
        delta    = res["combo"]["f1"] - baseline_f1
        verdict  = ("▲ DROP IT"   if delta >  0.02 else
                    ("▼ KEEP IT"  if delta < -0.02 else "~ neutral"))
        sc = res["combo"]
        print(f"  {dropped:<10}  {sc['f1']:>6.3f}  {delta:>+8.3f}  {verdict:<14}  "
              f"{sc['hits']:>2}/{sc['n_peaks']:<3}  {sc['false_alarms']:>4}  "
              f"{sc['avg_lead_days']:>6.0f}d")
        if delta <= 0.01:          # removing hurts or is neutral → keep
            survivors.append(dropped)

    dropped_features = [f for f in all_features if f not in survivors]
    print(f"\n  ✓ Survivors ({len(survivors)}): {survivors}")
    print(f"  ✗ Dropped   ({len(dropped_features)}): {dropped_features}")
    return survivors


# ── Stage 2: combinatorial ────────────────────────────────────────────────────

def _stage2_sweep(features: list[str], peaks: list[pd.Timestamp],
                  workers: int, min_features: int = 3) -> list[dict]:
    combos = []
    for k in range(min_features, len(features) + 1):
        combos.extend(list(c) for c in itertools.combinations(features, k))
    print("\n" + "═" * 108)
    print(f"  STAGE 2 — COMBINATORIAL SWEEP  "
          f"({len(combos)} combinations of {min_features}..{len(features)} features)")
    print("═" * 108)
    return _run_parallel(combos, peaks, workers, "COMBO SWEEP")


# ── SPX peak auto-detection ───────────────────────────────────────────────────

def _detect_spx_peaks(spx: pd.Series, train_start: pd.Timestamp,
                      prominence: float = 0.12) -> list[pd.Timestamp]:
    from scipy.signal import find_peaks
    sub = spx[spx.index >= train_start].dropna()
    if sub.empty:
        return []
    idx, _ = find_peaks(np.log(sub.values), distance=63,
                        prominence=np.log(1 + prominence))
    return [sub.index[i] for i in idx]


# ── Leaderboard ───────────────────────────────────────────────────────────────

def _print_leaderboard(results: list[dict], top_n: int = 25) -> None:
    ranked = sorted(
        [r for r in results if "error" not in r],
        key=lambda r: (r["combo"]["f1"], r["combo"]["avg_lead_days"]),
        reverse=True,
    )
    print("\n" + "═" * 115)
    print(f"  TOP-{top_n} FEATURE COMBINATIONS FOR TOP DETECTION")
    print("═" * 115)
    print(f"  {'Rank':>4}  {'Features':<60}  {'n':>2}  {'Best signal':<14}  "
          f"{'F1':>6}  {'Hits':>6}  {'FA':>4}  {'Lead':>7}  {'Flag%':>6}")
    print("  " + "─" * 111)

    for rank, res in enumerate(ranked[:top_n], 1):
        sc       = res["combo"]
        feat_str = ",".join(res["features"])[:59]
        best_sig = res.get("best_signal", "?")[:14]
        print(
            f"  {rank:>4}.  {feat_str:<60}  {res['n_features']:>2}  "
            f"{best_sig:<14}  {sc['f1']:>6.3f}  "
            f"{sc['hits']:>2}/{sc['n_peaks']:<3}  {sc['false_alarms']:>4}  "
            f"{sc['avg_lead_days']:>6.0f}d  {sc['pct_flagged']*100:>5.1f}%"
        )

    if ranked:
        best = ranked[0]
        print(f"\n  ★  WINNER: {best['features']}")
        print(f"     n_states={best['n_states']}  ci_anchor={best['ci_anchor']}  "
              f"best_signal={best.get('best_signal','?')}")
        sc = best["combo"]
        print(f"     Best:       F1={sc['f1']:.3f}  "
              f"hits={sc['hits']}/{sc['n_peaks']}  FA={sc['false_alarms']}  "
              f"lead={sc['avg_lead_days']:.0f}d  flagged={sc['pct_flagged']*100:.1f}%")
        for sig_key, label in [("regime_exit","Regime-exit"), ("ll_stress","LL-stress"),
                                ("combo_or","OR-combo"),     ("combo_and","AND-combo")]:
            s = best.get(sig_key, {})
            if s:
                print(f"     {label:<12} F1={s['f1']:.3f}  "
                      f"hits={s['hits']}/{s['n_peaks']}  FA={s['false_alarms']}  "
                      f"lead={s['avg_lead_days']:.0f}d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global _FULL_MATRIX

    parser = argparse.ArgumentParser(description="Top-signal HMM feature search")
    parser.add_argument("--full",         action="store_true",
                        help="Sweep all 1023 combinations (slow)")
    parser.add_argument("--no-ablation",  action="store_true",
                        help="Skip stage 1 (use all features in stage 2)")
    parser.add_argument("--workers",      type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel threads (default: CPU count - 1)")
    parser.add_argument("--min-features", type=int, default=3,
                        help="Min features per combination in stage 2 (default: 3)")
    args = parser.parse_args()

    print("═" * 108)
    print("  TOP-SIGNAL HMM FEATURE SEARCH")
    print(f"  Mode: {'FULL SWEEP' if args.full else 'TWO-STAGE'}  |  "
          f"Workers: {args.workers}  |  Output: {_OUT_PATH}")
    print("═" * 108)

    print("\n[1/4] Loading all features into shared matrix...")
    _FULL_MATRIX = _build_full_matrix()
    train_start  = _FULL_MATRIX.index[0]
    train_end    = _FULL_MATRIX.index[-1]
    print(f"  Matrix shape: {_FULL_MATRIX.shape}  "
          f"({train_start.date()} → {train_end.date()})")
    print(f"  Columns: {list(_FULL_MATRIX.columns)}")

    print("\n[2/4] Loading SPX and detecting peaks...")
    spx = _load_spx()

    hardcoded = [pd.Timestamp(d) for d, _ in _KNOWN_PEAKS
                 if train_start <= pd.Timestamp(d) <= train_end]
    auto_peaks = _detect_spx_peaks(spx, train_start)

    all_peaks: set[pd.Timestamp] = set(hardcoded)
    for ap in auto_peaks:
        if not any(abs((ap - hp).days) < 60 for hp in all_peaks):
            all_peaks.add(ap)

    peaks = sorted(all_peaks)
    print(f"  Using {len(peaks)} peaks:")
    for p in peaks:
        label = next((l for d, l in _KNOWN_PEAKS if pd.Timestamp(d) == p),
                     "auto-detected")
        print(f"    {p.date()}  {label}")

    print("\n[3/4] Running search...")
    all_results: list[dict] = []

    if args.full:
        all_combos = []
        for k in range(args.min_features, len(_ALL_FEATURES) + 1):
            all_combos.extend(list(c)
                              for c in itertools.combinations(_ALL_FEATURES, k))
        all_results = _run_parallel(all_combos, peaks, args.workers, "FULL SWEEP")
    else:
        if not args.no_ablation:
            survivors = _stage1_ablation(_ALL_FEATURES, peaks, args.workers)
        else:
            survivors = _ALL_FEATURES

        if len(survivors) >= args.min_features:
            stage2 = _stage2_sweep(survivors, peaks, args.workers, args.min_features)
            all_results.extend(stage2)
        else:
            print(f"  Only {len(survivors)} survivors — running as single combo")
            all_results.append(_worker((survivors, peaks)))

        # Always include the all-features baseline
        print("\n  Adding full-feature baseline...")
        bl = _worker((_ALL_FEATURES, peaks))
        if "error" not in bl:
            all_results.append(bl)

    print(f"\n[4/4] Saving {len(all_results)} results...")
    _print_leaderboard(all_results)

    output = {
        "generated_at":           datetime.utcnow().isoformat(),
        "mode":                   "full" if args.full else "two-stage",
        "train_start":            train_start.strftime("%Y-%m-%d"),
        "train_end":              train_end.strftime("%Y-%m-%d"),
        "peaks_used":             [p.strftime("%Y-%m-%d") for p in peaks],
        "signal_params": {
            "ll_stress_window":        _LL_TREND_WINDOW,
            "ll_stress_thresh":        _LL_TREND_THRESH,
            "peak_lead_window_days":   _PEAK_LEAD_WINDOW,
            "peak_trail_window_days":  _PEAK_TRAIL_WINDOW,
            "late_cycle_labels":       sorted(_LATE_CYCLE_LABELS),
        },
        "n_combinations_tested":  len([r for r in all_results if "error" not in r]),
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

    print(f"\n  ✓ Results written to {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
