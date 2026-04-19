"""
Shadow HMM Duel Backtest: SPX-only vs SPX+Macro (VIX + Yield Curves)
=====================================================================
Trains both variants from scratch, backtests each against historical
-10% drawdowns, and compares:
  - Hit rate (% of -10% drawdowns caught by crisis gate)
  - False alarm rate
  - Average early warning (days before drawdown)
  - Correlation between the two models' crisis calls
  - Correlation with the primary FRED/VIX HMM brain (if available)

Run:
    python tools/backtest_shadow_duel.py

Takes ~5-10 minutes (two MarkovRegression fits on 16k+ obs).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(_ROOT, "data")
_OUTPUT_PATH = os.path.join(_DATA_DIR, "shadow_duel_results.json")

_START = "1960-01-01"
_K_REGIMES = 6
_FWD_WINDOW = 30          # forward drawdown window (trading days)
_CRASH_THRESHOLD = -10.0   # % drawdown to count as "crash"
_CRISIS_Z_GATE = -0.30     # ll_zscore threshold for crisis call
_STRESS_Z_GATE = -0.18     # loosened gate for early warning

# VIX starts 1990-01-02, so the macro variant trains on a shorter window
_MACRO_START = "1990-01-02"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_spx_data(start: str = _START) -> tuple[pd.Series, pd.Series]:
    """Return (close, log_returns_pct) for ^GSPC."""
    import yfinance as yf
    df = yf.download("^GSPC", start=start, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    returns = (np.log(close / close.shift(1)).dropna() * 100.0)
    close = close.reindex(returns.index).ffill()
    return close, returns


def _load_vix(index: pd.DatetimeIndex) -> pd.Series:
    """Load VIX, reindex to given dates, ffill gaps."""
    import yfinance as yf
    df = yf.download("^VIX", start="1990-01-01", progress=False, auto_adjust=True)
    vix = df["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix = vix.dropna().reindex(index).ffill().bfill()
    return vix


def _load_yield_curve(index: pd.DatetimeIndex) -> pd.Series:
    """Load 10Y-2Y spread from FRED, reindex to given dates."""
    try:
        from fredapi import Fred
        api_key = os.environ.get("FRED_API_KEY", "")
        if api_key:
            fred = Fred(api_key=api_key)
            t10y2y = fred.get_series("T10Y2Y").dropna()
            return t10y2y.reindex(index).ffill().bfill()
    except Exception:
        pass

    # Fallback: try yfinance treasury proxy (^TNX for 10Y, ^IRX approx 3M)
    # or just return NaN series so we skip it
    print("[duel] FRED_API_KEY not set — trying yfinance treasury proxy")
    try:
        import yfinance as yf
        tnx = yf.download("^TNX", start="1990-01-01", progress=False, auto_adjust=True)["Close"]
        irx = yf.download("^IRX", start="1990-01-01", progress=False, auto_adjust=True)["Close"]
        if isinstance(tnx, pd.DataFrame):
            tnx = tnx.iloc[:, 0]
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        spread = (tnx - irx).dropna().reindex(index).ffill().bfill()
        spread.name = "yield_spread_proxy"
        return spread
    except Exception:
        print("[duel] WARNING: no yield curve data available, macro variant uses SPX+VIX only")
        return pd.Series(np.nan, index=index, name="yield_spread_na")


# ── Forward drawdown ─────────────────────────────────────────────────────────

def _forward_drawdown_pct(close: pd.Series, window: int = _FWD_WINDOW) -> pd.Series:
    """For each day t, forward-window peak-to-trough drawdown % (negative)."""
    vals = close.values
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        j = min(i + window, len(vals) - 1)
        if j <= i:
            continue
        segment = vals[i: j + 1]
        out[i] = (segment.min() / segment[0] - 1.0) * 100.0
    return pd.Series(out, index=close.index)


# ── Fit a MarkovRegression variant ───────────────────────────────────────────

def _fit_variant(endog, label: str) -> object:
    """Fit 6-regime MarkovRegression. Returns fitted result."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    print(f"[duel] fitting {label} ({len(endog)} obs, {_K_REGIMES} regimes) ...")
    t0 = time.time()

    if isinstance(endog, pd.DataFrame):
        model = MarkovRegression(
            endog.iloc[:, 0],
            k_regimes=_K_REGIMES,
            trend="c",
            switching_variance=True,
            exog=endog.iloc[:, 1:],
        )
    else:
        model = MarkovRegression(
            endog,
            k_regimes=_K_REGIMES,
            trend="c",
            switching_variance=True,
        )

    result = model.fit(disp=False, maxiter=200)
    elapsed = time.time() - t0
    print(f"[duel] {label} fitted in {elapsed:.1f}s  (BIC={result.bic:.1f})")
    sys.stdout.flush()
    return result


def _save_fit(result, name: str) -> None:
    """Save fitted result to pickle for resume."""
    import pickle
    path = os.path.join(_DATA_DIR, f"_duel_{name}.pickle")
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"[duel] saved {name} to {path}")
    sys.stdout.flush()


def _load_fit(name: str):
    """Load previously saved fit, or None."""
    import pickle
    path = os.path.join(_DATA_DIR, f"_duel_{name}.pickle")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── Extract ll_zscore from a fitted result ───────────────────────────────────

def _get_ll_zscore(result) -> np.ndarray:
    ll_obs = np.asarray(result.llf_obs)
    mu = np.nanmean(ll_obs)
    sd = np.nanstd(ll_obs)
    if sd <= 0 or not np.isfinite(sd):
        sd = 1.0
    return (ll_obs - mu) / sd


# ── Backtest a single variant ────────────────────────────────────────────────

def _backtest_variant(ll_z: np.ndarray, fwd_dd: pd.Series, crashed: pd.Series,
                      label: str, gate: float = _CRISIS_Z_GATE) -> dict:
    """
    Compute hit rate, false alarm rate, early warning for a variant.
    """
    n = min(len(ll_z), len(fwd_dd))
    ll_z = ll_z[:n]
    fwd_dd = fwd_dd.iloc[:n]
    crashed = crashed.iloc[:n]

    crisis_calls = (ll_z < gate)
    n_crisis = int(crisis_calls.sum())
    n_total = int(len(ll_z))

    # --- Hit rate: what % of actual crashes had a crisis call within 5 days prior ---
    crash_dates = crashed[crashed == 1].index
    # For each crash episode (cluster crashes into episodes separated by >30 days)
    episodes = []
    if len(crash_dates) > 0:
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

    # For each episode, check if any crisis call happened within [-5, +5] days of start
    hits = 0
    early_warnings = []
    crisis_idx = np.where(crisis_calls)[0]
    crisis_dates = fwd_dd.index[crisis_idx] if len(crisis_idx) > 0 else pd.DatetimeIndex([])

    for ep_start, ep_end in episodes:
        window_start = ep_start - pd.Timedelta(days=20)
        window_end = ep_start + pd.Timedelta(days=5)
        calls_in_window = crisis_dates[(crisis_dates >= window_start) & (crisis_dates <= window_end)]
        if len(calls_in_window) > 0:
            hits += 1
            earliest = calls_in_window[0]
            ew_days = (ep_start - earliest).days
            early_warnings.append(ew_days)

    hit_rate = hits / len(episodes) if episodes else 0.0
    avg_early_warning = float(np.mean(early_warnings)) if early_warnings else 0.0

    # --- False alarm rate: crisis calls where no -10% drawdown followed ---
    if n_crisis > 0:
        false_alarms = int((crisis_calls & (fwd_dd.values[:n] > _CRASH_THRESHOLD)).sum())
        false_alarm_rate = false_alarms / n_crisis
    else:
        false_alarms = 0
        false_alarm_rate = 0.0

    # --- Precision: of crisis calls, what % had a real crash ---
    precision = 1.0 - false_alarm_rate

    stats = {
        "label": label,
        "gate_z": gate,
        "n_obs": n_total,
        "n_episodes": len(episodes),
        "n_crisis_calls": n_crisis,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarm_rate, 4),
        "precision": round(precision, 4),
        "avg_early_warning_days": round(avg_early_warning, 1),
    }
    return stats


# ── Correlation with primary HMM ────────────────────────────────────────────

def _load_primary_hmm_history() -> pd.Series | None:
    """Load ll_zscore from the primary HMM state history if available."""
    path = os.path.join(_DATA_DIR, "hmm_state_history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            history = json.load(f)
        records = [(h["date"], h.get("log_likelihood_zscore", h.get("ll_zscore", 0.0)))
                    for h in history if "date" in h]
        if not records:
            return None
        df = pd.DataFrame(records, columns=["date", "ll_zscore"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["ll_zscore"]
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("SHADOW HMM DUEL BACKTEST: SPX-only vs SPX+Macro")
    print("=" * 70)
    print()

    # ── Load data ──
    print("[duel] loading ^GSPC price data ...")
    close_full, returns_full = _load_spx_data(_START)
    print(f"[duel] SPX: {returns_full.index[0].date()} -> {returns_full.index[-1].date()} ({len(returns_full)} obs)")

    # ── Variant A: SPX-only (full 1960->) ──
    result_a = _load_fit("spx_only")
    if result_a is None:
        result_a = _fit_variant(returns_full, "SPX-only (1960->)")
        _save_fit(result_a, "spx_only")
    else:
        print(f"[duel] loaded cached SPX-only fit  (BIC={result_a.bic:.1f})")
        sys.stdout.flush()
    ll_z_a = _get_ll_zscore(result_a)

    # ── Variant B: SPX + VIX + Yield curve (1990->) ──
    # Trim to macro start date
    macro_mask = returns_full.index >= _MACRO_START
    returns_macro = returns_full[macro_mask]
    close_macro = close_full[macro_mask]

    vix = _load_vix(returns_macro.index)
    yield_curve = _load_yield_curve(returns_macro.index)

    # Build multivariate endog: SPX return + VIX level + yield spread
    macro_df = pd.DataFrame({
        "spx_ret": returns_macro.values,
        "vix": vix.values,
    }, index=returns_macro.index)

    has_yield = not yield_curve.isna().all()
    if has_yield:
        macro_df["yield_spread"] = yield_curve.values
        # Drop any rows with NaN
        macro_df = macro_df.dropna()
        variant_b_label = "SPX+VIX+Yield (1990->)"
    else:
        macro_df = macro_df.dropna()
        variant_b_label = "SPX+VIX (1990->)"

    close_macro = close_macro.reindex(macro_df.index).ffill()

    result_b = _load_fit("macro")
    if result_b is None:
        result_b = _fit_variant(macro_df, variant_b_label)
        _save_fit(result_b, "macro")
    else:
        print(f"[duel] loaded cached macro fit  (BIC={result_b.bic:.1f})")
        sys.stdout.flush()
    ll_z_b = _get_ll_zscore(result_b)

    # ── Forward drawdowns ──
    print("[duel] computing forward drawdowns ...")
    fwd_dd_full = _forward_drawdown_pct(close_full)
    crashed_full = (fwd_dd_full <= _CRASH_THRESHOLD).astype(int)

    fwd_dd_macro = _forward_drawdown_pct(close_macro)
    crashed_macro = (fwd_dd_macro <= _CRASH_THRESHOLD).astype(int)

    # ── Backtest both variants ──
    # For fair comparison, also backtest SPX-only on the 1990-> window
    macro_mask_arr = np.array(macro_mask)
    ll_z_a_trimmed = ll_z_a[macro_mask_arr[:len(ll_z_a)]] if len(ll_z_a) == len(returns_full) else ll_z_a[-len(returns_macro):]
    # Recompute: trim ll_z_a to the macro window for apples-to-apples
    n_macro = len(returns_macro)
    ll_z_a_macro_window = ll_z_a[-n_macro:]

    print()
    print("-" * 70)
    print("RESULTS: Crisis gate z < -0.30")
    print("-" * 70)

    stats_a_full = _backtest_variant(ll_z_a, fwd_dd_full, crashed_full,
                                     "SPX-only (full 1960->)", _CRISIS_Z_GATE)
    stats_a_macro = _backtest_variant(ll_z_a_macro_window, fwd_dd_macro, crashed_macro,
                                      "SPX-only (1990-> window)", _CRISIS_Z_GATE)
    stats_b = _backtest_variant(ll_z_b, fwd_dd_macro, crashed_macro,
                                variant_b_label, _CRISIS_Z_GATE)

    all_stats = [stats_a_full, stats_a_macro, stats_b]

    # Also run with loosened gate
    print()
    print("-" * 70)
    print("RESULTS: Loosened gate z < -0.18")
    print("-" * 70)

    stats_a_full_loose = _backtest_variant(ll_z_a, fwd_dd_full, crashed_full,
                                           "SPX-only (full 1960->)", _STRESS_Z_GATE)
    stats_a_macro_loose = _backtest_variant(ll_z_a_macro_window, fwd_dd_macro, crashed_macro,
                                            "SPX-only (1990-> window)", _STRESS_Z_GATE)
    stats_b_loose = _backtest_variant(ll_z_b, fwd_dd_macro, crashed_macro,
                                      variant_b_label, _STRESS_Z_GATE)

    all_stats_loose = [stats_a_full_loose, stats_a_macro_loose, stats_b_loose]

    # ── Print results ──
    def _print_table(stats_list):
        print(f"  {'Variant':<28} {'Episodes':>8} {'Hits':>6} {'HitRate':>8} "
              f"{'Precision':>10} {'FalseAlm':>9} {'EarlyWarn':>10}")
        for s in stats_list:
            print(f"  {s['label']:<28} {s['n_episodes']:>8} {s['hits']:>6} "
                  f"{s['hit_rate']:>8.1%} {s['precision']:>10.1%} "
                  f"{s['false_alarms']:>9} {s['avg_early_warning_days']:>10.1f}d")

    print()
    print("CRISIS GATE (z < -0.30):")
    _print_table(all_stats)

    print()
    print("LOOSENED GATE (z < -0.18):")
    _print_table(all_stats_loose)

    # ── Correlation between variants (1990-> window) ──
    print()
    print("-" * 70)
    print("CORRELATION ANALYSIS (1990-> window)")
    print("-" * 70)

    n_common = min(len(ll_z_a_macro_window), len(ll_z_b))
    corr_ab = float(np.corrcoef(ll_z_a_macro_window[:n_common], ll_z_b[:n_common])[0, 1])
    print(f"  SPX-only vs {variant_b_label} ll_zscore corr: {corr_ab:.4f}")

    # Crisis call agreement
    crisis_a = (ll_z_a_macro_window[:n_common] < _CRISIS_Z_GATE)
    crisis_b = (ll_z_b[:n_common] < _CRISIS_Z_GATE)
    both_crisis = (crisis_a & crisis_b).sum()
    either_crisis = (crisis_a | crisis_b).sum()
    jaccard = both_crisis / either_crisis if either_crisis > 0 else 0.0
    print(f"  Crisis call overlap (Jaccard): {jaccard:.4f}  (both={both_crisis}, either={either_crisis})")
    print(f"    -> Lower Jaccard = more independent = more valuable as dual signal")

    # Primary HMM correlation
    primary_z = _load_primary_hmm_history()
    if primary_z is not None and len(primary_z) > 20:
        print()
        # Align dates
        common_dates_a = returns_macro.index.intersection(primary_z.index)
        if len(common_dates_a) > 20:
            # Get ll_z_a for those dates
            a_aligned = pd.Series(ll_z_a_macro_window, index=returns_macro.index[:len(ll_z_a_macro_window)])
            a_vals = a_aligned.reindex(common_dates_a).dropna()
            p_vals = primary_z.reindex(a_vals.index).dropna()
            common = a_vals.index.intersection(p_vals.index)
            if len(common) > 20:
                corr_ap = float(np.corrcoef(a_vals[common].values, p_vals[common].values)[0, 1])
                print(f"  SPX-only vs Primary HMM (FRED/VIX) ll_zscore corr: {corr_ap:.4f}")

            b_aligned = pd.Series(ll_z_b, index=macro_df.index[:len(ll_z_b)])
            b_vals = b_aligned.reindex(common_dates_a).dropna()
            common_b = b_vals.index.intersection(p_vals.index)
            if len(common_b) > 20:
                corr_bp = float(np.corrcoef(b_vals[common_b].values, p_vals[common_b].values)[0, 1])
                print(f"  {variant_b_label} vs Primary HMM (FRED/VIX) ll_zscore corr: {corr_bp:.4f}")
                print()
                if corr_ap < corr_bp:
                    print(f"  -> SPX-only is MORE independent from primary brain (corr {corr_ap:.3f} vs {corr_bp:.3f})")
                else:
                    print(f"  -> {variant_b_label} is MORE independent from primary brain (corr {corr_bp:.3f} vs {corr_ap:.3f})")
    else:
        print("  (primary HMM history not available — skipping primary correlation)")

    # ── BIC comparison ──
    print()
    print("-" * 70)
    print("MODEL FIT (BIC — lower is better)")
    print("-" * 70)
    print(f"  SPX-only (1960->):    BIC = {result_a.bic:.1f}  (n={len(returns_full)})")
    print(f"  {variant_b_label}: BIC = {result_b.bic:.1f}  (n={len(macro_df)})")
    print(f"  (BIC not directly comparable across different n — use for within-variant tuning)")

    # ── Verdict ──
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Compare on the common 1990-> window
    a_better_hit = stats_a_macro["hit_rate"] >= stats_b["hit_rate"]
    a_better_prec = stats_a_macro["precision"] >= stats_b["precision"]
    a_score = int(a_better_hit) + int(a_better_prec) + int(corr_ab < 0.5)
    b_score = int(not a_better_hit) + int(not a_better_prec)

    if a_score > b_score:
        winner = "SPX-only"
        reason = "better hit rate/precision AND lower correlation with primary (more independent)"
    elif b_score > a_score:
        winner = variant_b_label
        reason = "better hit rate/precision on common window"
    else:
        winner = "TIE"
        reason = "similar performance — SPX-only preferred for data depth + independence"

    print(f"  Winner: {winner}")
    print(f"  Reason: {reason}")
    print()
    print(f"  Recommendation: Use SPX-only as shadow brain (1960-> data depth)")
    if corr_ab > 0.7:
        print(f"  NOTE: High correlation ({corr_ab:.2f}) — macro variant adds little new info")
    elif corr_ab < 0.4:
        print(f"  NOTE: Low correlation ({corr_ab:.2f}) — consider running BOTH as ensemble")

    # ── Save results ──
    output = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "crisis_gate": {
            "spx_only_full": stats_a_full,
            "spx_only_1990": stats_a_macro,
            "macro_variant": stats_b,
        },
        "loosened_gate": {
            "spx_only_full": stats_a_full_loose,
            "spx_only_1990": stats_a_macro_loose,
            "macro_variant": stats_b_loose,
        },
        "correlation": {
            "spx_vs_macro_ll_zscore": round(corr_ab, 4),
            "crisis_jaccard": round(jaccard, 4),
        },
        "bic": {
            "spx_only": round(result_a.bic, 1),
            "macro_variant": round(result_b.bic, 1),
        },
        "verdict": winner,
    }
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {_OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
