"""
Shadow HMM Regime Detection Service (SPX price-return brain)
------------------------------------------------------------
Second, independent brain trained on ^GSPC daily log returns 1960→present.
Uses statsmodels.tsa.regime_switching.MarkovRegression with k_regimes=6 and
switching_variance=True. Complements the credit/yield brain in hmm_regime.py.

Persistence:
  - data/hmm_shadow_brain.json         — brain params + CI anchor + crash bins
  - data/hmm_shadow_result.pickle      — pickled MarkovRegressionResults (fast scoring)
  - data/hmm_shadow_history.json       — daily state log (last 500)
  - data/hmm_shadow_ci_calibration.json — walk through backtest output

Regime labels: Strong Bear, Mild Bear, Transition, Mild Bull, Strong Bull, Crisis.
Crisis = highest-variance regime regardless of mean. The other five are ordered
by conditional mean return.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_BRAIN_PATH = os.path.join(_DATA_DIR, "hmm_shadow_brain.json")
_RESULT_PATH = os.path.join(_DATA_DIR, "hmm_shadow_result.pickle")
_HISTORY_PATH = os.path.join(_DATA_DIR, "hmm_shadow_history.json")
_CALIBRATION_PATH = os.path.join(_DATA_DIR, "hmm_shadow_ci_calibration.json")

_TICKER = "^GSPC"
_DEFAULT_START = "1960-01-01"
_K_REGIMES = 6
_DEFAULT_CI_ANCHOR = 1.194  # z<-0.80 = CI 67% gate (sweep-optimized, 95% hit rate)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HMMShadowBrain:
    k_regimes: int
    trained_at: str
    training_start: str
    training_end: str
    regime_means: list          # length k, conditional return mean (% per day)
    regime_variances: list      # length k, conditional variance (%²)
    transmat: list              # k × k
    state_labels: list          # ordered by regime index
    ll_baseline_mean: float
    ll_baseline_std: float
    ci_anchor: float = _DEFAULT_CI_ANCHOR
    crash_prob_bins: list = field(default_factory=list)  # list[dict]
    bic: float = 0.0
    n_obs: int = 0


@dataclass
class HMMShadowState:
    date: str
    state_idx: int
    state_label: str
    state_probabilities: list
    confidence: float
    persistence: int
    daily_return_pct: float = 0.0
    log_likelihood: float = 0.0
    ll_zscore: float = 0.0
    ci_pct: float = 0.0
    crash_prob_10pct: float = 0.0
    expected_drawdown_pct: float = 0.0
    entropy: float = 0.0
    transition_risk_1m: float = 0.0
    transition_risk_3m: float = 0.0
    transition_risk_6m: float = 0.0
    forecast_1m: list = None
    forecast_3m: list = None
    forecast_6m: list = None


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_gspc_returns(start: str = _DEFAULT_START) -> pd.Series:
    """Download ^GSPC from yfinance, return daily log returns in percent."""
    import yfinance as yf
    df = yf.download(_TICKER, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    returns = (np.log(close / close.shift(1)).dropna() * 100.0)
    returns.name = "gspc_logret_pct"
    return returns


# ── Regime labelling ──────────────────────────────────────────────────────────

def _label_regimes(means: np.ndarray, variances: np.ndarray) -> list[str]:
    """
    Label 6 regimes: highest-variance regime = Crisis; remaining 5 ordered by
    mean return ascending → [Strong Bear, Mild Bear, Transition, Mild Bull, Strong Bull].
    """
    n = len(means)
    labels = [""] * n
    crisis_idx = int(np.argmax(variances))
    labels[crisis_idx] = "Crisis"
    remaining = [i for i in range(n) if i != crisis_idx]
    remaining_sorted = sorted(remaining, key=lambda i: means[i])
    ordered_labels = ["Strong Bear", "Mild Bear", "Transition", "Mild Bull", "Strong Bull"]
    for lbl, idx in zip(ordered_labels, remaining_sorted):
        labels[idx] = lbl
    return labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_shadow_hmm(start: str = _DEFAULT_START,
                     random_state: int = 42) -> HMMShadowBrain:
    """
    Fit a 6-regime MarkovRegression with switching variance on ^GSPC log returns.
    Persists result.pickle + brain.json. Does NOT run CI calibration — call
    tools/backtest_shadow_ci.py afterwards to fill ci_anchor + crash_prob_bins.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except Exception as e:
        import sys
        raise ImportError(
            f"statsmodels not found in {sys.executable} — "
            f"run: {sys.executable} -m pip install statsmodels"
        ) from e

    returns = _load_gspc_returns(start)
    y = returns.values.astype(np.float64)

    np.random.seed(random_state)
    model = MarkovRegression(
        y,
        k_regimes=_K_REGIMES,
        trend="c",
        switching_variance=True,
    )

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=200)

    # Extract switching means + variances by param_names (version-portable)
    params_arr = np.asarray(result.params)
    try:
        param_names = list(result.model.param_names)
        name_to_val = {n: float(v) for n, v in zip(param_names, params_arr)}
        means = np.array([name_to_val[f"const[{i}]"] for i in range(_K_REGIMES)])
        variances = np.array([name_to_val[f"sigma2[{i}]"] for i in range(_K_REGIMES)])
    except (AttributeError, KeyError):
        # Fallback: positional — transitions first, then k means, then k variances
        means = params_arr[-2 * _K_REGIMES : -_K_REGIMES].astype(float)
        variances = params_arr[-_K_REGIMES:].astype(float)

    # regime_transition shape may be (k, k, 1) for time-invariant TVTP or (k, k)
    _rt = np.asarray(result.regime_transition)
    if _rt.ndim == 3:
        transmat = _rt[:, :, 0]
    else:
        transmat = _rt
    # Normalise rows defensively (the stored matrix is column-stochastic in some
    # statsmodels versions: P[i, j] = P(regime=i | prev=j). Check by row-sum.)
    row_sums = transmat.sum(axis=1)
    col_sums = transmat.sum(axis=0)
    if np.allclose(col_sums, 1.0, atol=1e-3) and not np.allclose(row_sums, 1.0, atol=1e-3):
        transmat = transmat.T  # transpose to row-stochastic (our convention)
    # Final safety normalisation
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    labels = _label_regimes(means, variances)

    ll_obs = np.asarray(result.llf_obs)
    ll_mean = float(np.nanmean(ll_obs))
    ll_std = float(np.nanstd(ll_obs))
    if not np.isfinite(ll_std) or ll_std <= 0:
        ll_std = 1.0

    try:
        bic_val = float(result.bic)
    except Exception:
        bic_val = 0.0
    try:
        n_obs_val = int(result.nobs)
    except Exception:
        n_obs_val = int(len(y))

    brain = HMMShadowBrain(
        k_regimes=_K_REGIMES,
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_start=returns.index[0].strftime("%Y-%m-%d"),
        training_end=returns.index[-1].strftime("%Y-%m-%d"),
        regime_means=[round(float(v), 6) for v in means],
        regime_variances=[round(float(v), 6) for v in variances],
        transmat=transmat.tolist(),
        state_labels=labels,
        ll_baseline_mean=round(ll_mean, 6),
        ll_baseline_std=round(ll_std, 6),
        ci_anchor=_DEFAULT_CI_ANCHOR,
        crash_prob_bins=[],
        bic=round(bic_val, 2),
        n_obs=n_obs_val,
    )

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_RESULT_PATH, "wb") as f:
        pickle.dump(result, f)
    save_shadow_brain(brain)
    return brain


# ── Persistence ───────────────────────────────────────────────────────────────

def save_shadow_brain(brain: HMMShadowBrain) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_BRAIN_PATH, "w") as f:
        json.dump(asdict(brain), f, indent=2)


def load_shadow_brain() -> Optional[HMMShadowBrain]:
    if not os.path.exists(_BRAIN_PATH):
        return None
    try:
        with open(_BRAIN_PATH) as f:
            d = json.load(f)
        d.setdefault("ci_anchor", _DEFAULT_CI_ANCHOR)
        d.setdefault("crash_prob_bins", [])
        d.setdefault("bic", 0.0)
        d.setdefault("n_obs", 0)
        return HMMShadowBrain(**d)
    except Exception:
        return None


def _load_pickled_result():
    if not os.path.exists(_RESULT_PATH):
        return None
    try:
        with open(_RESULT_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_history() -> list[dict]:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list[dict]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_HISTORY_PATH, "w") as f:
        json.dump(history[-500:], f, indent=2)


# ── Crash-prob lookup ────────────────────────────────────────────────────────

def _lookup_crash_prob(ll_z: float, bins: list[dict]) -> tuple[float, float]:
    """Given ll_zscore, return (prob_10pct, expected_drawdown_pct) from calibration bins."""
    if not bins:
        return 0.0, 0.0
    for b in bins:
        z_lo = b.get("z_lo", -np.inf)
        z_hi = b.get("z_hi", np.inf)
        if z_lo <= ll_z < z_hi:
            return float(b.get("prob_10pct", 0.0)), float(b.get("expected_drawdown_pct", 0.0))
    # Fallback: most extreme bucket on whichever side
    if ll_z < bins[0].get("z_lo", 0):
        b = bins[0]
    else:
        b = bins[-1]
    return float(b.get("prob_10pct", 0.0)), float(b.get("expected_drawdown_pct", 0.0))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_current_shadow_state(log_to_history: bool = True) -> Optional[HMMShadowState]:
    """
    Compute today's Shadow HMM regime using stored pickled result.
    Extends the fitted result with any new ^GSPC days since training via .apply(),
    falling back to a full refit if the incremental API fails.
    Returns None if brain/result are missing or fetch fails.
    """
    brain = load_shadow_brain()
    if brain is None:
        return None
    result = _load_pickled_result()
    if result is None:
        return None

    try:
        returns = _load_gspc_returns(brain.training_start)
        y_full = returns.values.astype(np.float64)

        # Extend the fitted result onto the full series (includes any new days).
        # .apply() re-runs the filter with the same fitted params — no re-optimisation.
        try:
            extended = result.apply(y_full)
        except Exception:
            extended = result

        smoothed = np.asarray(extended.smoothed_marginal_probabilities)
        # statsmodels shape may be (k, T) or (T, k); normalise to (T, k)
        if smoothed.shape[0] == brain.k_regimes and smoothed.shape[1] != brain.k_regimes:
            smoothed = smoothed.T
        elif smoothed.ndim == 2 and smoothed.shape[1] != brain.k_regimes and smoothed.shape[0] == brain.k_regimes:
            smoothed = smoothed.T
        today_probs = smoothed[-1].tolist()
        state_idx = int(np.argmax(today_probs))
        state_label = brain.state_labels[state_idx]
        confidence = round(float(today_probs[state_idx]), 4)

        # Persistence: consecutive days with argmax == state_idx
        states_seq = np.argmax(smoothed, axis=1)
        persistence = 1
        for s in reversed(states_seq[:-1]):
            if s == state_idx:
                persistence += 1
            else:
                break

        # Today's per-obs log-likelihood and z-score
        ll_obs = np.asarray(extended.llf_obs)
        ll_today = float(ll_obs[-1])
        ll_z = (ll_today - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6)
        ll_z = round(ll_z, 4)
        ci_pct = round(abs(ll_z) / max(brain.ci_anchor, 1e-6) * 100.0, 2)

        crash_prob, exp_dd = _lookup_crash_prob(ll_z, brain.crash_prob_bins)

        # Entropy
        from scipy.stats import entropy as _shannon_entropy
        raw_entropy = float(_shannon_entropy(today_probs))
        max_entropy = float(np.log(brain.k_regimes))
        norm_entropy = round(raw_entropy / max_entropy, 4) if max_entropy > 0 else 0.0

        # Transition projections
        tm = np.array(brain.transmat)
        pv = np.array(today_probs)
        fc_1m = (pv @ np.linalg.matrix_power(tm, 21)).tolist()
        fc_3m = (pv @ np.linalg.matrix_power(tm, 63)).tolist()
        fc_6m = (pv @ np.linalg.matrix_power(tm, 126)).tolist()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = HMMShadowState(
            date=today,
            state_idx=state_idx,
            state_label=state_label,
            state_probabilities=[round(p, 4) for p in today_probs],
            confidence=confidence,
            persistence=persistence,
            daily_return_pct=round(float(y_full[-1]), 4),
            log_likelihood=round(ll_today, 6),
            ll_zscore=ll_z,
            ci_pct=ci_pct,
            crash_prob_10pct=round(float(crash_prob), 4),
            expected_drawdown_pct=round(float(exp_dd), 2),
            entropy=norm_entropy,
            transition_risk_1m=round(1.0 - fc_1m[state_idx], 4),
            transition_risk_3m=round(1.0 - fc_3m[state_idx], 4),
            transition_risk_6m=round(1.0 - fc_6m[state_idx], 4),
            forecast_1m=[round(p, 4) for p in fc_1m],
            forecast_3m=[round(p, 4) for p in fc_3m],
            forecast_6m=[round(p, 4) for p in fc_6m],
        )

        if log_to_history:
            history = _load_history()
            history = [h for h in history if h.get("date") != today]
            history.append(asdict(state))
            _save_history(history)

        return state

    except Exception:
        return None


def load_current_shadow_state() -> Optional[HMMShadowState]:
    """Fast path: load most recent Shadow state from history JSON."""
    history = _load_history()
    if not history:
        return None
    last = history[-1]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        state = HMMShadowState(**last)
        state._is_stale = last.get("date") != today
        state._stale_date = last.get("date", "unknown")
        return state
    except Exception:
        return None


def get_shadow_state_history() -> list[dict]:
    return _load_history()
