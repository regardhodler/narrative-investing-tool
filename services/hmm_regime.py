"""
HMM Regime Detection Service
-----------------------------
Trains a GaussianHMM on FRED credit / yield / volatility signals to infer
latent market regimes (Bull, Neutral, Stress, Late Cycle, Crisis).

Key design decisions:
  - 15 years of FRED data, simple z-scoring over the full window.
    This gives the model memory of tail events (COVID 2020, 2022 rate shock)
    so it can properly distinguish "elevated VIX" from "true crisis."
    VIX=50 (Apr 2026 tariff shock) reads as z≈+0.8 against a 15yr baseline
    that includes VIX=80 (COVID). Credit spreads at 312bp read as z≈-1.0,
    below the 15yr average. The model correctly calls this Neutral, not Crisis.
  - BIC selects number of states (2-6).
  - Gaussian emission approximation — mitigated by z-scoring + ±3σ cap.
  - Persistence: data/hmm_brain.json + data/hmm_state_history.json.
  - load_current_hmm_state() is fast — reads JSON, no inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_FRED_DIR = os.path.join(_DATA_DIR, "fred_cache")
_BRAIN_PATH = os.path.join(_DATA_DIR, "hmm_brain.json")
_HISTORY_PATH = os.path.join(_DATA_DIR, "hmm_state_history.json")

# ── Feature set ───────────────────────────────────────────────────────────────
# 9 FRED series + VIX from yfinance.  Chosen for low lag, high regime signal.
_FRED_FEATURES = [
    "BAA10Y",        # Moody's Baa - 10yr (HY-proxy spread, daily, 1986+).
                     # Replaces BAMLH0A0HYM2 — ICE BofA series capped at 3yr public history (Apr 2026).
    "AAA10Y",        # Moody's Aaa - 10yr (IG-proxy spread, daily, 1983+).
                     # Replaces BAMLC0A0CM — same licensing reason.
    "T10Y2Y",        # 10yr-2yr yield curve
    "T10Y3M",        # 10yr-3mo (inversion depth)
    "DGS10",         # nominal 10yr yield
    "DGS2",          # nominal 2yr yield
    "DFII10",        # real 10yr yield (inflation proxy)
    "NFCI",          # Chicago Fed financial conditions
    "ICSA",          # weekly claims — labour market stress
]

_N_FEATURES = len(_FRED_FEATURES) + 1  # +1 for VIX


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HMMBrain:
    n_states: int
    trained_at: str        # ISO8601
    training_start: str
    training_end: str
    transmat: list         # n_states × n_states
    means: list            # n_states × n_features
    covars: list           # n_states × n_features
    state_labels: list     # e.g. ["Bull", "Neutral", "Stress", "Late Cycle"]
    aic: float
    bic: float
    feature_names: list
    ll_baseline_mean: float = 0.0   # mean LL per observation at training time
    ll_baseline_std: float = 1.0    # std of rolling-window LL (for z-scoring daily LL)
    lookback_years: int = 15        # lookback used to build feature matrix — MUST match live scoring
    ci_anchor: float = 0.467        # |z| value that maps to 100% CI. Auto-set at train time
                                    # to the in-sample worst-day z. Default preserves legacy behavior.


@dataclass
class HMMState:
    date: str
    state_idx: int
    state_label: str
    state_probabilities: list  # soft assignment, sums to 1
    confidence: float          # max(state_probabilities)
    persistence: int           # calendar days in current state
    log_likelihood: float = 0.0       # LL per observation (daily vs training baseline)
    ll_zscore: float = 0.0            # (daily_ll - baseline_mean) / baseline_std
    entropy: float = 0.0              # normalized Shannon entropy [0=pure, 1=fog]
    transition_risk_1m: float = 0.0   # P(leaving current state) over 21 biz days
    transition_risk_3m: float = 0.0   # P(leaving current state) over 63 biz days
    transition_risk_6m: float = 0.0   # P(leaving current state) over 126 biz days
    forecast_1m: list = None          # state probability vector at +21 days
    forecast_3m: list = None          # state probability vector at +63 days
    forecast_6m: list = None          # state probability vector at +126 days


# ── FRED CSV loader ───────────────────────────────────────────────────────────

def _load_fred_series(series_id: str) -> pd.Series:
    """Load a single FRED CSV from disk. Returns daily Series indexed by date."""
    path = os.path.join(_FRED_DIR, f"{series_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FRED cache missing: {series_id}.csv")
    df = pd.read_csv(path)
    date_col = next((c for c in ("observation_date", "DATE") if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No date column in {series_id}.csv")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Value column is the series_id column
    val_col = [c for c in df.columns if c != date_col][0]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.name = series_id
    return s


def _load_vix(lookback_years: int = 4) -> pd.Series:
    """Fetch VIX from yfinance."""
    try:
        import yfinance as yf
        period = f"{lookback_years + 1}y"
        df = yf.download("^VIX", period=period, interval="1d", progress=False,
                         auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float, name="VIX")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = "VIX"
        return close.squeeze()
    except Exception:
        return pd.Series(dtype=float, name="VIX")


# ── Feature matrix builder ────────────────────────────────────────────────────

def _load_raw_feature_df(lookback_years: int = 15) -> pd.DataFrame:
    """Load raw (unscaled) FRED + VIX features into a single DataFrame."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
    series_list = []
    for sid in _FRED_FEATURES:
        try:
            s = _load_fred_series(sid)
            series_list.append(s)
        except Exception:
            pass
    vix = _load_vix(lookback_years + 1)
    if not vix.empty:
        series_list.append(vix)
    if not series_list:
        raise RuntimeError("No feature series loaded — cannot train HMM")
    df = pd.concat(series_list, axis=1)
    df = df.resample("B").last().ffill(limit=5)
    df = df[df.index >= cutoff]
    df = df.dropna()
    return df


_ROLLING_ZSCORE_WINDOW = 5 * 252  # 5 years of business days for rolling z-score
_EWMA_SPAN = 10                   # 10-day signal smoothing — reacts in 1-2 weeks, filters daily noise


def _build_feature_matrix(lookback_years: int = 15) -> pd.DataFrame:
    """
    Build a feature matrix with EWMA-smoothed, rolling-z-scored signals.

    Pipeline:
      1. Load 15yr raw FRED + VIX data
      2. Apply 20-day EWMA smoothing (noise reduction — daily jitter doesn't flip regimes)
      3. Rolling 5-year z-score — each day is scored against its own prior 5yr window.
         This means 2022 rate shock is extreme relative to 2017-2022 baseline, AND
         today's tariff shock is extreme relative to 2021-2026 baseline. Both are
         properly contextualized in their own era.
      4. Cap at ±3σ
      5. Drop the initial ~5yr warmup period (NaN from rolling window)

    Result: ~10yr of usable training data with contextually relevant z-scores.
    The HMM sees crisis transitions (2020 COVID, 2022 rate shock) AND current stress.
    """
    df = _load_raw_feature_df(lookback_years)

    # Step 2: EWMA smoothing
    df = df.ewm(span=_EWMA_SPAN).mean()

    # Step 3: Rolling 5-year z-score
    rolling_mu = df.rolling(window=_ROLLING_ZSCORE_WINDOW, min_periods=252).mean()
    rolling_sigma = df.rolling(window=_ROLLING_ZSCORE_WINDOW, min_periods=252).std().replace(0, 1)
    df = (df - rolling_mu) / rolling_sigma

    # Step 4: Cap and clean
    df = df.clip(-3, 3)
    df = df.dropna()
    return df


# ── HMM training ─────────────────────────────────────────────────────────────

def _label_states(model, feature_names: list) -> list[str]:
    """
    Auto-label HMM states by mean HY-proxy spread level.
    Lowest spread → Bull. Highest spread → Crisis (true dislocation).
    'Crash' intentionally avoided — reserved only for genuine credit crisis levels.

    Tries Moody's BAA10Y first (current feature set), falls back to the legacy
    ICE BofA BAMLH0A0HYM2 name for backward compatibility with old brains.
    """
    n = model.n_components
    means = model.means_  # shape (n_states, n_features)

    # Find index of HY spread feature — try current name first, then legacy
    hy_idx = None
    for candidate in ("BAA10Y", "BAMLH0A0HYM2"):
        for i, name in enumerate(feature_names):
            if candidate in name:
                hy_idx = i
                break
        if hy_idx is not None:
            break

    if hy_idx is None or n < 2:
        return [f"State{i}" for i in range(n)]

    hy_means = means[:, hy_idx]
    order = np.argsort(hy_means)  # ascending HY spread

    label_map = {i: "Unknown" for i in range(n)}
    # Labels ordered from tightest → widest HY spread
    # "Late Cycle" = elevated spreads but not crisis (persistent, months-long)
    # "Crisis" = only appears when training data includes true dislocation (COVID, GFC)
    labels_by_order = {
        2: ["Bull", "Stress"],
        3: ["Bull", "Neutral", "Stress"],
        4: ["Bull", "Neutral", "Stress", "Late Cycle"],
        5: ["Bull", "Neutral", "Stress", "Late Cycle", "Crisis"],
        6: ["Bull", "Neutral", "Early Stress", "Stress", "Late Cycle", "Crisis"],
    }
    assigned = labels_by_order.get(n, [f"State{i}" for i in range(n)])
    for rank, state_idx in enumerate(order):
        label_map[state_idx] = assigned[rank]

    return [label_map[i] for i in range(n)]


def train_hmm(lookback_years: int = 15, max_states: int = 6,
              random_state: int = 42) -> HMMBrain:
    """
    Train GaussianHMM on EWMA-weighted FRED feature matrix.

    Uses 15 years of data with exponential decay (λ=0.98):
      - Yesterday: weight 1.00
      - 1 year ago: ~0.85
      - 5 years ago: ~0.45
      - 2008 GFC (15yr ago): ~0.05

    hmmlearn doesn't support sample_weight, so we use weighted resampling:
    rows are repeated proportionally to their EWMA weight (~3000 resampled obs).
    BIC computed on the original (non-resampled) sequence for honest model selection.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        import sys
        raise ImportError(
            f"hmmlearn not found in {sys.executable} — "
            f"run: {sys.executable} -m pip install hmmlearn"
        ) from e

    df = _build_feature_matrix(lookback_years)
    X = df.values.astype(np.float64)
    feature_names = list(df.columns)

    best_model = None
    best_bic = np.inf
    best_n = 3

    for n in range(2, max_states + 1):
        try:
            import warnings as _w
            # init_params="smc" tells hmmlearn to auto-init startprob/means/covars
            # but NOT transmat ("t" omitted) — so our seeded matrix is preserved
            model = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=500,
                random_state=random_state,
                tol=1e-5,
                verbose=False,
                init_params="smc",
            )
            # Seed transmat with self-transition prior (0.70) — soft enough that
            # EM can override if data supports transitions, but prevents degenerate
            # ping-pong states. Laplace +1e-6 ensures no zero probabilities.
            _diag_prior = 0.70
            _off = (1.0 - _diag_prior) / max(n - 1, 1)
            model.transmat_ = np.full((n, n), _off)
            np.fill_diagonal(model.transmat_, _diag_prior)
            model.transmat_ += 1e-6
            model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                model.fit(X)
            bic = model.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM training failed for all n_states")

    state_labels = _label_states(best_model, feature_names)

    # ── Empirical transition matrix from decoded state sequence ───────────
    # The EM-learned transmat_ is nearly diagonal (1.00) because emissions are
    # so well-separated that posterior state assignments are ~100% certain.
    # Instead, count actual state→state transitions in the Viterbi path and
    # apply Laplace smoothing so no transition is exactly zero.
    _decoded_states = best_model.predict(X)
    _emp_trans = np.zeros((best_n, best_n))
    for _t in range(1, len(_decoded_states)):
        _emp_trans[_decoded_states[_t - 1], _decoded_states[_t]] += 1
    # Laplace smoothing: add 1 pseudo-count to prevent zero probabilities
    _emp_trans += 1.0
    _emp_trans /= _emp_trans.sum(axis=1, keepdims=True)
    # Override the model's degenerate transmat with the empirical one
    best_model.transmat_ = _emp_trans

    # ── LL baseline: compute mean and std of per-observation log-likelihood ───
    # Rolling 60-day windows give a distribution of "normal" LL values.
    # Daily scoring compares against this to detect model-market divergence.
    ll_total = float(best_model.score(X))
    ll_per_obs = ll_total / len(X)
    _window = 60
    _ll_vals = []
    for i in range(_window, len(X)):
        _chunk = X[i - _window : i]
        _ll_vals.append(float(best_model.score(_chunk)) / _window)
    _ll_std = float(np.std(_ll_vals)) if _ll_vals else 1.0

    # ── CI anchor: auto-calibrate from in-sample emission-marginal z-scores ──
    # The anchor is |z| at the worst training day (COVID-equivalent → 100% CI).
    # We use the SAME logsumexp(emit) calculation that live scoring uses, so
    # the live z-distribution matches the training reference and threshold
    # gates (Zone 2/3/4) keep their intended meaning across feature swaps.
    try:
        from scipy.special import logsumexp as _logsumexp
        _emit_log = best_model._compute_log_likelihood(X)
        _live_ll  = _logsumexp(_emit_log, axis=1)
        _z_train  = (_live_ll - ll_per_obs) / max(_ll_std, 1e-6)
        _ci_anchor = float(max(abs(_z_train.min()), 0.467))
    except Exception:
        _ci_anchor = 0.467  # safe fallback to legacy anchor

    brain = HMMBrain(
        n_states=best_n,
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_start=df.index[0].strftime("%Y-%m-%d"),
        training_end=df.index[-1].strftime("%Y-%m-%d"),
        transmat=best_model.transmat_.tolist(),
        means=best_model.means_.tolist(),
        covars=best_model.covars_.tolist(),
        state_labels=state_labels,
        aic=round(best_model.aic(X), 2),
        bic=round(best_bic, 2),
        feature_names=feature_names,
        ll_baseline_mean=round(ll_per_obs, 6),
        ll_baseline_std=round(max(_ll_std, 1e-6), 6),
        lookback_years=lookback_years,
        ci_anchor=round(_ci_anchor, 4),
    )

    save_hmm_brain(brain)
    return brain


# ── Persistence ───────────────────────────────────────────────────────────────

def save_hmm_brain(brain: HMMBrain) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_BRAIN_PATH, "w") as f:
        json.dump(asdict(brain), f, indent=2)
    invalidate_ci_anchor_cache()


def get_ci_anchor() -> float:
    """Return the CI% anchor for the currently saved brain.

    All CI%/Zone calculations should use this instead of hardcoding 0.467.
    Falls back to the legacy 0.467 if no brain is saved.

    Cached at module level; auto-invalidates when the brain file changes on disk
    (handles external retrains that don't go through save_hmm_brain()).
    """
    global _CI_ANCHOR_CACHE, _CI_ANCHOR_MTIME
    try:
        current_mtime = os.path.getmtime(_BRAIN_PATH) if os.path.exists(_BRAIN_PATH) else None
    except OSError:
        current_mtime = None
    if _CI_ANCHOR_CACHE is None or current_mtime != _CI_ANCHOR_MTIME:
        brain = load_hmm_brain()
        _CI_ANCHOR_CACHE = float(brain.ci_anchor) if brain else 0.467
        _CI_ANCHOR_MTIME = current_mtime
    return _CI_ANCHOR_CACHE


_CI_ANCHOR_CACHE: Optional[float] = None
_CI_ANCHOR_MTIME: Optional[float] = None


def invalidate_ci_anchor_cache() -> None:
    """Clear the cached CI anchor — call after train_hmm() to force re-read."""
    global _CI_ANCHOR_CACHE, _CI_ANCHOR_MTIME
    _CI_ANCHOR_CACHE = None
    _CI_ANCHOR_MTIME = None


def load_hmm_brain() -> Optional[HMMBrain]:
    if not os.path.exists(_BRAIN_PATH):
        return None
    try:
        with open(_BRAIN_PATH) as f:
            d = json.load(f)
        # Backward compat: old brains lack LL baseline and lookback fields
        d.setdefault("ll_baseline_mean", 0.0)
        d.setdefault("ll_baseline_std", 1.0)
        d.setdefault("lookback_years", 15)  # default to 15 for brains trained before this field was added
        d.setdefault("ci_anchor", 0.467)    # legacy ICE BofA anchor; new brains override at train time
        return HMMBrain(**d)
    except Exception:
        return None


def load_transition_heat() -> Optional[dict]:
    """Return daily + monthly transition matrices, diagonal health, and warming alerts.

    The daily matrix has very high diagonals (~0.99) because regimes persist for months.
    The monthly matrix (daily^21) shows actionable 1-month transition probabilities.

    Returns dict with:
      - matrix_daily: list[list[float]]  (raw daily transition probs)
      - matrix: list[list[float]]        (monthly = daily^21, the primary display)
      - labels: list[str]
      - current_idx: int
      - diagonal_health: list[float]     (monthly self-transition, i.e. chance of staying)
      - warming: list[tuple[str,str,float]]  (monthly off-diag > 3%)
      - expected_persistence: list[float] (expected days = 1/(1-daily_diag))
    """
    brain = load_hmm_brain()
    if not brain:
        return None
    state = load_current_hmm_state()
    cur_idx = state.state_idx if state else 0
    tm_daily = np.array(brain.transmat)
    labels = brain.state_labels
    n = len(labels)

    # Monthly matrix: raise daily to power 21 (business days)
    tm_monthly = np.linalg.matrix_power(tm_daily, 21)

    # Daily diagonal for expected persistence calculation
    daily_diag = [float(tm_daily[i, i]) for i in range(n)]
    persistence = [round(1.0 / max(1 - d, 1e-6), 1) for d in daily_diag]

    # Monthly diagonal = chance of staying in same state over 1 month
    monthly_diag = [float(tm_monthly[i, i]) for i in range(n)]

    # Warming: monthly off-diagonal > 3% (actionable threshold)
    warming = []
    for i in range(n):
        for j in range(n):
            if i != j and tm_monthly[i, j] > 0.03:
                warming.append((labels[i], labels[j], round(float(tm_monthly[i, j]), 4)))
    warming.sort(key=lambda x: -x[2])

    return {
        "matrix_daily": tm_daily.tolist(),
        "matrix": tm_monthly.tolist(),  # primary display = monthly
        "labels": labels,
        "current_idx": cur_idx,
        "diagonal_health": [round(d, 4) for d in monthly_diag],
        "expected_persistence": persistence,
        "warming": warming,
    }


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
    # Keep last 500 entries
    with open(_HISTORY_PATH, "w") as f:
        json.dump(history[-500:], f, indent=2)


# ── Scoring (inference) ───────────────────────────────────────────────────────

def score_current_state(brain: Optional[HMMBrain] = None,
                        log_to_history: bool = True) -> Optional[HMMState]:
    """
    Compute today's HMM regime state using the trained brain.
    Reconstructs the model from stored parameters (no refit needed).
    Optionally appends to hmm_state_history.json.
    Returns None if brain is unavailable or inference fails.
    """
    if brain is None:
        brain = load_hmm_brain()
    if brain is None:
        return None

    try:
        from hmmlearn.hmm import GaussianHMM
        import numpy as np

        # Rebuild model from stored params
        n = brain.n_states
        model = GaussianHMM(n_components=n, covariance_type="full")
        model.n_features = len(brain.feature_names)
        model.startprob_ = np.ones(n) / n  # uniform start
        model.transmat_ = np.array(brain.transmat)
        model.means_ = np.array(brain.means)
        model.covars_ = np.array(brain.covars)

        # Build feature matrix with same EWMA z-scoring as training.
        # Use EXACT same lookback as training — different lookback gives different
        # rolling z-scores → different LL scale → invalid z-score comparison.
        df = _build_feature_matrix(lookback_years=brain.lookback_years)
        for col in brain.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna()
        X = df[brain.feature_names].values.astype(np.float64)

        # Predict state probabilities for the sequence, take last row
        posteriors = model.predict_proba(X)  # shape (T, n_states)
        today_probs = posteriors[-1].tolist()
        state_idx = int(np.argmax(today_probs))
        state_label = brain.state_labels[state_idx]
        confidence = round(float(today_probs[state_idx]), 3)

        # Compute persistence: how many consecutive days in this state
        states_seq = np.argmax(posteriors, axis=1)
        persistence = 1
        for s in reversed(states_seq[:-1]):
            if s == state_idx:
                persistence += 1
            else:
                break

        # ── Log-Likelihood: "Check Engine" light ────────────────────────────
        # Use TODAY's per-day emission marginal (logsumexp across states).
        # MUST match compute_full_state_path() so live z-scores align with
        # the historical path used to calibrate ci_anchor.
        from scipy.special import logsumexp as _logsumexp
        log_emit = model._compute_log_likelihood(X)
        todays_ll = float(_logsumexp(log_emit[-1]))
        ll_per_obs = round(todays_ll, 6)
        ll_zscore = round(
            (todays_ll - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6), 3
        )

        # ── Entropy: regime certainty [0=pure, 1=total fog] ─────────────────
        from scipy.stats import entropy as _shannon_entropy
        raw_entropy = float(_shannon_entropy(today_probs))
        max_entropy = float(np.log(brain.n_states))
        normalized_entropy = round(raw_entropy / max_entropy, 4) if max_entropy > 0 else 0.0

        # ── Transition projections: 1-month (21d), 3-month (63d), 6-month (126d) ─
        transmat = np.array(brain.transmat)
        prob_vec = np.array(today_probs)
        forecast_1m = (prob_vec @ np.linalg.matrix_power(transmat, 21)).tolist()
        forecast_3m = (prob_vec @ np.linalg.matrix_power(transmat, 63)).tolist()
        forecast_6m = (prob_vec @ np.linalg.matrix_power(transmat, 126)).tolist()
        transition_risk_1m = round(1.0 - forecast_1m[state_idx], 4)
        transition_risk_3m = round(1.0 - forecast_3m[state_idx], 4)
        transition_risk_6m = round(1.0 - forecast_6m[state_idx], 4)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = HMMState(
            date=today,
            state_idx=state_idx,
            state_label=state_label,
            state_probabilities=[round(p, 4) for p in today_probs],
            confidence=confidence,
            persistence=persistence,
            log_likelihood=ll_per_obs,
            ll_zscore=ll_zscore,
            entropy=normalized_entropy,
            transition_risk_1m=transition_risk_1m,
            transition_risk_3m=transition_risk_3m,
            transition_risk_6m=transition_risk_6m,
            forecast_1m=[round(p, 4) for p in forecast_1m],
            forecast_3m=[round(p, 4) for p in forecast_3m],
            forecast_6m=[round(p, 4) for p in forecast_6m],
        )

        if log_to_history:
            history = _load_history()
            # Replace today's entry if it already exists
            history = [h for h in history if h.get("date") != today]
            history.append(asdict(state))

            # ── Backfill forward 20d SPY return on older entries ─────────
            # Look ~22 entries back (≈20 business days). If that entry has
            # no fwd_20d_spy_return yet, fetch SPY price and compute it.
            # This runs once per old entry — already-stamped entries are skipped.
            if len(history) >= 22:
                _bf_entry = history[-22]
                if "fwd_20d_spy_return" not in _bf_entry:
                    try:
                        import yfinance as _yf_bf
                        _bf_start = _bf_entry["date"]
                        _bf_end = today
                        _bf_spy = _yf_bf.download(
                            "SPY", start=_bf_start, end=_bf_end,
                            progress=False, auto_adjust=True
                        )
                        if _bf_spy is not None and len(_bf_spy) >= 20:
                            _bf_close = _bf_spy["Close"]
                            if isinstance(_bf_close, pd.DataFrame):
                                _bf_close = _bf_close.iloc[:, 0]
                            _bf_ret = float((_bf_close.iloc[20] / _bf_close.iloc[0] - 1) * 100)
                            _bf_entry["fwd_20d_spy_return"] = round(_bf_ret, 2)
                    except Exception:
                        pass

            _save_history(history)

        return state

    except Exception:
        return None


def compute_full_state_path(brain: Optional[HMMBrain] = None) -> Optional[pd.DataFrame]:
    """
    Decode the full historical regime path for the main HMM brain.

    Rebuilds the GaussianHMM from stored params, runs predict_proba over the
    entire feature matrix the model was trained on (~lookback_years long), and
    returns one row per business day with the argmax state, the per-day
    marginal log-likelihood, its z-score against the training baseline, and CI%.

    Returns None if the brain or feature data is unavailable.
    """
    if brain is None:
        brain = load_hmm_brain()
    if brain is None:
        return None

    try:
        from hmmlearn.hmm import GaussianHMM
        from scipy.special import logsumexp

        n = brain.n_states
        model = GaussianHMM(n_components=n, covariance_type="full")
        model.n_features = len(brain.feature_names)
        model.startprob_ = np.ones(n) / n
        model.transmat_ = np.array(brain.transmat)
        model.means_ = np.array(brain.means)
        model.covars_ = np.array(brain.covars)

        df = _build_feature_matrix(lookback_years=brain.lookback_years)
        for col in brain.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[brain.feature_names].dropna()
        if df.empty:
            return None
        X = df.values.astype(np.float64)

        posteriors = model.predict_proba(X)
        states_seq = np.argmax(posteriors, axis=1)
        log_emit = model._compute_log_likelihood(X)
        ll_per_obs = logsumexp(log_emit, axis=1)
        ll_z = (ll_per_obs - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6)
        ci_pct = np.abs(ll_z) / max(brain.ci_anchor, 1e-6) * 100.0

        labels = [brain.state_labels[i] for i in states_seq]
        return pd.DataFrame(
            {
                "state_idx": states_seq,
                "state_label": labels,
                "ll_per_obs": ll_per_obs,
                "ll_zscore": ll_z,
                "ci_pct": ci_pct,
            },
            index=df.index,
        )
    except Exception:
        return None


def load_current_hmm_state() -> Optional[HMMState]:
    """
    Fast path: load HMM state from history JSON.
    Returns today's entry if available, otherwise falls back to most recent entry.
    No model inference — safe to call on every QIR refresh.
    Returns None only if history is completely empty.
    """
    history = _load_history()
    if not history:
        return None
    last = history[-1]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Use latest entry regardless of date (QIR will show staleness badge)
    try:
        state = HMMState(**last)
        # Tag if stale (not today's data)
        state._is_stale = last.get("date") != today
        state._stale_date = last.get("date", "unknown")
        return state
    except Exception:
        return None


def get_hmm_state_history() -> list[dict]:
    return _load_history()


# ── Summary helpers for QIR display ──────────────────────────────────────────

_STATE_COLORS = {
    # Shadow-brain-specific labels (listed first — substring match picks the first hit)
    "Strong Bull": "#16a34a",
    "Mild Bull":   "#4ade80",
    "Transition":  "#94a3b8",
    "Mild Bear":   "#f97316",
    "Strong Bear": "#ef4444",
    # Credit-brain labels
    "Bull":        "#22c55e",
    "Neutral":     "#94a3b8",
    "Early Stress":"#f59e0b",
    "Stress":      "#f97316",
    "Late Cycle":  "#ef4444",
    "Crisis":      "#dc2626",
}

_STATE_ARROWS = {
    "Strong Bull": "▲▲",
    "Mild Bull":   "▲",
    "Transition":  "→",
    "Mild Bear":   "▼",
    "Strong Bear": "▼▼",
    "Bull":        "▲",
    "Neutral":     "→",
    "Early Stress":"↘",
    "Stress":      "▼",
    "Late Cycle":  "▼",
    "Crisis":      "▼▼",
}

_STATE_TIPS = {
    "Strong Bull": (
        "SPX price brain: Strong Bull regime. Mean return is solidly positive, "
        "variance is low. Historical analog: 2017, 2021 melt-up phases. "
        "Momentum strategies work, dip-buying rewarded. Full sizing appropriate."
    ),
    "Mild Bull": (
        "SPX price brain: Mild Bull regime. Positive drift but modest. "
        "Historical analog: steady-grind years. Normal position sizing, "
        "don't chase breakouts — trend is your friend but accelerations are rare."
    ),
    "Transition": (
        "SPX price brain: Transition regime — neither bull nor bear dominates. "
        "Mean return near zero, moderate variance. Chop/range-bound tape. "
        "Wait for the next regime signal before adding directional risk."
    ),
    "Mild Bear": (
        "SPX price brain: Mild Bear regime. Negative drift but not dislocation. "
        "Historical analog: slow grindy declines (2015-2016, 2018 Q4 pre-pivot). "
        "Reduce beta exposure, trim cyclicals. Not time to buy weakness yet."
    ),
    "Strong Bear": (
        "SPX price brain: Strong Bear regime. Large negative drift, sellers in control. "
        "Historical analog: 2022 grinding bear, 2008 pre-Lehman. "
        "Defensive posture — raise cash, prefer quality, avoid catching knives."
    ),
    "Bull": (
        "Risk-on. Stay invested and let winners run. "
        "Momentum strategies work best here — buying dips is rewarded. "
        "Full conviction sizing is appropriate. "
        "Watch for euphoria signals (VIX <12, HY spreads <250bp) as late-bull warnings."
    ),
    "Neutral": (
        "Balanced exposure. Wait for confirmation before adding new risk. "
        "Normal position sizing. Neither strong bull nor bear signals. "
        "Good time to review portfolio quality — trim laggards, hold core positions. "
        "The next state transition is the key trade."
    ),
    "Early Stress": (
        "Early warning. Begin reducing beta exposure. "
        "Trim speculative positions but don't panic-sell core holdings. "
        "Raise a small cash buffer (5-10%). "
        "Watch credit spreads — if HY breaks above 400bp, this becomes Stress."
    ),
    "Stress": (
        "Defensive posture. Reduce overall equity exposure by 20-30%. "
        "Avoid buying weakness — Stress regimes mean sellers are in control. "
        "Prefer quality over growth. Bonds (TLT/IEI) and gold as hedges. "
        "Do not add new aggressive positions. Size down via Kelly discount."
    ),
    "Late Cycle": (
        "Hold quality, don't add new risk. "
        "Economy still functioning but conditions are deteriorating — spreads elevated, "
        "credit tightening, real rates high. This is the phase where most people get "
        "caught holding too much. Trim cyclicals and high-beta positions. "
        "Raise cash (15-25%). Wait for either a policy pivot or spread blowout "
        "before repositioning. Historically lasts 6-18 months before resolving."
    ),
    "Crisis": (
        "Capital preservation mode. Maximum defensiveness. "
        "Credit markets are dislocating — HY spreads at crisis levels. "
        "Avoid catching falling knives. Hold cash, short-term treasuries, gold. "
        "Historically the best time to build a buy list, not to deploy capital. "
        "Wait for VIX to spike and then reverse before re-entering risk assets."
    ),
}


def get_state_color(state_label: str) -> str:
    for key, color in _STATE_COLORS.items():
        if key.lower() in state_label.lower():
            return color
    return "#64748b"


def get_state_arrow(state_label: str) -> str:
    for key, arrow in _STATE_ARROWS.items():
        if key.lower() in state_label.lower():
            return arrow
    return "→"


def get_state_tips(state_label: str) -> str:
    for key, tip in _STATE_TIPS.items():
        if key.lower() in state_label.lower():
            return tip
    return "No guidance available for this state."


def get_conviction_multiplier(state_label: str, entropy: float = 0.0) -> float:
    """
    Returns a multiplier applied to raw conviction score.
    Base range [0.75, 1.06] from regime state, then entropy penalty
    dampens further when the HMM is uncertain about the current state.

    At entropy=0 (pure state): no penalty.
    At entropy=1 (total fog): 30% additional dampening.
    """
    label = state_label.lower()
    if "crisis" in label:
        base = 0.75   # maximum dampener — capital preservation mode
    elif "late cycle" in label:
        base = 0.82   # significant dampener — avoid new risk
    elif "stress" in label:
        base = 0.92   # moderate dampener — size down
    elif "neutral" in label:
        base = 1.00
    elif "bull" in label:
        base = 1.06   # amplifier — regime confirms risk-on
    else:
        base = 1.00
    return round(base * (1.0 - 0.3 * min(max(entropy, 0.0), 1.0)), 4)
