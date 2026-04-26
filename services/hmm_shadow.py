"""
Shadow HMM Regime Detection Service (SPX + VIX brain)
------------------------------------------------------
Second, independent brain trained on ^GSPC daily log returns + VIX level
from 1990 -> present. Uses hmmlearn GaussianHMM with n_components=6 and
covariance_type="full". Complements the credit/yield brain in hmm_regime.py.

Features:
  - gspc_logret_pct : SPX daily log return (%)
  - vix_level       : CBOE VIX closing level

Both features are z-scored before training (same pipeline as main brain).
VIX limits the training window to 1990 (vs 1960 for the old SPX-only brain).

Persistence:
  - data/hmm_shadow_brain.json         -- brain params + CI anchor + crash bins
  - data/hmm_shadow_history.json       -- daily state log (last 500)
  - data/hmm_shadow_ci_calibration.json -- walk-through backtest output

Regime labels: Strong Bear, Mild Bear, Transition, Mild Bull, Strong Bull, Crisis.
Crisis = highest SPX-return variance regime. The other five are ordered by
conditional mean SPX return.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_BRAIN_PATH = os.path.join(_DATA_DIR, "hmm_shadow_brain.json")
_HISTORY_PATH = os.path.join(_DATA_DIR, "hmm_shadow_history.json")
_CALIBRATION_PATH = os.path.join(_DATA_DIR, "hmm_shadow_ci_calibration.json")

_TICKER = "^GSPC"
_VIX_TICKER = "^VIX"
_DEFAULT_START = "1990-01-01"   # VIX history begins ~1990; BAA10Y from 1986
_N_STATES = 6
_FEATURE_NAMES = ["gspc_logret_pct", "vix_level"]
_DEFAULT_CI_ANCHOR = 1.194      # updated after calibration


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HMMShadowBrain:
    n_states: int
    feature_names: list
    trained_at: str
    training_start: str
    training_end: str
    means: list             # (n_states, n_features) GaussianHMM means
    covars: list            # (n_states, n_features, n_features) covariances
    transmat: list          # n_states x n_states
    state_labels: list
    ll_baseline_mean: float
    ll_baseline_std: float
    feature_means: list     # per-feature training mean (for z-scoring new data)
    feature_stds: list      # per-feature training std
    ci_anchor: float = _DEFAULT_CI_ANCHOR
    crash_prob_bins: list = field(default_factory=list)
    regime_means: list = field(default_factory=list)    # SPX return dim, for display
    regime_variances: list = field(default_factory=list)
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


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_shadow_features(start: str = _DEFAULT_START,
                           feat_means: Optional[list] = None,
                           feat_stds: Optional[list] = None) -> tuple[pd.DataFrame, list, list]:
    """Download SPX + VIX, align, z-score, return (df, means, stds).

    Features:
      gspc_logret_pct : SPX daily log return (%)
      vix_level       : CBOE VIX close

    If feat_means/feat_stds are provided (from saved brain), apply them
    instead of computing from the data — ensures live scoring uses the
    same scale as training.
    """
    import yfinance as yf

    spx = yf.download(_TICKER, start=start, progress=False, auto_adjust=True)["Close"]
    vix = yf.download(_VIX_TICKER, start=start, progress=False, auto_adjust=True)["Close"]

    if isinstance(spx, pd.DataFrame):
        spx = spx.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]

    if spx.index.tz is not None:
        spx.index = spx.index.tz_localize(None)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)

    logret = (np.log(spx / spx.shift(1)) * 100.0).rename("gspc_logret_pct")
    vix_s = vix.rename("vix_level")

    df = pd.concat([logret, vix_s], axis=1).dropna()

    # Z-score using training stats (or compute if not provided)
    if feat_means is None or feat_stds is None:
        out_means = df.mean().tolist()
        out_stds = df.std().tolist()
    else:
        out_means = feat_means
        out_stds = feat_stds

    for i, col in enumerate(df.columns):
        df[col] = (df[col] - out_means[i]) / max(out_stds[i], 1e-6)

    return df, out_means, out_stds


# ── Regime labelling ──────────────────────────────────────────────────────────

def _label_regimes(means: np.ndarray, covars: np.ndarray) -> list[str]:
    """Label 6 regimes using SPX-return dimension (index 0).

    Crisis = highest SPX-return variance. Remaining 5 ordered by SPX mean return.
    """
    n = means.shape[0]
    spx_means = means[:, 0]
    spx_vars = covars[:, 0, 0]

    labels = [""] * n
    crisis_idx = int(np.argmax(spx_vars))
    labels[crisis_idx] = "Crisis"

    remaining = [i for i in range(n) if i != crisis_idx]
    remaining_sorted = sorted(remaining, key=lambda i: spx_means[i])
    ordered_labels = ["Strong Bear", "Mild Bear", "Transition", "Mild Bull", "Strong Bull"]
    for lbl, idx in zip(ordered_labels, remaining_sorted):
        labels[idx] = lbl
    return labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_shadow_hmm(start: str = _DEFAULT_START,
                     n_states: int = _N_STATES,
                     random_state: int = 42) -> HMMShadowBrain:
    """Fit a GaussianHMM on SPX log returns + VIX. Persists brain.json.

    After training run tools/backtest_shadow_ci.py to calibrate ci_anchor
    and crash_prob_bins.
    """
    from hmmlearn.hmm import GaussianHMM
    from scipy.special import logsumexp

    print(f"[shadow] building feature matrix from {start} ...")
    df, feat_means, feat_stds = _build_shadow_features(start)
    X = df.values.astype(np.float64)
    print(f"[shadow] {len(X)} obs, features: {list(df.columns)}")

    best_brain = None
    best_ll = -np.inf

    # Train multiple restarts, keep best log-likelihood
    for seed in [random_state, random_state + 1, random_state + 7]:
        np.random.seed(seed)
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=seed,
        )
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                model.fit(X)
            total_ll = model.score(X)
            if total_ll > best_ll:
                best_ll = total_ll
                best_model = model
        except Exception:
            continue

    model = best_model

    # Per-day emission marginals for LL baseline
    from scipy.special import logsumexp
    log_emit = model._compute_log_likelihood(X)
    ll_per_obs = logsumexp(log_emit, axis=1)
    ll_mean = float(np.mean(ll_per_obs))
    ll_std = float(np.std(ll_per_obs))
    if not np.isfinite(ll_std) or ll_std <= 0:
        ll_std = 1.0

    ll_z_train = (ll_per_obs - ll_mean) / max(ll_std, 1e-6)
    ci_anchor = float(max(abs(ll_z_train.min()), _DEFAULT_CI_ANCHOR))

    labels = _label_regimes(model.means_, model.covars_)

    # SPX-return dimension stats for display
    regime_means = [round(float(model.means_[i, 0] * feat_stds[0] + feat_means[0]), 4)
                    for i in range(n_states)]
    regime_variances = [round(float(model.covars_[i, 0, 0] * feat_stds[0] ** 2), 6)
                        for i in range(n_states)]

    brain = HMMShadowBrain(
        n_states=n_states,
        feature_names=list(df.columns),
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_start=df.index[0].strftime("%Y-%m-%d"),
        training_end=df.index[-1].strftime("%Y-%m-%d"),
        means=model.means_.tolist(),
        covars=model.covars_.tolist(),
        transmat=model.transmat_.tolist(),
        state_labels=labels,
        ll_baseline_mean=round(ll_mean, 6),
        ll_baseline_std=round(ll_std, 6),
        feature_means=feat_means,
        feature_stds=feat_stds,
        ci_anchor=round(ci_anchor, 4),
        crash_prob_bins=[],
        regime_means=regime_means,
        regime_variances=regime_variances,
        n_obs=len(X),
    )

    os.makedirs(_DATA_DIR, exist_ok=True)
    save_shadow_brain(brain)
    print(f"[shadow] trained. ci_anchor={ci_anchor:.4f}  ll_mean={ll_mean:.4f}  ll_std={ll_std:.4f}")
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
        # Backward compat: old brains used k_regimes, lacked GaussianHMM fields
        if "k_regimes" in d and "n_states" not in d:
            d["n_states"] = d.pop("k_regimes")
        d.setdefault("n_states", _N_STATES)
        d.setdefault("feature_names", ["gspc_logret_pct"])
        d.setdefault("means", [])
        d.setdefault("covars", [])
        d.setdefault("feature_means", [0.0])
        d.setdefault("feature_stds", [1.0])
        d.setdefault("ci_anchor", _DEFAULT_CI_ANCHOR)
        d.setdefault("crash_prob_bins", [])
        d.setdefault("regime_means", [])
        d.setdefault("regime_variances", [])
        d.setdefault("n_obs", 0)
        # Remove old-only fields not in new dataclass
        d.pop("bic", None)
        d.pop("k_regimes", None)
        return HMMShadowBrain(**d)
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


# ── Crash-prob lookup ─────────────────────────────────────────────────────────

def _lookup_crash_prob(ll_z: float, bins: list[dict]) -> tuple[float, float]:
    if not bins:
        return 0.0, 0.0
    for b in bins:
        z_lo = b.get("z_lo", -np.inf)
        z_hi = b.get("z_hi", np.inf)
        if z_lo <= ll_z < z_hi:
            return float(b.get("prob_10pct", 0.0)), float(b.get("expected_drawdown_pct", 0.0))
    if ll_z < bins[0].get("z_lo", 0):
        b = bins[0]
    else:
        b = bins[-1]
    return float(b.get("prob_10pct", 0.0)), float(b.get("expected_drawdown_pct", 0.0))


def _reconstruct_model(brain: HMMShadowBrain):
    """Rebuild a GaussianHMM from stored brain params."""
    from hmmlearn.hmm import GaussianHMM
    n = brain.n_states
    model = GaussianHMM(n_components=n, covariance_type="full")
    model.n_features = len(brain.feature_names)
    model.startprob_ = np.ones(n) / n
    model.transmat_ = np.array(brain.transmat)
    model.means_ = np.array(brain.means)
    model.covars_ = np.array(brain.covars)
    return model


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_current_shadow_state(log_to_history: bool = True) -> Optional[HMMShadowState]:
    """Compute today's Shadow HMM regime using stored brain params (no pickle needed)."""
    from scipy.special import logsumexp

    brain = load_shadow_brain()
    if brain is None or not brain.means:
        return None

    try:
        df, _, _ = _build_shadow_features(
            brain.training_start,
            feat_means=brain.feature_means,
            feat_stds=brain.feature_stds,
        )
        X = df.values.astype(np.float64)

        model = _reconstruct_model(brain)

        posteriors = model.predict_proba(X)
        today_probs = posteriors[-1].tolist()
        state_idx = int(np.argmax(today_probs))
        state_label = brain.state_labels[state_idx]
        confidence = round(float(today_probs[state_idx]), 4)

        states_seq = np.argmax(posteriors, axis=1)
        persistence = 1
        for s in reversed(states_seq[:-1]):
            if s == state_idx:
                persistence += 1
            else:
                break

        log_emit = model._compute_log_likelihood(X)
        ll_per_obs = logsumexp(log_emit, axis=1)
        ll_today = float(ll_per_obs[-1])
        ll_z = round((ll_today - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6), 4)
        ci_pct = round(abs(ll_z) / max(brain.ci_anchor, 1e-6) * 100.0, 2)

        crash_prob, exp_dd = _lookup_crash_prob(ll_z, brain.crash_prob_bins)

        from scipy.stats import entropy as _shannon_entropy
        raw_entropy = float(_shannon_entropy(today_probs))
        max_entropy = float(np.log(brain.n_states))
        norm_entropy = round(raw_entropy / max_entropy, 4) if max_entropy > 0 else 0.0

        tm = np.array(brain.transmat)
        pv = np.array(today_probs)
        fc_1m = (pv @ np.linalg.matrix_power(tm, 21)).tolist()
        fc_3m = (pv @ np.linalg.matrix_power(tm, 63)).tolist()
        fc_6m = (pv @ np.linalg.matrix_power(tm, 126)).tolist()

        # Unscaled SPX return for display
        raw_logret = float(df["gspc_logret_pct"].iloc[-1] * brain.feature_stds[0] + brain.feature_means[0])

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = HMMShadowState(
            date=today,
            state_idx=state_idx,
            state_label=state_label,
            state_probabilities=[round(p, 4) for p in today_probs],
            confidence=confidence,
            persistence=persistence,
            daily_return_pct=round(raw_logret, 4),
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


def compute_full_shadow_state_path(brain: Optional[HMMShadowBrain] = None) -> Optional[pd.DataFrame]:
    """Decode the full historical regime path for the Shadow HMM brain."""
    from scipy.special import logsumexp

    if brain is None:
        brain = load_shadow_brain()
    if brain is None or not brain.means:
        return None

    try:
        df, _, _ = _build_shadow_features(
            brain.training_start,
            feat_means=brain.feature_means,
            feat_stds=brain.feature_stds,
        )
        X = df.values.astype(np.float64)

        model = _reconstruct_model(brain)

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


def load_current_shadow_state() -> Optional[HMMShadowState]:
    """Fast path: load most recent Shadow state from history JSON."""
    history = _load_history()
    if not history:
        return None
    last = history[-1]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        # Strip fields not in dataclass (old entries may have extras)
        valid = {f.name for f in HMMShadowState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in last.items() if k in valid}
        state = HMMShadowState(**filtered)
        state._is_stale = last.get("date") != today
        state._stale_date = last.get("date", "unknown")
        return state
    except Exception:
        return None


def get_shadow_state_history() -> list[dict]:
    return _load_history()
