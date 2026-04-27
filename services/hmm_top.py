"""
Top Brain HMM Service (VIX + NFCI + BAA10Y + T10Y3M)
------------------------------------------------------
A 4-feature GaussianHMM trained on FRED macro features optimised for early
market TOP detection. Distinct from the Main Brain (10 features, regime engine)
and Shadow Brain (SPX log returns, bottom detection).

Signal logic:
  sig_and = regime ∈ {Late Cycle, Early Stress, Stress, Crisis}
            AND 40-day rolling mean(ll_z) < -0.20

Backtest (validated 2012-2026, 8 peaks):
  F1 = 0.556 · hits 5/8 · 4 false alarms · avg lead 107 days

Persistence:
  data/hmm_top_brain.json   -- brain params
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR   = os.path.join(_BASE_DIR, "data")
_BRAIN_PATH = os.path.join(_DATA_DIR, "hmm_top_brain.json")

# ── Brain constants ────────────────────────────────────────────────────────────
_FEATURES       = ["VIX", "NFCI", "BAA10Y", "T10Y3M"]
_N_STATES_MIN   = 2
_N_STATES_MAX   = 6
_N_RESTARTS     = 3
_DEFAULT_CI_ANCHOR = 4.754

# ── Signal constants ──────────────────────────────────────────────────────────
_LL_ROLL_WINDOW = 40      # days for rolling LL mean
_LL_ROLL_THRESH = -0.20   # 40-day mean must be below this to fire sig_ll
_LATE_LABELS    = {"Late Cycle", "Early Stress", "Stress", "Crisis"}

# ── Hardcoded backtest stats (update after retrain) ───────────────────────────
_BT_HITS       = 5
_BT_PEAKS      = 8
_BT_HIT_PCT    = 62
_BT_FA         = 4
_BT_AVG_LEAD   = 107  # days


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class HMMTopBrain:
    n_states: int
    feature_names: list
    trained_at: str
    training_start: str
    training_end: str
    means: list           # (n_states, n_features) GaussianHMM means
    covars: list          # (n_states, n_features, n_features)
    transmat: list        # n_states × n_states
    state_labels: list
    ll_baseline_mean: float
    ll_baseline_std: float
    feature_means: list   # per-feature z-score mean from training
    feature_stds: list    # per-feature z-score std from training
    ci_anchor: float = _DEFAULT_CI_ANCHOR
    n_obs: int = 0


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_top_features(feat_means: Optional[list] = None,
                        feat_stds:  Optional[list] = None,
                        lookback_years: int = 15
                        ) -> tuple[pd.DataFrame, list, list]:
    """Build the 4-feature FRED matrix used by the Top Brain.

    Uses the same pipeline as the Main Brain (_build_feature_matrix) but slices
    to _FEATURES only.  feat_means/feat_stds from the saved brain are applied
    during live scoring so the z-scores stay on the training scale.
    """
    from services.hmm_regime import _build_feature_matrix

    df_full = _build_feature_matrix(lookback_years=lookback_years)
    # VIX is named VIX in the main matrix; others match _FEATURES exactly
    missing = [f for f in _FEATURES if f not in df_full.columns]
    if missing:
        raise ValueError(f"Top Brain features missing from matrix: {missing}")

    df = df_full[_FEATURES].dropna().copy()

    if feat_means is None or feat_stds is None:
        out_means = df.mean().tolist()
        out_stds  = df.std().tolist()
    else:
        out_means = feat_means
        out_stds  = feat_stds

    for i, col in enumerate(df.columns):
        df[col] = (df[col] - out_means[i]) / max(out_stds[i], 1e-6)

    return df, out_means, out_stds


# ── State labelling ───────────────────────────────────────────────────────────

def _label_states(means: np.ndarray, feature_names: list) -> list[str]:
    """Label states by BAA10Y mean (index of that feature in _FEATURES).

    Lowest BAA10Y spread → Bull.  Highest → Crisis.  Others interpolated.
    Falls back to generic labels if BAA10Y not found.
    """
    n = means.shape[0]
    try:
        idx = feature_names.index("BAA10Y")
        baa_means = means[:, idx]
        order = np.argsort(baa_means)   # ascending: tightest spread first
    except ValueError:
        order = np.arange(n)

    pool = ["Bull", "Neutral", "Late Cycle", "Early Stress", "Stress", "Crisis"]
    # Pick evenly spaced labels from pool based on n_states
    step = max(1, (len(pool) - 1) / max(n - 1, 1))
    labels = [""] * n
    for rank, state_idx in enumerate(order):
        pool_idx = min(round(rank * step), len(pool) - 1)
        labels[state_idx] = pool[pool_idx]
    return labels


# ── Model reconstruction ──────────────────────────────────────────────────────

def _reconstruct_model(brain: HMMTopBrain):
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=brain.n_states, covariance_type="full")
    m.n_features  = len(brain.feature_names)
    m.startprob_  = np.ones(brain.n_states) / brain.n_states
    m.transmat_   = np.array(brain.transmat)
    m.means_      = np.array(brain.means)
    m.covars_     = np.array(brain.covars)
    return m


# ── Training ──────────────────────────────────────────────────────────────────

def train_top_brain(lookback_years: int = 15) -> HMMTopBrain:
    """BIC-sweep GaussianHMM over _FEATURES, return a trained HMMTopBrain."""
    import warnings
    from hmmlearn.hmm import GaussianHMM
    from scipy.special import logsumexp

    print(f"[top_brain] building feature matrix (lookback={lookback_years}yr) ...")
    df, feat_means, feat_stds = _build_top_features(lookback_years=lookback_years)
    X = df.values.astype(np.float64)
    print(f"[top_brain] {len(X)} obs  features: {list(df.columns)}")

    best_model = None
    best_bic   = np.inf
    best_n     = _N_STATES_MIN

    for n in range(_N_STATES_MIN, _N_STATES_MAX + 1):
        for seed in [42, 43, 47][:_N_RESTARTS]:
            np.random.seed(seed)
            model = GaussianHMM(
                n_components=n, covariance_type="full",
                n_iter=200, tol=1e-4, random_state=seed,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X)
                n_params = n * n + n * len(_FEATURES) + n * len(_FEATURES) ** 2
                bic = -2 * model.score(X) * len(X) + n_params * np.log(len(X))
                if bic < best_bic:
                    best_bic   = bic
                    best_model = model
                    best_n     = n
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Top Brain training failed — all restarts diverged")

    print(f"[top_brain] best n_states={best_n}  BIC={best_bic:.1f}")

    from scipy.special import logsumexp
    log_emit   = best_model._compute_log_likelihood(X)
    ll_per_obs = logsumexp(log_emit, axis=1)
    ll_mean    = float(np.mean(ll_per_obs))
    ll_std     = float(np.std(ll_per_obs))
    if not np.isfinite(ll_std) or ll_std <= 0:
        ll_std = 1.0

    ll_z_train = (ll_per_obs - ll_mean) / max(ll_std, 1e-6)
    ci_anchor  = float(max(abs(ll_z_train.min()), 0.467))
    labels     = _label_states(best_model.means_, list(df.columns))
    print(f"[top_brain] ci_anchor={ci_anchor:.4f}  labels={labels}")

    brain = HMMTopBrain(
        n_states        = best_n,
        feature_names   = list(df.columns),
        trained_at      = datetime.now(timezone.utc).isoformat(),
        training_start  = df.index[0].strftime("%Y-%m-%d"),
        training_end    = df.index[-1].strftime("%Y-%m-%d"),
        means           = best_model.means_.tolist(),
        covars          = best_model.covars_.tolist(),
        transmat        = best_model.transmat_.tolist(),
        state_labels    = labels,
        ll_baseline_mean = round(ll_mean, 6),
        ll_baseline_std  = round(ll_std, 6),
        feature_means   = feat_means,
        feature_stds    = feat_stds,
        ci_anchor       = round(ci_anchor, 4),
        n_obs           = len(X),
    )
    return brain


# ── Persistence ───────────────────────────────────────────────────────────────

def save_top_brain(brain: HMMTopBrain) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_BRAIN_PATH, "w") as f:
        json.dump(asdict(brain), f, indent=2)
    print(f"[top_brain] saved to {_BRAIN_PATH}")


def load_top_brain() -> Optional[HMMTopBrain]:
    if not os.path.exists(_BRAIN_PATH):
        return None
    try:
        with open(_BRAIN_PATH) as f:
            d = json.load(f)
        d.setdefault("n_states", 4)
        d.setdefault("feature_names", _FEATURES)
        d.setdefault("means", [])
        d.setdefault("covars", [])
        d.setdefault("feature_means", [0.0] * len(_FEATURES))
        d.setdefault("feature_stds",  [1.0] * len(_FEATURES))
        d.setdefault("ci_anchor", _DEFAULT_CI_ANCHOR)
        d.setdefault("n_obs", 0)
        return HMMTopBrain(**d)
    except Exception:
        return None


# ── Full path ─────────────────────────────────────────────────────────────────

def compute_full_top_state_path(brain: Optional[HMMTopBrain] = None) -> Optional[pd.DataFrame]:
    """Decode the full historical regime path for the Top Brain.

    Returns DataFrame indexed by date with columns:
      state_label, ll_zscore, ci_pct, ll_z_roll, sig_regime, sig_ll, sig_and
    """
    from scipy.special import logsumexp

    if brain is None:
        brain = load_top_brain()
    if brain is None or not brain.means:
        return None

    try:
        df, _, _ = _build_top_features(
            feat_means     = brain.feature_means,
            feat_stds      = brain.feature_stds,
            lookback_years = 15,
        )
        X = df.values.astype(np.float64)

        model      = _reconstruct_model(brain)
        states_seq = model.predict(X)
        log_emit   = model._compute_log_likelihood(X)
        ll_per_obs = logsumexp(log_emit, axis=1)
        ll_z       = (ll_per_obs - brain.ll_baseline_mean) / max(brain.ll_baseline_std, 1e-6)
        ci_pct     = np.abs(ll_z) / max(brain.ci_anchor, 1e-6) * 100.0
        labels     = [brain.state_labels[i] for i in states_seq]

        path = pd.DataFrame({
            "state_label": labels,
            "ll_zscore":   ll_z,
            "ci_pct":      ci_pct,
        }, index=df.index)

        path["ll_z_roll"]   = path["ll_zscore"].rolling(_LL_ROLL_WINDOW, min_periods=10).mean()
        path["sig_regime"]  = path["state_label"].isin(_LATE_LABELS).astype(int)
        path["sig_ll"]      = (path["ll_z_roll"] < _LL_ROLL_THRESH).astype(int)
        path["sig_and"]     = ((path["sig_regime"] == 1) & (path["sig_ll"] == 1)).astype(int)

        return path
    except Exception:
        return None


# ── Today's signal ────────────────────────────────────────────────────────────

def compute_top_signal_today(brain: Optional[HMMTopBrain] = None) -> Optional[dict]:
    """Compute today's Top Brain signal.

    Returns dict with keys:
      ll_z, ll_z_roll, ci_pct, regime_label,
      sig_regime, sig_ll, sig_and,
      days_in_stress, roll_fill_pct, ci_anchor,
      training_end
    """
    if brain is None:
        brain = load_top_brain()

    path = compute_full_top_state_path(brain)
    if path is None or path.empty:
        return None

    row = path.iloc[-1]

    # Consecutive days sig_and has been ON (streak ending today)
    sig_col = path["sig_and"].values
    days_on = 0
    for v in reversed(sig_col):
        if v == 1:
            days_on += 1
        else:
            break

    # Progress bar fill: how far is roll toward the threshold?
    roll = float(row["ll_z_roll"]) if pd.notna(row["ll_z_roll"]) else 0.0
    # fill = roll / thresh * 100 — thresh is negative so flip sign
    roll_fill_pct = min(abs(roll) / abs(_LL_ROLL_THRESH) * 100.0, 150.0)
    if roll >= 0:
        roll_fill_pct = 0.0

    return {
        "ll_z":          float(row["ll_zscore"]),
        "ll_z_roll":     roll,
        "ci_pct":        float(row["ci_pct"]),
        "regime_label":  str(row["state_label"]),
        "sig_regime":    bool(row["sig_regime"]),
        "sig_ll":        bool(row["sig_ll"]),
        "sig_and":       bool(row["sig_and"]),
        "days_in_stress": days_on,
        "roll_fill_pct": roll_fill_pct,
        "ci_anchor":     brain.ci_anchor,
        "training_end":  brain.training_end,
    }


# ── Cached signal (keyed by trained_at so it refreshes after retrain) ────────

@lru_cache(maxsize=4)
def _cached_top_signal(trained_at: str) -> Optional[dict]:
    """LRU-cached wrapper — one FRED matrix build per brain version per process."""
    brain = load_top_brain()
    if brain is None or brain.trained_at != trained_at:
        return None
    return compute_top_signal_today(brain)


def get_top_signal_cached() -> Optional[dict]:
    """Call this from QIR / any hot path instead of compute_top_signal_today()."""
    brain = load_top_brain()
    if brain is None:
        return None
    return _cached_top_signal(brain.trained_at)
