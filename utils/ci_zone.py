"""
CI% (Crisis Intensity) zone classifier.

Single source of truth for the Zone 1/2/3/4 mapping documented in CLAUDE.md.
Replaces inline classification logic that was previously duplicated across
modules (originally at quick_run.py:3083-3090).

Formula: CI% = abs(ll_zscore) / brain.ci_anchor * 100   (uncapped)
The anchor is auto-calibrated at training time — get it via
services.hmm_regime.get_ci_anchor().
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CIZone:
    zone: int                # 1-4
    label: str               # human-readable name
    color: str               # hex
    crash_prob_pct: float    # historical 30-day crash probability
    z_threshold: str         # human-readable z-score range (anchor-relative)
    description: str         # one-line explanation


_ZONES = {
    1: CIZone(1, "Normal",            "#22c55e", 3.0,  "CI < 22%",
              "Conviction signals suppressed."),
    2: CIZone(2, "Model Stress",      "#f59e0b", 6.0,  "22% <= CI < 40%",
              "Early-warning watch zone. Signals shown as context."),
    3: CIZone(3, "Crisis Gate Open",  "#ef4444", 9.25, "40% <= CI <= 100%",
              "Operational gate. Backtest: 75% historical crash detection, 0% false alarms."),
    4: CIZone(4, "Beyond Training",   "#a855f7", 12.0, "CI > 100%",
              "Post-training extremes (in-sample worst day = 100% CI)."),
}


def classify_ci_zone(ci_pct: float) -> CIZone:
    """Map a CI% value to its Zone (1-4).

    Boundaries calibrated from the LL-gate backtest against historical SPX
    crashes (post-Apr-2026 retrain on Moody's BAA10Y/AAA10Y features):
        Zone 1: CI < 22%        — Normal (~92% of training days)
        Zone 2: 22% <= CI < 40% — Stress watch (~6% of days, 88% recall)
        Zone 3: 40% <= CI <= 100% — Crisis Gate (75% recall, 0% FP)
        Zone 4: CI > 100%       — Beyond training extremes
    """
    if ci_pct > 100.0:
        return _ZONES[4]
    if ci_pct >= 40.0:    # was 67.0 (legacy ICE-BofA brain) — recalibrated 2026-04
        return _ZONES[3]
    if ci_pct >= 22.0:
        return _ZONES[2]
    return _ZONES[1]


def classify_from_ll_zscore(ll_zscore: float, anchor: float = 0.467) -> CIZone:
    """Convenience: classify directly from a raw LL z-score using the standard
    main-brain anchor. Pass `anchor=brain.ci_anchor` for the shadow brain."""
    if ll_zscore >= 0:
        return _ZONES[1]
    ci = abs(ll_zscore) / max(anchor, 1e-6) * 100.0
    return classify_ci_zone(ci)
