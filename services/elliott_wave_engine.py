"""
Elliott Wave Engine — SPY Primary Degree Wave Counting

Pipeline:
  detect_pivots()         -> ATR-filtered swing highs/lows
  find_wave_sequences()   -> candidate 5-wave impulse + 3-wave corrective windows
  score_sequence()        -> hard EW rules + Fibonacci ratio scoring
  get_best_wave_count()   -> highest-confidence valid count
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Pivot:
    date: pd.Timestamp
    price: float      # closing price at pivot bar
    type: str         # "high" | "low"


@dataclass
class WaveSequence:
    pivots: list[Pivot]   # 6 for impulse, 4 for corrective
    wave_type: str        # "impulse" | "corrective"
    labels: list[str]     # ["0","1","2","3","4","5"] or ["0","A","B","C"]


@dataclass
class BestCount:
    sequence: WaveSequence
    confidence: int              # 0-100
    wave_position: str           # e.g. "In Wave 3 of Primary Impulse (incomplete)"
    current_wave_label: str      # e.g. "3" or "C"
    invalidation_level: float    # price that breaks this count
    fibonacci_hits: list[str] = field(default_factory=list)


def _atr14(series: pd.Series) -> pd.Series:
    """Approximate ATR(14) using daily close-to-close range."""
    diff = series.diff().abs()
    return diff.rolling(14).mean()


def detect_pivots(series: pd.Series, atr_multiplier: float = 1.0) -> list[Pivot]:
    """
    ATR(14)-filtered zigzag pivot detection using closing prices.

    A high pivot is confirmed when price subsequently closes >= atr_multiplier * ATR(14)
    below the candidate high's close.
    A low pivot is confirmed when price subsequently closes >= atr_multiplier * ATR(14)
    above the candidate low's close.

    Returns alternating list of Pivot objects (high, low, high, low, ...).
    Target density: 20-40 pivots on 252 trading days with default atr_multiplier=1.0.
    """
    if len(series) < 20:
        return []

    atr = _atr14(series)
    closes = series.values
    dates = series.index
    n = len(closes)

    pivots: list[Pivot] = []
    candidate_idx: int | None = None
    candidate_type: str | None = None  # "high" | "low"

    # Determine initial direction from first meaningful move
    for i in range(1, min(n, 30)):
        atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        if atr_val == 0:
            continue
        if closes[i] > closes[0] + atr_val:
            candidate_idx = 0
            candidate_type = "low"
            break
        if closes[i] < closes[0] - atr_val:
            candidate_idx = 0
            candidate_type = "high"
            break

    if candidate_idx is None:
        return []

    for i in range(1, n):
        atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
        if atr_val == 0:
            continue

        threshold = atr_multiplier * atr_val

        if candidate_type == "high":
            if closes[i] > closes[candidate_idx]:
                # Extend the candidate higher
                candidate_idx = i
            elif closes[i] <= closes[candidate_idx] - threshold:
                # Confirm the high pivot
                pivots.append(Pivot(
                    date=pd.Timestamp(dates[candidate_idx]),
                    price=float(closes[candidate_idx]),
                    type="high",
                ))
                candidate_idx = i
                candidate_type = "low"

        else:  # candidate_type == "low"
            if closes[i] < closes[candidate_idx]:
                # Extend the candidate lower
                candidate_idx = i
            elif closes[i] >= closes[candidate_idx] + threshold:
                # Confirm the low pivot
                pivots.append(Pivot(
                    date=pd.Timestamp(dates[candidate_idx]),
                    price=float(closes[candidate_idx]),
                    type="low",
                ))
                candidate_idx = i
                candidate_type = "high"

    return pivots


def find_wave_sequences(pivots: list[Pivot]) -> list[WaveSequence]:
    """
    Sliding window over last 30 pivots to generate candidate wave sequences.

    Impulse (5-wave): window of 6 consecutive pivots starting on a low pivot.
    Corrective (3-wave): window of 4 consecutive pivots starting on a high pivot.
    """
    if len(pivots) < 4:
        return []

    # Restrict to last 30 pivots
    window = pivots[-30:]
    sequences: list[WaveSequence] = []

    for i in range(len(window) - 5):
        # Impulse candidate: 6 pivots starting on a low (bullish) or high (bearish)
        chunk6 = window[i:i + 6]
        if chunk6[0].type == "low" or chunk6[0].type == "high":
            sequences.append(WaveSequence(
                pivots=chunk6,
                wave_type="impulse",
                labels=["0", "1", "2", "3", "4", "5"],
            ))

    for i in range(len(window) - 3):
        # Corrective candidate: 4 pivots starting on a high (bearish ABC) or low (bullish ABC)
        chunk4 = window[i:i + 4]
        if chunk4[0].type == "high" or chunk4[0].type == "low":
            sequences.append(WaveSequence(
                pivots=chunk4,
                wave_type="corrective",
                labels=["0", "A", "B", "C"],
            ))

    return sequences


_FIB_TOLERANCE = 0.15  # +/-15%


def _fib_close(ratio: float, target: float) -> bool:
    """Return True if ratio is within FIB_TOLERANCE of target."""
    return abs(ratio - target) / max(target, 1e-9) <= _FIB_TOLERANCE


def score_sequence(seq: WaveSequence) -> tuple[bool, int, list[str]]:
    """
    Validate a wave sequence against EW hard rules and score Fibonacci ratios.

    Returns:
        (is_valid, confidence_0_to_100, fibonacci_hits)
    """
    p = seq.pivots

    if seq.wave_type == "impulse":
        # Wave lengths (absolute price moves, close-to-close)
        w1 = abs(p[1].price - p[0].price)
        w2 = abs(p[2].price - p[1].price)
        w3 = abs(p[3].price - p[2].price)
        w4 = abs(p[4].price - p[3].price)
        w5 = abs(p[5].price - p[4].price)

        # Hard Rules
        # 1. Wave 3 is not the shortest impulse wave
        if w3 <= w1 and w3 <= w5:
            return False, 0, []

        # 2. Wave 4 does not overlap wave 1
        if p[0].type == "low":
            # Bullish: wave 4 low must not drop below wave 1 high
            if p[4].price < p[1].price:
                return False, 0, []
        else:
            # Bearish: wave 4 high must not exceed wave 1 low
            if p[4].price > p[1].price:
                return False, 0, []

        # 3. Wave 2 never retraces beyond wave 0
        if p[0].type == "low":
            # Bullish: wave 2 low must not drop below wave 0 low
            if p[2].price < p[0].price:
                return False, 0, []
        else:
            # Bearish: wave 2 high must not exceed wave 0 high
            if p[2].price > p[0].price:
                return False, 0, []

        # Fibonacci Scoring
        raw_score = 0
        hits: list[str] = []

        if w1 > 0:
            r31 = w3 / w1
            if _fib_close(r31, 1.618):
                raw_score += 20
                hits.append("Wave 3 = 1.618 x Wave 1 (+20pts)")
            elif _fib_close(r31, 2.618):
                raw_score += 15
                hits.append("Wave 3 = 2.618 x Wave 1 (+15pts)")

        w1_retrace = w2 / w1 if w1 > 0 else 0
        if _fib_close(w1_retrace, 0.618):
            raw_score += 15
            hits.append("Wave 2 retraces 0.618 of Wave 1 (+15pts)")

        if w3 > 0:
            w3_retrace = w4 / w3
            if _fib_close(w3_retrace, 0.382):
                raw_score += 15
                hits.append("Wave 4 retraces 0.382 of Wave 3 (+15pts)")
            elif _fib_close(w3_retrace, 0.500):
                raw_score += 10
                hits.append("Wave 4 retraces 0.500 of Wave 3 (+10pts)")

        if w1 > 0 and _fib_close(w5 / w1, 1.0):
            raw_score += 10
            hits.append("Wave 5 = Wave 1 (+10pts)")

        # Normalize: max possible raw score = 85
        confidence = int(round(raw_score / 85 * 100))
        return True, confidence, hits

    else:  # corrective
        wa = abs(p[1].price - p[0].price)
        wb = abs(p[2].price - p[1].price)
        wc = abs(p[3].price - p[2].price)

        # C should move in same direction as A
        a_dir = 1 if p[1].price < p[0].price else -1
        c_dir = 1 if p[3].price < p[2].price else -1
        if a_dir != c_dir:
            return False, 0, []

        raw_score = 0
        hits: list[str] = []

        if wa > 0:
            if _fib_close(wc / wa, 1.0):
                raw_score += 30
                hits.append("Wave C = Wave A (+30pts)")
            elif _fib_close(wc / wa, 0.618):
                raw_score += 20
                hits.append("Wave C = 0.618 x Wave A (+20pts)")

        confidence = int(round(min(raw_score / 50 * 100, 100)))
        return True, confidence, hits


def get_best_wave_count(series: pd.Series) -> BestCount | None:
    """
    Full pipeline: detect pivots -> find sequences -> score -> return best valid count.

    Returns None if:
    - Fewer than 60 bars in series
    - Fewer than 6 pivots detected
    - No sequence passes hard EW rules
    """
    if len(series) < 60:
        return None

    pivots = detect_pivots(series)
    if len(pivots) < 6:
        return None

    sequences = find_wave_sequences(pivots)
    if not sequences:
        return None

    best_seq: WaveSequence | None = None
    best_confidence: int = -1
    best_hits: list[str] = []

    for seq in sequences:
        is_valid, confidence, hits = score_sequence(seq)
        if is_valid and confidence > best_confidence:
            best_seq = seq
            best_confidence = confidence
            best_hits = hits

    if best_seq is None:
        return None

    # Determine wave position
    most_recent_pivot = pivots[-1]
    last_seq_pivot = best_seq.pivots[-1]
    is_complete = (last_seq_pivot.date == most_recent_pivot.date)

    current_wave_label = best_seq.labels[-1]
    wave_type_name = "Primary Impulse" if best_seq.wave_type == "impulse" else "Primary Correction"

    if is_complete:
        wave_position = f"Wave {current_wave_label} of {wave_type_name} complete"
    else:
        wave_position = f"In Wave {current_wave_label} of {wave_type_name} (incomplete)"

    # Invalidation level: origin pivot 0 for both impulse and corrective
    invalidation_level = float(best_seq.pivots[0].price)

    return BestCount(
        sequence=best_seq,
        confidence=best_confidence,
        wave_position=wave_position,
        current_wave_label=current_wave_label,
        invalidation_level=invalidation_level,
        fibonacci_hits=best_hits,
    )
