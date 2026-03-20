"""
Elliott Wave Engine — SPY Multi-Degree Wave Counting

Pipeline:
  detect_pivots()         -> ATR-filtered swing highs/lows
  find_wave_sequences()   -> candidate 5-wave impulse + 3-wave corrective windows
  score_sequence()        -> hard EW rules + Fibonacci ratio scoring
  get_best_wave_count()   -> highest-confidence valid count

Multi-degree support:
  DEGREE_CONFIGS          -> lookback + ATR multiplier per degree
  DEGREE_WAVE_LABELS      -> standard EW notation per degree
  get_degree_wave_count() -> impulse count at a specific degree
"""

from __future__ import annotations
from collections import OrderedDict
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

    # Restrict to last 50 pivots — wider window finds better structural counts
    window = pivots[-50:]
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


_FIB_RETRACE_TOL = 0.07   # tighter: retracements (W2, W4, WB) are more predictable
_FIB_EXTEND_TOL  = 0.12   # wider: extensions (W3, WC, W5) vary more in practice


def _retrace_close(ratio: float, target: float) -> bool:
    """True if ratio is within FIB_RETRACE_TOL of target."""
    return abs(ratio - target) / max(target, 1e-9) <= _FIB_RETRACE_TOL


def _extend_close(ratio: float, target: float) -> bool:
    """True if ratio is within FIB_EXTEND_TOL of target."""
    return abs(ratio - target) / max(target, 1e-9) <= _FIB_EXTEND_TOL


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

        # Fibonacci Scoring — EWF ratios
        raw_score = 0
        hits: list[str] = []

        # Wave 2: 61.8%, 50%, 76.4%, 85.4% of Wave 1 (retracement — tight tolerance)
        w2_retrace = w2 / w1 if w1 > 0 else 0
        if _retrace_close(w2_retrace, 0.618):
            raw_score += 15; hits.append("Wave 2 = 61.8% of Wave 1 (+15pts)")
        elif _retrace_close(w2_retrace, 0.500):
            raw_score += 12; hits.append("Wave 2 = 50% of Wave 1 (+12pts)")
        elif _retrace_close(w2_retrace, 0.764):
            raw_score += 10; hits.append("Wave 2 = 76.4% of Wave 1 (+10pts)")
        elif _retrace_close(w2_retrace, 0.854):
            raw_score += 8; hits.append("Wave 2 = 85.4% of Wave 1 (+8pts)")

        # Wave 3: 161.8%, 261.8%, 200%, 323.6% of Wave 1 (extension — wide tolerance)
        w3_ext = w3 / w1 if w1 > 0 else 0
        if _extend_close(w3_ext, 1.618):
            raw_score += 25; hits.append("Wave 3 = 161.8% of Wave 1 (+25pts)")
        elif _extend_close(w3_ext, 2.618):
            raw_score += 22; hits.append("Wave 3 = 261.8% of Wave 1 (+22pts)")
        elif _extend_close(w3_ext, 2.000):
            raw_score += 18; hits.append("Wave 3 = 200% of Wave 1 (+18pts)")
        elif _extend_close(w3_ext, 3.236):
            raw_score += 15; hits.append("Wave 3 = 323.6% of Wave 1 (+15pts)")

        # Wave 4: 38.2%, 23.6%, 14.6% of Wave 3 (retracement — tight tolerance)
        w4_retrace = w4 / w3 if w3 > 0 else 0
        if w4_retrace <= 0.50:
            if _retrace_close(w4_retrace, 0.382):
                raw_score += 15; hits.append("Wave 4 = 38.2% of Wave 3 (+15pts)")
            elif _retrace_close(w4_retrace, 0.236):
                raw_score += 12; hits.append("Wave 4 = 23.6% of Wave 3 (+12pts)")
            elif _retrace_close(w4_retrace, 0.146):
                raw_score += 10; hits.append("Wave 4 = 14.6% of Wave 3 (+10pts)")

        # Wave 5: equal to Wave 1, 61.8% of Wave 1+3, or 1.618× Wave 1 (extension)
        w5_vs_w1    = w5 / w1 if w1 > 0 else 0
        w5_vs_w1w3  = w5 / (w1 + w3) if (w1 + w3) > 0 else 0
        if _extend_close(w5_vs_w1, 1.618):
            raw_score += 12; hits.append("Wave 5 = 161.8% of Wave 1 (+12pts)")
        elif _extend_close(w5_vs_w1, 1.000):
            raw_score += 12; hits.append("Wave 5 = Wave 1 (+12pts)")
        elif _extend_close(w5_vs_w1w3, 0.618):
            raw_score += 10; hits.append("Wave 5 = 61.8% of Wave 1+3 (+10pts)")
        elif _extend_close(w5_vs_w1w3, 0.382):
            raw_score += 8;  hits.append("Wave 5 = 38.2% of Wave 1+3 (+8pts)")

        # Wave Alternation bonus: W2 and W4 should differ in depth
        # Deep W2 (>50%) pairs with shallow W4 (<38.2%), or vice versa
        w2_deep = w2_retrace >= 0.50
        w4_deep = w4_retrace >= 0.382
        if (w2_deep and not w4_deep) or (not w2_deep and w4_deep):
            raw_score += 8
            hits.append("Wave alternation confirmed: W2/W4 differ in depth (+8pts)")

        # Normalize: max possible raw score = 75 (15+25+15+12+8 alternation)
        confidence = int(round(raw_score / 75 * 100))
        return True, confidence, hits

    else:  # corrective (Zigzag 5-3-5)
        wa = abs(p[1].price - p[0].price)
        wb = abs(p[2].price - p[1].price)
        wc = abs(p[3].price - p[2].price)

        # C must move in same direction as A
        a_dir = 1 if p[1].price < p[0].price else -1
        c_dir = 1 if p[3].price < p[2].price else -1
        if a_dir != c_dir:
            return False, 0, []

        raw_score = 0
        hits: list[str] = []

        # Wave B: 61.8%, 50%, 76.4%, 85.4% of Wave A (retracement — tight tolerance)
        wb_retrace = wb / wa if wa > 0 else 0
        if _retrace_close(wb_retrace, 0.618):
            raw_score += 20; hits.append("Wave B = 61.8% of Wave A (+20pts)")
        elif _retrace_close(wb_retrace, 0.500):
            raw_score += 16; hits.append("Wave B = 50% of Wave A (+16pts)")
        elif _retrace_close(wb_retrace, 0.764):
            raw_score += 14; hits.append("Wave B = 76.4% of Wave A (+14pts)")
        elif _retrace_close(wb_retrace, 0.854):
            raw_score += 12; hits.append("Wave B = 85.4% of Wave A (+12pts)")

        # Wave C: 100%, 61.8%, 123.6% of Wave A (extension — wide tolerance)
        wc_vs_wa = wc / wa if wa > 0 else 0
        if _extend_close(wc_vs_wa, 1.000):
            raw_score += 30; hits.append("Wave C = 100% of Wave A (+30pts)")
        elif _extend_close(wc_vs_wa, 0.618):
            raw_score += 24; hits.append("Wave C = 61.8% of Wave A (+24pts)")
        elif _extend_close(wc_vs_wa, 1.236):
            raw_score += 24; hits.append("Wave C = 123.6% of Wave A (+24pts)")
        elif _extend_close(wc_vs_wa, 1.618):
            raw_score += 20; hits.append("Wave C = 161.8% of Wave A (+20pts)")

        confidence = int(round(min(raw_score / 50 * 100, 100)))
        return True, confidence, hits


def _build_best_count(
    seq: WaveSequence, confidence: int, hits: list[str], all_pivots: list[Pivot]
) -> BestCount:
    """Build a BestCount from a scored sequence and the full pivot list."""
    most_recent_pivot = all_pivots[-1]
    last_seq_pivot = seq.pivots[-1]
    is_complete = (last_seq_pivot.date == most_recent_pivot.date)

    current_wave_label = seq.labels[-1]
    wave_type_name = "Primary Impulse" if seq.wave_type == "impulse" else "Primary Correction"

    if is_complete:
        wave_position = f"Wave {current_wave_label} of {wave_type_name} complete"
    else:
        wave_position = f"In Wave {current_wave_label} of {wave_type_name} (incomplete)"

    invalidation_level = float(seq.pivots[0].price)

    return BestCount(
        sequence=seq,
        confidence=confidence,
        wave_position=wave_position,
        current_wave_label=current_wave_label,
        invalidation_level=invalidation_level,
        fibonacci_hits=hits,
    )


def _sequences_overlap(s1: WaveSequence, s2: WaveSequence, min_shared: int = 4) -> bool:
    """Check if two sequences share min_shared or more pivot dates."""
    dates1 = {p.date for p in s1.pivots}
    dates2 = {p.date for p in s2.pivots}
    return len(dates1 & dates2) >= min_shared


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

    return _build_best_count(best_seq, best_confidence, best_hits, pivots)


def get_top_wave_counts(series: pd.Series, top_n: int = 5) -> list[BestCount]:
    """
    Full pipeline returning up to top_n deduplicated wave counts sorted by confidence.

    Deduplication: if two sequences share 4+ pivot dates, keep the higher-confidence one.
    Returns empty list if no valid counts found.
    """
    if len(series) < 60:
        return []

    pivots = detect_pivots(series)
    if len(pivots) < 6:
        return []

    sequences = find_wave_sequences(pivots)
    if not sequences:
        return []

    # Collect all valid scored sequences
    scored: list[tuple[WaveSequence, int, list[str]]] = []
    for seq in sequences:
        is_valid, confidence, hits = score_sequence(seq)
        if is_valid:
            scored.append((seq, confidence, hits))

    if not scored:
        return []

    # Sort by confidence descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate overlapping sequences (keep higher confidence)
    kept: list[tuple[WaveSequence, int, list[str]]] = []
    for seq, conf, hits in scored:
        overlaps = any(_sequences_overlap(seq, k[0]) for k in kept)
        if not overlaps:
            kept.append((seq, conf, hits))
        if len(kept) >= top_n:
            break

    return [_build_best_count(seq, conf, hits, pivots) for seq, conf, hits in kept]


# ── Multi-degree support ──────────────────────────────────────────────────────

# Lookback in trading days and ATR multiplier per degree.
# Larger degrees use longer lookbacks and higher multipliers to capture fewer,
# larger pivots that represent multi-year structural swings.
DEGREE_CONFIGS: OrderedDict = OrderedDict([
    ("Grand Supercycle", {"lookback": None, "atr_mult": 4.0}),   # full history
    ("Supercycle",       {"lookback": 5040, "atr_mult": 2.8}),   # ~20 years
    ("Cycle",            {"lookback": 1260, "atr_mult": 2.0}),   # ~5 years
    ("Primary",          {"lookback": 504,  "atr_mult": 1.5}),   # ~2 years
    ("Intermediate",     {"lookback": 252,  "atr_mult": 1.0}),   # ~1 year
    ("Minor",            {"lookback": 126,  "atr_mult": 0.7}),   # ~6 months
    ("Minute",           {"lookback": 63,   "atr_mult": 0.4}),   # ~3 months
])

# Standard EW notation: Grand Supercycle [[I]], Supercycle (I), Cycle I,
# Primary [1], Intermediate (1), Minor 1, Minute i
DEGREE_WAVE_LABELS: dict[str, list[str]] = {
    "Grand Supercycle": ["[[I]]",  "[[II]]",  "[[III]]",  "[[IV]]",  "[[V]]"],
    "Supercycle":       ["(I)",    "(II)",    "(III)",    "(IV)",    "(V)"],
    "Cycle":            ["I",      "II",      "III",      "IV",      "V"],
    "Primary":          ["[1]",    "[2]",     "[3]",      "[4]",     "[5]"],
    "Intermediate":     ["(1)",    "(2)",     "(3)",      "(4)",     "(5)"],
    "Minor":            ["1",      "2",       "3",        "4",       "5"],
    "Minute":           ["i",      "ii",      "iii",      "iv",      "v"],
}

DEGREE_CORRECTIVE_LABELS: dict[str, list[str]] = {
    "Grand Supercycle": ["[[A]]", "[[B]]", "[[C]]"],
    "Supercycle":       ["(A)",   "(B)",   "(C)"],
    "Cycle":            ["A",     "B",     "C"],
    "Primary":          ["[A]",   "[B]",   "[C]"],
    "Intermediate":     ["(a)",   "(b)",   "(c)"],
    "Minor":            ["a",     "b",     "c"],
    "Minute":           ["((a))", "((b))", "((c))"],
}


def _build_degree_count(
    seq: WaveSequence,
    confidence: int,
    hits: list[str],
    all_pivots: list[Pivot],
    degree: str,
) -> BestCount:
    """Build a BestCount with degree-specific labeling and wave position string."""
    last = seq.pivots[-1]
    is_complete = last.date == all_pivots[-1].date
    label = seq.labels[-1]
    status = "complete" if is_complete else "in progress"
    wave_position = f"{degree} · Wave {label} {status}"
    return BestCount(
        sequence=seq,
        confidence=confidence,
        wave_position=wave_position,
        current_wave_label=label,
        invalidation_level=float(seq.pivots[0].price),
        fibonacci_hits=hits,
    )


def get_degree_wave_count(series: pd.Series, degree: str) -> BestCount | None:
    """
    Detect the best impulse (1-2-3-4-5) wave count at a specific EW degree.

    Uses degree-appropriate ATR multiplier and lookback window.
    Only considers impulse sequences — no A-B-C corrective counts.
    Returns None if insufficient data or no valid count found.
    """
    if degree not in DEGREE_CONFIGS:
        return None

    config = DEGREE_CONFIGS[degree]
    lookback = config["lookback"]
    data = series if lookback is None else series.iloc[-min(lookback, len(series)):]

    if len(data) < 60:
        return None

    pivots = detect_pivots(data, atr_multiplier=config["atr_mult"])
    if len(pivots) < 6:
        return None

    sequences = find_wave_sequences(pivots)
    impulse_seqs = [s for s in sequences if s.wave_type == "impulse"]
    if not impulse_seqs:
        return None

    best_seq: WaveSequence | None = None
    best_conf = -1
    best_hits: list[str] = []

    for seq in impulse_seqs:
        valid, conf, hits = score_sequence(seq)
        if valid and conf > best_conf:
            best_seq, best_conf, best_hits = seq, conf, hits

    if best_seq is None:
        return None

    labels = ["0"] + DEGREE_WAVE_LABELS[degree]
    labeled_seq = WaveSequence(
        pivots=best_seq.pivots,
        wave_type="impulse",
        labels=labels,
    )

    return _build_degree_count(labeled_seq, best_conf, best_hits, pivots, degree)


# ── Backtesting ───────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    date: pd.Timestamp
    wave_label: str
    direction: str  # "Bullish" | "Bearish"
    entry_price: float
    target_price: float
    invalidation_price: float
    outcome: str | None = None  # "Target Hit" | "Invalidated" | "Indeterminate"
    pnl_pct: float | None = None


def backtest_wave_counts(
    series: pd.Series,
    degree: str,
    test_period_bars: int = 252
) -> list[BacktestResult]:
    """
    Run a historical backtest of wave counts for a specific degree.
    """
    if len(series) < test_period_bars + 100:
        return []

    results: list[BacktestResult] = []
    
    # Start iterating from `test_period_bars` ago
    start_idx = len(series) - test_period_bars
    if start_idx < 100: start_idx = 100
    
    # State tracking to avoid duplicate signals on consecutive days
    # We track the pivot date of the active wave start to identify "new" waves
    last_signal_wave_start_date = None
    
    # Iterate through history day by day
    # We step every 5 days to speed up backtest loop, or 1 day for precision?
    # Let's do every 3 days to balance speed/precision
    step = 3
    
    for i in range(start_idx, len(series) - 5, step):
        # Slice history up to current simulated day i
        history = series.iloc[:i+1]
        current_date = history.index[-1]
        current_price = float(history.iloc[-1])
        
        # Get wave count at this point in time
        count = get_degree_wave_count(history, degree)
        
        # We only care about active impulse waves (In Wave 3 or 5)
        if not count or "complete" in count.wave_position:
            continue
            
        label = count.current_wave_label  # "3" or "5"
        if label not in ["3", "5"]:
            continue
            
        # Check if this is a new signal or same as last week
        # Use the date of the pivot that started this wave
        # Wave 3 starts at pivot 2. Wave 5 starts at pivot 4.
        pivots = count.sequence.pivots
        wave_start_idx = 2 if label == "3" else 4
        if len(pivots) <= wave_start_idx:
            continue
            
        wave_start_date = pivots[wave_start_idx].date
        
        if wave_start_date == last_signal_wave_start_date:
            continue # Already traded this wave
            
        last_signal_wave_start_date = wave_start_date

        # Determine Trend Direction
        # Wave 1 is defined by pivots 0 -> 1.
        p0 = pivots[0]
        p1 = pivots[1]
        
        is_bullish = p1.price > p0.price
        direction = "Bullish" if is_bullish else "Bearish"
        
        # Calculate Target Price
        # Wave 3 target: 1.618 * W1 length
        # Wave 5 target: 1.0 * W1 length
        w1_len = abs(p1.price - p0.price)
        target_price = 0.0
        
        if label == "3":
            p2 = pivots[2]
            extension = w1_len * 1.618
            target_price = (p2.price + extension) if is_bullish else (p2.price - extension)
        elif label == "5":
            p4 = pivots[4]
            extension = w1_len * 1.0
            target_price = (p4.price + extension) if is_bullish else (p4.price - extension)

        invalidation = count.invalidation_level
        
        # Validate R:R (Risk:Reward)
        # If target is already hit or invalidation is super close, skip?
        # For backtest, we record it.
        
        # Create Result Object
        res = BacktestResult(
            date=pd.Timestamp(current_date),
            wave_label=f"Wave {label}",
            direction=direction,
            entry_price=current_price,
            target_price=target_price,
            invalidation_price=invalidation
        )
        
        # Check Future Outcome
        # Look ahead from i+1 to end of series
        future_data = series.iloc[i+1:]
        outcome = "Indeterminate"
        exit_price = current_price
        
        # We need a loop to find first touch of target or invalidation
        for date, price in future_data.items():
            price = float(price)
            
            if is_bullish:
                if price <= invalidation:
                    outcome = "Invalidated"
                    exit_price = invalidation # Assume stopped out at level
                    break
                if price >= target_price:
                    outcome = "Target Hit"
                    exit_price = target_price
                    break
            else: # Bearish
                if price >= invalidation:
                    outcome = "Invalidated"
                    exit_price = invalidation
                    break
                if price <= target_price:
                    outcome = "Target Hit"
                    exit_price = target_price
                    break
        
        res.outcome = outcome
        if outcome == "Target Hit":
            res.pnl_pct = abs(target_price - current_price) / current_price * 100
        elif outcome == "Invalidated":
             res.pnl_pct = -abs(current_price - invalidation) / current_price * 100
        else:
             # Mark to market at end of data if indeterminate
             end_price = float(series.iloc[-1])
             if is_bullish:
                 res.pnl_pct = (end_price - current_price) / current_price * 100
             else:
                 res.pnl_pct = (current_price - end_price) / current_price * 100

        results.append(res)
        
    return results


def get_degree_corrective_count(series: pd.Series, degree: str) -> BestCount | None:
    """
    Detect the best ABC corrective (zigzag) count at a specific EW degree.

    Uses degree-appropriate ATR multiplier and lookback window.
    Only considers corrective sequences — no 1-2-3-4-5 impulse.
    Returns None if insufficient data or no valid count found.
    """
    if degree not in DEGREE_CONFIGS:
        return None

    config = DEGREE_CONFIGS[degree]
    lookback = config["lookback"]
    data = series if lookback is None else series.iloc[-min(lookback, len(series)):]

    if len(data) < 60:
        return None

    pivots = detect_pivots(data, atr_multiplier=config["atr_mult"])
    if len(pivots) < 4:
        return None

    sequences = find_wave_sequences(pivots)
    corrective_seqs = [s for s in sequences if s.wave_type == "corrective"]
    if not corrective_seqs:
        return None

    best_seq: WaveSequence | None = None
    best_conf = -1
    best_hits: list[str] = []

    for seq in corrective_seqs:
        valid, conf, hits = score_sequence(seq)
        if valid and conf > best_conf:
            best_seq, best_conf, best_hits = seq, conf, hits

    if best_seq is None:
        return None

    labels = ["0"] + DEGREE_CORRECTIVE_LABELS[degree]
    labeled_seq = WaveSequence(
        pivots=best_seq.pivots,
        wave_type="corrective",
        labels=labels,
    )

    return _build_degree_count(labeled_seq, best_conf, best_hits, pivots, degree)


@dataclass
class WaveForecast:
    """Projected price path from the current active wave position."""
    wave_label: str          # current wave label e.g. "3"
    direction: str           # "Bullish" | "Bearish"
    current_price: float

    # Primary scenario (trend continuation)
    primary_target: float    # Fibonacci extension target
    primary_label: str       # e.g. "Wave 3 target (1.618× W1)"
    primary_probability: int # 0-100, derived from confidence

    # Intermediate waypoint (next corrective before final target)
    waypoint_target: float | None     # e.g. Wave 4 pullback level
    waypoint_label: str | None        # e.g. "Wave 4 pullback (38.2% of W3)"

    # Alternate scenario (reversal)
    alternate_target: float  # invalidation extended
    alternate_label: str     # e.g. "Breakdown below Wave 0"
    alternate_probability: int  # 100 - primary_probability

    invalidation: float
    rationale: str           # one-line explanation


def build_wave_forecast(count: BestCount, current_price: float) -> "WaveForecast | None":
    """
    Build a price forecast from the current BestCount.

    Logic:
      - Wave 3 in progress → primary target = Wave 2 low + 1.618×W1; waypoint = Wave 4 (38.2% pullback of W3 so far); alternate = below Wave 0
      - Wave 4 in progress → primary target = Wave 4 end + W1 length (W5 equality); waypoint = None; alternate = overlap with W1 (invalidation)
      - Wave 5 in progress → primary target = Wave 4 end + 0.618×W1; waypoint = None; alternate = below Wave 3 top
      - Wave A/B/C corrective → primary target = C = 100% of A; waypoint = B retrace (61.8% of A); alternate = C extends to 161.8% of A
    """
    if count is None:
        return None

    pivots = count.sequence.pivots
    label = count.current_wave_label
    n = len(pivots)
    confidence = count.confidence

    # Determine trend direction from W0→W1
    p0 = pivots[0]
    p1 = pivots[1] if n > 1 else None
    if p1 is None:
        return None

    is_bullish = p1.price > p0.price
    direction = "Bullish" if is_bullish else "Bearish"
    w1_len = abs(p1.price - p0.price)

    # Strip degree notation to get raw label character
    raw = label.strip("[]()i ")
    # Normalize: [[I]] → I, (I) → I, [1] → 1, (1) → 1, i → 5 equivalent
    for ch in ["[[", "]]", "(", ")", "[", "]"]:
        raw = raw.replace(ch, "")
    raw = raw.strip()

    # Map roman/letter to numeric for logic
    roman_map = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
                 "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
    numeric = roman_map.get(raw, raw)

    invalidation = count.invalidation_level
    primary_target = current_price
    primary_label = ""
    waypoint_target = None
    waypoint_label = None
    alternate_target = current_price
    alternate_label = ""
    rationale = ""

    sign = 1 if is_bullish else -1

    if numeric == "3" and n >= 3:
        p2 = pivots[2]
        primary_target = p2.price + sign * w1_len * 1.618
        primary_label = f"Wave 3 target (1.618× W1 = ${primary_target:.2f})"
        w3_so_far = abs(current_price - p2.price)
        waypoint_target = current_price - sign * w3_so_far * 0.382
        waypoint_label = f"Wave 4 pullback est. (38.2% of W3 = ${waypoint_target:.2f})"
        alternate_target = p0.price - sign * w1_len * 0.5
        alternate_label = f"Breakdown / W0 breach (${alternate_target:.2f})"
        rationale = f"Wave 3 in progress — expect extension to ${primary_target:.2f} (1.618× W1), then Wave 4 pullback ~${waypoint_target:.2f}"

    elif numeric == "4" and n >= 4:
        p3 = pivots[3] if n > 3 else None
        if p3:
            primary_target = current_price + sign * w1_len
            primary_label = f"Wave 5 target (W1 equality = ${primary_target:.2f})"
            alternate_target = p1.price  # W4 overlapping W1 = invalidation
            alternate_label = f"W4/W1 overlap invalidation (${alternate_target:.2f})"
            rationale = f"Wave 4 correction — expect resumption in Wave 5 toward ${primary_target:.2f}"
        else:
            return None

    elif numeric == "5" and n >= 5:
        p4 = pivots[4] if n > 4 else None
        if p4:
            primary_target = p4.price + sign * w1_len * 0.618
            primary_label = f"Wave 5 target (61.8% of W1 = ${primary_target:.2f})"
            alternate_target = pivots[3].price if n > 3 else invalidation
            alternate_label = f"Reversal below Wave 3 top (${alternate_target:.2f})"
            rationale = f"Wave 5 final leg — target ${primary_target:.2f}, watch for momentum divergence"
        else:
            return None

    elif raw in ("A", "a", "[A]", "(A)", "[[A]]", "(a)", "((a))") and n >= 2:
        wa_len = abs(p1.price - p0.price)
        # B retrace 61.8% of A
        waypoint_target = p1.price + sign * wa_len * 0.618
        waypoint_label = f"Wave B retrace (61.8% of A = ${waypoint_target:.2f})"
        # C = 100% of A from B
        primary_target = p1.price - sign * wa_len  # approximate C end
        primary_label = f"Wave C target (100% of A = ${primary_target:.2f})"
        alternate_target = p1.price - sign * wa_len * 1.618
        alternate_label = f"Wave C extends (161.8% of A = ${alternate_target:.2f})"
        rationale = f"Wave A down — expect B retrace to ~${waypoint_target:.2f} then C down to ~${primary_target:.2f}"

    elif raw in ("B", "b", "[B]", "(B)", "[[B]]", "(b)", "((b))") and n >= 3:
        p2 = pivots[2]
        wa_len = abs(p1.price - p0.price)
        primary_target = p2.price - sign * wa_len
        primary_label = f"Wave C target (100% of A = ${primary_target:.2f})"
        alternate_target = p2.price - sign * wa_len * 1.618
        alternate_label = f"Extended Wave C (161.8% of A = ${alternate_target:.2f})"
        rationale = f"Wave B retracing — final Wave C expected toward ${primary_target:.2f}"

    elif raw in ("C", "c", "[C]", "(C)", "[[C]]", "(c)", "((c))") and n >= 3:
        p2 = pivots[2]
        wa_len = abs(p1.price - p0.price)
        primary_target = p2.price - sign * wa_len
        primary_label = f"Wave C completion (100% of A = ${primary_target:.2f})"
        alternate_target = p2.price - sign * wa_len * 1.618
        alternate_label = f"Extended C target (161.8% of A = ${alternate_target:.2f})"
        rationale = f"Wave C in progress — target ${primary_target:.2f}, completion ends the correction"

    else:
        return None

    primary_probability = max(20, min(85, confidence))
    alternate_probability = 100 - primary_probability

    # Derive direction from target vs current price (not historical W0→W1)
    # This correctly handles Wave 4 corrections (bearish move within a bull trend)
    if primary_target != current_price:
        direction = "Bullish" if primary_target > current_price else "Bearish"

    return WaveForecast(
        wave_label=label,
        direction=direction,
        current_price=current_price,
        primary_target=primary_target,
        primary_label=primary_label,
        primary_probability=primary_probability,
        waypoint_target=waypoint_target,
        waypoint_label=waypoint_label,
        alternate_target=alternate_target,
        alternate_label=alternate_label,
        alternate_probability=alternate_probability,
        invalidation=invalidation,
        rationale=rationale,
    )
