"""
Wyckoff Method Engine — Phase Detection

Detects Accumulation, Distribution, Markup, and Markdown phases
using price structure and volume behavior on any timeframe OHLCV data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Interval-adaptive parameters ─────────────────────────────────────────────

def _interval_params(interval: str) -> tuple[int, int]:
    """Return (atr_window, min_bars) scaled to the interval granularity.

    Shorter timeframes use tighter windows so the ATR baseline is calculated
    over a representative session-equivalent span rather than calendar time.
    min_bars is also relaxed so intraday consolidations (which are shorter)
    are not missed.
    """
    return {
        "1m":  (30,  5),
        "2m":  (30,  5),
        "5m":  (20,  5),
        "15m": (20,  6),
        "30m": (18,  6),
        "1h":  (24,  8),
        "4h":  (30,  8),
        "1d":  (50, 10),
        "1wk": (50, 10),
        "1mo": (50, 10),
    }.get(interval, (50, 10))


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TradingRange:
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    upper_bound: float        # resistance
    lower_bound: float        # support
    preceding_trend: str      # "up" | "down" | "neutral"


@dataclass
class WyckoffEvent:
    date: pd.Timestamp
    price: float
    event_type: str           # "SC","AR","ST","Spring","SOS","BC","UTAD","SOW"
    description: str


@dataclass
class WyckoffPhase:
    phase: str                # "Accumulation" | "Distribution" | "Markup" | "Markdown"
    confidence: int           # 0-100
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    key_levels: dict          # {"support": ..., "resistance": ...}
    events: list[WyckoffEvent] = field(default_factory=list)
    sub_phase: str = ""           # "A" | "B" | "C" | "D" | "E" | ""
    cause_target: float | None = None   # P&F-style price target
    demand_line: tuple | None = None    # (date1, price1, date2, price2)
    supply_line: tuple | None = None    # (date1, price1, date2, price2)


@dataclass
class VSABar:
    date: pd.Timestamp
    signal: str          # "Buying Climax" | "Selling Climax" | "No Supply" | "No Demand" | "Effort No Result" | "Strength" | "Weakness"
    description: str
    strength: int        # 1-3 (1=weak, 3=strong)


@dataclass
class WyckoffAnalysis:
    current_phase: WyckoffPhase
    all_phases: list[WyckoffPhase]
    phase_history: list[tuple]  # (start, end, phase_name)
    vsa_bars: list[VSABar] = field(default_factory=list)
    effort_vs_result: list[tuple] = field(default_factory=list)  # (date, description)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _atr(close: pd.Series, period: int = 14) -> pd.Series:
    """ATR approximation using close-to-close absolute diff rolling mean."""
    return close.diff().abs().rolling(period).mean()


def detect_trading_ranges(
    close: pd.Series, high: pd.Series, low: pd.Series,
    atr_window: int = 50, min_bars: int = 10,
) -> list[TradingRange]:
    """Find consolidation zones via ATR contraction.

    atr_window / min_bars are interval-adaptive (see _interval_params).
    Shorter timeframes use a tighter rolling window so session-length
    consolidations are detected instead of being swamped by daily ATR norms.
    """
    atr = _atr(close, period=14)
    atr_ma = atr.rolling(atr_window).mean()
    ratio = atr / atr_ma.replace(0, np.nan)

    # Find consecutive windows where ratio < 0.85
    contracted = ratio < 0.85
    ranges: list[TradingRange] = []
    start = None

    for i in range(len(contracted)):
        if contracted.iloc[i]:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_bars:
                span = float(high.iloc[start:i].max() - low.iloc[start:i].min())
                atr_val = float(atr.iloc[start]) if not np.isnan(atr.iloc[start]) else 0.0
                if atr_val == 0 or span <= atr_val * 2.5:
                    _add_range(ranges, close, high, low, start, i - 1)
            start = None
    if start is not None and (len(contracted) - start) >= min_bars:
        span = float(high.iloc[start:].max() - low.iloc[start:].min())
        atr_val = float(atr.iloc[start]) if not np.isnan(atr.iloc[start]) else 0.0
        if atr_val == 0 or span <= atr_val * 2.5:
            _add_range(ranges, close, high, low, start, len(contracted) - 1)

    return ranges


def _add_range(
    ranges: list[TradingRange],
    close: pd.Series, high: pd.Series, low: pd.Series,
    start: int, end: int,
) -> None:
    upper = float(high.iloc[start:end + 1].max())
    lower = float(low.iloc[start:end + 1].min())

    # Preceding trend: 20-bar SMA slope before range start
    lookback_start = max(0, start - 20)
    if start - lookback_start >= 5:
        segment = close.iloc[lookback_start:start]
        slope = np.polyfit(range(len(segment)), segment.values, 1)[0]
        if slope > 0.1:
            trend = "up"
        elif slope < -0.1:
            trend = "down"
        else:
            trend = "neutral"
    else:
        trend = "neutral"

    ranges.append(TradingRange(
        start_idx=start,
        end_idx=end,
        start_date=close.index[start],
        end_date=close.index[end],
        upper_bound=upper,
        lower_bound=lower,
        preceding_trend=trend,
    ))


def classify_volume_behavior(
    close: pd.Series, high: pd.Series, low: pd.Series,
    volume: pd.Series, tr: TradingRange,
) -> str:
    """VSA-weighted volume classification within a trading range.

    Each bar's volume is weighted by where price closed within the bar's
    high-low range (close position).  Closing near the high on a given day
    implies buying pressure even if the net close-to-close change is small.
    This avoids the naive up-day / down-day split which can be misleading
    on inside-bar or spinning-top days.
    """
    sl = slice(tr.start_idx, tr.end_idx + 1)
    c  = close.iloc[sl]
    h  = high.iloc[sl]
    lo = low.iloc[sl]
    v  = volume.iloc[sl]

    spread = h - lo
    # Close position 0 = bar low, 1 = bar high; default to 0.5 for zero-spread bars
    close_pos = ((c - lo) / spread.replace(0, np.nan)).fillna(0.5)

    # Weighted volume: weight > 0.5 means bullish pressure
    bull_vol = (v * close_pos).sum()
    bear_vol = (v * (1 - close_pos)).sum()
    total = bull_vol + bear_vol

    if total == 0:
        return "neutral"

    bull_ratio = bull_vol / total
    if bull_ratio > 0.55:   # >55% weighted toward highs = accumulation
        return "accumulation"
    elif bull_ratio < 0.45: # <45% = distribution
        return "distribution"
    return "neutral"


def identify_wyckoff_events(
    close: pd.Series, high: pd.Series, low: pd.Series,
    volume: pd.Series, tr: TradingRange, phase_type: str,
) -> list[WyckoffEvent]:
    """Detect Wyckoff events within a trading range."""
    sl = slice(tr.start_idx, tr.end_idx + 1)
    c = close.iloc[sl]
    h = high.iloc[sl]
    lo = low.iloc[sl]
    v = volume.iloc[sl]
    avg_vol = v.mean()
    range_height = tr.upper_bound - tr.lower_bound

    events: list[WyckoffEvent] = []

    if phase_type == "accumulation":
        _detect_accumulation_events(c, h, lo, v, avg_vol, range_height, tr, events)
    elif phase_type == "distribution":
        _detect_distribution_events(c, h, lo, v, avg_vol, range_height, tr, events)

    return events


def _detect_accumulation_events(
    c: pd.Series, h: pd.Series, lo: pd.Series, v: pd.Series,
    avg_vol: float, range_height: float, tr: TradingRange,
    events: list[WyckoffEvent],
) -> None:
    if len(c) < 3:
        return

    # SC: bar in the bottom 15% of the range with volume > 1.5× avg
    # (Avoids picking any low as SC — requires climactic volume)
    bottom_15 = tr.lower_bound + range_height * 0.15
    sc_idx = None
    sc_price = None
    for idx in lo.index:
        if float(lo[idx]) <= bottom_15 and float(v.get(idx, 0)) > avg_vol * 1.5:
            if sc_idx is None or float(lo[idx]) < sc_price:
                sc_idx = idx
                sc_price = float(lo[idx])
    if sc_idx is not None:
        events.append(WyckoffEvent(sc_idx, sc_price, "SC", "Selling Climax — high-volume capitulation low"))

    # AR: First significant high after SC (rise > 0.5 × range height)
    if sc_idx is not None:
        sc_pos = c.index.get_loc(sc_idx) if sc_idx in c.index else 0
        for i in range(sc_pos + 1, len(h)):
            if float(h.iloc[i]) - sc_price > 0.5 * range_height:
                events.append(WyckoffEvent(h.index[i], float(h.iloc[i]), "AR", "Automatic Rally — sharp bounce after SC"))
                break

    # ST: Low near SC price (±2%) on lower volume
    if sc_idx is not None:
        sc_pos = c.index.get_loc(sc_idx) if sc_idx in c.index else 0
        for i in range(sc_pos + 2, len(lo)):
            if abs(float(lo.iloc[i]) - sc_price) / max(sc_price, 1e-9) < 0.02:
                if float(v.iloc[i]) < avg_vol:
                    events.append(WyckoffEvent(lo.index[i], float(lo.iloc[i]), "ST", "Secondary Test — retest of SC on lower volume"))
                    break

    # Spring: Close below support then recovers within 3 bars;
    # confirmed by a subsequent low-volume test within 5 bars
    support = tr.lower_bound
    for i in range(len(c) - 3):
        if float(c.iloc[i]) < support:
            recovered = any(float(c.iloc[i + j]) > support for j in range(1, min(4, len(c) - i)))
            if recovered:
                # Look for low-volume test bar within the next 5 bars
                test_confirmed = any(
                    float(v.iloc[i + k]) < avg_vol * 0.7
                    and float(lo.iloc[i + k]) > float(lo.iloc[i]) * 0.99
                    for k in range(1, min(6, len(c) - i))
                )
                desc = (
                    "Spring (tested) — false break below support + low-vol test confirmed"
                    if test_confirmed
                    else "Spring — false break below support with recovery"
                )
                events.append(WyckoffEvent(c.index[i], float(c.iloc[i]), "Spring", desc))
                break

    # SOS: Close above resistance on above-avg volume
    resistance = tr.upper_bound
    for i in range(len(c)):
        if float(c.iloc[i]) > resistance and float(v.iloc[i]) > avg_vol:
            events.append(WyckoffEvent(c.index[i], float(c.iloc[i]), "SOS", "Sign of Strength — breakout above resistance on volume"))
            break


def _detect_distribution_events(
    c: pd.Series, h: pd.Series, lo: pd.Series, v: pd.Series,
    avg_vol: float, range_height: float, tr: TradingRange,
    events: list[WyckoffEvent],
) -> None:
    if len(c) < 3:
        return

    # BC: Bar in the top 15% of the range with volume > 1.5× avg
    # (Avoids picking any high as BC — requires climactic volume)
    top_15 = tr.upper_bound - range_height * 0.15
    bc_idx = None
    bc_price = None
    for idx in h.index:
        if float(h[idx]) >= top_15 and float(v.get(idx, 0)) > avg_vol * 1.5:
            if bc_idx is None or float(h[idx]) > bc_price:
                bc_idx = idx
                bc_price = float(h[idx])
    if bc_idx is not None:
        events.append(WyckoffEvent(bc_idx, bc_price, "BC", "Buying Climax — high-volume euphoria high"))

    # AR: First significant low after BC
    if bc_idx is not None:
        bc_pos = c.index.get_loc(bc_idx) if bc_idx in c.index else 0
        for i in range(bc_pos + 1, len(lo)):
            if bc_price - float(lo.iloc[i]) > 0.5 * range_height:
                events.append(WyckoffEvent(lo.index[i], float(lo.iloc[i]), "AR", "Automatic Reaction — sharp drop after BC"))
                break

    # ST: High near BC price (±2%) on lower volume
    if bc_idx is not None:
        bc_pos = c.index.get_loc(bc_idx) if bc_idx in c.index else 0
        for i in range(bc_pos + 2, len(h)):
            if abs(float(h.iloc[i]) - bc_price) / max(bc_price, 1e-9) < 0.02:
                if float(v.iloc[i]) < avg_vol:
                    events.append(WyckoffEvent(h.index[i], float(h.iloc[i]), "ST", "Secondary Test — retest of BC on lower volume"))
                    break

    # UTAD: Close above resistance then failure within 3 bars
    resistance = tr.upper_bound
    for i in range(len(c) - 3):
        if float(c.iloc[i]) > resistance:
            failed = any(float(c.iloc[i + j]) < resistance for j in range(1, min(4, len(c) - i)))
            if failed:
                events.append(WyckoffEvent(c.index[i], float(c.iloc[i]), "UTAD", "Upthrust After Distribution — false breakout above resistance"))
                break

    # SOW: Close below support on above-avg volume
    support = tr.lower_bound
    for i in range(len(c)):
        if float(c.iloc[i]) < support and float(v.iloc[i]) > avg_vol:
            events.append(WyckoffEvent(c.index[i], float(c.iloc[i]), "SOW", "Sign of Weakness — breakdown below support on volume"))
            break


def analyze_vsa(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    lookback: int = 20,
) -> list[VSABar]:
    """
    Volume Spread Analysis — checks spread (high-low) vs volume vs close position.
    Only returns the most recent `lookback` bars worth of signals.
    """
    bars = []
    vol_ma = volume.rolling(20).mean()
    spread = high - low
    spread_ma = spread.rolling(20).mean()

    recent = close.index[-lookback:]

    for i in range(20, len(close)):
        date = close.index[i]
        if date not in recent:
            continue

        c = float(close.iloc[i])
        h = float(high.iloc[i])
        lo = float(low.iloc[i])
        v = float(volume.iloc[i])
        vm = float(vol_ma.iloc[i]) if not pd.isna(vol_ma.iloc[i]) else v
        sp = h - lo
        sp_ma = float(spread_ma.iloc[i]) if not pd.isna(spread_ma.iloc[i]) else sp

        high_vol = v > vm * 1.5
        low_vol = v < vm * 0.7
        wide_spread = sp > sp_ma * 1.3
        narrow_spread = sp < sp_ma * 0.7
        close_upper = (c - lo) / sp > 0.7 if sp > 0 else False
        close_lower = (c - lo) / sp < 0.3 if sp > 0 else False

        if high_vol and wide_spread and close_upper:
            bars.append(VSABar(date, "Strength", "Wide spread up bar on high volume — demand entering", 3))
        elif high_vol and wide_spread and close_lower:
            bars.append(VSABar(date, "Weakness", "Wide spread down bar on high volume — supply entering", 3))
        elif high_vol and narrow_spread:
            bars.append(VSABar(date, "Effort No Result", "High volume but narrow spread — absorption/reversal warning", 2))
        elif low_vol and narrow_spread and close_upper:
            bars.append(VSABar(date, "No Supply", "Low volume narrow spread closing near high — no selling pressure", 2))
        elif low_vol and narrow_spread and close_lower:
            bars.append(VSABar(date, "No Demand", "Low volume narrow spread closing near low — no buying interest", 2))

    return bars


def detect_effort_vs_result(
    close: pd.Series, volume: pd.Series, lookback: int = 60,
) -> list[tuple]:
    """
    Detect divergences where effort (volume) doesn't match result (price move).
    Returns list of (date, description) tuples for the most significant divergences.
    """
    divergences = []
    vol_ma = volume.rolling(10).mean()
    price_change = close.pct_change().abs()
    price_change_ma = price_change.rolling(10).mean()

    recent_start = len(close) - lookback

    for i in range(max(10, recent_start), len(close)):
        v = float(volume.iloc[i])
        vm = float(vol_ma.iloc[i]) if not pd.isna(vol_ma.iloc[i]) else v
        pc = float(price_change.iloc[i])
        pcm = float(price_change_ma.iloc[i]) if not pd.isna(price_change_ma.iloc[i]) else pc

        if vm == 0 or pcm == 0:
            continue

        high_effort = v > vm * 2.0
        low_result = pc < pcm * 0.4

        if high_effort and low_result:
            divergences.append((
                close.index[i],
                f"High effort (vol {v/vm:.1f}×avg) but tiny price move ({pc*100:.2f}%) — absorption likely"
            ))

    return divergences[-5:]  # return last 5 most recent


def detect_sub_phase(
    phase: WyckoffPhase, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
) -> str:
    """
    Determine which Wyckoff sub-phase (A-E) the current range is in.

    Accumulation:
      A = Stopping the downtrend (SC, AR detected)
      B = Building the cause (ST, tests of support)
      C = Testing (Spring or Shakeout)
      D = Dominance of demand (SOS, LPS)
      E = Markup beginning (price leaves range)

    Distribution:
      A = Stopping the uptrend (BC, AR)
      B = Building the cause (ST, tests of resistance)
      C = Testing supply (UTAD)
      D = Dominance of supply (SOW, LPSY)
      E = Markdown beginning
    """
    events = [e.event_type for e in phase.events]
    is_accum = phase.phase == "Accumulation"

    if is_accum:
        if "SOS" in events:
            return "D/E"
        if "Spring" in events:
            return "C"
        if "ST" in events:
            return "B"
        if "SC" in events or "AR" in events:
            return "A"
        return "A"
    else:  # Distribution
        if "SOW" in events:
            return "D/E"
        if "UTAD" in events:
            return "C"
        if "ST" in events:
            return "B"
        if "BC" in events or "AR" in events:
            return "A"
        return "A"


def calculate_cause_target(phase: WyckoffPhase, current_price: float) -> float | None:
    """
    Wyckoff Cause & Effect: width of trading range × a duration-scaled multiplier
    estimates the price target (simplified horizontal point count method).

    Longer-duration causes justify proportionally larger price targets.
    Multiplier = 1.5 × sqrt(range_bars / 30), capped at 3×.
    """
    support = phase.key_levels["support"]
    resistance = phase.key_levels["resistance"]
    width = resistance - support
    if width <= 0:
        return None

    # Estimate bars in cause using date range (approx trading days)
    range_bars = max(
        int((phase.end_date - phase.start_date).days * 5 / 7), 1
    )
    # Scale: longer cause → larger target, capped at 3× the range width
    scale = min(1.5 * (range_bars / 30) ** 0.5, 3.0)

    if phase.phase == "Accumulation":
        return resistance + width * scale
    elif phase.phase == "Distribution":
        return support - width * scale
    return None


def calculate_demand_supply_lines(
    phase: WyckoffPhase, close: pd.Series, high: pd.Series, low: pd.Series,
) -> tuple[tuple | None, tuple | None]:
    """
    Demand line: connects the two most significant lows within accumulation range.
    Supply line: connects the two most significant highs within distribution range.
    Returns (demand_line, supply_line) where each is (date1, price1, date2, price2) or None.
    """
    # Find the slice for this phase
    mask = (close.index >= phase.start_date) & (close.index <= phase.end_date)
    c_slice = close[mask]
    h_slice = high[mask]
    l_slice = low[mask]

    if len(c_slice) < 10:
        return None, None

    demand_line = None
    supply_line = None

    if phase.phase in ("Accumulation", "Markup"):
        # Find two lowest lows
        lows_sorted = l_slice.nsmallest(2)
        if len(lows_sorted) == 2:
            d1, d2 = sorted(lows_sorted.index)
            demand_line = (d1, float(l_slice[d1]), d2, float(l_slice[d2]))

    if phase.phase in ("Distribution", "Markdown"):
        # Find two highest highs
        highs_sorted = h_slice.nlargest(2)
        if len(highs_sorted) == 2:
            d1, d2 = sorted(highs_sorted.index)
            supply_line = (d1, float(h_slice[d1]), d2, float(h_slice[d2]))

    return demand_line, supply_line


def _classify_trend_phase(
    close: pd.Series, start_idx: int, end_idx: int,
) -> str:
    """Classify inter-range gap as Markup or Markdown via regression slope."""
    if end_idx <= start_idx:
        return "Markup"
    segment = close.iloc[start_idx:end_idx + 1]
    if len(segment) < 3:
        return "Markup"
    slope = np.polyfit(range(len(segment)), segment.values, 1)[0]
    return "Markup" if slope >= 0 else "Markdown"


def determine_phases(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    atr_window: int = 50, min_bars: int = 10,
) -> list[WyckoffPhase]:
    """Orchestrate: ranges → volume → events → phases with confidence scoring."""
    ranges = detect_trading_ranges(close, high, low, atr_window=atr_window, min_bars=min_bars)
    phases: list[WyckoffPhase] = []

    for i, tr in enumerate(ranges):
        vol_behavior = classify_volume_behavior(close, high, low, volume, tr)

        if vol_behavior == "accumulation":
            phase_name = "Accumulation"
        elif vol_behavior == "distribution":
            phase_name = "Distribution"
        elif tr.preceding_trend == "down":
            phase_name = "Accumulation"
        elif tr.preceding_trend == "up":
            phase_name = "Distribution"
        else:
            phase_name = "Accumulation"

        event_type = "accumulation" if phase_name == "Accumulation" else "distribution"
        events = identify_wyckoff_events(close, high, low, volume, tr, event_type)

        # Confidence: base 40 + 10 per event (max 50) + 10 for volume confirm
        confidence = 40
        confidence += min(len(events) * 10, 50)
        if vol_behavior in ("accumulation", "distribution"):
            confidence += 10
        confidence = min(confidence, 100)

        phases.append(WyckoffPhase(
            phase=phase_name,
            confidence=confidence,
            start_date=tr.start_date,
            end_date=tr.end_date,
            key_levels={"support": tr.lower_bound, "resistance": tr.upper_bound},
            events=events,
        ))

        # Sub-phase
        sub = detect_sub_phase(phases[-1], close, high, low, volume)
        phases[-1].sub_phase = sub

        # Cause & Effect target
        phases[-1].cause_target = calculate_cause_target(phases[-1], float(close.iloc[tr.end_idx]))

        # Demand/Supply lines
        demand, supply = calculate_demand_supply_lines(phases[-1], close, high, low)
        phases[-1].demand_line = demand
        phases[-1].supply_line = supply

        # Add trend phase between ranges
        if i < len(ranges) - 1:
            next_tr = ranges[i + 1]
            gap_start = tr.end_idx + 1
            gap_end = next_tr.start_idx - 1
            if gap_end > gap_start:
                trend_name = _classify_trend_phase(close, gap_start, gap_end)
                # Trend confidence
                segment = close.iloc[gap_start:gap_end + 1]
                t_confidence = 50
                if len(segment) >= 5:
                    diffs = segment.diff().dropna()
                    consistency = (diffs > 0).mean() if trend_name == "Markup" else (diffs < 0).mean()
                    if consistency > 0.6:
                        t_confidence += 25
                    vol_seg = volume.iloc[gap_start:gap_end + 1]
                    vol_trend = vol_seg.rolling(5).mean()
                    if len(vol_trend.dropna()) > 2:
                        vol_slope = np.polyfit(range(len(vol_trend.dropna())), vol_trend.dropna().values, 1)[0]
                        if (trend_name == "Markup" and vol_slope > 0) or (trend_name == "Markdown" and vol_slope > 0):
                            t_confidence += 25
                t_confidence = min(t_confidence, 100)

                phases.append(WyckoffPhase(
                    phase=trend_name,
                    confidence=t_confidence,
                    start_date=close.index[gap_start],
                    end_date=close.index[gap_end],
                    key_levels={"support": float(low.iloc[gap_start:gap_end + 1].min()),
                                "resistance": float(high.iloc[gap_start:gap_end + 1].max())},
                ))

    # Sort by start_date
    phases.sort(key=lambda p: p.start_date)
    return phases


def analyze_wyckoff(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    interval: str = "1d",
) -> WyckoffAnalysis | None:
    """Public entry point. Returns WyckoffAnalysis or None if insufficient data.

    Pass *interval* (e.g. '15m', '1h', '1d') so ATR window and minimum
    consolidation length adapt to the selected timeframe granularity.
    """
    atr_window, min_bars = _interval_params(interval)
    min_len = max(atr_window + 20, 60)
    if len(close) < min_len:
        return None

    phases = determine_phases(close, high, low, volume, atr_window=atr_window, min_bars=min_bars)
    if not phases:
        return None

    phase_history = [(p.start_date, p.end_date, p.phase) for p in phases]
    vsa_bars = analyze_vsa(close, high, low, volume, lookback=30)
    effort_vs_result = detect_effort_vs_result(close, volume, lookback=60)

    return WyckoffAnalysis(
        current_phase=phases[-1],
        all_phases=phases,
        phase_history=phase_history,
        vsa_bars=vsa_bars,
        effort_vs_result=effort_vs_result,
    )
