"""One-time backfill script: populates data/tactical_score_history.json with ~1 year of
historical tactical scores computed from yfinance price data.

Signals 1-7 are computed faithfully from historical prices.
Signals 8 (Fear & Greed) and 9 (AAII) default to neutral (0.0) — no free historical API.
Their combined weight is ~15% of the total, so scores are close approximations.

Run from the project root:
    python tools/backfill_tactical_history.py
"""
import json
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tactical_history import _HISTORY_PATH, _MAX_DAYS, load_history

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 365          # how many calendar days back to backfill
TICKERS = ["^VIX", "^VIX3M", "^VIX9D", "^VIX6M", "SPY", "RSP", "^SKEW"]
WEIGHTS = [2.0, 2.0, 1.5, 1.5, 1.0, 1.5, 1.0, 1.0, 0.8]


# ── Helpers (mirrors modules/risk_regime.py) ──────────────────────────────────

def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _clamp_score(value: float, scale: float) -> float:
    return _clamp(value / max(scale, 1e-6))


def _zscore_score(series: pd.Series, invert: bool = False, lookback: int = 252) -> float:
    clean = series.dropna()
    n = min(lookback, len(clean))
    if n < 20:
        return 0.0
    window = clean.iloc[-n:]
    std = float(window.std())
    if std < 1e-9:
        return 0.0
    z = (float(clean.iloc[-1]) - float(window.mean())) / std
    if invert:
        z = -z
    return _clamp(z / 2.0)


def _label_from_score(score: int, p10: int, p35: int, p65: int) -> str:
    if score >= p65:
        return "Favorable Entry"
    elif score >= p35:
        return "Neutral / Hold"
    elif score >= p10:
        return "Caution / Reduce"
    return "Risk-Off Signal"


# ── Signal computation for a single day ──────────────────────────────────────

def _compute_day_score(i: int, data: dict[str, pd.Series]) -> int | None:
    """Compute tactical score. data values are pre-sliced up to the target date.
    Parameter i is unused (kept for compatibility) — function uses full series as-is.
    """

    def get(ticker: str) -> pd.Series | None:
        s = data.get(ticker)
        if s is None or len(s) == 0:
            return None
        return s

    # ── Signal 1: VIX Level + 5d Trend ───────────────────────────────────────
    vix_s = get("^VIX")
    if vix_s is None or len(vix_s) < 6:
        return None  # can't compute without VIX
    vix_level = float(vix_s.iloc[-1])

    if len(vix_s) >= 20:
        vix_level_score = _zscore_score(vix_s, invert=True)
    else:
        vix_level_score = _clamp_score(20.0 - vix_level, 10.0)

    vix_5d_chg = float(vix_s.iloc[-1] - vix_s.iloc[-6])
    vix_trend_score = _clamp(-vix_5d_chg / 3.0)
    sig1 = _clamp(vix_level_score * 0.6 + vix_trend_score * 0.4)

    # ── Signal 2: VIX Term Structure (VIX/VIX3M) ─────────────────────────────
    vix3m_s = get("^VIX3M")
    sig2 = 0.0
    if vix3m_s is not None and len(vix3m_s) >= 1:
        vix3m_level = float(vix3m_s.iloc[-1])
        if vix3m_level > 0:
            ts_ratio = vix_level / vix3m_level
            sig2 = _clamp((1.0 - ts_ratio) / 0.15)

    # ── Signal 3: SPY vs 20d/50d MA + slope ──────────────────────────────────
    spy_s = get("SPY")
    sig3 = 0.0
    if spy_s is not None and len(spy_s) >= 50:
        spy_price = float(spy_s.iloc[-1])
        ma20 = float(spy_s.tail(20).mean())
        ma50 = float(spy_s.tail(50).mean())
        pct20 = (spy_price / ma20 - 1.0) * 100
        pct50 = (spy_price / ma50 - 1.0) * 100
        ma20_5d_ago = float(spy_s.iloc[-6:-1].mean()) if len(spy_s) >= 6 else ma20
        slope_pct = (ma20 - ma20_5d_ago) / ma20_5d_ago * 100 if ma20_5d_ago else 0.0
        ma_score = _clamp_score(pct20 * 0.5 + pct50 * 0.5, 4.0)
        slope_score = _clamp(slope_pct / 0.5)
        sig3 = _clamp(ma_score * 0.7 + slope_score * 0.3)

    # ── Signal 4: Short-term Momentum ────────────────────────────────────────
    sig4 = 0.0
    if spy_s is not None and len(spy_s) >= 21:
        roc5 = float((spy_s.iloc[-1] / spy_s.iloc[-6] - 1) * 100)
        roc20 = float((spy_s.iloc[-1] / spy_s.iloc[-21] - 1) * 100)
        accel = roc5 - (roc20 / 4)
        mom_score = _clamp_score(roc5, 3.0)
        accel_score = _clamp(accel / 2.0)
        sig4 = _clamp(mom_score * 0.6 + accel_score * 0.4)

    # ── Signal 5: Breadth Trend (RSP/SPY 5d ratio change) ────────────────────
    rsp_s = get("RSP")
    sig5 = 0.0
    if rsp_s is not None and spy_s is not None and len(rsp_s) >= 6 and len(spy_s) >= 6:
        brd = pd.DataFrame({"rsp": rsp_s, "spy": spy_s}).dropna()
        if len(brd) >= 6:
            ratio = brd["rsp"] / brd["spy"]
            ratio_5d_chg = float((ratio.iloc[-1] / ratio.iloc[-6] - 1) * 100)
            sig5 = _clamp(ratio_5d_chg / 1.0)

    # ── Signal 6: VIX Full Curve ──────────────────────────────────────────────
    vix9d_s = get("^VIX9D")
    vix6m_s = get("^VIX6M")
    sig6 = 0.0
    vix9d_level = float(vix9d_s.iloc[-1]) if vix9d_s is not None and len(vix9d_s) >= 1 else None
    vix6m_level = float(vix6m_s.iloc[-1]) if vix6m_s is not None and len(vix6m_s) >= 1 else None
    vix3m_level_cur = float(get("^VIX3M").iloc[-1]) if (get("^VIX3M") is not None and len(get("^VIX3M")) >= 1) else None

    if vix9d_level and vix3m_level_cur and vix6m_level:
        short_inv = vix9d_level / vix_level
        mid_inv = vix_level / vix3m_level_cur
        long_inv = vix3m_level_cur / vix6m_level
        contango_avg = (short_inv + mid_inv + long_inv) / 3.0
        sig6 = _clamp((1.0 - contango_avg) / 0.15)
    elif vix9d_level:
        ratio9d = vix9d_level / vix_level
        sig6 = _clamp((1.0 - ratio9d) / 0.15)

    # ── Signal 7: CBOE SKEW ───────────────────────────────────────────────────
    skew_s = get("^SKEW")
    sig7 = 0.0
    if skew_s is not None and len(skew_s) >= 1:
        skew_val = float(skew_s.iloc[-1])
        sig7 = _clamp((120.0 - skew_val) / 20.0)

    # ── Signals 8 & 9: Fear & Greed + AAII — no historical data → neutral ─────
    sig8 = 0.0
    sig9 = 0.0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scores = [sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8, sig9]
    agg = float(np.average(scores, weights=WEIGHTS))
    return int(round((agg + 1.0) * 50))


def _write_historical_entry(day_str: str, score: int, label: str, existing: list[dict]) -> None:
    """Write a historical score entry directly, preserving the correct date."""
    existing.append({
        "date":      day_str,
        "score":     score,
        "label":     label,
        "logged_at": datetime.utcnow().isoformat(),
        "backfilled": True,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Tactical Score Backfill")
    print("=" * 60)

    # Check what dates already exist
    existing = load_history()
    existing_dates = {e["date"] for e in existing}
    print(f"Existing entries: {len(existing_dates)}")

    # Download historical data
    print(f"\nDownloading {LOOKBACK_DAYS} days of price data for: {', '.join(TICKERS)}")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    start = date.today() - timedelta(days=LOOKBACK_DAYS + 100)  # extra buffer for warmup

    # Download each ticker individually (batch download can hang for some tickers)
    raw_series: dict[str, pd.Series] = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=start.isoformat(), progress=False, auto_adjust=True)
            if not df.empty:
                close_col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                raw_series[ticker] = close_col.squeeze().dropna()
                print(f"  {ticker}: {len(raw_series[ticker])} bars")
            else:
                print(f"  {ticker}: no data")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")

    if "^VIX" not in raw_series or "SPY" not in raw_series:
        print("ERROR: Missing critical tickers (^VIX, SPY). Cannot proceed.")
        sys.exit(1)

    data = raw_series

    # Build aligned trading day index from SPY (most complete)
    spy_index = data["SPY"].index
    trading_days = spy_index
    print(f"Data spans {spy_index[0].date()} → {spy_index[-1].date()} ({len(spy_index)} trading days)")

    # Static thresholds for labeling during backfill (percentile thresholds don't exist yet)
    P10, P35, P65 = 38, 52, 65

    # Compute scores for each trading day (need at least 50 bars warmup for 50d MA)
    WARMUP = 50
    computed = 0
    skipped_warmup = 0
    skipped_existing = 0

    print(f"\nComputing scores (skipping {WARMUP} warmup bars)...")

    entries = list(existing)  # start with what's already on disk
    for i in range(len(trading_days)):
        day_str = str(trading_days[i].date())

        if i < WARMUP:
            skipped_warmup += 1
            continue

        if day_str in existing_dates:
            skipped_existing += 1
            continue

        # Build per-ticker slices up to day i
        day_data: dict[str, pd.Series] = {}
        for ticker, s in data.items():
            s_slice = s[s.index <= trading_days[i]]
            if len(s_slice) > 0:
                day_data[ticker] = s_slice

        vix_slice = day_data.get("^VIX")
        if vix_slice is None or len(vix_slice) == 0:
            continue
        score = _compute_day_score(len(vix_slice) - 1, day_data)
        if score is None:
            continue

        label = _label_from_score(score, P10, P35, P65)
        _write_historical_entry(day_str, score, label, entries)
        computed += 1

    # ── Write all entries to disk at once ─────────────────────────────────────
    if computed > 0:
        cutoff = str(date.today() - timedelta(days=_MAX_DAYS))
        entries = [e for e in entries if e.get("date", "") >= cutoff]
        entries.sort(key=lambda x: x.get("date", ""))
        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    print(f"\nResults:")
    print(f"  Warmup bars skipped:   {skipped_warmup}")
    print(f"  Already existing:      {skipped_existing}")
    print(f"  Scores computed:       {computed}")

    # Check final state
    final = load_history()
    print(f"  Total history entries: {len(final)}")
    print("\nDone! Dynamic percentile thresholds will now activate on next app load.")


if __name__ == "__main__":
    main()
