"""Free public data sources — Fear & Greed, AAII sentiment, FedSpeak RSS, VIX term structure.

All functions are safe to call from background threads (no st.* calls).
Each returns a plain dict or None on failure.
"""

import requests
import feedparser
from datetime import datetime, timezone


# ── Fear & Greed Index ────────────────────────────────────────────────────────

def fetch_fear_greed() -> dict | None:
    """Fetch Fear & Greed Index.

    Primary: alternative.me/fng (free, updated daily, crypto-market correlated)
    Fallback: reconstruct from yfinance (VIX, SPY momentum, HYG/LQD spread)

    Returns dict with:
        score      int    0-100
        label      str    "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
        change_7d  int    score change vs 7 days ago
        source     str    "alternative.me" or "synthetic"
        fetched_at str    ISO timestamp
    """
    def _label(s: int) -> str:
        if s <= 25:  return "Extreme Fear"
        if s <= 45:  return "Fear"
        if s <= 55:  return "Neutral"
        if s <= 75:  return "Greed"
        return "Extreme Greed"

    # Primary: alternative.me (free, no key needed)
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=8",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            current = int(data[0]["value"])
            week_ago = int(data[7]["value"]) if len(data) >= 8 else current
            return {
                "score":      current,
                "label":      _label(current),
                "change_7d":  current - week_ago,
                "source":     "alternative.me",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        pass

    # Fallback: synthetic from yfinance signals
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        raw = yf.download(["^VIX", "SPY", "HYG", "LQD", "RSP"],
                          period="60d", interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        scores = []
        # VIX component (inverted): VIX 12=100, VIX 40=0
        if "^VIX" in close.columns:
            vix = float(close["^VIX"].dropna().iloc[-1])
            scores.append(max(0, min(100, int((40 - vix) / 28 * 100))))
        # SPY momentum vs 50d MA
        if "SPY" in close.columns:
            spy = close["SPY"].dropna()
            if len(spy) >= 50:
                pct = (float(spy.iloc[-1]) / float(spy.tail(50).mean()) - 1) * 100
                scores.append(max(0, min(100, int(50 + pct * 5))))
        # HYG/LQD spread proxy (inverted — high ratio = greed)
        if "HYG" in close.columns and "LQD" in close.columns:
            hyg = close["HYG"].dropna(); lqd = close["LQD"].dropna()
            if len(hyg) >= 20 and len(lqd) >= 20:
                ratio = float(hyg.iloc[-1]) / float(lqd.iloc[-1])
                mean_r = float((hyg.tail(20) / lqd.tail(20)).mean())
                spread_score = max(0, min(100, int(50 + (ratio - mean_r) / mean_r * 2000)))
                scores.append(spread_score)
        # Breadth RSP/SPY
        if "RSP" in close.columns and "SPY" in close.columns:
            rsp = close["RSP"].dropna(); spy = close["SPY"].dropna()
            if len(rsp) >= 20 and len(spy) >= 20:
                ratio = float(rsp.iloc[-1]) / float(spy.iloc[-1])
                mean_r = float((rsp.tail(20) / spy.tail(20)).mean())
                brd_score = max(0, min(100, int(50 + (ratio - mean_r) / mean_r * 1000)))
                scores.append(brd_score)

        if not scores:
            return None
        synth = int(round(sum(scores) / len(scores)))
        return {
            "score":      synth,
            "label":      _label(synth),
            "change_7d":  0,
            "source":     "synthetic",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


# ── AAII Sentiment Survey ─────────────────────────────────────────────────────

def fetch_aaii_sentiment() -> dict | None:
    """Fetch AAII individual investor sentiment survey (weekly).

    Returns dict with:
        bull_pct     float   % bullish
        neutral_pct  float   % neutral
        bear_pct     float   % bearish
        bull_bear_spread float  bull - bear
        label        str    "Bullish" / "Neutral" / "Bearish"
        fetched_at   str    ISO timestamp
    """
    try:
        url = "https://www.aaii.com/sentimentsurvey/sent_results"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        import re
        text = resp.text

        # AAII table has three consecutive percentage values per row
        pattern = r'(\d{1,2}\.\d)%\s*</td>\s*<td[^>]*>\s*(\d{1,2}\.\d)%\s*</td>\s*<td[^>]*>\s*(\d{1,2}\.\d)%'
        match = re.search(pattern, text)
        if match:
            bull    = float(match.group(1))
            neutral = float(match.group(2))
            bear    = float(match.group(3))
        else:
            pcts = re.findall(r'(\d{1,2}\.\d)%', text)
            if len(pcts) >= 3:
                bull, neutral, bear = float(pcts[0]), float(pcts[1]), float(pcts[2])
            else:
                return None

        spread = round(bull - bear, 1)
        label = "Bullish" if spread > 10 else ("Bearish" if spread < -10 else "Neutral")

        return {
            "bull_pct":         bull,
            "neutral_pct":      neutral,
            "bear_pct":         bear,
            "bull_bear_spread": spread,
            "label":            label,
            "fetched_at":       datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


# ── FedSpeak RSS ──────────────────────────────────────────────────────────────

def fetch_fedspeech_rss(max_items: int = 8) -> list[dict]:
    """Fetch recent Fed governor speeches from the Federal Reserve RSS feed.

    Returns list of dicts: {title, link, published, summary, speaker}
    """
    try:
        url = "https://www.federalreserve.gov/feeds/press_all.xml"
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items * 3]:
            title = entry.get("title", "")
            title_lower = title.lower()
            if not any(kw in title_lower for kw in ["speech", "remark", "testimony", "statement", "address"]):
                continue
            import re
            m = re.search(r'(Governor|Chair|President|Vice Chair)\s+\w+', title, re.I)
            speaker = m.group(0)[:60] if m else ""
            items.append({
                "title":     title[:120],
                "link":      entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary":   (entry.get("summary", "") or "")[:300],
                "speaker":   speaker,
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


# ── VIX Term Structure (full curve: VIX9D / VIX / VIX3M / VIX6M) ─────────────

def fetch_vix_term_structure() -> dict | None:
    """Fetch the full VIX term structure from yfinance.

    Returns dict with:
        vix9d              float | None
        vix                float | None
        vix3m              float | None
        vix6m              float | None
        contango_9d_3m     float | None   VIX9D / VIX3M
        contango_spot_3m   float | None   VIX / VIX3M
        structure          str   "Deep Contango" / "Contango" / "Flat" / "Backwardation" / "Deep Backwardation"
        fetched_at         str
    """
    try:
        import yfinance as yf
        import pandas as pd

        tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
        data = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None

        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data

        def _last(t):
            if t in close.columns:
                s = close[t].dropna()
                return round(float(s.iloc[-1]), 2) if len(s) else None
            return None

        vix9d = _last("^VIX9D")
        vix   = _last("^VIX")
        vix3m = _last("^VIX3M")
        vix6m = _last("^VIX6M")

        ratio_9d_3m   = round(vix9d / vix3m, 3) if (vix9d and vix3m) else None
        ratio_spot_3m = round(vix   / vix3m, 3) if (vix   and vix3m) else None

        r = ratio_spot_3m
        if r is None:
            structure = "Unknown"
        elif r < 0.80:
            structure = "Deep Contango"
        elif r < 0.95:
            structure = "Contango"
        elif r < 1.05:
            structure = "Flat"
        elif r < 1.20:
            structure = "Backwardation"
        else:
            structure = "Deep Backwardation"

        return {
            "vix9d":            vix9d,
            "vix":              vix,
            "vix3m":            vix3m,
            "vix6m":            vix6m,
            "contango_9d_3m":   ratio_9d_3m,
            "contango_spot_3m": ratio_spot_3m,
            "structure":        structure,
            "fetched_at":       datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


# ── Combined sentiment snapshot ───────────────────────────────────────────────

def fetch_sentiment_snapshot() -> dict:
    """Fetch all free sentiment signals. Returns combined dict.

    Keys: fear_greed, aaii, vix_curve, fedspeech
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        fg_fut   = pool.submit(fetch_fear_greed)
        aaii_fut = pool.submit(fetch_aaii_sentiment)
        vix_fut  = pool.submit(fetch_vix_term_structure)
        fed_fut  = pool.submit(fetch_fedspeech_rss)
    return {
        "fear_greed": fg_fut.result(),
        "aaii":       aaii_fut.result(),
        "vix_curve":  vix_fut.result(),
        "fedspeech":  fed_fut.result(),
    }

