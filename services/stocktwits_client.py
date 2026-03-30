"""StockTwits enterprise API client.

Base URL: https://api-gw-prd.stocktwits.com
Auth: HTTP Basic (username:password)
Credentials: contact enterprise-support@stocktwits.com

Set in .env:
  STOCKTWITS_USERNAME=your_username
  STOCKTWITS_PASSWORD=your_password

Key functions:
  get_trending_symbols()        → top trending tickers with scores + price data
  get_symbol_sentiment()        → 0-100 sentiment score + label for a ticker
  get_why_trending()            → AI-generated "why it's trending" summary
  run_quick_stocktwits()        → background-safe QIR helper, returns digest dict
"""

import os
import base64
import requests

_BASE = "https://api-gw-prd.stocktwits.com"
_TIMEOUT = 12

_USERNAME = os.getenv("STOCKTWITS_USERNAME", "").strip()
_PASSWORD = os.getenv("STOCKTWITS_PASSWORD", "").strip()


def _headers() -> dict:
    h = {"User-Agent": "NarrativeInvestingTool/1.0"}
    if _USERNAME and _PASSWORD:
        creds = base64.b64encode(f"{_USERNAME}:{_PASSWORD}".encode()).decode()
        h["Authorization"] = f"Basic {creds}"
    return h


def _check_creds():
    if not (_USERNAME and _PASSWORD):
        raise RuntimeError(
            "STOCKTWITS_USERNAME / STOCKTWITS_PASSWORD not set — "
            "contact enterprise-support@stocktwits.com to get credentials, "
            "then add both to your .env"
        )


def get_trending_symbols(limit: int = 20, region: str = "US") -> list[dict]:
    """Fetch top trending equities from StockTwits.

    Returns list of {symbol, title, trending_score, trending_summary,
                     price, change_pct, watchlist_count}.
    """
    _check_creds()
    resp = requests.get(
        f"{_BASE}/api-middleware/external/api/2/trending/symbols_enhanced.json",
        headers=_headers(),
        params={"limit": limit, "class": "equities", "regions": region, "payloads": "prices"},
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise RuntimeError(f"StockTwits trending HTTP {resp.status_code}")

    symbols = resp.json().get("symbols", [])
    result = []
    for s in symbols:
        trends = s.get("trends") or {}
        price_data = s.get("price_data") or {}
        result.append({
            "symbol": s.get("symbol", ""),
            "title": s.get("title", s.get("symbol", "")),
            "trending_score": trends.get("all", 0),
            "trending_summary": trends.get("summary", ""),
            "watchlist_count": s.get("watchlist_count", 0),
            "price": price_data.get("Last", 0),
            "change_pct": price_data.get("PercentChange", 0),
            "sector": s.get("sector", ""),
        })
    return result


def get_symbol_sentiment(ticker: str) -> dict:
    """Fetch 0-100 sentiment score + label for a ticker.

    Returns {symbol, sentiment_score, sentiment_label, volume_score, volume_label}.
    """
    _check_creds()
    try:
        resp = requests.get(
            f"{_BASE}/api-middleware/external/sentiment/v2/{ticker.upper()}/detail",
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            return {"symbol": ticker.upper(), "sentiment_score": 50, "sentiment_label": "Neutral"}
        data = resp.json().get("data") or {}
        sent = (data.get("sentiment") or {}).get("15m") or {}
        vol = (data.get("messageVolume") or {}).get("15m") or {}
        return {
            "symbol": ticker.upper(),
            "sentiment_score": sent.get("valueNormalized", 50),
            "sentiment_label": sent.get("labelNormalized", "Neutral"),
            "volume_score": vol.get("valueNormalized", 0),
            "volume_label": vol.get("labelNormalized", ""),
        }
    except Exception:
        return {"symbol": ticker.upper(), "sentiment_score": 50, "sentiment_label": "Neutral"}


def get_why_trending(ticker: str) -> str:
    """Fetch AI-generated 'why it's trending' summary for a symbol.
    Returns summary string or empty string if unavailable."""
    _check_creds()
    try:
        resp = requests.get(
            f"{_BASE}/api-middleware/external/api/2/symbols/trending/{ticker.upper()}.json",
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            return ""
        symbol_data = resp.json().get("symbol") or {}
        return (symbol_data.get("trends") or {}).get("summary", "")
    except Exception:
        return ""


def run_quick_stocktwits() -> dict:
    """Background-safe QIR helper — fetches StockTwits trending + sentiment.

    No st.* calls. Raises RuntimeError on failure so QIR can report the reason.
    Returns digest dict with trending tickers, sentiment scores, and summary text.
    """
    trending = get_trending_symbols(limit=20)
    if not trending:
        raise RuntimeError("StockTwits returned no trending symbols")

    # Enrich top 6 with per-symbol sentiment scores (keeps API calls low)
    enriched = []
    for item in trending[:6]:
        sent = get_symbol_sentiment(item["symbol"])
        enriched.append({**item, **sent})

    # Aggregate market mood from sentiment scores
    scores = [e["sentiment_score"] for e in enriched if e.get("sentiment_score") is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else 50
    market_mood = (
        "bullish" if avg_score >= 60
        else ("bearish" if avg_score <= 40 else "mixed")
    )

    top_bullish = sorted(enriched, key=lambda x: x.get("sentiment_score", 50), reverse=True)[:3]
    top_bearish = sorted(enriched, key=lambda x: x.get("sentiment_score", 50))[:3]

    # Build summary text for AI prompts
    trending_str = ", ".join(
        f"{e['symbol']} ({e.get('sentiment_label','?')} {e.get('sentiment_score','?')}/100)"
        for e in enriched
    )
    bull_names = ", ".join(e["symbol"] for e in top_bullish[:2])
    bear_names = ", ".join(e["symbol"] for e in top_bearish[:2])

    # Include trending summaries for top 3
    trend_context = []
    for e in enriched[:3]:
        if e.get("trending_summary"):
            trend_context.append(f"{e['symbol']}: {e['trending_summary'][:120]}")

    summary = (
        f"StockTwits Social Sentiment: avg score {avg_score}/100 ({market_mood}). "
        f"Trending: {trending_str}. "
        + (f"Most bullish crowd: {bull_names}. " if bull_names else "")
        + (f"Most bearish crowd: {bear_names}. " if bear_names else "")
        + (" | ".join(trend_context) if trend_context else "")
    )

    return {
        "trending_tickers": enriched,
        "all_trending_symbols": [t["symbol"] for t in trending],
        "market_mood": market_mood,
        "avg_sentiment_score": avg_score,
        "overall_bull_pct": avg_score,           # kept for backward compat with downstream consumers
        "top_bullish": [e["symbol"] for e in top_bullish],
        "top_bearish": [e["symbol"] for e in top_bearish],
        "summary": summary,
    }
