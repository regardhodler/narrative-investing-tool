"""Financial RSS news feed aggregator + file-based inbox for current events injection."""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests
import streamlit as st

_GIST_RAW_URL = os.getenv("NEWS_GIST_RAW_URL", "")  # Set in .env: full raw URL of financial_intel.json

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_INBOX_FILE = os.path.join(_DATA_DIR, "news_inbox.json")
_INBOX_MAX = 50

_RSS_FEEDS = [
    ("Reuters",     "https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Barron's",    "https://www.barrons.com/xml/rss/3_7510.xml"),
    ("Investopedia","https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline"),
    ("FT",          "https://www.ft.com/?format=rss"),
]


def _parse_rss_xml(xml_text: str, source: str) -> list[dict]:
    """Parse generic RSS XML into headline dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title:
            continue
        try:
            dt = parsedate_to_datetime(pub_date)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
            sort_key = dt.timestamp()
        except Exception:
            date_str = pub_date[:16]
            sort_key = 0.0
        items.append({
            "title": title,
            "date": date_str,
            "url": link,
            "source": source,
            "_sort_key": sort_key,
        })
    items.sort(key=lambda x: x["_sort_key"], reverse=True)
    return items


@st.cache_data(ttl=1800)
def fetch_financial_headlines(max_per_feed: int = 6) -> list[dict]:
    """
    Pull latest headlines from financial RSS feeds.
    Returns list of {title, url, source, date}, newest first.
    Cached 30 minutes. Falls back gracefully per feed.
    """
    all_items = []
    for source, url in _RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=6, headers={"User-Agent": "NarrativeInvestingTool/1.0"})
            if resp.ok:
                parsed = _parse_rss_xml(resp.text, source=source)
                all_items.extend(parsed[:max_per_feed])
        except Exception:
            continue

    all_items.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in all_items:
        item.pop("_sort_key", None)
    return all_items


def load_news_inbox() -> list[dict]:
    """
    Read data/news_inbox.json.
    Each item: {text, source, ts}
    Returns [] if file missing or corrupt.
    """
    if not os.path.exists(_INBOX_FILE):
        return []
    try:
        with open(_INBOX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_to_inbox(text: str, source: str = "manual") -> None:
    """Append a new item to news_inbox.json. Caps at _INBOX_MAX items (rolling)."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    items = load_news_inbox()
    items.append({
        "text": text.strip(),
        "source": source,
        "ts": datetime.now().isoformat(),
    })
    # Keep newest _INBOX_MAX items
    items = items[-_INBOX_MAX:]
    with open(_INBOX_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def clear_inbox() -> None:
    """Delete all inbox items."""
    if os.path.exists(_INBOX_FILE):
        with open(_INBOX_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


@st.cache_data(ttl=900)  # 15-minute cache — matches bot's run interval
def fetch_gist_intel() -> dict | None:
    """
    Fetch the latest bot intel snapshot from the GitHub Gist.
    Returns dict with keys: updated_at, narrative, top_alerts, polymarket.
    Set NEWS_GIST_RAW_URL in .env to the raw URL of financial_intel.json.
    Returns None if not configured or on failure.
    """
    url = _GIST_RAW_URL
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "NarrativeInvestingTool/1.0"})
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def headlines_to_text(headlines: list[dict], max_items: int = 20) -> str:
    """Format headlines list as plain text for AI prompt injection."""
    lines = []
    for h in headlines[:max_items]:
        lines.append(f"[{h['source']}] {h['title']} ({h['date']})")
    return "\n".join(lines)


def inbox_to_text(inbox: list[dict], max_items: int = 10) -> str:
    """Format inbox items as plain text for AI prompt injection."""
    lines = []
    for item in inbox[-max_items:]:
        ts = item.get("ts", "")[:16].replace("T", " ")
        lines.append(f"[{ts}] {item.get('text', '')}")
    return "\n".join(lines)


def polymarket_to_text(markets: list[dict]) -> str:
    """Format Polymarket markets as plain text for AI prompt injection."""
    if not markets:
        return ""
    lines = ["Polymarket Prediction Markets:"]
    for m in markets:
        prob = m.get("probability")
        prob_str = f"{prob:.0%}" if prob is not None else "N/A"
        lines.append(f"  • {m.get('question', '')} → {prob_str}")
    return "\n".join(lines)
