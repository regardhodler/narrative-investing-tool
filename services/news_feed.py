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

# Inbox Gist persistence (survives Streamlit Cloud redeploys)
_INBOX_GIST_ID  = os.getenv("INBOX_GIST_ID", "")
_INBOX_GIST_RAW = os.getenv("INBOX_GIST_RAW_URL", "")
_GIST_TOKEN     = (os.getenv("GIST_TOKEN") or os.getenv("GITHUB_GIST_TOKEN") or "").strip()
_INBOX_GIST_FILENAME = "news_inbox.json"

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


def load_news_inbox(prefer_gist: bool = False) -> list[dict]:
    """
    Load inbox.
    Default: local file first (fast, consistent within a session), Gist as cold-start fallback.
    prefer_gist=True: Gist first (used on fresh Streamlit Cloud deploy where local file is absent).
    """
    # If local file exists and we're not forcing Gist, use it — avoids stale-Gist overwrite bug
    if not prefer_gist and os.path.exists(_INBOX_FILE):
        try:
            with open(_INBOX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # Cold-start: try Gist (Streamlit Cloud fresh deploy — no local file yet)
    if _INBOX_GIST_RAW:
        try:
            resp = requests.get(
                _INBOX_GIST_RAW, timeout=8,
                headers={"User-Agent": "NarrativeInvestingTool/1.0", "Cache-Control": "no-cache"},
            )
            if resp.ok:
                data = resp.json()
                if isinstance(data, list):
                    # Seed local file from Gist so subsequent calls use local
                    os.makedirs(_DATA_DIR, exist_ok=True)
                    with open(_INBOX_FILE, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    return data
        except Exception:
            pass

    return []


def save_to_inbox(text: str, source: str = "manual", message_id: int = 0) -> None:
    """
    Append a new item to inbox. Deduplicates Telegram messages by message_id.
    Writes local file + Gist (Gist debounced to 2 min).
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    items = load_news_inbox()

    # Deduplicate Telegram messages
    if message_id:
        existing_ids = {i.get("message_id", 0) for i in items}
        if message_id in existing_ids:
            return

    items.append({
        "text": text.strip(),
        "source": source,
        "ts": datetime.now().isoformat(),
        **({"message_id": message_id} if message_id else {}),
    })
    items = items[-_INBOX_MAX:]

    payload_str = json.dumps(items, indent=2)

    # Always write local file
    with open(_INBOX_FILE, "w", encoding="utf-8") as f:
        f.write(payload_str)

    # Write Gist (debounced via streamlit session_state)
    if _INBOX_GIST_ID and _GIST_TOKEN:
        try:
            import streamlit as _st
            last = _st.session_state.get("_inbox_gist_saved_at")
            now = datetime.now()
            if last is None or (now - last).total_seconds() > 120:
                requests.patch(
                    f"https://api.github.com/gists/{_INBOX_GIST_ID}",
                    json={"files": {_INBOX_GIST_FILENAME: {"content": payload_str}}},
                    headers={"Authorization": f"Bearer {_GIST_TOKEN}",
                             "Accept": "application/vnd.github+json"},
                    timeout=10,
                )
                _st.session_state["_inbox_gist_saved_at"] = now
        except Exception:
            pass


def clear_inbox() -> None:
    """Delete all inbox items (local + Gist)."""
    payload_str = json.dumps([], indent=2)
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_INBOX_FILE, "w", encoding="utf-8") as f:
        f.write(payload_str)
    if _INBOX_GIST_ID and _GIST_TOKEN:
        try:
            requests.patch(
                f"https://api.github.com/gists/{_INBOX_GIST_ID}",
                json={"files": {_INBOX_GIST_FILENAME: {"content": payload_str}}},
                headers={"Authorization": f"Bearer {_GIST_TOKEN}",
                         "Accept": "application/vnd.github+json"},
                timeout=10,
            )
        except Exception:
            pass


def _extract_tweet_content(url: str) -> str | None:
    """Use Twitter oEmbed API to get tweet text without requiring JavaScript."""
    try:
        r = requests.get(
            "https://publish.twitter.com/oembed",
            params={"url": url, "omit_script": "true"},
            timeout=8,
        )
        if r.ok:
            html = r.json().get("html", "")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # oEmbed returns an HTML blockquote — extract the text
            text = soup.get_text(" ", strip=True)
            if text and len(text) > 20:
                author = r.json().get("author_name", "")
                return f"@{author}: {text}" if author else text
    except Exception:
        pass
    return None


def _extract_url_content(url: str, max_chars: int = 2000) -> str | None:
    """
    Fetch a URL and extract the main article text using BeautifulSoup.
    Returns extracted text or None on failure.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            url, timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
            allow_redirects=True,
        )
        if not resp.ok:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        # Try article > main > body in order
        for selector in ["article", "main", "[role='main']", ".article-body", ".post-content", "body"]:
            container = soup.select_one(selector)
            if container:
                text = " ".join(container.get_text(" ", strip=True).split())
                if len(text) > 200:
                    return text[:max_chars]
        return None
    except Exception:
        return None


def sync_telegram_to_inbox() -> int:
    """
    Poll Telegram for new messages and save them to the inbox.
    If a message is a URL, fetches the page and extracts article text.
    Returns number of new messages added.
    Only runs if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are configured.
    """
    import re
    try:
        from services.telegram_client import poll_new_messages, is_configured
        if not is_configured():
            return 0

        # Find highest message_id already in inbox to avoid re-importing
        items = load_news_inbox()
        seen_ids = {i.get("message_id", 0) for i in items if i.get("message_id")}
        since_id = max(seen_ids) if seen_ids else 0

        new_msgs = poll_new_messages(since_message_id=since_id)
        _url_re = re.compile(r'https?://\S+')

        for msg in new_msgs:
            text = msg["text"]
            urls = _url_re.findall(text)

            if urls:
                # Extract content from each URL found in the message
                extracted_parts = []
                for url in urls[:2]:  # max 2 URLs per message
                    # Special handler for X / Twitter
                    if any(d in url for d in ("x.com", "twitter.com")):
                        content = _extract_tweet_content(url) or _extract_url_content(url)
                    else:
                        content = _extract_url_content(url)
                    # Reject JS-wall responses
                    if content and "javascript" in content.lower()[:80]:
                        content = None
                    if content:
                        extracted_parts.append(f"[From {url}]\n{content}")

                if extracted_parts:
                    # Prepend any non-URL text the user wrote as context
                    user_note = _url_re.sub("", text).strip()
                    full_text = "\n\n".join(
                        ([f"Note: {user_note}"] if user_note else []) + extracted_parts
                    )
                    save_to_inbox(full_text, source="📱 Telegram (article)", message_id=msg["message_id"])
                else:
                    # URL fetch failed — save the raw URL with a note
                    save_to_inbox(f"[Link — could not extract]\n{text}", source="📱 Telegram", message_id=msg["message_id"])
            else:
                save_to_inbox(text, source="📱 Telegram", message_id=msg["message_id"])

        return len(new_msgs)
    except Exception:
        return 0


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
