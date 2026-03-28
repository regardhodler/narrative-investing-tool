"""Telegram Bot integration — outbound alerts + inbound field notes polling."""

import os
import requests

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
_BASE = f"https://api.telegram.org/bot{_TOKEN}"


def is_configured() -> bool:
    return bool(_TOKEN and _CHAT_ID)


def send_alert(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the user via Telegram bot. Returns True on success."""
    if not is_configured():
        return False
    try:
        r = requests.post(
            f"{_BASE}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": message, "parse_mode": parse_mode},
            timeout=8,
        )
        return r.ok
    except Exception:
        return False


def send_document(filename: str, content: str, caption: str = "") -> bool:
    """Send a text file as a document to the user via Telegram bot."""
    if not is_configured():
        return False
    try:
        r = requests.post(
            f"{_BASE}/sendDocument",
            data={"chat_id": _CHAT_ID, "caption": caption},
            files={"document": (filename, content.encode("utf-8"), "text/plain")},
            timeout=15,
        )
        return r.ok
    except Exception:
        return False
    """
    Poll getUpdates for messages from the user's chat.
    Returns list of {message_id, text, date_iso} — newest last.
    Filters to TELEGRAM_CHAT_ID only. Skips bot commands (/start etc).
    since_message_id: only return messages with message_id > this value.
    """
    if not is_configured():
        return []
    try:
        r = requests.get(
            f"{_BASE}/getUpdates",
            params={"limit": 100, "timeout": 0},
            timeout=10,
        )
        if not r.ok:
            return []
        updates = r.json().get("result", [])
        messages = []
        for upd in updates:
            msg = upd.get("message", {})
            if str(msg.get("chat", {}).get("id", "")) != _CHAT_ID:
                continue
            mid = msg.get("message_id", 0)
            if mid <= since_message_id:
                continue
            text = (msg.get("text") or msg.get("caption") or "").strip()
            if not text or text.startswith("/"):
                continue
            from datetime import datetime, timezone
            ts = msg.get("date", 0)
            date_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else datetime.now(timezone.utc).isoformat()
            messages.append({"message_id": mid, "text": text, "date_iso": date_iso})
        messages.sort(key=lambda x: x["message_id"])
        return messages
    except Exception:
        return []
