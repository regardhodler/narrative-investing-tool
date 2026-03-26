"""
Telegram Inbox Bot — forwards messages to data/news_inbox.json for the investing tool.

SETUP:
1. Install: pip install python-telegram-bot
2. Create a bot: message @BotFather on Telegram → /newbot → copy token
3. Set TELEGRAM_BOT_TOKEN in your .env file
4. Run: pythonw tools/telegram_inbox_bot.py   (background, no console window)
   Or add to Windows Task Scheduler to run on login.

USAGE:
- Open Telegram on your phone
- Find your bot and send it any message
- Text, forwarded posts from X, articles — anything
- Next time you open the investing tool, it appears in the Current Events inbox
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve data dir relative to this script
_ROOT = Path(__file__).parent.parent
_INBOX_FILE = _ROOT / "data" / "news_inbox.json"
_INBOX_MAX = 50

# Load .env if present
_env_file = _ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
    sys.exit(1)


def _load_inbox():
    if _INBOX_FILE.exists():
        try:
            return json.loads(_INBOX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_inbox(items):
    _INBOX_FILE.parent.mkdir(exist_ok=True)
    items = items[-_INBOX_MAX:]
    _INBOX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _append_item(text: str, source: str = "telegram"):
    items = _load_inbox()
    items.append({"text": text.strip(), "source": source, "ts": datetime.now().isoformat()})
    _save_inbox(items)


try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text or update.message.caption or ""
        # Handle forwarded messages — include original source if available
        fwd = update.message.forward_origin
        if fwd:
            source = f"telegram_fwd"
        else:
            source = "telegram"
        if text.strip():
            _append_item(text, source=source)
            await update.message.reply_text("✅ Saved to investing tool inbox.")
        else:
            await update.message.reply_text("⚠ No text found — send text or captions only.")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, handle_message))
    print(f"Bot running. Send messages to your Telegram bot to add to inbox → {_INBOX_FILE}")
    app.run_polling()

except ImportError:
    print("ERROR: python-telegram-bot not installed. Run: pip install python-telegram-bot")
    sys.exit(1)
