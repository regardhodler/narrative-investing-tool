import json
import os

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "watchlist.json")


def load_watchlist() -> list[dict]:
    """Load watchlist from JSON. Each entry: {"ticker": "AAPL", "narrative": "AI Play", "added": "2026-03-15"}"""
    if os.path.exists(_FILE):
        with open(_FILE) as f:
            return json.load(f)
    return []


def save_watchlist(items: list[dict]):
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    with open(_FILE, "w") as f:
        json.dump(items, f, indent=2)


def add_to_watchlist(ticker: str, narrative: str = ""):
    items = load_watchlist()
    if any(item["ticker"] == ticker for item in items):
        return False  # already exists
    from datetime import date
    items.append({"ticker": ticker, "narrative": narrative, "added": str(date.today())})
    save_watchlist(items)
    return True


def remove_from_watchlist(ticker: str):
    items = [i for i in load_watchlist() if i["ticker"] != ticker]
    save_watchlist(items)


def is_in_watchlist(ticker: str) -> bool:
    return any(i["ticker"] == ticker for i in load_watchlist())
