import json
import os
from uuid import uuid4
from datetime import date

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_journal.json")


def load_journal() -> list[dict]:
    if os.path.exists(_FILE):
        with open(_FILE) as f:
            return json.load(f)
    return []


def save_journal(trades: list[dict]):
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    with open(_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def add_trade(ticker: str, direction: str, entry_price: float, position_size: float,
              signal_source: str = "Manual", notes: str = "", entry_date: str | None = None,
              thesis: str = "", regime_at_entry: str = "") -> dict:
    trades = load_journal()
    trade = {
        "id": str(uuid4()),
        "ticker": ticker.upper().strip(),
        "direction": direction,
        "entry_date": entry_date or str(date.today()),
        "entry_price": entry_price,
        "exit_date": None,
        "exit_price": None,
        "position_size": position_size,
        "signal_source": signal_source,
        "notes": notes,
        "thesis": thesis,
        "regime_at_entry": regime_at_entry,
        "status": "open",
    }
    trades.append(trade)
    save_journal(trades)
    return trade


def update_trade(trade_id: str, **updates) -> bool:
    trades = load_journal()
    for t in trades:
        if t["id"] == trade_id:
            t.update(updates)
            save_journal(trades)
            return True
    return False


def close_trade(trade_id: str, exit_price: float) -> bool:
    return update_trade(trade_id, exit_price=exit_price, exit_date=str(date.today()), status="closed")


def delete_trade(trade_id: str) -> bool:
    trades = load_journal()
    filtered = [t for t in trades if t["id"] != trade_id]
    if len(filtered) < len(trades):
        save_journal(filtered)
        return True
    return False
