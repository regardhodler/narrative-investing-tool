"""
Alert evaluation and delivery via Telegram Bot API.

Checks triggers (regime flip, insider clusters, options P/C ratio, stress score)
and sends notifications via Telegram. Called on page load with 1-hour cooldown.
"""

import time
import requests
from datetime import datetime
from utils.alerts_config import load_config, save_config, add_alert_history
from utils.watchlist import load_watchlist


def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a message via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _check_regime_flip(cfg: dict) -> str | None:
    """Check if risk regime has flipped since last check."""
    import json
    import os
    history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json")
    if not os.path.exists(history_file):
        return None
    try:
        with open(history_file) as f:
            history = json.load(f)
        if not history:
            return None
        latest = sorted(history, key=lambda x: x.get("date", ""))[-1]
        current_regime = latest.get("regime", "")
        last_regime = cfg.get("last_regime")
        if last_regime and current_regime != last_regime:
            cfg["last_regime"] = current_regime
            return f"REGIME FLIP: {last_regime} → {current_regime}"
        cfg["last_regime"] = current_regime
    except Exception:
        pass
    return None


def _check_insider_clusters(cfg: dict) -> list[str]:
    """Check watchlist tickers for insider buying clusters."""
    alerts = []
    watchlist = load_watchlist()
    if not watchlist:
        return alerts

    from services.sec_client import get_insider_trades
    import pandas as pd

    for item in watchlist[:10]:  # limit to 10 tickers
        ticker = item["ticker"]
        try:
            df = get_insider_trades(ticker)
            if df.empty:
                continue
            buys = df[df["type"] == "Purchase"].copy()
            if len(buys) < 3:
                continue
            buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
            recent = buys[buys["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
            if len(recent) >= 3:
                total_val = recent["value"].sum()
                alerts.append(f"INSIDER CLUSTER: {ticker} — {len(recent)} insider buys in 30d (${total_val:,.0f})")
        except Exception:
            continue
    return alerts


def _check_options_pc(cfg: dict) -> list[str]:
    """Check watchlist for extreme P/C ratios."""
    alerts = []
    threshold = cfg.get("thresholds", {}).get("pc_ratio", 1.5)
    watchlist = load_watchlist()
    if not watchlist:
        return alerts

    import yfinance as yf

    for item in watchlist[:10]:
        ticker = item["ticker"]
        try:
            tk = yf.Ticker(ticker)
            expiries = list(tk.options or [])
            if not expiries:
                continue
            chain = tk.option_chain(expiries[0])
            call_oi = chain.calls["openInterest"].sum() if chain.calls is not None else 0
            put_oi = chain.puts["openInterest"].sum() if chain.puts is not None else 0
            if call_oi > 0:
                pc = put_oi / call_oi
                if pc > threshold:
                    alerts.append(f"OPTIONS ALERT: {ticker} P/C ratio {pc:.2f} > {threshold} (contrarian bullish)")
        except Exception:
            continue
    return alerts


def check_and_send_alerts() -> list[str]:
    """Main entry point. Check all triggers and send alerts. Returns list of alert messages sent."""
    cfg = load_config()

    # 1-hour cooldown
    last_check = cfg.get("last_check")
    if last_check:
        try:
            elapsed = time.time() - float(last_check)
            if elapsed < 3600:
                return []
        except (ValueError, TypeError):
            pass

    cfg["last_check"] = str(time.time())
    triggers = cfg.get("triggers", {})
    bot_token = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")

    all_alerts = []

    if triggers.get("regime_flip"):
        msg = _check_regime_flip(cfg)
        if msg:
            all_alerts.append(msg)

    if triggers.get("insider_cluster"):
        all_alerts.extend(_check_insider_clusters(cfg))

    if triggers.get("options_pc_ratio"):
        all_alerts.extend(_check_options_pc(cfg))

    # Send via Telegram
    sent = []
    for msg in all_alerts:
        full_msg = f"📡 <b>HRT Alert</b>\n\n{msg}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        if send_telegram(bot_token, chat_id, full_msg):
            sent.append(msg)
        add_alert_history(msg)

    save_config(cfg)
    return sent
