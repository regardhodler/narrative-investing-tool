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


def _check_stress_threshold(cfg: dict) -> str | None:
    """Trigger when regime score crosses into stress territory (score < -1.0 = strong Risk-Off)."""
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
        score = float(latest.get("score", 0))
        threshold = float(cfg.get("thresholds", {}).get("stress_score", 70))
        # stress_score threshold of 70 maps to regime score < -1.0
        # (score ranges ~-3 to +3; below -1.0 is meaningfully stressed)
        stress_score_normalized = max(0, min(100, 50 - score * 20))
        if stress_score_normalized >= threshold:
            regime = latest.get("regime", "")
            return f"STRESS ALERT: Regime score {score:+.2f} (stress level {stress_score_normalized:.0f}/100) — {regime}"
    except Exception:
        pass
    return None


def _check_tactical_threshold(cfg: dict) -> list[str]:
    """Fire when tactical score crosses below 38 (Risk-Off) or above 65 (Favorable Entry)."""
    last = cfg.get("last_tactical_score")
    curr = cfg.get("current_tactical_score")
    if last is None or curr is None:
        return []
    alerts = []
    risk_off_th = cfg.get("thresholds", {}).get("tactical_risk_off", 38)
    entry_th    = cfg.get("thresholds", {}).get("tactical_entry", 65)
    if last >= risk_off_th and curr < risk_off_th:
        alerts.append(
            f"⚠️ <b>TACTICAL: Risk-Off Threshold Crossed</b>\n"
            f"Score dropped {last} → {curr}/100 (below {risk_off_th}). Consider defensive posture."
        )
    if last <= entry_th and curr > entry_th:
        alerts.append(
            f"✅ <b>TACTICAL: Favorable Entry Signal</b>\n"
            f"Score rose {last} → {curr}/100 (above {entry_th}). Conditions support adding risk."
        )
    return alerts


def _check_data_quality(cfg: dict) -> str | None:
    """Fire once when data quality score drops below 60. Resets latch when score recovers."""
    score = cfg.get("current_data_quality_score")
    if score is None:
        return None
    if score < 60 and not cfg.get("data_quality_alerted", False):
        cfg["data_quality_alerted"] = True
        return (
            f"🔴 <b>DATA QUALITY WARNING</b>\n"
            f"Score: {score}/100 — AI analysis may be unreliable. Check FRED/yfinance connectivity."
        )
    if score >= 60:
        cfg["data_quality_alerted"] = False
    return None


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

    if triggers.get("stress_threshold"):
        msg = _check_stress_threshold(cfg)
        if msg:
            all_alerts.append(msg)

    if triggers.get("tactical_threshold"):
        all_alerts.extend(_check_tactical_threshold(cfg))

    if triggers.get("data_quality_alert"):
        msg = _check_data_quality(cfg)
        if msg:
            all_alerts.append(msg)

    # Send via Telegram
    sent = []
    for msg in all_alerts:
        full_msg = f"📡 <b>HRT Alert</b>\n\n{msg}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        if send_telegram(bot_token, chat_id, full_msg):
            sent.append(msg)
        add_alert_history(msg)

    save_config(cfg)
    return sent


# ── Background Worker ──────────────────────────────────────────────────────────

def _check_price_alerts(cfg: dict) -> list[str]:
    """Check price targets — fires when ticker crosses target. Safe to call outside Streamlit."""
    import yfinance as yf
    alerts = []
    targets = cfg.get("price_targets", [])
    changed = False
    for i, pt in enumerate(targets):
        if not pt.get("active", True):
            continue
        ticker = pt.get("ticker", "")
        target = float(pt.get("target", 0))
        direction = pt.get("direction", "above")
        if not ticker or not target:
            continue
        try:
            info = yf.Ticker(ticker).info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            if not price:
                continue
            hit = (direction == "above" and price >= target) or \
                  (direction == "below" and price <= target)
            if hit:
                alerts.append(
                    f"PRICE ALERT: {ticker} ${price:.2f} — "
                    f"{'above' if direction == 'above' else 'below'} target ${target:.2f}"
                )
                cfg["price_targets"][i]["active"] = False  # deactivate to prevent repeats
                changed = True
        except Exception:
            continue
    if changed:
        save_config(cfg)
    return alerts


def _write_heartbeat(sent: int = 0) -> None:
    """Write worker heartbeat to data/worker_heartbeat.json."""
    import json as _json
    import os as _os
    _hb = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "worker_heartbeat.json")
    try:
        _existing = {}
        if _os.path.exists(_hb):
            with open(_hb) as _f:
                _existing = _json.load(_f)
        _existing["last_run"] = datetime.now().isoformat()
        _existing["checks_run"] = _existing.get("checks_run", 0) + 1
        _existing["alerts_sent"] = _existing.get("alerts_sent", 0) + sent
        _os.makedirs(_os.path.dirname(_hb), exist_ok=True)
        with open(_hb, "w") as _f:
            _json.dump(_existing, _f, indent=2)
    except Exception:
        pass


def run_worker_cycle() -> list[str]:
    """Run one full alert check cycle. No cooldown — designed for background worker use.

    Returns list of alert messages that were sent via Telegram.
    Skips any check that requires a Streamlit context (insider cluster may fail — caught silently).
    """
    cfg = load_config()
    triggers = cfg.get("triggers", {})
    bot_token = cfg.get("telegram_bot_token", "")
    chat_id   = cfg.get("telegram_chat_id", "")

    all_alerts: list[str] = []

    # Regime flip — pure json/os, always safe
    if triggers.get("regime_flip"):
        try:
            msg = _check_regime_flip(cfg)
            if msg:
                all_alerts.append(msg)
        except Exception:
            pass

    # Stress threshold — pure json/os, always safe
    if triggers.get("stress_threshold"):
        try:
            msg = _check_stress_threshold(cfg)
            if msg:
                all_alerts.append(msg)
        except Exception:
            pass

    # Price targets — yfinance only, always safe
    try:
        all_alerts.extend(_check_price_alerts(cfg))
    except Exception:
        pass

    # Options P/C — raw yfinance, usually safe outside Streamlit
    if triggers.get("options_pc_ratio"):
        try:
            all_alerts.extend(_check_options_pc(cfg))
        except Exception:
            pass

    # Insider cluster — calls sec_client with @st.cache_data, may fail outside Streamlit
    if triggers.get("insider_cluster"):
        try:
            all_alerts.extend(_check_insider_clusters(cfg))
        except Exception:
            pass  # silently skip — will still run on page load

    # Tactical threshold — reads from alerts_config (persisted by QIR), always safe
    if triggers.get("tactical_threshold"):
        try:
            all_alerts.extend(_check_tactical_threshold(cfg))
        except Exception:
            pass

    # Data quality — reads from alerts_config (persisted by QIR), always safe
    if triggers.get("data_quality_alert"):
        try:
            msg = _check_data_quality(cfg)
            if msg:
                all_alerts.append(msg)
        except Exception:
            pass

    # Send via Telegram
    sent = []
    for msg in all_alerts:
        full_msg = f"📡 <b>HRT Alert</b>\n\n{msg}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        if send_telegram(bot_token, chat_id, full_msg):
            sent.append(msg)
        add_alert_history(msg)

    save_config(cfg)
    _write_heartbeat(sent=len(sent))
    return sent


def run_alert_worker(interval_minutes: int = 15) -> None:
    """Infinite loop — run alert checks every interval_minutes. Call from tools/alert_worker.py."""
    import time as _time
    print(f"[HRT Alert Worker] Started — checking every {interval_minutes} minutes. Ctrl+C to stop.")
    while True:
        try:
            _sent = run_worker_cycle()
            _ts = datetime.now().strftime("%H:%M:%S")
            if _sent:
                print(f"[{_ts}] {len(_sent)} alert(s) sent: {_sent[0][:60]}{'…' if len(_sent[0]) > 60 else ''}")
            else:
                print(f"[{_ts}] No alerts triggered. Next check in {interval_minutes}m.")
        except KeyboardInterrupt:
            print("\n[HRT Alert Worker] Stopped.")
            break
        except Exception as _e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker error: {_e}")
        _time.sleep(interval_minutes * 60)
