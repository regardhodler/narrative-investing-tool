"""Alerts Settings — configure Telegram alerts and manage triggers."""

import json
import os
from datetime import datetime

import streamlit as st
from utils.theme import COLORS
from utils.alerts_config import load_config, save_config
from services.alerts_service import send_telegram

_HEARTBEAT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "worker_heartbeat.json"
)


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">ALERTS</div>',
        unsafe_allow_html=True,
    )

    cfg = load_config()

    # --- Telegram Setup ---
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">TELEGRAM SETUP</div>', unsafe_allow_html=True)

    with st.expander("How to set up Telegram alerts", expanded=False):
        st.markdown("""
**Step 1:** Open Telegram, search for **@BotFather**

**Step 2:** Send `/newbot` and follow prompts to create your bot

**Step 3:** Copy the **bot token** (looks like `123456:ABC-DEF1234`)

**Step 4:** Start a chat with your new bot, send any message

**Step 5:** Get your chat ID:
- Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
- Find `"chat":{"id": 123456}` — that number is your chat ID

**Step 6:** Paste both values below and click TEST
        """)

    col1, col2 = st.columns(2)
    with col1:
        bot_token = st.text_input("Bot Token", value=cfg.get("telegram_bot_token", ""),
                                   type="password", key="alert_token")
    with col2:
        chat_id = st.text_input("Chat ID", value=cfg.get("telegram_chat_id", ""), key="alert_chatid")

    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("SAVE CONFIG", key="alert_save"):
            cfg["telegram_bot_token"] = bot_token
            cfg["telegram_chat_id"] = chat_id
            save_config(cfg)
            st.success("Config saved.")

    with col_test:
        if st.button("SEND TEST", key="alert_test"):
            if bot_token and chat_id:
                ok = send_telegram(bot_token, chat_id,
                                   "📡 <b>HRT Test</b>\n\nTelegram alerts are working!")
                if ok:
                    st.success("Test message sent!")
                else:
                    st.error("Failed to send. Check your token and chat ID.")
            else:
                st.warning("Enter bot token and chat ID first.")

    # --- Triggers ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">ALERT TRIGGERS</div>', unsafe_allow_html=True)

    triggers = cfg.get("triggers", {})
    thresholds = cfg.get("thresholds", {})

    col1, col2 = st.columns(2)
    with col1:
        regime_flip = st.checkbox("Risk Regime Flip", value=triggers.get("regime_flip", True),
                                   key="trig_regime", help="Alert when regime changes (Risk-On / Risk-Off)")
        insider_cluster = st.checkbox("Insider Buying Cluster", value=triggers.get("insider_cluster", True),
                                       key="trig_insider", help="Alert when 3+ insider buys in 30 days on watchlist")
    with col2:
        pc_ratio = st.checkbox("Options P/C Ratio", value=triggers.get("options_pc_ratio", False),
                                key="trig_pc", help="Alert when P/C ratio exceeds threshold")
        stress = st.checkbox("Stress Score Threshold", value=triggers.get("stress_threshold", False),
                              key="trig_stress", help="Alert when stress score crosses threshold")
        tactical_threshold = st.checkbox("Tactical Score Crossing", value=triggers.get("tactical_threshold", False),
                              key="trig_tactical", help="Alert when tactical score crosses below 38 (Risk-Off) or above 65 (Favorable Entry)")
        data_quality_alert = st.checkbox("Data Quality Warning", value=triggers.get("data_quality_alert", False),
                              key="trig_dq", help="Alert when data quality drops below 60 — AI analysis may be unreliable")

    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin:8px 0 4px 0;">'
                f'THRESHOLDS</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pc_thresh = st.number_input("P/C Ratio Threshold", value=thresholds.get("pc_ratio", 1.5),
                                     min_value=0.5, max_value=5.0, step=0.1, key="thresh_pc")
    with col2:
        stress_thresh = st.number_input("Stress Score Threshold", value=thresholds.get("stress_score", 70),
                                         min_value=0, max_value=100, step=5, key="thresh_stress")

    if st.button("SAVE TRIGGERS", type="primary", key="alert_save_triggers"):
        cfg["triggers"] = {
            "regime_flip": regime_flip,
            "insider_cluster": insider_cluster,
            "options_pc_ratio": pc_ratio,
            "stress_threshold": stress,
            "tactical_threshold": tactical_threshold,
            "data_quality_alert": data_quality_alert,
        }
        cfg["thresholds"] = {
            "pc_ratio": pc_thresh,
            "stress_score": stress_thresh,
            "tactical_risk_off": thresholds.get("tactical_risk_off", 38),
            "tactical_entry": thresholds.get("tactical_entry", 65),
        }
        save_config(cfg)
        st.success("Triggers saved.")

    # --- Alert History ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">ALERT HISTORY</div>', unsafe_allow_html=True)

    history = cfg.get("alert_history", [])
    if not history:
        st.info("No alerts sent yet.")
    else:
        for entry in history[:20]:
            time_str = entry.get("time", "")[:16].replace("T", " ")
            msg = entry.get("message", "")
            st.markdown(
                f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="color:{COLORS["text_dim"]};">{time_str}</span> '
                f'<span style="color:{COLORS["text"]};">{msg}</span></div>',
                unsafe_allow_html=True,
            )

    # --- Price Targets ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">PRICE TARGETS</div>', unsafe_allow_html=True)
    st.caption("Fires once when the ticker crosses your target price. Auto-deactivates on trigger.")

    price_targets = cfg.get("price_targets", [])

    # Add new target
    with st.expander("Add Price Target", expanded=False):
        pt_col1, pt_col2, pt_col3 = st.columns([2, 2, 2])
        with pt_col1:
            pt_ticker = st.text_input("Ticker", key="pt_ticker", placeholder="e.g. NVDA").upper().strip()
        with pt_col2:
            pt_price = st.number_input("Target Price ($)", min_value=0.01, value=100.0,
                                       step=0.5, key="pt_price")
        with pt_col3:
            pt_dir = st.radio("Direction", ["above", "below"], horizontal=True, key="pt_dir")
        if st.button("ADD TARGET", key="pt_add"):
            if pt_ticker:
                price_targets.append({
                    "ticker": pt_ticker,
                    "target": round(pt_price, 2),
                    "direction": pt_dir,
                    "active": True,
                })
                cfg["price_targets"] = price_targets
                save_config(cfg)
                st.success(f"Added: {pt_ticker} {pt_dir} ${pt_price:.2f}")
                st.rerun()
            else:
                st.warning("Enter a ticker symbol.")

    # List active targets
    active_targets = [t for t in price_targets if t.get("active", True)]
    inactive_targets = [t for t in price_targets if not t.get("active", True)]

    if active_targets:
        for i, pt in enumerate(price_targets):
            if not pt.get("active", True):
                continue
            _dir_color = COLORS["positive"] if pt["direction"] == "above" else COLORS["negative"]
            _col_a, _col_b = st.columns([5, 1])
            with _col_a:
                st.markdown(
                    f'<div style="font-size:12px;padding:3px 0;">'
                    f'<span style="color:#f1f5f9;font-weight:700;">{pt["ticker"]}</span> '
                    f'<span style="color:{COLORS["text_dim"]};">alert when</span> '
                    f'<span style="color:{_dir_color};font-weight:700;">{pt["direction"]} ${pt["target"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _col_b:
                if st.button("✕", key=f"pt_del_{i}"):
                    cfg["price_targets"][i]["active"] = False
                    save_config(cfg)
                    st.rerun()
    else:
        st.caption("No active price targets.")

    if inactive_targets:
        with st.expander(f"Triggered targets ({len(inactive_targets)})", expanded=False):
            for pt in inactive_targets:
                st.markdown(
                    f'<div style="font-size:11px;color:{COLORS["text_dim"]};">'
                    f'✓ {pt["ticker"]} {pt["direction"]} ${pt["target"]:.2f} — triggered</div>',
                    unsafe_allow_html=True,
                )

    # --- Background Worker Status ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">BACKGROUND WORKER</div>', unsafe_allow_html=True)

    # Read heartbeat
    _hb = {}
    if os.path.exists(_HEARTBEAT_FILE):
        try:
            with open(_HEARTBEAT_FILE) as _f:
                _hb = json.load(_f)
        except Exception:
            pass

    if _hb:
        _last_run_str = _hb.get("last_run", "")
        _checks = _hb.get("checks_run", 0)
        _sent = _hb.get("alerts_sent", 0)
        # Compute staleness
        _stale = True
        _mins_ago = None
        if _last_run_str:
            try:
                _last_dt = datetime.fromisoformat(_last_run_str)
                _mins_ago = (datetime.now() - _last_dt).total_seconds() / 60
                _stale = _mins_ago > 30  # consider stale if > 30 min
            except Exception:
                pass

        _status_color = COLORS["negative"] if _stale else COLORS["positive"]
        _status_label = "STALE" if _stale else "LIVE"
        _ago_str = f"{_mins_ago:.0f}m ago" if _mins_ago is not None else "unknown"

        st.markdown(
            f'<div style="border:1px solid {_status_color}44;border-radius:6px;'
            f'padding:10px 14px;background:#0f172a;display:flex;gap:20px;flex-wrap:wrap;'
            f'align-items:center;">'
            f'<span style="color:{_status_color};font-weight:700;font-size:12px;">⬤ {_status_label}</span>'
            f'<span style="font-size:12px;color:{COLORS["text_dim"]};">Last run: '
            f'<span style="color:{COLORS["text"]};">{_ago_str}</span></span>'
            f'<span style="font-size:12px;color:{COLORS["text_dim"]};">Checks: '
            f'<span style="color:{COLORS["text"]};">{_checks}</span></span>'
            f'<span style="font-size:12px;color:{COLORS["text_dim"]};">Alerts sent: '
            f'<span style="color:{COLORS["text"]};">{_sent}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="border:1px solid {COLORS["border"]};border-radius:6px;'
            f'padding:10px 14px;background:#0f172a;">'
            f'<span style="color:{COLORS["text_dim"]};font-size:12px;">⬤ NOT RUNNING — '
            f'worker has never sent a heartbeat</span></div>',
            unsafe_allow_html=True,
        )

    with st.expander("How to run the background worker", expanded=False):
        st.markdown("""
The background worker checks alerts every 15 minutes — even when the app is closed.
It fires: **regime flips, stress alerts, price targets, options P/C ratio**.

**Quick test (run once):**
```
python tools/alert_worker.py --once
```

**Run continuously (every 15 min):**
```
python tools/alert_worker.py
```

**Background — Windows (no console window):**
```
pythonw tools/alert_worker.py
```

**Background — Mac/Linux:**
```
nohup python tools/alert_worker.py &
```

**Windows Task Scheduler (auto-start on login):**
- Action: `pythonw.exe`
- Arguments: `"C:\\path\\to\\tools\\alert_worker.py"`
- Trigger: At log on

**Custom interval (e.g. every 5 min):**
```
python tools/alert_worker.py --interval 5
```

> Note: Insider cluster alerts still require the app to be open (SEC client uses Streamlit cache).
        """)
