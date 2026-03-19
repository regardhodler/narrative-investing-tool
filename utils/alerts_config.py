import json
import os

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alerts_config.json")

_DEFAULTS = {
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "triggers": {
        "regime_flip": True,
        "insider_cluster": True,
        "options_pc_ratio": False,
        "stress_threshold": False,
    },
    "thresholds": {
        "pc_ratio": 1.5,
        "stress_score": 70,
    },
    "last_check": None,
    "last_regime": None,
    "alert_history": [],
}


def load_config() -> dict:
    if os.path.exists(_FILE):
        with open(_FILE) as f:
            cfg = json.load(f)
        # Merge defaults for any missing keys
        for k, v in _DEFAULTS.items():
            if k not in cfg:
                cfg[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in cfg[k]:
                        cfg[k][kk] = vv
        return cfg
    return dict(_DEFAULTS)


def save_config(cfg: dict):
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    with open(_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def add_alert_history(message: str):
    from datetime import datetime
    cfg = load_config()
    cfg["alert_history"].insert(0, {"time": datetime.now().isoformat(), "message": message})
    cfg["alert_history"] = cfg["alert_history"][:100]  # keep last 100
    save_config(cfg)
