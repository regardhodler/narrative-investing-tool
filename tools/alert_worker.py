"""
HRT Alert Worker — runs alert checks in the background without the app being open.

SETUP:
1. Ensure TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (or stored in alerts_config.json) are configured
2. Run once to test:
       python tools/alert_worker.py --once
3. Run continuously (every 15 min):
       python tools/alert_worker.py
4. Background (no console window, Windows):
       pythonw tools/alert_worker.py
5. Background (Mac/Linux):
       nohup python tools/alert_worker.py &
6. Windows Task Scheduler (auto-start on login):
   - Action: pythonw.exe
   - Arguments: "C:\\path\\to\\tools\\alert_worker.py"
   - Trigger: At log on / At startup

NOTES:
- Checks: regime flip, stress threshold, price targets, options P/C ratio
- Insider cluster check is skipped outside Streamlit (st.cache_data not available) — it still
  fires on page load as usual.
- Heartbeat written to data/worker_heartbeat.json — the Alerts Settings page reads this to
  show worker liveness.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Load .env ──────────────────────────────────────────────────────────────────
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Main ───────────────────────────────────────────────────────────────────────
from services.alerts_service import run_worker_cycle, run_alert_worker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="HRT Alert Worker")
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Check interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single check cycle and exit (useful for cron/Task Scheduler)"
    )
    args = parser.parse_args()

    if args.once:
        sent = run_worker_cycle()
        if sent:
            print(f"[HRT Alert Worker] {len(sent)} alert(s) sent: {sent}")
        else:
            print("[HRT Alert Worker] No alerts triggered.")
        return

    run_alert_worker(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
