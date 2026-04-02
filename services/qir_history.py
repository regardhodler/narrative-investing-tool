"""QIR run history — persists each run's key signals to data/qir_run_history.json."""
import json
import os
from dataclasses import dataclass, asdict

_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "qir_run_history.json")
_MAX_RUNS = 30  # keep last 30 runs


@dataclass
class QIRRunRecord:
    run_id: str
    timestamp: str          # ISO format
    pattern: str
    conviction: str         # BULLISH | BEARISH | MIXED | UNCERTAIN | ""
    tactical_score: int
    options_score: float
    regime_label: str
    quadrant: str
    n_ok: int
    n_total: int
    engine: str             # Freeloader / Regard / Highly Regarded


def load_qir_history() -> list[dict]:
    """Load run history from disk. Returns list of dicts, newest first."""
    path = os.path.normpath(_HISTORY_PATH)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(reversed(data)) if isinstance(data, list) else []
    except Exception:
        return []


def append_qir_run(record: QIRRunRecord) -> None:
    """Append a run record, keeping only the last _MAX_RUNS entries."""
    path = os.path.normpath(_HISTORY_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(asdict(record))
    if len(existing) > _MAX_RUNS:
        existing = existing[-_MAX_RUNS:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
