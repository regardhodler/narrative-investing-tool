"""Rolling history store for options flow metrics (P/C ratio, IV skew).

Used for z-score normalization in VIX-adaptive scoring.
Max 30 rows per metric — older rows are pruned automatically.
"""
import sqlite3, os, datetime
from typing import Optional

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "options_flow_history.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=5)
    c.execute("""
        CREATE TABLE IF NOT EXISTS flow_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL
        )
    """)
    c.commit()
    return c


def append_reading(metric: str, value: float) -> None:
    """Append a new reading and prune to last 30 rows per metric."""
    with _conn() as c:
        c.execute(
            "INSERT INTO flow_history (ts, metric, value) VALUES (?, ?, ?)",
            (datetime.datetime.utcnow().isoformat(), metric, value)
        )
        # Keep only last 30 readings per metric
        c.execute("""
            DELETE FROM flow_history
            WHERE metric = ? AND id NOT IN (
                SELECT id FROM flow_history WHERE metric = ?
                ORDER BY id DESC LIMIT 30
            )
        """, (metric, metric))
        c.commit()


def get_history(metric: str) -> list[float]:
    """Return list of recent values for a metric, oldest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT value FROM flow_history WHERE metric = ? ORDER BY id ASC",
            (metric,)
        ).fetchall()
    return [r[0] for r in rows]


def compute_ewma_stats(values: list[float], span: int = 20) -> tuple[float, float]:
    """Compute EWMA mean and std with given span. Returns (mean, std)."""
    if not values:
        return 0.0, 1.0
    alpha = 2.0 / (span + 1)
    ewma = values[0]
    sq_ewma = values[0] ** 2
    for v in values[1:]:
        ewma = alpha * v + (1 - alpha) * ewma
        sq_ewma = alpha * (v ** 2) + (1 - alpha) * sq_ewma
    variance = max(sq_ewma - ewma ** 2, 1e-8)
    return ewma, variance ** 0.5


def zscore(value: float, metric: str, span: int = 20) -> tuple[float, int]:
    """Compute z-score of value vs EWMA history. Returns (z, n_samples)."""
    hist = get_history(metric)
    if len(hist) < 5:
        return 0.0, len(hist)  # insufficient history — return neutral
    mu, sigma = compute_ewma_stats(hist, span)
    if sigma < 1e-8:
        return 0.0, len(hist)
    return (value - mu) / sigma, len(hist)


def get_vix_level() -> float:
    """Fetch current VIX level. Returns 20.0 as default if unavailable."""
    try:
        import yfinance as yf
        vx = yf.Ticker("^VIX")
        info = vx.fast_info
        v = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        if v and v > 0:
            return float(v)
    except Exception:
        pass
    return 20.0


def get_vix_regime_weights(vix: float) -> dict:
    """Return signal weights conditioned on VIX level.

    In high-vol (VIX>35): IV skew dominates — it's the most information-dense signal.
    In low-vol (VIX<15): P/C flow dominates.
    In crisis (VIX>50): drop unusual activity (too noisy), max skew weight.
    """
    if vix > 50:
        return {"pc": 1.5, "gamma": 1.0, "skew": 4.5, "unusual": 0.0}
    elif vix > 35:
        return {"pc": 2.0, "gamma": 1.0, "skew": 4.0, "unusual": 1.0}
    elif vix > 25:
        return {"pc": 2.5, "gamma": 1.5, "skew": 3.0, "unusual": 1.0}
    elif vix > 15:
        return {"pc": 3.0, "gamma": 1.5, "skew": 2.0, "unusual": 1.5}
    else:  # low vol < 15
        return {"pc": 3.5, "gamma": 1.5, "skew": 1.5, "unusual": 1.5}
