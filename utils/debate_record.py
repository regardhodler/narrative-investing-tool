"""Commander Wincyl's Court Record — debate verdict ledger with outcome tracking.

Stores every debate verdict in SQLite. Auto-resolves verdicts after 5 days
by comparing SPX price at verdict time vs current price.

This is how Commander Wincyl learns (or doesn't) — the record is injected into
every future debate prompt so the AI knows its track record.
"""

from __future__ import annotations

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "debate_record.db"


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verdicts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,
            verdict       TEXT    NOT NULL,
            confidence    INTEGER NOT NULL DEFAULT 5,
            regime        TEXT,
            quadrant      TEXT,
            regime_score  REAL,
            spx_at_verdict REAL,
            spx_at_resolve REAL,
            outcome       TEXT    DEFAULT 'pending',
            resolved_at   TEXT
        )
    """)
    conn.commit()
    return conn


def _get_spx_price() -> float | None:
    try:
        import yfinance as yf
        h = yf.Ticker("^GSPC").history(period="1d", interval="1m")
        if h is not None and not h.empty:
            return round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


def log_verdict(
    verdict: str,
    confidence: int,
    regime: str = "",
    quadrant: str = "",
    regime_score: float = 0.0,
) -> int:
    """Log a new debate verdict. Returns the row id."""
    spx = _get_spx_price()
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO verdicts (ts, verdict, confidence, regime, quadrant, regime_score, spx_at_verdict) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), verdict, confidence, regime, quadrant, regime_score, spx),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def resolve_old_verdicts() -> int:
    """Auto-resolve verdicts older than 5 days using SPX price comparison.

    Returns number of verdicts resolved.
    """
    cutoff = (datetime.now() - timedelta(days=5)).isoformat()
    conn = _conn()
    pending = conn.execute(
        "SELECT id, ts, verdict, spx_at_verdict FROM verdicts "
        "WHERE outcome = 'pending' AND verdict != 'CONTESTED' AND ts < ?",
        (cutoff,),
    ).fetchall()

    if not pending:
        conn.close()
        return 0

    spx_now = _get_spx_price()
    if spx_now is None:
        conn.close()
        return 0

    resolved = 0
    for row in pending:
        spx_then = row["spx_at_verdict"]
        if spx_then is None:
            continue
        went_up = spx_now > spx_then
        if row["verdict"] == "BULL WINS":
            outcome = "correct" if went_up else "wrong"
        elif row["verdict"] == "BEAR WINS":
            outcome = "correct" if not went_up else "wrong"
        else:
            continue

        conn.execute(
            "UPDATE verdicts SET outcome=?, spx_at_resolve=?, resolved_at=? WHERE id=?",
            (outcome, spx_now, datetime.now().isoformat(), row["id"]),
        )
        resolved += 1

    conn.commit()
    conn.close()
    return resolved


def get_recent_verdicts(n: int = 10) -> list[dict]:
    """Return the n most recent verdicts."""
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT * FROM verdicts ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_stats() -> dict:
    """Return Commander Wincyl's win/loss/accuracy record."""
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT outcome, COUNT(*) as cnt FROM verdicts GROUP BY outcome"
        ).fetchall()
        conn.close()
        counts = {r["outcome"]: r["cnt"] for r in rows}
        correct  = counts.get("correct", 0)
        wrong    = counts.get("wrong", 0)
        pending  = counts.get("pending", 0)
        total_resolved = correct + wrong
        accuracy = round(correct / total_resolved * 100) if total_resolved > 0 else None
        return {
            "correct": correct,
            "wrong": wrong,
            "pending": pending,
            "total": correct + wrong + pending,
            "accuracy_pct": accuracy,
        }
    except Exception:
        return {"correct": 0, "wrong": 0, "pending": 0, "total": 0, "accuracy_pct": None}


def get_record_summary() -> str:
    """One-line record string for prompt injection — e.g. '7W-3L (70% accuracy, 4 pending)'"""
    s = get_stats()
    if s["total"] == 0:
        return "No prior verdicts on record."
    acc = f"{s['accuracy_pct']}% accuracy" if s["accuracy_pct"] is not None else "unresolved"
    return (
        f"Commander Wincyl court record: {s['correct']}W-{s['wrong']}L "
        f"({acc}, {s['pending']} pending resolution)"
    )
