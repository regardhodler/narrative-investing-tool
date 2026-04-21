"""Thesis tracker — persists generated nth-order theses and scores them forward.

Each saved thesis becomes a list of ticker-level predictions with entry price,
upside target, duration, order level, and timestamps. `score_open_theses()`
prices open tickers against live data and computes a rolling hit-rate per order
level — the user's personal track record on this tool.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date
from uuid import uuid4

_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "thesis_tracker.json"
)


def _load() -> list[dict]:
    if not os.path.exists(_FILE):
        return []
    try:
        with open(_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    with open(_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def save_thesis(thesis: dict, entry_prices: dict[str, float]) -> str:
    """Persist a generated thesis with per-ticker entries for forward tracking.

    thesis: full output from generate_nth_order_thesis()
    entry_prices: {ticker: price_at_generation}
    Returns the thesis_id.
    """
    thesis_id = str(uuid4())
    generated_at = thesis.get("generated_at") or datetime.utcnow().isoformat()
    primary = thesis.get("primary", "")

    tickers_flat: list[dict] = []
    for play in thesis.get("orders", []):
        order_level = int(play.get("order", 2))
        name = play.get("name", "")
        upside_range = play.get("upside_pct_range") or [0, 0]
        downside_range = play.get("downside_pct_range") or [0, 0]
        duration = int(play.get("duration_months") or 12)
        prob = int(play.get("probability_score") or 0)
        for tk in play.get("tickers") or []:
            tk_u = tk.upper()
            ep = entry_prices.get(tk_u) or entry_prices.get(tk)
            if not ep or ep <= 0:
                continue
            tickers_flat.append({
                "ticker":             tk_u,
                "order":              order_level,
                "play_name":          name,
                "entry_price":        float(ep),
                "upside_pct_low":     float(upside_range[0] if upside_range else 0),
                "upside_pct_high":    float(upside_range[-1] if upside_range else 0),
                "downside_pct_low":   float(downside_range[0] if downside_range else 0),
                "downside_pct_high":  float(downside_range[-1] if downside_range else 0),
                "duration_months":    duration,
                "probability_score":  prob,
                "status":             "open",
            })

    entry = {
        "thesis_id":     thesis_id,
        "primary":       primary,
        "generated_at":  generated_at,
        "generated_date": str(date.today()),
        "regime_snapshot": thesis.get("regime_snapshot", {}),
        "tickers":       tickers_flat,
    }
    all_entries = _load()
    all_entries.append(entry)
    _save(all_entries)
    return thesis_id


def score_open_theses(live_prices: dict[str, float]) -> dict:
    """Score every open ticker entry against live prices.

    Returns aggregate hit-rate stats per order level plus the raw scored entries.
    A ticker "hits" if current return ≥ upside_pct_low. It "fails" if current
    return ≤ -downside_pct_high. Otherwise it remains open. Duration gates
    closure: once duration_months has elapsed, the ticker is marked resolved
    as hit/fail based on the final return.
    """
    all_entries = _load()
    today = date.today()

    by_order: dict[int, dict] = {
        2: {"n": 0, "hits": 0, "misses": 0, "open": 0, "returns": []},
        3: {"n": 0, "hits": 0, "misses": 0, "open": 0, "returns": []},
        4: {"n": 0, "hits": 0, "misses": 0, "open": 0, "returns": []},
    }

    for entry in all_entries:
        try:
            gen_date = date.fromisoformat(entry.get("generated_date") or "")
        except Exception:
            continue
        days_held = (today - gen_date).days

        for tk_entry in entry.get("tickers", []):
            tk = tk_entry["ticker"]
            order = tk_entry.get("order", 2)
            if order not in by_order:
                continue
            ep = tk_entry.get("entry_price") or 0
            cp = live_prices.get(tk)
            if not ep or ep <= 0 or cp is None or cp <= 0:
                continue

            ret_pct = (cp - ep) / ep * 100.0
            up_low = tk_entry.get("upside_pct_low", 0)
            down_high = tk_entry.get("downside_pct_high", 0)
            duration_days = int(tk_entry.get("duration_months", 12)) * 30

            by_order[order]["n"] += 1
            by_order[order]["returns"].append(ret_pct)

            if ret_pct >= up_low:
                by_order[order]["hits"] += 1
                tk_entry["status"] = "hit"
                tk_entry["final_return_pct"] = round(ret_pct, 1)
            elif ret_pct <= -down_high:
                by_order[order]["misses"] += 1
                tk_entry["status"] = "miss"
                tk_entry["final_return_pct"] = round(ret_pct, 1)
            elif days_held >= duration_days:
                if ret_pct > 0:
                    by_order[order]["hits"] += 1
                    tk_entry["status"] = "hit_time"
                else:
                    by_order[order]["misses"] += 1
                    tk_entry["status"] = "miss_time"
                tk_entry["final_return_pct"] = round(ret_pct, 1)
            else:
                by_order[order]["open"] += 1
                tk_entry["status"] = "open"
                tk_entry["current_return_pct"] = round(ret_pct, 1)

    _save(all_entries)

    stats_per_order: dict[int, dict] = {}
    for order, s in by_order.items():
        closed = s["hits"] + s["misses"]
        hit_rate = round(s["hits"] / closed * 100, 1) if closed > 0 else None
        avg_return = round(sum(s["returns"]) / len(s["returns"]), 1) if s["returns"] else None
        stats_per_order[order] = {
            "n_total":   s["n"],
            "n_open":    s["open"],
            "n_closed":  closed,
            "hits":      s["hits"],
            "misses":    s["misses"],
            "hit_rate":  hit_rate,
            "avg_return_pct": avg_return,
        }
    return {
        "per_order":   stats_per_order,
        "total_theses": len(all_entries),
    }


def get_track_record_summary(live_prices: dict[str, float] | None = None) -> dict:
    """Lightweight read of the tracker for UI display. Optionally rescores if
    live_prices provided."""
    if live_prices:
        return score_open_theses(live_prices)
    # Fast path: just count
    all_entries = _load()
    by_order: dict[int, int] = {2: 0, 3: 0, 4: 0}
    for e in all_entries:
        for tk in e.get("tickers", []):
            o = tk.get("order", 2)
            if o in by_order:
                by_order[o] += 1
    return {
        "per_order":   {o: {"n_total": n, "n_open": n, "n_closed": 0, "hits": 0, "misses": 0, "hit_rate": None, "avg_return_pct": None} for o, n in by_order.items()},
        "total_theses": len(all_entries),
    }


def match_analog(analog_name: str) -> dict | None:
    """Find the best matching analog from data/analog_library.json for a
    free-form analog name the LLM produced. Returns the analog dict or None."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "analog_library.json"
    )
    try:
        with open(path) as f:
            lib = json.load(f)
    except Exception:
        return None

    analogs = lib.get("analogs", {})
    if not analogs or not analog_name:
        return None

    norm = analog_name.lower()
    # Score each analog by word overlap with its name and key/slug
    best = None
    best_score = 0
    for slug, a in analogs.items():
        score = 0
        for word in (slug.replace("_", " ")).lower().split():
            if word in norm:
                score += 2
        for word in a.get("name", "").lower().split():
            if len(word) >= 4 and word in norm:
                score += 1
        if score > best_score:
            best_score = score
            best = dict(a)
            best["slug"] = slug
    return best if best_score >= 2 else None
