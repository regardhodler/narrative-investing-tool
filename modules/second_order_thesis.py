"""Nth-Order Thesis Builder — hedge-fund-grade 2nd/3rd/4th-order play finder.

Reads QIR aggregate context (HMM state, CI%, macro score, Current Events
sentiment) and produces non-consensus plays one to three steps removed from the
user's active narrative. Every play carries a composite probability score, a
Kelly-sized suggestion, a steel-manned bear case, disconfirming evidence, and a
historical analog with real back-test numbers.

This module reads — it does not recompute. All signals come from session state
and existing persisted files. The module contributes its own output to
`data/thesis_tracker.json` for forward scoring.
"""
from __future__ import annotations

import json
import os
import streamlit as st
from datetime import datetime, timedelta, date
from html import escape

from utils.session import get_narrative, set_ticker
from utils.theme import COLORS
from utils.ai_tier import render_ai_tier_selector
from utils.journal import add_trade


# ─────────────────────────────────────────────────────────────────────────────
# Persistent on-disk thesis cache — survives restarts, avoids re-running LLM
# ─────────────────────────────────────────────────────────────────────────────

_THESIS_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "nth_thesis_cache.json"
)


def _load_disk_cache() -> dict:
    if not os.path.exists(_THESIS_CACHE_FILE):
        return {}
    try:
        with open(_THESIS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_disk_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_THESIS_CACHE_FILE), exist_ok=True)
        with open(_THESIS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
    except Exception:
        pass


def _cache_put(narrative_key: str, thesis: dict, engine_label: str) -> None:
    cache = _load_disk_cache()
    cache[narrative_key] = {
        "saved_at":    datetime.utcnow().isoformat(),
        "engine":      engine_label,
        "thesis":      thesis,
    }
    _save_disk_cache(cache)


def _cache_get(narrative_key: str) -> dict | None:
    return _load_disk_cache().get(narrative_key)


def _cache_delete(narrative_key: str) -> None:
    cache = _load_disk_cache()
    if narrative_key in cache:
        cache.pop(narrative_key)
        _save_disk_cache(cache)


# ─────────────────────────────────────────────────────────────────────────────
# Regime context builder — reads from session state / data files (no compute)
# ─────────────────────────────────────────────────────────────────────────────

def _build_regime_ctx() -> dict:
    """Aggregate the current regime context from session state + HMM brain file."""
    ctx: dict = {}

    # HMM state from brain file
    try:
        from services.hmm_regime import load_current_hmm_state
        hmm = load_current_hmm_state()
        if hmm:
            ctx["hmm_state"] = hmm.state_label
            ctx["hmm_confidence"] = round(getattr(hmm, "confidence", 0) * 100)
            ll_z = getattr(hmm, "ll_zscore", 0)
            from services.hmm_regime import get_ci_anchor as _gca
            ci_pct = abs(ll_z) / max(_gca(), 1e-6) * 100 if ll_z else 0
            ctx["ci_pct"] = round(ci_pct, 1)
            ctx["_hmm_stale"] = getattr(hmm, "_is_stale", False)
    except Exception:
        pass

    # Full macro scorecard (written by Macro Scorecard module)
    msc = st.session_state.get("_macro_scorecard") or {}
    if msc:
        ctx["macro_regime"] = msc.get("regime", "")
        ctx["macro_score"] = msc.get("total_score")
        ctx["macro_factors"] = msc.get("factors", {})
        ctx["macro_trigger"] = msc.get("trigger", "")
        ctx["macro_trigger_confidence"] = msc.get("trigger_confidence", "")
        ctx["macro_actions"] = msc.get("actions", [])
    else:
        # Fallback: scalar from QIR / risk regime
        for key in ("_macro_score", "_regime_score", "_risk_regime_score"):
            v = st.session_state.get(key)
            if v is not None:
                try:
                    ctx["macro_score"] = float(v)
                    break
                except Exception:
                    continue

    # Contagion score (from risk regime or correlation monitor)
    for key in ("_contagion_score", "_correlation_contagion"):
        v = st.session_state.get(key)
        if v is not None:
            try:
                ctx["contagion_score"] = float(v)
                break
            except Exception:
                continue

    # Current Events digest + sentiment
    ctx["events_digest"] = st.session_state.get("_current_events_digest", "") or ""
    ctx["events_sentiment"] = st.session_state.get("_events_sentiment_score", {}) or {}
    ts = st.session_state.get("_current_events_digest_ts")
    ctx["events_ts"] = ts

    return ctx


def _events_freshness_hours(ts) -> float | None:
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            t = ts
        else:
            return None
        delta = datetime.now() - t.replace(tzinfo=None) if t.tzinfo else datetime.now() - t
        return round(delta.total_seconds() / 3600.0, 1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dependability layers — computed in Python (not LLM self-report)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_crowding(ticker: str) -> dict:
    """Fetch short_pct_float + analyst_count and derive low/med/high crowding."""
    try:
        from services.market_data import fetch_ticker_fundamentals
        f = fetch_ticker_fundamentals(ticker)
    except Exception:
        return {"label": None, "short_pct_float": None, "analyst_count": None}
    short_pct = f.get("short_pct_float")
    analyst_n = f.get("analyst_count")
    short_pct_100 = (short_pct * 100) if (short_pct is not None and short_pct < 1.5) else short_pct

    label = None
    if analyst_n is not None and short_pct_100 is not None:
        if analyst_n <= 5 and short_pct_100 < 3:
            label = "low"
        elif analyst_n >= 20 or short_pct_100 > 8:
            label = "high"
        else:
            label = "medium"
    elif analyst_n is not None:
        label = "low" if analyst_n <= 5 else ("high" if analyst_n >= 20 else "medium")
    elif short_pct_100 is not None:
        label = "low" if short_pct_100 < 3 else ("high" if short_pct_100 > 8 else "medium")
    return {
        "label":           label,
        "short_pct_float": round(short_pct_100, 2) if short_pct_100 is not None else None,
        "analyst_count":   analyst_n,
    }


def _aggregate_crowding(tickers: list[str]) -> dict:
    """Average crowding across the tickers in a play."""
    if not tickers:
        return {"label": None, "n": 0}
    results = [_compute_crowding(t) for t in tickers[:5]]
    labels = [r["label"] for r in results if r.get("label")]
    if not labels:
        return {"label": None, "n": 0, "components": results}
    # Mode
    counts: dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    label = max(counts, key=counts.get)
    avg_short = [r["short_pct_float"] for r in results if r.get("short_pct_float") is not None]
    avg_analysts = [r["analyst_count"] for r in results if r.get("analyst_count") is not None]
    return {
        "label":       label,
        "n":           len(labels),
        "avg_short":   round(sum(avg_short) / len(avg_short), 1) if avg_short else None,
        "avg_analysts": int(round(sum(avg_analysts) / len(avg_analysts))) if avg_analysts else None,
        "components":  results,
    }


def _regime_alignment(play: dict, regime_ctx: dict) -> int:
    """Rule-based -2..+2 alignment score between a play and the current regime."""
    hmm = (regime_ctx.get("hmm_state") or "").lower()
    size = (play.get("size_bucket") or "").lower()
    moonshot = bool(play.get("moonshot"))
    order = int(play.get("order", 2))

    score = 0
    if "bull" in hmm:
        score += 2 if order == 2 else (1 if order == 3 else 0)
        if moonshot:
            score += 1
    elif "neutral" in hmm:
        score += 1 if order in (2, 3) else 0
    elif "late cycle" in hmm:
        # Late cycle favors 2nd/3rd-order capex beneficiaries, not tails
        score += 2 if order == 2 else (1 if order == 3 else -1)
        if moonshot:
            score -= 2
    elif "stress" in hmm or "early stress" in hmm:
        score += 0 if order == 2 else (-1 if order >= 3 else 0)
        if moonshot:
            score -= 2
    elif "crisis" in hmm:
        # Crisis: avoid tails entirely
        score += -1
        if moonshot or order == 4:
            score -= 2

    # Events sentiment modulation
    sent = (regime_ctx.get("events_sentiment") or {}).get("sentiment")
    if sent is not None:
        try:
            sent_f = float(sent)
            if sent_f < -0.3 and moonshot:
                score -= 1
            if sent_f > 0.3 and order in (3, 4):
                score += 1
        except Exception:
            pass

    return max(-3, min(3, score))


def _probability_score(play: dict) -> dict:
    """Composite 0-100 probability from the dependability layers."""
    # conviction_pct — from N-run ensemble (0-100)
    conviction = float(play.get("conviction_pct") or 50)

    # regime_alignment — -3..+3 mapped to 0..100
    ra = play.get("regime_alignment", 0)
    ra_pct = max(0.0, min(100.0, (ra + 3) / 6 * 100))

    # crowding_inverse — low=85, medium=55, high=25
    crowding = (play.get("crowding_label") or "medium").lower()
    ci_pct = {"low": 85.0, "medium": 55.0, "high": 25.0}.get(crowding, 50.0)

    # analog_hit_rate — "3 of 4" → 75
    ahr_str = play.get("analog_hit_rate_str") or ""
    ahr_pct = 50.0
    if ahr_str:
        try:
            parts = ahr_str.lower().split("of")
            if len(parts) == 2:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den > 0:
                    ahr_pct = num / den * 100
        except Exception:
            pass

    # evidence_balance — if evidence_against is non-trivial, penalize
    ea = play.get("evidence_against") or []
    ea_real = [e for e in ea if e and "no disconfirming" not in str(e).lower()]
    if len(ea_real) == 0:
        eb_pct = 65.0  # either too confident or LLM skipped it — moderate penalty
    elif len(ea_real) <= 2:
        eb_pct = 55.0  # acknowledged real counter-evidence
    else:
        eb_pct = 35.0  # lots of counter-evidence — thesis weaker

    composite = (
        conviction * 0.25 +
        ra_pct     * 0.25 +
        ci_pct     * 0.15 +
        ahr_pct    * 0.20 +
        eb_pct     * 0.15
    )
    return {
        "probability":   int(round(composite)),
        "components": {
            "conviction":    int(round(conviction)),
            "regime_align":  int(round(ra_pct)),
            "crowding_inv":  int(round(ci_pct)),
            "analog_hits":   int(round(ahr_pct)),
            "evidence_bal":  int(round(eb_pct)),
        },
    }


def _kelly_size(probability_pct: int, upside_pct: float, downside_pct: float) -> float:
    """Simple Kelly fraction in % of portfolio. Half-Kelly, capped at 15%."""
    if probability_pct <= 0 or downside_pct <= 0 or upside_pct <= 0:
        return 0.0
    p = probability_pct / 100.0
    b = upside_pct / downside_pct
    q = 1.0 - p
    kelly_full = (b * p - q) / b
    if kelly_full <= 0:
        return 0.0
    half = kelly_full * 0.5
    return round(min(half * 100, 15.0), 1)


def _bucket_from_kelly(kelly_pct: float, moonshot: bool) -> str:
    if moonshot:
        return "tail"
    if kelly_pct >= 3.0:
        return "core"
    if kelly_pct >= 1.0:
        return "trading"
    return "tail"


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

MONO = "'JetBrains Mono', Consolas, Courier New, monospace"


def _pill(text: str, bg: str, fg: str = "#fff", border: str | None = None) -> str:
    bd = f"border:1px solid {border};" if border else ""
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};{bd}'
        f'font-family:{MONO};font-size:10px;font-weight:700;letter-spacing:0.04em;'
        f'padding:3px 8px;border-radius:3px;margin-right:6px;text-transform:uppercase;">'
        f'{escape(text)}</span>'
    )


def _crowding_pill(label: str | None) -> str:
    if not label:
        return _pill("CROWDING ?", COLORS["surface"], COLORS["text_dim"], COLORS["border"])
    color = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}.get(label.lower(), COLORS["text_dim"])
    return _pill(f"CROWDING {label.upper()}", color, "#0b0d10")


def _probability_pill(pct: int) -> str:
    if pct >= 70:
        color = "#22c55e"
    elif pct >= 50:
        color = "#f59e0b"
    elif pct >= 30:
        color = "#f97316"
    else:
        color = "#ef4444"
    return _pill(f"PROB {pct}/100", color, "#0b0d10")


def _order_badge(order: int) -> str:
    bg = {2: "#4B9FFF", 3: "#8b5cf6", 4: "#ec4899"}.get(order, COLORS["text_dim"])
    return _pill(f"{order}ND ORDER" if order == 2 else (f"{order}RD ORDER" if order == 3 else f"{order}TH ORDER"), bg, "#0b0d10")


def _size_pill(bucket: str, kelly_pct: float | None) -> str:
    color_map = {"core": "#00D4AA", "trading": "#FFD700", "tail": "#8b5cf6"}
    color = color_map.get(bucket.lower(), COLORS["text_dim"])
    text = f"{bucket.upper()}"
    if kelly_pct is not None:
        text += f" · K {kelly_pct:.1f}%"
    return _pill(text, color, "#0b0d10")


def _moonshot_pill() -> str:
    return _pill("MOONSHOT", "#ec4899", "#0b0d10")


def _regime_incoherent_pill() -> str:
    return _pill("REGIME-INCOHERENT", "#ef4444", "#fff")


# ─────────────────────────────────────────────────────────────────────────────
# Context strip
# ─────────────────────────────────────────────────────────────────────────────

def _render_context_strip(regime_ctx: dict, narrative: str | None):
    hmm = regime_ctx.get("hmm_state") or "—"
    ci = regime_ctx.get("ci_pct")
    ms = regime_ctx.get("macro_score")
    sent = (regime_ctx.get("events_sentiment") or {}).get("sentiment")
    theme = (regime_ctx.get("events_sentiment") or {}).get("dominant_theme")
    hours = _events_freshness_hours(regime_ctx.get("events_ts"))

    # Freshness badge color
    if hours is None:
        fresh_color, fresh_text = "#ef4444", "NO NEWS DIGEST"
    elif hours > 24:
        fresh_color, fresh_text = "#f59e0b", f"TAPE {hours:.0f}H OLD"
    else:
        fresh_color, fresh_text = "#22c55e", f"TAPE {hours:.0f}H OLD"

    def _item(label, value, color=None):
        col = color or COLORS["text"]
        return (
            f'<div style="display:inline-block;margin-right:14px;'
            f'font-family:{MONO};font-size:10px;">'
            f'<span style="color:{COLORS["text_dim"]};">{label}</span> '
            f'<span style="color:{col};font-weight:700;">{value}</span></div>'
        )

    parts = []
    parts.append(_item("NARRATIVE", (narrative or "—").upper()))
    parts.append(_item("HMM", hmm.upper(), COLORS["bloomberg_orange"]))
    if ci is not None:
        ci_color = "#22c55e" if ci < 22 else ("#f59e0b" if ci < 67 else "#ef4444")
        parts.append(_item("CI%", f"{ci:.1f}", ci_color))
    if ms is not None:
        parts.append(_item("MACRO", f"{ms:+.1f}" if abs(ms) < 10 else f"{ms:.0f}"))
    if sent is not None:
        s_color = "#22c55e" if sent > 0.15 else ("#ef4444" if sent < -0.15 else COLORS["text"])
        parts.append(_item("EVENTS", f"{sent:+.2f}", s_color))
    if theme:
        parts.append(_item("THEME", escape(str(theme)[:24])))
    parts.append(_item("", fresh_text, fresh_color))

    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-left:3px solid {COLORS["bloomberg_orange"]};padding:10px 14px;'
        f'border-radius:3px;margin-bottom:14px;">' + "".join(parts) + '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regime template card
# ─────────────────────────────────────────────────────────────────────────────

def _render_regime_template(rt: dict):
    if not rt:
        return
    name = rt.get("analog_name") or rt.get("analog_slug") or "—"
    period = rt.get("analog_period") or ""
    reasoning = rt.get("reasoning") or ""
    ret = rt.get("analog_return_pct")
    dur = rt.get("analog_duration_months")
    dd = rt.get("analog_max_drawdown")
    hr = rt.get("analog_hit_rate")
    note = rt.get("analog_note")

    stat = ""
    if any(v is not None for v in (ret, dur, dd, hr)):
        s_items = []
        if ret is not None:
            col = "#22c55e" if ret > 0 else "#ef4444"
            s_items.append(f'<span style="color:{COLORS["text_dim"]};">RETURN</span> <span style="color:{col};font-weight:700;">{ret:+.0f}%</span>')
        if dur is not None:
            s_items.append(f'<span style="color:{COLORS["text_dim"]};">DURATION</span> <span style="color:{COLORS["text"]};font-weight:700;">{dur} mo</span>')
        if dd is not None:
            s_items.append(f'<span style="color:{COLORS["text_dim"]};">MAX DD</span> <span style="color:#ef4444;font-weight:700;">{dd:.0f}%</span>')
        if hr:
            s_items.append(f'<span style="color:{COLORS["text_dim"]};">HIT RATE</span> <span style="color:{COLORS["text"]};font-weight:700;">{escape(str(hr))}</span>')
        stat = (
            f'<div style="font-family:{MONO};font-size:11px;margin-top:8px;'
            f'padding-top:8px;border-top:1px dashed {COLORS["border"]};'
            f'display:flex;gap:18px;flex-wrap:wrap;">' + " · ".join(s_items) + '</div>'
        )

    text_dim = COLORS["text_dim"]
    text_col = COLORS["text"]
    surface = COLORS["surface"]
    border = COLORS["border"]
    period_html = (
        f' <span style="color:{text_dim};font-weight:400;font-size:12px;">({escape(period)})</span>'
        if period else ""
    )
    note_html = (
        f'<div style="font-family:{MONO};font-size:11px;color:{text_dim};'
        f'font-style:italic;margin-top:8px;line-height:1.5;">{escape(note)}</div>'
        if note else ""
    )
    st.markdown(
        f'<div style="background:{surface};border:1px solid {border};'
        f'border-left:3px solid #f59e0b;padding:14px 16px;border-radius:3px;'
        f'margin-bottom:14px;">'
        f'<div style="font-family:{MONO};font-size:10px;color:#f59e0b;'
        f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">'
        f'REGIME TEMPLATE · DALIO ANALOG</div>'
        f'<div style="font-family:{MONO};font-size:15px;color:{text_col};'
        f'font-weight:700;margin-bottom:4px;">{escape(name)}{period_html}</div>'
        f'<div style="font-family:{MONO};font-size:12px;color:{text_col};'
        f'line-height:1.5;">{escape(reasoning)}</div>'
        f'{stat}{note_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ticker quick-card (shown inside st.popover)
# ─────────────────────────────────────────────────────────────────────────────

def _render_ticker_quick_card(ticker: str) -> None:
    """Compact Bloomberg-style fundamentals card rendered inside a popover."""
    from services.market_data import fetch_ticker_fundamentals, get_yf_info_safe

    def _fmt(v, fmt="{:.1f}", suffix="", prefix=""):
        if v is None or (isinstance(v, float) and v != v):
            return '<span style="color:#475569;">—</span>'
        try:
            return f"{prefix}{fmt.format(float(v))}{suffix}"
        except Exception:
            return '<span style="color:#475569;">—</span>'

    info = get_yf_info_safe(ticker) or {}
    fund = fetch_ticker_fundamentals(ticker) or {}

    name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or ""
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    w52_lo = info.get("fiftyTwoWeekLow")
    w52_hi = info.get("fiftyTwoWeekHigh")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    day_chg = ((price - prev_close) / prev_close * 100) if price and prev_close else None
    desc = (info.get("longBusinessSummary") or "")[:240]

    pe_t  = fund.get("pe_trailing")
    pe_f  = fund.get("pe_forward")
    pb    = fund.get("pb_ratio")
    peg   = fund.get("peg")
    ps    = fund.get("ps_ratio")
    ev_eb = fund.get("ev_ebitda")
    roe   = fund.get("roe")
    fcfy  = fund.get("fcf_yield")
    de    = fund.get("debt_to_equity")
    tgt   = fund.get("target_mean")
    n_an  = fund.get("analyst_count")

    arrow = "▲" if (day_chg or 0) >= 0 else "▼"
    chg_color = "#22c55e" if (day_chg or 0) >= 0 else "#ef4444"
    implied_up = ((tgt - price) / price * 100) if tgt and price and price > 0 else None

    price_str = f"${price:,.2f}" if price else "—"
    chg_str = (
        f'<span style="color:{chg_color};font-weight:700;">{arrow} {day_chg:+.1f}%</span>'
        if day_chg is not None else ""
    )
    range_str = (
        f'52w: <span style="color:#94a3b8;">${w52_lo:.2f} – ${w52_hi:.2f}</span>'
        if w52_lo and w52_hi else ""
    )

    tgt_str = ""
    if tgt:
        n_str = f" (n={n_an})" if n_an else ""
        up_str = f' → <span style="color:{"#22c55e" if (implied_up or 0)>0 else "#ef4444"};font-weight:700;">{implied_up:+.1f}%</span>' if implied_up is not None else ""
        tgt_str = (
            f'<div style="font-family:{MONO};font-size:11px;margin-top:6px;">'
            f'<span style="color:{COLORS["text_dim"]};">STREET TARGET</span> '
            f'<span style="color:{COLORS["text"]};font-weight:700;">${tgt:.2f}{n_str}</span>{up_str}</div>'
        )

    desc_html = (
        f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text_dim"]};'
        f'line-height:1.5;margin-top:8px;padding-top:8px;'
        f'border-top:1px dashed {COLORS["border"]};">{escape(desc)}{"…" if len(desc)==240 else ""}</div>'
    ) if desc else ""

    def _cell(label, val_html):
        return (
            f'<div style="min-width:80px;">'
            f'<div style="font-size:9px;color:{COLORS["text_dim"]};letter-spacing:0.06em;">{label}</div>'
            f'<div style="font-size:13px;color:{COLORS["text"]};font-weight:700;">{val_html}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="font-family:{MONO};padding:4px 0 8px 0;">'
        # Header
        f'<div style="font-size:15px;font-weight:700;color:{COLORS["text"]};">{escape(name)}'
        f'{"  <span style=\"font-size:11px;color:" + COLORS["text_dim"] + ";font-weight:400;\">· " + escape(sector) + "</span>" if sector else ""}'
        f'</div>'
        # Price row
        f'<div style="font-size:12px;margin-top:2px;">'
        f'<span style="color:{COLORS["text"]};font-weight:700;">{price_str}</span> '
        f'{chg_str}{"  " if chg_str else ""}{range_str}</div>'
        # Divider
        f'<div style="border-top:1px solid {COLORS["border"]};margin:8px 0;"></div>'
        # Multiples grid row 1
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
        f'{_cell("PE (ttm)", _fmt(pe_t))}'
        f'{_cell("PE (fwd)", _fmt(pe_f))}'
        f'{_cell("PB", _fmt(pb))}'
        f'</div>'
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">'
        f'{_cell("PEG", _fmt(peg))}'
        f'{_cell("PS", _fmt(ps))}'
        f'{_cell("EV/EBITDA", _fmt(ev_eb, "{:.1f}x"))}'
        f'</div>'
        # Divider
        f'<div style="border-top:1px solid {COLORS["border"]};margin:8px 0;"></div>'
        # Quality row
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
        f'{_cell("ROE", _fmt(roe, "{:.1f}%"))}'
        f'{_cell("FCF yield", _fmt(fcfy, "{:.1f}%"))}'
        f'{_cell("D/E", _fmt(de))}'
        f'</div>'
        f'{tgt_str}{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Set as active ticker →",
        key=f"_quick_set_{ticker}_{id(ticker)}",
        use_container_width=True,
        type="primary",
    ):
        set_ticker(ticker)
        st.toast(f"Active ticker set to {ticker}", icon="🎯")


# ─────────────────────────────────────────────────────────────────────────────
# Play card
# ─────────────────────────────────────────────────────────────────────────────

def _render_play_card(play: dict, key_prefix: str):
    name = play.get("name", "—")
    order = int(play.get("order", 2))
    moonshot = bool(play.get("moonshot"))
    regime_align = play.get("regime_alignment", 0)
    prob_pct = play.get("probability_score", 50)
    kelly_pct = play.get("kelly_pct")
    size_bucket = play.get("size_bucket") or "trading"
    crowding_label = play.get("crowding_label")

    # Header
    header_pills = _order_badge(order) + _probability_pill(prob_pct) + _crowding_pill(crowding_label) + _size_pill(size_bucket, kelly_pct)
    if moonshot:
        header_pills += _moonshot_pill()
    if regime_align < 0:
        header_pills += _regime_incoherent_pill()

    # Build a container
    with st.container():
        st.markdown(
            f'<div style="background:{COLORS["surface_dark"]};border:1px solid {COLORS["border"]};'
            f'border-radius:4px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
            f'margin-bottom:10px;flex-wrap:wrap;gap:6px;">'
            f'<div style="font-family:{MONO};font-size:15px;color:{COLORS["text"]};'
            f'font-weight:700;flex:1;min-width:200px;">{escape(name)}</div>'
            f'<div style="text-align:right;">{header_pills}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Consensus vs Non-Consensus two columns
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div style="background:{COLORS["surface"]};border-left:2px solid {COLORS["text_dim"]};'
                f'padding:10px 12px;border-radius:3px;margin-bottom:8px;">'
                f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["text_dim"]};'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">CONSENSUS</div>'
                f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};'
                f'line-height:1.55;">{escape(play.get("consensus_view",""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="background:{COLORS["surface"]};border-left:2px solid #8b5cf6;'
                f'padding:10px 12px;border-radius:3px;margin-bottom:8px;">'
                f'<div style="font-family:{MONO};font-size:10px;color:#8b5cf6;'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">OUR VIEW · NON-CONSENSUS</div>'
                f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};'
                f'line-height:1.55;">{escape(play.get("non_consensus_view",""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Catalyst path
        cp = play.get("catalyst_path") or []
        if cp:
            cp_html = ""
            for i, step in enumerate(cp, 1):
                cp_html += (
                    f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};'
                    f'line-height:1.55;padding:2px 0;">'
                    f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{i}.</span> '
                    f'{escape(str(step))}</div>'
                )
            st.markdown(
                f'<div style="background:{COLORS["surface"]};padding:10px 12px;border-radius:3px;'
                f'margin-bottom:8px;border:1px solid {COLORS["border"]};">'
                f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["bloomberg_orange"]};'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:6px;">CATALYST PATH</div>'
                f'{cp_html}</div>',
                unsafe_allow_html=True,
            )

        # R/R + Kelly bar
        up = play.get("upside_pct_range") or [0, 0]
        dn = play.get("downside_pct_range") or [0, 0]
        up_mid = (float(up[0]) + float(up[-1])) / 2 if up else 0
        dn_mid = (float(dn[0]) + float(dn[-1])) / 2 if dn else 0
        rr = (up_mid / dn_mid) if dn_mid > 0 else 0
        dur_mo = play.get("duration_months", 0)

        rr_html = (
            f'<div style="background:{COLORS["surface"]};padding:10px 12px;border-radius:3px;'
            f'margin-bottom:8px;border:1px solid {COLORS["border"]};display:flex;'
            f'justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">'
            f'<div style="font-family:{MONO};font-size:12px;">'
            f'<span style="color:#22c55e;font-weight:700;">+{up[0]:.0f}% → +{up[-1]:.0f}%</span>'
            f' <span style="color:{COLORS["text_dim"]};">UP</span>'
            f' &nbsp;·&nbsp; '
            f'<span style="color:#ef4444;font-weight:700;">-{dn[0]:.0f}% → -{dn[-1]:.0f}%</span>'
            f' <span style="color:{COLORS["text_dim"]};">DN</span>'
            f' &nbsp;·&nbsp; '
            f'<span style="color:{COLORS["text"]};font-weight:700;">{dur_mo} mo</span>'
            f'</div>'
            f'<div style="font-family:{MONO};font-size:12px;">'
            f'<span style="color:{COLORS["text_dim"]};">R/R</span> '
            f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{rr:.1f}x</span>'
            f' &nbsp; '
            f'<span style="color:{COLORS["text_dim"]};">KELLY</span> '
            f'<span style="color:#22c55e;font-weight:700;">{(kelly_pct or 0):.1f}%</span>'
            f'</div></div>'
        )
        st.markdown(rr_html, unsafe_allow_html=True)

        # Bear case
        bc = play.get("bear_case") or ""
        if bc:
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.08);border-left:2px solid #ef4444;'
                f'padding:10px 12px;border-radius:3px;margin-bottom:8px;">'
                f'<div style="font-family:{MONO};font-size:10px;color:#ef4444;'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">BEAR CASE</div>'
                f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};'
                f'line-height:1.55;font-style:italic;">{escape(bc)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Evidence against
        ea = play.get("evidence_against") or []
        if ea:
            ea_items = "".join(
                f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};'
                f'line-height:1.5;padding:2px 0;">'
                f'<span style="color:#f59e0b;">▪</span> {escape(str(e))}</div>'
                for e in ea
            )
            st.markdown(
                f'<div style="background:rgba(245,158,11,0.07);border-left:2px solid #f59e0b;'
                f'padding:10px 12px;border-radius:3px;margin-bottom:8px;">'
                f'<div style="font-family:{MONO};font-size:10px;color:#f59e0b;'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">'
                f'EVIDENCE AGAINST · TRUE TODAY</div>'
                f'{ea_items}</div>',
                unsafe_allow_html=True,
            )

        # Data points to watch
        dp = play.get("data_points") or []
        if dp:
            dp_items = "".join(
                f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text"]};'
                f'line-height:1.5;padding:1px 0;">'
                f'<span style="color:{COLORS["text_dim"]};">◦</span> {escape(str(d))}</div>'
                for d in dp
            )
            st.markdown(
                f'<div style="background:{COLORS["surface"]};padding:8px 12px;border-radius:3px;'
                f'margin-bottom:8px;border:1px dashed {COLORS["border"]};">'
                f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["text_dim"]};'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">DATA TO WATCH</div>'
                f'{dp_items}</div>',
                unsafe_allow_html=True,
            )

        # Vehicles / tickers
        vehicles_note = play.get("vehicles") or ""
        tickers = play.get("tickers") or []
        if tickers:
            st.markdown(
                f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["text_dim"]};'
                f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">TICKERS · CLICK TO ACTIVATE</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(tickers), 5))
            for i, tk in enumerate(tickers[:5]):
                with cols[i]:
                    with st.popover(tk, use_container_width=True):
                        _render_ticker_quick_card(tk)
        if vehicles_note:
            st.markdown(
                f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text_dim"]};'
                f'line-height:1.5;font-style:italic;margin-top:6px;">'
                f'📎 {escape(vehicles_note)}</div>',
                unsafe_allow_html=True,
            )

        # Probability breakdown (collapsible)
        comps = play.get("probability_components") or {}
        if comps:
            with st.expander(f"Probability breakdown ({prob_pct}/100)", expanded=False):
                st.markdown(
                    f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text"]};line-height:1.6;">'
                    + "".join(
                        f'<div><span style="color:{COLORS["text_dim"]};">'
                        f'{escape(k.replace("_"," ").title())}:</span> '
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{v}</span></div>'
                        for k, v in comps.items()
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Track record strip
# ─────────────────────────────────────────────────────────────────────────────

def _render_track_record():
    try:
        from services.thesis_tracker import get_track_record_summary
        stats = get_track_record_summary()
    except Exception:
        return
    total = stats.get("total_theses", 0)
    if total == 0:
        st.markdown(
            f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text_dim"]};'
            f'padding:10px 12px;border-top:1px dashed {COLORS["border"]};margin-top:18px;">'
            f'📋 Track Record — No theses saved yet. Save one below to start scoring.</div>',
            unsafe_allow_html=True,
        )
        return

    per_order = stats.get("per_order") or {}
    rows = []
    for order in (2, 3, 4):
        s = per_order.get(order) or {}
        n = s.get("n_total", 0)
        hr = s.get("hit_rate")
        avg = s.get("avg_return_pct")
        if n == 0:
            continue
        hr_txt = f"{hr:.0f}%" if hr is not None else "—"
        avg_txt = f"{avg:+.0f}%" if avg is not None else "—"
        rows.append(
            f'<div style="display:inline-block;margin-right:18px;font-family:{MONO};font-size:11px;">'
            f'<span style="color:{COLORS["text_dim"]};">{order}ND ORDER</span> '
            f'<span style="color:{COLORS["text"]};font-weight:700;">n={n}</span> '
            f'<span style="color:{COLORS["text_dim"]};">hit</span> '
            f'<span style="color:#22c55e;font-weight:700;">{hr_txt}</span> '
            f'<span style="color:{COLORS["text_dim"]};">avg</span> '
            f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{avg_txt}</span>'
            f'</div>'
        )
    if not rows:
        return
    st.markdown(
        f'<div style="margin-top:18px;padding:10px 12px;background:{COLORS["surface"]};'
        f'border:1px solid {COLORS["border"]};border-radius:3px;">'
        f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.08em;font-weight:700;margin-bottom:6px;">📋 PERSONAL TRACK RECORD</div>'
        + "".join(rows) + '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_plays(thesis: dict, regime_ctx: dict) -> dict:
    """Add regime_alignment, crowding, probability_score, kelly_pct, size_bucket
    to each play. Mutates and returns the thesis."""
    plays = thesis.get("orders") or []
    rt = thesis.get("regime_template") or {}
    hit_rate_str = rt.get("analog_hit_rate") or ""

    for p in plays:
        # Regime alignment (needs size_bucket preview — use moonshot as hint)
        ra = _regime_alignment(p, regime_ctx)
        p["regime_alignment"] = ra

        # Crowding (computed)
        tickers = p.get("tickers") or []
        crowd = _aggregate_crowding(tickers)
        p["crowding_label"] = crowd.get("label")
        p["crowding_data"]  = crowd

        # Carry analog hit rate down into each play for probability math
        p["analog_hit_rate_str"] = hit_rate_str

        # First-pass probability (Kelly needs it)
        prob = _probability_score(p)
        p["probability_score"]       = prob["probability"]
        p["probability_components"]  = prob["components"]

        # Kelly size
        up = p.get("upside_pct_range") or [0, 0]
        dn = p.get("downside_pct_range") or [0, 0]
        up_mid = (float(up[0]) + float(up[-1])) / 2 if up else 0
        dn_mid = (float(dn[0]) + float(dn[-1])) / 2 if dn else 0
        k = _kelly_size(prob["probability"], up_mid, dn_mid)
        p["kelly_pct"]   = k
        p["size_bucket"] = _bucket_from_kelly(k, bool(p.get("moonshot")))

    return thesis


def _save_to_journal_and_tracker(thesis: dict, regime_ctx: dict):
    from services.market_data import fetch_ticker_fundamentals
    import yfinance as yf

    # Gather entry prices
    all_tickers = []
    for p in thesis.get("orders") or []:
        all_tickers.extend(p.get("tickers") or [])
    all_tickers = list(dict.fromkeys([t.upper() for t in all_tickers if t]))
    entry_prices: dict[str, float] = {}
    for t in all_tickers:
        try:
            tk_obj = yf.Ticker(t)
            fast = getattr(tk_obj, "fast_info", None)
            px = None
            if fast is not None:
                px = getattr(fast, "last_price", None) or fast.get("lastPrice") if isinstance(fast, dict) else getattr(fast, "last_price", None)
            if not px:
                info = tk_obj.info or {}
                px = info.get("regularMarketPrice") or info.get("currentPrice")
            if px and px > 0:
                entry_prices[t] = float(px)
        except Exception:
            continue

    # Write to thesis_tracker
    try:
        from services.thesis_tracker import save_thesis
        thesis_id = save_thesis(thesis, entry_prices)
    except Exception:
        thesis_id = None

    # Write summary to journal as markdown-ish text
    primary = thesis.get("primary", "")
    rt = thesis.get("regime_template") or {}
    kill = thesis.get("what_would_kill_it") or ""
    lines = [
        f"**Nth-Order Thesis — {primary}**",
        f"Analog: {rt.get('analog_name','')} ({rt.get('analog_period','')})",
        f"Analog return: {rt.get('analog_return_pct','?')}% over {rt.get('analog_duration_months','?')} mo",
        "",
    ]
    for p in thesis.get("orders") or []:
        lines.append(
            f"— {p.get('order')}°: {p.get('name','')} · Prob {p.get('probability_score','')}/100 · "
            f"Kelly {p.get('kelly_pct',0):.1f}% · Tickers: {', '.join(p.get('tickers') or [])}"
        )
    lines.append("")
    lines.append(f"Kill switch: {kill}")
    thesis_md = "\n".join(lines)

    return thesis_id


def render():
    narrative = get_narrative()

    # Header
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{COLORS["header_bg"]},{COLORS["surface"]});'
        f'border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["bloomberg_orange"]};'
        f'padding:14px 18px;border-radius:3px;margin-bottom:14px;">'
        f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.1em;font-weight:700;">NTH-ORDER THESIS · RESEARCH NOTE</div>'
        f'<div style="font-family:{MONO};font-size:22px;color:{COLORS["text"]};'
        f'font-weight:700;margin-top:4px;">2nd · 3rd · 4th Order Plays</div>'
        f'<div style="font-family:{MONO};font-size:11px;color:{COLORS["text_dim"]};'
        f'margin-top:2px;">Mapping the current narrative to plays two or three steps downstream — '
        f'before the herd prices them.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    regime_ctx = _build_regime_ctx()
    _render_context_strip(regime_ctx, narrative)

    # QIR freshness check — upstream regime aggregator
    try:
        from services.qir_history import load_qir_history
        _qir_hist = load_qir_history()
    except Exception:
        _qir_hist = []
    _qir_hours: float | None = None
    if _qir_hist:
        try:
            _qir_ts = _qir_hist[0].get("timestamp", "")
            _qir_dt = datetime.fromisoformat(_qir_ts.replace("Z", "+00:00"))
            _qir_now = datetime.now(_qir_dt.tzinfo) if _qir_dt.tzinfo else datetime.utcnow()
            _qir_hours = (_qir_now - _qir_dt).total_seconds() / 3600.0
        except Exception:
            _qir_hours = None
    if not _qir_hist:
        st.error(
            "🚨 **No QIR run found.** Run the Quick Intel Run module first — it aggregates the "
            "regime signals (HMM, CI%, conviction, tactical score) that anchor this thesis. "
            "Without it, the nth-order mapping is flying blind."
        )
    elif _qir_hours is not None and _qir_hours > 12:
        st.warning(
            f"⏱ QIR last run **{_qir_hours:.1f}h ago**. Regime context may be stale — "
            "consider rerunning QIR before building the thesis for best signal quality."
        )

    # Stale news warning / one-click refresh
    hours = _events_freshness_hours(regime_ctx.get("events_ts"))
    if hours is None or hours > 24:
        with st.container():
            st.warning(
                "News digest is stale or missing. Your thesis is only as fresh as the tape. "
                "Run Current Events module first to refresh the digest for this session."
            )

    # Consume pending narrative load (from Saved searches) BEFORE the widget is
    # instantiated — Streamlit forbids mutating a widget's key after creation.
    _pending = st.session_state.pop("_nth_pending_narrative", None)
    if _pending is not None:
        st.session_state["_nth_narrative_input"] = _pending

    # Input row
    col_a, col_b = st.columns([3, 2])
    with col_a:
        narrative_input = st.text_input(
            "Primary narrative",
            value=narrative or "",
            placeholder="e.g. AI, reshoring, stablecoin regulation, de-dollarization",
            key="_nth_narrative_input",
        )
    with col_b:
        use_claude, model = render_ai_tier_selector(
            key="_nth_ai_tier",
            label="AI engine",
            default=2,
            include_dd_scholar=True,
            recommendation="👑 Highly Regarded (Haiku) default · 📜 DD Scholar (Sonnet 4.6) for deep chains",
        )

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        build_clicked = st.button("🚀 Build Thesis", type="primary", use_container_width=True)
    with col_btn2:
        force_refresh = st.button("Force Refresh", use_container_width=True)

    # Recent searches — one-click load of previously cached theses
    _disk_cache = _load_disk_cache()
    if _disk_cache:
        with st.expander(f"📁 Saved searches ({len(_disk_cache)})", expanded=False):
            _entries = sorted(
                _disk_cache.items(),
                key=lambda kv: kv[1].get("saved_at", ""),
                reverse=True,
            )
            for _key, _entry in _entries[:20]:
                try:
                    _ts = datetime.fromisoformat(_entry.get("saved_at", ""))
                    _ago_h = (datetime.utcnow() - _ts).total_seconds() / 3600.0
                    _ago = f"{_ago_h:.1f}h" if _ago_h < 24 else f"{_ago_h/24:.1f}d"
                except Exception:
                    _ago = "?"
                _eng = _entry.get("engine", "")
                _c1, _c2, _c3 = st.columns([4, 1, 1])
                with _c1:
                    st.markdown(
                        f'<div style="font-family:{MONO};font-size:12px;color:{COLORS["text"]};padding-top:6px;">'
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{escape(_key)}</span>'
                        f'<span style="color:{COLORS["text_dim"]};"> · {_ago} ago{f" · {escape(_eng)}" if _eng else ""}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with _c2:
                    if st.button("Load", key=f"_load_cached_{_key}", use_container_width=True):
                        st.session_state["_nth_pending_narrative"] = _key
                        st.session_state[f"_nth_thesis_{_key}"] = _entry.get("thesis")
                        st.rerun()
                with _c3:
                    if st.button("🗑", key=f"_del_cached_{_key}", use_container_width=True, help="Delete"):
                        _cache_delete(_key)
                        st.session_state.pop(f"_nth_thesis_{_key}", None)
                        st.rerun()

    narrative_key = (narrative_input or "").strip().lower()
    cache_key = f"_nth_thesis_{narrative_key}"
    thesis = st.session_state.get(cache_key)

    # Fall back to disk cache if RAM is empty (survives restarts)
    if thesis is None and narrative_key:
        _disk_entry = _cache_get(narrative_key)
        if _disk_entry and _disk_entry.get("thesis"):
            thesis = _disk_entry["thesis"]
            st.session_state[cache_key] = thesis
            try:
                _saved_dt = datetime.fromisoformat(_disk_entry.get("saved_at", ""))
                _age_h = (datetime.utcnow() - _saved_dt).total_seconds() / 3600.0
                _eng = _disk_entry.get("engine", "")
                if _age_h < 24:
                    _age_str = f"{_age_h:.1f}h ago"
                elif _age_h < 24 * 30:
                    _age_str = f"{_age_h/24:.1f}d ago"
                else:
                    _age_str = _saved_dt.strftime("%Y-%m-%d")
                st.info(
                    f"📁 Loaded cached thesis for **{narrative_key}** (generated {_age_str}"
                    f"{f' · {_eng}' if _eng else ''}). Click **Force Refresh** to regenerate."
                )
            except Exception:
                pass

    if build_clicked and not narrative_input.strip():
        st.error("Enter a primary narrative first (or set one via Narrative Discovery).")
        return

    if build_clicked or (force_refresh and narrative_input.strip()):
        with st.spinner(
            f"Running 5-run ensemble on {narrative_input.strip()} — this takes 60–90 seconds…"
        ):
            try:
                from services.claude_client import generate_nth_order_thesis
                raw = generate_nth_order_thesis(
                    primary_narrative=narrative_input.strip(),
                    regime_ctx=regime_ctx,
                    contagion_score=regime_ctx.get("contagion_score"),
                    qir_snapshot=None,
                    n_runs=5,
                    use_claude=use_claude,
                    model=model,
                )
            except Exception as e:
                st.error(f"Thesis generation failed: {e}")
                return

            if not raw or not raw.get("orders"):
                st.error("No thesis returned. Try a different narrative or engine tier.")
                return

            thesis = _enrich_plays(raw, regime_ctx)
            st.session_state[cache_key] = thesis
            # Persist to disk — survives restart
            _engine_label = st.session_state.get("_nth_ai_tier", "")
            _cache_put(narrative_key, thesis, _engine_label)

    if not thesis:
        st.info(
            "Enter a narrative above and click **Build Thesis**. Output is a structured "
            "research note with 2nd/3rd/4th-order plays, probability-scored and Kelly-sized."
        )
        _render_track_record()
        return

    # Generation metadata
    gen_at = thesis.get("generated_at", "")
    gen_age = ""
    if gen_at:
        try:
            g = datetime.fromisoformat(gen_at)
            age_h = (datetime.utcnow() - g).total_seconds() / 3600
            gen_age = f"{age_h:.1f}h ago" if age_h >= 1 else f"{age_h*60:.0f}m ago"
        except Exception:
            pass
    n_runs = thesis.get("n_runs", "?")
    st.markdown(
        f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["text_dim"]};'
        f'text-align:right;margin-bottom:10px;">Generated {gen_age} · {n_runs}-run ensemble</div>',
        unsafe_allow_html=True,
    )

    # Regime template
    _render_regime_template(thesis.get("regime_template") or {})

    # Primary thesis card
    pt = thesis.get("primary_thesis", "")
    if pt:
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {COLORS["bloomberg_orange"]};padding:14px 16px;'
            f'border-radius:3px;margin-bottom:14px;">'
            f'<div style="font-family:{MONO};font-size:10px;color:{COLORS["bloomberg_orange"]};'
            f'letter-spacing:0.08em;font-weight:700;margin-bottom:4px;">PRIMARY THESIS</div>'
            f'<div style="font-family:{MONO};font-size:14px;color:{COLORS["text"]};'
            f'line-height:1.6;font-weight:500;">{escape(pt)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Order stepper tabs
    plays = thesis.get("orders") or []
    by_order = {2: [], 3: [], 4: []}
    for p in plays:
        o = int(p.get("order", 2))
        if o in by_order:
            by_order[o].append(p)

    counts = {o: len(by_order[o]) for o in (2, 3, 4)}
    tab_labels = [
        f"2nd Order · {counts[2]} plays",
        f"3rd Order · {counts[3]} plays",
        f"4th Order · {counts[4]} plays",
    ]
    tabs = st.tabs(tab_labels)
    for idx, order in enumerate((2, 3, 4)):
        with tabs[idx]:
            order_plays = by_order[order]
            if not order_plays:
                st.info(f"No {order}-order plays proposed this run.")
                continue
            for i, play in enumerate(order_plays):
                _render_play_card(play, key_prefix=f"o{order}_i{i}")

    # Kill switch
    kill = thesis.get("what_would_kill_it", "")
    if kill:
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.12);border:1px solid #ef4444;'
            f'padding:12px 16px;border-radius:3px;margin-top:14px;">'
            f'<div style="font-family:{MONO};font-size:10px;color:#ef4444;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">🔴 WHAT WOULD KILL IT</div>'
            f'<div style="font-family:{MONO};font-size:13px;color:{COLORS["text"]};'
            f'line-height:1.55;font-weight:500;">{escape(kill)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Save to journal + tracker
    st.markdown("---")
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        if st.button("💾 Save to Journal & Tracker", type="primary", use_container_width=True):
            tid = _save_to_journal_and_tracker(thesis, regime_ctx)
            st.success(
                f"Saved thesis {tid[:8] if tid else ''} to trade journal and forward tracker."
                if tid else
                "Saved to journal."
            )

    # Track record
    _render_track_record()
