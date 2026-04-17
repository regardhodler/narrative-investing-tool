"""Forecast Accuracy Tracker — did the AI's calls actually work?

Logs AI predictions from any module, auto-evaluates outcomes against price moves,
and shows accuracy stats by signal type, model, and confidence tier.
"""

import io
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric


# ── Inline log button (importable by any module) ───────────────────────────────

def render_log_button(
    signal_type: str,
    prediction: str,
    confidence: int,
    summary: str,
    model: str = "Groq Llama 3.3",
    ticker: str | None = None,
    target_price: float | None = None,
    horizon_days: int = 30,
    key: str = "log_btn",
    label: str = "📌 Log Forecast",
):
    """Render a one-click forecast logging button. Import this into any module."""
    from services.forecast_tracker import log_forecast
    if st.button(label, key=key, use_container_width=True, help="Log this signal to Forecast Accuracy Tracker"):
        fid = log_forecast(
            signal_type=signal_type,
            prediction=prediction,
            confidence=confidence,
            summary=summary,
            model=model,
            ticker=ticker,
            target_price=target_price,
            horizon_days=horizon_days,
        )
        st.toast(f"📌 Logged! [{fid}] — check Forecast Tracker to track outcome", icon="✅")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _outcome_badge(outcome: str | None) -> str:
    if outcome == "correct":
        return f'<span style="background:{COLORS["positive"]}22;color:{COLORS["positive"]};border:1px solid {COLORS["positive"]}44;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">✓ CORRECT</span>'
    if outcome == "incorrect":
        return f'<span style="background:{COLORS["negative"]}22;color:{COLORS["negative"]};border:1px solid {COLORS["negative"]}44;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">✗ WRONG</span>'
    if outcome == "expired":
        return f'<span style="background:{COLORS["text_dim"]}22;color:{COLORS["text_dim"]};border:1px solid {COLORS["text_dim"]}44;padding:2px 8px;border-radius:3px;font-size:11px;">EXPIRED</span>'
    return f'<span style="background:{COLORS["blue"]}22;color:{COLORS["blue"]};border:1px solid {COLORS["blue"]}44;padding:2px 8px;border-radius:3px;font-size:11px;">PENDING</span>'


def _type_color(signal_type: str) -> str:
    return {
        "valuation": COLORS["bloomberg_orange"],
        "squeeze":   COLORS["yellow"],
        "regime":    COLORS["blue"],
        "fed":       COLORS["accent"],
        "manual":    COLORS["text_dim"],
    }.get(signal_type, COLORS["text_dim"])


def _read_current_signals() -> dict:
    """Pull relevant fields from session state to pre-fill the log form."""
    out: dict = {}

    # Valuation result (last ticker run)
    val = st.session_state.get("_last_valuation_result")
    if isinstance(val, dict) and "rating" in val:
        out["valuation"] = val

    # Regime context
    regime = st.session_state.get("_regime_context")
    if isinstance(regime, dict) and "regime" in regime:
        out["regime"] = regime

    # Dominant rate path
    rp = st.session_state.get("_dominant_rate_path")
    if isinstance(rp, dict):
        out["fed"] = rp

    # Tactical context
    tac = st.session_state.get("_tactical_context")
    if isinstance(tac, dict):
        out["tactical"] = tac

    # Short squeeze thesis (from Short Squeeze Radar)
    sq = st.session_state.get("sq_thesis")
    if isinstance(sq, dict) and sq.get("ticker") and sq.get("text"):
        out["squeeze"] = sq

    return out


# ── Tab: Log Forecast ──────────────────────────────────────────────────────────

def _render_log_tab():
    from services.forecast_tracker import log_forecast, MODEL_LABELS

    st.markdown(
        f'<div style="color:{COLORS["text_dim"]};font-size:12px;margin-bottom:12px;">'
        "Log an AI prediction so it can be validated against future price action."
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick-capture from session state
    signals = _read_current_signals()
    if signals:
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;'
            f'font-weight:700;letter-spacing:0.08em;margin-bottom:6px;">⚡ QUICK CAPTURE FROM SESSION</div>',
            unsafe_allow_html=True,
        )
        caps = []
        if "valuation" in signals:
            v = signals["valuation"]
            ticker = st.session_state.get("active_ticker", "")
            caps.append((
                f"💹 Valuation: {ticker} → {v.get('rating','?')} ({v.get('confidence','?')}% confidence)",
                "valuation", ticker,
                v.get("rating", ""), v.get("confidence", 70),
                v.get("summary", ""), v.get("engine", ""),
                v.get("key_levels", {}).get("resistance"),
            ))
        if "regime" in signals:
            r = signals["regime"]
            caps.append((
                f"📡 Regime: {r.get('quadrant','?')} — score {r.get('score',0):+.2f}",
                "regime", None,
                r.get("quadrant", ""), 60,
                r.get("signal_summary", "")[:200], "",
                None,
            ))
        if "fed" in signals:
            f_ = signals["fed"]
            label_map = {"cut_25": "25bp cut", "cut_50": "50bp cut", "hold": "Hold", "hike_25": "25bp hike"}
            sc = label_map.get(f_.get("scenario", ""), f_.get("scenario", ""))
            caps.append((
                f"🏛 Fed: dominant path → {sc} ({f_.get('prob_pct', 0):.0f}% prob)",
                "fed", None,
                sc, int(f_.get("prob_pct", 50)),
                f"Fed dominant rate path: {sc}", "",
                None,
            ))
        if "squeeze" in signals:
            sq = signals["squeeze"]
            sq_ticker = sq.get("ticker", "?")
            sq_preview = sq.get("text", "")[:120].replace("\n", " ")
            caps.append((
                f"🔥 Squeeze: {sq_ticker} — {sq_preview}…",
                "squeeze", sq_ticker,
                f"Squeeze candidate: {sq_ticker}", 65,
                sq.get("text", "")[:500], "",
                None,
            ))

        for label, sig_type, ticker, prediction, conf, summary, engine, target in caps:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {_type_color(sig_type)};'
                    f'padding:8px 12px;border-radius:4px;font-size:12px;margin-bottom:4px;">'
                    f'{label}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Log", key=f"qc_{sig_type}_{prediction}", use_container_width=True):
                    fid = log_forecast(
                        signal_type=sig_type,
                        prediction=prediction,
                        confidence=conf,
                        summary=summary,
                        model=engine or "Groq Llama 3.3",
                        ticker=ticker,
                        target_price=target,
                    )
                    st.success(f"Logged! ID: {fid}")
                    st.rerun()

        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

    # Manual entry form
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;'
        f'font-weight:700;letter-spacing:0.08em;margin-bottom:8px;">✏️ MANUAL ENTRY</div>',
        unsafe_allow_html=True,
    )

    with st.form("log_forecast_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            sig_type = st.selectbox(
                "Signal Type",
                ["valuation", "squeeze", "regime", "fed", "manual"],
            )
            ticker_in = st.text_input("Ticker (optional)", placeholder="AAPL")
        with c2:
            prediction = st.text_input("Prediction", placeholder="Buy / Goldilocks / Hold…")
            confidence = st.slider("Confidence", 0, 100, 65)

        # Smart default horizon — only used for regime/fed/manual (not ticker-based ATR signals)
        _is_atr_signal = sig_type in ("valuation", "squeeze")
        _default_horizons = {"regime": 14, "fed": 30, "manual": 30}
        _horizon_options = [7, 14, 21, 30, 60, 90]
        _default_idx = _horizon_options.index(_default_horizons.get(sig_type, 30)) if _default_horizons.get(sig_type, 30) in _horizon_options else 3

        c3, c4 = st.columns(2)
        with c3:
            if _is_atr_signal:
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {COLORS["bloomberg_orange"]};padding:8px 10px;border-radius:4px;font-size:11px;">'
                    f'<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;margin-bottom:2px;">📡 ATR EXIT — No fixed date</div>'
                    f'<div style="color:{COLORS["text_dim"]};">Closes on 🎯 target (3×ATR) or 🛑 trailing stop (2×ATR)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                horizon = None  # no horizon for ATR-based signals
            else:
                horizon = st.selectbox("Evaluation Horizon", _horizon_options, index=_default_idx, format_func=lambda x: f"{x} days")
        with c4:
            model_sel = st.selectbox("Model Used", list(MODEL_LABELS.values()) + ["Other"])

        summary_in = st.text_area("Thesis / Summary", height=80, placeholder="Why is this the right call?")
        notes_in = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("📌 Log Forecast", use_container_width=True)
        if submitted:
            if not prediction.strip():
                st.error("Prediction cannot be empty.")
            else:
                kwargs = dict(
                    signal_type=sig_type,
                    prediction=prediction.strip(),
                    confidence=confidence,
                    summary=summary_in,
                    model=model_sel,
                    ticker=ticker_in.strip().upper() or None,
                    notes=notes_in,
                )
                if horizon is not None:
                    kwargs["horizon_days"] = horizon
                fid = log_forecast(**kwargs)
                st.success(f"Forecast logged! ID: **{fid}**")
                st.rerun()


# ── Tab: Dashboard ─────────────────────────────────────────────────────────────

# ── Pattern Accuracy block ─────────────────────────────────────────────────────

def _render_pattern_accuracy_block():
    """Per-QIR-pattern win rate table, read from trade_journal.json pattern_at_entry field."""
    trades = _load_trade_journal()
    closed = [
        t for t in trades
        if t.get("status") == "closed"
        and t.get("entry_price")
        and t.get("exit_price")
        and t.get("pattern_at_entry")
    ]
    if not closed:
        return  # silently skip — no data yet

    # Aggregate by pattern
    from collections import defaultdict
    by_pattern: dict[str, dict] = defaultdict(lambda: {"wins": [], "losses": [], "returns": []})
    for t in closed:
        pat  = (t.get("pattern_at_entry") or "UNKNOWN").upper().strip()
        ep   = float(t["entry_price"])
        xp   = float(t["exit_price"])
        dirn = (t.get("direction") or "Long").lower()
        ret  = (xp - ep) / ep * 100 if dirn == "long" else (ep - xp) / ep * 100
        by_pattern[pat]["returns"].append(ret)
        if ret > 0:
            by_pattern[pat]["wins"].append(ret)
        else:
            by_pattern[pat]["losses"].append(ret)

    if not by_pattern:
        return

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:20px 0 12px 0;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">🎯 QIR PATTERN ACCURACY</div>'
        f'<div style="color:{COLORS["text_dim"]};font-size:11px;margin-bottom:10px;">'
        f'When QIR fires a pattern, how often do SPY trades from that pattern win?</div>',
        unsafe_allow_html=True,
    )

    # Sort by sample size descending, then win rate
    rows = []
    for pat, d in by_pattern.items():
        n      = len(d["returns"])
        nw     = len(d["wins"])
        wr     = round(nw / n * 100) if n else 0
        avg_r  = round(sum(d["returns"]) / n, 2) if n else 0
        avg_w  = round(sum(d["wins"]) / len(d["wins"]), 2) if d["wins"] else None
        avg_l  = round(sum(d["losses"]) / len(d["losses"]), 2) if d["losses"] else None
        real_b = round(avg_w / abs(avg_l), 2) if avg_w and avg_l else None
        rows.append((pat, n, nw, wr, avg_r, avg_w, avg_l, real_b))

    rows.sort(key=lambda x: (-x[1], -x[3]))  # n desc, then wr desc

    # Header
    st.markdown(
        f'<div style="display:grid;grid-template-columns:160px 40px 60px 80px 80px 70px 70px 60px;'
        f'gap:6px;padding:4px 8px;font-size:9px;color:{COLORS["text_dim"]};font-weight:700;'
        f'letter-spacing:0.08em;border-bottom:1px solid {COLORS["border"]};margin-bottom:4px;">'
        f'<div>PATTERN</div><div>n</div><div>WIN RATE</div><div>AVG RETURN</div>'
        f'<div>AVG WIN</div><div>AVG LOSS</div><div>b RATIO</div><div>VERDICT</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for pat, n, nw, wr, avg_r, avg_w, avg_l, real_b in rows:
        wr_col   = COLORS["positive"] if wr >= 55 else (COLORS["yellow"] if wr >= 45 else COLORS["negative"])
        ar_col   = COLORS["positive"] if avg_r > 0 else COLORS["negative"]
        b_str    = f"{real_b}" if real_b else ("ATR 1.5" if n < 10 else "—")
        verdict  = (
            "✅ Edge" if wr >= 55 and avg_r > 0 else
            "⚠️ Mixed" if wr >= 45 or avg_r > 0 else
            "❌ No edge" if n >= 3 else
            "⏳ Building"
        )
        verd_col = (
            COLORS["positive"] if "Edge" in verdict else
            COLORS["yellow"]   if "Mixed" in verdict or "Building" in verdict else
            COLORS["negative"]
        )
        # Win rate bar
        bar_w = min(wr, 100)
        bar_html = (
            f'<div style="background:{COLORS["surface"]};border-radius:2px;height:10px;overflow:hidden;">'
            f'<div style="width:{bar_w}%;background:{wr_col};height:100%;border-radius:2px;"></div>'
            f'</div>'
        )

        st.markdown(
            f'<div style="display:grid;grid-template-columns:160px 40px 60px 80px 80px 70px 70px 60px;'
            f'gap:6px;padding:5px 8px;font-size:11px;background:{COLORS["surface"]};'
            f'border:1px solid {COLORS["border"]};border-radius:3px;margin-bottom:3px;align-items:center;">'
            f'<div style="color:{COLORS["text"]};font-size:10px;font-weight:700;">{pat}</div>'
            f'<div style="color:{COLORS["text_dim"]};">{n}</div>'
            f'<div>'
            f'<div style="color:{wr_col};font-size:10px;font-weight:700;margin-bottom:2px;">{wr}%</div>'
            f'{bar_html}'
            f'</div>'
            f'<div style="color:{ar_col};font-weight:700;">{avg_r:+.2f}%</div>'
            f'<div style="color:{COLORS["positive"]};">{f"+{avg_w:.2f}%" if avg_w is not None else "—"}</div>'
            f'<div style="color:{COLORS["negative"]};">{f"{avg_l:.2f}%" if avg_l is not None else "—"}</div>'
            f'<div style="color:{COLORS["bloomberg_orange"]};">{b_str}</div>'
            f'<div style="color:{verd_col};font-size:10px;font-weight:700;">{verdict}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(rows) > 0:
        total_n  = sum(r[1] for r in rows)
        total_nw = sum(r[2] for r in rows)
        overall_wr = round(total_nw / total_n * 100) if total_n else 0
        ov_col = COLORS["positive"] if overall_wr >= 55 else (COLORS["yellow"] if overall_wr >= 45 else COLORS["negative"])
        st.markdown(
            f'<div style="font-size:10px;color:{COLORS["text_dim"]};margin-top:6px;">'
            f'Overall across all patterns: '
            f'<span style="color:{ov_col};font-weight:700;">{overall_wr}% win rate</span>'
            f' ({total_nw}W / {total_n - total_nw}L of {total_n} closed SPY trades)'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tab: Accuracy Dashboard ────────────────────────────────────────────────────

def _render_dashboard_tab():
    from services.forecast_tracker import get_stats, evaluate_pending

    # Auto-evaluate on load — capture newly resolved for notification
    log_before = [e for e in (st.session_state.get("_forecast_log") or []) if e.get("outcome") not in (None, "pending")]
    ids_before  = {e.get("id") for e in log_before}
    updated = evaluate_pending()

    # ATR fire notification banner
    if updated:
        log_after   = [e for e in (st.session_state.get("_forecast_log") or []) if e.get("outcome") not in (None, "pending")]
        newly_fired = [e for e in log_after if e.get("id") not in ids_before]
        if newly_fired:
            with st.container():
                _banner_lines = []
                for e in newly_fired:
                    _exit = {"profit_target": "🎯 Target hit", "trailing_stop": "🛑 Stop hit"}.get(e.get("exit_reason",""), "⏱ Resolved")
                    _ret  = f"{e['return_pct']:+.1f}%" if e.get("return_pct") is not None else ""
                    _out  = "✅ CORRECT" if e.get("outcome") == "correct" else "❌ WRONG"
                    _banner_lines.append(f"**{e.get('ticker','MACRO')}** {e.get('prediction','')} — {_exit} {_ret} {_out}")
                st.success("**🔔 ATR exit triggered on " + str(len(newly_fired)) + " trade(s):**\n\n" + "\n\n".join(_banner_lines))
        else:
            st.toast(f"Auto-evaluated {updated} forecast(s)", icon="🔍")

    stats = get_stats()

    if stats["total"] == 0:
        st.info("No forecasts logged yet. Use the **Log Forecast** tab to start tracking.")
        return

    # ── Row 1: Overall counts ─────────────────────────────────────────────────
    cols = st.columns(3)
    for col, (label, val, color) in zip(cols, [
        ("TOTAL LOGGED", str(stats["total"]),          None),
        ("RESOLVED",     str(stats["total_resolved"]), None),
        ("PENDING",      str(stats["pending"]),        COLORS["blue"]),
    ]):
        with col:
            st.markdown(bloomberg_metric(label, val, color), unsafe_allow_html=True)

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:10px 0 8px 0;"></div>', unsafe_allow_html=True)

    # ── Row 2: Signal accuracy (macro — no price) ─────────────────────────────
    st.markdown(
        f'<div style="color:{COLORS["blue"]};font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">📡 SIGNAL ACCURACY — REGIME / FED / MACRO</div>',
        unsafe_allow_html=True,
    )
    macro_acc   = stats.get("macro_accuracy")
    macro_res   = stats.get("macro_resolved", 0)
    macro_color = COLORS["positive"] if macro_acc is not None and macro_acc >= 55 else (COLORS["yellow"] if macro_acc is not None and macro_acc >= 45 else COLORS["negative"]) if macro_acc is not None else COLORS["text_dim"]
    cols_macro = st.columns(3)
    for col, (label, val, color) in zip(cols_macro, [
        ("MACRO ACCURACY",  f"{macro_acc:.1f}%" if macro_acc is not None else "—",  macro_color),
        ("MACRO RESOLVED",  str(macro_res),                                           None),
        ("MACRO CORRECT",   str(stats.get("correct", 0) - sum(1 for e in stats.get("log",[]) if e.get("outcome") == "correct" and e.get("signal_type") in ("valuation","squeeze"))), COLORS["positive"]),
    ]):
        with col:
            st.markdown(bloomberg_metric(label, val, color), unsafe_allow_html=True)

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:10px 0 8px 0;"></div>', unsafe_allow_html=True)

    # ── Row 3: Price accuracy (ATR-based — valuation / squeeze) ───────────────
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">📈 PRICE ACCURACY — VALUATION / SQUEEZE (ATR EXIT)</div>',
        unsafe_allow_html=True,
    )
    price_acc   = stats.get("price_accuracy")
    price_res   = stats.get("price_resolved", 0)
    price_color = COLORS["positive"] if price_acc is not None and price_acc >= 55 else (COLORS["yellow"] if price_acc is not None and price_acc >= 45 else COLORS["negative"]) if price_acc is not None else COLORS["text_dim"]
    avg_alpha      = stats.get("avg_alpha")
    pos_alpha_rate = stats.get("positive_alpha_rate")
    alpha_color    = COLORS["positive"] if avg_alpha is not None and avg_alpha > 0 else (COLORS["negative"] if avg_alpha is not None and avg_alpha < 0 else COLORS["text_dim"])
    alpha_calls    = sum(1 for e in stats.get("log", []) if e.get("alpha_pct") is not None)

    cols_price = st.columns(6)
    for col, (label, val, color) in zip(cols_price, [
        ("PRICE ACCURACY",   f"{price_acc:.1f}%" if price_acc is not None else "—",                                                       price_color),
        ("PRICE RESOLVED",   str(price_res),                                                                                               None),
        ("AVG RETURN (✓)",   f"{stats['avg_return_correct']:+.1f}%"   if stats.get("avg_return_correct")   is not None else "—",          COLORS["positive"]),
        ("AVG RETURN (✗)",   f"{stats['avg_return_incorrect']:+.1f}%" if stats.get("avg_return_incorrect") is not None else "—",          COLORS["negative"]),
        ("AVG ALPHA vs SPY", f"{avg_alpha:+.2f}%" if avg_alpha is not None else "—",                                                      alpha_color),
        ("BEAT SPY RATE",    f"{pos_alpha_rate:.0f}%" if pos_alpha_rate is not None else "—",                                             alpha_color),
    ]):
        with col:
            st.markdown(bloomberg_metric(label, val, color), unsafe_allow_html=True)

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:10px 0 8px 0;"></div>', unsafe_allow_html=True)

    # ── Row 4: Streaks ────────────────────────────────────────────────────────
    streak_type  = stats.get("current_streak_type")
    streak_n     = stats.get("current_streak", 0)
    streak_color = COLORS["positive"] if streak_type == "correct" else (COLORS["negative"] if streak_type == "incorrect" else COLORS["text_dim"])
    streak_label = f"{'🔥' if streak_type == 'correct' else '❄️'} {streak_n} {streak_type or '—'} in a row" if streak_n else "—"

    cols2 = st.columns(3)
    for col, (label, val, color) in zip(cols2, [
        ("CURRENT STREAK",    streak_label,                               streak_color),
        ("BEST WIN STREAK",   str(stats.get("best_correct_streak", 0)),   COLORS["positive"]),
        ("WORST LOSS STREAK", str(stats.get("worst_incorrect_streak", 0)), COLORS["negative"]),
    ]):
        with col:
            st.markdown(bloomberg_metric(label, val, color), unsafe_allow_html=True)

    st.markdown(f'<div style="height:16px;"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # By type
    with col_left:
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">ACCURACY BY SIGNAL TYPE</div>',
            unsafe_allow_html=True,
        )
        if stats["by_type"]:
            for st_, d in sorted(stats["by_type"].items(), key=lambda x: -x[1]["accuracy"]):
                acc = d["accuracy"]
                bar_color = COLORS["positive"] if acc >= 55 else (COLORS["yellow"] if acc >= 45 else COLORS["negative"])
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<div style="width:80px;font-size:11px;color:{_type_color(st_)};text-transform:uppercase;">{st_}</div>'
                    f'<div style="flex:1;background:{COLORS["surface"]};border-radius:2px;height:14px;overflow:hidden;">'
                    f'<div style="width:{acc}%;background:{bar_color};height:100%;border-radius:2px;transition:width 0.3s;"></div>'
                    f'</div>'
                    f'<div style="width:60px;font-size:11px;color:{bar_color};text-align:right;">{acc}% ({d["total"]})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f'<div style="color:{COLORS["text_dim"]};font-size:12px;">No resolved forecasts yet.</div>', unsafe_allow_html=True)

    # By model
    with col_right:
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">ACCURACY BY AI MODEL</div>',
            unsafe_allow_html=True,
        )
        if stats["by_model"]:
            for m, d in sorted(stats["by_model"].items(), key=lambda x: -x[1]["accuracy"]):
                acc = d["accuracy"]
                bar_color = COLORS["positive"] if acc >= 55 else (COLORS["yellow"] if acc >= 45 else COLORS["negative"])
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<div style="width:100px;font-size:11px;color:{COLORS["text"]};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{m}</div>'
                    f'<div style="flex:1;background:{COLORS["surface"]};border-radius:2px;height:14px;overflow:hidden;">'
                    f'<div style="width:{acc}%;background:{bar_color};height:100%;border-radius:2px;"></div>'
                    f'</div>'
                    f'<div style="width:60px;font-size:11px;color:{bar_color};text-align:right;">{acc}% ({d["total"]})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f'<div style="color:{COLORS["text_dim"]};font-size:12px;">No resolved forecasts yet.</div>', unsafe_allow_html=True)

    # Win rate by regime context (price calls only)
    by_regime = stats.get("by_regime", {})
    if by_regime:
        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 12px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:4px;">WIN RATE BY REGIME CONTEXT</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:11px;margin-bottom:10px;">'
            f'Price calls (valuation/squeeze) grouped by market quadrant + VIX at time of logging</div>',
            unsafe_allow_html=True,
        )
        for key, d in sorted(by_regime.items(), key=lambda x: -x[1]["accuracy"]):
            acc = d["accuracy"]
            bar_color = COLORS["positive"] if acc >= 55 else (COLORS["yellow"] if acc >= 45 else COLORS["negative"])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<div style="width:220px;font-size:11px;color:{COLORS["text"]};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{key}</div>'
                f'<div style="flex:1;background:{COLORS["surface"]};border-radius:2px;height:14px;overflow:hidden;">'
                f'<div style="width:{acc}%;background:{bar_color};height:100%;border-radius:2px;"></div>'
                f'</div>'
                f'<div style="width:60px;font-size:11px;color:{bar_color};text-align:right;">{acc}% ({d["total"]})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Confidence calibration chart
    if stats["calibration"]:
        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 12px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">CONFIDENCE CALIBRATION</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:11px;margin-bottom:10px;">'
            f'Are high-confidence calls actually more accurate? (diagonal = perfectly calibrated)</div>',
            unsafe_allow_html=True,
        )
        cal = stats["calibration"]
        labels = [c["label"] for c in cal]
        actual_acc = [c["accuracy"] for c in cal]
        ns = [c["n"] for c in cal]

        fig = go.Figure()
        # Perfect calibration diagonal
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            line=dict(color=COLORS["text_dim"], dash="dash", width=1),
            name="Perfect calibration",
        ))
        fig.add_trace(go.Bar(
            x=labels, y=actual_acc,
            marker_color=[
                COLORS["positive"] if a >= 55 else (COLORS["yellow"] if a >= 45 else COLORS["negative"])
                for a in actual_acc
            ],
            text=[f"n={n}" for n in ns],
            textposition="outside",
            textfont=dict(size=10),
            name="Actual accuracy",
        ))
        fig.update_layout(
            **{k: v for k, v in {
                "paper_bgcolor": COLORS["bg"],
                "plot_bgcolor": COLORS["bg"],
                "font": dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"], size=11),
                "xaxis": dict(title="Confidence range stated", gridcolor=COLORS["grid"]),
                "yaxis": dict(title="Actual accuracy %", range=[0, 110], gridcolor=COLORS["grid"]),
                "margin": dict(l=40, r=20, t=20, b=40),
                "height": 260,
                "showlegend": False,
            }.items()}
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pattern Accuracy block ────────────────────────────────────────────────
    _render_pattern_accuracy_block()

    # Methodology explainer
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:20px 0 12px 0;"></div>', unsafe_allow_html=True)
    with st.expander("📖 How this tracker works & why the data is reliable", expanded=False):
        st.markdown(
            f"""<div style="color:{COLORS["text_dim"]};font-size:12px;line-height:1.7;">

<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;letter-spacing:0.08em;margin-bottom:6px;">
WHY CALENDAR DATES ARE UNRELIABLE
</div>
Traditional backtests close a trade on day 30, 60, or 90 regardless of what the market is doing.
That introduces arbitrary noise — a great call can be early, a bad call can look right by accident.
This tracker eliminates that bias entirely for price-based signals.

<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;letter-spacing:0.08em;margin-top:12px;margin-bottom:6px;">
HOW ATR EXIT WORKS
</div>
Every ticker-based forecast (Valuation, Squeeze) is evaluated using a <b style="color:{COLORS["text"]};">real price simulation</b>
that walks actual daily OHLC data from your log date forward:

<ul style="margin:6px 0 6px 18px;padding:0;">
<li><b style="color:{COLORS["positive"]};">🎯 Profit Target</b> — entry price ± <b>3×ATR(14)</b>. Triggered when intraday high/low touches the level. ATR(14) is the 14-day Average True Range at the time of logging — a volatility-normalized target that automatically adjusts to each ticker's behavior.</li>
<li><b style="color:{COLORS["negative"]};">🛑 Trailing Stop</b> — begins at ± <b>2×ATR(14)</b> from entry, then trails the high watermark (longs) or low watermark (shorts). It only moves in your favor — never against you.</li>
<li><b style="color:{COLORS["text"]};">Still open</b> — if neither has triggered, the trade stays pending. No forced close.</li>
</ul>

This gives a <b style="color:{COLORS["text"]};">1.5:1 reward-to-risk ratio</b> on every trade entry,
and lets winners run while cutting losers at a volatility-defined level.

<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;letter-spacing:0.08em;margin-top:12px;margin-bottom:6px;">
DATA SOURCE & ACCURACY
</div>
Prices come from <b style="color:{COLORS["text"]};">Yahoo Finance (yfinance)</b> — adjusted OHLC data, the same source used
by most quant platforms. ATR is computed from True Range: <code style="color:{COLORS["text"]};background:{COLORS["surface"]};padding:1px 4px;border-radius:2px;">max(H-L, |H-Prev_Close|, |L-Prev_Close|)</code> rolled over 14 days.
The trailing stop and target levels are locked at log time and stored — they don't change retroactively.

<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;letter-spacing:0.08em;margin-top:12px;margin-bottom:6px;">
SPY ALPHA
</div>
Every resolved call shows alpha vs SPY — the return above (or below) what a passive SPY hold would have returned
over the same period. This is the only honest measure of whether AI signal generation actually adds value.
A 60% accuracy rate that lags SPY by 3% annually is not useful. One that beats SPY is.

<div style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;letter-spacing:0.08em;margin-top:12px;margin-bottom:6px;">
REGIME & FED CALLS
</div>
Macro calls (Regime, Fed path) don't have a price to trail, so they use a horizon-based window.
Outcome is determined by comparing the predicted regime/rate-path to the current live state
when the window closes.

</div>""",
            unsafe_allow_html=True,
        )


# ── Tab: History ───────────────────────────────────────────────────────────────

def _render_history_tab():
    from services.forecast_tracker import get_stats, evaluate_pending, mark_outcome, delete_forecast

    # Auto-evaluate on load
    updated = evaluate_pending()
    if updated:
        st.toast(f"Auto-evaluated {updated} forecast(s)", icon="🔍")

    log = st.session_state.get("_forecast_log") or []

    if not log:
        st.info("No forecasts logged yet.")
        return

    # Controls row
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        filter_type = st.selectbox("Filter by type", ["All"] + ["valuation", "squeeze", "regime", "fed", "manual"], key="fa_filter_type")
    with c2:
        filter_outcome = st.selectbox("Filter by outcome", ["All", "pending", "correct", "incorrect", "expired"], key="fa_filter_outcome")
    with c3:
        if st.button("🔄 Evaluate All", use_container_width=True, help="Fetch current prices and resolve past-horizon forecasts"):
            n = evaluate_pending(force=False)
            st.toast(f"Evaluated {n} forecast(s)")
            st.rerun()
    with c4:
        # CSV export
        try:
            import pandas as pd
            df_rows = []
            for e in log:
                ts = e.get("timestamp")
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")
                ctx = e.get("market_context") or {}
                df_rows.append({
                    "id": e.get("id"), "type": e.get("signal_type"), "ticker": e.get("ticker"),
                    "prediction": e.get("prediction"), "confidence": e.get("confidence"),
                    "model": e.get("model"), "horizon_days": e.get("horizon_days"),
                    "timestamp": ts_str, "outcome": e.get("outcome"),
                    "return_pct": e.get("return_pct"), "spy_return_pct": e.get("spy_return_pct"),
                    "alpha_pct": e.get("alpha_pct"),
                    "price_at_forecast": e.get("price_at_forecast"),
                    "price_at_eval": e.get("price_at_eval"),
                    "spy_price_at_forecast": e.get("spy_price_at_forecast"),
                    "spy_price_at_eval": e.get("spy_price_at_eval"),
                    "regime_at_log": ctx.get("regime"), "quadrant_at_log": ctx.get("quadrant"),
                    "fg_at_log": ctx.get("fear_greed_score"), "vix_structure_at_log": ctx.get("vix_structure"),
                    "notes": e.get("notes"),
                })
            csv_buf = io.StringIO()
            pd.DataFrame(df_rows).to_csv(csv_buf, index=False)
            st.download_button(
                "⬇ CSV", data=csv_buf.getvalue(),
                file_name=f"forecast_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
        except Exception:
            pass

    # Apply filters
    filtered = log
    if filter_type != "All":
        filtered = [e for e in filtered if e.get("signal_type") == filter_type]
    if filter_outcome != "All":
        if filter_outcome == "pending":
            filtered = [e for e in filtered if e.get("outcome") in (None, "pending")]
        else:
            filtered = [e for e in filtered if e.get("outcome") == filter_outcome]

    st.markdown(
        f'<div style="color:{COLORS["text_dim"]};font-size:11px;margin-bottom:10px;">'
        f'Showing {len(filtered)} of {len(log)} forecasts</div>',
        unsafe_allow_html=True,
    )

    for entry in filtered:
        ts = entry.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "—"

        ticker = entry.get("ticker", "")
        sig_type = entry.get("signal_type", "manual")
        prediction = entry.get("prediction", "")
        confidence = entry.get("confidence", 0)
        outcome = entry.get("outcome")
        ret = entry.get("return_pct")
        ret_str = f"{ret:+.1f}%" if ret is not None else "—"
        model = entry.get("model", "—")
        horizon = entry.get("horizon_days", 30)
        fid = entry.get("id", "")

        eval_due_str = "—"
        days_elapsed = 0
        days_open_str = "—"
        is_atr_signal = sig_type in ("valuation", "squeeze") and ticker
        if ts:
            now_dt = datetime.now()
            days_elapsed = max(0, (now_dt - ts).days)
            days_open_str = f"{days_elapsed}d"
            if not is_atr_signal:
                horizon = entry.get("horizon_days", 30)
                eval_due = ts + timedelta(days=horizon)
                eval_due_str = eval_due.strftime("%Y-%m-%d")

        price_at = entry.get("price_at_forecast")
        price_eval = entry.get("price_at_eval")
        price_str = f"${price_at:.2f}" if price_at else "—"
        price_eval_str = f"${price_eval:.2f}" if price_eval else "—"

        tc = _type_color(sig_type)
        _open_flag = f" · {days_open_str} open" if outcome in (None, "pending") else ""
        with st.expander(
            f"[{fid}] {sig_type.upper()} · {ticker or 'MACRO'} · {prediction} · {ts_str}{_open_flag}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                # ATR live status for pending ticker signals
                if outcome in (None, "pending") and is_atr_signal:
                    stop_l  = entry.get("stop_at_log")
                    tgt_l   = entry.get("target_at_log")
                    p_at    = entry.get("price_at_forecast")
                    if stop_l and tgt_l and p_at:
                        is_short = prediction in ("Sell", "Strong Sell")
                        risk     = abs(p_at - stop_l)
                        reward   = abs(tgt_l - p_at)
                        rr_str   = f"{reward/risk:.1f}:1" if risk > 0 else "—"
                        st.markdown(
                            f'<div style="display:flex;gap:10px;margin-bottom:8px;font-size:11px;">'
                            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["negative"]}33;'
                            f'border-left:3px solid {COLORS["negative"]};padding:6px 10px;border-radius:4px;flex:1;">'
                            f'<div style="color:{COLORS["negative"]};font-size:10px;text-transform:uppercase;letter-spacing:0.08em;">🛑 Stop</div>'
                            f'<div style="color:{COLORS["text"]};font-weight:700;">${stop_l:.2f}</div></div>'
                            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["positive"]}33;'
                            f'border-left:3px solid {COLORS["positive"]};padding:6px 10px;border-radius:4px;flex:1;">'
                            f'<div style="color:{COLORS["positive"]};font-size:10px;text-transform:uppercase;letter-spacing:0.08em;">🎯 Target</div>'
                            f'<div style="color:{COLORS["text"]};font-weight:700;">${tgt_l:.2f}</div></div>'
                            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                            f'padding:6px 10px;border-radius:4px;flex:1;">'
                            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:10px;text-transform:uppercase;letter-spacing:0.08em;">R:R</div>'
                            f'<div style="color:{COLORS["text"]};font-weight:700;">{rr_str}</div></div>'
                            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                            f'padding:6px 10px;border-radius:4px;flex:1;">'
                            f'<div style="color:{COLORS["text_dim"]};font-size:10px;text-transform:uppercase;letter-spacing:0.08em;">Open</div>'
                            f'<div style="color:{COLORS["text"]};font-weight:700;">{days_open_str}</div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                elif outcome in (None, "pending") and not is_atr_signal:
                    # Horizon progress bar for regime/fed/manual
                    horizon = entry.get("horizon_days", 30)
                    if horizon and horizon > 0:
                        pct = min(100, int(days_elapsed / horizon * 100))
                        days_remaining = max(0, horizon - days_elapsed)
                        bar_color = COLORS["yellow"] if pct < 75 else COLORS["bloomberg_orange"]
                        st.markdown(
                            f'<div style="margin-bottom:8px;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:10px;color:{COLORS["text_dim"]};margin-bottom:3px;">'
                            f'<span>Horizon progress: day {days_elapsed}/{horizon} ({pct}%)</span>'
                            f'<span style="color:{COLORS["bloomberg_orange"]}">{days_remaining}d remaining</span>'
                            f'</div>'
                            f'<div style="background:{COLORS["surface"]};border-radius:3px;height:6px;overflow:hidden;">'
                            f'<div style="width:{pct}%;background:{bar_color};height:100%;border-radius:3px;transition:width 0.3s;"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
            with col1:
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;">'
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {tc};padding:8px 10px;border-radius:4px;">'
                    f'<div style="font-size:10px;color:{COLORS["bloomberg_orange"]};text-transform:uppercase;letter-spacing:0.08em;">Type</div>'
                    f'<div style="font-size:14px;color:{tc};font-weight:700;margin-top:2px;">{sig_type}</div></div>'
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};padding:8px 10px;border-radius:4px;">'
                    f'<div style="font-size:10px;color:{COLORS["bloomberg_orange"]};text-transform:uppercase;letter-spacing:0.08em;">Confidence</div>'
                    f'<div style="font-size:14px;color:{COLORS["text"]};font-weight:700;margin-top:2px;">{confidence}%</div></div>'
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};padding:8px 10px;border-radius:4px;">'
                    f'<div style="font-size:10px;color:{COLORS["bloomberg_orange"]};text-transform:uppercase;letter-spacing:0.08em;">Return</div>'
                    f'<div style="font-size:14px;color:{COLORS["positive"] if ret and ret > 0 else COLORS["negative"] if ret and ret < 0 else COLORS["text"]};font-weight:700;margin-top:2px;">{ret_str}</div></div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:6px;margin-bottom:10px;font-size:11px;">'
                    f'<div><span style="color:{COLORS["text_dim"]}">Model:</span> <span style="color:{COLORS["text"]}">{model}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Price@Log:</span> <span style="color:{COLORS["text"]}">{price_str}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Price@Eval:</span> <span style="color:{COLORS["text"]}">{price_eval_str}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">{"Days Open" if is_atr_signal else "Eval Due"}:</span> <span style="color:{COLORS["text"]}">{days_open_str if is_atr_signal else eval_due_str}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Alpha vs SPY:</span> <span style="color:{COLORS["positive"] if entry.get("alpha_pct") and entry["alpha_pct"] > 0 else COLORS["negative"] if entry.get("alpha_pct") and entry["alpha_pct"] < 0 else COLORS["text_dim"]}">{f"{entry["alpha_pct"]:+.2f}%" if entry.get("alpha_pct") is not None else "—"}</span></div>'
                    f'</div>'
                    + (
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-bottom:10px;font-size:11px;">'
                        f'<div><span style="color:{COLORS["text_dim"]}">ATR@Log:</span> <span style="color:{COLORS["text"]}">{f"${entry["atr_at_log"]:.2f}" if entry.get("atr_at_log") else "—"}</span></div>'
                        f'<div><span style="color:{COLORS["negative"]}">Stop:</span> <span style="color:{COLORS["text"]}">{f"${entry["stop_at_log"]:.2f}" if entry.get("stop_at_log") else "—"}</span></div>'
                        f'<div><span style="color:{COLORS["positive"]}">Target:</span> <span style="color:{COLORS["text"]}">{f"${entry["target_at_log"]:.2f}" if entry.get("target_at_log") else "—"}</span></div>'
                        f'<div><span style="color:{COLORS["text_dim"]}">Exit:</span> <span style="color:{COLORS["bloomberg_orange"]}">'
                        + {"profit_target": "🎯 Target", "trailing_stop": "🛑 Trailing Stop", "horizon_end": "⏱ Horizon", "horizon_end_fallback": "⏱ Horizon"}.get(entry.get("exit_reason", ""), entry.get("exit_reason") or "—")
                        + f'</span> <span style="color:{COLORS["text_dim"]};font-size:10px;">{entry.get("exit_date") or ""}</span></div>'
                        f'</div>'
                        if entry.get("atr_at_log") or entry.get("exit_reason") else ""
                    ),
                    unsafe_allow_html=True,
                )

                summary = entry.get("summary", "")
                if summary:
                    st.markdown(
                        f'<div style="color:{COLORS["text_dim"]};font-size:11px;'
                        f'background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                        f'padding:8px 10px;border-radius:4px;margin-bottom:8px;">{summary[:300]}</div>',
                        unsafe_allow_html=True,
                    )

                # Market context at log time
                ctx = entry.get("market_context")
                if ctx:
                    ctx_parts = []
                    if ctx.get("quadrant"): ctx_parts.append(f"Regime: {ctx['quadrant']}")
                    if ctx.get("fear_greed_score") is not None: ctx_parts.append(f"F&G: {ctx['fear_greed_score']} ({ctx.get('fear_greed_label','')})")
                    if ctx.get("vix_structure"): ctx_parts.append(f"VIX: {ctx['vix_structure']}")
                    if ctx.get("fed_path"): ctx_parts.append(f"Fed: {ctx['fed_path']} ({ctx.get('fed_path_prob',0):.0f}%)")
                    if ctx_parts:
                        st.markdown(
                            f'<div style="color:{COLORS["text_dim"]};font-size:10px;'
                            f'background:{COLORS["surface"]};border:1px solid {COLORS["border"]}44;'
                            f'padding:6px 10px;border-radius:4px;margin-bottom:8px;">'
                            f'📸 Context at log: {" · ".join(ctx_parts)}</div>',
                            unsafe_allow_html=True,
                        )

                # Post-mortem (stored)
                postmortem = entry.get("postmortem")
                if postmortem:
                    st.markdown(
                        f'<div style="color:{COLORS["yellow"]};font-size:10px;font-weight:700;'
                        f'letter-spacing:0.06em;margin-bottom:4px;">🧠 AI POST-MORTEM</div>'
                        f'<div style="color:{COLORS["text_dim"]};font-size:11px;'
                        f'background:{COLORS["surface"]};border:1px solid {COLORS["yellow"]}33;'
                        f'padding:8px 10px;border-radius:4px;margin-bottom:8px;">{postmortem}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(_outcome_badge(outcome), unsafe_allow_html=True)

            with col2:
                st.markdown(f'<div style="font-size:10px;color:{COLORS["text_dim"]};margin-bottom:6px;">MANUAL MARK</div>', unsafe_allow_html=True)
                if outcome in (None, "pending"):
                    if st.button("✓ Correct", key=f"mc_{fid}", use_container_width=True):
                        mark_outcome(fid, "correct")
                        st.rerun()
                    if st.button("✗ Wrong", key=f"mi_{fid}", use_container_width=True):
                        mark_outcome(fid, "incorrect")
                        st.rerun()

                # AI post-mortem for incorrect calls
                if outcome == "incorrect" and not entry.get("postmortem"):
                    if st.button("🧠 Analyze", key=f"pm_{fid}", use_container_width=True, help="AI post-mortem: why was this call wrong?"):
                        with st.spinner("Analyzing..."):
                            _run_postmortem(entry)
                        st.rerun()

                if st.button("🗑 Delete", key=f"del_{fid}", use_container_width=True):
                    delete_forecast(fid)
                    st.rerun()


# ── AI Post-mortem ─────────────────────────────────────────────────────────────

def _run_postmortem(entry: dict) -> None:
    """Call Groq to generate a post-mortem on a wrong forecast. Writes to entry in-place."""
    try:
        import os, requests
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.warning("GROQ_API_KEY not set — post-mortem unavailable.")
            return
        ctx = entry.get("market_context") or {}
        ctx_str = ", ".join(f"{k}: {v}" for k, v in ctx.items() if v is not None) or "unavailable"
        ret = entry.get("return_pct")
        ret_str = f"{ret:+.1f}%" if ret is not None else "unknown"
        prompt = (
            f"You are an investment analyst reviewing a failed AI prediction.\n\n"
            f"ORIGINAL CALL: {entry.get('signal_type','').upper()} → {entry.get('prediction','?')}\n"
            f"TICKER: {entry.get('ticker') or 'MACRO'}\n"
            f"CONFIDENCE: {entry.get('confidence',0)}%\n"
            f"THESIS: {(entry.get('summary') or '')[:400]}\n"
            f"MARKET CONTEXT AT LOG TIME: {ctx_str}\n"
            f"ACTUAL RETURN: {ret_str} (call was WRONG)\n\n"
            f"In 3-4 sentences, identify the most likely reason this call failed. "
            f"Be specific: was it the wrong regime, mean-reversion, macro shift, overconfidence? "
            f"Do not hedge. Give a direct post-mortem."
        )
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 220},
            timeout=20,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        entry["postmortem"] = result
        log = st.session_state.get("_forecast_log") or []
        for e in log:
            if e.get("id") == entry.get("id"):
                e["postmortem"] = result
                break
        st.session_state["_forecast_log"] = log
    except Exception as ex:
        st.warning(f"Post-mortem failed: {ex}")


# ── Tab: P&L Simulation ────────────────────────────────────────────────────────

def _render_pnl_tab():
    from services.forecast_tracker import get_stats

    stats = get_stats()
    log = stats.get("log", [])
    resolved = [e for e in log if e.get("outcome") in ("correct", "incorrect") and e.get("return_pct") is not None]

    if len(resolved) < 2:
        st.info("Need at least 2 resolved forecasts with price data to simulate P&L.")
        return

    st.markdown(
        f'<div style="color:{COLORS["text_dim"]};font-size:12px;margin-bottom:12px;">'
        f'Simulates equal-weight $1,000 per resolved valuation/squeeze call. '
        f'Assumes you entered at price@log and exited at price@eval.</div>',
        unsafe_allow_html=True,
    )

    # Filter to tradeable types
    tradeable = [e for e in resolved if e.get("signal_type") in ("valuation", "squeeze")]
    if not tradeable:
        st.info("No resolved valuation or squeeze forecasts with price data yet.")
        return

    # Sort chronologically
    tradeable.sort(key=lambda e: e.get("timestamp") or "")

    position_size = 1000.0
    cumulative = 0.0
    spy_cumulative = 0.0
    curve_x, curve_y, spy_curve_y, alpha_curve_y = [], [], [], []
    total_invested = 0.0
    wins, losses = 0, 0

    for e in tradeable:
        ts = e.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts or "")[:10]
        ret_pct = e.get("return_pct", 0) or 0
        spy_ret_pct = e.get("spy_return_pct") or 0
        prediction = e.get("prediction", "")
        if prediction in ("Sell", "Strong Sell"):
            ret_pct = -ret_pct
            spy_ret_pct = -spy_ret_pct  # benchmark also inverted (you were short)
        pnl = position_size * (ret_pct / 100)
        spy_pnl = position_size * (spy_ret_pct / 100)
        cumulative += pnl
        spy_cumulative += spy_pnl
        total_invested += position_size
        if e.get("outcome") == "correct":
            wins += 1
        else:
            losses += 1
        label = f"{ts_str} [{e.get('ticker','?')}]"
        curve_x.append(label)
        curve_y.append(round(cumulative, 2))
        spy_curve_y.append(round(spy_cumulative, 2))
        alpha_curve_y.append(round(cumulative - spy_cumulative, 2))

    total_return_pct = round(cumulative / total_invested * 100, 2) if total_invested else 0
    spy_return_total = round(spy_cumulative / total_invested * 100, 2) if total_invested else 0
    alpha_total = round(total_return_pct - spy_return_total, 2)

    # Summary metrics
    pnl_color = COLORS["positive"] if cumulative >= 0 else COLORS["negative"]
    alpha_color = COLORS["positive"] if alpha_total > 0 else COLORS["negative"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(bloomberg_metric("TOTAL P&L", f"${cumulative:+,.0f}", pnl_color), unsafe_allow_html=True)
    with col2:
        st.markdown(bloomberg_metric("YOUR RETURN", f"{total_return_pct:+.1f}%", pnl_color), unsafe_allow_html=True)
    with col3:
        st.markdown(bloomberg_metric("SPY RETURN", f"{spy_return_total:+.1f}%", COLORS["text_dim"]), unsafe_allow_html=True)
    with col4:
        st.markdown(bloomberg_metric("TOTAL ALPHA", f"{alpha_total:+.2f}%", alpha_color), unsafe_allow_html=True)
    with col5:
        st.markdown(bloomberg_metric("WINNING TRADES", str(wins), COLORS["positive"]), unsafe_allow_html=True)
    with col6:
        st.markdown(bloomberg_metric("LOSING TRADES", str(losses), COLORS["negative"]), unsafe_allow_html=True)

    st.markdown(f'<div style="height:12px;"></div>', unsafe_allow_html=True)

    # Equity curve with SPY overlay
    bar_colors = []
    prev = 0.0
    for y in curve_y:
        bar_colors.append(COLORS["positive"] if y >= prev else COLORS["negative"])
        prev = y

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=curve_x, y=curve_y,
        marker_color=bar_colors,
        name="Your calls P&L",
        opacity=0.8,
    ))
    if any(v != 0 for v in spy_curve_y):
        fig.add_trace(go.Scatter(
            x=curve_x, y=spy_curve_y,
            mode="lines+markers",
            line=dict(color=COLORS["text_dim"], width=2, dash="dot"),
            marker=dict(size=5),
            name="SPY equivalent",
        ))
        fig.add_trace(go.Scatter(
            x=curve_x, y=alpha_curve_y,
            mode="lines",
            line=dict(color=COLORS["bloomberg_orange"], width=2),
            name="Alpha (your - SPY)",
        ))
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"], size=10),
        xaxis=dict(title="Trade (date · ticker)", gridcolor=COLORS["grid"], tickangle=-45),
        yaxis=dict(title="Cumulative P&L ($)", gridcolor=COLORS["grid"]),
        legend=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["border"], borderwidth=1, font=dict(size=10)),
        margin=dict(l=50, r=20, t=20, b=100),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div style="color:{COLORS["text_dim"]};font-size:10px;">'
        f'* $1,000 position size per trade. Sell/Strong Sell calls short the position. '
        f'Does not account for slippage, fees, or compounding.</div>',
        unsafe_allow_html=True,
    )





# ── Tab: SPY Trade Log ─────────────────────────────────────────────────────────

def _load_trade_journal() -> list:
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_journal.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save_trade_journal(trades: list) -> None:
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_journal.json")
    with open(path, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def _auto_evaluate_spy_trades() -> int:
    """Walk OHLC history for all open SPY trades; auto-close if ATR×2 stop or ATR×3 target hit.

    Returns count of trades auto-closed this run.
    """
    from services.forecast_tracker import _fetch_atr_and_history, _ATR_MULT_STOP, _ATR_MULT_TARGET

    trades  = _load_trade_journal()
    open_spy = [
        t for t in trades
        if t.get("ticker", "").upper() == "SPY"
        and t.get("status") == "open"
        and t.get("entry_price")
        and t.get("entry_date")
    ]
    if not open_spy:
        return 0

    n_closed = 0
    for trade in open_spy:
        try:
            ep     = float(trade["entry_price"])
            is_short = (trade.get("direction") or "Long").lower() == "short"
            ts     = datetime.fromisoformat(str(trade["entry_date"]) + "T00:00:00")

            data = _fetch_atr_and_history("SPY", ts)
            if not data or not data["closes"]:
                continue

            atr = data["atr"]
            if not atr or atr <= 0:
                continue

            stop_dist   = _ATR_MULT_STOP   * atr
            target_dist = _ATR_MULT_TARGET * atr

            if is_short:
                stop_level   = ep + stop_dist
                target_level = ep - target_dist
            else:
                stop_level   = ep - stop_dist
                target_level = ep + target_dist

            # Store ATR levels if not already set
            if not trade.get("atr_at_log"):
                trade["atr_at_log"]    = round(atr, 4)
                trade["stop_at_log"]   = round(stop_level, 2)
                trade["target_at_log"] = round(target_level, 2)

            watermark     = ep
            trailing_stop = stop_level
            result        = None

            highs, lows, closes, dates = data["highs"], data["lows"], data["closes"], data["dates"]

            for h, l, c, d in zip(highs, lows, closes, dates):
                if is_short:
                    if l < watermark:
                        watermark     = l
                        trailing_stop = watermark + stop_dist
                    if l <= target_level:
                        ret    = round((ep - target_level) / ep * 100, 2)
                        result = {"exit_price": target_level, "exit_reason": "Hit Target (ATR×3)",
                                  "exit_date": d, "return_pct": ret}
                        break
                    if h >= trailing_stop:
                        ret    = round((ep - trailing_stop) / ep * 100, 2)
                        result = {"exit_price": round(trailing_stop, 2),
                                  "exit_reason": "Hit Stop (ATR×2)",
                                  "exit_date": d, "return_pct": ret}
                        break
                else:
                    if h > watermark:
                        watermark     = h
                        trailing_stop = watermark - stop_dist
                    if h >= target_level:
                        ret    = round((target_level - ep) / ep * 100, 2)
                        result = {"exit_price": target_level, "exit_reason": "Hit Target (ATR×3)",
                                  "exit_date": d, "return_pct": ret}
                        break
                    if l <= trailing_stop:
                        ret    = round((trailing_stop - ep) / ep * 100, 2)
                        result = {"exit_price": round(trailing_stop, 2),
                                  "exit_reason": "Hit Stop (ATR×2)",
                                  "exit_date": d, "return_pct": ret}
                        break

            if result:
                trade["status"]     = "closed"
                trade["exit_price"] = round(result["exit_price"], 2)
                trade["exit_reason"]= result["exit_reason"]
                trade["closed_at"]  = str(result["exit_date"])[:10]
                trade["return_pct"] = result["return_pct"]
                n_closed += 1

        except Exception:
            continue

    if n_closed:
        _save_trade_journal(trades)

    return n_closed


def _render_spy_trade_tab():
    import uuid

    # Auto-evaluate open SPY trades against ATR stop/target
    _n_auto_closed = _auto_evaluate_spy_trades()
    if _n_auto_closed:
        st.success(f"🔔 **{_n_auto_closed} SPY trade(s) auto-closed** — ATR stop/target triggered. See table below.")

    trades = _load_trade_journal()
    spy_trades = [t for t in trades if t.get("ticker", "").upper() == "SPY"]

    # ── Stats bar ──────────────────────────────────────────────────────────────
    closed = [t for t in spy_trades if t.get("status") == "closed" and t.get("entry_price") and t.get("exit_price")]
    wins   = []
    losses = []
    for t in closed:
        ep, xp = float(t["entry_price"]), float(t["exit_price"])
        direction = (t.get("direction") or "Long").lower()
        ret = (xp - ep) / ep * 100 if direction == "long" else (ep - xp) / ep * 100
        if ret > 0:
            wins.append(ret)
        else:
            losses.append(ret)

    n_closed   = len(closed)
    win_rate   = round(len(wins) / n_closed * 100) if n_closed else 0
    avg_win    = round(sum(wins) / len(wins), 2) if wins else None
    avg_loss   = round(sum(losses) / len(losses), 2) if losses else None
    real_b     = round(avg_win / abs(avg_loss), 2) if avg_win and avg_loss else None
    wr_color   = COLORS["positive"] if win_rate >= 55 else (COLORS["yellow"] if win_rate >= 45 else COLORS["negative"])

    cols = st.columns(6)
    metrics = [
        ("SPY TRADES",    str(len(spy_trades)),                              None),
        ("CLOSED",        str(n_closed),                                     None),
        ("WIN RATE",      f"{win_rate}%" if n_closed else "—",               wr_color),
        ("AVG WIN",       f"+{avg_win:.2f}%" if avg_win is not None else "—", COLORS["positive"]),
        ("AVG LOSS",      f"{avg_loss:.2f}%" if avg_loss is not None else "—", COLORS["negative"]),
        ("REAL b RATIO",  f"{real_b}" if real_b else "ATR 1.5 (boot)",      COLORS["bloomberg_orange"]),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        with col:
            st.markdown(bloomberg_metric(label, val, color), unsafe_allow_html=True)

    # ATR bootstrap progress bar
    needed = max(0, 5 - min(len(wins), len(losses)))
    if n_closed < 10:
        pct_done = min(100, int(n_closed / 10 * 100))
        bar_col  = COLORS["blue"] if pct_done < 50 else COLORS["bloomberg_orange"]
        st.markdown(
            f'<div style="margin:8px 0 4px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;color:{COLORS["text_dim"]};margin-bottom:3px;">'
            f'<span>Kelly bootstrap progress — {n_closed}/10 closed trades</span>'
            f'<span style="color:{COLORS["bloomberg_orange"]};">'
            + (f"Need {needed} more on each side for real b ratio" if needed > 0 else "Real b ratio active ✓")
            + f'</span></div>'
            f'<div style="background:{COLORS["surface"]};border-radius:3px;height:5px;overflow:hidden;">'
            f'<div style="width:{pct_done}%;background:{bar_col};height:100%;border-radius:3px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:10px 0 12px;"></div>', unsafe_allow_html=True)

    # ── Log new trade form ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">📝 LOG SPY TRADE</div>',
        unsafe_allow_html=True,
    )

    # Fetch live SPY price
    _live_spy = None
    try:
        import yfinance as _yf2
        _sh = _yf2.Ticker("SPY").history(period="1d", interval="1m")
        if _sh is not None and not _sh.empty:
            _live_spy = round(float(_sh["Close"].iloc[-1]), 2)
    except Exception:
        pass

    if _live_spy:
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["positive"]}33;'
            f'border-left:3px solid {COLORS["positive"]};padding:7px 12px;border-radius:4px;'
            f'margin-bottom:10px;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:10px;color:{COLORS["text_dim"]};font-weight:700;letter-spacing:0.08em;">SPY LIVE</span>'
            f'<span style="font-size:20px;font-weight:900;color:{COLORS["positive"]};">${_live_spy:.2f}</span>'
            f'<span style="font-size:10px;color:{COLORS["text_dim"]};">Use this as your entry price below</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Auto-pull from QIR session state
    _qir_pattern = st.session_state.get("_qir_pattern") or st.session_state.get("_last_pattern") or "—"
    _kelly_pct   = None
    try:
        from services.portfolio_sizing import compute_qir_kelly, get_trade_kelly_stats
        _kly = compute_qir_kelly(
            st.session_state.get("_conviction_score") or 50,
            st.session_state.get("_fear_composite") or {},
            st.session_state.get("_regime_context") or {},
        )
        _kelly_pct = _kly.get("kelly_half_pct")
    except Exception:
        pass

    if _qir_pattern != "—" or _kelly_pct is not None:
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {COLORS["bloomberg_orange"]};padding:7px 10px;border-radius:4px;'
            f'font-size:11px;margin-bottom:10px;display:flex;gap:16px;">'
            + (f'<span><span style="color:{COLORS["text_dim"]};">Pattern: </span>'
               f'<span style="color:{COLORS["text"]};font-weight:700;">{_qir_pattern}</span></span>' if _qir_pattern != "—" else "")
            + (f'<span><span style="color:{COLORS["text_dim"]};">Kelly suggested: </span>'
               f'<span style="color:{COLORS["positive"] if _kelly_pct and _kelly_pct >= 4 else COLORS["yellow"]};font-weight:700;">'
               f'{_kelly_pct:.1f}%</span></span>' if _kelly_pct is not None else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

    with st.form("spy_trade_log_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            direction  = st.selectbox("Direction", ["Long", "Short"])
            entry_date = st.date_input("Entry Date", value=datetime.now().date())
        with c2:
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, step=0.01, format="%.2f",
                                          value=float(_live_spy or 0.01))
            actual_size = st.number_input("Actual Size (% of portfolio)", min_value=0.0, max_value=100.0, step=0.5, value=float(_kelly_pct or 0))
        with c3:
            exit_price = st.number_input("Exit Price ($ — leave 0 if still open)", min_value=0.0, step=0.01, format="%.2f")
            exit_reason = st.selectbox("Exit Reason", ["Open", "Hit Target (ATR×3)", "Hit Stop (ATR×2)", "Manual Exit", "Time Exit"])

        pattern_used = st.text_input("QIR Pattern at Entry", value=_qir_pattern if _qir_pattern != "—" else "", placeholder="e.g. BULL_CONFIRMED")
        notes = st.text_area("Comments", placeholder="e.g. tariff news, earnings week, macro catalyst, why you took this trade…", height=80)

        submitted = st.form_submit_button("📌 Log SPY Trade", use_container_width=True)
        if submitted:
            if entry_price <= 0:
                st.error("Entry price required.")
            else:
                trade_id  = str(uuid.uuid4())[:8].upper()
                is_closed = exit_price > 0 and exit_reason != "Open"

                # Fetch ATR at entry date to store stop/target levels
                _atr_at_log = _stop_at_log = _target_at_log = None
                try:
                    from services.forecast_tracker import _fetch_atr_and_history, _ATR_MULT_STOP, _ATR_MULT_TARGET
                    _ts_entry = datetime.fromisoformat(str(entry_date) + "T00:00:00")
                    _atr_data = _fetch_atr_and_history("SPY", _ts_entry)
                    if _atr_data and _atr_data.get("atr", 0) > 0:
                        _atr = _atr_data["atr"]
                        _is_short = direction.lower() == "short"
                        _atr_at_log    = round(_atr, 4)
                        _stop_at_log   = round(entry_price + _ATR_MULT_STOP * _atr if _is_short else entry_price - _ATR_MULT_STOP * _atr, 2)
                        _target_at_log = round(entry_price - _ATR_MULT_TARGET * _atr if _is_short else entry_price + _ATR_MULT_TARGET * _atr, 2)
                except Exception:
                    pass

                new_trade = {
                    "id":              trade_id,
                    "ticker":          "SPY",
                    "direction":       direction,
                    "entry_price":     round(entry_price, 2),
                    "entry_date":      str(entry_date),
                    "exit_price":      round(exit_price, 2) if is_closed else None,
                    "exit_reason":     exit_reason if is_closed else None,
                    "status":          "closed" if is_closed else "open",
                    "position_size":   round(actual_size, 2),
                    "kelly_suggested": round(_kelly_pct, 1) if _kelly_pct is not None else None,
                    "pattern_at_entry": pattern_used.strip() or None,
                    "notes":           notes.strip() or None,
                    "logged_at":       datetime.now().isoformat(),
                    "atr_at_log":      _atr_at_log,
                    "stop_at_log":     _stop_at_log,
                    "target_at_log":   _target_at_log,
                }
                trades.append(new_trade)
                _save_trade_journal(trades)
                _atr_msg = f" · ATR={_atr_at_log:.2f}, Stop=${_stop_at_log:.2f}, Target=${_target_at_log:.2f}" if _atr_at_log else ""
                st.success(f"✅ Logged SPY {'closed' if is_closed else 'open'} trade [{trade_id}]{_atr_msg}")
                st.rerun()

    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:14px 0 10px;"></div>', unsafe_allow_html=True)

    # ── Trade table ────────────────────────────────────────────────────────────
    if not spy_trades:
        st.markdown(f'<div style="color:{COLORS["text_dim"]};font-size:12px;">No SPY trades logged yet.</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:8px;">📋 SPY TRADE HISTORY</div>',
        unsafe_allow_html=True,
    )

    # Header row
    st.markdown(
        f'<div style="display:grid;grid-template-columns:60px 55px 70px 70px 70px 55px 80px 80px 1fr;'
        f'gap:6px;padding:4px 8px;font-size:9px;color:{COLORS["text_dim"]};font-weight:700;'
        f'letter-spacing:0.08em;border-bottom:1px solid {COLORS["border"]};margin-bottom:4px;">'
        f'<div>DATE</div><div>DIR</div><div>ENTRY</div><div>EXIT</div>'
        f'<div>RETURN %</div><div>SIZE %</div><div>KELLY %</div><div>PATTERN</div><div>EXIT REASON</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sort newest first
    for t in sorted(spy_trades, key=lambda x: x.get("logged_at", ""), reverse=True):
        ep   = t.get("entry_price")
        xp   = t.get("exit_price")
        dirn = (t.get("direction") or "Long").lower()
        status = t.get("status", "open")

        ret_str   = "—"
        ret_color = COLORS["text_dim"]
        if ep and xp and status == "closed":
            ret = (xp - ep) / ep * 100 if dirn == "long" else (ep - xp) / ep * 100
            ret_str   = f"{ret:+.2f}%"
            ret_color = COLORS["positive"] if ret > 0 else COLORS["negative"]

        status_badge = (
            f'<span style="color:{COLORS["positive"]};font-size:9px;font-weight:700;">CLOSED</span>'
            if status == "closed" else
            f'<span style="color:{COLORS["blue"]};font-size:9px;font-weight:700;">OPEN</span>'
        )

        dir_color = COLORS["positive"] if dirn == "long" else COLORS["negative"]
        pattern   = (t.get("pattern_at_entry") or "—")[:14]
        kelly_sug = f'{t["kelly_suggested"]:.1f}%' if t.get("kelly_suggested") is not None else "—"
        exit_rsn  = (t.get("exit_reason") or "open").replace("Hit ", "").replace(" (ATR×3)", " 🎯").replace(" (ATR×2)", " 🛑")
        entry_date = str(t.get("entry_date") or t.get("logged_at") or "")[:10]
        notes_txt  = t.get("notes") or ""

        st.markdown(
            f'<div style="display:grid;grid-template-columns:60px 55px 70px 70px 70px 55px 80px 80px 1fr;'
            f'gap:6px;padding:5px 8px;font-size:11px;background:{COLORS["surface"]};'
            f'border:1px solid {COLORS["border"]};border-radius:3px;'
            f'{"margin-bottom:1px;" if notes_txt else "margin-bottom:3px;"}'
            f'align-items:center;">'
            f'<div style="color:{COLORS["text_dim"]};">{entry_date}</div>'
            f'<div style="color:{dir_color};font-weight:700;">{dirn.upper()}</div>'
            f'<div style="color:{COLORS["text"]};">${ep:.2f}</div>'
            f'<div style="color:{COLORS["text"]};">{f"${xp:.2f}" if xp else "—"}</div>'
            f'<div style="color:{ret_color};font-weight:700;">{ret_str}</div>'
            f'<div style="color:{COLORS["text"]};">{t.get("position_size") or "—"}%</div>'
            f'<div style="color:{COLORS["bloomberg_orange"]};">{kelly_sug}</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:10px;">{pattern}</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:10px;">{exit_rsn}</div>'
            f'</div>'
            + (
                f'<div style="padding:4px 8px 6px 8px;font-size:10px;color:{COLORS["text_dim"]};'
                f'background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-top:none;border-radius:0 0 3px 3px;margin-bottom:3px;">'
                f'💬 {notes_txt}</div>'
                if notes_txt else ""
            ),
            unsafe_allow_html=True,
        )

    # Close open trades section
    open_trades = [t for t in spy_trades if t.get("status") == "open"]
    if open_trades:
        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:14px 0 10px;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">🔓 CLOSE OPEN TRADES</div>',
            unsafe_allow_html=True,
        )
        for t in open_trades:
            tid   = t.get("id", "?")
            ep    = t.get("entry_price", 0)
            dirn  = t.get("direction", "Long")
            date  = str(t.get("entry_date") or "")[:10]
            stop  = t.get("stop_at_log")
            tgt   = t.get("target_at_log")
            atr   = t.get("atr_at_log")
            with st.expander(f"SPY {dirn} @ ${ep:.2f} — entry {date} [{tid}]", expanded=False):
                # Show auto-close levels
                if stop and tgt:
                    st.markdown(
                        f'<div style="display:flex;gap:8px;margin-bottom:10px;font-size:11px;">'
                        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["negative"]}44;'
                        f'border-left:3px solid {COLORS["negative"]};padding:6px 10px;border-radius:4px;flex:1;">'
                        f'<div style="color:{COLORS["negative"]};font-size:9px;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">🛑 AUTO-STOP (ATR×2)</div>'
                        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:14px;">${stop:.2f}</div>'
                        f'</div>'
                        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["positive"]}44;'
                        f'border-left:3px solid {COLORS["positive"]};padding:6px 10px;border-radius:4px;flex:1;">'
                        f'<div style="color:{COLORS["positive"]};font-size:9px;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">🎯 AUTO-TARGET (ATR×3)</div>'
                        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:14px;">${tgt:.2f}</div>'
                        f'</div>'
                        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                        f'padding:6px 10px;border-radius:4px;flex:1;">'
                        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:9px;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">ATR AT ENTRY</div>'
                        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:14px;">${atr:.2f}</div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="color:{COLORS["text_dim"]};font-size:10px;margin-bottom:8px;">'
                        f'🤖 This trade auto-closes when SPY hits either level. Tab refresh triggers evaluation.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="color:{COLORS["text_dim"]};font-size:10px;margin-bottom:8px;">'
                        f'⚠️ ATR levels not stored — will be computed on next refresh.</div>',
                        unsafe_allow_html=True,
                    )
                c1, c2, c3 = st.columns(3)
                with c1:
                    close_price = st.number_input(f"Manual Exit Price", min_value=0.01, step=0.01, format="%.2f", key=f"close_price_{tid}")
                with c2:
                    close_reason = st.selectbox("Exit Reason", ["Hit Target (ATR×3)", "Hit Stop (ATR×2)", "Manual Exit", "Time Exit"], key=f"close_reason_{tid}")
                with c3:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button(f"Close Manually", key=f"close_btn_{tid}", use_container_width=True):
                        for trade in trades:
                            if trade.get("id") == tid:
                                trade["exit_price"]  = round(close_price, 2)
                                trade["exit_reason"] = close_reason
                                trade["status"]      = "closed"
                                trade["closed_at"]   = datetime.now().isoformat()
                                break
                        _save_trade_journal(trades)
                        st.success(f"Trade [{tid}] manually closed at ${close_price:.2f}")
                        st.rerun()




def render():
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;">'
        f'<span style="font-size:22px;font-weight:700;color:{COLORS["text"]};">Forecast Accuracy Tracker</span>'
        f'<span style="font-size:13px;color:{COLORS["text_dim"]};margin-left:12px;">'
        f'did the AI\'s calls actually work?</span></div>'
        f'<div style="height:2px;margin:8px 0 16px 0;'
        f'background:linear-gradient(90deg,{COLORS["bloomberg_orange"]},{COLORS["bloomberg_orange"]}44,transparent);'
        f'border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )

    tab_dash, tab_spy, tab_pnl, tab_log, tab_hist = st.tabs([
        "📊 Accuracy Dashboard", "🎯 SPY Trade Log", "💰 P&L Simulation", "🔮 Log Forecast", "📋 History"
    ])

    with tab_dash:
        _render_dashboard_tab()

    with tab_spy:
        _render_spy_trade_tab()

    with tab_pnl:
        _render_pnl_tab()

    with tab_log:
        _render_log_tab()

    with tab_hist:
        _render_history_tab()
