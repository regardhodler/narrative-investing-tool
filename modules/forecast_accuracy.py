"""Forecast Accuracy Tracker — did the AI's calls actually work?

Logs AI predictions from any module, auto-evaluates outcomes against price moves,
and shows accuracy stats by signal type, model, and confidence tier.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric


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

        c3, c4 = st.columns(2)
        with c3:
            horizon = st.selectbox("Evaluation Horizon", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"{x} days")
        with c4:
            model_sel = st.selectbox("Model Used", list(MODEL_LABELS.values()) + ["Other"])

        summary_in = st.text_area("Thesis / Summary", height=80, placeholder="Why is this the right call?")
        notes_in = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("📌 Log Forecast", use_container_width=True)
        if submitted:
            if not prediction.strip():
                st.error("Prediction cannot be empty.")
            else:
                fid = log_forecast(
                    signal_type=sig_type,
                    prediction=prediction.strip(),
                    confidence=confidence,
                    summary=summary_in,
                    model=model_sel,
                    ticker=ticker_in.strip().upper() or None,
                    horizon_days=horizon,
                    notes=notes_in,
                )
                st.success(f"Forecast logged! ID: **{fid}**")
                st.rerun()


# ── Tab: Dashboard ─────────────────────────────────────────────────────────────

def _render_dashboard_tab():
    from services.forecast_tracker import get_stats, evaluate_pending

    # Auto-evaluate on load
    updated = evaluate_pending()
    if updated:
        st.toast(f"Auto-evaluated {updated} forecast(s)", icon="🔍")

    stats = get_stats()

    if stats["total"] == 0:
        st.info("No forecasts logged yet. Use the **Log Forecast** tab to start tracking.")
        return

    # Top metrics
    acc_color = COLORS["positive"] if stats["accuracy"] >= 55 else (COLORS["yellow"] if stats["accuracy"] >= 45 else COLORS["negative"])
    cols = st.columns(5)
    metrics = [
        ("TOTAL LOGGED", str(stats["total"]), None),
        ("RESOLVED", str(stats["total_resolved"]), None),
        ("PENDING", str(stats["pending"]), COLORS["blue"]),
        ("ACCURACY", f"{stats['accuracy']}%", acc_color),
        ("AVG RETURN (✓)", f"{stats['avg_return_correct']:+.1f}%" if stats["avg_return_correct"] is not None else "—", COLORS["positive"]),
    ]
    for col, (label, val, color) in zip(cols, metrics):
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


# ── Tab: History ───────────────────────────────────────────────────────────────

def _render_history_tab():
    from services.forecast_tracker import get_stats, evaluate_pending, mark_outcome, delete_forecast

    log = st.session_state.get("_forecast_log") or []

    if not log:
        st.info("No forecasts logged yet.")
        return

    # Controls
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        filter_type = st.selectbox("Filter by type", ["All"] + ["valuation", "squeeze", "regime", "fed", "manual"], key="fa_filter_type")
    with c2:
        filter_outcome = st.selectbox("Filter by outcome", ["All", "pending", "correct", "incorrect", "expired"], key="fa_filter_outcome")
    with c3:
        if st.button("🔄 Evaluate All", use_container_width=True, help="Fetch current prices and resolve past-horizon forecasts"):
            n = evaluate_pending(force=False)
            st.toast(f"Evaluated {n} forecast(s)")
            st.rerun()

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
        if ts:
            eval_due = ts + timedelta(days=horizon)
            eval_due_str = eval_due.strftime("%Y-%m-%d")

        price_at = entry.get("price_at_forecast")
        price_eval = entry.get("price_at_eval")
        price_str = f"${price_at:.2f}" if price_at else "—"
        price_eval_str = f"${price_eval:.2f}" if price_eval else "—"

        tc = _type_color(sig_type)
        with st.expander(
            f"[{fid}] {sig_type.upper()} · {ticker or 'MACRO'} · {prediction} · {ts_str}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])
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
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-bottom:10px;font-size:11px;">'
                    f'<div><span style="color:{COLORS["text_dim"]}">Model:</span> <span style="color:{COLORS["text"]}">{model}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Price@Log:</span> <span style="color:{COLORS["text"]}">{price_str}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Price@Eval:</span> <span style="color:{COLORS["text"]}">{price_eval_str}</span></div>'
                    f'<div><span style="color:{COLORS["text_dim"]}">Eval due:</span> <span style="color:{COLORS["text"]}">{eval_due_str}</span></div>'
                    f'</div>',
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
                if st.button("🗑 Delete", key=f"del_{fid}", use_container_width=True):
                    delete_forecast(fid)
                    st.rerun()


# ── Main render ────────────────────────────────────────────────────────────────

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

    tab_dash, tab_log, tab_hist = st.tabs(["📊 Accuracy Dashboard", "🔮 Log Forecast", "📋 History"])

    with tab_dash:
        _render_dashboard_tab()

    with tab_log:
        _render_log_tab()

    with tab_hist:
        _render_history_tab()
