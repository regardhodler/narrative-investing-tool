"""Quick Intel Run — one button runs Regime + Rate-Path Plays + Current Events + Doom Briefing."""

import streamlit as st
from utils.theme import COLORS


def render():
    _oc = COLORS["bloomberg_orange"]

    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:2px;">⚡ QUICK INTEL RUN</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Runs Risk Regime + Fed Rate Path + Policy Transmission + Current Events + Doom Briefing + Whale Activity + Black Swans + Macro Synopsis in sequence. "
        "Navigate to Portfolio Intelligence when done."
    )

    # ── Highly Regarded gate: check if conditions warrant Sonnet ──────────────
    def _hr_gate_check() -> tuple[bool, str]:
        """Returns (unlocked, reason). Unlocked when macro stress or event warrants Sonnet."""
        reasons = []
        ctx = st.session_state.get("_regime_context", {})
        regime = ctx.get("regime", "")
        score = ctx.get("score", 0)
        quadrant = ctx.get("quadrant", "")

        if "Risk-Off" in regime:
            reasons.append("Risk-Off regime")
        if score < -0.3:
            reasons.append(f"bearish score ({score:+.2f})")
        if quadrant in ("Stagflation", "Deflation"):
            reasons.append(f"{quadrant} quadrant")

        try:
            from datetime import date, timedelta
            from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
            import datetime as _dt
            today = date.today()
            window = today + timedelta(days=1)
            for fn, label in [(get_next_fomc, "FOMC"), (get_next_cpi, "CPI"), (get_next_nfp, "NFP")]:
                ev = fn()
                ev_date = ev.get("date")
                if ev_date:
                    if isinstance(ev_date, str):
                        ev_date = _dt.datetime.strptime(ev_date[:10], "%Y-%m-%d").date()
                    if today <= ev_date <= window:
                        reasons.append(f"{label} day")
        except Exception:
            pass

        doom = st.session_state.get("_doom_briefing", "")
        if doom and any(w in doom.upper() for w in ["CRITICAL", "SEVERE", "HIGH STRESS", "EXTREME"]):
            reasons.append("elevated stress signals")

        return bool(reasons), " · ".join(reasons)

    # ── Engine selector ────────────────────────────────────────────────────────
    import os
    _has_xai      = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    _hr_unlocked, _hr_reason = _hr_gate_check()

    _tier_opts = ["⚡ Groq (fast, free)"]
    if _has_xai:
        _tier_opts.append("🧠 Regard Mode")
    if _has_anthropic and _hr_unlocked:
        _tier_opts.append("👑 Highly Regarded Mode")

    _tier_map = {
        "⚡ Groq (fast, free)":      (False, None),
        "🧠 Regard Mode":            (True, "grok-4-1-fast-reasoning"),
        "👑 Highly Regarded Mode":   (True, "claude-sonnet-4-6"),
    }
    _rec_map = {
        "⚡ Groq (fast, free)":      "Daily routine — all 8 modules in ~90s, completely free.",
        "🧠 Regard Mode":            "Active day — Grok 4.1 reasoning for deeper synthesis + live X feed in Current Events.",
        "👑 Highly Regarded Mode":   "High conviction — Sonnet on all 8 modules before running Portfolio.",
    }
    _sel = st.radio("Engine", _tier_opts, horizontal=True, key="qr_engine")
    st.markdown(
        '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
        'margin-top:-10px;margin-bottom:2px;">'
        '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"💡 {_rec_map.get(_sel, '')}")

    # Show gate status
    if _has_xai and _has_anthropic and not _hr_unlocked:
        st.markdown(
            f'<div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:#f59e0b;margin-bottom:4px;">'
            f'🔒 <b>Highly Regarded Mode locked</b> — unlocks on Risk-Off regime, '
            f'Stagflation/Deflation quadrant, bearish macro score, or FOMC/CPI/NFP day</div>',
            unsafe_allow_html=True,
        )
    elif _hr_unlocked and _has_anthropic:
        st.markdown(
            f'<div style="background:#1a0d00;border:1px solid #FF881166;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:{_oc};margin-bottom:4px;">'
            f'🔓 <b>Highly Regarded Mode unlocked</b> — {_hr_reason}</div>',
            unsafe_allow_html=True,
        )
    elif not _has_xai and not _has_anthropic:
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:4px;'
            f'padding:6px 12px;font-size:10px;color:#8b949e;margin-bottom:4px;">'
            f'ℹ️ Add <b>XAI_API_KEY</b> to unlock Regard Mode (Grok 4.1) · '
            f'Add <b>ANTHROPIC_API_KEY</b> for Highly Regarded (Sonnet)</div>',
            unsafe_allow_html=True,
        )

    # Confirmation gate when Highly Regarded is selected
    _use_claude, _cl_model = _tier_map[_sel]
    if _sel == "👑 Highly Regarded Mode":
        st.warning("👑 Highly Regarded uses Claude Sonnet — reserve for elevated volatility or high-conviction sessions.")
        _confirmed = st.checkbox("I confirm this is a high-conviction session", key="qr_hr_confirm")
        if not _confirmed:
            _use_claude, _cl_model = True, "grok-4-1-fast-reasoning"
            st.caption("*Running in Regard Mode until confirmed.*")

    # ── Signal readiness ───────────────────────────────────────────────────────
    _signal_keys = ["_regime_context", "_dominant_rate_path", "_rp_plays_result", "_fed_plays_result", "_current_events_digest", "_doom_briefing", "_chain_narration", "_custom_swans", "_whale_summary", "_macro_synopsis"]
    _signal_labels = ["Regime", "Fed Rate Path", "Rate-Path Plays", "Fed Plays", "News Digest", "Doom Briefing", "Policy Trans.", "Black Swans", "Whale Activity", "Macro Synopsis"]
    _populated = [(k, l) for k, l in zip(_signal_keys, _signal_labels) if st.session_state.get(k)]

    if _populated:
        _badges = " &nbsp;".join(
            f'<span style="background:#052e16;color:#22c55e;border:1px solid #22c55e44;'
            f'border-radius:3px;padding:1px 7px;font-size:10px;">{l}</span>'
            for _, l in _populated
        )
        st.markdown(
            f'<div style="margin:6px 0 10px 0;">{_badges}</div>',
            unsafe_allow_html=True,
        )

    # ── Run button ─────────────────────────────────────────────────────────────
    if st.button("⚡ RUN ALL INTEL MODULES", type="primary", key="qr_run_all", use_container_width=True):

        _results = {}
        _macro_ctx, _fred_data = {}, {}

        # Step 1: Risk Regime + Rate-Path Plays
        with st.spinner("📡 Fetching market + FRED data, computing regime..."):
            try:
                from modules.risk_regime import run_quick_regime
                _macro_ctx, _fred_data = run_quick_regime(use_claude=_use_claude, model=_cl_model)
                _results["regime"] = True
                st.success("✅ Risk Regime + Rate-Path Plays — done")
            except Exception as e:
                _results["regime"] = False
                st.error(f"❌ Regime failed: {e}")

        # Step 2: Fed Rate Path (uses regime data from step 1)
        with st.spinner("📈 Computing Fed rate path + sector plays..."):
            try:
                from modules.fed_forecaster import run_quick_fed
                run_quick_fed(_macro_ctx, _fred_data, use_claude=_use_claude, model=_cl_model)
                _results["fed"] = True
                st.success("✅ Fed Rate Path — done")
            except Exception as e:
                _results["fed"] = False
                st.error(f"❌ Fed Rate Path failed: {e}")

        # Step 2b: Policy Transmission (uses _rate_path_probs from step 2)
        with st.spinner("🔗 Narrating policy transmission path..."):
            try:
                from modules.fed_forecaster import run_quick_chain
                ok = run_quick_chain(use_claude=_use_claude, model=_cl_model)
                _results["chain"] = ok
                if ok:
                    st.success("✅ Policy Transmission — done")
                else:
                    st.warning("⚠ Policy Transmission skipped — rate path not available")
            except Exception as e:
                _results["chain"] = False
                st.error(f"❌ Policy Transmission failed: {e}")

        # Step 3: Current Events Digest
        with st.spinner("🗞 Fetching headlines + generating digest..."):
            try:
                from modules.current_events import run_quick_digest
                ok = run_quick_digest(use_claude=_use_claude, model=_cl_model)
                _results["digest"] = ok
                if ok:
                    st.success("✅ Current Events Digest — done")
                else:
                    st.warning("⚠ Digest skipped — no content available (check Gist URL + RSS)")
            except Exception as e:
                _results["digest"] = False
                st.error(f"❌ Digest failed: {e}")

        # Step 4: Doom Briefing (uses current events context from step 3)
        with st.spinner("💀 Fetching stress signals + generating briefing..."):
            try:
                from modules.stress_signals import run_quick_doom
                run_quick_doom(use_claude=_use_claude, model=_cl_model)
                _results["doom"] = True
                st.success("✅ Doom Briefing — done")
            except Exception as e:
                _results["doom"] = False
                st.error(f"❌ Doom Briefing failed: {e}")

        # Step 5: Whale Activity (13F scan + AI summary)
        with st.spinner("🐋 Scanning 13F whale filings..."):
            try:
                from modules.whale_buyers import run_quick_whale
                ok = run_quick_whale(use_claude=_use_claude, model=_cl_model)
                _results["whale"] = ok
                if ok:
                    st.success("✅ Whale Activity — done")
                else:
                    st.warning("⚠ Whale scan returned no data — SEC EDGAR may be slow")
            except Exception as e:
                _results["whale"] = False
                st.error(f"❌ Whale Activity failed: {e}")

        # Step 6: Black Swans (auto-generate regime-relevant scenarios)
        with st.spinner("🦢 Generating regime-relevant black swan scenarios..."):
            try:
                from modules.fed_forecaster import run_quick_swans
                ok = run_quick_swans(use_claude=_use_claude, model=_cl_model)
                _results["swans"] = ok
                if ok:
                    _bs_count = len(st.session_state.get("_custom_swans", {}))
                    st.success(f"✅ Black Swans — {_bs_count} scenarios ready")
                else:
                    st.warning("⚠ Black Swan generation returned no results")
            except Exception as e:
                _results["swans"] = False
                st.error(f"❌ Black Swans failed: {e}")

        # Step 7: Macro Conviction Synopsis (cross-signal coherence check)
        with st.spinner("🧠 Synthesizing signals → macro conviction..."):
            try:
                import datetime as _syndt
                from services.claude_client import generate_macro_synopsis as _gen_synopsis

                # Build signal summary from session state
                _rc = st.session_state.get("_regime_context") or {}
                _dp = st.session_state.get("_dominant_rate_path") or {}
                _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
                _sig_parts = []
                if _rc.get("regime"):
                    _sig_parts.append(f"REGIME: {_rc['regime']} (score {_rc.get('score',0):+.2f}) | Quadrant: {_rc.get('quadrant','')}")
                if _dp.get("scenario"):
                    _sig_parts.append(f"FED RATE PATH: {_dp_labels.get(_dp['scenario'], _dp['scenario'])} ({_dp.get('prob_pct',0):.0f}% probability)")
                _doom = st.session_state.get("_doom_briefing", "")
                if _doom:
                    _sig_parts.append(f"DOOM BRIEFING: {_doom[:400]}")
                _digest = st.session_state.get("_current_events_digest", "")
                if _digest:
                    _sig_parts.append(f"NEWS DIGEST: {_digest[:400]}")
                _chain = st.session_state.get("_chain_narration", "")
                if _chain:
                    _sig_parts.append(f"POLICY TRANSMISSION: {_chain[:300]}")
                _whale = st.session_state.get("_whale_summary", "")
                if _whale:
                    _sig_parts.append(f"WHALE ACTIVITY: {_whale[:300]}")
                _bs = st.session_state.get("_custom_swans", {})
                if _bs:
                    _bs_summary = "; ".join(
                        f"{k} ({v.get('probability_pct', 0):.1f}% annual)"
                        for k, v in list(_bs.items())[:3]
                    )
                    _sig_parts.append(f"BLACK SWANS: {_bs_summary}")

                _synopsis = _gen_synopsis("\n\n".join(_sig_parts), use_claude=_use_claude, model=_cl_model)
                _syn_tier = "👑 Highly Regarded Mode" if (_use_claude and _cl_model == "claude-sonnet-4-6") \
                    else ("🧠 Regard Mode" if _use_claude else "⚡ Groq")
                st.session_state["_macro_synopsis"] = _synopsis
                st.session_state["_macro_synopsis_ts"] = _syndt.datetime.now()
                st.session_state["_macro_synopsis_engine"] = _syn_tier
                _results["synopsis"] = True
                st.success(f"✅ Macro Conviction Synopsis — {_synopsis.get('conviction', '?')}")
            except Exception as e:
                _results["synopsis"] = False
                st.error(f"❌ Synopsis failed: {e}")

        # ── Completion summary ─────────────────────────────────────────────────
        _n_ok = sum(1 for v in _results.values() if v)
        if _n_ok == 8:
            st.markdown(
                f'<div style="background:#052e16;border:1px solid #22c55e;border-radius:6px;'
                f'padding:12px 16px;margin-top:10px;">'
                f'<div style="color:#22c55e;font-weight:700;font-size:13px;">✅ All intel modules ready</div>'
                f'<div style="color:#86efac;font-size:11px;margin-top:4px;">'
                f'Navigate to <b>Portfolio Intelligence</b> to run your final analysis.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"{_n_ok}/8 modules completed — check errors above.")

        # ── Signal Coverage Panel ──────────────────────────────────────────────
        from datetime import datetime as _dt2
        _coverage_signals = [
            ("Regime",             "_regime_context",        "_regime_context_ts"),
            ("Fed Rate Path",      "_dominant_rate_path",    "_rate_path_probs_ts"),
            ("Fed Funds Rate",     "_fed_funds_rate",        None),
            ("Rate-Path Plays",    "_fed_plays_result",      "_fed_plays_result_ts"),
            ("Regime Plays",       "_rp_plays_result",       None),
            ("Doom Briefing",      "_doom_briefing",         "_doom_briefing_ts"),
            ("Policy Trans.",      "_chain_narration",       None),
            ("Black Swans",        "_custom_swans",          "_custom_swans_ts"),
            ("Whale Activity",     "_whale_summary",         "_whale_summary_ts"),
            ("Current Events",     "_current_events_digest", "_current_events_digest_ts"),
        ]
        _now2 = _dt2.now()
        _n_loaded = sum(1 for _, k, _ in _coverage_signals if st.session_state.get(k))
        _bar_pct = int(_n_loaded / len(_coverage_signals) * 100)
        _bar_color = "#22c55e" if _n_loaded == len(_coverage_signals) else ("#f59e0b" if _n_loaded >= 7 else "#ef4444")

        _left = _coverage_signals[:5]
        _right = _coverage_signals[5:]
        _rows_html = ""
        for (lbl_l, k_l, ts_l), (lbl_r, k_r, ts_r) in zip(_left, _right):
            def _sig_cell(lbl, k, ts_k):
                ok = bool(st.session_state.get(k))
                icon = f'<span style="color:#22c55e;">✓</span>' if ok else '<span style="color:#ef4444;">✗</span>'
                age = ""
                if ok and ts_k:
                    _ts = st.session_state.get(ts_k)
                    if _ts:
                        _m = int((_now2 - _ts).total_seconds() / 60)
                        age = f' <span style="color:#555;font-size:10px;">· {"just now" if _m < 1 else (f"{_m}m ago" if _m < 60 else f"{_m//60}h ago")}</span>'
                color = "#e2e8f0" if ok else "#475569"
                return f'<td style="padding:2px 12px 2px 0;white-space:nowrap;">{icon} <span style="color:{color};">{lbl}</span>{age}</td>'
            _rows_html += f"<tr>{_sig_cell(lbl_l, k_l, ts_l)}{_sig_cell(lbl_r, k_r, ts_r)}</tr>"

        st.markdown(
            f'<div style="border:1px solid #334155;border-radius:6px;padding:10px 14px;margin-top:10px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase;">Prompt Context</span>'
            f'<span style="font-size:10px;color:{_bar_color};font-weight:600;">{_n_loaded}/{len(_coverage_signals)} signals loaded</span>'
            f'</div>'
            f'<div style="height:2px;background:#1e293b;border-radius:1px;margin-bottom:8px;">'
            f'<div style="height:2px;width:{_bar_pct}%;background:{_bar_color};border-radius:1px;"></div>'
            f'</div>'
            f'<table style="width:100%;font-size:11px;font-family:monospace;border-collapse:collapse;">'
            f'{_rows_html}</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Previews ───────────────────────────────────────────────────────────
        _digest = st.session_state.get("_current_events_digest", "")
        _doom = st.session_state.get("_doom_briefing", "")
        _plays = st.session_state.get("_rp_plays_result") or {}

        _synopsis = st.session_state.get("_macro_synopsis") or {}
        if _synopsis.get("conviction"):
            _conv = _synopsis["conviction"]
            _conv_color = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "MIXED": "#f59e0b", "UNCERTAIN": "#94a3b8"}.get(_conv, "#94a3b8")
            _conv_bg   = {"BULLISH": "#052e16", "BEARISH": "#1a0000", "MIXED": "#1a1200", "UNCERTAIN": "#0d1117"}.get(_conv, "#0d1117")
            _kp_html = "".join(
                f'<div style="color:#94a3b8;font-size:11px;padding:2px 0;"> · {kp}</div>'
                for kp in _synopsis.get("key_points", [])
            )
            _ct_html = ""
            for ct in _synopsis.get("contradictions", []):
                _ct_html += (
                    f'<div style="color:#f59e0b;font-size:10px;padding:2px 0;">'
                    f'⚠ {ct}</div>'
                )
            _syn_eng = st.session_state.get("_macro_synopsis_engine", "")
            _eng_span = f'<span style="font-size:10px;color:#555;margin-left:auto;">{_syn_eng}</span>' if _syn_eng else ""
            st.markdown(
                f'<div style="background:{_conv_bg};border:2px solid {_conv_color};border-radius:8px;'
                f'padding:14px 18px;margin-top:14px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="background:{_conv_color};color:black;font-weight:800;font-size:12px;'
                f'padding:3px 12px;border-radius:4px;letter-spacing:0.08em;">{_conv}</span>'
                f'<span style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.08em;">MACRO CONVICTION</span>'
                f'{_eng_span}'
                f'</div>'
                f'<div style="color:#e2e8f0;font-size:12px;line-height:1.6;margin-bottom:6px;">{_synopsis.get("summary","")}</div>'
                f'{_kp_html}'
                f'{_ct_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        if _digest or _doom or _plays or st.session_state.get("_chain_narration") or st.session_state.get("_whale_summary"):
            st.markdown(
                f'<div style="border-top:1px solid {COLORS["border"]};margin:14px 0 10px 0;"></div>',
                unsafe_allow_html=True,
            )

        _dp = st.session_state.get("_dominant_rate_path") or {}
        if _dp:
            _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
            _dp_label = _dp_labels.get(_dp.get("scenario", ""), _dp.get("scenario", ""))
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid {COLORS["border"]};border-radius:4px;'
                f'padding:8px 12px;font-size:11px;color:{COLORS["text_dim"]};margin-bottom:6px;">'
                f'📈 <b style="color:{COLORS["bloomberg_orange"]}">Fed Rate Path</b> — '
                f'Dominant: <b style="color:{COLORS["text"]}">{_dp_label}</b> '
                f'({_dp.get("prob_pct", 0):.0f}% probability)</div>',
                unsafe_allow_html=True,
            )

        if _plays:
            _regime = st.session_state.get("_regime_context", {})
            _regime_label = _regime.get("regime", "")
            _quad = _regime.get("quadrant", "")
            with st.expander(f"📡 Regime: {_regime_label} · {_quad}", expanded=True):
                _sectors = _plays.get("sectors", [])
                _stocks = _plays.get("stocks", [])
                if _sectors:
                    st.markdown("**Sectors:** " + " · ".join(
                        f"{s.get('name','')} ({'★'*s.get('conviction',1)})" for s in _sectors[:4]
                    ))
                if _stocks:
                    st.markdown("**Stocks:** " + " · ".join(
                        f"{s.get('ticker','')} ({'★'*s.get('conviction',1)})" for s in _stocks[:4]
                    ))

        if _digest:
            with st.expander("🗞 News Digest", expanded=True):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_digest}</div>',
                    unsafe_allow_html=True,
                )

        if _doom:
            with st.expander("💀 Doom Briefing", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_doom}</div>',
                    unsafe_allow_html=True,
                )

        _chain = st.session_state.get("_chain_narration", "")
        if _chain:
            with st.expander("🔗 Policy Transmission", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_chain}</div>',
                    unsafe_allow_html=True,
                )

        _whale = st.session_state.get("_whale_summary", "")
        if _whale:
            with st.expander("🐋 Whale Activity", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_whale}</div>',
                    unsafe_allow_html=True,
                )

        _bs = st.session_state.get("_custom_swans", {})
        if _bs:
            _bs_names = " · ".join(list(_bs.keys())[:3])
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #4B5EAA44;border-radius:4px;'
                f'padding:8px 12px;font-size:11px;color:{COLORS["text_dim"]};margin-top:6px;">'
                f'🦢 <b style="color:#8899CC">Black Swans</b> — '
                f'{len(_bs)} scenarios: <span style="color:{COLORS["text"]}">{_bs_names}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Data Flow Legend ───────────────────────────────────────────────────────
    with st.expander("📊 Data Flow", expanded=False):
        _oc = COLORS["bloomberg_orange"]
        st.markdown(
            f"""
<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;line-height:1.9;">

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">DATA SOURCES</div>
<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">
  {''.join(f'<span style="background:#1e293b;border:1px solid #334155;border-radius:3px;padding:2px 8px;color:#94a3b8;">{s}</span>' for s in ['yfinance','FRED API','RSS feeds','Polymarket Gist','📱 Telegram inbox','SEC EDGAR'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">SIGNAL LAYER <span style="font-weight:400;color:#475569;">(Quick Intel Run generates these)</span></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Regime + Quadrant','Rate Path Plays','Fed Funds Rate','Current Events Digest','Doom Briefing','Whale Activity','Black Swans','Policy Transmission','Trending Narratives','Auto-Trending Groups'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">CONSUMERS</div>
<div style="display:flex;flex-direction:column;gap:4px;">
  <div style="background:#0c1a0c;border:1px solid #22c55e44;border-radius:4px;padding:6px 10px;">
    <span style="color:#22c55e;font-weight:700;">Portfolio Intelligence</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">uses ALL signals — regime · rate path · news digest · doom · whales · swans · current events · trending narratives · auto-trending groups</span>
  </div>
  <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;padding:6px 10px;">
    <span style="color:#f59e0b;font-weight:700;">Discovery</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · rate path · fed plays · regime plays · doom · whales · swans · current events · trending narratives · auto-trending groups (feeds Cross-Signal Plays)</span>
  </div>
  <div style="background:#0d1117;border:1px solid #3b82f644;border-radius:4px;padding:6px 10px;">
    <span style="color:#3b82f6;font-weight:700;">Valuation</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · rate path · fed plays · regime plays · doom · whales · swans · current events · trending narratives · auto-trending groups · DCF + Elliott Wave + Wyckoff</span>
  </div>
</div>

<div style="margin-top:12px;padding-top:8px;border-top:1px solid #1e293b;color:#334155;font-size:10px;">
  📱 Telegram inbox → Current Events Digest → Portfolio Intelligence + Doom Briefing
</div>

</div>""",
            unsafe_allow_html=True,
        )
