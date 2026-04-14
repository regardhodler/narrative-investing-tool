"""Quick Intel Run — one button runs Regime + Rate-Path Plays + Current Events + Doom Briefing."""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.theme import COLORS
from utils.ai_tier import TIER_OPTS, TIER_MAP, MODEL_HINT_HTML
from utils.components import (
    render_rr_score_mode_toggle,
    render_intel_health_bar,
    render_action_queue,
    apply_confidence_penalty,
)


# ── QIR Dashboard helpers ────────────────────────────────────────────────────

_RATE_PATH_SCALE = {
    "CUT_75": 1.0, "CUT_50": 0.75, "CUT_25": 0.5,
    "HOLD": 0.0,
    "HIKE_25": -0.5, "HIKE_50": -0.75, "HIKE_100": -1.0,
}

_PATTERNS = {
    "BULLISH_CONFIRMATION": {
        "label": "BULLISH CONFIRMATION",
        "color": "#22c55e",
        "bullish": True,
        "interpretation": "All three timing layers aligned bullish — highest-conviction long entry.",
        "buy_tier": "STRONG",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("QQQ / SPY", "Broad market long — simplest expression of risk-on"),
            ("XLK", "Tech leads in risk-on / Goldilocks — amplified beta"),
            ("SPY Calls", "ATM or 5% OTM, 30–60 DTE — leveraged with defined risk"),
            ("TQQQ / UPRO", "Max leverage for highest-conviction only — short duration"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Enter on pullbacks to 20d MA or prior breakout level — not at all-time highs\n"
            "Confirm with breadth: >60% of S&P 500 above 50d MA\n"
            "Stop: close below 20d MA or -5% from entry, whichever is tighter\n"
            "Scale: 50% at entry, add 25% on first successful retest"
        ),
        "entry_short": "All layers lean bullish — avoid net short exposure.\nIf forced: only extreme overbought bounces, very tight stops.",
    },
    "BEARISH_CONFIRMATION": {
        "label": "BEARISH CONFIRMATION",
        "color": "#ef4444",
        "bullish": False,
        "interpretation": "All three layers aligned bearish — highest-conviction short or cash environment.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "STRONG",
        "instruments_buy": [],
        "instruments_short": [
            ("SH", "1× inverse S&P 500 — no decay, suits multi-week holds"),
            ("SPY Puts", "ATM or 5% OTM, 30–60 DTE — defined risk, leverage"),
            ("SDS", "2× inverse S&P — short duration only (decay risk)"),
            ("SQQQ", "3× inverse Nasdaq — tech-led selloff only, size small"),
        ],
        "entry_buy": "No layers confirm bullish conditions — hold cash or hedge.\nNo new longs until at least Tactical turns.",
        "entry_short": (
            "Enter on dead-cat bounces into resistance — not into freefall\n"
            "Confirm with failed rally: rejection at 50d MA on volume\n"
            "Stop: close above last swing high or 50d MA\n"
            "Scale out: 50% at first target, trail stop on remainder"
        ),
    },
    "PULLBACK_IN_UPTREND": {
        "label": "PULLBACK IN UPTREND",
        "color": "#f59e0b",
        "bullish": True,
        "interpretation": "Regime and Options Flow confirm bull trend — Tactical dip is a buy-the-dip setup.",
        "buy_tier": "STRONG",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("QQQ / SPY", "Broad market — buy the dip in the uptrend"),
            ("Sector leaders", "XLK, XLY — highest-momentum sectors in the current regime"),
            ("SPY Calls", "30–60 DTE calls capture the bounce with defined risk"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Scale in on red days — Tactical weakness is your entry window\n"
            "Stop: close below 20d MA\n"
            "Target: prior swing high / all-time high\n"
            "Wait for Tactical to turn ≥55 before adding full size"
        ),
        "entry_short": "Trend is up — pullbacks are entries, not reversals.\nAvoiding fighting the regime.",
    },
    "OPTIONS_FLOW_DIVERGENCE": {
        "label": "OPTIONS FLOW DIVERGENCE",
        "color": "#f59e0b",
        "bullish": True,
        "interpretation": "Regime and Tactical bullish but options crowd is hedging — smart money buying protection.",
        "buy_tier": "MODERATE",
        "short_tier": "SELECTIVE",
        "instruments_buy": [
            ("SPY", "Broad market — reduced size until Options Flow confirms"),
            ("MSFT / GOOGL", "Defensive growth — quality names hold up if crowd is right"),
        ],
        "instruments_short": [
            ("Weak sector ETFs", "XLE, XLU — hedge only, not directional short"),
        ],
        "entry_buy": (
            "Smaller position size — wait for Options Flow to confirm ≥65\n"
            "Prefer quality (MSFT, GOOGL) over high-beta names\n"
            "Stop: tighter than normal — -3% from entry"
        ),
        "entry_short": "Treat as hedge only — not a directional short setup.\nOptions crowd may be wrong; regime and tactical are bullish.",
    },
    "BEAR_MARKET_BOUNCE": {
        "label": "BEAR MARKET BOUNCE",
        "color": "#f97316",
        "bullish": False,
        "interpretation": "Short-term momentum against the macro trend — take profits quickly, don't chase.",
        "buy_tier": "SELECTIVE",
        "short_tier": "MODERATE",
        "instruments_buy": [
            ("QQQ / ARKK", "Momentum names — bounce candidates, not trend reversals"),
        ],
        "instruments_short": [
            ("SH", "1× inverse S&P — size for the eventual trend resumption"),
            ("XLE / XLF Puts", "Weak sectors underperform when the bounce fades"),
        ],
        "entry_buy": (
            "Tight stops — this is a counter-trend trade\n"
            "Take 50% profits at first resistance\n"
            "Do not add to winners — the macro trend is down"
        ),
        "entry_short": (
            "Wait for the bounce to fade — sell into strength, not freefall\n"
            "Stop: above the bounce high\n"
            "Scale in: build position in tranches as momentum rolls over"
        ),
    },
    "LATE_CYCLE_SQUEEZE": {
        "label": "LATE CYCLE SQUEEZE",
        "color": "#ef4444",
        "bullish": False,
        "interpretation": "Options crowd squeezing higher against a bearish regime and tactical — high risk of reversal.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "STRONG",
        "instruments_buy": [],
        "instruments_short": [
            ("SH / SDS", "Build short in tranches — fade the squeeze"),
            ("QQQ Puts", "45–60 DTE — defined risk on the squeeze reversal"),
        ],
        "entry_buy": "Don't chase the squeeze — regime and tactical are bearish.\nWait for regime to confirm before entering longs.",
        "entry_short": (
            "Build short position in tranches — don't front-run the squeeze\n"
            "Stop: above squeeze high\n"
            "Target: regime-implied support level\n"
            "Patience required — squeezes can persist 1–3 sessions"
        ),
    },
    "GENUINE_UNCERTAINTY": {
        "label": "GENUINE UNCERTAINTY",
        "color": "#7c3aed",
        "bullish": None,
        "interpretation": "Signals are conflicting — but uncertainty is quantified, not an excuse for inaction. A probabilistic lean is forced below. Size down to 25–40% of normal. Act small, act smart.",
        "buy_tier": "LEAN SETUP — SIZE DOWN",
        "short_tier": "LEAN SETUP — SIZE DOWN",
        "instruments_buy": [
            ("SPY (25% size)", "Broad exposure at reduced size — captures upside if lean is correct"),
            ("Cash-secured puts (OTM)", "Collect premium in uncertain tape — defined max loss"),
        ],
        "instruments_short": [
            ("Collars on longs", "Buy OTM puts against existing longs — hedge without exiting"),
            ("VXX / UVXY (small)", "Vol insurance if uncertainty resolves bearishly — defined risk"),
        ],
        "entry_buy": (
            "Size to 25–40% of normal — uncertainty demands margin of safety\n"
            "Enter only if lean probability ≥55% from the Uncertainty Panel below\n"
            "Stop: -3% hard stop — no exceptions in uncertain tape\n"
            "Wait for one more layer to confirm before adding size"
        ),
        "entry_short": (
            "Size to 25–40% of normal — hedges only, not directional short\n"
            "Prefer collars or puts against existing positions over outright shorts\n"
            "Stop: close above prior swing high\n"
            "Remove hedge if lean probability flips or regime resolves bullish"
        ),
    },
}


def _regime_velocity_pts(current_macro_score: float, is_bullish: bool | None) -> float:
    """Return directional velocity pts (±8 max) from tactical_score_history.json."""
    import json as _json, os as _os
    try:
        _path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "tactical_score_history.json")
        with open(_path) as _f:
            _hist = _json.load(_f)
        if not _hist or len(_hist) < 6:
            return 0.0
        _old_score = float(_hist[-6].get("score", 50))
        velocity = current_macro_score - _old_score
        vel_norm = max(-1.0, min(1.0, velocity / 25.0))
        if is_bullish is True:
            vel_dir = 1 if vel_norm > 0 else (-1 if vel_norm < 0 else 0)
        elif is_bullish is False:
            vel_dir = 1 if vel_norm < 0 else (-1 if vel_norm > 0 else 0)
        else:
            vel_dir = 0
        return vel_dir * abs(vel_norm) * 8
    except Exception:
        return 0.0


def _classify_signals(regime_ctx: dict, tac_ctx: dict, of_ctx: dict) -> dict:
    """Map three timing signal contexts to one of 7 dashboard patterns.

    Uses identical direction thresholds to the existing Short/Buy panel logic.
    Returns full pattern dict with label, color, tiers, instruments, and entry rules.
    """
    regime_ctx = regime_ctx or {}
    tac_ctx    = tac_ctx    or {}
    of_ctx     = of_ctx     or {}

    score        = regime_ctx.get("score", 0.0)
    regime_label = regime_ctx.get("regime", "")
    tac_score    = tac_ctx.get("tactical_score", 50) if tac_ctx else 50
    of_score     = of_ctx.get("options_score", 50)   if of_ctx  else 50

    # Thresholds match existing lines 221–226 of quick_run.py
    r_bull = "Risk-On"  in regime_label or score >  0.3
    r_bear = "Risk-Off" in regime_label or score < -0.3
    t_bull = tac_score >= 65
    t_bear = tac_score <  38
    o_bull = of_score  >= 65
    o_bear = of_score  <  38

    if   r_bull and t_bull and o_bull: pattern = "BULLISH_CONFIRMATION"
    elif r_bear and t_bear and o_bear: pattern = "BEARISH_CONFIRMATION"
    elif r_bull and t_bear and o_bull: pattern = "PULLBACK_IN_UPTREND"
    elif r_bull and t_bull and o_bear: pattern = "OPTIONS_FLOW_DIVERGENCE"
    elif r_bear and t_bull and o_bull: pattern = "BEAR_MARKET_BOUNCE"
    elif r_bear and t_bear and o_bull: pattern = "LATE_CYCLE_SQUEEZE"
    else:                              pattern = "GENUINE_UNCERTAINTY"

    pat_meta = _PATTERNS[pattern]

    # ── Conviction Score ──────────────────────────────────────────────────────
    # Measures HOW STRONG the pattern is, not just WHICH pattern fired.
    # Binary thresholds above decide the pattern; signal magnitudes decide conviction.
    # GENUINE_UNCERTAINTY uses its own domain math — no conviction score needed.
    conviction_score = None
    conviction_size_mult = None
    conviction_size_label = None
    leading_warning = None
    _cv_comps = {}

    if pattern != "GENUINE_UNCERTAINTY":
        _regime_pts  = abs(score) * 40                              # max 40 — regime strength
        _tac_pts     = abs(tac_score - 50) / 50.0 * 30             # max 30 — tactical strength
        _opts_pts    = abs(of_score - 50)  / 50.0 * 20             # max 20 — options strength

        # Staleness weight: regime data older than 24h gets downweighted
        import time as _time
        try:
            _regime_ts  = st.session_state.get("_regime_context_ts") or 0
            _age_hours  = (_time.time() - float(_regime_ts)) / 3600.0
            _age_weight = max(0.75, 1.0 - (_age_hours / 48.0))
        except Exception:
            _age_weight = 1.0
        _regime_pts *= _age_weight

        # Leading divergence: does the early-warning direction confirm or contradict?
        _leading_div = regime_ctx.get("leading_divergence", 0) or 0
        _is_bullish  = pat_meta.get("bullish")
        if _is_bullish is True:
            _dir_match = 1 if _leading_div > 0 else (-1 if _leading_div < 0 else 0)
        elif _is_bullish is False:
            _dir_match = 1 if _leading_div < 0 else (-1 if _leading_div > 0 else 0)
        else:
            _dir_match = 0
        _lead_pts = _dir_match * min(abs(_leading_div) / 20.0, 1.0) * 10  # ±10 pts

        # Rate path: normalized -1..+1, directional vs pattern, max ±8 pts
        try:
            _rate_path = st.session_state.get("_dominant_rate_path") or "HOLD"
            _rate_val  = _RATE_PATH_SCALE.get(str(_rate_path).upper(), 0.0)
            _rate_dir  = 1 if ((_is_bullish is True  and _rate_val > 0) or
                               (_is_bullish is False and _rate_val < 0)) else -1
            _rate_pts  = _rate_dir * abs(_rate_val) * 8
        except Exception:
            _rate_pts = 0.0

        # Whale flow: bull_pct normalized, directional, max ±10 pts
        try:
            _wf        = st.session_state.get("_whale_flow_score") or {}
            _bull_pct  = float(_wf.get("bull_pct", 50))
            _wf_norm   = (_bull_pct - 50) / 50.0
            _wf_dir    = 1 if ((_is_bullish is True  and _wf_norm > 0) or
                               (_is_bullish is False and _wf_norm < 0)) else -1
            _whale_pts = _wf_dir * abs(_wf_norm) * 10
        except Exception:
            _whale_pts = 0.0

        # Regime velocity — how fast is the regime moving? max ±8 pts
        _macro_score_0to100 = (score + 1.0) * 50.0
        _vel_pts = _regime_velocity_pts(_macro_score_0to100, _is_bullish)

        # Sum and apply fear multiplier
        _raw = (_regime_pts + _rate_pts + _tac_pts + _opts_pts +
                _whale_pts + _lead_pts + _vel_pts)

        try:
            _fc         = st.session_state.get("_fear_composite") or {}
            _fear_score = float(_fc.get("score", 50))
            _fear_ext   = abs(_fear_score - 50) / 50.0
            _fear_mult  = 1.0 - _fear_ext * 0.30
        except Exception:
            _fear_mult = 1.0

        conviction_score = int(max(0, min(100, round(_raw * _fear_mult))))

        # Store component breakdown for the UI card
        _cv_comps = {
            "regime":   round(_regime_pts, 1),
            "tactical": round(_tac_pts, 1),
            "options":  round(_opts_pts, 1),
            "leading":  round(_lead_pts, 1),
            "rate":     round(_rate_pts, 1),
            "whale":    round(_whale_pts, 1),
            "velocity": round(_vel_pts, 1),
            "fear_mult": round(_fear_mult, 3),
            "hmm_mult":  1.0,
        }

        # HMM regime state modifier — passive, secondary dampener
        try:
            from services.hmm_regime import load_current_hmm_state, get_conviction_multiplier
            _hmm_state = load_current_hmm_state()
            if _hmm_state is not None:
                _hmm_entropy = getattr(_hmm_state, "entropy", 0.0) or 0.0
                _hmm_mult = get_conviction_multiplier(_hmm_state.state_label, entropy=_hmm_entropy)
                _cv_comps["hmm_mult"] = round(_hmm_mult, 3)
                # Persistence bonus: >10 days in same state adds confidence
                if _hmm_state.persistence > 10:
                    conviction_score = min(int(round(conviction_score * _hmm_mult)) + 3, 100)
                else:
                    conviction_score = int(max(0, min(100, round(conviction_score * _hmm_mult))))
        except Exception:
            _hmm_state = None

        # Position sizing from conviction
        if conviction_score >= 75:
            conviction_size_mult, conviction_size_label = 0.50, "50% SIZE"
        elif conviction_score >= 55:
            conviction_size_mult, conviction_size_label = 0.40, "40% SIZE"
        elif conviction_score >= 40:
            conviction_size_mult, conviction_size_label = 0.30, "30% SIZE"
        else:
            conviction_size_mult, conviction_size_label = 0.20, "20% SIZE"

        # Warning: leading divergence is running against the pattern direction
        if _dir_match == -1 and abs(_leading_div) > 5:
            _dir_word = "below" if _is_bullish else "above"
            leading_warning = (
                f"Leading sub-score running {abs(_leading_div)} pts {_dir_word} composite — "
                f"fast signals contradict this pattern. Conviction reduced by ~{abs(int(_lead_pts))} pts. "
                f"Fast signals may be early or this pattern may be near exhaustion."
            )

    return {
        "pattern": pattern,
        "conviction_score": conviction_score,
        "conviction_size_mult": conviction_size_mult,
        "conviction_size_label": conviction_size_label,
        "leading_warning": leading_warning,
        "conviction_components": _cv_comps if pattern != "GENUINE_UNCERTAINTY" else {},
        **pat_meta,
    }


def _classify_entry_recommendation(
    leading_score: int,
    macro_score: int,
    tactical_score: int,
    options_score: int,
    divergence_label: str,
    divergence_pts: int,
    velocity_delta: int = 0,
) -> dict:
    """Map leading/lagging indicator divergence + tactical + options + macro velocity into a single entry verdict.

    velocity_delta: 5-day macro score change (positive = strengthening, negative = deteriorating).
    Rapid deterioration (< -8) overrides bullish setups to WAIT; rapid improvement (> 8) reinforces
    bullish setups and softens cautious holds.
    """
    leading_bull  = leading_score  >= 55
    leading_bear  = leading_score  <  44
    macro_bear    = macro_score    <  40
    tac_dip       = tactical_score <  48
    tac_rip       = tactical_score >= 62
    opts_bearish  = options_score  <  40
    opts_bullish  = options_score  >= 60
    early_risk_on  = divergence_label == "Early Risk-On Setup"
    early_risk_off = divergence_label == "Early Risk-Off Warning"
    large_div      = abs(divergence_pts) >= 8

    # Velocity regime — captures rate-of-change of macro conditions.
    # Threshold of ±8 pts/5d corresponds to roughly 1σ of weekly swing.
    vel_accelerating = velocity_delta >  8   # macro rapidly improving
    vel_deteriorating = velocity_delta < -8  # macro rapidly deteriorating

    if leading_bear and tac_rip and not macro_bear:
        verdict = "SELL THE RIP"
    elif early_risk_off and large_div and tac_rip:
        verdict = "SELL THE RIP"
    # Rapidly deteriorating macro overrides buy signals regardless of leading/tactical positioning.
    # Exception: if leading is still strongly bull AND deterioration is shallow (<-15), allow a WAIT.
    elif vel_deteriorating and not leading_bull:
        verdict = "SELL THE RIP" if tac_rip else "WAIT"
    elif vel_deteriorating and leading_bull and not macro_bear:
        # Leading still holding up but macro is sliding — downgrade to WAIT, not a full buy
        verdict = "WAIT"
    elif leading_bull and tac_dip and not macro_bear:
        # Options flow is a same-day confirming signal — bearish flow blocks unless early risk-on or accelerating
        if opts_bearish and not early_risk_on and not vel_accelerating:
            verdict = "WAIT"
        else:
            verdict = "BUY THE DIP"
    elif early_risk_on and large_div and tac_dip and not macro_bear:
        # Even with early risk-on divergence, very bearish options flow is a veto
        verdict = "WAIT" if opts_bearish else "BUY THE DIP"
    elif early_risk_off and large_div:
        verdict = "WAIT"
    elif leading_bear and not tac_rip:
        verdict = "WAIT"
    elif leading_bull and tac_dip and macro_bear:
        verdict = "WAIT"
    elif not leading_bull and not leading_bear and tac_dip and opts_bearish:
        verdict = "WAIT"
    else:
        verdict = "HOLD"
        if opts_bullish and leading_bull and tac_dip:
            verdict = "BUY THE DIP"
        elif opts_bearish and not leading_bull:
            # Options flow bearish + no bullish leadership → downgrade hold to wait
            verdict = "WAIT"
        # Rapidly improving macro softens a HOLD to a positive lean — note added to reasoning

    _meta = {
        "BUY THE DIP":  ("#22c55e", "#052e16", "▲"),
        "HOLD":         ("#4B9FFF", "#0a1628", "◆"),
        "WAIT":         ("#FFD700", "#1a1200", "◌"),
        "SELL THE RIP": ("#ef4444", "#2d0a0a", "▼"),
    }
    color, bg, icon = _meta[verdict]

    div_sign  = f"+{divergence_pts}" if divergence_pts >= 0 else str(divergence_pts)
    vel_sign  = f"+{velocity_delta}" if velocity_delta >= 0 else str(velocity_delta)
    opts_note = ""
    if opts_bearish:
        opts_note = f" Options flow bearish ({options_score}/100) — confirms caution."
    elif opts_bullish:
        opts_note = f" Options flow bullish ({options_score}/100) — confirms setup."
    vel_note  = ""
    if vel_accelerating:
        vel_note = f" Macro velocity {vel_sign} pts/5d — regime accelerating, supports entry."
    elif vel_deteriorating:
        vel_note = f" Macro velocity {vel_sign} pts/5d — regime deteriorating fast, raises bar for new longs."

    if verdict == "BUY THE DIP":
        reasoning = (
            f"Leading indicators healthy at {leading_score}/100 vs macro {macro_score}/100 "
            f"({div_sign} pts divergence). "
            f"Tactical pullback to {tactical_score}/100 creates a favorable entry before lagging confirms."
            f"{opts_note}{vel_note}"
        )
    elif verdict == "HOLD":
        reasoning = (
            f"All layers aligned — leading {leading_score}/100, macro {macro_score}/100, "
            f"tactical {tactical_score}/100. "
            f"No new entry trigger or exit signal; maintain existing positions."
            f"{opts_note}{vel_note}"
        )
    elif verdict == "WAIT":
        reasoning = (
            f"Leading score ({leading_score}/100) diverging {div_sign} pts from composite — "
            f"fast signals have weakened. "
            f"Hold new entries until divergence resolves or macro catches down."
            f"{opts_note}{vel_note}"
        )
    else:
        reasoning = (
            f"Leading indicators cracking ({leading_score}/100) while price remains elevated "
            f"(tactical {tactical_score}/100). "
            f"Use current strength to reduce exposure before lagging composite confirms the move."
            f"{opts_note}{vel_note}"
        )

    return {
        "verdict":          verdict,
        "color":            color,
        "bg":               bg,
        "icon":             icon,
        "reasoning":        reasoning,
        "leading_score":    leading_score,
        "macro_score":      macro_score,
        "divergence_pts":   divergence_pts,
        "divergence_label": divergence_label,
        "velocity_delta":   velocity_delta,
    }


def _build_uncertainty_profile(rc: dict, tac: dict, of_ctx: dict) -> dict:
    """Decompose the GENUINE_UNCERTAINTY state into a 5-domain uncertainty profile.

    Derives all data from already-populated session_state + the three core signal
    dicts. No API calls. Returns a profile dict consumed by _render_genuine_uncertainty_panel.
    """
    rc      = rc      or {}
    tac     = tac     or {}
    of_ctx  = of_ctx  or {}

    # ── Raw scores (0-100 normalised) ────────────────────────────────────────
    regime_score_raw = rc.get("score", 0.0)           # z-score, roughly -1..+1
    regime_score_100 = min(100, max(0, int((regime_score_raw + 1) / 2 * 100)))
    tac_score        = tac.get("tactical_score", 50)
    of_score         = of_ctx.get("options_score", 50)

    # ── Macro domain ─────────────────────────────────────────────────────────
    quadrant   = rc.get("quadrant", "")
    regime_lbl = rc.get("regime", "")
    doom       = st.session_state.get("_doom_briefing", "") or ""
    fear_greed = st.session_state.get("_fear_greed") or {}
    fg_score   = fear_greed.get("score", 50) if isinstance(fear_greed, dict) else 50

    # Blend composite score with Fear & Greed, then nudge by leading divergence.
    # leading_divergence > 0 means fast signals are ahead of lagging ones → slight bullish bias.
    _leading_div = rc.get("leading_divergence", 0) or 0
    _leading_lbl = rc.get("leading_label", "Aligned") or "Aligned"
    _leading_nudge = int(_leading_div * 0.25)  # max ±5 pts at ±20 divergence
    macro_score = int(regime_score_100 * 0.5 + fg_score * 0.5) + _leading_nudge
    macro_score = max(0, min(100, macro_score))
    macro_flag  = (
        "CONFLICTED" if 38 <= macro_score <= 62 else
        "MILD BULL"  if macro_score > 62 else
        "MILD BEAR"
    )
    _leading_suffix = f" | Leading: {_leading_lbl} ({_leading_div:+d})" if _leading_lbl != "Aligned" else ""
    macro_detail = f"{regime_lbl or 'Regime unclear'} | Quadrant: {quadrant or '—'} | F&G: {fg_score}{_leading_suffix}"

    # ── Technical domain ──────────────────────────────────────────────────────
    tech_score  = tac_score
    tech_flag   = (
        "NEUTRAL"    if 38 <= tech_score <= 62 else
        "MILD BULL"  if tech_score > 62 else
        "MILD BEAR"
    )
    tech_detail = f"Tactical {tech_score}/100 — {tac.get('label', '')}"

    # ── Options Flow domain ───────────────────────────────────────────────────
    opts_score  = of_score
    ua_sent     = st.session_state.get("_unusual_activity_sentiment") or {}
    ua_call_pct = ua_sent.get("call_pct", 50) if isinstance(ua_sent, dict) else 50
    opts_score  = int(opts_score * 0.6 + ua_call_pct * 0.4)
    opts_flag   = (
        "NEUTRAL"    if 38 <= opts_score <= 62 else
        "CALL-SKEW"  if opts_score > 62 else
        "PUT-SKEW"
    )
    opts_detail = f"Flow score {of_ctx.get('options_score', 50)}/100 | Unusual call%: {ua_call_pct:.0f}%"

    # ── Sentiment domain ──────────────────────────────────────────────────────
    aaii   = st.session_state.get("_aaii_sentiment") or {}
    bull_s = aaii.get("bull_pct", 50) if isinstance(aaii, dict) else 50
    sent_score = int(bull_s * 0.5 + fg_score * 0.5)
    sent_flag  = (
        "NEUTRAL"    if 38 <= sent_score <= 62 else
        "MILD BULL"  if sent_score > 62 else
        "MILD BEAR"
    )
    sent_detail = f"AAII bull: {bull_s:.0f}% | Fear&Greed: {fg_score}"

    # ── Event Risk domain ─────────────────────────────────────────────────────
    event_risk_score = 50
    event_unknowns   = []
    try:
        from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
        for fn, lbl in [(get_next_fomc, "FOMC"), (get_next_cpi, "CPI"), (get_next_nfp, "NFP")]:
            try:
                ev = fn()
                d  = ev.get("days_away", 99)
                if d <= 1:
                    event_risk_score = max(0, event_risk_score - 30)
                    event_unknowns.append(f"{lbl} {'today' if d == 0 else 'tomorrow'} — outcome unknown")
                elif d <= 5:
                    event_risk_score = max(0, event_risk_score - 15)
                    event_unknowns.append(f"{lbl} in {d}d — market may pre-position")
                elif d <= 14:
                    event_risk_score = max(5, event_risk_score - 5)
            except Exception:
                pass
    except Exception:
        pass
    event_flag = (
        "HIGH RISK"  if event_risk_score < 30 else
        "ELEVATED"   if event_risk_score < 50 else
        "MODERATE"
    )
    event_detail = f"Event uncertainty score: {event_risk_score}/100"

    # ── Directional lean (weighted, regime-aware) ──────────────────────────
    _lean_weights = {"macro": 0.30, "tech": 0.25, "opts": 0.20, "sent": 0.15, "event": 0.10}
    _lean_scores = {
        "macro": macro_score, "tech": tech_score, "opts": opts_score,
        "sent": sent_score, "event": event_risk_score,
    }
    domain_avg = sum(_lean_scores[k] * _lean_weights[k] for k in _lean_weights)

    # Fast leading score nudge: if leading diverges from macro by >10pts, blend toward it
    _fast_ls = int(rc.get("leading_score_fast") or rc.get("leading_score") or 50)
    if abs(_fast_ls - macro_score) > 10:
        domain_avg = domain_avg * 0.85 + _fast_ls * 0.15  # 15% nudge

    # Neutral zone: 47-53 → NEUTRAL
    if domain_avg > 53:
        lean = "BULLISH"
    elif domain_avg < 47:
        lean = "BEARISH"
    else:
        lean = "NEUTRAL"

    # HMM state guardrails
    try:
        from services.hmm_regime import load_current_hmm_state as _lean_hmm_load
        _lean_hmm = _lean_hmm_load()
        if _lean_hmm:
            _lean_state = _lean_hmm.state_label
            _lean_conf = getattr(_lean_hmm, "confidence", 0.5) or 0.5
            if _lean_state in ("Crisis", "Deep Stress") and lean == "BULLISH":
                lean = "NEUTRAL"
            elif _lean_state == "Bull" and _lean_conf > 0.7 and lean == "BEARISH":
                lean = "NEUTRAL"
    except Exception:
        pass

    # Lean percentage: distance from 50, scaled at 0.7x
    lean_pct = int(50 + abs(domain_avg - 50) * 0.7)

    # Reversal risk integration: degrade lean if reversal risk is elevated
    try:
        _tb_prox = st.session_state.get("_top_bottom_proximity") or {}
        _top_sc = _tb_prox.get("top_pct", 0)
        _bot_sc = _tb_prox.get("bottom_pct", 0)
        if lean == "BULLISH" and _top_sc > 40:
            lean_pct = max(51, lean_pct - int(_top_sc * 0.15))
        elif lean == "BEARISH" and _bot_sc > 40:
            lean_pct = max(51, lean_pct - int(_bot_sc * 0.15))
    except Exception:
        pass

    # Strong agreement bonus: if top 3 domains all agree (>65 or <35), expand range
    _core3 = [macro_score, tech_score, opts_score]
    if all(s > 65 for s in _core3) or all(s < 35 for s in _core3):
        lean_pct = min(85, lean_pct)
    else:
        lean_pct = min(75, lean_pct)
    lean_pct = max(51, lean_pct) if lean != "NEUTRAL" else 50

    # ── Overall uncertainty score (how uncertain, NOT how bullish) ────────────
    # Distance of each domain from 50 → lower distance = more uncertain
    distances      = [abs(s - 50) for s in [macro_score, tech_score, opts_score, sent_score, event_risk_score]]
    avg_distance   = sum(distances) / len(distances)
    uncertainty_sc = int(100 - avg_distance * 2)  # 100 = totally uncertain, 0 = crystal clear
    uncertainty_sc = min(100, max(0, uncertainty_sc))

    # ── Position size ─────────────────────────────────────────────────────────
    if uncertainty_sc >= 80:
        size_mult, size_label = 0.20, "20% SIZE"
    elif uncertainty_sc >= 65:
        size_mult, size_label = 0.30, "30% SIZE"
    elif uncertainty_sc >= 50:
        size_mult, size_label = 0.40, "40% SIZE"
    else:
        size_mult, size_label = 0.50, "50% SIZE"

    # ── Known unknowns ────────────────────────────────────────────────────────
    known_unknowns = list(event_unknowns)
    if macro_flag == "CONFLICTED":
        known_unknowns.append(f"Regime direction unresolved ({regime_lbl or 'no clear label'}) — next macro print could flip")
    if abs(_leading_div) > 7:
        _dir_word = "ahead of" if _leading_div > 0 else "below"
        known_unknowns.append(f"Leading indicators running {abs(_leading_div)} pts {_dir_word} composite ({_leading_lbl}) — lagging data may confirm soon")
    if opts_flag in ("PUT-SKEW", "CALL-SKEW") and tech_flag == "NEUTRAL":
        known_unknowns.append("Options crowd and price action disagree — one of them is early")
    if not known_unknowns:
        known_unknowns.append("No single dominant unknown — uncertainty is distributed across all 5 domains")
    known_unknowns = known_unknowns[:3]

    return {
        "lean":             lean,
        "lean_pct":         lean_pct,
        "uncertainty_score": uncertainty_sc,
        "size_mult":        size_mult,
        "size_label":       size_label,
        "domains": [
            {"name": "Macro",        "score": macro_score,  "flag": macro_flag,  "detail": macro_detail},
            {"name": "Technical",    "score": tech_score,   "flag": tech_flag,   "detail": tech_detail},
            {"name": "Options Flow", "score": opts_score,   "flag": opts_flag,   "detail": opts_detail},
            {"name": "Sentiment",    "score": sent_score,   "flag": sent_flag,   "detail": sent_detail},
            {"name": "Event Risk",   "score": event_risk_score, "flag": event_flag, "detail": event_detail},
        ],
        "known_unknowns":   known_unknowns,
    }


def _render_genuine_uncertainty_panel(profile: dict) -> None:
    """Render the anti-ambiguity-aversion Genuine Uncertainty intelligence panel.

    Always forces a lean — never outputs 'unclear'. Sizes the trade to match
    the measured uncertainty. Reads only from the pre-built profile dict.
    The lean header is rendered inline in the main verdict block (above the setups);
    this function renders the domain breakdown, known unknowns, and decision rules.
    """
    lean         = profile["lean"]
    lean_pct     = profile["lean_pct"]
    unc_score    = profile["uncertainty_score"]
    size_label   = profile["size_label"]
    domains      = profile["domains"]
    unknowns     = profile["known_unknowns"]

    lean_color = "#22c55e" if lean == "BULLISH" else ("#ef4444" if lean == "BEARISH" else "#f59e0b")

    unc_color = "#22c55e" if unc_score < 40 else ("#f59e0b" if unc_score < 65 else "#ef4444")

    flag_colors = {
        "MILD BULL":  "#22c55e", "CALL-SKEW":   "#22c55e",
        "MILD BEAR":  "#ef4444", "PUT-SKEW":     "#ef4444",
        "CONFLICTED": "#f59e0b", "NEUTRAL":      "#f59e0b",
        "HIGH RISK":  "#ef4444", "ELEVATED":     "#f97316",
        "MODERATE":   "#64748b",
    }

    # ── 5-domain breakdown ────────────────────────────────────────────────────
    domain_rows = ""
    for d in domains:
        d_score = d["score"]
        d_flag  = d["flag"]
        d_col   = flag_colors.get(d_flag, "#64748b")
        d_bg    = "#052e16" if d["score"] > 62 else ("#2d0a0a" if d["score"] < 38 else "#1a1200")
        bar_pct = d_score  # score IS 0-100, maps directly to bar width
        domain_rows += (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
            f'<span style="font-size:10px;font-weight:700;color:#94a3b8;">{d["name"].upper()}</span>'
            f'<span style="font-size:9px;font-weight:700;color:{d_col};background:{d_bg};'
            f'padding:1px 6px;border-radius:3px;">{d_flag}</span>'
            f'</div>'
            f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-bottom:2px;">'
            f'<div style="background:{d_col};width:{bar_pct}%;height:5px;border-radius:3px;'
            f'transition:width 0.3s;"></div>'
            f'</div>'
            f'<div style="font-size:9px;color:#475569;">{d["detail"]}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #1e293b;border-radius:6px;padding:12px 14px;margin:6px 0;">'
        f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;margin-bottom:10px;">UNCERTAINTY DOMAIN BREAKDOWN</div>'
        f'{domain_rows}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Known unknowns ────────────────────────────────────────────────────────
    with st.expander("🔍 Known Unknowns — what would flip the call", expanded=False):
        for i, ku in enumerate(unknowns, 1):
            st.markdown(
                f'<div style="background:#0f0f1a;border-left:3px solid #7c3aed;'
                f'padding:6px 12px;margin-bottom:6px;font-size:11px;color:#94a3b8;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'<span style="color:#7c3aed;font-weight:700;">[{i}]</span> {ku}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "These are the only factors that would flip the forced lean. "
            "Until they resolve, the lean stands. Don't manufacture uncertainty that isn't here."
        )

    # ── Decision rule strip ───────────────────────────────────────────────────
    stop_pct = "−3%" if unc_score >= 65 else "−4%"
    target   = "first resistance / prior swing high" if lean == "BULLISH" else ("prior swing low" if lean == "BEARISH" else "nearest support/resistance")
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #7c3aed44;border-radius:6px;'
        f'padding:10px 14px;margin-top:6px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">'
        f'<div><div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">IF LEAN CORRECT</div>'
        f'<div style="font-size:10px;color:#22c55e;margin-top:3px;">Take 50% off at {target}. Trail stop on rest.</div></div>'
        f'<div><div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">IF LEAN WRONG</div>'
        f'<div style="font-size:10px;color:#ef4444;margin-top:3px;">Hard stop at {stop_pct}. No averaging in. No exceptions.</div></div>'
        f'<div><div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">WHEN TO UPSIZE</div>'
        f'<div style="font-size:10px;color:#f59e0b;margin-top:3px;">Only after 2 of 5 domains shift to confirm. Not before.</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Crash Pattern Fingerprints ────────────────────────────────────────────────
_CRASH_FINGERPRINTS = [
    {
        "name": "GFC 2008",
        "lead_days": 481,
        "max_drawdown": -56.8,
        "conditions": [
            ("credit_hy_widening", lambda rc: (rc.get("score", 0) < -0.10), "Regime score turning negative"),
            ("vix_elevated", lambda _: (st.session_state.get("_vix_curve") or {}).get("vix_spot", 0) > 22, "VIX elevated above 22"),
            ("hmm_stress", lambda _: (st.session_state.get("_hmm_state") or {}).get("state_label", "") in ("Early Stress", "Stress", "Late Cycle", "Crisis"), "HMM detecting stress regime"),
            ("fear_high", lambda _: (st.session_state.get("_fear_composite") or {}).get("score", 50) < 30, "Fear composite in extreme fear"),
            ("tactical_weak", lambda _: (st.session_state.get("_tactical_context") or {}).get("tactical_score", 50) < 40, "Tactical score bearish"),
            ("stress_z_high", lambda _: (st.session_state.get("_stress_zscore") or {}).get("z", 0) > 1.0, "Stress z-score elevated >1σ"),
        ],
    },
    {
        "name": "COVID 2020",
        "lead_days": 33,
        "max_drawdown": -33.9,
        "conditions": [
            ("vix_spike", lambda _: (st.session_state.get("_vix_curve") or {}).get("vix_spot", 0) > 30, "VIX spiking above 30"),
            ("regime_risk_off", lambda rc: rc.get("score", 0) < -0.25, "Regime deep Risk-Off"),
            ("fear_extreme", lambda _: (st.session_state.get("_fear_composite") or {}).get("score", 50) < 20, "Fear composite extreme"),
            ("tactical_crash", lambda _: (st.session_state.get("_tactical_context") or {}).get("tactical_score", 50) < 30, "Tactical score collapsed"),
            ("options_bearish", lambda _: (st.session_state.get("_options_flow_context") or {}).get("options_score", 50) < 35, "Options flow bearish"),
        ],
    },
    {
        "name": "Rate Shock 2022",
        "lead_days": 282,
        "max_drawdown": -25.4,
        "conditions": [
            ("regime_negative", lambda rc: rc.get("score", 0) < -0.15, "Regime turning Risk-Off"),
            ("hmm_late_cycle", lambda _: (st.session_state.get("_hmm_state") or {}).get("state_label", "") in ("Late Cycle", "Stress", "Early Stress"), "HMM late cycle / stress"),
            ("fear_elevated", lambda _: (st.session_state.get("_fear_composite") or {}).get("score", 50) < 35, "Fear composite elevated"),
            ("whale_bearish", lambda _: (st.session_state.get("_whale_flow_score") or {}).get("bull_pct", 50) < 40, "Whale flow leaning bearish"),
            ("stress_rising", lambda _: (st.session_state.get("_stress_zscore") or {}).get("z", 0) > 0.5, "Stress z-score rising"),
        ],
    },
    {
        "name": "Carry Unwind 2024",
        "lead_days": 20,
        "max_drawdown": -8.5,
        "conditions": [
            ("vix_spike", lambda _: (st.session_state.get("_vix_curve") or {}).get("vix_spot", 0) > 25, "VIX spike above 25"),
            ("regime_flip", lambda rc: rc.get("score", 0) < -0.10, "Regime flipping negative"),
            ("gex_negative", lambda _: (st.session_state.get("_gex_dealer_context") or {}).get("composite", 0) < -0.3, "GEX deeply negative (dealers amplify)"),
            ("tactical_drop", lambda _: (st.session_state.get("_tactical_context") or {}).get("tactical_score", 50) < 40, "Tactical score dropped sharply"),
        ],
    },
]


def _check_crash_patterns(rc: dict) -> list[dict]:
    """Check current signals against historical crash fingerprints."""
    matches = []
    for fp in _CRASH_FINGERPRINTS:
        matched = []
        total = len(fp["conditions"])
        for cond_name, check_fn, desc in fp["conditions"]:
            try:
                if check_fn(rc):
                    matched.append(desc)
            except Exception:
                pass
        pct = len(matched) / total * 100 if total > 0 else 0
        if pct >= 50:
            matches.append({
                "name": fp["name"],
                "lead_days": fp["lead_days"],
                "max_drawdown": fp["max_drawdown"],
                "match_pct": round(pct),
                "matched": len(matched),
                "total": total,
                "details": matched,
            })
    matches.sort(key=lambda m: m["match_pct"], reverse=True)
    return matches


def _render_qir_dashboard() -> None:
    """Render the QIR Intelligence Dashboard.

    Always visible — greyed shell before QIR, activated with glow border after.
    Reads from session state only — no API calls.
    """
    _rc  = st.session_state.get("_regime_context")       or {}
    _tac = st.session_state.get("_tactical_context")     or {}
    _of  = st.session_state.get("_options_flow_context") or {}
    _er  = st.session_state.get("_qir_earnings_risk")    or []
    _populated = bool(_rc or _tac or _of)

    # ── Signal classification ─────────────────────────────────────────────
    _cls = _classify_signals(_rc, _tac, _of)
    _border_color = _cls["color"] if _populated else "#1e293b"
    _border_glow  = f"0 0 8px {_cls['color']}44" if _populated else "none"

    # ── Timing Stack column ───────────────────────────────────────────────
    _regime_score = _rc.get("score", 0)
    _regime_label = _rc.get("regime", "")
    _tac_score    = _tac.get("tactical_score", 50) if _tac else 50
    _of_score     = _of.get("options_score", 50)   if _of  else 50

    _t1 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">TIMING STACK</div>'
    if _rc:
        _r_bull = "Risk-On"  in _regime_label or _regime_score >  0.3
        _r_bear = "Risk-Off" in _regime_label or _regime_score < -0.3
        _rc_color = "#22c55e" if _r_bull else ("#ef4444" if _r_bear else "#f59e0b")
        _rc_arrow = "▲" if _r_bull else ("▼" if _r_bear else "◆")
        _t1 += (f'<div style="color:{_rc_color};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_rc_arrow} 📡 Regime: <span style="color:#e2e8f0;">{_regime_label[:28]}</span>'
                f'<span style="color:{_rc_color};font-size:10px;"> ({_regime_score:+.2f})</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ 📡 Regime — run QIR</div>'

    if _tac:
        _t_bull = _tac_score >= 65; _t_bear = _tac_score < 38
        _tc = "#22c55e" if _t_bull else ("#ef4444" if _t_bear else "#f59e0b")
        _ta = "▲" if _t_bull else ("▼" if _t_bear else "◆")
        _t1 += (f'<div style="color:{_tc};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_ta} ⚡ Tactical: <span style="color:#e2e8f0;">{_tac.get("label","")}</span>'
                f'<span style="color:{_tc};font-size:10px;"> ({_tac_score}/100)</span></div>')
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ ⚡ Tactical — run QIR</div>'

    if _of:
        _o_bull = _of_score >= 65; _o_bear = _of_score < 38
        _oc = "#22c55e" if _o_bull else ("#ef4444" if _o_bear else "#f59e0b")
        _oa = "▲" if _o_bull else ("▼" if _o_bear else "◆")
        _t1 += (f'<div style="color:{_oc};font-size:11px;padding:1px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'{_oa} 📊 Opt Flow: <span style="color:#e2e8f0;">{_of.get("label","")}</span>'
                f'<span style="color:{_oc};font-size:10px;"> ({_of_score}/100)</span></div>')
    elif _of.get("data_unavailable"):
        _t1 += '<div style="color:#475569;font-size:11px;padding:1px 0;">◆ 📊 Opt Flow — market closed</div>'
    else:
        _t1 += '<div style="color:#374151;font-size:11px;padding:1px 0;">◌ 📊 Opt Flow — run QIR</div>'

    # ── Macro Events column ───────────────────────────────────────────────
    _t2 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">MACRO EVENTS</div>'
    try:
        from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
        for _ev_label, _ev_fn in (("FOMC", get_next_fomc), ("CPI", get_next_cpi), ("NFP", get_next_nfp)):
            try:
                _ev = _ev_fn()
                _d  = _ev.get("days_away", 99)
                _dt = _ev.get("date", "")[:6]
                if   _d == 0:  _ec = "#ef4444"; _ds = "TODAY"
                elif _d == 1:  _ec = "#f97316"; _ds = "TMRW"
                elif _d <= 5:  _ec = "#f59e0b"; _ds = f"{_d}d"
                else:          _ec = "#475569"; _ds = f"{_d}d"
                _t2 += (f'<div style="font-size:11px;padding:1px 0;'
                        f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                        f'<span style="color:#64748b;">{_ev_label}</span>'
                        f'<span style="color:{_ec};"> {_ds}</span>'
                        f'<span style="color:#475569;font-size:10px;"> {_dt}</span></div>')
            except Exception:
                _t2 += f'<div style="color:#374151;font-size:11px;padding:1px 0;">{_ev_label} —</div>'
    except Exception:
        _t2 += '<div style="color:#374151;font-size:11px;">Macro events unavailable</div>'

    # ── Earnings Risk column ──────────────────────────────────────────────
    _t3 = '<div style="margin-bottom:2px;font-size:9px;font-weight:700;letter-spacing:0.1em;color:#475569;">EARNINGS RISK</div>'
    if _er:
        for _e in _er[:4]:
            _ed = _e["days_away"]
            _em = _e.get("expected_move_pct")
            _em_str = f" ±{_em:.1f}%" if _em else ""
            if   _ed <= 3: _ec2 = "#ef4444"; _eicon = "⚠"
            elif _ed <= 7: _ec2 = "#f59e0b"; _eicon = "⚠"
            else:          _ec2 = "#475569"; _eicon = "📅"
            _t3 += (f'<div style="font-size:11px;padding:1px 0;'
                    f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                    f'<span style="color:{_ec2};">{_eicon} {_e["ticker"]}</span>'
                    f'<span style="color:#64748b;"> {_ed}d</span>'
                    f'<span style="color:{_ec2};font-size:10px;">{_em_str}</span></div>')
    else:
        _t3 += '<div style="color:#374151;font-size:11px;">No earnings ≤21d</div>'

    # ── Verdict section ───────────────────────────────────────────────────
    _gu_profile = None  # computed inside if _populated when pattern == GENUINE_UNCERTAINTY
    _entry_rec_html = ""
    if _populated:
        _verdict_color = _cls["color"]
        _verdict_label = _cls["label"]
        _verdict_interp = _cls["interpretation"]
        _buy_tier   = _cls["buy_tier"]
        _short_tier = _cls["short_tier"]
        _instr_buy  = list(_cls["instruments_buy"])
        _instr_shrt = list(_cls["instruments_short"])
        _entry_buy  = _cls["entry_buy"]
        _entry_shrt = _cls["entry_short"]

        # Quadrant-specific instrument override
        _quadrant = _rc.get("quadrant", "")
        if _cls["pattern"] in ("BULLISH_CONFIRMATION", "PULLBACK_IN_UPTREND"):
            if _quadrant == "Goldilocks":
                _instr_buy = [("XLK / QQQ", "Tech leads in Goldilocks — low rates, strong growth"),
                              ("XLY", "Consumer discretionary benefits from spending confidence")] + _instr_buy
            elif _quadrant == "Overheating":
                _instr_buy = [("XLE", "Energy outperforms in overheating / commodity-driven growth"),
                              ("XLB", "Materials benefit from rising input prices")] + _instr_buy
            elif _quadrant == "Reflation":
                _instr_buy = [("XLE / XLB", "Commodity producers lead in reflation"),
                              ("XLF", "Financials benefit from steepening yield curve")] + _instr_buy
        elif _cls["pattern"] in ("BEARISH_CONFIRMATION", "LATE_CYCLE_SQUEEZE"):
            if _quadrant == "Stagflation":
                _instr_shrt = _instr_shrt + [("XLY Puts", "Consumer discretionary crushed by stagflation"),
                                              ("QQQ Puts", "Growth multiples compress fastest under stagflation")]
            elif _quadrant in ("Deflation", "Recession"):
                _instr_shrt = _instr_shrt + [("XLF Puts", "Credit losses mount in deflation/recession"),
                                              ("XLE Puts", "Demand collapses before supply adjusts")]

        # VIX note
        _vix_note = ""
        try:
            import re as _vre
            _vix_raw = (_tac.get("signals") or [{}])[0].get("Value", "") if _tac else ""
            _vm = _vre.search(r"(\d+\.?\d*)", str(_vix_raw))
            if _vm:
                _vix = float(_vm.group(1))
                if _vix > 28:
                    _vix_note = f"⚠ VIX {_vix:.0f} — premiums elevated. Prefer ETFs over options."
                elif _vix < 15:
                    _vix_note = f"ℹ VIX {_vix:.0f} — low vol. Options cheap: calls over ETFs for leverage efficiency."
        except Exception:
            pass

        # Earnings footer caveat (≤14 days)
        _earn_caveats = [
            f"⚠ {e['ticker']} earnings in {e['days_away']}d — size to survive "
            f"±{e['expected_move_pct']:.1f}% gap before the print"
            for e in _er
            if e["days_away"] <= 14 and e.get("expected_move_pct")
        ]

        # Pre-compute uncertainty profile for GENUINE_UNCERTAINTY to avoid calling twice
        _gu_profile = (
            _build_uncertainty_profile(_rc, _tac, _of)
            if _cls["pattern"] == "GENUINE_UNCERTAINTY" else None
        )

        # Store conviction score so _capture_market_context() can read it on log
        st.session_state["_qir_conviction_score"] = _cls.get("conviction_score")

        # ── Lean Tracker: log daily snapshot ─────────────────────────────────
        try:
            import json as _lt_json, os as _lt_os
            from datetime import date as _lt_date
            _lt_path = _lt_os.path.join(_lt_os.path.dirname(_lt_os.path.dirname(__file__)), "data", "lean_tracker.json")
            _lt_today = str(_lt_date.today())
            _lt_history = []
            if _lt_os.path.exists(_lt_path):
                with open(_lt_path, "r") as _lt_f:
                    _lt_history = _lt_json.load(_lt_f)
            if not any(h.get("date") == _lt_today for h in _lt_history):
                if _gu_profile:
                    _lt_lean = _gu_profile["lean"]
                    _lt_lean_pct = _gu_profile["lean_pct"]
                    _lt_domains = {d["name"].lower().replace(" ", "_"): d["score"] for d in _gu_profile["domains"]}
                else:
                    # Weighted lean — same logic as _build_uncertainty_profile
                    _lt_m = float(_rc.get("macro_score") or 50)
                    _lt_t = float((_tac or {}).get("tactical_score") or 50)
                    _lt_o = float((_of or {}).get("options_score") or 50)
                    _lt_s = 50.0  # sentiment unavailable outside GU
                    _lt_e = 50.0  # event risk unavailable outside GU
                    _lt_wavg = _lt_m * 0.30 + _lt_t * 0.25 + _lt_o * 0.20 + _lt_s * 0.15 + _lt_e * 0.10
                    # Fast leading nudge
                    _lt_fast = float(_rc.get("leading_score_fast") or _rc.get("leading_score") or 50)
                    if abs(_lt_fast - _lt_m) > 10:
                        _lt_wavg = _lt_wavg * 0.85 + _lt_fast * 0.15
                    # Neutral zone
                    if _lt_wavg > 53:
                        _lt_lean = "BULLISH"
                    elif _lt_wavg < 47:
                        _lt_lean = "BEARISH"
                    else:
                        _lt_lean = "NEUTRAL"
                    _lt_lean_pct = int(50 + abs(_lt_wavg - 50) * 0.7)
                    # HMM guardrails
                    try:
                        from services.hmm_regime import load_current_hmm_state as _cal_hmm_load
                        _cal_hmm = _cal_hmm_load()
                        if _cal_hmm:
                            _cal_state = _cal_hmm.state_label
                            _cal_conf = getattr(_cal_hmm, "confidence", 0.5) or 0.5
                            if _cal_state in ("Crisis", "Deep Stress") and _lt_lean == "BULLISH":
                                _lt_lean = "NEUTRAL"
                                _lt_lean_pct = 50
                            elif _cal_state == "Bull" and _cal_conf > 0.7 and _lt_lean == "BEARISH":
                                _lt_lean = "NEUTRAL"
                                _lt_lean_pct = 50
                    except Exception:
                        pass
                    # Strong agreement bonus
                    _lt_core3 = [_lt_m, _lt_t, _lt_o]
                    if all(s > 65 for s in _lt_core3) or all(s < 35 for s in _lt_core3):
                        _lt_lean_pct = min(85, _lt_lean_pct)
                    else:
                        _lt_lean_pct = min(75, _lt_lean_pct)
                    _lt_lean_pct = max(51, _lt_lean_pct) if _lt_lean != "NEUTRAL" else 50
                    _lt_domains = {
                        "macro": _lt_m,
                        "technical": _lt_t,
                        "options_flow": _lt_o,
                        "sentiment": _lt_s,
                        "event_risk": _lt_e,
                    }
                _lt_hmm_state = ""
                _lt_hmm_entropy = 0.0
                try:
                    from services.hmm_regime import load_current_hmm_state as _lt_hmm
                    _lt_hs = _lt_hmm()
                    if _lt_hs:
                        _lt_hmm_state = _lt_hs.state_label
                        _lt_hmm_entropy = getattr(_lt_hs, "entropy", 0.0) or 0.0
                except Exception:
                    pass
                _lt_gex = (st.session_state.get("_gex_dealer_context") or {}).get("composite", 0.0)
                _lt_entry = {
                    "date": _lt_today,
                    "lean": _lt_lean,
                    "lean_pct": _lt_lean_pct,
                    "domain_avg": round(sum(_lt_domains.values()) / max(len(_lt_domains), 1), 1),
                    "macro_score": _lt_domains.get("macro", 50),
                    "tech_score": _lt_domains.get("technical", 50),
                    "opts_score": _lt_domains.get("options_flow", 50),
                    "sent_score": _lt_domains.get("sentiment", 50),
                    "event_score": _lt_domains.get("event_risk", 50),
                    "regime": _rc.get("regime", ""),
                    "hmm_state": _lt_hmm_state,
                    "hmm_entropy": round(_lt_hmm_entropy, 4),
                    "gex_composite": round(_lt_gex or 0.0, 3),
                    "conviction_score": _cls.get("conviction_score"),
                    "pattern": _cls.get("pattern", ""),
                    "fwd_5d_spy_return": None,
                    "fwd_20d_spy_return": None,
                }
                _lt_history.append(_lt_entry)
                import pandas as _lt_pd
                if len(_lt_history) >= 6:
                    _lt_bf5 = _lt_history[-6]
                    if _lt_bf5.get("fwd_5d_spy_return") is None:
                        try:
                            import yfinance as _lt_yf
                            _lt_spy = _lt_yf.download("SPY", start=_lt_bf5["date"], end=_lt_today, progress=False, auto_adjust=True)
                            if _lt_spy is not None and len(_lt_spy) >= 5:
                                _lt_c = _lt_spy["Close"]
                                if isinstance(_lt_c, _lt_pd.DataFrame):
                                    _lt_c = _lt_c.iloc[:, 0]
                                _lt_bf5["fwd_5d_spy_return"] = round(float((_lt_c.iloc[5] / _lt_c.iloc[0] - 1) * 100), 2)
                        except Exception:
                            pass
                if len(_lt_history) >= 22:
                    _lt_bf20 = _lt_history[-22]
                    if _lt_bf20.get("fwd_20d_spy_return") is None:
                        try:
                            import yfinance as _lt_yf2
                            _lt_spy2 = _lt_yf2.download("SPY", start=_lt_bf20["date"], end=_lt_today, progress=False, auto_adjust=True)
                            if _lt_spy2 is not None and len(_lt_spy2) >= 20:
                                _lt_c2 = _lt_spy2["Close"]
                                if isinstance(_lt_c2, _lt_pd.DataFrame):
                                    _lt_c2 = _lt_c2.iloc[:, 0]
                                _lt_bf20["fwd_20d_spy_return"] = round(float((_lt_c2.iloc[20] / _lt_c2.iloc[0] - 1) * 100), 2)
                        except Exception:
                            pass
                with open(_lt_path, "w") as _lt_fw:
                    _lt_json.dump(_lt_history, _lt_fw, indent=2)
        except Exception:
            pass

        # Build verdict HTML helpers
        def _tier_badge(tier):
            _tc = {"STRONG": "#22c55e", "MODERATE": "#f59e0b",
                   "SELECTIVE": "#f97316", "NOT A BUYING ENV": "#ef4444",
                   "NOT A SHORTING ENV": "#22c55e"}.get(tier, "#64748b")
            return (f'<span style="background:{_tc};color:black;font-weight:800;'
                    f'font-size:9px;padding:1px 7px;border-radius:3px;">{tier}</span>')

        def _instruments_html(instruments):
            if not instruments:
                return ""
            rows = "".join(
                f'<div style="padding:2px 0;border-bottom:1px solid #1e293b;">'
                f'<span style="color:#f1f5f9;font-weight:700;font-size:10px;">{t}</span>'
                f'<span style="color:#94a3b8;font-size:10px;"> — {d}</span></div>'
                for t, d in instruments[:4]
            )
            return (f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin:6px 0 2px;">INSTRUMENTS</div>'
                    f'<div style="margin-bottom:6px;">{rows}</div>')

        def _entry_html(entry_text, label="ENTRY / RISK RULES"):
            rows = "".join(
                f'<div style="color:#94a3b8;font-size:10px;padding:1px 0;">· {r.strip()}</div>'
                for r in entry_text.strip().split("\n") if r.strip()
            )
            return (f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin:6px 0 2px;">{label}</div>{rows}')

        _conviction_score  = _cls.get("conviction_score")
        _conviction_size_label = _cls.get("conviction_size_label")
        _leading_warning   = _cls.get("leading_warning")
        _conviction_comps  = _cls.get("conviction_components") or {}

        # Entry signal recommendation (leading vs lagging synthesis)
        _leading_s = int(_rc.get("leading_score") or 50)
        _macro_s   = int(_rc.get("macro_score") or 50)
        _div_pts   = int(_rc.get("leading_divergence") or 0)
        _div_label = _rc.get("leading_label") or "Aligned"
        _tac_s     = int(_tac.get("tactical_score", 50)) if _tac else 50
        _opts_s    = int(_of.get("options_score", 50))   if _of  else 50
        # Velocity: prefer 5-day macro trend (wider signal); fall back to 1-day delta
        _vel_for_entry = int(_rc.get("score_5d_trend") or _rc.get("velocity") or 0)
        _entry_rec = _classify_entry_recommendation(
            _leading_s, _macro_s, _tac_s, _opts_s, _div_label, _div_pts,
            velocity_delta=_vel_for_entry,
        )

        _verdict_html = (
            f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
            f'<div style="font-size:13px;font-weight:800;color:{_verdict_color};'
            f'letter-spacing:0.04em;margin-bottom:4px;">{_verdict_label}</div>'
            f'<div style="color:#94a3b8;font-size:11px;margin-bottom:10px;">{_verdict_interp}</div>'
        )

        # Conviction and leading warning are now shown in the dedicated MEDIUM conviction card.
        # Keep only the VIX note and leading divergence warning here for pattern context.

        # Leading divergence warning
        if _leading_warning:
            _verdict_html += (
                f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                f'padding:6px 10px;font-size:10px;color:#f59e0b;margin-bottom:8px;">'
                f'⚠ {_leading_warning}</div>'
            )

        if _vix_note:
            _verdict_html += (
                f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                f'padding:5px 10px;font-size:10px;color:#f59e0b;margin-bottom:8px;">{_vix_note}</div>'
            )

        # Tactical Kelly pcts — computed later in Kelly block; init for badge guards
        _tac_long_kelly_pct = None
        _tac_short_kelly_pct = None

        # Pre-compute Kelly for GENUINE_UNCERTAINTY tactical badges
        if _cls.get("pattern") == "GENUINE_UNCERTAINTY" and _gu_profile:
            try:
                from services.portfolio_sizing import compute_triple_kelly as _pre_triple_kelly
                try:
                    from services.hmm_regime import load_current_hmm_state as _pre_hmm_load
                    _pre_hmm_label = getattr(_pre_hmm_load(), "state_label", None)
                except Exception:
                    _pre_hmm_label = None
                _pre_tkly = _pre_triple_kelly(
                    fear_composite=st.session_state.get("_fear_composite") or {},
                    regime_ctx=st.session_state.get("_regime_context") or {},
                    hmm_state_label=_pre_hmm_label,
                    forced_lean=_gu_profile.get("lean", "BEARISH"),
                    lean_pct=float(_gu_profile.get("lean_pct", 53)),
                    uncertainty_score=int(_gu_profile.get("uncertainty_score", 60)),
                    macro_score=int((_rc or {}).get("macro_score") or 50),
                    leading_score=int((_rc or {}).get("leading_score") or 50),
                )
                _tac_short_kelly_pct = _pre_tkly["tactical_short"]["half_pct"]
                _tac_long_kelly_pct  = _pre_tkly["tactical_long"]["half_pct"]
            except Exception:
                pass

        # Tactical Long Kelly badge — only for GENUINE_UNCERTAINTY bullish lean
        _tac_long_badge = ""
        if (
            _cls.get("pattern") == "GENUINE_UNCERTAINTY"
            and _gu_profile
            and _tac_long_kelly_pct is not None
        ):
            _tlk_col = "#22c55e" if _tac_long_kelly_pct >= 4 else "#f59e0b" if _tac_long_kelly_pct >= 2 else "#94a3b8"
            _tlk_label = "TACTICAL LONG KELLY" if _gu_profile.get("lean") == "BULLISH" else "SCALP KELLY"
            _tlk_note = "days/weeks" if _gu_profile.get("lean") == "BULLISH" else "1-3 days · uncertainty-penalised"
            _tac_long_badge = (
                f'<div style="display:flex;align-items:baseline;gap:6px;'
                f'background:#0a1a0a;border:1px solid #22c55e33;border-radius:4px;'
                f'padding:5px 8px;margin:5px 0 6px;">'
                f'<span style="font-size:8px;color:#22c55e;font-weight:700;'
                f'letter-spacing:0.08em;">{_tlk_label}</span>'
                f'<span style="font-size:22px;font-weight:900;color:{_tlk_col};">'
                f'{_tac_long_kelly_pct}%</span>'
                f'<span style="font-size:9px;color:#475569;">of portfolio · {_tlk_note}</span>'
                f'</div>'
            )
        _buy_html = (
            f'<div style="padding:8px;background:#0a1628;border:1px solid {_cls["color"]}22;border-radius:5px;">'
            f'<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">BUY SETUP</div>'
            f'{_tier_badge(_buy_tier)}'
            f'{_tac_long_badge}'
            f'{_instruments_html(_instr_buy)}'
            f'{_entry_html(_entry_buy)}'
            f'</div>'
        )
        # Tactical Short Kelly badge — only for GENUINE_UNCERTAINTY bearish lean
        _tac_short_badge = ""
        if (
            _cls.get("pattern") == "GENUINE_UNCERTAINTY"
            and _gu_profile
            and _gu_profile.get("lean") == "BEARISH"
            and _tac_short_kelly_pct is not None
        ):
            _tsk_col = "#22c55e" if _tac_short_kelly_pct >= 6 else "#f59e0b" if _tac_short_kelly_pct >= 3 else "#ef4444"
            _tac_short_badge = (
                f'<div style="display:flex;align-items:baseline;gap:6px;'
                f'background:#0a0a1a;border:1px solid #ef444433;border-radius:4px;'
                f'padding:5px 8px;margin:5px 0 6px;">'
                f'<span style="font-size:8px;color:#ef4444;font-weight:700;'
                f'letter-spacing:0.08em;">TACTICAL SHORT KELLY</span>'
                f'<span style="font-size:22px;font-weight:900;color:{_tsk_col};">'
                f'{_tac_short_kelly_pct}%</span>'
                f'<span style="font-size:9px;color:#475569;">of portfolio · days/weeks</span>'
                f'</div>'
            )
        _short_html = (
            f'<div style="padding:8px;background:#160a0a;border:1px solid {_cls["color"]}22;border-radius:5px;">'
            f'<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">SHORT SETUP</div>'
            f'{_tier_badge(_short_tier)}'
            f'{_tac_short_badge}'
            f'{_instruments_html(_instr_shrt)}'
            f'{_entry_html(_entry_shrt)}'
            f'</div>'
        )

        # GENUINE_UNCERTAINTY: grey out the opposing side — lean-aligned panel is active, other is dimmed.
        # NEUTRAL lean: both sides get mild dimming with equal weight.
        if _cls["pattern"] == "GENUINE_UNCERTAINTY" and _gu_profile:
            _gu_lean     = _gu_profile["lean"]
            _gu_lean_pct = _gu_profile["lean_pct"]
            _gu_unc      = _gu_profile["uncertainty_score"]
            _gu_size_lbl = _gu_profile["size_label"]
            _gu_lc  = "#22c55e" if _gu_lean == "BULLISH" else ("#ef4444" if _gu_lean == "BEARISH" else "#f59e0b")
            _gu_arr = "▲" if _gu_lean == "BULLISH" else ("▼" if _gu_lean == "BEARISH" else "◆")
            _gu_unc_col = "#22c55e" if _gu_unc < 40 else ("#f59e0b" if _gu_unc < 65 else "#ef4444")
            _verdict_html += (
                f'<div style="background:#0d1117;border:1px solid {_gu_lc}33;border-left:3px solid {_gu_lc};'
                f'border-radius:5px;padding:8px 12px;margin:6px 0;display:flex;align-items:center;gap:12px;">'
                f'<div>'
                f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:2px;">'
                f'{"DIRECTIONAL LEAN" if _gu_lean != "NEUTRAL" else "NO DIRECTIONAL EDGE"}</div>'
                f'<div style="font-size:15px;font-weight:800;color:{_gu_lc};letter-spacing:0.04em;">'
                f'{_gu_arr} {_gu_lean} {_gu_lean_pct}%</div>'
                f'</div>'
                f'<div style="flex:1;border-left:1px solid #1e293b;padding-left:12px;">'
                f'<div style="font-size:8px;color:#64748b;font-weight:700;margin-bottom:2px;">ANTI-AMBIGUITY OVERRIDE</div>'
                f'<div style="font-size:9px;color:#475569;line-height:1.5;">'
                f'Uncertainty is quantified — not an excuse for inaction. '
                f'Act at <span style="color:#94a3b8;font-weight:700;">{_gu_size_lbl}</span> of normal. Wrong? Stop early.'
                f'</div>'
                f'</div>'
                f'<div style="text-align:right;min-width:48px;">'
                f'<div style="font-size:8px;color:#334155;font-weight:700;letter-spacing:0.08em;">UNC</div>'
                f'<div style="font-size:14px;font-weight:800;color:{_gu_unc_col};font-family:monospace;">{_gu_unc}</div>'
                f'<div style="font-size:8px;color:#334155;">/100</div>'
                f'</div>'
                f'</div>'
            )

            def _dim(html: str) -> str:
                return html.replace(
                    'background:#0a1628', 'background:#0d1117'
                ).replace(
                    'background:#160a0a', 'background:#0d1117'
                )

            if _gu_lean == "NEUTRAL":
                # Both sides equally viable — mild dim, no "NOT RECOMMENDED"
                _buy_display = (
                    f'<div style="opacity:0.70;">'
                    f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin-bottom:3px;">LEAN NEUTRAL — size down both sides</div>'
                    f'{_buy_html}</div>'
                )
                _shrt_display = (
                    f'<div style="opacity:0.70;">'
                    f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin-bottom:3px;">LEAN NEUTRAL — size down both sides</div>'
                    f'{_short_html}</div>'
                )
            else:
                _dimmed_buy   = _gu_lean != "BULLISH"
                _dimmed_shrt  = _gu_lean != "BEARISH"
                _buy_display  = _buy_html  if not _dimmed_buy  else (
                    f'<div style="opacity:0.35;filter:grayscale(0.7);">'
                    f'<div style="font-size:9px;color:#374151;font-weight:700;'
                    f'letter-spacing:0.06em;margin-bottom:3px;">NOT RECOMMENDED — lean is bearish</div>'
                    f'{_dim(_buy_html)}</div>'
                )
                _shrt_display = _short_html if not _dimmed_shrt else (
                    f'<div style="opacity:0.35;filter:grayscale(0.7);">'
                    f'<div style="font-size:9px;color:#374151;font-weight:700;'
                    f'letter-spacing:0.06em;margin-bottom:3px;">NOT RECOMMENDED — lean is bullish</div>'
                    f'{_dim(_short_html)}</div>'
                )
            _verdict_html += (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">'
                f'{_buy_display}{_shrt_display}</div>'
            )
        else:
            _verdict_html += (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">'
                f'{_buy_html}{_short_html}</div>'
            )

        if _earn_caveats:
            _verdict_html += "".join(
                f'<div style="background:#1a0a00;border-left:3px solid #ef4444;'
                f'padding:5px 10px;font-size:10px;color:#ef4444;margin-top:4px;">{c}</div>'
                for c in _earn_caveats
            )

        # ── Zone 2 blocks (populated below) ──────────────────────────────────
        _conviction_block = ""
        _velocity_block   = ""
        _signal_breakdown_block = ""
        _top_bottom_block = ""
        _hmm_block        = ""
        _gex_block        = ""
        _lean_card        = ""
        _fast_setups_html = ""

        # Conviction block — full card with component breakdown
        if _conviction_score is not None:
            _cv_color2 = "#22c55e" if _conviction_score >= 75 else (
                "#f59e0b" if _conviction_score >= 55 else (
                "#f97316" if _conviction_score >= 40 else "#ef4444"
            ))
            # Component rows (sorted best → worst absolute contribution)
            _cv_rows_data = [
                ("Regime",    _conviction_comps.get("regime",   0), 40,  "macro z-score strength"),
                ("Tactical",  _conviction_comps.get("tactical", 0), 30,  "how far from neutral 50"),
                ("Options",   _conviction_comps.get("options",  0), 20,  "flow conviction"),
                ("Leading",   _conviction_comps.get("leading",  0), 10,  "fast signal divergence"),
                ("Whale",     _conviction_comps.get("whale",    0), 10,  "institutional flow"),
                ("Rate Path", _conviction_comps.get("rate",     0), 8,   "Fed path alignment"),
                ("Velocity",  _conviction_comps.get("velocity", 0), 8,   "momentum direction"),
            ]
            _cv_comp_rows = ""
            for _cvr_name, _cvr_val, _cvr_max, _cvr_tip in _cv_rows_data:
                _cvr_col = "#22c55e" if _cvr_val > 0 else ("#ef4444" if _cvr_val < 0 else "#334155")
                _cvr_bar_pct = int(min(abs(_cvr_val) / max(_cvr_max, 1) * 100, 100))
                _cvr_sign = f"+{_cvr_val:.1f}" if _cvr_val >= 0 else f"{_cvr_val:.1f}"
                _cv_comp_rows += (
                    f'<div style="display:grid;grid-template-columns:60px 1fr 36px;'
                    f'align-items:center;gap:5px;margin-bottom:3px;">'
                    f'<div style="font-size:8px;color:#475569;text-align:right;">{_cvr_name}</div>'
                    f'<div style="background:#1e293b;border-radius:2px;height:6px;position:relative;">'
                    f'<div style="background:{_cvr_col};width:{_cvr_bar_pct}%;height:100%;'
                    f'border-radius:2px;{"float:right;" if _cvr_val < 0 else ""}"></div>'
                    f'</div>'
                    f'<div style="font-size:8px;color:{_cvr_col};font-weight:700;'
                    f'font-family:monospace;text-align:right;">{_cvr_sign}</div>'
                    f'</div>'
                )
            # Multiplier pills
            _fear_mult_v  = _conviction_comps.get("fear_mult", 1.0)
            _hmm_mult_v   = _conviction_comps.get("hmm_mult",  1.0)
            _fear_pill_c  = "#22c55e" if _fear_mult_v >= 0.95 else ("#f59e0b" if _fear_mult_v >= 0.80 else "#ef4444")
            _hmm_pill_c   = "#22c55e" if _hmm_mult_v >= 1.0 else ("#f59e0b" if _hmm_mult_v >= 0.85 else "#ef4444")
            _warn_row = ""
            if _leading_warning:
                _warn_row = (
                    f'<div style="background:#1a1200;border-left:3px solid #f59e0b;'
                    f'padding:5px 8px;font-size:9px;color:#f59e0b;margin-top:5px;'
                    f'border-radius:0 3px 3px 0;line-height:1.4;">⚠ {_leading_warning}</div>'
                )
            _conviction_block = (
                f'<div style="background:#0f172a;border:1px solid {_cv_color2}44;'
                f'border-left:3px solid {_cv_color2};border-radius:6px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                # Header row
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">'
                f'<div>'
                f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.12em;margin-bottom:3px;">'
                f'CONVICTION</div>'
                f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:28px;font-weight:900;color:{_cv_color2};line-height:1;">'
                f'{_conviction_score}</div>'
                f'<div style="font-size:8px;color:#64748b;">/100</div>'
                f'</div>'
                f'</div>'
                # Master bar
                f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-bottom:8px;">'
                f'<div style="background:{_cv_color2};width:{_conviction_score}%;height:5px;border-radius:3px;"></div>'
                f'</div>'
                # Size label
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<div style="background:{_cv_color2}22;border:1px solid {_cv_color2}55;'
                f'border-radius:4px;padding:3px 10px;font-size:10px;font-weight:800;color:{_cv_color2};">'
                f'→ {_conviction_size_label}</div>'
                f'<span style="font-size:9px;color:#334155;">recommended max position size</span>'
                f'</div>'
                # Component breakdown
                f'<div style="border-top:1px solid #1e293b;padding-top:7px;margin-bottom:4px;">'
                f'<div style="font-size:7px;color:#334155;font-weight:700;letter-spacing:0.1em;margin-bottom:5px;">'
                f'COMPONENT BREAKDOWN  · +max / −drag</div>'
                f'{_cv_comp_rows}'
                f'</div>'
                # Multiplier pills
                f'<div style="display:flex;gap:6px;margin-top:5px;">'
                f'<div style="background:#0a0f1a;border:1px solid #1e293b;border-radius:3px;'
                f'padding:2px 7px;font-size:8px;color:{_fear_pill_c};">'
                f'fear ×{_fear_mult_v:.2f}</div>'
                f'<div style="background:#0a0f1a;border:1px solid #1e293b;border-radius:3px;'
                f'padding:2px 7px;font-size:8px;color:{_hmm_pill_c};">'
                f'HMM ×{_hmm_mult_v:.2f}</div>'
                f'</div>'
                f'{_warn_row}'
                f'</div>'
            )

        # ── Regime Velocity strip (horizontal) ───────────────────────────────
        import json as _vjson, os as _vos
        try:
            _vpath = _vos.path.join(_vos.path.dirname(_vos.path.dirname(__file__)), "data", "tactical_score_history.json")
            with open(_vpath) as _vf:
                _vhist = _vjson.load(_vf)
            if _vhist and len(_vhist) >= 6:
                _v_now = float(_rc.get("macro_score") or 50)
                _v_old = float(_vhist[-6].get("score", 50))
                _v_delta = round(_v_now - _v_old, 1)
                _v_abs = abs(_v_delta)
                _v_color = "#22c55e" if _v_delta > 3 else ("#ef4444" if _v_delta < -3 else "#f59e0b")
                _v_arrow = "▲" if _v_delta > 3 else ("▼" if _v_delta < -3 else "►")
                _v_label = ("ACCELERATING" if _v_abs > 15 else
                            "FLIPPING" if _v_abs > 8 else
                            "DRIFTING" if _v_abs > 3 else "STABLE")
                _v_bar_w = min(100, int(_v_abs * 3))
                _v_note = {
                    "ACCELERATING": "major shift — trade with conviction",
                    "FLIPPING": "transition zone — watch for confirmation",
                    "DRIFTING": "gradual shift — stay alert",
                    "STABLE": "regime holding steady",
                }[_v_label]

                # ── Previous velocity + acceleration ────────────────────────
                _v_accel_html = ""
                _v_spark_html = ""
                if len(_vhist) >= 12:
                    _v_prev_delta = round(
                        float(_vhist[-7].get("score", 50)) - float(_vhist[-12].get("score", 50)), 1
                    )
                    _v_accel = round(_v_delta - _v_prev_delta, 1)
                    _v_ac_color = "#22c55e" if _v_accel > 2 else ("#ef4444" if _v_accel < -2 else "#64748b")
                    _v_ac_arrow = "↑" if _v_accel > 2 else ("↓" if _v_accel < -2 else "~")
                    _v_ac_label = ("SPEEDING UP" if _v_accel > 5 else
                                   "SLOWING DOWN" if _v_accel < -5 else
                                   "ACCELERATING" if _v_accel > 2 else
                                   "DECELERATING" if _v_accel < -2 else "STEADY")
                    _v_accel_html = (
                        f'<span style="font-size:8px;color:#475569;margin-left:10px;">'
                        f'prev: <span style="color:#64748b;">{_v_prev_delta:+.1f}</span></span>'
                        f'<span style="font-size:8px;color:{_v_ac_color};font-weight:700;margin-left:6px;">'
                        f'{_v_ac_arrow} {_v_accel:+.1f} · {_v_ac_label}</span>'
                    )

                # Sparkline
                if len(_vhist) >= 12:
                    _SPARK_CHARS = " ▁▂▃▄▅▆▇█"
                    _spark_vals = []
                    for _si in range(min(8, len(_vhist) // 6)):
                        _s_end = len(_vhist) - 1 - _si * 6
                        _s_start = _s_end - 5
                        if _s_start >= 0:
                            _sv = float(_vhist[_s_end].get("score", 50)) - float(_vhist[_s_start].get("score", 50))
                            _spark_vals.insert(0, _sv)
                    _spark_vals.append(_v_delta)
                    if _spark_vals:
                        _sp_max = max(abs(v) for v in _spark_vals) or 1
                        _sp_spans = []
                        for _sv in _spark_vals:
                            _sp_idx = min(8, int(abs(_sv) / _sp_max * 8))
                            _sp_char = _SPARK_CHARS[_sp_idx]
                            _sp_col = "#22c55e" if _sv > 2 else ("#ef4444" if _sv < -2 else "#64748b")
                            _sp_spans.append(f'<span style="color:{_sp_col};">{_sp_char}</span>')
                        _v_spark_html = (
                            f'<span style="font-family:monospace;font-size:12px;letter-spacing:2px;'
                            f'margin-left:10px;" title="Weekly velocity (left=oldest)">'
                            f'{"".join(_sp_spans)}</span>'
                        )

                # ── Conviction pill ──────────────────────────────────────────
                _v_conv_html = ""
                if _conviction_score is not None:
                    _cv_c2 = ("#22c55e" if _conviction_score >= 75 else
                               "#f59e0b" if _conviction_score >= 55 else
                               "#f97316" if _conviction_score >= 40 else "#ef4444")
                    _v_conv_html = (
                        f'<span style="background:#1e293b;border-radius:4px;padding:1px 7px;'
                        f'font-size:9px;color:{_cv_c2};font-weight:700;margin-left:10px;">'
                        f'CONVICTION {_conviction_score}/100 · {_conviction_size_label}</span>'
                    )

                # ── Horizontal strip ─────────────────────────────────────────
                _velocity_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:6px 12px;margin-bottom:8px;display:flex;align-items:center;'
                    f'flex-wrap:wrap;gap:4px 0;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-right:8px;">REGIME VELOCITY</span>'
                    f'<span style="font-size:13px;font-weight:800;color:{_v_color};">'
                    f'{_v_arrow} {_v_delta:+.1f}/wk</span>'
                    f'<span style="font-size:9px;color:{_v_color};margin-left:5px;">'
                    f'· {_v_label}</span>'
                    f'<span style="font-size:8px;color:#475569;margin-left:8px;">({_v_note})</span>'
                    f'{_v_accel_html}'
                    f'{_v_spark_html}'
                    f'{_v_conv_html}'
                    f'<div style="width:100%;background:#1e293b;border-radius:2px;height:2px;margin-top:4px;">'
                    f'<div style="background:{_v_color};width:{_v_bar_w}%;height:2px;border-radius:2px;"></div>'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

        # Fallback: if file-based velocity failed, build from regime context (always available)
        if not _velocity_block:
            _rc_vel = _rc.get("score_5d_trend") if _rc.get("score_5d_trend") is not None else _rc.get("velocity")
            if _rc_vel is not None:
                _v_delta = float(_rc_vel)
                _v_abs = abs(_v_delta)
                _v_color = "#22c55e" if _v_delta > 3 else ("#ef4444" if _v_delta < -3 else "#f59e0b")
                _v_arrow = "▲" if _v_delta > 3 else ("▼" if _v_delta < -3 else "►")
                _v_label = ("ACCELERATING" if _v_abs > 15 else
                            "FLIPPING" if _v_abs > 8 else
                            "DRIFTING" if _v_abs > 3 else "STABLE")
                _v_bar_w = min(100, int(_v_abs * 3))
                _v_src = "5d" if _rc.get("score_5d_trend") is not None else "1d"
                _v_conv_html = ""
                if _conviction_score is not None:
                    _cv_c2 = ("#22c55e" if _conviction_score >= 75 else
                               "#f59e0b" if _conviction_score >= 55 else
                               "#f97316" if _conviction_score >= 40 else "#ef4444")
                    _v_conv_html = (
                        f'<span style="background:#1e293b;border-radius:4px;padding:1px 7px;'
                        f'font-size:9px;color:{_cv_c2};font-weight:700;margin-left:10px;">'
                        f'CONVICTION {_conviction_score}/100 · {_conviction_size_label}</span>'
                    )
                _velocity_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:6px 12px;margin-bottom:8px;display:flex;align-items:center;'
                    f'flex-wrap:wrap;gap:4px 0;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-right:8px;">REGIME VELOCITY</span>'
                    f'<span style="font-size:13px;font-weight:800;color:{_v_color};">'
                    f'{_v_arrow} {_v_delta:+.0f}pt</span>'
                    f'<span style="font-size:9px;color:{_v_color};margin-left:5px;">· {_v_label}</span>'
                    f'<span style="font-size:8px;color:#475569;margin-left:8px;">({_v_src} macro trend)</span>'
                    f'{_v_conv_html}'
                    f'<div style="width:100%;background:#1e293b;border-radius:2px;height:2px;margin-top:4px;">'
                    f'<div style="background:{_v_color};width:{_v_bar_w}%;height:2px;border-radius:2px;"></div>'
                    f'</div>'
                    f'</div>'
                )

        # Final fallback: if velocity is unavailable but conviction exists, show conviction-only strip
        if not _velocity_block and _conviction_score is not None:
            _cv_c2 = ("#22c55e" if _conviction_score >= 75 else
                       "#f59e0b" if _conviction_score >= 55 else
                       "#f97316" if _conviction_score >= 40 else "#ef4444")
            _velocity_block = (
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:6px 12px;margin-bottom:8px;display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">REGIME VELOCITY</span>'
                f'<span style="font-size:8px;color:#334155;">— insufficient history</span>'
                f'<span style="background:#1e293b;border-radius:4px;padding:1px 7px;'
                f'font-size:9px;color:{_cv_c2};font-weight:700;">'
                f'CONVICTION {_conviction_score}/100 · {_conviction_size_label}</span>'
                f'</div>'
            )

        # ── Signal Breakdown card ────────────────────────────────────────────
        _raw_sigs = st.session_state.get("_regime_raw_signals") or {}
        _meta_keys = {"macro_score_norm", "macro_regime", "quadrant", "leading_score",
                       "leading_divergence", "leading_label", "score_5d_trend",
                       "fear_greed", "fear_greed_label", "tactical_score"}
        _sig_items = [(k, v) for k, v in _raw_sigs.items()
                      if k not in _meta_keys and isinstance(v, (int, float))]
        if _sig_items:
            _SIG_PROB = {
                "vix": ("75%", "Best early warning — fired before 6/8 peaks"),
                "real_yield": ("62%", "Very early signal — avg 212d lead time"),
                "fedfunds": ("50%", "Policy-driven crashes — 69d avg lead"),
                "credit_ig": ("50%", "Credit stress detector — 64d avg lead"),
                "credit_hy": ("50%", "Risk appetite gauge — 38d avg lead"),
                "yield_curve": ("50%", "Recession predictor — 105d avg lead"),
                "credit_impulse": ("50%", "Credit cycle turn — 166d avg lead"),
                "umcsent": ("50%", "Slow burn detector — 145d avg lead"),
                "spy_trend": ("38%", "Confirms, doesn't lead — 25d avg lead"),
                "fci": ("25%", "Late confirmer — fires after peak"),
                "yield_curve_3m": ("25%", "Late confirmer — fires after peak"),
                "permit": ("25%", "Niche — only fired in GFC + EU Debt"),
                "icsa": ("0%", "Lagging — always fires after peak"),
                "indpro": ("0%", "Lagging — always fires after peak"),
            }
            _sig_items.sort(key=lambda x: x[1])
            _sb_rows = ""
            for _sk, _sv in _sig_items:
                _s_name = _sk.replace("_", " ").title()
                _s_color = "#22c55e" if _sv > 0.3 else ("#ef4444" if _sv < -0.3 else "#f59e0b")
                _s_arrow = "▲" if _sv > 0.3 else ("▼" if _sv < -0.3 else "►")
                _s_bar_w = min(100, int(abs(_sv) * 50))
                _prob_info = _SIG_PROB.get(_sk)
                _prob_html = ""
                if _prob_info and _sv < -0.3:
                    _prob_html = (
                        f'<div style="font-size:7px;color:#64748b;padding-left:16px;">'
                        f'Pre-peak probability: {_prob_info[0]} · {_prob_info[1]}</div>'
                    )
                _sb_rows += (
                    f'<div style="display:flex;align-items:center;gap:6px;padding:2px 0;'
                    f'border-bottom:1px solid #1e293b22;">'
                    f'<span style="color:{_s_color};font-size:9px;width:10px;">{_s_arrow}</span>'
                    f'<span style="color:#94a3b8;font-size:9px;width:120px;white-space:nowrap;'
                    f'overflow:hidden;text-overflow:ellipsis;">{_s_name}</span>'
                    f'<span style="color:{_s_color};font-size:10px;font-weight:700;width:45px;'
                    f'text-align:right;font-family:monospace;">{_sv:+.2f}</span>'
                    f'<div style="flex:1;background:#1e293b;border-radius:2px;height:3px;">'
                    f'<div style="background:{_s_color};width:{_s_bar_w}%;height:3px;border-radius:2px;'
                    f'margin-{"left:auto" if _sv >= 0 else "right:auto"};"></div></div>'
                    f'</div>'
                    f'{_prob_html}'
                )
            _signal_breakdown_block = (
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;'
                f'margin-bottom:6px;">SIGNAL HEALTH MONITOR</div>'
                f'{_sb_rows}'
                f'<div style="font-size:7px;color:#475569;margin-top:4px;">'
                f'Z-scores sorted worst-first · Green &gt;0.3 · Red &lt;-0.3 · '
                f'Pre-peak % = how often this signal fired before the peak across 8 historical crashes</div>'
                f'</div>'
            )

        # ── Kelly Criterion card ──────────────────────────────────────────────
        _kelly_block = ""
        _bimodal_block = ""
        _kly_half = None
        _tac_short_kelly_pct = None
        _tac_long_kelly_pct  = None
        def _build_kelly_ref_table(base_pct: float) -> str:
            """Compact scenario reference grid: alignment × HMM state → half-Kelly %."""
            _A_ROWS = [
                ("4/4", 1.00), ("3/4", 0.90), ("2/4", 0.75), ("1/4", 0.50), ("0/4", 0.25),
            ]
            _H_COLS = [
                ("Bull", 1.10), ("Neut", 1.00), ("Str", 0.85), ("Late", 0.75), ("Cris", 0.60),
            ]
            _header = "".join(
                f'<th style="padding:2px 5px;font-size:7px;color:#475569;'
                f'font-weight:700;text-align:center;">{h}</th>'
                for h, _ in _H_COLS
            )
            _body = ""
            for _a_lbl, _a_m in _A_ROWS:
                _cells = ""
                for _h_lbl, _h_m in _H_COLS:
                    _val = min(base_pct * _a_m * _h_m, 15.0)
                    _col = (
                        "#22c55e" if _val >= 10 else
                        "#f59e0b" if _val >= 6 else
                        "#ef4444"
                    )
                    _cells += (
                        f'<td style="padding:2px 5px;font-size:8px;font-weight:700;'
                        f'color:{_col};text-align:center;">{_val:.1f}%</td>'
                    )
                _body += (
                    f'<tr><td style="padding:2px 5px;font-size:7px;color:#475569;'
                    f'font-weight:700;white-space:nowrap;">{_a_lbl}</td>{_cells}</tr>'
                )
            return (
                f'<div style="border-top:1px solid #1e293b;margin:6px 0 4px;"></div>'
                f'<div style="font-size:7px;color:#334155;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:3px;">'
                f'HALF-KELLY REFERENCE · base {base_pct:.1f}% · rows=alignment cols=HMM</div>'
                f'<table style="border-collapse:collapse;width:100%;">'
                f'<thead><tr><th style="padding:2px 5px;"></th>{_header}</tr></thead>'
                f'<tbody>{_body}</tbody></table>'
            )

        def _build_kelly_chain(kly: dict, hmm_lbl: str = "N/A", align_col: str = "#f59e0b") -> str:
            """Visual step-by-step Kelly sizing chain: raw → stress → alignment → HMM → final."""
            _raw      = kly.get("kelly_half_raw_pct", kly.get("kelly_half_base_pct", 0))
            _base     = kly.get("kelly_half_base_pct", _raw)
            _final    = kly.get("kelly_half_pct", 0)
            _stress_d = kly.get("stress_discount_pct", 0)
            _fear     = kly.get("fear_score", 50)
            _a_m      = kly.get("align_multiplier", 1.0)
            _h_m      = kly.get("hmm_multiplier", 1.0)
            _n_ag     = kly.get("n_signals_agree", 0)
            _n_tot    = kly.get("n_signals_total", 0)
            _hmm_l    = hmm_lbl or "N/A"
            _capped   = kly.get("capped", False)

            def _node(val: float, label: str, sublabel: str, mult: str, color: str, arrow: bool = True) -> str:
                _mc = "#22c55e" if mult.startswith("×1") or mult.startswith("+") else (
                      "#ef4444" if any(mult.startswith(f"×0.{d}") for d in ["2","3","4","5","6","7"]) else "#f59e0b"
                )
                return (
                    f'<div style="display:flex;flex-direction:column;align-items:center;min-width:52px;">'
                    f'<div style="font-size:12px;font-weight:900;color:{color};">{val:.1f}%</div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;white-space:nowrap;">{label}</div>'
                    f'<div style="font-size:7px;color:#334155;white-space:nowrap;">{sublabel}</div>'
                    f'<div style="font-size:8px;font-weight:700;color:{_mc};margin-top:1px;">{mult}</div>'
                    f'</div>'
                    + (f'<div style="color:#334155;font-size:14px;padding:0 2px;align-self:center;">→</div>' if arrow else '')
                )

            _after_stress = _base
            _after_align  = round(_after_stress * _a_m, 1)
            _stress_mult  = f"−{_stress_d:.0f}% fear" if _stress_d > 0 else "×1.00"
            _align_mult   = f"×{_a_m:.2f} ({_n_ag}/{_n_tot})"
            _hmm_mult     = f"×{_h_m:.2f} {_hmm_l}"
            _cap_note     = ' <span style="color:#f59e0b;font-size:7px;">⚠ 15% cap</span>' if _capped else ""
            _final_col    = "#22c55e" if _final >= 10 else ("#f59e0b" if _final >= 5 else "#ef4444")

            _chain = (
                _node(_raw,         "HALF-KELLY",  f"p={kly.get('p',0)*100:.0f}% b={kly.get('b',1.0)}", "raw",          "#94a3b8")
                + _node(_after_stress, "AFTER STRESS", f"fear {_fear:.0f}/100",                             _stress_mult,   "#64748b")
                + _node(_after_align,  "AFTER ALIGN",  f"{_n_ag}/{_n_tot} signals",                          _align_mult,    align_col)
                + _node(_final,        "FINAL",          f"HMM {_hmm_l}",                                    _hmm_mult,      _final_col, arrow=False)
            )
            return (
                f'<div style="display:flex;align-items:stretch;gap:0;overflow-x:auto;'
                f'background:#0a0f1a;border:1px solid #1e293b;border-radius:4px;padding:6px 8px;">'
                f'{_chain}'
                f'</div>'
                f'<div style="font-size:7px;color:#334155;margin-top:3px;">'
                f'Half-Kelly = (b·p − q)/b ÷ 2 &nbsp;·&nbsp; '
                f'stress = fear/100 × 30% discount &nbsp;·&nbsp; '
                f'alignment ×signal confluence &nbsp;·&nbsp; '
                f'HMM ×regime state{_cap_note}</div>'
            )

        try:
            from services.portfolio_sizing import compute_qir_kelly as _compute_kelly, compute_triple_kelly as _compute_triple_kelly
            try:
                from services.hmm_regime import load_current_hmm_state as _hmm_load_ks
                _hmm_state_for_kelly = _hmm_load_ks()
                _hmm_label_for_kelly = getattr(_hmm_state_for_kelly, "state_label", None)
            except Exception:
                _hmm_label_for_kelly = None

            _is_gu = _cls.get("pattern") == "GENUINE_UNCERTAINTY"
            _triple_kelly_html = ""
            if _is_gu and _gu_profile:
                try:
                    _tkly = _compute_triple_kelly(
                        fear_composite=st.session_state.get("_fear_composite") or {},
                        regime_ctx=st.session_state.get("_regime_context") or {},
                        hmm_state_label=_hmm_label_for_kelly,
                        forced_lean=_gu_profile.get("lean", "BEARISH"),
                        lean_pct=float(_gu_profile.get("lean_pct", 53)),
                        uncertainty_score=int(_gu_profile.get("uncertainty_score", 60)),
                        macro_score=int((_rc or {}).get("macro_score") or 50),
                        leading_score=int((_rc or {}).get("leading_score") or 50),
                    )
                    _tk_accordions = ""
                    for _tk in (_tkly["structural"], _tkly["tactical_short"], _tkly["tactical_long"]):
                        _tk_col  = _tk["color"]
                        _tk_half = _tk["half_pct"]
                        _tk_accordions += (
                            f'<details style="border-bottom:1px solid #1e293b;margin:0;">'
                            f'<summary style="list-style:none;cursor:pointer;padding:7px 8px;'
                            f'display:flex;align-items:center;gap:10px;">'
                            f'<span style="font-size:10px;font-weight:700;color:{_tk_col};flex:0 0 140px;">'
                            f'{_tk["label"]}</span>'
                            f'<span style="font-size:20px;font-weight:900;color:{_tk_col};flex:0 0 60px;">'
                            f'{_tk_half}%</span>'
                            f'<span style="font-size:8px;color:#64748b;">{_tk["timeframe"]}</span>'
                            f'</summary>'
                            f'<div style="padding:6px 12px 10px 12px;background:#0a0f1a;">'
                            f'<div style="font-size:9px;color:#94a3b8;margin-bottom:3px;">'
                            f'p = {_tk["p"]*100:.0f}% &nbsp;·&nbsp; b = {_tk["b"]}</div>'
                            f'<div style="font-size:9px;color:#475569;">{_tk["note"]}</div>'
                            f'<div style="font-size:8px;color:#334155;margin-top:3px;">'
                            f'% of total portfolio to size this leg</div>'
                            f'</div>'
                            f'</details>'
                        )
                    _tac_short_kelly_pct = _tkly["tactical_short"]["half_pct"]
                    _tac_long_kelly_pct  = _tkly["tactical_long"]["half_pct"]
                    _triple_kelly_html = (
                        f'<div style="background:#0f172a;border:1px solid #334155;'
                        f'border-radius:6px;padding:10px 14px;margin-top:6px;">'
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
                        f'<span style="font-size:10px;color:#f59e0b;font-weight:700;letter-spacing:0.1em;">'
                        f'⚡ BIMODAL SIZING — TRIPLE KELLY</span>'
                        f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                        f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⚡ FAST · HOURS/DAYS</span>'
                        f'</div>'
                        f'<div style="font-size:8px;color:#334155;margin-bottom:8px;">'
                        f'Genuine Uncertainty active — three concurrent position legs · '
                        f'% shown = half-Kelly, i.e. recommended portfolio allocation per leg</div>'
                        f'<div style="border:1px solid #1e293b;border-radius:4px;overflow:hidden;">'
                        f'{_tk_accordions}'
                        f'</div>'
                        f'</div>'
                    )
                except Exception as _tk_err:
                    _triple_kelly_html = (
                        f'<div style="background:#1a0a00;border:1px solid #7f1d1d;'
                        f'border-radius:5px;padding:6px 10px;margin-bottom:6px;">'
                        f'<div style="font-size:8px;color:#ef4444;">⚠ Triple Kelly error: {_tk_err}</div>'
                        f'</div>'
                    )

            _kly = _compute_kelly(
                _conviction_score,
                st.session_state.get("_fear_composite") or {},
                st.session_state.get("_regime_context") or {},
                options_score=(st.session_state.get("_options_flow_context") or {}).get("options_score"),
                tactical_score=(st.session_state.get("_tactical_context") or {}).get("tactical_score"),
                hmm_state_label=_hmm_label_for_kelly,
            )
            _kly_half   = _kly["kelly_half_pct"]
            _kly_full   = _kly["kelly_full_pct"]
            _kly_p      = _kly["p"]
            _kly_b      = _kly["b"]
            _kly_psrc   = _kly["p_source"]
            _kly_bsrc   = _kly["b_source"]
            _kly_stress = _kly["stress_discount_pct"]
            _kly_cap    = _kly["capped"]
            _kly_viable = _kly["viable"]
            _kly_n_agree  = _kly.get("n_signals_agree", 0)
            _kly_n_total  = _kly.get("n_signals_total", 0)
            _kly_align_m  = _kly.get("align_multiplier", 1.0)
            _kly_hmm_m    = _kly.get("hmm_multiplier", 1.0)
            _kly_sdirs    = _kly.get("signal_dirs", {})

            _SIG_ORDER = [
                ("options",   "Options",  "Fast"),
                ("tactical",  "Tactical", "Med"),
                ("regime",    "Regime",   "Slow"),
                ("conviction","Conviction","Very Slow"),
            ]
            _verdict_dir_for_display = 1 if _kly_p > 0.55 else (-1 if _kly_p < 0.45 else 0)
            _sq_html = ""
            _lbl_html = ""
            for _sig_key, _sig_name, _sig_speed in _SIG_ORDER:
                _d = _kly_sdirs.get(_sig_key)
                if _d is None:
                    _sq_col = "#334155"; _sq_char = "—"
                elif _verdict_dir_for_display == 0:
                    # Kelly is neutral — no signal can truly "disagree"; directional = divergence, not error
                    _sq_col = "#f59e0b" if _d != 0 else "#334155"
                    _sq_char = "~"
                elif _d == _verdict_dir_for_display:
                    _sq_col = "#22c55e"; _sq_char = "✓"
                elif _d != 0:
                    _sq_col = "#ef4444"; _sq_char = "✗"
                else:
                    _sq_col = "#f59e0b"; _sq_char = "~"
                _dir_arrow = "↑" if _d == 1 else ("↓" if _d == -1 else ("~" if _d == 0 else "?"))
                _sq_html += (
                    f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
                    f'margin-right:6px;">'
                    f'<div style="width:18px;height:18px;background:{_sq_col};border-radius:3px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:10px;font-weight:900;color:#0f172a;">{_sq_char}</div>'
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;white-space:nowrap;">'
                    f'{_sig_speed}</div>'
                    f'</div>'
                )
                _lbl_html += (
                    f'<span style="font-size:9px;color:#64748b;">'
                    f'{_sig_name} <span style="color:#94a3b8;">{_dir_arrow}</span></span>  '
                )

            _align_pct_txt = f"{_kly_n_agree}/{_kly_n_total} agree"
            _align_mult_col = "#22c55e" if _kly_align_m >= 0.90 else ("#f59e0b" if _kly_align_m >= 0.75 else "#ef4444")
            _hmm_mult_col = "#22c55e" if _kly_hmm_m >= 1.0 else ("#f59e0b" if _kly_hmm_m >= 0.85 else "#ef4444")
            _hmm_lbl_txt = _hmm_label_for_kelly or "N/A"

            try:
                from services.forecast_tracker import get_stats as _get_fstats
                _fst = _get_fstats()
                def _streak_badge(n, stype, label):
                    if not n:
                        return (f'<span style="font-size:9px;color:#334155;">'
                                f'{label} <span style="color:#475569;">—</span></span>')
                    _sc = "#22c55e" if stype == "correct" else "#ef4444"
                    _icon = "🔥" if stype == "correct" else "❄️"
                    return (f'<span style="font-size:9px;color:#64748b;">{label} '
                            f'<span style="color:{_sc};font-weight:700;">{_icon}{n}</span></span>')
                _streak_row_html = (
                    f'<div style="border-top:1px solid #1e293b;margin:5px 0 4px;"></div>'
                    f'<div style="display:flex;gap:14px;align-items:center;">'
                    f'{_streak_badge(_fst.get("price_streak",0), _fst.get("price_streak_type"), "Price")}'
                    f'{_streak_badge(_fst.get("macro_streak",0), _fst.get("macro_streak_type"), "Signal")}'
                    f'<span style="font-size:8px;color:#334155;">streaks</span>'
                    f'</div>'
                )
            except Exception:
                _streak_row_html = ""

            if _kly_viable:
                _kly_col  = "#22c55e" if _kly_half >= 8 else "#f59e0b" if _kly_half >= 4 else "#94a3b8"
                _kly_cap_badge = (
                    '<span style="color:#f59e0b;font-size:9px;"> · 15% cap applied</span>'
                    if _kly_cap else ""
                )
                _kelly_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:6px 10px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">KELLY SIZING</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · WEEKS/MONTHS</span>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:4px;">'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">HALF-KELLY</div>'
                    f'<div style="font-size:20px;font-weight:900;color:{_kly_col};">{_kly_half}%</div>'
                    f'<div style="font-size:9px;color:#475569;">portfolio{_kly_cap_badge}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">FULL KELLY</div>'
                    f'<div style="font-size:20px;font-weight:900;color:#334155;">{_kly_full}%</div>'
                    f'<div style="font-size:9px;color:#334155;">reference</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">WIN/LOSS</div>'
                    f'<div style="font-size:20px;font-weight:900;color:#94a3b8;">b={_kly_b}</div>'
                    f'<div style="font-size:9px;color:#475569;">{_kly_bsrc}</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#475569;line-height:1.5;">'
                    f'p={_kly_p*100:.0f}% · {_kly_psrc}'
                    f'{"  ·  stress −" + str(_kly_stress) + "%" if _kly_stress > 0 else ""}'
                    f'</div>'
                    f'{_streak_row_html}'
                    f'<div style="border-top:1px solid #1e293b;margin:6px 0 5px;"></div>'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">SIGNAL ALIGNMENT</div>'
                    f'<div style="display:flex;align-items:flex-start;margin-bottom:4px;">{_sq_html}</div>'
                    f'<div style="margin-bottom:6px;">{_lbl_html}</div>'
                    f'<div style="border-top:1px solid #1e293b;margin:5px 0 6px;"></div>'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:5px;">SIZING CHAIN</div>'
                    f'{_build_kelly_chain(_kly, hmm_lbl=_hmm_lbl_txt, align_col=_align_mult_col)}'
                    f'{_build_kelly_ref_table(_kly.get("kelly_half_base_pct", _kly_half))}'
                    f'</div>'
                )
            else:
                _kelly_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">KELLY SIZING</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · WEEKS/MONTHS</span>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#ef444466;">Negative expectancy — no position suggested</div>'
                    f'<div style="font-size:10px;color:#334155;margin-top:2px;">p={_kly_p*100:.0f}% · b={_kly_b} · {_kly_psrc}</div>'
                    f'</div>'
                )
            _kelly_block       = _kelly_html        # structural sizing → MEDIUM
            _fast_setups_html  = _triple_kelly_html  # Buy/Short Setup → FAST
            _bimodal_block     = ""
        except Exception:
            pass

        # ── Entry Signal card ────────────────────────────────────────────────
        _entry_rec_html = ""
        _er_color = _entry_rec["color"]
        _er_bg    = _entry_rec["bg"]
        _er_icon  = _entry_rec["icon"]
        _er_verb  = _entry_rec["verdict"]
        _er_rsn   = _entry_rec["reasoning"]
        _er_ldg   = _entry_rec["leading_score"]
        _er_mac   = _entry_rec["macro_score"]
        _er_dpts  = _entry_rec["divergence_pts"]
        _er_dlbl  = _entry_rec["divergence_label"]
        _er_dsign = f"+{_er_dpts}" if _er_dpts >= 0 else str(_er_dpts)
        _er_div_color = (
            "#22c55e" if _er_dlbl == "Early Risk-On Setup" else
            "#ef4444" if _er_dlbl == "Early Risk-Off Warning" else
            "#64748b"
        )
        _er_div_fg = "#052e16" if _er_div_color == "#22c55e" else "white"
        _er_div_badge = (
            f'<span style="background:{_er_div_color};color:{_er_div_fg};'
            f'font-weight:800;font-size:8px;padding:1px 6px;border-radius:3px;letter-spacing:0.05em;">'
            f'{_er_dsign} pts · {_er_dlbl.upper()}</span>'
        )
        _entry_rec_html = (
            f'<div style="background:{_er_bg};border:1px solid {_er_color}44;'
            f'border-radius:6px;padding:10px 14px;margin:0 0 10px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="font-size:13px;color:#475569;font-weight:700;letter-spacing:0.1em;">ENTRY SIGNAL</span>'
            f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
            f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · MACRO+LEADING AGGREGATOR</span>'
            f'</div>'
            f'<div style="font-size:20px;font-weight:900;color:{_er_color};'
            f'letter-spacing:0.04em;margin-bottom:8px;">{_er_icon} {_er_verb}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:6px;">'
            f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">LEADING</div>'
            f'<div style="font-size:15px;font-weight:800;color:#f1f5f9;">{_er_ldg}/100</div>'
            f'<div style="font-size:7px;color:#475569;">fast signals</div></div>'
            f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">MACRO</div>'
            f'<div style="font-size:15px;font-weight:800;color:#f1f5f9;">{_er_mac}/100</div>'
            f'<div style="font-size:7px;color:#475569;">slow guard</div></div>'
            f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">TACTICAL</div>'
            f'<div style="font-size:15px;font-weight:800;'
            f'color:{"#22c55e" if _tac_s >= 62 else "#ef4444" if _tac_s < 40 else "#f1f5f9"};">'
            f'{_tac_s}/100</div>'
            f'<div style="font-size:7px;color:#475569;">dip · rip detector</div></div>'
            f'<div><div style="font-size:9px;color:#f59e0b;font-weight:700;letter-spacing:0.08em;margin-bottom:2px;">DIVERGENCE</div>'
            f'<div style="margin-top:2px;">{_er_div_badge}</div></div>'
            f'</div>'
            f'<div style="font-size:9px;color:#334155;margin-bottom:6px;padding:3px 0 0 2px;">'
            f'<span style="color:#475569;">options confirm · </span>'
            f'<span style="color:{"#22c55e" if _opts_s >= 60 else "#ef4444" if _opts_s < 40 else "#64748b"};">'
            f'{_opts_s}/100</span>'
            f'<span style="color:#334155;"> same-day flow — not a verdict driver</span>'
            f'</div>'
            f'<div style="background:#0d1117;border-left:3px solid {_er_color}44;'
            f'padding:7px 10px;font-size:11px;color:#94a3b8;line-height:1.6;border-radius:0 3px 3px 0;">'
            f'{_er_rsn}</div>'
            f'</div>'
        )

        # ── Market Top/Bottom Proximity card ──────────────────────────────────
        try:
            from services.hmm_regime import load_current_hmm_state as _tb_hmm_load
            _tb_hmm = _tb_hmm_load()
            _tb_entropy = getattr(_tb_hmm, "entropy", 0.0) or 0.0 if _tb_hmm else 0.0
            _tb_ll_z = getattr(_tb_hmm, "ll_zscore", 0.0) or 0.0 if _tb_hmm else 0.0
            _tb_conf = getattr(_tb_hmm, "confidence", 0.5) or 0.5 if _tb_hmm else 0.5
            _tb_hmm_label = getattr(_tb_hmm, "state_label", "") if _tb_hmm else ""
        except Exception:
            _tb_entropy, _tb_ll_z, _tb_conf, _tb_hmm_label = 0.0, 0.0, 0.5, ""

        _tb_regime = float(_rc.get("score") or 0)
        _tb_macro = float(_rc.get("macro_score") or 50)
        _tb_conv = float(_cls.get("conviction_score") or 0) if _cls.get("conviction_score") is not None else 0

        _tb_vel = 0.0
        try:
            import json as _tb_json, os as _tb_os
            _tb_vpath = _tb_os.path.join(_tb_os.path.dirname(_tb_os.path.dirname(__file__)), "data", "tactical_score_history.json")
            with open(_tb_vpath) as _tb_vf:
                _tb_vhist = _tb_json.load(_tb_vf)
            if _tb_vhist and len(_tb_vhist) >= 6:
                _tb_vel = float(_rc.get("macro_score") or 50) - float(_tb_vhist[-6].get("score", 50))
        except Exception:
            pass

        # ── Wyckoff signals ────────────────────────────────────────────────────
        _tb_wyckoff = None
        try:
            from services.market_data import fetch_wyckoff_spy as _fw_spy
            _tb_wyckoff = _fw_spy()
        except Exception:
            pass

        # ── HY credit spreads ──────────────────────────────────────────────────
        _tb_hy = None
        try:
            from services.market_data import fetch_hy_spread as _fhs
            _tb_hy = _fhs()
        except Exception:
            pass

        # ── Market breadth (% SPX above 200MA) ────────────────────────────────
        _tb_breadth = None
        try:
            from services.market_data import fetch_breadth_pct as _fbp
            _tb_breadth = _fbp()
        except Exception:
            pass

        # ── AAII sentiment ─────────────────────────────────────────────────────
        _tb_aaii = st.session_state.get("_aaii_sentiment") or {}

        _top_signals = []
        _bottom_signals = []

        # ── TOP signals — calibrated thresholds (empirical avg at 8 known peaks) ──
        # regime avg=+0.14 (88% hit), entropy avg=0.71 (75%), conviction avg=17 (75%)
        # ll_z avg=-6.0 — tightened from -0.5 to -3.0 to reduce false fires
        if _tb_regime > 0.05:
            _top_signals.append(("Regime elevated", min(100, _tb_regime * 180)))
        if _tb_vel < -3:
            _top_signals.append(("Velocity turning negative", min(100, abs(_tb_vel) * 5)))
        if _tb_entropy > 0.68:
            _top_signals.append(("High regime entropy", min(100, (_tb_entropy - 0.45) * 200)))
        if _tb_conv < 22:
            _top_signals.append(("Low conviction", min(100, (22 - _tb_conv) * 5)))
        if _tb_ll_z < -3.0:
            _top_signals.append(("LL deteriorating", min(100, abs(_tb_ll_z) * 8)))
        # Late Cycle → top only (not bottom) — empirically fires at peaks
        if _tb_hmm_label in ("Late Cycle", "Stress", "Early Stress"):
            _top_signals.append(("HMM late/stress state", 65))

        # Wyckoff top signals — only Distribution is reliable (38% hit at peaks)
        # Accumulation at peaks = 62% false positive → NOT a top signal
        if _tb_wyckoff:
            _wk_phase = _tb_wyckoff.get("phase", "")
            _wk_conf  = _tb_wyckoff.get("confidence", 0)
            _wk_sub   = _tb_wyckoff.get("sub_phase", "")
            _wk_res   = _tb_wyckoff.get("resistance")
            _wk_tgt   = _tb_wyckoff.get("cause_target")
            _wk_last  = _tb_wyckoff.get("spy_last")
            if _wk_phase == "Distribution":
                _top_signals.append((f"Wyckoff Distribution {_wk_sub} ({_wk_conf}% conf)", min(100, _wk_conf)))
            if _wk_phase == "Markup" and _wk_sub in ("D", "E"):
                _top_signals.append((f"Wyckoff Markup late phase {_wk_sub}", min(80, _wk_conf)))
            if _wk_res and _wk_last and _wk_res > 0:
                _res_prox = (_wk_last - _wk_res) / _wk_res * 100
                if -2.0 <= _res_prox <= 1.5:
                    _top_signals.append((f"SPY at Wyckoff resistance ${_wk_res:.0f}", min(90, 50 + _wk_conf // 2)))
            if _wk_tgt and _wk_last and _wk_phase == "Distribution":
                if _wk_tgt < _wk_last * 0.98:
                    _top_signals.append((f"Wyckoff downside target ${_wk_tgt:.0f}", min(80, _wk_conf)))

        # ── HY Credit Spread signals ───────────────────────────────────────────
        # Tight spreads = complacency → TOP zone; wide spreads = max fear → BOTTOM zone
        if _tb_hy:
            _hy_level = _tb_hy.get("level")
            _hy_z     = _tb_hy.get("zscore")
            if _hy_level is not None:
                if _hy_level < 3.5:
                    _top_signals.append((f"HY spreads historically tight ({_hy_level:.1f}%)", 75))
                elif _hy_level < 4.5 and _hy_z is not None and _hy_z < -0.5:
                    _top_signals.append((f"HY spreads tight + compressing ({_hy_level:.1f}%)", 55))

        # ── AAII sentiment (contrarian) ────────────────────────────────────────
        _aaii_bull = float(_tb_aaii.get("bull_pct", 0) or 0)
        if _aaii_bull > 55:
            _top_signals.append((f"AAII extreme bulls ({_aaii_bull:.0f}%)", min(80, int((_aaii_bull - 40) * 2))))

        # ── Market breadth ─────────────────────────────────────────────────────
        if _tb_breadth:
            _brd_pct = _tb_breadth.get("pct", 50)
            if _brd_pct > 80:
                _top_signals.append((f"Breadth extended — {_brd_pct:.0f}% above 200MA", min(70, int((_brd_pct - 60) * 2))))

        # ── BOTTOM signals — calibrated thresholds (empirical avg at 8 known troughs) ──
        # regime avg=-0.35 (100% hit), macro avg=32.8 (88%), conviction avg=34.4 (88%)
        # ll_z avg=-20.9 — tightened from -5 to -8 to reduce noise
        # Late Cycle removed from bottom — it fires at tops too, causing double-counting
        if _tb_regime < -0.17:
            _bottom_signals.append(("Regime deep negative", min(100, abs(_tb_regime) * 220)))
        if _tb_vel > 3:
            _bottom_signals.append(("Velocity turning positive", min(100, _tb_vel * 5)))
        if _tb_macro < 37:
            _bottom_signals.append(("Macro crushed", min(100, (37 - _tb_macro) * 6)))
        if _tb_conv > 24:
            _bottom_signals.append(("Conviction building", min(100, _tb_conv * 2)))
        if _tb_ll_z < -8:
            _bottom_signals.append(("Extreme LL stress", min(100, abs(_tb_ll_z) * 3)))
        # Crisis only for bottom HMM (not Late Cycle — it's a top indicator)
        if _tb_hmm_label in ("Crisis",):
            _bottom_signals.append(("HMM Crisis state", 75))

        # Wyckoff bottom signals
        if _tb_wyckoff:
            _wk_phase = _tb_wyckoff.get("phase", "")
            _wk_conf  = _tb_wyckoff.get("confidence", 0)
            _wk_sub   = _tb_wyckoff.get("sub_phase", "")
            _wk_sup   = _tb_wyckoff.get("support")
            _wk_tgt   = _tb_wyckoff.get("cause_target")
            _wk_last  = _tb_wyckoff.get("spy_last")
            if _wk_phase == "Accumulation":
                _bottom_signals.append((f"Wyckoff Accumulation {_wk_sub} ({_wk_conf}% conf)", min(100, _wk_conf)))
            if _wk_phase == "Markdown" and _wk_sub in ("D", "E"):
                _bottom_signals.append((f"Wyckoff Markdown exhaustion {_wk_sub}", min(80, _wk_conf)))
            if _wk_sup and _wk_last and _wk_sup > 0:
                _sup_prox = (_wk_last - _wk_sup) / _wk_sup * 100
                if -1.5 <= _sup_prox <= 2.0:
                    _bottom_signals.append((f"SPY at Wyckoff support ${_wk_sup:.0f}", min(90, 50 + _wk_conf // 2)))
            if _wk_tgt and _wk_last and _wk_phase == "Accumulation":
                if _wk_tgt > _wk_last * 1.02:
                    _bottom_signals.append((f"Wyckoff upside target ${_wk_tgt:.0f}", min(80, _wk_conf)))

        # ── HY Credit Spread signals ───────────────────────────────────────────
        if _tb_hy:
            _hy_level = _tb_hy.get("level")
            _hy_z     = _tb_hy.get("zscore")
            if _hy_level is not None:
                if _hy_level > 7.0:
                    _bottom_signals.append((f"HY spreads at crisis level ({_hy_level:.1f}%)", 80))
                elif _hy_level > 5.5 and _hy_z is not None and _hy_z > 1.5:
                    _bottom_signals.append((f"HY spreads elevated + rising ({_hy_level:.1f}%)", 60))

        # ── AAII sentiment (contrarian) ────────────────────────────────────────
        _aaii_bear = float(_tb_aaii.get("bear_pct", 0) or 0)
        if _aaii_bear > 50:
            _bottom_signals.append((f"AAII extreme bears ({_aaii_bear:.0f}%)", min(85, int((_aaii_bear - 35) * 2))))

        # ── Market breadth ─────────────────────────────────────────────────────
        if _tb_breadth:
            _brd_pct = _tb_breadth.get("pct", 50)
            if _brd_pct < 20:
                _bottom_signals.append((f"Breadth washed out — {_brd_pct:.0f}% above 200MA", min(80, int((20 - _brd_pct) * 3))))

        _top_score = round(sum(s for _, s in _top_signals) / max(1, len(_top_signals))) if _top_signals else 0
        _bot_score = round(sum(s for _, s in _bottom_signals) / max(1, len(_bottom_signals))) if _bottom_signals else 0

        if _top_signals or _bottom_signals:
            _tb_rows = ""
            if _top_signals:
                _top_color = "#ef4444" if _top_score >= 50 else ("#f59e0b" if _top_score >= 25 else "#22c55e")
                _tb_rows += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:4px 0;border-bottom:1px solid #1e293b33;">'
                    f'<span style="color:#ef4444;font-size:10px;font-weight:700;">MARKET TOP</span>'
                    f'<span style="color:{_top_color};font-size:12px;font-weight:800;">{_top_score}%</span>'
                    f'</div>'
                )
                for _ts_name, _ts_val in _top_signals:
                    _tb_rows += (
                        f'<div style="font-size:8px;color:#64748b;padding:1px 0 1px 8px;">'
                        f'{"●" if _ts_val >= 50 else "○"} {_ts_name}</div>'
                    )
            if _bottom_signals:
                _bot_color = "#22c55e" if _bot_score >= 50 else ("#f59e0b" if _bot_score >= 25 else "#64748b")
                _tb_rows += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:4px 0;border-bottom:1px solid #1e293b33;margin-top:4px;">'
                    f'<span style="color:#22c55e;font-size:10px;font-weight:700;">MARKET BOTTOM</span>'
                    f'<span style="color:{_bot_color};font-size:12px;font-weight:800;">{_bot_score}%</span>'
                    f'</div>'
                )
                for _bs_name, _bs_val in _bottom_signals:
                    _tb_rows += (
                        f'<div style="font-size:8px;color:#64748b;padding:1px 0 1px 8px;">'
                        f'{"●" if _bs_val >= 50 else "○"} {_bs_name}</div>'
                    )

            _top_bottom_block = (
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">TOP / BOTTOM PROXIMITY</span>'
                f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</span>'
                f'</div>'
                f'{_tb_rows}'
                f'<div style="font-size:7px;color:#475569;margin-top:4px;">'
                f'Empirically calibrated from 8 crash peaks &amp; troughs · '
                f'Peak avg: regime +0.14, entropy 0.71, conviction 17 · '
                f'Trough avg: regime -0.35, macro 33, conviction 34, LL_z -21 · '
                f'Wyckoff S/R · HY credit spreads · AAII sentiment · SPX breadth</div>'
                f'</div>'
            )

            # Auto-log proximity scores to history file
            try:
                import json as _al_json, os as _al_os
                from datetime import date as _al_date
                _al_path = _al_os.path.join(_al_os.path.dirname(_al_os.path.dirname(__file__)), "data", "top_bottom_history.json")
                _al_today = str(_al_date.today())
                _al_hist = []
                if _al_os.path.exists(_al_path):
                    with open(_al_path, "r") as _al_f:
                        _al_hist = _al_json.load(_al_f)
                if not any(h.get("date") == _al_today for h in _al_hist):
                    _al_hist.append({
                        "date": _al_today,
                        "top_pct": _top_score,
                        "bottom_pct": _bot_score,
                        "regime_score": round(_tb_regime, 3),
                        "velocity": round(_tb_vel, 1),
                        "entropy": round(_tb_entropy, 3),
                        "ll_z": round(_tb_ll_z, 2),
                        "conviction": round(_tb_conv, 0),
                        "hmm_state": _tb_hmm_label,
                        "wyckoff_phase": (_tb_wyckoff or {}).get("phase", "N/A"),
                        "wyckoff_conf": (_tb_wyckoff or {}).get("confidence", 0),
                        "top_signals": [s[0] for s in _top_signals],
                        "bottom_signals": [s[0] for s in _bottom_signals],
                    })
                    _al_hist = _al_hist[-365:]
                    with open(_al_path, "w") as _al_f:
                        _al_json.dump(_al_hist, _al_f, indent=1)
                st.session_state["_top_bottom_proximity"] = {
                    "top_pct": _top_score, "bottom_pct": _bot_score,
                    "top_signals": [s[0] for s in _top_signals],
                    "bottom_signals": [s[0] for s in _bottom_signals],
                }
            except Exception:
                pass

        # ── HMM Brain State card ──────────────────────────────────────────────
        try:
            from services.hmm_regime import (
                load_current_hmm_state, load_hmm_brain,
                get_state_color, get_state_arrow, get_state_tips,
                get_hmm_state_history, get_conviction_multiplier,
            )
            _hmm_s = load_current_hmm_state()
            _hmm_b = load_hmm_brain()
            if _hmm_s is not None and _hmm_b is not None:
                _hs_col = get_state_color(_hmm_s.state_label)
                _hs_arr = get_state_arrow(_hmm_s.state_label)
                _hs_conf = int(_hmm_s.confidence * 100)
                _hs_pers = _hmm_s.persistence
                _hs_trained = _hmm_b.trained_at[:10] if _hmm_b.trained_at else "—"

                _hs_retrain_due = False
                try:
                    from datetime import datetime, timezone as _hmm_tz
                    _hs_dt = datetime.fromisoformat(_hmm_b.trained_at.replace("Z", "+00:00"))
                    _hs_days_since = (datetime.now(_hmm_tz.utc) - _hs_dt).days
                    _hs_retrain_due = _hs_days_since >= 90
                except Exception:
                    pass
                _hs_retrain_badge = (
                    f'<span style="background:#92400e;color:#fcd34d;font-size:8px;'
                    f'font-weight:700;padding:1px 6px;border-radius:3px;'
                    f'letter-spacing:0.05em;margin-left:6px;">RETRAIN DUE</span>'
                    if _hs_retrain_due else ""
                )

                _trans_row = _hmm_b.transmat[_hmm_s.state_idx]
                def _fmt_prob(v):
                    pct = v * 100
                    if pct >= 1:
                        return f"{pct:.0f}%"
                    elif pct > 0:
                        return f"<1%"
                    return "0%"
                _trans_cells = "".join(
                    f'<span style="font-size:9px;color:#64748b;">'
                    f'{_hmm_b.state_labels[j][:4]} '
                    f'<span style="color:#94a3b8;">{_fmt_prob(_trans_row[j])}</span>'
                    f'</span>  '
                    for j in range(_hmm_b.n_states)
                )

                # ── Live Sensor: LL + Entropy + Transition Projections ────────
                _ll_z = getattr(_hmm_s, "ll_zscore", 0.0) or 0.0
                _entropy = getattr(_hmm_s, "entropy", 0.0) or 0.0
                _tr_1m = getattr(_hmm_s, "transition_risk_1m", 0.0) or 0.0
                _tr_3m = getattr(_hmm_s, "transition_risk_3m", 0.0) or 0.0
                _tr_6m = getattr(_hmm_s, "transition_risk_6m", 0.0) or 0.0
                _fc_1m = getattr(_hmm_s, "forecast_1m", None)
                _fc_3m = getattr(_hmm_s, "forecast_3m", None)
                _fc_6m = getattr(_hmm_s, "forecast_6m", None)

                _ll_col = "#22c55e" if _ll_z > -1 else ("#f59e0b" if _ll_z > -2 else "#ef4444")
                _ll_label = "Normal" if _ll_z > -1 else ("Caution" if _ll_z > -2 else "CHECK ENGINE")

                _ent_col = "#22c55e" if _entropy < 0.3 else ("#f59e0b" if _entropy < 0.6 else "#ef4444")
                _ent_label = "Pure" if _entropy < 0.3 else ("Mixed" if _entropy < 0.6 else "Fog")
                _ent_bar_w = int(min(_entropy * 100, 100))

                _tr1_col = "#22c55e" if _tr_1m < 0.03 else ("#f59e0b" if _tr_1m < 0.10 else "#ef4444")
                _tr3_col = "#22c55e" if _tr_3m < 0.10 else ("#f59e0b" if _tr_3m < 0.25 else "#ef4444")
                _tr6_col = "#22c55e" if _tr_6m < 0.20 else ("#f59e0b" if _tr_6m < 0.40 else "#ef4444")

                def _build_forecast_bars(forecast, horizon_label, tr_pct, tr_col):
                    if not forecast or not _hmm_b:
                        return f'<div style="font-size:9px;color:#334155;">{horizon_label}: N/A</div>'
                    bars = ""
                    for j, p in enumerate(forecast):
                        bar_w = max(int(p * 100), 1)
                        lbl = _hmm_b.state_labels[j][:4] if j < len(_hmm_b.state_labels) else f"S{j}"
                        s_col = get_state_color(_hmm_b.state_labels[j]) if j < len(_hmm_b.state_labels) else "#64748b"
                        is_current = (j == _hmm_s.state_idx)
                        border = f"border:1px solid {s_col};" if is_current else ""
                        bars += (
                            f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px;">'
                            f'<span style="font-size:8px;color:#64748b;width:32px;text-align:right;">{lbl}</span>'
                            f'<div style="background:#1e293b;border-radius:2px;flex:1;height:10px;{border}">'
                            f'<div style="background:{s_col};width:{bar_w}%;height:100%;border-radius:2px;'
                            f'opacity:{"1.0" if is_current else "0.5"};"></div>'
                            f'</div>'
                            f'<span style="font-size:8px;color:#94a3b8;width:28px;">{p*100:.1f}%</span>'
                            f'</div>'
                        )
                    return (
                        f'<div>'
                        f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                        f'letter-spacing:0.08em;margin-bottom:4px;">{horizon_label}</div>'
                        f'<div style="font-size:10px;color:{tr_col};font-weight:700;margin-bottom:4px;">'
                        f'Transition Risk: {tr_pct*100:.1f}%</div>'
                        f'{bars}'
                        f'</div>'
                    )

                _forecast_1m_html = _build_forecast_bars(_fc_1m, "1-MONTH OUTLOOK", _tr_1m, _tr1_col)
                _forecast_3m_html = _build_forecast_bars(_fc_3m, "3-MONTH OUTLOOK", _tr_3m, _tr3_col)
                _forecast_6m_html = _build_forecast_bars(_fc_6m, "6-MONTH OUTLOOK", _tr_6m, _tr6_col)

                _live_sensor_html = (
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;'
                    f'margin-top:8px;padding-top:8px;border-top:1px solid #1e293b;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:3px;">LOG-LIKELIHOOD</div>'
                    f'<div style="font-size:16px;font-weight:900;color:{_ll_col};">'
                    f'{_ll_z:+.2f}z</div>'
                    f'<div style="font-size:9px;color:{_ll_col};font-weight:600;">{_ll_label}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:3px;">ENTROPY</div>'
                    f'<div style="display:flex;align-items:center;gap:6px;">'
                    f'<div style="font-size:16px;font-weight:900;color:{_ent_col};">'
                    f'{_entropy:.2f}</div>'
                    f'<div style="font-size:9px;color:{_ent_col};font-weight:600;">{_ent_label}</div>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:2px;height:4px;margin-top:3px;">'
                    f'<div style="background:{_ent_col};width:{_ent_bar_w}%;height:100%;'
                    f'border-radius:2px;"></div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-top:6px;padding:5px 8px;background:#0a0f1a;'
                    f'border-radius:3px;border:1px solid #1e293b;">'
                    f'<div style="font-size:8px;color:#334155;line-height:1.6;">'
                    f'<b style="color:#3b4f6b;">Log-Likelihood</b> measures how well today\'s '
                    f'market data fits the trained model. A dropping LL (z &lt; -1) means the '
                    f'market is behaving unusually — the "Check Engine" light. '
                    f'<b style="color:#3b4f6b;">Entropy</b> measures regime certainty: '
                    f'0 = the model is sure of the current state, 1 = total fog between states. '
                    f'Rising entropy dampens the Kelly multiplier automatically.<br>'
                    f'<span style="color:#475569;">Note: The HMM only sees credit spreads, yields, '
                    f'and VIX — not equity prices. Equity-only corrections (e.g. Apr 2025) '
                    f'that don\'t spill into credit markets will not flip the regime state. '
                    f'Watch the LL for early divergence.</span>'
                    f'</div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'
                    f'margin-top:8px;padding-top:8px;border-top:1px solid #1e293b;">'
                    f'{_forecast_1m_html}'
                    f'{_forecast_3m_html}'
                    f'{_forecast_6m_html}'
                    f'</div>'
                )

                _hmm_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-left:3px solid {_hs_col};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<span style="font-size:13px;color:#475569;font-weight:700;letter-spacing:0.1em;">HMM BRAIN STATE</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · WEEKS/MONTHS</span>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:6px;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">REGIME</div>'
                    f'<div style="font-size:20px;font-weight:900;color:{_hs_col};'
                    f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                    f'{_hs_arr}</div>'
                    f'<div style="font-size:11px;font-weight:700;color:{_hs_col};">'
                    f'{_hmm_s.state_label}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">CONFIDENCE</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#94a3b8;">'
                    f'{_hs_conf}%</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">PERSISTENCE</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#94a3b8;">'
                    f'{_hs_pers}d</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:2px;">'
                    f'→ {_trans_cells}'
                    f'</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:3px;">'
                    f'{_hmm_b.n_states}-state model · trained {_hs_trained} · '
                    f'BIC {_hmm_b.bic:,.0f}{_hs_retrain_badge}</div>'
                    f'{_live_sensor_html}'
                    f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #1e293b;'
                    f'font-size:10px;color:#64748b;line-height:1.6;">'
                    f'<span style="color:{_hs_col};font-weight:700;">What to do: </span>'
                    f'{get_state_tips(_hmm_s.state_label)}</div>'
                    f'<div style="margin-top:8px;padding:6px 10px;background:#0a0f1a;'
                    f'border-radius:4px;border:1px solid #1e293b;">'
                    f'<div style="font-size:8px;color:#1e293b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:4px;">ARCHITECTURE NOTES</div>'
                    f'<div style="font-size:8px;color:#1e3a5f;line-height:1.7;">'
                    f'HMM is a <span style="color:#1e4a7a;">MULTIPLIER</span>, not a signal — '
                    f'it gates Kelly sizing only, never the entry direction.<br>'
                    f'<span style="color:#1e4a7a;">Structural Long</span> always uses long-side b '
                    f'(_REGIME_B_IMPLIED). The forced lean must NOT flip the structural b — '
                    f'lean is short-duration noise (sentiment, options, event risk); '
                    f'structural is a weeks/months regime trade.<br>'
                    f'<span style="color:#1e4a7a;">Tactical Short</span> uses short-side b '
                    f'(_SHORT_B_IMPLIED) + fear boost. It expresses the lean, not the regime.<br>'
                    f'<span style="color:#1e4a7a;">init_params="smc"</span> — the "t" is excluded '
                    f'so hmmlearn does not overwrite the seeded diagonal prior (0.70) on each fit. '
                    f'Diagonal prior + Laplace 1e-6 prevent ping-pong states.</div>'
                    f'</div>'
                )

                # ── State Calibration Table ──
                _cal_html = ""
                try:
                    _cal_hist = get_hmm_state_history()
                    _cal_data = {}
                    for _ch in _cal_hist:
                        if "fwd_20d_spy_return" in _ch:
                            _cl = _ch.get("state_label", "?")
                            _cal_data.setdefault(_cl, []).append(_ch["fwd_20d_spy_return"])
                    if _cal_data:
                        _cal_rows = ""
                        _all_labels = ["Bull", "Neutral", "Stress", "Late Cycle", "Crisis"]
                        for _cl in _all_labels:
                            _rets = _cal_data.get(_cl, [])
                            _mult = get_conviction_multiplier(_cl)
                            if _rets:
                                _avg = sum(_rets) / len(_rets)
                                _avg_col = "#22c55e" if _avg >= 0 else "#ef4444"
                                _cal_rows += (
                                    f'<tr><td style="color:{get_state_color(_cl)};font-weight:700;">{_cl}</td>'
                                    f'<td style="color:{_avg_col};font-weight:700;">{_avg:+.2f}%</td>'
                                    f'<td>{len(_rets)}</td>'
                                    f'<td>{_mult:.2f}x</td></tr>'
                                )
                            else:
                                _cal_rows += (
                                    f'<tr><td style="color:{get_state_color(_cl)};font-weight:700;">{_cl}</td>'
                                    f'<td style="color:#334155;">waiting</td>'
                                    f'<td>0</td>'
                                    f'<td>{_mult:.2f}x</td></tr>'
                                )
                        _cal_html = (
                            f'<div style="margin-top:8px;padding:6px 10px;background:#0a0f1a;'
                            f'border-radius:4px;border:1px solid #1e293b;">'
                            f'<div style="font-size:8px;color:#3b4f6b;font-weight:700;'
                            f'letter-spacing:0.08em;margin-bottom:4px;">STATE CALIBRATION (20d fwd SPY)</div>'
                            f'<table style="width:100%;font-size:9px;color:#64748b;border-collapse:collapse;">'
                            f'<tr style="border-bottom:1px solid #1e293b;">'
                            f'<th style="text-align:left;padding:2px 4px;">State</th>'
                            f'<th style="text-align:left;padding:2px 4px;">Avg Ret</th>'
                            f'<th style="text-align:left;padding:2px 4px;">N</th>'
                            f'<th style="text-align:left;padding:2px 4px;">Mult</th></tr>'
                            f'{_cal_rows}</table></div>'
                        )
                except Exception:
                    pass

                _hmm_html = _hmm_html + _cal_html + f'</div>'
                _hmm_block = _hmm_html
            elif _hmm_b is None:
                _hmm_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:5px;padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">HMM BRAIN STATE</div>'
                    f'<div style="font-size:11px;color:#ef444466;">'
                    f'No model trained — click Retrain HMM below to build your regime brain</div>'
                    f'</div>'
                )
        except Exception as _hmm_exc:
            import traceback as _hmm_tb
            _hmm_block = (
                f'<div style="background:#1a0000;border:1px solid #ef4444;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="font-size:11px;color:#ef4444;font-weight:700;">HMM BLOCK ERROR</div>'
                f'<pre style="font-size:9px;color:#f87171;white-space:pre-wrap;">'
                f'{"".join(_hmm_tb.format_exception(type(_hmm_exc), _hmm_exc, _hmm_exc.__traceback__))}'
                f'</pre></div>'
            )

        # ── GEX Dealer Positioning card ──────────────────────────────────────
        try:
            _rc_fresh = st.session_state.get("_regime_context") or {}
            _gex_data = _rc_fresh.get("gex_profile") or st.session_state.get("_gex_profile_spx") or _rc.get("gex_profile")
            if not _gex_data:
                try:
                    from services.market_data import fetch_gex_profile as _fgp_fallback
                    _fgp_fn = getattr(_fgp_fallback, "__wrapped__", _fgp_fallback)
                    _gex_data = _fgp_fn("SPY", 3)
                except Exception:
                    pass
            if _gex_data:
                import numpy as _np_gex
                _gx_spot = _gex_data.get("spot", 0)
                _gx_flip = _gex_data.get("gamma_flip", _gx_spot)
                _gx_cw = _gex_data.get("call_wall", _gx_spot)
                _gx_pw = _gex_data.get("put_wall", _gx_spot)
                _gx_total = _gex_data.get("total_gex", 0)
                _gx_delta = _gex_data.get("dealer_net_delta", 0)
                _gx_zone = _gex_data.get("zone", "")

                _gx_zone_s = float(_np_gex.tanh(_gx_total / 2000.0))
                _gx_flip_pct = ((_gx_spot - _gx_flip) / max(_gx_spot, 1)) * 100
                _gx_flip_s = max(-1, min(1, _gx_flip_pct / 3.0))
                _gx_delta_s = max(-1, min(1, _gx_delta))
                _gx_wall_pct = ((_gx_cw - _gx_pw) / max(_gx_spot, 1)) * 100
                _gx_width_s = max(-1, min(1, (_gx_wall_pct - 5) / 5.0))
                _gx_composite = max(-1, min(1,
                    0.35 * _gx_zone_s + 0.25 * _gx_flip_s + 0.25 * _gx_delta_s + 0.15 * _gx_width_s
                ))

                _gx_comp_col = "#22c55e" if _gx_composite > 0.1 else ("#ef4444" if _gx_composite < -0.1 else "#f59e0b")
                _gx_zone_col = "#22c55e" if "Positive" in _gx_zone else "#ef4444"
                _gx_delta_col = "#22c55e" if _gx_delta >= 0 else "#ef4444"
                _gx_flip_col = "#22c55e" if _gx_flip_pct > 0.5 else ("#ef4444" if _gx_flip_pct < -0.5 else "#f59e0b")

                def _gex_factor_bar(label, value, color, detail):
                    bar_w = int(min(abs(value) * 100, 100))
                    bar_dir = "right" if value >= 0 else "left"
                    return (
                        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
                        f'<span style="font-size:8px;color:#64748b;width:42px;text-align:right;">{label}</span>'
                        f'<div style="background:#1e293b;border-radius:2px;flex:1;height:8px;position:relative;">'
                        f'<div style="background:{color};width:{bar_w}%;height:100%;border-radius:2px;'
                        f'float:{bar_dir};"></div>'
                        f'</div>'
                        f'<span style="font-size:8px;color:#94a3b8;width:50px;">{detail}</span>'
                        f'</div>'
                    )

                _gex_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-left:3px solid {_gx_comp_col};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;">GEX DEALER POSITIONING</div>'
                    f'<div style="display:flex;gap:14px;align-items:flex-end;">'
                    f'<div style="text-align:right;">'
                    f'<div style="font-size:14px;font-weight:800;color:{_gx_delta_col};">{_gx_delta:+.3f}</div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;">DELTA</div>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<div style="font-size:18px;font-weight:900;color:{_gx_comp_col};">'
                    f'{_gx_composite:+.2f}</div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">COMPOSITE</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:6px;margin-bottom:8px;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">SPOT</div>'
                    f'<div style="font-size:13px;font-weight:800;color:#94a3b8;">${_gx_spot:,.0f}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">FLIP</div>'
                    f'<div style="font-size:13px;font-weight:800;color:{_gx_flip_col};">${_gx_flip:,.0f}</div>'
                    f'<div style="font-size:8px;color:{_gx_flip_col};">{_gx_flip_pct:+.1f}%</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">DELTA</div>'
                    f'<div style="font-size:13px;font-weight:800;color:{_gx_delta_col};">{_gx_delta:+.3f}</div>'
                    f'<div style="font-size:7px;color:#3b5998;">call/put OI ratio</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">PUT WALL</div>'
                    f'<div style="font-size:13px;font-weight:800;color:#ef4444;">${_gx_pw:,.0f}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">CALL WALL</div>'
                    f'<div style="font-size:13px;font-weight:800;color:#22c55e;">${_gx_cw:,.0f}</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-bottom:4px;">'
                    f'{_gex_factor_bar("Zone", _gx_zone_s, _gx_zone_col, f"{_gx_total:+.0f}M")}'
                    f'{_gex_factor_bar("Flip", _gx_flip_s, _gx_flip_col, f"{_gx_flip_pct:+.1f}%")}'
                    f'{_gex_factor_bar("Delta", _gx_delta_s, _gx_delta_col, f"{_gx_delta:+.3f}")}'
                    f'{_gex_factor_bar("Width", _gx_width_s, "#94a3b8", f"{_gx_wall_pct:.1f}%")}'
                    f'</div>'
                    f'<div style="font-size:9px;color:{_gx_zone_col};font-weight:600;">'
                    f'{_gx_zone}</div>'
                    f'<div style="font-size:8px;color:#334155;margin-top:3px;">'
                    f'{_gex_data.get("zone_detail", "")}</div>'
                    f'<div style="margin-top:6px;padding:5px 8px;background:#0a0f1a;'
                    f'border-radius:3px;border:1px solid #1e293b;'
                    f'font-size:8px;color:#3b5998;line-height:1.7;">'
                    f'<b style="color:#4a6fa5;">Zone</b> = gamma at spot strike (do dealers dampen or amplify moves <i>here</i>). '
                    f'<b style="color:#4a6fa5;">Delta</b> = net directional lean across ALL strikes (are dealers long or short the market). '
                    f'<b style="color:#4a6fa5;">Flip</b> = distance to gamma sign change — how far price must move before dealers switch behavior. '
                    f'<b style="color:#4a6fa5;">Width</b> = call wall minus put wall — wider = more dealer control, tighter = breakout risk.<br>'
                    f'Zone and Delta can disagree: positive gamma at spot (dealers absorb small moves) '
                    f'with negative delta (dealers are net short overall) means calm near spot but directional pressure building underneath.'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

        # ── Lean Accuracy card ───────────────────────────────────────────────
        try:
            import json as _la_json, os as _la_os
            _la_path = _la_os.path.join(_la_os.path.dirname(_la_os.path.dirname(__file__)), "data", "lean_tracker.json")
            if _la_os.path.exists(_la_path):
                with open(_la_path, "r") as _la_f:
                    _la_hist = _la_json.load(_la_f)
                _la_stats = {}
                _la_regime_stats = {}
                for _la_e in _la_hist:
                    _la_l = _la_e.get("lean", "?")
                    if _la_l not in _la_stats:
                        _la_stats[_la_l] = {"5d": [], "20d": [], "count": 0}
                    _la_stats[_la_l]["count"] += 1
                    if _la_e.get("fwd_5d_spy_return") is not None:
                        _la_stats[_la_l]["5d"].append(_la_e["fwd_5d_spy_return"])
                    if _la_e.get("fwd_20d_spy_return") is not None:
                        _la_stats[_la_l]["20d"].append(_la_e["fwd_20d_spy_return"])
                    _la_hmm = _la_e.get("hmm_state", "?")
                    _la_rk = (_la_hmm, _la_l)
                    if _la_rk not in _la_regime_stats:
                        _la_regime_stats[_la_rk] = {"5d": [], "count": 0}
                    _la_regime_stats[_la_rk]["count"] += 1
                    if _la_e.get("fwd_5d_spy_return") is not None:
                        _la_regime_stats[_la_rk]["5d"].append(_la_e["fwd_5d_spy_return"])

                _la_streak = 0
                _la_cur_lean = _la_hist[-1].get("lean", "?") if _la_hist else "?"
                for _la_e2 in reversed(_la_hist):
                    if _la_e2.get("lean") == _la_cur_lean:
                        _la_streak += 1
                    else:
                        break
                _la_cur_davg = _la_hist[-1].get("domain_avg", 50) if _la_hist else 50
                _la_streak_col = "#22c55e" if _la_cur_lean == "BULLISH" else ("#ef4444" if _la_cur_lean == "BEARISH" else "#f59e0b")

                _la_has_data = any(s["5d"] for s in _la_stats.values())

                _la_rows = ""
                for _la_dir in ["BULLISH", "NEUTRAL", "BEARISH"]:
                    _la_s = _la_stats.get(_la_dir, {"5d": [], "20d": [], "count": 0})
                    _la_n = _la_s["count"]
                    if _la_n == 0:
                        continue
                    _la_col = "#22c55e" if _la_dir == "BULLISH" else ("#ef4444" if _la_dir == "BEARISH" else "#f59e0b")
                    if _la_s["5d"]:
                        _la_5avg = sum(_la_s["5d"]) / len(_la_s["5d"])
                        _la_5col = "#22c55e" if _la_5avg >= 0 else "#ef4444"
                        _la_5str = f'<span style="color:{_la_5col};font-weight:700;">{_la_5avg:+.2f}%</span>'
                    else:
                        _la_5str = '<span style="color:#334155;">—</span>'
                    if _la_s["20d"]:
                        _la_20avg = sum(_la_s["20d"]) / len(_la_s["20d"])
                        _la_20col = "#22c55e" if _la_20avg >= 0 else "#ef4444"
                        _la_20str = f'<span style="color:{_la_20col};font-weight:700;">{_la_20avg:+.2f}%</span>'
                    else:
                        _la_20str = '<span style="color:#334155;">—</span>'
                    _la_rows += (
                        f'<tr>'
                        f'<td style="padding:2px 6px;color:{_la_col};font-weight:700;">{_la_dir}</td>'
                        f'<td style="padding:2px 6px;text-align:center;">{_la_5str}</td>'
                        f'<td style="padding:2px 6px;text-align:center;">{_la_20str}</td>'
                        f'<td style="padding:2px 6px;text-align:center;color:#94a3b8;">{_la_n}</td>'
                        f'</tr>'
                    )

                _la_regime_rows = ""
                for (_la_rh, _la_rl), _la_rs in sorted(_la_regime_stats.items()):
                    if len(_la_rs["5d"]) >= 3:
                        _la_r5 = sum(_la_rs["5d"]) / len(_la_rs["5d"])
                        _la_r5c = "#22c55e" if _la_r5 >= 0 else "#ef4444"
                        _la_rlc = "#22c55e" if _la_rl == "BULLISH" else ("#ef4444" if _la_rl == "BEARISH" else "#f59e0b")
                        _la_regime_rows += (
                            f'<tr>'
                            f'<td style="padding:1px 6px;font-size:8px;color:#64748b;">{_la_rh}</td>'
                            f'<td style="padding:1px 6px;font-size:8px;color:{_la_rlc};">{_la_rl}</td>'
                            f'<td style="padding:1px 6px;font-size:8px;color:{_la_r5c};font-weight:700;">{_la_r5:+.2f}%</td>'
                            f'<td style="padding:1px 6px;font-size:8px;color:#94a3b8;">n={len(_la_rs["5d"])}</td>'
                            f'</tr>'
                        )

                _la_regime_html = ""
                if _la_regime_rows:
                    _la_regime_html = (
                        f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #1e293b;">'
                        f'<div style="font-size:8px;color:#3b4f6b;font-weight:700;letter-spacing:0.08em;margin-bottom:3px;">BY REGIME × LEAN (5d avg)</div>'
                        f'<table style="width:100%;font-size:9px;color:#64748b;border-collapse:collapse;">'
                        f'{_la_regime_rows}</table></div>'
                    )

                _la_status = "" if _la_has_data else (
                    '<div style="font-size:8px;color:#334155;margin-top:4px;">'
                    'accumulating data... forward returns appear after 5+ daily QIR runs</div>'
                )

                _lean_card = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-left:3px solid {_la_streak_col};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;">LEAN ACCURACY</div>'
                    f'<div style="font-size:10px;color:{_la_streak_col};font-weight:700;">'
                    f'{_la_cur_lean} × {_la_streak}d | avg {_la_cur_davg:.0f}</div>'
                    f'</div>'
                    f'<table style="width:100%;font-size:9px;color:#64748b;border-collapse:collapse;">'
                    f'<tr style="border-bottom:1px solid #1e293b;">'
                    f'<th style="text-align:left;padding:2px 6px;font-size:8px;">Lean</th>'
                    f'<th style="text-align:center;padding:2px 6px;font-size:8px;">Avg 5d</th>'
                    f'<th style="text-align:center;padding:2px 6px;font-size:8px;">Avg 20d</th>'
                    f'<th style="text-align:center;padding:2px 6px;font-size:8px;">Count</th>'
                    f'</tr>{_la_rows}</table>'
                    f'{_la_regime_html}'
                    f'{_la_status}'
                    f'</div>'
                )
        except Exception:
            pass

    else:
        # No fresh QIR run — show stale data if available, or minimal placeholder
        _stale_syn = st.session_state.get("_macro_synopsis") or {}
        if _stale_syn.get("conviction"):
            _sc = _stale_syn["conviction"]
            _sc_color = {"BULLISH": "#22c55e44", "BEARISH": "#ef444444", "MIXED": "#f59e0b44", "UNCERTAIN": "#94a3b844"}.get(_sc, "#94a3b844")
            _sc_text  = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "MIXED": "#f59e0b", "UNCERTAIN": "#94a3b8"}.get(_sc, "#94a3b8")
            _verdict_html = (
                f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
                f'<div style="background:{_sc_color};border-radius:4px;padding:6px 10px;margin-bottom:6px;">'
                f'<span style="color:{_sc_text};font-weight:700;font-size:11px;">{_sc}</span>'
                f'<span style="color:#475569;font-size:9px;margin-left:8px;">(last run — re-run QIR to refresh)</span>'
                f'</div>'
                f'<div style="color:#64748b;font-size:11px;line-height:1.5;">{_stale_syn.get("summary","")}</div>'
            )
        else:
            _verdict_html = (
                f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
                f'<div style="border:1px dashed #1e293b;border-radius:4px;padding:8px;text-align:center;">'
                f'<span style="color:#334155;font-size:11px;">Run QIR for conviction analysis</span>'
                f'</div>'
            )

    # ── Signal freshness row ──────────────────────────────────────────────
    _sig_checks = [
        ("Regime",    "_regime_context"),
        ("Tactical",  "_tactical_context"),
        ("Opt Flow",  "_options_flow_context"),
        ("Rate Path", "_dominant_rate_path"),
        ("Events",    "_current_events_digest"),
        ("Doom",      "_doom_briefing"),
        ("Whales",    "_whale_summary"),
        ("Activism",  "_activism_digest"),
        ("Risk Snap", "_portfolio_risk_snapshot"),
        ("Earnings",  "_qir_earnings_risk"),
        ("F&G Index", "_fear_greed"),
        ("AAII",      "_aaii_sentiment"),
        ("VIX Curve", "_vix_curve"),
    ]
    _fresh_count = sum(1 for _, k in _sig_checks if st.session_state.get(k))
    _total_sigs = len(_sig_checks)
    _frac = _fresh_count / _total_sigs
    _fc = "#22c55e" if _frac == 1.0 else ("#f59e0b" if _frac >= 0.5 else "#ef4444")
    _dots = "".join(
        f'<span style="color:{"#22c55e" if st.session_state.get(k) else "#1e293b"};font-size:9px;">●</span>'
        for _, k in _sig_checks
    )
    _freshness_html = (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
        f'<span style="color:{_fc};font-size:10px;font-weight:700;font-family:monospace;">'
        f'{_fresh_count}/{_total_sigs}</span>'
        f'<span style="letter-spacing:2px;">{_dots}</span>'
        f'<span style="color:#334155;font-size:10px;">'
        f'{"all signals loaded" if _fresh_count == _total_sigs else f"{_total_sigs - _fresh_count} signal(s) missing — run QIR"}'
        f'</span></div>'
    )

    # ── Compose zone 2: slow signals (Kelly+HMM+GEX+lean+signal) — top_bottom rendered separately ──
    # ── SLOW: just the HMM regime brain (LL + entropy + forecasts) ──────────
    _slow_html = _hmm_block if _populated else ""

    # ── MEDIUM: structural Kelly sizing + contextual signals + entry card ────
    _medium_html = ""
    if _populated:
        _medium_parts = []
        if _conviction_block:
            _medium_parts.append(_conviction_block)
        if _kelly_block:
            _medium_parts.append(f'<div style="margin-bottom:10px;">{_kelly_block}</div>')
        _medium_parts.append(_gex_block + _lean_card + _signal_breakdown_block)
        _medium_html = "".join(_medium_parts)

    # ── Crash pattern alert ──────────────────────────────────────────────
    _crash_alert_html = ""
    if _populated:
        _crash_matches = _check_crash_patterns(_rc)
        if _crash_matches:
            _alert_parts = []
            for _cm in _crash_matches:
                _sev_color = "#ef4444" if _cm["match_pct"] >= 80 else ("#f59e0b" if _cm["match_pct"] >= 60 else "#f97316")
                _icon = "🚨" if _cm["match_pct"] >= 80 else "⚠"
                _detail_str = " · ".join(_cm["details"][:3])
                _alert_parts.append(
                    f'<div style="padding:4px 0;border-bottom:1px solid #1e293b22;">'
                    f'<span style="color:{_sev_color};font-weight:800;font-size:11px;">'
                    f'{_icon} {_cm["name"]} — {_cm["match_pct"]}% match</span>'
                    f'<span style="color:#64748b;font-size:9px;margin-left:8px;">'
                    f'({_cm["matched"]}/{_cm["total"]} conditions) · {_cm["lead_days"]}d lead · '
                    f'{_cm["max_drawdown"]}% drawdown</span>'
                    f'<div style="font-size:8px;color:#475569;padding-left:20px;">{_detail_str}</div>'
                    f'</div>'
                )
            _crash_alert_html = (
                f'<div style="background:#1a0a0a;border:1px solid #ef444433;'
                f'border-radius:5px;padding:8px 12px;margin-bottom:10px;">'
                f'<div style="font-size:9px;color:#ef4444;font-weight:700;'
                f'letter-spacing:0.1em;margin-bottom:4px;">CRASH PATTERN ALERT</div>'
                f'{"".join(_alert_parts)}'
                f'</div>'
            )

    # ── Render the full dashboard — ordered SLOW → MEDIUM → FAST ─────────────
    def _tf_divider(label: str, color: str = "#1e293b") -> str:
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;">'
            f'<div style="flex:1;height:1px;background:{color};"></div>'
            f'<span style="font-size:7px;color:#334155;font-weight:700;letter-spacing:0.12em;'
            f'white-space:nowrap;">{label}</span>'
            f'<div style="flex:1;height:1px;background:{color};"></div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid {_border_color};border-radius:8px;'
        f'box-shadow:{_border_glow};padding:14px 16px;margin:8px 0 12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;'
        f'text-transform:uppercase;">QIR Intelligence Dashboard</div>'
        f'</div>'
        f'{_freshness_html}'
        f'{_crash_alert_html}'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
        f'<div>{_t1}</div><div>{_t2}</div><div>{_t3}</div>'
        f'</div>'
        # ── SLOW: Entry Signal (macro+leading+options aggregator) then HMM brain ─
        f'{_tf_divider("⏱  SLOW — REGIME BRAIN · WEEKS / MONTHS")}'
        f'{_entry_rec_html}'
        f'{_slow_html}'
        # ── MEDIUM: structural Kelly sizing + contextual signals ───────────────
        f'{_tf_divider("⏑  MEDIUM — STRUCTURAL SIZING · DAYS / WEEKS")}'
        f'{_medium_html}'
        f'{_verdict_html}'
        f'{_top_bottom_block if _populated else ""}'
        # ── FAST: Buy/Short Setups (Triple Kelly, lean-dependent) ─────────────
        f'{_tf_divider("⚡  FAST — TACTICAL SETUPS · HOURS / DAYS")}'
        f'{_velocity_block if _populated else ""}'
        f'{_fast_setups_html if _populated else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Genuine Uncertainty expansion panel ───────────────────────────────
    if _populated and _cls["pattern"] == "GENUINE_UNCERTAINTY" and _gu_profile:
        _render_genuine_uncertainty_panel(_gu_profile)

    # ── Inline log button for QIR macro verdict ───────────────────────────
    if _populated and _verdict_label:
        from modules.forecast_accuracy import render_log_button
        from services.forecast_tracker import log_forecast

        _qir_conf = int(min(95, max(40, (_tac_score + _of_score) / 2)))
        _qir_summary = (
            f"Regime: {_regime_label} ({_regime_score:+.2f}) | "
            f"Tactical: {_tac.get('label','')} ({_tac_score}/100) | "
            f"Opt Flow: {_of.get('label','')} ({_of_score}/100)"
        )

        # Map verdict to SPY direction — GENUINE_UNCERTAINTY uses the forced lean
        _buy_verdicts  = {"BULLISH CONFIRMATION", "PULLBACK IN UPTREND", "OPTIONS FLOW DIVERGENCE", "BEAR MARKET BOUNCE"}
        _sell_verdicts = {"BEARISH CONFIRMATION", "LATE CYCLE SQUEEZE"}
        if _cls["pattern"] == "GENUINE_UNCERTAINTY":
            _gu_lean = _gu_profile["lean"]
            _spy_prediction = "Buy" if _gu_lean == "BULLISH" else "Sell"
            _qir_conf = max(40, min(60, _gu_profile["lean_pct"]))
            _qir_summary += f" | Uncertainty lean: {_gu_lean} {_gu_profile['lean_pct']}% | {_gu_profile['size_label']}"
        else:
            _spy_prediction = (
                "Buy"  if _verdict_label in _buy_verdicts  else
                "Sell" if _verdict_label in _sell_verdicts else
                None
            )

        _qc1, _qc2, _qc3 = st.columns([3, 1, 1])
        with _qc2:
            render_log_button(
                signal_type="regime",
                prediction=_verdict_label,
                confidence=_qir_conf,
                summary=_qir_summary,
                model="QIR Composite",
                horizon_days=21,
                key=f"qir_log_{_verdict_label}_{_tac_score}",
                label="📌 Log Signal",
            )
        with _qc3:
            if _spy_prediction:
                _spy_label = "📈 Trade SPY Long" if _spy_prediction == "Buy" else "📉 Trade SPY Short"
                if _cls["pattern"] == "GENUINE_UNCERTAINTY":
                    _spy_label = ("📈 Lean Long (small)" if _spy_prediction == "Buy"
                                  else "📉 Lean Short (small)")
                if st.button(_spy_label, key=f"qir_spy_{_verdict_label}_{_tac_score}", use_container_width=True,
                             help="Log this QIR verdict as a SPY ATR trade in Forecast Tracker"):
                    _fid = log_forecast(
                        signal_type="valuation",
                        prediction=_spy_prediction,
                        confidence=_qir_conf,
                        summary=f"QIR verdict: {_verdict_label} | {_qir_summary}",
                        model="QIR Composite",
                        ticker="SPY",
                    )
                    st.toast(f"📌 SPY {_spy_prediction} logged! [{_fid}] — ATR trailing stop active", icon="✅")
            else:
                st.button("🚫 No SPY Trade", key=f"qir_spy_none_{_tac_score}", use_container_width=True,
                          disabled=True, help=f"{_verdict_label} — no clear directional edge for SPY")

    # ── Call Top / Call Bottom manual log buttons ─────────────────────────
    _tb_prox = st.session_state.get("_top_bottom_proximity")
    if _populated and _tb_prox:
        _tb_col1, _tb_col2 = st.columns(2)
        with _tb_col1:
            if _tb_prox["top_pct"] >= 20:
                with st.popover(f"📉 Call Market Top ({_tb_prox['top_pct']}%)", use_container_width=True):
                    st.markdown(
                        f'<div style="font-size:11px;color:#ef4444;font-weight:700;">CALLING MARKET TOP</div>'
                        f'<div style="font-size:10px;color:#94a3b8;margin:4px 0;">Top proximity: {_tb_prox["top_pct"]}%</div>'
                        f'<div style="font-size:9px;color:#64748b;">Signals: {" · ".join(_tb_prox["top_signals"])}</div>',
                        unsafe_allow_html=True,
                    )
                    _tb_top_conf = st.slider("Confidence", 30, 95, min(80, _tb_prox["top_pct"]), key="tb_top_conf")
                    _tb_top_notes = st.text_input("Notes", key="tb_top_notes", placeholder="Why I think this is a top...")
                    if st.button("Confirm Call Top", key="tb_top_confirm", type="primary", use_container_width=True):
                        from services.forecast_tracker import log_forecast as _tb_log
                        _tb_log(
                            signal_type="valuation",
                            prediction="Sell",
                            confidence=_tb_top_conf,
                            summary=f"MARKET TOP CALL | Top proximity: {_tb_prox['top_pct']}% | "
                                    f"Signals: {', '.join(_tb_prox['top_signals'])} | {_tb_top_notes}",
                            model="Top/Bottom Proximity",
                            ticker="SPY",
                            horizon_days=60,
                        )
                        st.toast(f"Market Top call logged at {_tb_prox['top_pct']}% proximity!", icon="📉")
        with _tb_col2:
            if _tb_prox["bottom_pct"] >= 20:
                with st.popover(f"📈 Call Market Bottom ({_tb_prox['bottom_pct']}%)", use_container_width=True):
                    st.markdown(
                        f'<div style="font-size:11px;color:#22c55e;font-weight:700;">CALLING MARKET BOTTOM</div>'
                        f'<div style="font-size:10px;color:#94a3b8;margin:4px 0;">Bottom proximity: {_tb_prox["bottom_pct"]}%</div>'
                        f'<div style="font-size:9px;color:#64748b;">Signals: {" · ".join(_tb_prox["bottom_signals"])}</div>',
                        unsafe_allow_html=True,
                    )
                    _tb_bot_conf = st.slider("Confidence", 30, 95, min(80, _tb_prox["bottom_pct"]), key="tb_bot_conf")
                    _tb_bot_notes = st.text_input("Notes", key="tb_bot_notes", placeholder="Why I think this is a bottom...")
                    if st.button("Confirm Call Bottom", key="tb_bot_confirm", type="primary", use_container_width=True):
                        from services.forecast_tracker import log_forecast as _tb_log2
                        _tb_log2(
                            signal_type="valuation",
                            prediction="Buy",
                            confidence=_tb_bot_conf,
                            summary=f"MARKET BOTTOM CALL | Bottom proximity: {_tb_prox['bottom_pct']}% | "
                                    f"Signals: {', '.join(_tb_prox['bottom_signals'])} | {_tb_bot_notes}",
                            model="Top/Bottom Proximity",
                            ticker="SPY",
                            horizon_days=60,
                        )
                        st.toast(f"Market Bottom call logged at {_tb_prox['bottom_pct']}% proximity!", icon="📈")


def render():
    _oc = COLORS["bloomberg_orange"]

    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:2px;">⚡ QUICK INTEL RUN</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Runs Risk Regime + Fed Rate Path + Policy Transmission + Current Events + Doom Briefing + Whale Activity + Black Swans + Macro Synopsis + Portfolio Risk Snapshot in sequence. "
        "Navigate to Portfolio Intelligence when done."
    )

    render_rr_score_mode_toggle(
        key="qir_rr_score_mode_ui",
        help_text="Affects the Risk Regime scoring used in this run.",
    )
    render_intel_health_bar()

    # ── Reading Guide ──────────────────────────────────────────────────────
    with st.expander("📖 How to read QIR", expanded=False):
        st.markdown(
            '<div style="font-size:11px;color:#475569;'
            'font-family:\'JetBrains Mono\',Consolas,monospace;line-height:2.0;">'

            '<span style="color:#334155;font-weight:700;letter-spacing:0.05em;">REGIME SCORE (0–100)</span><br>'
            'Weighted average of 26 macro signals z-scored against their own history. '
            '&lt;40 = Risk-Off · 40–60 = Neutral · &gt;60 = Risk-On.<br><br>'

            '<span style="color:#334155;font-weight:700;letter-spacing:0.05em;">LEADING SUB-SCORE</span><br>'
            'Fast signals only (VIX, credit spreads, LEI, credit impulse, real yields, etc.). '
            'Leads the composite by weeks to months.<br>'
            'Divergence +8 = fast signals already turning Risk-On → buy the dip before the flip<br>'
            'Divergence −8 = fast signals cracking → reduce before the composite turns red<br><br>'

            '<span style="color:#334155;font-weight:700;letter-spacing:0.05em;">5-SESSION TREND  C · L</span><br>'
            'Composite (C) and Leading (L) 5-day score change. '
            'Both rising = high conviction bull. Both falling = high conviction bear. '
            'Diverging = wait for alignment.<br><br>'

            '<span style="color:#334155;font-weight:700;letter-spacing:0.05em;">PATTERNS</span><br>'
            'BULLISH CONFIRMATION — all 3 layers aligned bull → strong long entry<br>'
            'PULLBACK IN UPTREND — regime + options bull, tactical dipping → buy the dip<br>'
            'BEAR MARKET BOUNCE — regime bear, tactical/options bull → fade the bounce<br>'
            'BEARISH CONFIRMATION — all 3 layers aligned bear → defense / short environment<br>'
            'GENUINE UNCERTAINTY — layers disagree → reduce size, wait for clarity<br><br>'

            '<span style="color:#334155;font-weight:700;letter-spacing:0.05em;">JUDGE JUDY DEBATE</span><br>'
            'Sir Doomburger 🐻 vs Sir Fukyerputs 🐂 argue the same ground truth numbers. '
            'Leading divergence and 5-session trend are explicit evidence in the debate — '
            'a large leading divergence gives the bull/bear case a concrete factual basis to argue.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── VIX live ticker ────────────────────────────────────────────────────────
    try:
        import yfinance as _yf
        @st.cache_data(ttl=120)
        def _fetch_vix_change():
            _v = _yf.Ticker("^VIX")
            _h = _v.history(period="2d", interval="1d")
            if _h is None or len(_h) < 2:
                return None, None, None
            _prev = float(_h["Close"].iloc[-2])
            _cur  = float(_h["Close"].iloc[-1])
            _chg  = _cur - _prev
            _pct  = (_chg / _prev) * 100
            return round(_cur, 2), round(_chg, 2), round(_pct, 2)

        _vix_cur, _vix_chg, _vix_pct = _fetch_vix_change()
        if _vix_cur is not None:
            if _vix_pct >= 10:
                _vc = "#ef4444"; _vglow = "0 0 10px #ef444488"; _vbg = "#2d0a0a"; _varrow = "▲"
            elif _vix_pct >= 3:
                _vc = "#f97316"; _vglow = "0 0 8px #f9731666"; _vbg = "#1f1000"; _varrow = "▲"
            elif _vix_pct > 0:
                _vc = "#f59e0b"; _vglow = "none"; _vbg = "#1a1200"; _varrow = "▲"
            elif _vix_pct <= -10:
                _vc = "#22c55e"; _vglow = "0 0 10px #22c55e88"; _vbg = "#052e16"; _varrow = "▼"
            elif _vix_pct <= -3:
                _vc = "#22c55e"; _vglow = "0 0 8px #22c55e55"; _vbg = "#0c1a0c"; _varrow = "▼"
            elif _vix_pct < 0:
                _vc = "#4ade80"; _vglow = "none"; _vbg = "#0c1a0c"; _varrow = "▼"
            else:
                _vc = "#475569"; _vglow = "none"; _vbg = "#0d1117"; _varrow = "◆"
            _vsign = "+" if _vix_chg >= 0 else ""
            st.markdown(
                f'<div style="background:{_vbg};border:1px solid {_vc}55;border-radius:6px;'
                f'box-shadow:{_vglow};padding:7px 14px;margin-bottom:8px;'
                f'display:inline-flex;align-items:center;gap:10px;">'
                f'<span style="color:#475569;font-size:10px;font-weight:700;letter-spacing:0.1em;">VIX</span>'
                f'<span style="color:{_vc};font-size:18px;font-weight:800;font-family:monospace;'
                f'text-shadow:{_vglow};">{_vix_cur}</span>'
                f'<span style="color:{_vc};font-size:12px;font-weight:700;">'
                f'{_varrow} {_vsign}{_vix_chg} ({_vsign}{_vix_pct}%)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

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

    _rec_map = {
        "⚡ Freeloader Mode":      "Daily routine — all 8 modules in ~90s, completely free.",
        "🧠 Regard Mode":            "Active day — Grok 4.1 reasoning for deeper synthesis + live X feed in Current Events.",
        "👑 Highly Regarded Mode":   "High conviction — Sonnet on all 8 modules before running Portfolio.",
    }
    _sel = st.radio("Engine", TIER_OPTS, horizontal=True, key="qr_engine")
    st.markdown(MODEL_HINT_HTML, unsafe_allow_html=True)
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
    _use_claude, _cl_model = TIER_MAP[_sel]
    if _sel == "👑 Highly Regarded Mode":
        st.warning("👑 Highly Regarded uses Claude Sonnet — reserve for elevated volatility or high-conviction sessions.")
        _confirmed = st.checkbox("I confirm this is a high-conviction session", key="qr_hr_confirm")
        if not _confirmed:
            _use_claude, _cl_model = True, "grok-4-1-fast-reasoning"
            st.caption("*Running in Regard Mode until confirmed.*")

    # ── Signal readiness ───────────────────────────────────────────────────────
    _signal_keys = ["_regime_context", "_tactical_context", "_options_flow_context", "_dominant_rate_path", "_rp_plays_result", "_fed_plays_result", "_current_events_digest", "_doom_briefing", "_chain_narration", "_custom_swans", "_whale_summary", "_activism_digest", "_sector_regime_digest", "_macro_synopsis", "_adversarial_debate", "_portfolio_risk_snapshot", "_stocktwits_digest", "_fear_greed", "_aaii_sentiment", "_vix_curve"]
    _signal_labels = ["Regime", "Tactical", "Opt Flow", "Fed Rate Path", "Rate-Path Plays", "Fed Plays", "News Digest", "Doom Briefing", "Policy Trans.", "Black Swans", "Whale Activity", "Activism", "Sector×Regime", "Macro Synopsis", "Debate", "Risk Snapshot", "Social Sentiment", "F&G Index", "AAII Sentiment", "VIX Curve"]
    _populated = [(k, l) for k, l in zip(_signal_keys, _signal_labels) if st.session_state.get(k)]

    if _populated:
        _badges = " &nbsp;".join(
            f'<span style="background:#052e16;color:#22c55e;border:1px solid #22c55e44;'
            f'border-radius:3px;padding:1px 7px;font-size:10px;">{l}</span>'
            for _, l in _populated
        )
        st.markdown(
            f'<div style="margin:6px 0 6px 0;">{_badges}</div>',
            unsafe_allow_html=True,
        )
        _dq_persistent = st.session_state.get("_data_quality") or {}
        if _dq_persistent:
            _dqp_col = "#22c55e" if _dq_persistent["score"] >= 80 else ("#f59e0b" if _dq_persistent["score"] >= 60 else "#ef4444")
            _dqp_bg  = "#052e16" if _dq_persistent["score"] >= 80 else ("#1a1200" if _dq_persistent["score"] >= 60 else "#2d0a0a")
            _dqp_label = _dq_persistent.get("label", "")
            _dqp_stale = _dq_persistent.get("stale_market", []) + _dq_persistent.get("stale_fred", [])
            _dqp_detail = f" · Stale: {', '.join(_dqp_stale[:4])}" if _dqp_stale else ""
            st.markdown(
                f'<div style="background:{_dqp_bg};border:1px solid {_dqp_col}66;border-radius:5px;'
                f'padding:7px 14px;margin-bottom:6px;display:flex;align-items:center;gap:12px;">'
                f'<span style="color:{_dqp_col};font-weight:800;font-size:15px;font-family:monospace;">'
                f'{_dq_persistent["score"]}/100</span>'
                f'<span style="color:{_dqp_col};font-size:11px;font-weight:700;letter-spacing:0.06em;">DATA QUALITY</span>'
                f'<span style="color:#94a3b8;font-size:11px;">{_dqp_label}{_dqp_detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _of_persistent = st.session_state.get("_options_flow_context") or {}
        if _of_persistent:
            _ofp_score = _of_persistent.get("options_score", 50)
            _ofp_col = "#22c55e" if _ofp_score >= 65 else ("#f59e0b" if _ofp_score >= 38 else "#ef4444")
            _ofp_bg  = "#052e16" if _ofp_score >= 65 else ("#1a1200" if _ofp_score >= 38 else "#2d0a0a")
            st.markdown(
                f'<div style="background:{_ofp_bg};border:1px solid {_ofp_col}66;border-radius:5px;'
                f'padding:7px 14px;margin-bottom:10px;display:flex;align-items:center;gap:12px;">'
                f'<span style="color:{_ofp_col};font-weight:800;font-size:15px;font-family:monospace;">'
                f'{_ofp_score}/100</span>'
                f'<span style="color:{_ofp_col};font-size:11px;font-weight:700;letter-spacing:0.06em;">OPTIONS FLOW</span>'
                f'<span style="color:#94a3b8;font-size:11px;">{_of_persistent.get("label","")} — {_of_persistent.get("action_bias","")[:60]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Score Interpretation Reference ─────────────────────────────────────────
    with st.expander("📖 How to read Tactical + Options Flow scores", expanded=False):
        st.markdown("""
**Both scores run 0–100. Higher = more bullish conditions.**

---

### Tactical Score (days/weeks layer)
Measures near-term market internals: momentum, breadth, credit spreads, bond yields, volatility.

| Score | Label | What to do |
|-------|-------|------------|
| ≥65 | Favorable Entry | Conditions support adding risk |
| 52–64 | Neutral | Hold existing positions, wait for clearer signal |
| 38–51 | Caution | Reduce new buys, tighten stops |
| <38 | Risk-Off | Defensive posture — consider exits or hedges |

---

### Options Flow Score (hours/days layer)
Measures what SPY options participants are doing *right now*: put/call ratio, gamma positioning, IV skew, and unusual flow.

| Score | Label | What it means |
|-------|-------|---------------|
| ≥65 | Call-Skewed Flow | Crowd positioned bullish — calls dominant, low fear skew |
| 52–64 | Neutral Flow | Mixed positioning, no strong lean |
| 38–51 | Put-Skewed Flow | Elevated hedging, cautious tone |
| <38 | Bearish Hedging | Heavy put buying, fear premium elevated |

**Sub-signals:**
- **P/C Ratio** (weight 3×) — Put vol ÷ call vol. Below 0.9 = bullish crowd. Above 1.2 = hedging.
- **Gamma Zone** (1.5×) — Positive gamma stabilizes prices (dealers fade moves). Negative gamma amplifies them (dealers chase).
- **IV Skew** (2×) — OTM put IV ÷ OTM call IV. Skew > 1.4 = expensive tail hedges = fear.
- **Unusual Activity Bias** (1.5×) — Vol/OI > 2× = unusual. Call-heavy unusual flow = bullish. Put-heavy = defensive.

---

### Reading them together

| Tactical | Opt Flow | Interpretation |
|----------|----------|----------------|
| ≥65 | ≥65 | **Strong alignment — highest conviction long entry** |
| ≥65 | 38–64 | Tactical setup good, crowd cautious — enter smaller |
| 38–64 | ≥65 | Options bullish but macro not yet confirmed — wait |
| <38 | <38 | **Avoid new longs — both layers warn** |
| <38 | ≥65 | Crowd complacent in a weak tape — possible contrarian short setup |

> **Rule of thumb:** Tactical tells you *when* to enter. Options Flow tells you *what the crowd is doing today*. Alignment between both = higher conviction. Divergence = reduce size or wait.
        """)

    # ── Run Options ────────────────────────────────────────────────────────────
    from utils.theme import COLORS as _QR_COLORS
    st.markdown(
        f'<div style="background:{_QR_COLORS["surface"]};border:1px solid {_QR_COLORS["border"]};'
        f'border-radius:6px;padding:10px 14px;margin-bottom:10px;">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:0.08em;'
        f'color:{_QR_COLORS["bloomberg_orange"]};margin-bottom:8px;">RUN OPTIONS</div>',
        unsafe_allow_html=True,
    )
    _qr_opt_c1, _qr_opt_c2 = st.columns(2)
    _include_swans = _qr_opt_c1.radio(
        "Black Swan Analysis",
        ["Skip (faster)", "Include — feeds Discovery, Valuation & Portfolio Intel"],
        index=st.session_state.get("_qir_include_swans_idx", 0),
        key="qir_swans_opt",
        help="When included, 3 regime-relevant black swan scenarios are generated and injected as context into downstream consumers.",
    )
    _qr_opt_c2.markdown(
        f'<div style="font-size:11px;color:{_QR_COLORS["text_dim"]};padding-top:6px;">'
        f'{"⚡ Faster run — skip tail risk generation" if "Skip" in _include_swans else "🦢 Tail risk scenarios generated for Discovery · Valuation · Portfolio Intel"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.session_state["_qir_include_swans_idx"] = 0 if "Skip" in _include_swans else 1
    st.markdown('</div>', unsafe_allow_html=True)
    _run_black_swans = "Include" in _include_swans

    # ── Run button ─────────────────────────────────────────────────────────────
    if st.button("⚡ RUN ALL INTEL MODULES", type="primary", key="qr_run_all", use_container_width=True):

        # Clear ALL cached data so every module fetches fresh on this run
        st.cache_data.clear()

        _results = {}
        _macro_ctx, _fred_data = {}, {}

        # ── Round 1 (parallel): Regime + Current Events + Whale + Options Flow + StockTwits ──
        # These five are fully independent — run concurrently to save time.
        from modules.risk_regime import run_quick_regime, run_quick_sector_regime
        from modules.current_events import run_quick_digest
        from modules.whale_buyers import run_quick_whale, run_quick_activism
        from modules.options_activity import run_quick_options_flow
        from services.stocktwits_client import run_quick_stocktwits
        from services.free_data import fetch_sentiment_snapshot

        def _run_sentiment() -> dict:
            """Fetch Fear & Greed + AAII + VIX curve in parallel."""
            snap = fetch_sentiment_snapshot()
            return {
                "_fear_greed":       snap.get("fear_greed"),
                "_aaii_sentiment":   snap.get("aaii"),
                "_vix_curve":        snap.get("vix_curve"),
                "_fedspeech":        snap.get("fedspeech") or [],
            }

        with st.spinner("📡 Round 1/4 — Regime · Current Events · Whale · Activism · Options Flow · Sector×Regime · Social · Sentiment (parallel)..."):
            _r1_errors = {}
            _macro_ctx, _fred_data = None, None
            _regime_ctx = None  # initialized here; set inside loop when regime future resolves
            import datetime as _dt_qir
            with ThreadPoolExecutor(max_workers=8) as _pool:
                _fut_regime   = _pool.submit(run_quick_regime, _use_claude, _cl_model)
                _fut_digest   = _pool.submit(run_quick_digest, _use_claude, _cl_model)
                _fut_whale    = _pool.submit(run_quick_whale,  _use_claude, _cl_model)
                _fut_activism = _pool.submit(run_quick_activism, _use_claude, _cl_model)
                _fut_opts     = _pool.submit(run_quick_options_flow, _use_claude, _cl_model)
                _fut_stwit    = _pool.submit(run_quick_stocktwits)
                _fut_sentiment = _pool.submit(_run_sentiment)
                # SPX GEX — institutional dealer book (background, no st.* calls)
                def _fetch_spx_gex():
                    try:
                        from services.market_data import fetch_gex_profile as _fgp
                        _fn = getattr(_fgp, "__wrapped__", _fgp)
                        return _fn("^SPX", 4)
                    except Exception:
                        return None
                _fut_gex_spx  = _pool.submit(_fetch_spx_gex)
                for _fut, _key in (
                    (_fut_regime,     "regime"),
                    (_fut_digest,     "digest"),
                    (_fut_whale,      "whale"),
                    (_fut_activism,   "activism"),
                    (_fut_opts,       "opts"),
                    (_fut_stwit,      "social"),
                    (_fut_sentiment,  "sentiment"),
                    (_fut_gex_spx,    "gex_spx"),
                ):
                    try:
                        _val = _fut.result()
                        if _key == "regime" and _val:
                            _macro_ctx, _fred_data, _tac_data, _tac_text, _regime_ctx, _plays, _tier, _dq, _raw_sigs = _val
                            # Write ALL regime data from main thread
                            st.session_state["_regime_context"] = _regime_ctx
                            st.session_state["_regime_context_ts"] = _dt_qir.datetime.now()
                            st.session_state["_regime_raw_signals"] = _raw_sigs
                            st.session_state["_rp_plays_result"] = _plays
                            st.session_state["_rp_plays_last_tier"] = _tier
                            if _tac_data:
                                st.session_state["_tactical_context"] = _tac_data
                                st.session_state["_tactical_context_ts"] = _dt_qir.datetime.now()
                            if _tac_text:
                                st.session_state["_tactical_analysis"] = _tac_text
                                st.session_state["_tactical_analysis_ts"] = _dt_qir.datetime.now()
                            if _dq:
                                st.session_state["_data_quality"] = _dq
                                st.session_state["_data_quality_ts"] = _dt_qir.datetime.now()
                                # Persist to alerts_config so background worker and alert checks can read it
                                try:
                                    from utils.alerts_config import load_config as _lc, save_config as _sc
                                    _ac = _lc()
                                    _ac["current_data_quality_score"] = _dq["score"]
                                    _ac["last_tactical_score"] = _ac.get("current_tactical_score")
                                    if _tac_data:
                                        _ac["current_tactical_score"] = _tac_data["tactical_score"]
                                    _sc(_ac)
                                except Exception:
                                    pass
                        elif _key == "digest" and _val:
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        elif _key == "whale" and _val:
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        elif _key == "activism" and _val:
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        elif _key == "opts" and _val:
                            st.session_state["_options_flow_context"] = _val
                            st.session_state["_options_flow_context_ts"] = _dt_qir.datetime.now()
                        elif _key == "social" and _val:
                            st.session_state["_stocktwits_digest"] = _val
                            st.session_state["_stocktwits_digest_ts"] = _dt_qir.datetime.now()
                        elif _key == "sentiment" and _val:
                            import datetime as _sdt
                            _now_s = _sdt.datetime.now()
                            for _sk in ("_fear_greed", "_aaii_sentiment", "_vix_curve", "_fedspeech"):
                                if _val.get(_sk) is not None:
                                    st.session_state[_sk] = _val[_sk]
                                    st.session_state[_sk + "_ts"] = _now_s
                        elif _key == "gex_spx" and _val:
                            st.session_state["_gex_profile_spx"] = _val
                            st.session_state["_gex_profile_spx_ts"] = _dt_qir.datetime.now()
                        _results[_key] = bool(_val)
                    except Exception as _e:
                        _results[_key] = False
                        _r1_errors[_key] = str(_e)

            # Sector×Regime runs after regime resolves — call directly (needs regime_ctx)
            try:
                _val_s = run_quick_sector_regime(_use_claude, _cl_model, regime_ctx=_regime_ctx)
                if _val_s and "_sector_regime_error" in _val_s:
                    # Error dict returned — surface the message, don't write to session state
                    _results["sector"] = False
                    _r1_errors["sector"] = _val_s["_sector_regime_error"]
                elif _val_s and "_sector_regime_digest" in _val_s:
                    for _k, _v in _val_s.items():
                        st.session_state[_k] = _v
                    _results["sector"] = True
                else:
                    _results["sector"] = False
            except Exception as _e:
                _results["sector"] = False
                _r1_errors["sector"] = str(_e)

        _regime_ok = _results.get("regime", False)
        if _regime_ok:
            st.success("✅ Risk Regime + Rate-Path Plays — done")
        else:
            st.error(f"❌ Regime failed: {_r1_errors.get('regime', '?')}")
        if _results.get("digest"):
            _x_live = st.session_state.get("_x_feed_injected", False)
            _x_badge = " · 𝕏 Live Feed ✓" if _x_live else ""
            _ev_engine = st.session_state.get("_current_events_engine", "")
            _ev_engine_str = f" [{_ev_engine}]" if _ev_engine else ""
            st.success(f"✅ Current Events Digest{_ev_engine_str}{_x_badge}")
            _ev_conflict = st.session_state.get("_current_events_conflict", "")
            if _ev_conflict:
                st.warning(_ev_conflict)
        else:
            st.warning(f"⚠ Digest skipped — {_r1_errors.get('digest', 'no content available')}")
        if _results.get("whale"):
            st.success("✅ Whale Activity — done")
        else:
            st.warning(f"⚠ Whale scan: {_r1_errors.get('whale', 'no data returned')}")
        if _results.get("opts"):
            _of_r1 = st.session_state.get("_options_flow_context") or {}
            st.success(f"✅ Options Flow — {_of_r1.get('label','?')} ({_of_r1.get('options_score','?')}/100)")
        else:
            st.warning(f"⚠ Options Flow: {_r1_errors.get('opts', 'SPY chain unavailable')}")
        if _results.get("social"):
            _st_r1 = st.session_state.get("_stocktwits_digest") or {}
            _st_mood = _st_r1.get("market_mood", "?")
            _st_bull = _st_r1.get("overall_bull_pct", "?")
            _st_top = ", ".join(_st_r1.get("top_bullish", [])[:3])
            st.success(f"✅ Social Sentiment — {_st_mood} ({_st_bull}% bull) · trending: {_st_top}")
        else:
            st.warning(f"⚠ Social Sentiment: {_r1_errors.get('social', 'StockTwits unavailable')}")
        if _results.get("sentiment"):
            _fg_r1 = st.session_state.get("_fear_greed") or {}
            _vc_r1 = st.session_state.get("_vix_curve") or {}
            _aaii_r1 = st.session_state.get("_aaii_sentiment") or {}
            _fg_str = f"F&G {_fg_r1.get('score','?')} ({_fg_r1.get('label','?')})" if _fg_r1 else "F&G N/A"
            _vc_str = f"VIX curve: {_vc_r1.get('structure','?')}" if _vc_r1 else "VIX curve N/A"
            _aa_str = f"AAII: {_aaii_r1.get('label','?')} (spread {_aaii_r1.get('bull_bear_spread','?')}%)" if _aaii_r1 else "AAII N/A"
            st.success(f"✅ Market Sentiment — {_fg_str} · {_vc_str} · {_aa_str}")
        else:
            st.warning(f"⚠ Market Sentiment: {_r1_errors.get('sentiment', 'data unavailable')}")
        if _results.get("sector"):
            st.success("✅ Sector×Regime Digest — done")
        else:
            st.warning(f"⚠ Sector×Regime: {_r1_errors.get('sector', 'regime not ready or no sector data')}")

        # ── Round 2 (parallel): Fed + Doom + Black Swans ──────────────────────
        # Fed uses regime output from Round 1. Doom reads digest from session_state.
        # Black Swans are optional (controlled by _run_black_swans toggle).
        from modules.fed_forecaster import run_quick_fed, run_quick_swans
        from modules.stress_signals import run_quick_doom

        _r2_spinner = "📈 Round 2/4 — Fed Rate Path · Doom Briefing" + (" · Black Swans" if _run_black_swans else "") + " (parallel)..."
        with st.spinner(_r2_spinner):
            _r2_errors = {}
            with ThreadPoolExecutor(max_workers=3) as _pool2:
                _fut_fed  = _pool2.submit(run_quick_fed, _macro_ctx, _fred_data, _use_claude, _cl_model)
                _fut_doom = _pool2.submit(run_quick_doom, _use_claude, _cl_model)
                _r2_jobs  = [(_fut_fed, "fed"), (_fut_doom, "doom")]
                if _run_black_swans:
                    _fut_swans = _pool2.submit(run_quick_swans, _use_claude, _cl_model)
                    _r2_jobs.append((_fut_swans, "swans"))
                for _fut, _key in _r2_jobs:
                    try:
                        _val = _fut.result()
                        if _val and isinstance(_val, dict):
                            for _k, _v in _val.items():
                                st.session_state[_k] = _v
                        _results[_key] = bool(_val)
                    except Exception as _e:
                        _results[_key] = False
                        _r2_errors[_key] = str(_e)

        if _results.get("fed"):
            st.success("✅ Fed Rate Path — done")
        else:
            st.error(f"❌ Fed Rate Path failed: {_r2_errors.get('fed', '?')}")
        if _results.get("doom"):
            st.success("✅ Doom Briefing — done")
        else:
            st.error(f"❌ Doom Briefing failed: {_r2_errors.get('doom', '?')}")
        if _run_black_swans:
            if _results.get("swans"):
                _bs_count = len(st.session_state.get("_custom_swans", {}))
                st.success(f"✅ Black Swans — {_bs_count} scenarios ready → Discovery · Valuation · Portfolio Intel")
            else:
                st.warning(f"⚠ Black Swans: {_r2_errors.get('swans', 'no results')}")
        else:
            st.info("ℹ️ Black Swans skipped — enable in Run Options to feed downstream consumers")

        # ── Round 3: Policy Transmission (needs Fed output from Round 2) ───────
        with st.spinner("🔗 Round 3/4 — Policy transmission path..."):
            try:
                from modules.fed_forecaster import run_quick_chain
                ok = run_quick_chain(use_claude=_use_claude, model=_cl_model)
                _results["chain"] = ok
                if ok:
                    import datetime as _cdt
                    st.session_state["_chain_narration_ts"] = _cdt.datetime.now()
                    st.success("✅ Policy Transmission — done")
                else:
                    st.warning("⚠ Policy Transmission skipped — rate path not available")
            except Exception as e:
                _results["chain"] = False
                st.error(f"❌ Policy Transmission failed: {e}")

        # ── Round 4: Macro Conviction Synopsis (cross-signal coherence check) ───
        with st.spinner("🧠 Round 4/4 — Macro Conviction Synopsis..."):
            try:
                import datetime as _syndt
                from services.claude_client import generate_macro_synopsis as _gen_synopsis

                # Build signal summary from session state
                _rc = st.session_state.get("_regime_context") or {}
                _dp = st.session_state.get("_dominant_rate_path") or {}
                _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
                _sig_parts = []
                _dq_ctx = st.session_state.get("_data_quality") or {}
                if _dq_ctx:
                    _sig_parts.append(f"DATA QUALITY: {_dq_ctx['score']}/100 — {_dq_ctx['label']}")

                # ── Tactical Regime first — actionable timeframe, highest weight ──
                _tac = st.session_state.get("_tactical_context")
                if _tac:
                    _tac_sigs = _tac.get("signals", [])
                    _tac_sig_str = "  |  ".join(
                        f"{s['Signal'].split('(')[0].strip()}: {s['Value']} ({s['Direction']})"
                        for s in _tac_sigs
                    ) if _tac_sigs else ""
                    _tac_block = (
                        f"TACTICAL REGIME: {_tac['tactical_score']}/100 ({_tac['label']}) — {_tac['action_bias']}"
                    )
                    if _tac_sig_str:
                        _tac_block += f"\n  Signals: {_tac_sig_str}"
                    _tac_ai_text = st.session_state.get("_tactical_analysis", "")
                    if _tac_ai_text:
                        _tac_block += f"\n  AI Analysis: {_tac_ai_text[:500]}"
                    _sig_parts.append(_tac_block)

                # ── Options Flow (hours/days layer) ──────────────────────────────
                _of = st.session_state.get("_options_flow_context")
                if _of:
                    _of_block = f"OPTIONS FLOW: {_of['options_score']}/100 ({_of['label']}) — {_of['action_bias']}"
                    _of_sigs = "  |  ".join(
                        f"{s['Signal']}: {s['Value']} ({s['Direction']})"
                        for s in _of.get("signals", [])
                    )
                    if _of_sigs:
                        _of_block += f"\n  Signals: {_of_sigs}"
                    _sig_parts.append(_of_block)

                # ── GEX Dealer Positioning (convexity adjustment on options signals) ──
                # Prefer SPX (institutional dealer book); fall back to per-ticker _gex_profile
                _gex = st.session_state.get("_gex_profile_spx") or st.session_state.get("_gex_profile")
                if _gex:
                    _gz = _gex.get("zone", "")
                    _gflip = _gex.get("gamma_flip")
                    _cwall = _gex.get("call_wall")
                    _pwall = _gex.get("put_wall")
                    _gex_net = _gex.get("total_gex")
                    _g_interp = (
                        "POSITIVE (dealers long gamma — market self-stabilizing, mean-reverting regime; "
                        "directional options flow signals are DAMPENED by dealer hedging)"
                        if "Positive" in _gz else
                        "NEGATIVE (dealers short gamma — market trending/self-amplifying regime; "
                        "directional options flow signals have ELEVATED conviction)"
                        if "Negative" in _gz else _gz
                    )
                    _gex_parts = [f"GEX DEALER POSITIONING: {_g_interp}"]
                    if _gflip:
                        _gex_parts.append(f"  Gamma Flip Level: ${_gflip:.0f} (below = trending regime, above = pinned)")
                    if _cwall:
                        _gex_parts.append(f"  Call Wall (resistance): ${_cwall:.0f}")
                    if _pwall:
                        _gex_parts.append(f"  Put Wall (support): ${_pwall:.0f}")
                    if _gex_net is not None:
                        _gex_parts.append(f"  Net GEX: ${_gex_net/1e9:.2f}B ({'+' if _gex_net >= 0 else ''})")
                    _sig_parts.append("\n".join(_gex_parts))

                # ── Credit Risk (fixed income convexity) for held positions ──────
                try:
                    from utils.journal import load_journal as _lj_cr
                    from services.market_data import fetch_credit_metrics as _fcr
                    _cr_tks = list({t["ticker"].upper() for t in _lj_cr() if t.get("status") == "open"})[:6]
                    _cr_flags = []
                    for _ctk in _cr_tks:
                        try:
                            _cm = _fcr(_ctk)
                            if not _cm:
                                continue
                            _cov = _cm.get("interest_coverage")
                            _lev = _cm.get("debt_to_ebitda")
                            _mf  = _cm.get("maturity_flag", "")
                            _cf  = _cm.get("coverage_flag", "")
                            if _cov is not None and _cov < 3.0:
                                _cr_flags.append(
                                    f"{_ctk}: coverage {_cov:.1f}x"
                                    + (f", D/EBITDA {_lev:.1f}x" if _lev else "")
                                    + (f" — {_cf}" if _cf else "")
                                    + (" | " + _mf if _mf else "")
                                )
                            elif _mf and "HIGH" in _mf:
                                _cr_flags.append(f"{_ctk}: {_mf}")
                        except Exception:
                            continue
                    if _cr_flags:
                        _sig_parts.append(
                            "CREDIT RISK IN PORTFOLIO (interest coverage < 3x or high refi risk — "
                            "options put skew understates tail risk for these positions):\n  "
                            + "\n  ".join(_cr_flags)
                        )
                except Exception:
                    pass

                # ── Macro Regime ──────────────────────────────────────────────────
                if _rc.get("regime"):
                    _sig_parts.append(f"MACRO REGIME: {_rc['regime']} (score {_rc.get('score',0):+.2f}) | Quadrant: {_rc.get('quadrant','')}")
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
                _activism = st.session_state.get("_activism_digest", "")
                if _activism:
                    _sig_parts.append(f"ACTIVISM CAMPAIGNS: {_activism[:300]}")
                _sector_reg = st.session_state.get("_sector_regime_digest", "")
                if _sector_reg:
                    _sig_parts.append(f"SECTOR×REGIME: {_sector_reg[:300]}")
                _bs = st.session_state.get("_custom_swans", {})
                if _bs:
                    _bs_summary = "; ".join(
                        f"{k} ({v.get('probability_pct', 0):.1f}% annual)"
                        for k, v in list(_bs.items())[:3]
                    )
                    _sig_parts.append(f"BLACK SWANS: {_bs_summary}")

                # ── Macro calendar context ────────────────────────────────
                try:
                    from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
                    _fomc = get_next_fomc(); _cpi = get_next_cpi(); _nfp = get_next_nfp()
                    _sig_parts.append(
                        f"MACRO CALENDAR: FOMC in {_fomc.get('days_away','?')}d ({_fomc.get('date','')})"
                        f" | CPI in {_cpi.get('days_away','?')}d ({_cpi.get('date','')})"
                        f" | NFP in {_nfp.get('days_away','?')}d ({_nfp.get('date','')})"
                    )
                except Exception:
                    pass

                # ── Earnings risk context ─────────────────────────────────
                _er_sig = st.session_state.get("_qir_earnings_risk") or []
                if _er_sig:
                    _er_parts = [
                        f"{e['ticker']} in {e['days_away']}d"
                        + (f" (±{e['expected_move_pct']:.1f}%)" if e.get('expected_move_pct') else "")
                        for e in _er_sig[:5]
                    ]
                    _sig_parts.append(f"EARNINGS RISK: {', '.join(_er_parts)}")

                _signals_text_for_debate = "\n\n".join(_sig_parts)
                _synopsis = _gen_synopsis(_signals_text_for_debate, use_claude=_use_claude, model=_cl_model)
                _syn_tier = "👑 Highly Regarded Mode" if (_use_claude and _cl_model == "claude-sonnet-4-6") \
                    else ("🧠 Regard Mode" if _use_claude else "⚡ Freeloader Mode")
                st.session_state["_macro_synopsis"] = _synopsis
                st.session_state["_macro_synopsis_ts"] = _syndt.datetime.now()
                st.session_state["_macro_synopsis_engine"] = _syn_tier
                _results["synopsis"] = True
                st.success(f"✅ Macro Conviction Synopsis — {_synopsis.get('conviction', '?')}")
            except Exception as e:
                _results["synopsis"] = False
                st.error(f"❌ Synopsis failed: {e}")

        # ── Round 5: Portfolio Risk Snapshot (headless risk matrix) ──────────
        with st.spinner("📊 Round 5/5 — Portfolio Risk Snapshot + AI Interpretation..."):
            try:
                from modules.trade_journal import run_quick_risk_snapshot
                _risk_ok = run_quick_risk_snapshot(use_claude=_use_claude, model=_cl_model)
                _results["risk_snapshot"] = _risk_ok
                if _risk_ok:
                    import datetime as _rsdt
                    st.session_state["_portfolio_risk_snapshot_ts"] = _rsdt.datetime.now()
                    _snap = st.session_state.get("_portfolio_risk_snapshot") or {}
                    _interp = st.session_state.get("_risk_matrix_interpretation") or {}
                    _alert = _interp.get("alert_level", "")
                    _alert_str = f" · Risk Alert: {_alert}" if _alert else ""
                    _beta = _snap.get("beta", "?")
                    st.success(f"✅ Portfolio Risk Snapshot — Beta {_beta}{_alert_str}")
                else:
                    st.info("ℹ️ Risk Snapshot skipped — no open positions in Trade Journal")
            except Exception as e:
                _results["risk_snapshot"] = False
                st.error(f"❌ Risk Snapshot failed: {e}")

            # ── Earnings Risk: fetch per held position ────────────────────
            try:
                from utils.journal import load_journal as _lj_r5
                from services.market_data import fetch_earnings_intelligence as _fei
                import datetime as _er_dt
                _open_tks = list({t["ticker"].upper() for t in _lj_r5() if t.get("status") == "open"})
                _earn_risk = []
                for _etk in _open_tks:
                    try:
                        _ei = _fei(_etk)
                        _ne = _ei.get("next_earnings") or {}
                        _em = _ei.get("expected_move") or {}
                        _days = _ne.get("days_away")
                        if _days is not None and _days <= 21:
                            _earn_risk.append({
                                "ticker": _etk,
                                "days_away": _days,
                                "date": _ne.get("date", ""),
                                "expected_move_pct": _em.get("pct"),
                                "expected_move_dollar": _em.get("dollar"),
                            })
                    except Exception:
                        continue
                _earn_risk.sort(key=lambda x: x["days_away"])
                if _earn_risk:
                    st.session_state["_qir_earnings_risk"] = _earn_risk
                    st.session_state["_qir_earnings_risk_ts"] = _er_dt.datetime.now()
                    _er_names = ", ".join(
                        f"{e['ticker']} ({e['days_away']}d)" for e in _earn_risk[:3]
                    )
                    st.success(f"✅ Earnings Risk — {len(_earn_risk)} position(s) flagged: {_er_names}")
                else:
                    st.info("ℹ️ No held positions with earnings in the next 21 days")
            except Exception as _er_e:
                st.warning(f"⚠ Earnings Risk scan failed: {_er_e}")

        # ── Store completion result for persistent display (outside button handler) ─
        _n_ok = sum(1 for v in _results.values() if v)
        st.session_state["_qir_last_n_ok"]    = _n_ok
        st.session_state["_qir_last_n_total"] = len(_results)
        if _n_ok == len(_results):
            try:
                from services.telegram_client import send_alert as _tg_qir
                _rc_f  = st.session_state.get("_regime_context") or {}
                _tac_f = st.session_state.get("_tactical_context") or {}
                _dq_f  = st.session_state.get("_data_quality") or {}
                _of_f  = st.session_state.get("_options_flow_context") or {}

                # ── Pattern change alert ──────────────────────────────────
                _new_pat = _classify_signals(_rc_f, _tac_f, _of_f)
                _new_pat_name = _new_pat["pattern"]
                _prev_pat_name = st.session_state.get("_qir_last_pattern", "")
                if _prev_pat_name and _prev_pat_name != _new_pat_name:
                    _pat_emojis = {
                        "BULLISH_CONFIRMATION":   "🟢",
                        "BEARISH_CONFIRMATION":   "🔴",
                        "PULLBACK_IN_UPTREND":    "🟡",
                        "OPTIONS_FLOW_DIVERGENCE":"🟡",
                        "BEAR_MARKET_BOUNCE":     "🟠",
                        "LATE_CYCLE_SQUEEZE":     "🔴",
                        "GENUINE_UNCERTAINTY":    "⚪",
                    }
                    _e_old = _pat_emojis.get(_prev_pat_name, "◆")
                    _e_new = _pat_emojis.get(_new_pat_name, "◆")
                    _tg_qir(
                        f"🔄 <b>QIR Pattern Changed</b>\n"
                        f"{_e_old} {_prev_pat_name.replace('_',' ')} → {_e_new} {_new_pat_name.replace('_',' ')}\n"
                        f"Buy: {_new_pat['buy_tier']} | Short: {_new_pat['short_tier']}\n"
                        f"Regime: {_rc_f.get('regime','?')} | Tactical: {_tac_f.get('tactical_score','?')}/100 | Flow: {_of_f.get('options_score','?')}/100"
                    )
                st.session_state["_qir_last_pattern"] = _new_pat_name

                _tg_qir(
                    f"⚡ <b>Quick Intel Run Complete</b>\n"
                    f"Pattern: {_new_pat['label']}\n"
                    f"Regime: {_rc_f.get('regime','?')} | {_rc_f.get('quadrant','?')}\n"
                    f"Tactical: {_tac_f.get('tactical_score','?')}/100 — {_tac_f.get('label','?')}\n"
                    f"Options Flow: {_of_f.get('options_score','?')}/100 — {_of_f.get('label','')}\n"
                    f"Data Quality: {_dq_f.get('score','?')}/100 — {_dq_f.get('label','')}"
                )
            except Exception:
                pass

        # ── Save QIR run record to history ────────────────────────────────────
        try:
            import uuid as _uuid
            from services.qir_history import QIRRunRecord, append_qir_run as _append_run
            _rc_h  = st.session_state.get("_regime_context") or {}
            _tac_h = st.session_state.get("_tactical_context") or {}
            _of_h  = st.session_state.get("_options_flow_context") or {}
            _syn_h = st.session_state.get("_macro_synopsis") or {}
            _n_ok_h = st.session_state.get("_qir_last_n_ok", 0)
            _n_tot_h = st.session_state.get("_qir_last_n_total", 0)
            _eng_h = st.session_state.get("_macro_synopsis_engine", "")
            _append_run(QIRRunRecord(
                run_id=str(_uuid.uuid4())[:8],
                timestamp=_dt_qir.datetime.now().isoformat(),
                pattern=st.session_state.get("_qir_last_pattern", ""),
                conviction=_syn_h.get("conviction", ""),
                tactical_score=int(_tac_h.get("tactical_score", 0)),
                options_score=float(_of_h.get("options_score", 0)),
                regime_label=_rc_h.get("regime", ""),
                quadrant=_rc_h.get("quadrant", ""),
                n_ok=_n_ok_h,
                n_total=_n_tot_h,
                engine=_eng_h,
            ))
        except Exception:
            pass

        # Flag that a fresh run just completed — dashboard below will re-render with new data
        st.session_state["_qir_just_ran"] = True

    render_action_queue(max_items=5)

    # ── ⚔️ Adversarial Debate — standalone, no QIR run required ─────────────
    st.markdown(
        '<div style="border-top:1px solid #1e293b;margin:16px 0 10px 0;"></div>',
        unsafe_allow_html=True,
    )
    try:
        from utils.ai_tier import TIER_OPTS as _dbt_opts, TIER_MAP as _dbt_map
    except ImportError:
        _dbt_opts = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
        _dbt_map  = {
            "⚡ Freeloader Mode":      (False, None),
            "🧠 Regard Mode":          (True, "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
        }
    _dbt_c1, _dbt_c2, _dbt_c3 = st.columns([2, 1.5, 1])
    with _dbt_c1:
        st.markdown(
            '<span style="color:#64748b;font-size:11px;">'
            '⚔️ <b style="color:#94a3b8;">Adversarial Debate</b> — '
            'Sir Doomburger 🐻 vs Sir Fukyerputs 🐂, judged by Judge Judy ⚖️. '
            '<span style="color:#475569;">Runs independently · 3 LLM calls</span></span>'
            '<div style="color:#475569;font-size:10px;margin-top:3px;">💡 Best results: run ⚡ Quick Intel Run first to load regime, rate path, options flow &amp; macro synopsis</div>',
            unsafe_allow_html=True,
        )
    with _dbt_c2:
        _dbt_tier = st.selectbox("", _dbt_opts, key="qir_debate_tier", label_visibility="collapsed")
    with _dbt_c3:
        _run_dbt = st.button("⚔️ Run Debate", key="btn_qir_debate_standalone", use_container_width=True)
    if _run_dbt:
        _dbt_uc, _dbt_mdl = _dbt_map.get(_dbt_tier, (False, None))
        _dbt_engine = _dbt_tier
        from services.claude_client import generate_adversarial_debate as _gen_dbt_sa
        from utils.signal_block import build_macro_block as _build_dbt_sa
        _dbt_sigs = _build_dbt_sa()
        if _dbt_sigs and _dbt_sigs != "NO_REGIME_DATA":
            with st.spinner("⚔️ Debate — Sir Doomburger 🐻 vs Sir Fukyerputs 🐂..."):
                try:
                    _dbt_result = _gen_dbt_sa(
                        _dbt_sigs,
                        use_claude=_dbt_uc,
                        model=_dbt_mdl,
                        topic="Is the current macro regime bullish or bearish for risk assets? Which signals are most decisive right now?",
                    )
                    st.session_state["_adversarial_debate"] = _dbt_result
                    st.session_state["_adversarial_debate_engine"] = _dbt_engine
                    try:
                        from utils.debate_record import log_verdict as _log_dbt, resolve_old_verdicts as _resolve_dbt
                        _rc_dbt = st.session_state.get("_regime_context") or {}
                        _log_dbt(
                            verdict=_dbt_result.get("verdict", "CONTESTED"),
                            confidence=_dbt_result.get("confidence", 5),
                            regime=_rc_dbt.get("regime", ""),
                            quadrant=_rc_dbt.get("quadrant", ""),
                            regime_score=float(_rc_dbt.get("score", 0.0)),
                        )
                        _resolve_dbt()
                    except Exception:
                        pass
                    st.rerun()
                except Exception as _dbt_e:
                    st.error(f"❌ Debate failed: {_dbt_e}")
        else:
            st.warning("⚠️ Run Quick Intel first to load market signals, then debate away.")

    # ── Adversarial Debate Display ──────────────────────────────────────────────
    _debate = st.session_state.get("_adversarial_debate") or {}
    if _debate.get("bear_argument") or _debate.get("bull_argument"):
        _db_verdict = _debate.get("verdict", "CONTESTED")
        _db_conf    = _debate.get("confidence", 5)
        _db_conf_adj = apply_confidence_penalty(_db_conf)
        _db_bias    = _debate.get("contested_bias", "")
        _db_bias_reason = _debate.get("contested_bias_reason", "")
        _db_engine  = st.session_state.get("_adversarial_debate_engine", "")
        _db_engine_badge = (
            f'<span style="color:#64748b;font-size:10px;margin-left:8px;">{_db_engine}</span>'
            if _db_engine else ""
        )
        _vc = {"BULL WINS": "#22c55e", "BEAR WINS": "#ef4444", "CONTESTED": "#f59e0b"}.get(_db_verdict, "#f59e0b")
        _vbg = {"BULL WINS": "#020d06", "BEAR WINS": "#0d0000", "CONTESTED": "#0d0800"}.get(_db_verdict, "#0d0800")

        # ── Court formality strings ───────────────────────────────────────
        _winner = {"BULL WINS": "Sir Fukyerputs 🐂", "BEAR WINS": "Sir Doomburger 🐻", "CONTESTED": None}.get(_db_verdict)
        _loser  = {"BULL WINS": "Sir Doomburger 🐻", "BEAR WINS": "Sir Fukyerputs 🐂", "CONTESTED": None}.get(_db_verdict)
        _sentences = {
            "BULL WINS": "Sir Doomburger is hereby sentenced to 30 days of buying the dip and deep reflection on his pessimism.",
            "BEAR WINS": "Sir Fukyerputs is hereby sentenced to 30 days of holding cash and contemplating the dangers of leverage.",
            "CONTESTED": "Both parties are hereby remanded to gather additional evidence. Court is adjourned pending new data.",
        }
        _sentence = _sentences.get(_db_verdict, _sentences["CONTESTED"])

        # ── Verdict record ────────────────────────────────────────────────
        try:
            from utils.debate_record import get_stats as _get_stats, get_recent_verdicts as _get_recent
            _jj_stats = _get_stats()
            _jj_recent = _get_recent(5)
        except Exception:
            _jj_stats = {}
            _jj_recent = []

        _acc_str = f"{_jj_stats['accuracy_pct']}% accuracy" if _jj_stats.get("accuracy_pct") is not None else "unresolved"
        _record_str = f"{_jj_stats.get('correct',0)}W-{_jj_stats.get('wrong',0)}L · {_acc_str} · {_jj_stats.get('pending',0)} pending"

        # ── Outcome dots for recent verdicts ──────────────────────────────
        _dot_html = ""
        for _rv in reversed(_jj_recent):
            _oc = _rv.get("outcome", "pending")
            _rv_v = _rv.get("verdict", "")
            _dot_c = {"correct": "#22c55e", "wrong": "#ef4444", "pending": "#475569"}.get(_oc, "#475569")
            _dot_sym = {"BULL WINS": "▲", "BEAR WINS": "▼", "CONTESTED": "■"}.get(_rv_v, "·")
            _dot_html += f'<span style="color:{_dot_c};font-size:13px;margin-right:4px;" title="{_rv_v} — {_oc}">{_dot_sym}</span>'

        # ── Glow CSS ──────────────────────────────────────────────────────
        _glow = f"0 0 8px {_vc}, 0 0 20px {_vc}88, 0 0 40px {_vc}44"
        _contested_bias_html = ""
        if _db_verdict == "CONTESTED" and _db_bias:
            _contested_bias_html = (
                f'<div style="text-align:center;color:#94a3b8;font-size:10px;margin-top:-8px;margin-bottom:10px;">'
                f'{_db_bias}' + (f' · {_db_bias_reason}' if _db_bias_reason else '') +
                f'</div>'
            )

        st.markdown(
            f'<div style="background:{_vbg};border:2px solid {_vc};border-radius:10px;'
            f'padding:18px 20px;margin-top:12px;margin-bottom:4px;">'

            # Court header
            f'<div style="text-align:center;margin-bottom:14px;">'
            f'<div style="color:#64748b;font-size:9px;font-weight:700;letter-spacing:0.15em;margin-bottom:4px;">⚖️ IN THE COURT OF MACRO JUSTICE</div>'
            f'<div style="color:#94a3b8;font-size:9px;letter-spacing:0.08em;">THE HONORABLE JUDGE JUDY PRESIDING{_db_engine_badge}</div>'
            f'</div>'

            # Glowing verdict
            f'<div style="text-align:center;margin-bottom:14px;">'
            f'<div style="color:{_vc};font-size:28px;font-weight:900;letter-spacing:0.05em;'
            f'text-shadow:{_glow};line-height:1.1;">⚔️ {_db_verdict}</div>'
            f'<div style="color:{_vc}aa;font-size:11px;margin-top:4px;font-style:italic;">'
            + (f'The Court finds in favor of {_winner}' if _winner else 'The Court is divided') +
            f'</div></div>'
            f'{_contested_bias_html}'

            # Sentence
            f'<div style="background:#0a0a0a;border-left:3px solid {_vc};border-radius:4px;'
            f'padding:8px 12px;margin-bottom:12px;font-size:11px;color:#e2e8f0;font-style:italic;">'
            f'"{_sentence}"</div>'

            # Judge Judy record
            f'<div style="display:flex;align-items:center;justify-content:space-between;">'
            f'<div style="color:#475569;font-size:9px;font-weight:700;letter-spacing:0.1em;">JUDGE JUDY\'S RECORD</div>'
            f'<div style="color:#64748b;font-size:10px;">{_record_str}</div>'
            f'</div>'
            f'<div style="margin-top:4px;display:flex;align-items:center;gap:2px;">'
            f'<span style="color:#475569;font-size:9px;margin-right:6px;">RECENT:</span>{_dot_html}'
            f'<span style="color:#334155;font-size:9px;margin-left:6px;">▲=bull ▼=bear ■=contested · 🟢correct 🔴wrong ⚫pending</span>'
            f'</div>'

            # Confidence
            f'<div style="margin-top:10px;text-align:right;">'
            f'<span style="color:#475569;font-size:9px;">AI confidence in verdict: </span>'
            f'<span style="color:{_vc};font-weight:700;font-size:11px;">{_db_conf_adj}/10</span>'
            f'<span style="color:#334155;font-size:9px;margin-left:6px;">(Judge Judy self-rated — not a quant score)</span>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Arguments side by side ────────────────────────────────────────
        _col_bear, _col_bull = st.columns(2)

        with _col_bear:
            _bear_border = "#ef444499" if _db_verdict == "BEAR WINS" else "#ef444433"
            _bear_glow = f"box-shadow:0 0 12px #ef444444;" if _db_verdict == "BEAR WINS" else ""
            st.markdown(
                f'<div style="background:#1a0000;border:1px solid {_bear_border};border-radius:6px;padding:12px;{_bear_glow}">'
                f'<div style="color:#ef4444;font-weight:700;font-size:12px;margin-bottom:8px;">'
                + ("🏆 " if _db_verdict == "BEAR WINS" else "")
                + f'🐻 SIR DOOMBURGER'
                + (" — WINS" if _db_verdict == "BEAR WINS" else " — SENTENCED" if _db_verdict == "BULL WINS" else "") +
                f'</div>'
                f'<div style="color:#e2e8f0;font-size:11px;line-height:1.6;">{_debate.get("bear_argument","")}</div>'
                f'<div style="margin-top:8px;padding-top:6px;border-top:1px solid #ef444433;">'
                f'<span style="color:#ef4444;font-size:9px;font-weight:700;letter-spacing:0.1em;">STRONGEST POINT: </span>'
                f'<span style="color:#fca5a5;font-size:10px;">{_debate.get("bear_strongest","")}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with _col_bull:
            _bull_border = "#22c55e99" if _db_verdict == "BULL WINS" else "#22c55e33"
            _bull_glow = f"box-shadow:0 0 12px #22c55e44;" if _db_verdict == "BULL WINS" else ""
            st.markdown(
                f'<div style="background:#052e16;border:1px solid {_bull_border};border-radius:6px;padding:12px;{_bull_glow}">'
                f'<div style="color:#22c55e;font-weight:700;font-size:12px;margin-bottom:8px;">'
                + ("🏆 " if _db_verdict == "BULL WINS" else "")
                + f'🐂 SIR FUKYERPUTS'
                + (" — WINS" if _db_verdict == "BULL WINS" else " — SENTENCED" if _db_verdict == "BEAR WINS" else "") +
                f'</div>'
                f'<div style="color:#e2e8f0;font-size:11px;line-height:1.6;">{_debate.get("bull_argument","")}</div>'
                f'<div style="margin-top:8px;padding-top:6px;border-top:1px solid #22c55e33;">'
                f'<span style="color:#22c55e;font-size:9px;font-weight:700;letter-spacing:0.1em;">STRONGEST POINT: </span>'
                f'<span style="color:#86efac;font-size:10px;">{_debate.get("bull_strongest","")}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Judge Judy asymmetry ruling ───────────────────────────────────
        if _debate.get("asymmetry") or _debate.get("key_disagreement"):
            st.markdown(
                f'<div style="background:#0a0a1a;border:1px solid #475569;border-radius:6px;'
                f'padding:10px 14px;margin-top:8px;">'
                f'<div style="color:#94a3b8;font-size:9px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">⚖️ JUDGE JUDY — RISK/REWARD RULING</div>'
                f'<div style="color:#e2e8f0;font-size:11px;margin-bottom:4px;">{_debate.get("asymmetry","")}</div>'
                f'<div style="color:#64748b;font-size:10px;"><b>Key disagreement:</b> {_debate.get("key_disagreement","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── QIR Post-Run Summary (outside button handler — survives st.rerun()) ──
    _last_ok    = st.session_state.get("_qir_last_n_ok")
    _last_total = st.session_state.get("_qir_last_n_total")
    if _last_ok is not None and _last_total is not None:
        if _last_ok == _last_total:
            st.markdown(
                f'<div style="background:#052e16;border:1px solid #22c55e44;border-radius:6px;'
                f'padding:8px 14px;margin:10px 0 6px 0;font-size:12px;color:#22c55e;">'
                f'✅ All {_last_total} intel modules ready</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"{_last_ok}/{_last_total} modules completed — check errors above.")

        # ── Intelligence Dashboard (always renders fresh after run) ───────────────
        _render_qir_dashboard()

        _synopsis = st.session_state.get("_macro_synopsis") or {}
        if _synopsis.get("conviction"):
            _conv = _synopsis["conviction"]
            _conv_color = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "MIXED": "#f59e0b", "UNCERTAIN": "#94a3b8"}.get(_conv, "#94a3b8")
            _conv_bg    = {"BULLISH": "#052e16", "BEARISH": "#1a0000", "MIXED": "#1a1200", "UNCERTAIN": "#0d1117"}.get(_conv, "#0d1117")
            _kp_html = "".join(
                f'<div style="color:#94a3b8;font-size:11px;padding:2px 0;"> · {kp}</div>'
                for kp in _synopsis.get("key_points", [])
            )
            _ct_html = "".join(
                f'<div style="color:#f59e0b;font-size:10px;padding:2px 0;">⚠ {ct}</div>'
                for ct in _synopsis.get("contradictions", [])
            )
            _syn_eng = st.session_state.get("_macro_synopsis_engine", "")
            _eng_span = f'<span style="font-size:10px;color:#555;margin-left:auto;">{_syn_eng}</span>' if _syn_eng else ""
            st.markdown(
                f'<div style="background:{_conv_bg};border:2px solid {_conv_color};border-radius:8px;'
                f'padding:14px 18px;margin-top:10px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="background:{_conv_color};color:black;font-weight:800;font-size:12px;'
                f'padding:3px 12px;border-radius:4px;letter-spacing:0.08em;">{_conv}</span>'
                f'<span style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.08em;">MACRO CONVICTION</span>'
                f'{_eng_span}'
                f'</div>'
                f'<div style="color:#e2e8f0;font-size:12px;line-height:1.6;margin-bottom:6px;">{_synopsis.get("summary","")}</div>'
                f'{_kp_html}{_ct_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        _tac_ctx = st.session_state.get("_tactical_context", {})
        _tac_ai  = st.session_state.get("_tactical_analysis", "")
        if _tac_ctx:
            _ts_val   = _tac_ctx.get("tactical_score", 50)
            _tlabel   = _tac_ctx.get("label", "")
            _tbias    = _tac_ctx.get("action_bias", "")
            _tac_color = "#22c55e" if _ts_val >= 65 else ("#f59e0b" if _ts_val >= 38 else "#ef4444")
            _tac_bg    = "#0c1a0c" if _ts_val >= 65 else ("#1a1200" if _ts_val >= 38 else "#1a0000")
            with st.expander(f"⚡ Tactical Regime — {_tlabel} ({_ts_val}/100)", expanded=True):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                    f'<span style="background:{_tac_color};color:black;font-weight:800;font-size:11px;'
                    f'padding:3px 10px;border-radius:4px;letter-spacing:0.06em;">{_tlabel.upper()}</span>'
                    f'<span style="color:{_tac_color};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;">{_ts_val}/100</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:11px;">{_tbias}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _tac_sigs = _tac_ctx.get("signals", [])
                if _tac_sigs:
                    _sigs_html = ""
                    for _sr in _tac_sigs:
                        _sc = "#22c55e" if _sr["Score"] > 0.2 else ("#ef4444" if _sr["Score"] < -0.2 else "#94a3b8")
                        _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
                        _sigs_html += (
                            f'<div style="color:{_sc};font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;padding:1px 0;">{_arrow} {_sr["Signal"]}: '
                            f'<span style="color:{COLORS["text"]}">{_sr["Value"]}</span>'
                            f'<span style="color:#475569;"> ({_sr["Direction"]})</span></div>'
                        )
                    st.markdown(f'<div style="margin-bottom:8px;">{_sigs_html}</div>', unsafe_allow_html=True)
                if _tac_ai:
                    st.markdown(
                        f'<div style="background:{_tac_bg};border-left:3px solid {_tac_color};'
                        f'padding:10px 14px;font-size:12px;color:{COLORS["text"]};'
                        f'line-height:1.8;white-space:pre-line;">{_tac_ai}</div>',
                        unsafe_allow_html=True,
                    )

        _of_ctx = st.session_state.get("_options_flow_context") or {}
        if _of_ctx:
            _os_val   = _of_ctx.get("options_score", 50)
            _of_label = _of_ctx.get("label", "")
            _of_bias  = _of_ctx.get("action_bias", "")
            _of_color = "#22c55e" if _os_val >= 65 else ("#f59e0b" if _os_val >= 38 else "#ef4444")
            _of_bg    = "#0c1a0c" if _os_val >= 65 else ("#1a1200" if _os_val >= 38 else "#1a0000")
            with st.expander(f"📊 Options Flow — {_of_label} ({_os_val}/100)", expanded=True):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                    f'<span style="background:{_of_color};color:black;font-weight:800;font-size:11px;'
                    f'padding:3px 10px;border-radius:4px;letter-spacing:0.06em;">{_of_label.upper()}</span>'
                    f'<span style="color:{_of_color};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;">{_os_val}/100</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:11px;">{_of_bias}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _of_sigs = _of_ctx.get("signals", [])
                if _of_sigs:
                    _of_sigs_html = ""
                    for _sr in _of_sigs:
                        _sc = "#22c55e" if _sr["Score"] > 0.2 else ("#ef4444" if _sr["Score"] < -0.2 else "#94a3b8")
                        _arrow = "▲" if _sr["Score"] > 0.1 else ("▼" if _sr["Score"] < -0.1 else "◆")
                        _of_sigs_html += (
                            f'<div style="color:{_sc};font-family:\'JetBrains Mono\',Consolas,monospace;'
                            f'font-size:11px;padding:1px 0;">{_arrow} {_sr["Signal"]}: '
                            f'<span style="color:{COLORS["text"]}">{_sr["Value"]}</span>'
                            f'<span style="color:#475569;"> ({_sr["Direction"]})</span></div>'
                        )
                    st.markdown(f'<div style="margin-bottom:8px;">{_of_sigs_html}</div>', unsafe_allow_html=True)
                _vix_lv = _of_ctx.get("vix_level", "?")
                _vix_rg = _of_ctx.get("vix_regime", "Normal")
                _mode   = _of_ctx.get("scoring_mode", "static")
                _n_hist = _of_ctx.get("n_pc_hist", 0)
                st.caption(f"VIX {_vix_lv} · {_vix_rg} regime · {_mode} · {_n_hist} samples in history")

        _dp = st.session_state.get("_dominant_rate_path") or {}
        if _dp:
            _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
            _dp_label  = _dp_labels.get(_dp.get("scenario", ""), _dp.get("scenario", ""))
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid {COLORS["border"]};border-radius:4px;'
                f'padding:8px 12px;font-size:11px;color:{COLORS["text_dim"]};margin-top:8px;">'
                f'📈 <b style="color:{COLORS["bloomberg_orange"]}">Fed Rate Path</b> — '
                f'Dominant: <b style="color:{COLORS["text"]}">{_dp_label}</b> '
                f'({_dp.get("prob_pct", 0):.0f}% probability)</div>',
                unsafe_allow_html=True,
            )

        _plays = st.session_state.get("_rp_plays_result") or {}
        if _plays:
            _regime    = st.session_state.get("_regime_context", {})
            _rlabel    = _regime.get("regime", "")
            _quad      = _regime.get("quadrant", "")
            with st.expander(f"📡 Regime: {_rlabel} · {_quad}", expanded=True):
                _sectors = _plays.get("sectors", [])
                _stocks  = _plays.get("stocks", [])
                if _sectors:
                    st.markdown("**Sectors:** " + " · ".join(
                        f"{s.get('name','')} ({'★'*s.get('conviction',1)})" for s in _sectors[:4]
                    ))
                if _stocks:
                    st.markdown("**Stocks:** " + " · ".join(
                        f"{s.get('ticker','')} ({'★'*s.get('conviction',1)})" for s in _stocks[:4]
                    ))

        _digest = st.session_state.get("_current_events_digest", "")
        if _digest:
            with st.expander("🗞 News Digest", expanded=True):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_digest}</div>',
                    unsafe_allow_html=True,
                )

        _doom = st.session_state.get("_doom_briefing", "")
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

        _activism = st.session_state.get("_activism_digest", "")
        if _activism:
            with st.expander("🎯 Activism Campaigns (13D)", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_activism}</div>',
                    unsafe_allow_html=True,
                )

        _srd = st.session_state.get("_sector_regime_digest", "")
        if _srd:
            with st.expander("🔄 Sector×Regime Digest", expanded=False):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:11px;color:{COLORS["text"]};line-height:1.6;">{_srd}</div>',
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

    # ── QIR Run History — conviction dot row ──────────────────────────────
    try:
        from services.qir_history import load_qir_history as _load_hist
        _hist = _load_hist()
        if _hist:
            _dot_colors = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "MIXED": "#f59e0b", "UNCERTAIN": "#64748b"}
            _dots_html = ""
            for _h in _hist[:10][::-1]:
                _dc   = _dot_colors.get(_h.get("conviction", ""), "#334155")
                _date = _h.get("timestamp", "")[:10]
                _tac  = _h.get("tactical_score", 0)
                _conv_label = _h.get("conviction", "")
                _tip  = f"{_date} \u00b7 {_conv_label} \u00b7 Tac {_tac}"
                _dots_html += (
                    f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">'
                    f'<div style="width:14px;height:14px;border-radius:50%;background:{_dc};" '
                    f'title="{_tip}"></div>'
                    f'<span style="font-size:9px;color:#475569;">{_tac}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="margin-top:14px;">'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
                f'color:#475569;margin-bottom:6px;">RUN HISTORY &nbsp;'
                f'<span style="font-weight:400;color:#22c55e;">● BULL</span> &nbsp;'
                f'<span style="font-weight:400;color:#ef4444;">● BEAR</span> &nbsp;'
                f'<span style="font-weight:400;color:#f59e0b;">● MIX</span></div>'
                f'<div style="display:flex;gap:10px;align-items:flex-end;">{_dots_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── HMM Regime Brain — maintenance section ───────────────────────────────
    with st.expander("🧠 HMM Regime Brain", expanded=False):
        try:
            from services.hmm_regime import (
                load_hmm_brain as _load_brain,
                train_hmm as _train_hmm,
                score_current_state as _score_hmm,
                get_state_color as _hmm_col,
                get_state_arrow as _hmm_arr,
            )
            _brain = _load_brain()
            if _brain:
                _b_trained = _brain.trained_at[:10]
                _b_end     = _brain.training_end
                _b_start   = _brain.training_start
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:10px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">CURRENT BRAIN</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">STATES</div>'
                    f'<div style="font-size:18px;font-weight:900;color:#94a3b8;">{_brain.n_states}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">BIC</div>'
                    f'<div style="font-size:18px;font-weight:900;color:#94a3b8;">{_brain.bic:,.0f}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">TRAINED</div>'
                    f'<div style="font-size:11px;font-weight:700;color:#94a3b8;margin-top:4px;">{_b_trained}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">WINDOW</div>'
                    f'<div style="font-size:9px;font-weight:700;color:#64748b;margin-top:4px;">{_b_start}<br>→ {_b_end}</div></div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#475569;">State labels: '
                    f'{" | ".join(_brain.state_labels)}</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:2px;">'
                    f'Features: {", ".join(_brain.feature_names)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Transition matrix display
                _tm = _brain.transmat
                _n  = _brain.n_states
                _rows_html = ""
                for i in range(_n):
                    def _fmt_tm(v):
                        pct = v * 100
                        if pct >= 1:
                            return f"{pct:.1f}%"
                        elif pct > 0:
                            return "<1%"
                        return "0%"
                    _row_cells = "".join(
                        f'<td style="padding:3px 8px;font-size:10px;color:#94a3b8;'
                        f'background:rgba(255,255,255,{float(_tm[i][j])*0.15:.2f});'
                        f'text-align:center;">{_fmt_tm(_tm[i][j])}</td>'
                        for j in range(_n)
                    )
                    _state_col = _hmm_col(_brain.state_labels[i])
                    _rows_html += (
                        f'<tr><td style="padding:3px 8px;font-size:9px;color:{_state_col};'
                        f'font-weight:700;white-space:nowrap;">{_brain.state_labels[i]}</td>'
                        f'{_row_cells}</tr>'
                    )
                _col_headers = "".join(
                    f'<th style="padding:3px 8px;font-size:8px;color:#475569;'
                    f'font-weight:700;text-align:center;">{lbl[:4]}</th>'
                    for lbl in _brain.state_labels
                )
                st.markdown(
                    f'<div style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">TRANSITION MATRIX</div>'
                    f'<table style="border-collapse:collapse;background:#0a0a14;'
                    f'border-radius:4px;overflow:hidden;">'
                    f'<thead><tr><th style="padding:3px 8px;"></th>{_col_headers}</tr></thead>'
                    f'<tbody>{_rows_html}</tbody></table>'
                    f'<div style="font-size:9px;color:#334155;margin-top:4px;">'
                    f'Row = current state → columns = next-day probability</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="color:#ef444466;font-size:12px;padding:8px 0;">'
                    'No brain trained yet. Click Retrain to build the HMM regime model.</div>',
                    unsafe_allow_html=True,
                )

            # ── Retrain due? ──────────────────────────────────────────────
            _retrain_due = False
            _days_since   = None
            if _brain:
                try:
                    from datetime import datetime, timezone as _tz
                    _trained_dt = datetime.fromisoformat(_brain.trained_at.replace("Z", "+00:00"))
                    _days_since = (datetime.now(_tz.utc) - _trained_dt).days
                    _retrain_due = _days_since >= 90
                except Exception:
                    pass

            if _retrain_due:
                st.markdown(
                    f'<div style="background:#1a0a00;border:1px solid #f59e0b;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;display:flex;align-items:center;gap:10px;">'
                    f'<span style="font-size:18px;">⏰</span>'
                    f'<div>'
                    f'<div style="font-size:11px;font-weight:700;color:#f59e0b;">Quarterly retrain due</div>'
                    f'<div style="font-size:10px;color:#92400e;">'
                    f'Last trained {_days_since} days ago — click Retrain HMM to refresh the model</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<style>'
                    '#btn_hmm_retrain_wrapper button {'
                    '  box-shadow: 0 0 8px #f59e0b, 0 0 20px #f59e0b88 !important;'
                    '  border-color: #f59e0b !important;'
                    '}'
                    '</style>',
                    unsafe_allow_html=True,
                )

            _retrain_label = "🔄 Retrain HMM" if not _retrain_due else "🔄 Retrain HMM ⚠️"
            _retrain_help  = (
                f"Quarterly retrain due — {_days_since}d since last train"
                if _retrain_due else
                "Train GaussianHMM on 15yr FRED data. Run quarterly."
            )

            _hm_c1, _hm_c2 = st.columns([1, 2])
            with _hm_c1:
                if st.button(_retrain_label, key="btn_hmm_retrain", use_container_width=True,
                             help=_retrain_help, type="primary" if _retrain_due else "secondary"):
                    with st.spinner("Training HMM on 15yr FRED signal matrix..."):
                        try:
                            _new_brain = _train_hmm(lookback_years=15)
                            _new_state = _score_hmm(_new_brain, log_to_history=True)
                            _st_lbl = _new_state.state_label if _new_state else "unknown"
                            st.toast(
                                f"✅ HMM trained — {_new_brain.n_states} states (BIC {_new_brain.bic:,.0f}) "
                                f"· Today: {_st_lbl}",
                                icon="🧠",
                            )
                            st.rerun()
                        except Exception as _e:
                            st.error(f"HMM training failed: {_e}")
            with _hm_c2:
                if st.button("📍 Score Today", key="btn_hmm_score", use_container_width=True,
                             help="Run inference on today's signal vector (no refit)"):
                    if _brain is None:
                        st.warning("Train the model first (Retrain HMM).")
                    else:
                        with st.spinner("Scoring today's state..."):
                            _ts = _score_hmm(_brain, log_to_history=True)
                            if _ts:
                                st.toast(
                                    f"📍 {_ts.state_label} · {int(_ts.confidence*100)}% confidence "
                                    f"· {_ts.persistence}d persistence",
                                    icon="🧠",
                                )
                                st.rerun()
                            else:
                                st.error("Scoring failed — check console for details.")
        except ImportError:
            st.error("hmmlearn not installed. Run: pip install hmmlearn")
        except Exception as _hmm_e:
            st.error(f"HMM section error: {_hmm_e}")

    # ── Data Flow Legend ───────────────────────────────────────────────────────
    with st.expander("📊 Data Flow", expanded=False):
        _oc = COLORS["bloomberg_orange"]
        st.markdown(
            f"""
<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;line-height:1.9;">

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">DATA SOURCES</div>
<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">
  {''.join(f'<span style="background:#1e293b;border:1px solid #334155;border-radius:3px;padding:2px 8px;color:#94a3b8;">{s}</span>' for s in ['yfinance (prices, options, fundamentals)','FRED API (macro series)','RSS / news feeds','SEC EDGAR (13F, 13D, Form 4)','StockTwits social sentiment','📱 Telegram inbox','Polymarket Gist (Black Swans)','alternative.me (Fear & Greed)','AAII.com (investor survey)','FederalReserve.gov RSS (FedSpeak)'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">QIR ROUND 1 — parallel <span style="font-weight:400;color:#475569;">(runs simultaneously)</span></div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:6px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Regime + Quadrant','Tactical Regime','Current Events Digest (+ FedSpeak RSS)','Options Flow (SPY macro)','Whale Activity (13F)','Activism Digest (13D)','Social Sentiment (StockTwits)','Market Sentiment (F&G · AAII · VIX Curve)'])}
</div>
<div style="color:#64748b;font-size:10px;margin-bottom:2px;">↳ then sequentially:</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0d1a0d;border:1px solid #22c55e44;border-radius:3px;padding:3px 8px;color:#22c55e88;">{s}</div>' for s in ['Sector×Regime Digest (needs regime quadrant)'])}
</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">QIR ROUND 2 — parallel</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Fed Rate Path + Plays','Doom Briefing','Black Swans'])}
</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">QIR ROUND 3</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Policy Transmission','Trending Narratives','Auto-Trending Groups'])}
</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">QIR ROUND 4 + 5</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:14px;">
  {''.join(f'<div style="background:#0f172a;border:1px solid {_oc}44;border-radius:3px;padding:3px 8px;color:{_oc};">{s}</div>' for s in ['Macro Conviction Synopsis (all signals → coherence verdict)','Portfolio Risk Snapshot (beta · VaR · stress · flags)'])}
</div>

<div style="color:#334155;font-size:16px;margin-bottom:8px;padding-left:4px;">↓</div>

<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">SIGNAL CONSUMERS</div>
<div style="display:flex;flex-direction:column;gap:4px;">
  <div style="background:#0c1a0c;border:1px solid #22c55e44;border-radius:4px;padding:6px 10px;">
    <span style="color:#22c55e;font-weight:700;">Portfolio Intelligence</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · tactical · rate path · doom · whales · activism · sector×regime · options flow · news · swans · trending · auto-trending · risk snapshot · social sentiment · F&G · AAII · VIX curve</span>
  </div>
  <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;padding:6px 10px;">
    <span style="color:#f59e0b;font-weight:700;">Discovery (Cross-Signal Plays)</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · tactical · rate path · fed plays · regime plays · doom · whales · activism · sector×regime · options flow · news · swans · trending · auto-trending · risk snapshot · F&G · AAII · VIX curve</span>
  </div>
  <div style="background:#0d1117;border:1px solid #3b82f644;border-radius:4px;padding:6px 10px;">
    <span style="color:#3b82f6;font-weight:700;">Valuation (AI Rating)</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">regime · tactical · rate path · fed plays · doom · whales · activism · sector×regime · macro options flow · news · swans · trending · auto-trending · per-ticker options/insider/congress/institutional · portfolio risk · F&G · AAII · VIX curve</span>
  </div>
  <div style="background:#120d1a;border:1px solid #a855f744;border-radius:4px;padding:6px 10px;">
    <span style="color:#a855f7;font-weight:700;">Tactical Regime (9 signals)</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">VIX level · VIX term structure · SPY MAs · momentum · breadth · VIX full curve · CBOE SKEW · Fear&Greed (contrarian) · AAII (contrarian)</span>
  </div>
  <div style="background:#120d1a;border:1px solid #a855f744;border-radius:4px;padding:6px 10px;">
    <span style="color:#a855f7;font-weight:700;">Signal Audit</span>
    <span style="color:#475569;font-size:10px;margin-left:8px;">tracks all signals — age · engine · preview — with staleness warnings</span>
  </div>
</div>

<div style="margin-top:12px;padding-top:8px;border-top:1px solid #1e293b;color:#475569;font-size:10px;line-height:1.8;">
  📱 Telegram inbox → Current Events Digest + FedSpeak RSS (Fed.gov) → all consumers<br>
  🔄 Sector×Regime = 11 SPDR ETFs momentum × Dalio quadrant → confirms or flags divergence<br>
  📡 Options Flow (SPY) = macro P/C ratio + gamma exposure + put wall → market positioning layer<br>
  😱 Fear & Greed (alternative.me) + AAII survey → contrarian signals in Tactical Regime + Valuation<br>
  📈 VIX Term Structure (9D/VIX/3M/6M) → full vol curve shape → Tactical Regime signal
</div>

</div>""",
            unsafe_allow_html=True,
        )
