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
from services.hmm_regime import get_ci_anchor as _ci_anchor


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
    # ── Named GU sub-patterns: 2-of-3 agree, 1 neutral ──────────────────────
    "MOMENTUM_BUILDING": {
        "label": "MOMENTUM BUILDING",
        "color": "#22c55e",
        "bullish": True,
        "interpretation": "Regime and Tactical both bullish — Options Flow hasn't confirmed yet. Two of three layers aligned. The trade is valid but wait for options to follow.",
        "buy_tier": "MODERATE",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("SPY / QQQ", "Broad market — two layers agree, reduced size until options confirm"),
            ("Sector leaders", "Highest momentum names in the current regime"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Enter at 60–70% of normal size — wait for options ≥65 to add\n"
            "Stop: close below 20d MA\n"
            "Target: prior swing high"
        ),
        "entry_short": "Two layers bullish — avoid shorts until regime breaks.",
    },
    "MACRO_FLOW_BULLISH": {
        "label": "MACRO + FLOW BULLISH",
        "color": "#22c55e",
        "bullish": True,
        "interpretation": "Regime and Options Flow agree bullish — Tactical is in neutral (price hasn't moved yet). Smart money and macro aligned; price action confirming is the final trigger.",
        "buy_tier": "MODERATE",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("SPY", "Broad market — macro + flow lead, price confirmation pending"),
            ("SPY Calls (OTM)", "Options flow is already bullish — capture the pending move with leverage"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Enter 50–70% size — wait for Tactical ≥65 to go full size\n"
            "Stop: -4% from entry or break of regime support\n"
            "Options flow leading → price often follows within 3–5 sessions"
        ),
        "entry_short": "Macro and flow are bullish — no short setup.",
    },
    "TACTICAL_FLOW_SURGE": {
        "label": "TACTICAL + FLOW SURGE",
        "color": "#f59e0b",
        "bullish": True,
        "interpretation": "Short-term price and Options Flow both bullish — but macro regime hasn't confirmed. Could be early in a turn or a counter-trend move. Treat as tactical, not structural.",
        "buy_tier": "SELECTIVE",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("QQQ / SPY", "Short-duration momentum trade — not a structural long"),
            ("High-beta names", "Amplified upside if macro confirms — tight stops required"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Tactical trade only — 40–60% size, tight stops\n"
            "Stop: -3% from entry — exit if regime stays neutral after 5 sessions\n"
            "Add size only when regime confirms ≥+0.3"
        ),
        "entry_short": "Flow and tactical are up — no short case yet.",
    },
    "SELLING_PRESSURE": {
        "label": "SELLING PRESSURE",
        "color": "#ef4444",
        "bullish": False,
        "interpretation": "Regime and Tactical both bearish — Options Flow hasn't confirmed yet. Bearish bias is clear; options confirmation would lock in the short case.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "MODERATE",
        "instruments_buy": [],
        "instruments_short": [
            ("SH", "1× inverse S&P — hold until options confirm the bear"),
            ("Cash", "Defensive — wait for all three layers before pressing shorts"),
        ],
        "entry_buy": "Two layers bearish — avoid new longs. Raise cash.",
        "entry_short": (
            "60–70% short size — wait for options <38 to add conviction\n"
            "Stop: close above last swing high\n"
            "Target: prior support level"
        ),
    },
    "MACRO_FLOW_BEARISH": {
        "label": "MACRO + FLOW BEARISH",
        "color": "#ef4444",
        "bullish": False,
        "interpretation": "Regime and Options Flow both bearish — Tactical hasn't rolled over yet. Smart money hedging while price is still holding. Expect tactical to follow.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "MODERATE",
        "instruments_buy": [],
        "instruments_short": [
            ("SH / SPY Puts", "Defined risk — price hasn't broken yet, wait for Tactical <38 to size up"),
            ("Collars on longs", "Protect existing positions — don't fully exit until tactical breaks"),
        ],
        "entry_buy": "Macro and flow both bearish — protect longs, no new buys.",
        "entry_short": (
            "50–70% size — tactical breakdown is the add trigger\n"
            "Stop: close above prior resistance\n"
            "Price often follows flow within 3–5 sessions"
        ),
    },
    "FLOW_BREAKDOWN": {
        "label": "FLOW BREAKDOWN",
        "color": "#f97316",
        "bullish": False,
        "interpretation": "Tactical price and Options Flow both bearish — macro regime hasn't confirmed the move. Could be a short-term flush or early warning of regime change. Stay defensive.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "SELECTIVE",
        "instruments_buy": [],
        "instruments_short": [
            ("Cash / SH", "Reduce exposure — wait for regime to confirm before pressing short"),
            ("VXX small", "Vol hedge in case regime breaks — defined cost"),
        ],
        "entry_buy": "Flow and tactical both down — hold cash, not longs.",
        "entry_short": (
            "Tactical trade only — 40% size until regime breaks <-0.3\n"
            "Stop: close above 20d MA\n"
            "Exit if regime stays neutral — may be a washout, not a trend"
        ),
    },
    # ── Named GU sub-patterns: 3-way conflict (Bull/Bear/Bear or Bear/Bull/Bear) ──
    "DISTRIBUTION": {
        "label": "DISTRIBUTION",
        "color": "#ef4444",
        "bullish": False,
        "interpretation": "Macro regime still bullish but Tactical and Options Flow both bearish — classic distribution. Smart money selling into strength while price holds. Rotate defensive.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "STRONG",
        "instruments_buy": [],
        "instruments_short": [
            ("SPY Puts (OTM)", "Distribution → price break incoming — defined risk"),
            ("SH / SDS", "Build short in tranches as tactical stays bearish"),
            ("Collars on longs", "Protect existing positions before the break"),
        ],
        "entry_buy": "Regime looks bullish but price and flow say otherwise — trust the flow.",
        "entry_short": (
            "Build position on weak bounces — don't short freefall\n"
            "Stop: close above prior swing high\n"
            "Add on each failed rally attempt"
        ),
    },
    "ACCUMULATION": {
        "label": "ACCUMULATION",
        "color": "#22c55e",
        "bullish": True,
        "interpretation": "Macro regime bearish but Tactical and Options Flow both bullish — quiet accumulation against the trend. Risk: macro doesn't turn. Reward: early in a new cycle.",
        "buy_tier": "SELECTIVE",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("SPY / QQQ", "Accumulate at reduced size — let regime confirm before going full size"),
            ("Quality growth", "MSFT, GOOGL — least likely to re-break if macro stays bearish"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "50% size only — wait for regime to turn ≥+0.3 before adding\n"
            "Stop: -5% from entry or regime re-breaks lower\n"
            "This is early-cycle risk — patience required"
        ),
        "entry_short": "Tactical and flow are bullish — no short setup while accumulation is active.",
    },
    # ── Named GU sub-patterns: 1 signal only ────────────────────────────────
    "REGIME_ONLY_BULLISH": {
        "label": "REGIME ONLY BULLISH",
        "color": "#64748b",
        "bullish": True,
        "interpretation": "Only the macro regime is bullish — Tactical and Options Flow are flat. Macro has moved but price and options haven't followed. Wait for confirmation before committing.",
        "buy_tier": "WATCH",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("Cash-secured calls (far OTM)", "Cheap participation if macro is right — defined risk"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "20–30% size only — one layer is not a trade\n"
            "Wait for Tactical ≥65 OR Options ≥65 before adding\n"
            "Stop: regime breaks below +0.3"
        ),
        "entry_short": "Macro bullish — no short case.",
    },
    "REGIME_ONLY_BEARISH": {
        "label": "REGIME ONLY BEARISH",
        "color": "#64748b",
        "bullish": False,
        "interpretation": "Only the macro regime is bearish — price and options haven't confirmed. Raise cash, hold off on new longs, but don't press short until tactical joins.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "WATCH",
        "instruments_buy": [],
        "instruments_short": [
            ("Cash / reduced exposure", "Defensive posture — wait for tactical to break"),
        ],
        "entry_buy": "Macro bearish — raise cash, no new longs.",
        "entry_short": (
            "20–30% size — one layer is not enough\n"
            "Add when Tactical <38 OR Options <38"
        ),
    },
    "TACTICAL_ONLY_BULLISH": {
        "label": "TACTICAL ONLY BULLISH",
        "color": "#64748b",
        "bullish": True,
        "interpretation": "Price action is bullish short-term but macro and options are flat. Could be a short squeeze or intraday momentum. Treat as scalp, not a trend trade.",
        "buy_tier": "SCALP ONLY",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("SPY / QQQ (small)", "Short-duration scalp — no structural backing"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Scalp only — 20–30% size, 1–3 day hold max\n"
            "Stop: -2% hard stop\n"
            "Exit if regime stays flat after 3 sessions"
        ),
        "entry_short": "Tactical up — no short setup.",
    },
    "TACTICAL_ONLY_BEARISH": {
        "label": "TACTICAL ONLY BEARISH",
        "color": "#64748b",
        "bullish": False,
        "interpretation": "Price action is bearish short-term but macro and flow are flat. May be profit-taking or a brief flush. Hold cash — don't chase the drop.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "SCALP ONLY",
        "instruments_buy": [],
        "instruments_short": [
            ("Cash", "Hold off — one layer is not enough for a structural short"),
        ],
        "entry_buy": "Tactical down — hold cash, wait for clarity.",
        "entry_short": (
            "Scalp only — 20% size, tight stop above recent high\n"
            "Exit if macro or flow don't confirm within 2–3 sessions"
        ),
    },
    "FLOW_ONLY_BULLISH": {
        "label": "FLOW ONLY BULLISH",
        "color": "#64748b",
        "bullish": True,
        "interpretation": "Options flow is buying protection or calls but macro and tactical are flat. Flow often leads price by 3–5 sessions — watch for regime and tactical to follow.",
        "buy_tier": "WATCH",
        "short_tier": "NOT A SHORTING ENV",
        "instruments_buy": [
            ("SPY Calls (OTM, small)", "Flow is your early warning — 20–30% size"),
        ],
        "instruments_short": [],
        "entry_buy": (
            "Small position — flow leads, price follows\n"
            "Add when Tactical ≥65 confirms\n"
            "Stop: flow reverses below 50"
        ),
        "entry_short": "Options flow bullish — no short case.",
    },
    "FLOW_ONLY_BEARISH": {
        "label": "FLOW ONLY BEARISH",
        "color": "#64748b",
        "bullish": False,
        "interpretation": "Options flow is buying puts / hedging but macro and price are flat. Smart money is quietly hedging — take note. May be noise or may be early warning.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "WATCH",
        "instruments_buy": [],
        "instruments_short": [
            ("Collars on longs", "Flow is hedging — protect your book without full exit"),
        ],
        "entry_buy": "Flow bearish — no new longs, protect existing positions.",
        "entry_short": (
            "Collars or small puts only — flow alone is not a directional trade\n"
            "Add if Tactical or regime confirms"
        ),
    },
    "TRUE_NEUTRAL": {
        "label": "TRUE NEUTRAL",
        "color": "#475569",
        "bullish": None,
        "interpretation": "All three layers are in the neutral zone — no signal anywhere. Market is in a holding pattern. Best action: wait. Worst action: force a trade.",
        "buy_tier": "NO SIGNAL",
        "short_tier": "NO SIGNAL",
        "instruments_buy": [
            ("Cash / T-bills", "Earn the risk-free rate while waiting for a signal"),
        ],
        "instruments_short": [],
        "entry_buy": "No layer active — stay in cash. Don't create signal where there is none.",
        "entry_short": "No layer active — no short case either.",
    },
    # ── Named GU sub-patterns: conflict with one neutral ─────────────────────
    "MACRO_VS_PRICE": {
        "label": "MACRO VS PRICE",
        "color": "#f59e0b",
        "bullish": None,
        "interpretation": "Regime bullish, Tactical bearish, Options flat. Macro says buy — price says sell. Classic divergence. Wait for one to break.",
        "buy_tier": "WATCH",
        "short_tier": "WATCH",
        "instruments_buy": [
            ("Cash", "Wait — price needs to confirm macro before committing"),
        ],
        "instruments_short": [],
        "entry_buy": "Wait for Tactical ≥65 before buying — price hasn't confirmed macro.",
        "entry_short": "Wait for regime to break <-0.3 before shorting — macro hasn't broken.",
    },
    "MACRO_VS_FLOW": {
        "label": "MACRO VS FLOW",
        "color": "#f59e0b",
        "bullish": None,
        "interpretation": "Regime bullish, Options bearish, Tactical flat. Macro says risk-on — options crowd is hedging. Smart money and fundamentals in a standoff.",
        "buy_tier": "WATCH",
        "short_tier": "WATCH",
        "instruments_buy": [],
        "instruments_short": [
            ("Collars on longs", "Options are hedging — protect existing positions"),
        ],
        "entry_buy": "Wait for options ≥65 before adding longs — flow is warning you.",
        "entry_short": "Regime still bullish — no outright short. Collars only.",
    },
    "PRICE_VS_FLOW": {
        "label": "PRICE VS FLOW",
        "color": "#f59e0b",
        "bullish": None,
        "interpretation": "Tactical bullish, Options bearish, Regime flat. Price is moving up but options crowd is hedging — short-term momentum vs smart money caution.",
        "buy_tier": "SCALP ONLY",
        "short_tier": "WATCH",
        "instruments_buy": [
            ("SPY (small)", "Tactical scalp only — options crowd is warning against size"),
        ],
        "instruments_short": [],
        "entry_buy": "Scalp only — 20–30% size, tight stop, options crowd is not with you.",
        "entry_short": "Wait for tactical to roll over before shorting.",
    },
    "BEAR_BOUNCE_WARNING": {
        "label": "BEAR BOUNCE WARNING",
        "color": "#f97316",
        "bullish": False,
        "interpretation": "Regime bearish, Tactical bullish, Options flat. Classic bear market bounce forming — don't chase. The trend is still down.",
        "buy_tier": "NOT A BUYING ENV",
        "short_tier": "MODERATE",
        "instruments_buy": [],
        "instruments_short": [
            ("SH", "Build short — sell into the bounce, not the freefall"),
            ("SPY Puts (OTM)", "Defined risk on bounce failure"),
        ],
        "entry_buy": "Regime is bearish — this bounce is a selling opportunity, not a buy.",
        "entry_short": (
            "Sell into strength — wait for tactical to roll back below 65\n"
            "Stop: above bounce high\n"
            "Add if options confirms <38"
        ),
    },
    "FLOW_DEFIES_MACRO": {
        "label": "FLOW DEFIES MACRO",
        "color": "#f97316",
        "bullish": None,
        "interpretation": "Regime bearish, Options bullish, Tactical flat. Options market is betting on a recovery while macro says no. High risk setup — could be early bottom or a trap.",
        "buy_tier": "WATCH",
        "short_tier": "WATCH",
        "instruments_buy": [
            ("Cash-secured puts (OTM)", "Collect premium — undefined upside for option sellers if flow is right"),
        ],
        "instruments_short": [],
        "entry_buy": "Flow leading macro is a high-risk bet — 20% size max, wait for tactical ≥65.",
        "entry_short": "Options bullish — covering the macro short here is risky. Trail stops up.",
    },
    "FLOW_VS_PRICE": {
        "label": "FLOW VS PRICE",
        "color": "#f59e0b",
        "bullish": None,
        "interpretation": "Tactical bearish, Options bullish, Regime flat. Price is falling but options crowd is buying. Either flow is early or price is capitulating into a bottom.",
        "buy_tier": "WATCH",
        "short_tier": "WATCH",
        "instruments_buy": [
            ("Small SPY", "If flow is right — small position, price must stop falling first"),
        ],
        "instruments_short": [],
        "entry_buy": "Wait for price to stabilize (Tactical ≥50) before trusting the flow signal.",
        "entry_short": "Options crowd bullish — don't add to shorts here.",
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
    r_neut = not r_bull and not r_bear
    t_bull = tac_score >= 65
    t_bear = tac_score <  38
    t_neut = not t_bull and not t_bear
    o_bull = of_score  >= 65
    o_bear = of_score  <  38
    o_neut = not o_bull and not o_bear

    # ── 3-of-3 named patterns ─────────────────────────────────────────────────
    if   r_bull and t_bull and o_bull: pattern = "BULLISH_CONFIRMATION"
    elif r_bear and t_bear and o_bear: pattern = "BEARISH_CONFIRMATION"
    elif r_bull and t_bear and o_bull: pattern = "PULLBACK_IN_UPTREND"
    elif r_bull and t_bull and o_bear: pattern = "OPTIONS_FLOW_DIVERGENCE"
    elif r_bear and t_bull and o_bull: pattern = "BEAR_MARKET_BOUNCE"
    elif r_bear and t_bear and o_bull: pattern = "LATE_CYCLE_SQUEEZE"
    # ── Bull/Bear/Bear and Bear/Bull/Bear conflicts ───────────────────────────
    elif r_bull and t_bear and o_bear: pattern = "DISTRIBUTION"
    elif r_bear and t_bull and o_bear: pattern = "ACCUMULATION"
    # ── 2-of-3 agree, 1 neutral ───────────────────────────────────────────────
    elif r_bull and t_bull and o_neut: pattern = "MOMENTUM_BUILDING"
    elif r_bull and o_bull and t_neut: pattern = "MACRO_FLOW_BULLISH"
    elif t_bull and o_bull and r_neut: pattern = "TACTICAL_FLOW_SURGE"
    elif r_bear and t_bear and o_neut: pattern = "SELLING_PRESSURE"
    elif r_bear and o_bear and t_neut: pattern = "MACRO_FLOW_BEARISH"
    elif t_bear and o_bear and r_neut: pattern = "FLOW_BREAKDOWN"
    # ── 1 signal only ─────────────────────────────────────────────────────────
    elif r_bull and t_neut and o_neut: pattern = "REGIME_ONLY_BULLISH"
    elif r_bear and t_neut and o_neut: pattern = "REGIME_ONLY_BEARISH"
    elif t_bull and r_neut and o_neut: pattern = "TACTICAL_ONLY_BULLISH"
    elif t_bear and r_neut and o_neut: pattern = "TACTICAL_ONLY_BEARISH"
    elif o_bull and r_neut and t_neut: pattern = "FLOW_ONLY_BULLISH"
    elif o_bear and r_neut and t_neut: pattern = "FLOW_ONLY_BEARISH"
    # ── Conflict with one neutral ─────────────────────────────────────────────
    elif r_bull and t_bear and o_neut: pattern = "MACRO_VS_PRICE"
    elif r_bull and o_bear and t_neut: pattern = "MACRO_VS_FLOW"
    elif t_bull and o_bear and r_neut: pattern = "PRICE_VS_FLOW"
    elif r_bear and t_bull and o_neut: pattern = "BEAR_BOUNCE_WARNING"
    elif r_bear and o_bull and t_neut: pattern = "FLOW_DEFIES_MACRO"
    elif t_bear and o_bull and r_neut: pattern = "FLOW_VS_PRICE"
    # ── True neutral (all flat) ───────────────────────────────────────────────
    elif r_neut and t_neut and o_neut: pattern = "TRUE_NEUTRAL"
    # ── Residual genuine uncertainty (should be unreachable now) ──────────────
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
    forced_lean_score: float = 50.0,
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
        # Forced directional lean (from Uncertainty panel / Lean Tracker) breaks the HOLD tie.
        # Guards: don't buy a rip (tac ≥ 55) or into weak macro (< 45); no guard needed on WAIT.
        elif forced_lean_score >= 55 and tactical_score < 55 and macro_score >= 45:
            verdict = "BUY THE DIP"
        elif forced_lean_score <= 45:
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

        _OPTIONS_KEYWORDS = (
            "call", "put", "uvxy", "vxx", "collar", "straddle", "strangle",
            "otm", "atm", "dte", "premium", "vol insurance",
        )

        def _is_options_instrument(ticker: str, desc: str) -> bool:
            combined = (ticker + " " + desc).lower()
            return any(kw in combined for kw in _OPTIONS_KEYWORDS)

        def _instruments_html(instruments):
            if not instruments:
                return ""
            filtered = [(t, d) for t, d in instruments if not _is_options_instrument(t, d)]
            if not filtered:
                return (
                    f'<div style="font-size:9px;color:#f59e0b;font-weight:700;'
                    f'letter-spacing:0.06em;margin:6px 0 2px;">INSTRUMENTS</div>'
                    f'<div style="font-size:10px;color:#475569;margin-bottom:6px;font-style:italic;">'
                    f'Options flow is the primary signal — no equity instrument suggested. '
                    f'Monitor flow via the Options Activity module before acting.</div>'
                )
            rows = "".join(
                f'<div style="padding:2px 0;border-bottom:1px solid #1e293b;">'
                f'<span style="color:#f1f5f9;font-weight:700;font-size:10px;">{t}</span>'
                f'<span style="color:#94a3b8;font-size:10px;"> — {d}</span></div>'
                for t, d in filtered[:4]
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

        # ── Kritzman contagion adjustment (Skulls, FAJ 2010) ─────────────────
        # Calm Sharpe +0.41 vs turbulent -0.58 → crisis signal value = 0.41/(0.41+0.58)
        _kritzman_mult = 1.0
        _kritzman_label = ""
        try:
            from services.market_data import fetch_correlation_matrix as _fcm_k
            import numpy as _np_k
            _k_corr = _fcm_k(("SPY","QQQ","TLT","GLD","UUP","HYG","^VIX","USO","EEM"), period="1mo")
            if _k_corr is not None and not _k_corr.empty:
                _k_mask = ~_np_k.eye(len(_k_corr), dtype=bool)
                _k_score = min(100.0, _np_k.abs(_k_corr.values[_k_mask]).mean() / 0.8 * 100.0)
                if _k_score >= 80:
                    _kritzman_mult = 0.41
                elif _k_score >= 60:
                    _kritzman_mult = 0.60
                elif _k_score >= 30:
                    _kritzman_mult = 0.80
                if _kritzman_mult < 1.0:
                    _kritzman_label = f"×{_kritzman_mult:.2f} (contagion {_k_score:.0f})"
        except Exception:
            pass

        # Entry signal recommendation (leading vs lagging synthesis)
        _leading_s = int(_rc.get("leading_score") or 50)
        _macro_s   = int(_rc.get("macro_score") or 50)
        _div_pts   = int(_rc.get("leading_divergence") or 0)
        _div_label = _rc.get("leading_label") or "Aligned"
        _tac_s     = int(_tac.get("tactical_score", 50)) if _tac else 50
        _opts_s    = int(_of.get("options_score", 50))   if _of  else 50
        # Velocity: prefer 5-day macro trend (wider signal); fall back to 1-day delta
        _vel_for_entry = int(_rc.get("score_5d_trend") or _rc.get("velocity") or 0)
        # Forced directional lean — same weighted formula as Lean Tracker (macro 30% / tac 25% /
        # opts 20% / sent+event 25% held neutral when unavailable). Prefer the richer GU profile
        # lean when it exists, converting its magnitude+direction back to a 0-100 directional score.
        _fl_base = _macro_s * 0.30 + _tac_s * 0.25 + _opts_s * 0.20 + 50 * 0.25
        if _gu_profile and _gu_profile.get("lean") == "BULLISH":
            _forced_lean_score = float(_gu_profile.get("lean_pct") or 50)
        elif _gu_profile and _gu_profile.get("lean") == "BEARISH":
            _forced_lean_score = 100.0 - float(_gu_profile.get("lean_pct") or 50)
        else:
            _forced_lean_score = _fl_base
        _entry_rec = _classify_entry_recommendation(
            _leading_s, _macro_s, _tac_s, _opts_s, _div_label, _div_pts,
            velocity_delta=_vel_for_entry,
            forced_lean_score=_forced_lean_score,
        )

        _verdict_html = (
            f'<div style="border-top:1px solid #1e293b;margin:10px 0 8px;"></div>'
            f'<div style="font-size:13px;font-weight:800;color:{_verdict_color};'
            f'letter-spacing:0.04em;margin-bottom:4px;">{_verdict_label}</div>'
            f'<div style="color:#94a3b8;font-size:11px;margin-bottom:6px;">{_verdict_interp}</div>'
        )

        # Signal state tip — shows Regime / Tactical / Options states for this pattern
        _SIG_STATE_TIPS = {
            "BULLISH_CONFIRMATION":     ("↑ Bull", "↑ Bull", "↑ Bull"),
            "BEARISH_CONFIRMATION":     ("↓ Bear", "↓ Bear", "↓ Bear"),
            "PULLBACK_IN_UPTREND":      ("↑ Bull", "↓ Bear", "↑ Bull"),
            "OPTIONS_FLOW_DIVERGENCE":  ("↑ Bull", "↑ Bull", "↓ Bear"),
            "BEAR_MARKET_BOUNCE":       ("↓ Bear", "↑ Bull", "↑ Bull"),
            "LATE_CYCLE_SQUEEZE":       ("↓ Bear", "↓ Bear", "↑ Bull"),
            "GENUINE_UNCERTAINTY":      ("? Mix",  "? Mix",  "? Mix"),
            "MOMENTUM_BUILDING":        ("↑ Bull", "↑ Bull", "→ Neut"),
            "MACRO_FLOW_BULLISH":       ("↑ Bull", "→ Neut", "↑ Bull"),
            "TACTICAL_FLOW_SURGE":      ("→ Neut", "↑ Bull", "↑ Bull"),
            "SELLING_PRESSURE":         ("↓ Bear", "↓ Bear", "→ Neut"),
            "MACRO_FLOW_BEARISH":       ("↓ Bear", "→ Neut", "↓ Bear"),
            "FLOW_BREAKDOWN":           ("→ Neut", "↓ Bear", "↓ Bear"),
            "DISTRIBUTION":             ("↑ Bull", "↓ Bear", "↓ Bear"),
            "ACCUMULATION":             ("↓ Bear", "↑ Bull", "↓ Bear"),
            "REGIME_ONLY_BULLISH":      ("↑ Bull", "→ Neut", "→ Neut"),
            "REGIME_ONLY_BEARISH":      ("↓ Bear", "→ Neut", "→ Neut"),
            "TACTICAL_ONLY_BULLISH":    ("→ Neut", "↑ Bull", "→ Neut"),
            "TACTICAL_ONLY_BEARISH":    ("→ Neut", "↓ Bear", "→ Neut"),
            "FLOW_ONLY_BULLISH":        ("→ Neut", "→ Neut", "↑ Bull"),
            "FLOW_ONLY_BEARISH":        ("→ Neut", "→ Neut", "↓ Bear"),
            "MACRO_VS_PRICE":           ("↑ Bull", "↓ Bear", "→ Neut"),
            "MACRO_VS_FLOW":            ("↑ Bull", "→ Neut", "↓ Bear"),
            "PRICE_VS_FLOW":            ("→ Neut", "↑ Bull", "↓ Bear"),
            "BEAR_BOUNCE_WARNING":      ("↓ Bear", "↑ Bull", "→ Neut"),
            "FLOW_DEFIES_MACRO":        ("↓ Bear", "→ Neut", "↑ Bull"),
            "FLOW_VS_PRICE":            ("→ Neut", "↓ Bear", "↑ Bull"),
            "TRUE_NEUTRAL":             ("→ Neut", "→ Neut", "→ Neut"),
        }
        _st = _SIG_STATE_TIPS.get(_cls["pattern"], ("?", "?", "?"))
        def _sig_pill(label, val):
            _c = "#22c55e" if "Bull" in val else ("#ef4444" if "Bear" in val else "#475569")
            return (
                f'<span style="background:{_c}18;border:1px solid {_c}44;border-radius:3px;'
                f'padding:1px 6px;font-size:8px;font-weight:700;color:{_c};margin-right:4px;">'
                f'{label} {val}</span>'
            )
        _verdict_html += (
            f'<div style="margin-bottom:8px;">'
            + _sig_pill("REGIME", _st[0])
            + _sig_pill("TACTICAL", _st[1])
            + _sig_pill("OPTIONS", _st[2])
            + f'</div>'
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
                f'<div style="font-size:7px;color:#334155;">/100 · domain disagreement</div>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:7px;color:#334155;margin-top:5px;line-height:1.7;padding:0 2px;">'
                f'<b style="color:#3b4f6b;">Uncertainty</b> = how much the 5 signal domains disagree with each other '
                f'(macro · technical · options · sentiment · event risk). '
                f'High → layers conflict → size down. Not to be confused with Conviction (signal strength).'
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
        _ll_anchored_block = ""
        _hmm_block        = ""
        _shadow_block     = ""
        _top_block        = ""
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
                f'<div style="font-size:13px;color:#94a3b8;font-weight:800;letter-spacing:0.08em;margin-bottom:3px;">'
                f'CONVICTION <span style="color:{_cv_color2};">×</span> UNCERTAINTY</div>'
                f'<div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:28px;font-weight:900;color:{_cv_color2};line-height:1;">'
                f'{_conviction_score}</div>'
                f'<div style="font-size:8px;color:#64748b;">/100</div>'
                f'</div>'
                f'</div>'
                # Plain-English explanation
                f'<div style="background:#0a0f1a;border-left:3px solid {_cv_color2}44;'
                f'border-radius:0 4px 4px 0;padding:6px 10px;margin-bottom:8px;'
                f'font-size:10px;color:#64748b;line-height:1.6;">'
                + (
                    f'<span style="color:#22c55e;font-weight:700;">All layers agree — strong directional edge.</span> '
                    f'Regime, tactical, and options flow are aligned. Size up to full conviction.'
                    if _conviction_score >= 75 else
                    f'<span style="color:#f59e0b;font-weight:700;">Signals mostly aligned — moderate edge.</span> '
                    f'Most layers agree but one is hesitating. Trade at normal size, watch for confirmation.'
                    if _conviction_score >= 55 else
                    f'<span style="color:#f97316;font-weight:700;">Weak signal — mixed layers.</span> '
                    f'Some agreement but significant drag from conflicting inputs. Size down, use tighter stops.'
                    if _conviction_score >= 40 else
                    f'<span style="color:#ef4444;font-weight:700;">No edge — layers in conflict.</span> '
                    f'Regime, tactical, or options are pulling in opposite directions. Wait or size to minimum.'
                )
                + f'</div>'
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
                f'<div style="font-size:7px;color:#334155;margin-top:6px;line-height:1.7;">'
                f'<b style="color:#3b4f6b;">Conviction</b> = how strong the pattern signal is (signal amplitude). '
                f'High → size up. Low → reduce exposure regardless of direction. '
                f'Not to be confused with Uncertainty (domain agreement).'
                f'</div>'
                f'{_warn_row}'
                f'</div>'
            )
        elif _cls.get("pattern") == "GENUINE_UNCERTAINTY" and _gu_profile:
            # GU has no conviction score — show the uncertainty score as the sizing anchor instead
            _gu_unc_v   = _gu_profile.get("uncertainty_score", 75)
            _gu_szl     = _gu_profile.get("size_label", "30% SIZE")
            _gu_unc_c   = "#22c55e" if _gu_unc_v < 40 else ("#f59e0b" if _gu_unc_v < 65 else "#ef4444")
            _conviction_block = (
                f'<div style="background:#0f172a;border:1px solid #7c3aed44;'
                f'border-left:3px solid #7c3aed;border-radius:6px;padding:10px 14px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px;">'
                f'<div>'
                f'<div style="font-size:13px;color:#94a3b8;font-weight:800;letter-spacing:0.08em;margin-bottom:3px;">'
                f'CONVICTION <span style="color:#7c3aed;">×</span> UNCERTAINTY</div>'
                f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:16px;font-weight:900;color:#7c3aed;">N/A</div>'
                f'<div style="font-size:7px;color:#64748b;">GU mode</div>'
                f'</div>'
                f'</div>'
                # Plain-English GU explanation
                f'<div style="background:#0a0f1a;border-left:3px solid #7c3aed44;'
                f'border-radius:0 4px 4px 0;padding:6px 10px;margin-bottom:8px;'
                f'font-size:10px;color:#64748b;line-height:1.6;">'
                f'<span style="color:#7c3aed;font-weight:700;">Genuine Uncertainty — signals are pulling in opposite directions.</span> '
                f'Regime, tactical, and options flow do not form a clean pattern — at least one layer is in the middle band or contradicting the others. '
                f'Conviction score cannot be computed. Instead, the 5-domain uncertainty score below measures how much the layers disagree and sets your size.'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:10px;background:#0a0f1a;'
                f'border:1px solid #1e293b;border-radius:4px;padding:6px 10px;">'
                f'<div><div style="font-size:7px;color:#334155;font-weight:700;letter-spacing:0.08em;">UNCERTAINTY</div>'
                f'<div style="font-size:20px;font-weight:900;color:{_gu_unc_c};">{_gu_unc_v}</div>'
                f'<div style="font-size:7px;color:#334155;">/100 · domain disagreement</div></div>'
                f'<div style="flex:1;border-left:1px solid #1e293b;padding-left:10px;">'
                f'<div style="font-size:7px;color:#334155;margin-bottom:3px;">SIZING ANCHOR</div>'
                f'<div style="font-size:12px;font-weight:800;color:#7c3aed;">→ {_gu_szl}</div>'
                f'<div style="font-size:7px;color:#334155;margin-top:2px;">from uncertainty penalty, not conviction</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
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
                    f'padding:6px 12px;margin:6px 0 4px;display:flex;align-items:center;'
                    f'flex-wrap:wrap;gap:4px 0;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-right:8px;">REGIME VELOCITY</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;margin-right:8px;">⏑ MEDIUM · DAYS/WEEKS</span>'
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
                    f'padding:6px 12px;margin:6px 0 4px;display:flex;align-items:center;'
                    f'flex-wrap:wrap;gap:4px 0;">'
                    f'<span style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-right:8px;">REGIME VELOCITY</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;margin-right:8px;">⏑ MEDIUM · DAYS/WEEKS</span>'
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
                f'padding:6px 12px;margin:6px 0 4px;display:flex;align-items:center;gap:8px;">'
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

        # ── Initialize Kelly vars (will be set inside try block; defaults if exception occurs) ──
        _kly_half = 0.0
        _kly_viable = False
        _kly_full = 0.0
        _kly_p = 0.5
        _kly_b = 1.0
        _kly_psrc = ""
        _kly_bsrc = ""
        _kly_stress = 0.0
        _kly_cap = False
        _kly_n_agree = 0
        _kly_n_total = 0
        _kly_align_m = 1.0
        _kly_hmm_m = 1.0
        _kly_sdirs = {}
        _triple_kelly_html = ""
        _tac_short_kelly_pct = 0.0
        _tac_long_kelly_pct = 0.0
        _net_kelly_pct = 0.0
        _lt_full_signed = 0.0
        _lt_half_signed_pct = 0.0

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

            _sh_for_kelly = st.session_state.get("_shadow_state_obj")
            _sh_label_kelly = getattr(_sh_for_kelly, "state_label", None) if _sh_for_kelly else None
            _kly = _compute_kelly(
                _conviction_score,
                st.session_state.get("_fear_composite") or {},
                st.session_state.get("_regime_context") or {},
                options_score=(st.session_state.get("_options_flow_context") or {}).get("options_score"),
                tactical_score=(st.session_state.get("_tactical_context") or {}).get("tactical_score"),
                hmm_state_label=_hmm_label_for_kelly,
                shadow_state_label=_sh_label_kelly,
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

            # ── Net Kelly (signed composite) ──────────────────────────────────────
            # Formula: 0.6 × LT_half_signed + 0.4 × (TacLong − TacShort)
            # LT leg uses raw Kelly formula (NO floor) so it goes negative in Risk-Off.
            # Negative result = SHORT setup; abs value = position size %.
            _lt_full_signed = ((_kly_b * _kly_p - (1.0 - _kly_p)) / _kly_b
                               if _kly_b > 0 else 0.0)
            _lt_half_signed_pct  = _lt_full_signed * 0.5 * 100
            _tac_net_pct_for_nk  = (_tac_long_kelly_pct or 0.0) - (_tac_short_kelly_pct or 0.0)
            _net_kelly_pct       = round(0.6 * _lt_half_signed_pct + 0.4 * _tac_net_pct_for_nk, 1)

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

            # ── Trade Stats pill ───────────────────────────────────────
            _kly_nw    = _kly.get("n_wins", 0)
            _kly_nl    = _kly.get("n_losses", 0)
            _kly_nc    = _kly.get("n_closed", 0)
            _kly_aw    = _kly.get("avg_win_pct", 0.0)
            _kly_al    = _kly.get("avg_loss_pct", 0.0)
            _kly_wr    = round(_kly_nw / _kly_nc * 100) if _kly_nc else 0
            _kly_wr_col = "#22c55e" if _kly_wr >= 55 else "#f59e0b" if _kly_wr >= 45 else "#ef4444"
            _b_needed   = max(0, 5 - min(_kly_nw, _kly_nl))
            _boot_pct   = min(100, int(_kly_nc / 10 * 100))
            _boot_col   = "#4B9FFF" if _boot_pct < 50 else "#f59e0b"
            _stats_pill_html = (
                f'<div style="background:#080d14;border:1px solid #1e293b33;border-radius:4px;'
                f'padding:6px 8px;margin:5px 0 4px;">'
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:4px;">'
                f'<span style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.08em;">TRADE STATS</span>'
                f'<span style="font-size:9px;color:#475569;">n={_kly_nc} closed</span>'
                + (
                    f'<span style="font-size:9px;color:{_kly_wr_col};font-weight:700;">WR {_kly_wr}%</span>'
                    f'<span style="font-size:9px;color:#22c55e;">+{_kly_aw:.2f}% avg win</span>'
                    f'<span style="font-size:9px;color:#ef4444;">{_kly_al:.2f}% avg loss</span>'
                    if _kly_nc > 0 else
                    f'<span style="font-size:9px;color:#334155;">No closed trades yet</span>'
                )
                + f'</div>'
                f'<div style="background:#0f172a;border-radius:2px;height:3px;overflow:hidden;margin-bottom:3px;">'
                f'<div style="width:{_boot_pct}%;background:{_boot_col};height:100%;border-radius:2px;"></div>'
                f'</div>'
                f'<div style="font-size:8px;color:#334155;">'
                + (
                    f'Real b ratio active ✓ ({_kly_nw}W / {_kly_nl}L)'
                    if _kly_nw >= 5 and _kly_nl >= 5 else
                    f'Bootstrap: {_kly_nc}/10 trades · need {_b_needed} more per side for real b ratio'
                )
                + f'</div>'
                f'</div>'
            )

            if _kly_viable:
                _kly_col  = "#22c55e" if _kly_half >= 8 else "#f59e0b" if _kly_half >= 4 else "#94a3b8"
                _kly_cap_badge = (
                    '<span style="color:#f59e0b;font-size:9px;"> · 15% cap applied</span>'
                    if _kly_cap else ""
                )
                _kelly_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:13px;color:#94a3b8;font-weight:800;letter-spacing:0.08em;">LONG TERM KELLY</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · WEEKS/MONTHS</span>'
                    f'</div>'
                    f'<span style="font-size:8px;color:#334155;font-style:italic;white-space:nowrap;">your core portfolio allocation</span>'
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
                    f'{_stats_pill_html}'
                    f'{_streak_row_html}'
                    f'<div style="border-top:1px solid #1e293b;margin:6px 0 5px;"></div>'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">SIGNAL ALIGNMENT</div>'
                    f'<div style="display:flex;align-items:flex-start;margin-bottom:4px;">{_sq_html}</div>'
                    f'<div style="margin-bottom:4px;">{_lbl_html}</div>'
                    f'<div style="font-size:7px;color:#334155;line-height:1.8;margin-bottom:6px;">'
                    f'<span style="color:#22c55e;font-weight:700;">✓</span> agrees with Kelly direction&nbsp;&nbsp;'
                    f'<span style="color:#ef4444;font-weight:700;">✗</span> opposes Kelly direction&nbsp;&nbsp;'
                    f'<span style="color:#f59e0b;font-weight:700;">~</span> neutral or Kelly is neutral'
                    f'</div>'
                    f'<div style="border-top:1px solid #1e293b;margin:5px 0 6px;"></div>'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:5px;">SIZING CHAIN</div>'
                    f'{_build_kelly_chain(_kly, hmm_lbl=_hmm_lbl_txt, align_col=_align_mult_col)}'
                    f'{_build_kelly_ref_table(_kly.get("kelly_half_base_pct", _kly_half))}'
                    f'<div style="margin-top:7px;padding-top:6px;border-top:1px solid #1e293b33;'
                    f'display:flex;align-items:center;gap:6px;">'
                    f'<span style="font-size:8px;color:#334155;font-weight:700;letter-spacing:0.08em;">KELLY SIZES →</span>'
                    f'<span style="font-size:8px;color:#475569;letter-spacing:0.06em;">Entry Signal trigger ↓ below</span>'
                    f'</div>'
                    f'</div>'
                )
            else:
                _kelly_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:13px;color:#94a3b8;font-weight:800;letter-spacing:0.08em;">LONG TERM KELLY</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏱ SLOW · WEEKS/MONTHS</span>'
                    f'</div>'
                    f'<span style="font-size:8px;color:#334155;font-style:italic;white-space:nowrap;">your core portfolio allocation</span>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#ef444466;">Negative expectancy — no position suggested</div>'
                    f'<div style="font-size:10px;color:#334155;margin-top:2px;">p={_kly_p*100:.0f}% · b={_kly_b} · {_kly_psrc}</div>'
                    f'<div style="background:#0a0f1a;border-left:3px solid #ef444440;border-radius:3px;'
                    f'padding:7px 10px;margin:8px 0 4px;">'
                    f'<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">WHY 0%</div>'
                    f'<div style="font-size:10px;color:#475569;line-height:1.6;">'
                    f'Kelly formula: <span style="color:#94a3b8;">f = (b×p − q) / b</span><br>'
                    f'= ({_kly_b}×{_kly_p*100:.0f}% − {(1-_kly_p)*100:.0f}%) / {_kly_b} '
                    f'= <span style="color:#ef4444;">{(_kly_b*_kly_p-(1-_kly_p))/_kly_b*100:.1f}%</span> → negative<br>'
                    f'<span style="color:#64748b;">To fix: raise conviction (signals align) or close more trades to build win rate history.</span>'
                    f'</div>'
                    f'</div>'
                    f'{_stats_pill_html}'
                    f'<div style="margin-top:7px;padding-top:6px;border-top:1px solid #1e293b33;'
                    f'display:flex;align-items:center;gap:6px;">'
                    f'<span style="font-size:8px;color:#334155;font-weight:700;letter-spacing:0.08em;">KELLY SIZES →</span>'
                    f'<span style="font-size:8px;color:#475569;letter-spacing:0.06em;">Entry Signal trigger ↓ below</span>'
                    f'</div>'
                    f'</div>'
                )
            # ── Shadow Kelly pill (SPX price-brain sibling) ──────────────────
            _shadow_kelly_html = ""
            try:
                from services.portfolio_sizing import compute_shadow_kelly as _compute_shadow_kelly
                from services.hmm_shadow import load_current_shadow_state as _sh_load_ks
                from services.hmm_regime import get_state_color as _sh_get_color
                _sh_state_for_kelly = _sh_load_ks()
                _sh_label_for_kelly = getattr(_sh_state_for_kelly, "state_label", None)
                _sh_crash_for_kelly = getattr(_sh_state_for_kelly, "crash_prob_10pct", 0.0) or 0.0

                _skly = _compute_shadow_kelly(
                    _conviction_score,
                    st.session_state.get("_fear_composite") or {},
                    st.session_state.get("_regime_context") or {},
                    options_score=(st.session_state.get("_options_flow_context") or {}).get("options_score"),
                    tactical_score=(st.session_state.get("_tactical_context") or {}).get("tactical_score"),
                    shadow_state_label=_sh_label_for_kelly,
                    shadow_crash_prob=_sh_crash_for_kelly,
                )
                _skly_half = _skly["kelly_half_pct"]
                _skly_full = _skly["kelly_full_pct"]
                _skly_rmul = _skly.get("shadow_regime_multiplier", 1.0)
                _skly_cpen = _skly.get("crash_prob_penalty_pct", 0.0)
                _skly_col  = "#22c55e" if _skly_half >= 8 else "#f59e0b" if _skly_half >= 4 else "#94a3b8"
                _skly_delta = round(_skly_half - _kly_half, 1) if _kly_viable else None
                _skly_delta_txt = ""
                if _skly_delta is not None:
                    _d_col = "#22c55e" if _skly_delta > 0 else ("#ef4444" if _skly_delta < 0 else "#64748b")
                    _skly_delta_txt = (
                        f'<span style="font-size:9px;color:{_d_col};font-weight:700;">'
                        f'Δ vs QIR Kelly: {_skly_delta:+.1f}pp</span>'
                    )
                _sh_label_display = _sh_label_for_kelly or "N/A"
                _shadow_kelly_html = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-left:3px solid {_skly_col};'
                    f'border-radius:5px;padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="font-size:12px;color:#94a3b8;font-weight:800;letter-spacing:0.08em;">'
                    f'SHADOW KELLY</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">SPX PRICE BRAIN</span>'
                    f'{_skly_delta_txt}'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:4px;">'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">HALF-KELLY</div>'
                    f'<div style="font-size:20px;font-weight:900;color:{_skly_col};">{_skly_half}%</div>'
                    f'<div style="font-size:9px;color:#475569;">portfolio</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">SHADOW REGIME</div>'
                    f'<div style="font-size:14px;font-weight:900;color:{_sh_get_color(_sh_label_display)};">'
                    f'{_sh_label_display}</div>'
                    f'<div style="font-size:9px;color:#475569;">×{_skly_rmul:.2f} multiplier</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:1px;">CRASH PENALTY</div>'
                    f'<div style="font-size:20px;font-weight:900;color:#94a3b8;">−{_skly_cpen:.0f}%</div>'
                    f'<div style="font-size:9px;color:#475569;">from 30d crash prob</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:8px;color:#334155;line-height:1.5;">'
                    f'Same Bayesian p + win/loss b as QIR Kelly. Differs only in the final regime multiplier '
                    f'(SPX-based 6-regime map) and a crash-probability penalty from the Shadow backtest. '
                    f'Compare side-by-side — disagreement is informative.'
                    f'</div>'
                    f'</div>'
                )
            except Exception:
                _shadow_kelly_html = ""

            _kelly_block       = _kelly_html + _shadow_kelly_html   # structural sizing → MEDIUM
            _fast_setups_html  = _triple_kelly_html  # Buy/Short Setup → FAST
            _bimodal_block     = ""

            # ── Net Kelly card (appended below LT Kelly block) ────────────────────
            _nk_abs = abs(_net_kelly_pct)
            if _net_kelly_pct >= 10:
                _nk_dir, _nk_dot, _nk_col = "LONG",       "🟢", "#22c55e"
            elif _net_kelly_pct >= 3:
                _nk_dir, _nk_dot, _nk_col = "WEAK LONG",  "🟡", "#f59e0b"
            elif _net_kelly_pct > -3:
                _nk_dir, _nk_dot, _nk_col = "FLAT / CASH","⚪", "#94a3b8"
            elif _net_kelly_pct > -10:
                _nk_dir, _nk_dot, _nk_col = "WEAK SHORT", "🟡", "#f59e0b"
            else:
                _nk_dir, _nk_dot, _nk_col = "SHORT",      "🔴", "#ef4444"
            _nk_size_str = f"{_nk_abs:.1f}%" if _nk_abs >= 0.1 else "—"
            _net_kelly_html = (
                f'<div style="background:#0a0f1a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<span style="font-size:11px;color:#64748b;font-weight:800;letter-spacing:0.08em;">NET KELLY</span>'
                f'<span style="font-size:7px;color:#334155;font-weight:700;letter-spacing:0.06em;'
                f'background:#0f172a;padding:1px 5px;border-radius:2px;">60% LT · 40% TAC</span>'
                f'</div>'
                f'<span style="font-size:7px;color:#334155;font-style:italic;">signed composite signal</span>'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:22px;font-weight:900;color:{_nk_col};">{_nk_size_str}</span>'
                f'<div>'
                f'<div style="font-size:12px;font-weight:800;color:{_nk_col};">{_nk_dot} {_nk_dir}</div>'
                f'<div style="font-size:8px;color:#475569;margin-top:1px;">'
                f'LT {_lt_half_signed_pct:+.1f}% · Tac {_tac_net_pct_for_nk:+.1f}%'
                f'</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            _kelly_block += _net_kelly_html

            # ── Inject Kelly badges into Buy/Short Setup cards ────────
            # Only for non-GU patterns (GU uses triple-kelly badges built earlier)
            if _cls.get("pattern") != "GENUINE_UNCERTAINTY":
                _kly_badge_col = "#22c55e" if _kly_viable and _kly_half >= 8 else (
                    "#f59e0b" if _kly_viable and _kly_half >= 4 else (
                    "#94a3b8" if _kly_viable else "#ef444466"
                ))
                if _kly_viable:
                    _kly_badge_txt = f"{_kly_half:.1f}%"
                    _kly_badge_note = f"half-Kelly · {_kly_psrc}"
                else:
                    _kly_badge_txt = "0%"
                    _kly_badge_note = "neg expectancy — reduce size"

                def _make_kelly_badge(label: str, col: str, txt: str, note: str) -> str:
                    return (
                        f'<div style="display:flex;align-items:baseline;gap:6px;'
                        f'background:#0a0f1a;border:1px solid {col}33;border-radius:4px;'
                        f'padding:5px 8px;margin:5px 0 6px;">'
                        f'<span style="font-size:8px;color:{col};font-weight:700;'
                        f'letter-spacing:0.08em;">{label}</span>'
                        f'<span style="font-size:22px;font-weight:900;color:{col};">{txt}</span>'
                        f'<span style="font-size:9px;color:#475569;">of portfolio · weeks/months · {note}</span>'
                        f'</div>'
                    )

                _long_kelly_badge  = _make_kelly_badge("LONG KELLY",  _kly_badge_col, _kly_badge_txt, _kly_badge_note)
                _short_kelly_badge = _make_kelly_badge("SHORT KELLY", _kly_badge_col, _kly_badge_txt, _kly_badge_note)

                # _buy_html / _short_html are already baked into _verdict_html — replace there
                _verdict_html = _verdict_html.replace(
                    '<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">BUY SETUP</div>',
                    '<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">BUY SETUP</div>' + _long_kelly_badge,
                    1,
                )
                _verdict_html = _verdict_html.replace(
                    '<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">SHORT SETUP</div>',
                    '<div style="font-size:9px;color:#64748b;font-weight:700;letter-spacing:0.1em;margin-bottom:4px;">SHORT SETUP</div>' + _short_kelly_badge,
                    1,
                )
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
            f'<span style="font-size:7px;color:#334155;font-weight:600;letter-spacing:0.06em;margin-left:auto;">↑ sized by Kelly above</span>'
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

        # ── Bottom Watch signals (4-signal bottom detection) ───────────────────
        _tb_bw = None
        try:
            from services.market_data import fetch_bottom_watch_signals as _fbws
            _tb_bw = _fbws()
        except Exception:
            pass

        # ── LL-ANCHORED CRISIS DETECTION ─────────────────────────────────────────
        # Crisis Intensity (CI%) normalizes LL z-score to 0-100% scale.
        # Anchor lives on brain.ci_anchor (auto-calibrated at training time = |z|
        # at the worst in-sample day, set to map to 100% CI).
        #   Zone 1: CI < 22%    — Normal
        #   Zone 2: 22-40%      — Model Stress (early warning)
        #   Zone 3: 40-100%     — Crisis Gate Open (75% recall, 0% FP)
        #   Zone 4: > 100%      — Beyond training range
        # Formula: CI = abs(ll_z) / brain.ci_anchor * 100  (uncapped)
        def _build_early_warning_pill() -> str:
            """Pre-gate warning when faster signals detect stress before primary brain."""
            try:
                _ew_of = st.session_state.get("_options_flow_context") or {}
                _ew_of_score = _ew_of.get("options_score", 50) if _ew_of else 50
                _ew_sh = st.session_state.get("_shadow_state_obj")
                _ew_sh_ci = getattr(_ew_sh, "ci_pct", 0.0) if _ew_sh else 0.0
                _ew_sh_label = getattr(_ew_sh, "state_label", "") if _ew_sh else ""
                _ew_primary_ci = max(0.0, (abs(_tb_ll_z) / _ci_anchor() * 100.0) if _tb_ll_z < 0 else 0.0)

                # DIP WARNING: options bearish + shadow stressed + primary calm
                if _ew_of_score < 35 and _ew_sh_ci > 22 and _ew_primary_ci < 22:
                    return (
                        f'<div style="margin-top:6px;padding:5px 10px;background:#451a03;'
                        f'border:1px solid #f59e0b;border-radius:4px;">'
                        f'<div style="font-size:9px;color:#fcd34d;font-weight:800;letter-spacing:0.06em;">'
                        f'⚡ EARLY WARNING — DIP FORMING</div>'
                        f'<div style="font-size:8px;color:#f59e0b;margin-top:2px;line-height:1.5;">'
                        f'Options bearish ({_ew_of_score}/100) + Shadow stressed (CI {_ew_sh_ci:.0f}%) '
                        f'· Primary brain has not confirmed yet (CI {_ew_primary_ci:.0f}%)</div>'
                        f'</div>'
                    )

                # BOTTOM WARNING: options bullish + shadow transitioning + primary still stressed
                if (_ew_of_score >= 65 and _ew_primary_ci >= 22
                        and _ew_sh_label in ("Transition", "Mild Bull", "Strong Bull")):
                    return (
                        f'<div style="margin-top:6px;padding:5px 10px;background:#052e16;'
                        f'border:1px solid #22c55e;border-radius:4px;">'
                        f'<div style="font-size:9px;color:#4ade80;font-weight:800;letter-spacing:0.06em;">'
                        f'🔄 EARLY WARNING — BOTTOM FORMING</div>'
                        f'<div style="font-size:8px;color:#22c55e;margin-top:2px;line-height:1.5;">'
                        f'Options bullish ({_ew_of_score}/100) + Shadow transitioning ({_ew_sh_label}) '
                        f'· Primary brain still stressed (CI {_ew_primary_ci:.0f}%)</div>'
                        f'</div>'
                    )
            except Exception:
                pass
            return ""

        def _build_ll_anchored_block() -> str:
            # Crisis Intensity score — uncapped, COVID in-sample peak = 100%
            _ci_raw = (abs(_tb_ll_z) / _ci_anchor() * 100.0) if _tb_ll_z < 0 else 0.0
            _ci = max(0.0, _ci_raw)

            # Zone thresholds in CI% — classifier lives in utils.ci_zone
            from utils.ci_zone import classify_ci_zone as _classify_ci_zone
            _zone = _classify_ci_zone(_ci).zone

            # ── Conviction signals ────────────────────────────────────────────────
            _vix_val  = st.session_state.get("_market_snapshot", {}).get("VIX", {}).get("price", 20)
            _wk_phase = (_tb_wyckoff or {}).get("phase", "")
            _wk_conf  = (_tb_wyckoff or {}).get("confidence", 0)

            _signals = [
                ("Regime elevated",       f"+{_tb_regime:.3f}",  _tb_regime > 0.05),
                ("High entropy",          f"{_tb_entropy:.3f}",  _tb_entropy > 0.68),
                ("Low conviction",        f"{_tb_conv:.0f}",     _tb_conv < 22),
                ("VIX spike",             f"{_vix_val:.1f}",     _vix_val > 25),
                ("Wyckoff Distribution",  f"{_wk_conf:.0f}%",    _wk_phase == "Distribution"),
            ]
            _n_firing = sum(1 for _, _, fired in _signals if fired)

            # ── Zone styling ──────────────────────────────────────────────────────
            if _zone == 4:
                _bg, _border     = "#0d001a", "#7c3aed"
                _ci_color        = "#a855f7"
                _label           = "BEYOND TRAINING RANGE"
                _label_sub       = f"Model scoring post-training data — {_ci:.0f}% CI · exceeds COVID baseline"
            elif _zone == 3:
                _bg, _border     = "#100000", "#7f1d1d"
                _ci_color        = "#ef4444"
                _label           = "CRISIS CONFIRMED"
                _label_sub       = "Crisis Gate Open · 75% historical detection, 0% false alarms"
            elif _zone == 2:
                _bg, _border     = "#0f0e00", "#78350f"
                _ci_color        = "#f59e0b"
                _label           = "MODEL STRESS DETECTED"
                _label_sub       = f"Below crisis gate (40%) · watch for continuation"
            else:
                _bg, _border     = "#0f172a", "#1e293b"
                _ci_color        = "#22c55e"
                _label           = "NORMAL MARKET CONDITIONS"
                _label_sub       = "HMM model fits current data — no crisis signature"

            # ── Signal rows ───────────────────────────────────────────────────────
            def _sig_row(name, val, fired):
                if _zone == 1:
                    return (
                        f'<div style="display:flex;justify-content:space-between;padding:1px 0;">'
                        f'<span style="font-size:8px;color:#1e3a5f;">○ {name}</span>'
                        f'<span style="font-size:8px;color:#1e3a5f;">{val}</span>'
                        f'</div>'
                    )
                dot  = "●" if fired else "○"
                dcol = _ci_color if fired else "#334155"
                tcol = ("#94a3b8" if _zone == 2 else _ci_color) if fired else "#475569"
                vcol = _ci_color if fired else "#334155"
                badge_text = "STRESS" if _zone == 2 else ("EXTREME" if _zone == 4 else "CONFIRMED")
                badge_bg   = "#1a1000" if _zone == 2 else ("#1a0028" if _zone == 4 else "#1a0000")
                badge = (
                    f' <span style="font-size:6px;color:{_ci_color};background:{badge_bg};'
                    f'padding:0 3px;border-radius:2px;">{badge_text}</span>'
                ) if fired else ""
                return (
                    f'<div style="display:flex;justify-content:space-between;padding:1px 0;">'
                    f'<span style="font-size:8px;color:{tcol};">'
                    f'<span style="color:{dcol};">{dot}</span> {name}{badge}</span>'
                    f'<span style="font-size:8px;color:{vcol};font-weight:{"700" if fired else "400"};">{val}</span>'
                    f'</div>'
                )

            _sigs_html = "".join(_sig_row(n, v, f) for n, v, f in _signals)

            # ── Reference points on the CI scale ─────────────────────────────────
            _ref_events = [
                # CI% values rescaled to current brain.ci_anchor (~6.461 post-2026-04 retrain)
                ("Tariff 2025", 50, "#f59e0b"),    # 2025-04 z=-3.21 → 50% CI (above gate)
                ("Volmageddon", 34, "#f59e0b"),    # 2018-02 z=-2.22 → 34% CI
                ("Fed Panic",   41, "#ef4444"),    # 2018-12 z=-2.66 → 41% CI (just over gate)
                ("COVID",      100, "#ef4444"),    # |z| = brain.ci_anchor → 100% CI
            ]

            # ── Footer text ───────────────────────────────────────────────────────
            if _zone == 4:
                _explain = (
                    f"CI {_ci:.0f}% (LL z={_tb_ll_z:.3f}). Model is scoring data BEYOND its training range. "
                    f"100% = worst-ever in-sample (z=-{_ci_anchor():.3f}). Current reading exceeds that baseline by "
                    f"{_ci - 100:.0f}%. This occurs when post-training market data is structurally novel to the model — "
                    f"a stronger crisis signal than any event in the backtest history."
                )
            elif _zone == 3:
                _explain = (
                    f"CI {_ci:.0f}% (LL z={_tb_ll_z:.3f}) has breached the 40% confirmation gate. "
                    f"Identical signatures: Volmageddon 76%, Fed Panic 96%, COVID 100%. "
                    f"Backtested 3,408 days: zero false alarms. Rate shocks (2022) and tariff events "
                    f"stay below 22% — the HMM model recognizes them as known regimes."
                )
            elif _zone == 2:
                _explain = (
                    f"CI {_ci:.0f}% (LL z={_tb_ll_z:.3f}). Stress detected but below 40% gate. "
                    f"Historical misses (2022 bear, tariffs) peaked at 12-21% CI — well below here. "
                    f"{_n_firing}/5 conviction signals shown as context. "
                    f"40% is the crisis gate threshold (9.25% crash prob = 3x baseline)."
                )
            else:
                _explain = (
                    f"CI {_ci:.0f}% (LL z={_tb_ll_z:.3f}). Normal market — HMM fits well. "
                    f"Conviction signals suppressed: they fire every day alone (0% precision). "
                    f"Stress watch above 22% CI · Crisis confirmed above 40% CI · COVID was 100%."
                )

            # Bar fill capped at 100% visually — but CI number shows true value
            _bar_fill = min(100.0, _ci)
            # Zone 4 extra annotation above the bar
            _beyond_badge = (
                f'<div style="font-size:7px;color:#a855f7;font-weight:700;margin-bottom:2px;">'
                f'⚠ {_ci:.0f}% — EXCEEDS COVID BASELINE BY {_ci-100:.0f}%</div>'
            ) if _zone == 4 else ""

            return (
                f'<div style="background:{_bg};border:1px solid {_border};border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'

                # Header
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
                f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">'
                f'LL-ANCHORED CRISIS DETECTION</span>'
                f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</span>'
                f'</div>'

                # Big CI% + z-score + label
                f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">'
                f'<span style="font-size:26px;color:{_ci_color};font-weight:900;line-height:1;">'
                f'{_ci:.0f}%</span>'
                f'<span style="font-size:13px;color:{_ci_color};font-weight:700;opacity:0.7;">'
                f'z={_tb_ll_z:.3f}</span>'
                f'<div>'
                f'<div style="font-size:10px;color:{_ci_color};font-weight:800;">{_label}</div>'
                f'<div style="font-size:7px;color:#64748b;">{_label_sub}</div>'
                f'</div>'
                f'</div>'

                # CI progress bar with zone markers
                f'<div style="margin-bottom:8px;">'
                f'{_beyond_badge}'
                f'<div style="position:relative;height:8px;background:#0a0f1a;border-radius:4px;overflow:hidden;">'
                # Fill capped at 100% visually
                f'<div style="position:absolute;left:0;top:0;height:100%;width:{_bar_fill:.0f}%;'
                f'background:{_ci_color};border-radius:4px;"></div>'
                # Zone boundary at 22% (stress start)
                f'<div style="position:absolute;left:22%;top:0;width:1px;height:100%;background:#334155;"></div>'
                # Gate marker at 40% (recalibrated 2026-04 — was 67% under legacy ICE BofA brain)
                f'<div style="position:absolute;left:40%;top:0;width:2px;height:100%;background:#ef444488;"></div>'
                f'</div>'
                # Scale labels
                f'<div style="display:flex;justify-content:space-between;font-size:6px;'
                f'color:#334155;margin-top:2px;">'
                f'<span>0% Normal</span>'
                f'<span style="margin-left:10%;">22% Stress</span>'
                f'<span style="color:#ef444488;">40% ▲ GATE</span>'
                f'<span>100% COVID</span>'
                f'</div>'
                # Reference events
                f'<div style="display:flex;gap:6px;margin-top:3px;flex-wrap:wrap;">'
                + "".join(
                    f'<span style="font-size:6px;color:{c};">◆ {name} {pct}%</span>'
                    for name, pct, c in _ref_events
                )
                + f'</div>'
                f'</div>'

                # Conviction signals
                f'<div style="border-top:1px solid {_border}55;padding-top:5px;margin-bottom:4px;">'
                f'<div style="font-size:7px;color:#475569;font-weight:700;letter-spacing:0.08em;margin-bottom:3px;">'
                f'CONVICTION SIGNALS — {_n_firing}/5 FIRING'
                + (
                    " · suppressed in normal market" if _zone == 1 else
                    " · stress context (unvalidated individually)" if _zone == 2 else
                    " · supporting context"
                )
                + f'</div>'
                f'{_sigs_html}'
                f'</div>'

                # Early warning pre-gate (options + shadow stress before primary catches up)
                + _build_early_warning_pill()
                +
                # Footer — explanation text (larger, more visible)
                f'<div style="font-size:10px;color:#cbd5e1;margin-top:6px;line-height:1.6;'
                f'padding:6px 8px;background:#0a0f1a22;border-radius:3px;border-left:2px solid {_border};">'
                f'{_explain}</div>'
                f'</div>'
            )
        
        # Generate LL-anchored block (replaces broken proximity method)
        _ll_anchored_block = _build_ll_anchored_block()

        # Legacy proximity data for backwards compatibility (deprecated) 
        _top_signals = []
        _bottom_signals = []

        # ── OLD proximity thresholds (DEPRECATED - use LL-anchored instead) ──
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

        # Count firing signals (≥50 threshold)
        _top_count = sum(1 for _, val in _top_signals if val >= 50) if _top_signals else 0
        _bot_count = sum(1 for _, val in _bottom_signals if val >= 50) if _bottom_signals else 0
        _top_total = len(_top_signals)
        _bot_total = len(_bottom_signals)

        if _top_signals or _bottom_signals:
            _tb_rows = ""
            if _top_signals:
                _top_color = "#ef4444" if _top_count >= _top_total//2 else ("#f59e0b" if _top_count > 0 else "#64748b")
                _tb_rows += (
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-end;'
                    f'padding:4px 0;border-bottom:1px solid #1e293b33;">'
                    f'<span style="color:#ef4444;font-size:10px;font-weight:700;">MARKET TOP</span>'
                    f'<div style="text-align:right;">'
                    f'<span style="color:{_top_color};font-size:12px;font-weight:800;">{_top_count}/{_top_total} signals</span><br>'
                    f'<span style="color:#64748b;font-size:8px;">avg strength {_top_score}%</span>'
                    f'</div>'
                    f'</div>'
                )
                for _ts_name, _ts_val in _top_signals:
                    _tb_rows += (
                        f'<div style="font-size:8px;color:#64748b;padding:1px 0 1px 8px;">'
                        f'{"●" if _ts_val >= 50 else "○"} {_ts_name}</div>'
                    )
            if _bottom_signals:
                _bot_color = "#22c55e" if _bot_count >= _bot_total//2 else ("#f59e0b" if _bot_count > 0 else "#64748b")
                _tb_rows += (
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-end;'
                    f'padding:4px 0;border-bottom:1px solid #1e293b33;margin-top:4px;">'
                    f'<span style="color:#22c55e;font-size:10px;font-weight:700;">MARKET BOTTOM</span>'
                    f'<div style="text-align:right;">'
                    f'<span style="color:{_bot_color};font-size:12px;font-weight:800;">{_bot_count}/{_bot_total} signals</span><br>'
                    f'<span style="color:#64748b;font-size:8px;">avg strength {_bot_score}%</span>'
                    f'</div>'
                    f'</div>'
                )
                for _bs_name, _bs_val in _bottom_signals:
                    _tb_rows += (
                        f'<div style="font-size:8px;color:#64748b;padding:1px 0 1px 8px;">'
                        f'{"●" if _bs_val >= 50 else "○"} {_bs_name}</div>'
                    )

            # Net lean verdict
            _net_diff = _top_count - _bot_count
            if _net_diff > 0:
                _lean_text = f"NET LEAN: +{_net_diff} TOP"
                _lean_color = "#ef4444"
            elif _net_diff < 0:
                _lean_text = f"NET LEAN: +{abs(_net_diff)} BOT"
                _lean_color = "#22c55e"
            else:
                _lean_text = "NET LEAN: BALANCED"
                _lean_color = "#64748b"

            # Store legacy proximity data but don't use for display 
            _legacy_top_bottom_data = (
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;">TOP / BOTTOM PROXIMITY</span>'
                f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⏑ MEDIUM · DAYS/WEEKS</span>'
                f'</div>'
                f'{_tb_rows}'
                f'<div style="padding:6px 0;border-top:1px solid #1e293b33;margin-top:6px;text-align:center;">'
                f'<span style="font-size:9px;color:{_lean_color};font-weight:700;">{_lean_text}</span>'
                f'</div>'
                f'<div style="font-size:7px;color:#475569;margin-top:4px;">'
                f'Count method 88% accurate vs 56% for score method · 8 crash calibration · '
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
                # Store LL-anchored crisis data for backtesting
                st.session_state["_ll_anchored_crisis"] = {
                    "ll_zscore": _tb_ll_z,
                    "ll_strength": _ll_strength,
                    "gate_open": _ll_strength >= 40,
                    "confirmations": _confirmations if _ll_strength >= 40 else [],
                    "crisis_level": _crisis_level if _ll_strength >= 40 else "NORMAL",
                    "status_text": _status_text if _ll_strength >= 40 else "Normal market conditions",
                    "hmm_state": _tb_hmm_label,
                }
                
                # Store legacy data for backward compatibility
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
                _hs_stale_badge = (
                    f'<span style="background:#1e293b;color:#64748b;font-size:7px;'
                    f'font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;">'
                    f'as of {getattr(_hmm_s, "_stale_date", "?")} · run QIR to refresh</span>'
                    if getattr(_hmm_s, "_is_stale", False) else ""
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

                if _ll_z < -0.30:
                    _ll_status_text = f"CRISIS GATE OPEN (z={_ll_z:.2f})"
                    _ll_status_color = "#ef4444"
                elif _ll_z < -0.20:
                    _ll_status_text = f"Model stress elevated (z={_ll_z:.2f})"
                    _ll_status_color = "#f59e0b"
                elif _ll_z < -0.10:
                    _ll_status_text = f"Minor model tension (z={_ll_z:.2f})"
                    _ll_status_color = "#eab308"
                else:
                    _ll_status_text = f"Model fitting normally (z={_ll_z:.2f})"
                    _ll_status_color = "#22c55e"
                
                _ll_col = "#22c55e" if _ll_z > -0.10 else ("#f59e0b" if _ll_z > -0.30 else "#ef4444")
                _ll_label = "Normal" if _ll_z > -0.10 else ("Caution" if _ll_z > -0.30 else "GATE OPEN")

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
                    f'market data fits the trained model. When LL z-score drops below -0.30, '
                    f'it triggers the LL-anchored crisis detection gate (50% crash detection, 0% false alarms). '
                    f'Current status: {_ll_status_text.strip()}. '
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
                    f'{_hs_retrain_badge}{_hs_stale_badge}'
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

                _state_legend_html = (
                    f'<div style="margin-top:8px;padding:7px 8px;background:#0a0f1a;'
                    f'border-radius:4px;border:1px solid #0f1f2e;">'
                    f'<div style="font-size:7px;color:#1e3a5f;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:5px;">STATE GUIDE</div>'
                    f'<div style="font-size:8px;color:#1e3a5f;line-height:1.9;">'
                    f'<span style="color:#22c55e;font-weight:700;">Bull</span>'
                    f'<span style="color:#334155;"> — tight credit (HY &lt;350bp), low VIX, risk-on. Full conviction sizing ×1.10</span><br>'
                    f'<span style="color:#94a3b8;font-weight:700;">Neutral</span>'
                    f'<span style="color:#334155;"> — balanced, no extremes. Wait for confirmation. Baseline sizing ×1.00</span><br>'
                    f'<span style="color:#f59e0b;font-weight:700;">Early Stress</span>'
                    f'<span style="color:#334155;"> — spreads starting to widen, VIX creeping. Trim speculative. ×0.90</span><br>'
                    f'<span style="color:#f97316;font-weight:700;">Stress</span>'
                    f'<span style="color:#334155;"> — HY 400–600bp, sellers in control. Reduce equity, size down ×0.85</span><br>'
                    f'<span style="color:#ef4444;font-weight:700;">Late Cycle</span>'
                    f'<span style="color:#334155;"> — spreads elevated for weeks/months, economy weakening. Quality only ×0.75</span><br>'
                    f'<span style="color:#dc2626;font-weight:700;">Crisis</span>'
                    f'<span style="color:#334155;"> — HY &gt;600bp, credit dislocation. Capital preservation mode ×0.60</span>'
                    f'</div></div>'
                )

                _hmm_html = _hmm_html + _cal_html + _state_legend_html + f'</div>'
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


        # ── Shadow HMM Brain (SPX price brain) ────────────────────────────────
        _shadow_block = ""
        try:
            from services.hmm_shadow import (
                load_current_shadow_state, load_shadow_brain,
            )
            from services.hmm_regime import (
                get_state_color, get_state_arrow, get_state_tips,
            )
            _sh_s = load_current_shadow_state()
            _sh_b = load_shadow_brain()
            if _sh_s is not None:
                st.session_state["_shadow_state_obj"] = _sh_s
            if _sh_s is not None and _sh_b is not None:
                _sh_col = get_state_color(_sh_s.state_label)
                _sh_arr = get_state_arrow(_sh_s.state_label)
                _sh_conf = int(_sh_s.confidence * 100)
                _sh_pers = _sh_s.persistence
                _sh_z = getattr(_sh_s, "ll_zscore", 0.0) or 0.0
                _sh_ci = getattr(_sh_s, "ci_pct", 0.0) or 0.0
                _sh_crash = getattr(_sh_s, "crash_prob_10pct", 0.0) or 0.0
                _sh_exp_dd = getattr(_sh_s, "expected_drawdown_pct", 0.0) or 0.0
                _sh_ret = getattr(_sh_s, "daily_return_pct", 0.0) or 0.0

                _sh_trained = _sh_b.trained_at[:10] if _sh_b.trained_at else "—"
                _sh_retrain_due = False
                try:
                    from datetime import datetime as _sh_dt_mod, timezone as _sh_tz
                    _sh_dt = _sh_dt_mod.fromisoformat(_sh_b.trained_at.replace("Z", "+00:00"))
                    _sh_days_since = (_sh_dt_mod.now(_sh_tz.utc) - _sh_dt).days
                    _sh_retrain_due = _sh_days_since >= 90
                except Exception:
                    pass
                _sh_retrain_badge = (
                    f'<span style="background:#92400e;color:#fcd34d;font-size:8px;'
                    f'font-weight:700;padding:1px 6px;border-radius:3px;'
                    f'letter-spacing:0.05em;margin-left:6px;">RETRAIN DUE</span>'
                    if _sh_retrain_due else ""
                )
                _sh_stale_badge = (
                    f'<span style="background:#1e293b;color:#64748b;font-size:7px;'
                    f'font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;">'
                    f'as of {getattr(_sh_s, "_stale_date", "?")} · run QIR to refresh</span>'
                    if getattr(_sh_s, "_is_stale", False) else ""
                )

                _sh_ll_col = "#22c55e" if _sh_z > -0.26 else ("#f59e0b" if _sh_z > -0.80 else ("#ef4444" if _sh_z > -1.194 else "#a855f7"))
                _sh_ll_label = "Normal" if _sh_z > -0.26 else ("Stress" if _sh_z > -0.80 else ("CRISIS GATE" if _sh_z > -1.194 else "BEYOND"))
                _sh_ll_bar_pct = min(100.0, max(0.0, ((-_sh_z) / 1.194) * 100.0)) if _sh_z else 0.0
                _sh_crash_col = "#22c55e" if _sh_crash < 0.20 else ("#f59e0b" if _sh_crash < 0.55 else "#ef4444")
                _sh_ci_col = "#22c55e" if _sh_ci < 22 else ("#f59e0b" if _sh_ci < 67 else "#ef4444")

                _sh_trans_row = _sh_b.transmat[_sh_s.state_idx]
                def _sh_fmt_prob(v):
                    pct = v * 100
                    if pct >= 1:
                        return f"{pct:.0f}%"
                    elif pct > 0:
                        return "<1%"
                    return "0%"
                _sh_trans_cells = "".join(
                    f'<span style="font-size:9px;color:#64748b;">'
                    f'{_sh_b.state_labels[j][:4]} '
                    f'<span style="color:#94a3b8;">{_sh_fmt_prob(_sh_trans_row[j])}</span>'
                    f'</span>  '
                    for j in range(_sh_b.n_states)
                )

                _shadow_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-left:3px solid {_sh_col};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<span style="font-size:13px;color:#475569;font-weight:700;letter-spacing:0.1em;">'
                    f'SHADOW BRAIN <span style="color:#64748b;font-size:10px;">(SPX price · 1960→now)</span>'
                    f'</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">⚡ PRICE-RETURN REGIME</span>'
                    f'{_sh_retrain_badge}{_sh_stale_badge}'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:6px;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">REGIME</div>'
                    f'<div style="font-size:20px;font-weight:900;color:{_sh_col};'
                    f'font-family:\'JetBrains Mono\',Consolas,monospace;">{_sh_arr}</div>'
                    f'<div style="font-size:11px;font-weight:700;color:{_sh_col};">{_sh_s.state_label}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">CONFIDENCE</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#94a3b8;">{_sh_conf}%</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">PERSISTENCE</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#94a3b8;">{_sh_pers}d</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:2px;">'
                    f'→ {_sh_trans_cells}'
                    f'</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:3px;">'
                    f'{_sh_b.n_states}-regime GaussianHMM (SPX+VIX) · trained {_sh_trained} · '
                    f'window {_sh_b.training_start}–{_sh_b.training_end}</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(2, 1fr);gap:10px;'
                    f'margin-top:8px;padding-top:8px;border-top:1px solid #1e293b;">'
                    # Left column: LL + bar (full width)
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:3px;">LOG-LIKELIHOOD</div>'
                    f'<div style="display:flex;align-items:baseline;gap:8px;">'
                    f'<div style="font-size:16px;font-weight:900;color:{_sh_ll_col};">{_sh_z:+.2f}z</div>'
                    f'<div style="font-size:9px;color:{_sh_ll_col};font-weight:600;">{_sh_ll_label}</div>'
                    f'</div>'
                    f'<div style="position:relative;height:6px;background:#1e293b;border-radius:2px;'
                    f'margin-top:4px;overflow:hidden;border:1px solid #334155;width:100%;">'
                    f'<div style="position:absolute;left:0;top:0;height:100%;width:{_sh_ll_bar_pct}%;'
                    f'background:{_sh_ll_col};border-radius:2px;"></div>'
                    # Gate markers at -0.26 (21.8%) and -0.80 (67.2%)
                    f'<div style="position:absolute;left:21.8%;top:0;width:1px;height:100%;background:#334155;opacity:0.5;"></div>'
                    f'<div style="position:absolute;left:67.2%;top:0;width:1px;height:100%;background:#334155;opacity:0.5;"></div>'
                    f'</div>'
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;">← -1.194 CRISIS · -0.80 STRESS · -0.26 NORMAL →</div>'
                    f'</div>'
                    # Right column: CI%, Crash, Exp DD (in 3 rows)
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:3px;">CI% · CRASH · DRAWDOWN</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:6px;">'
                    f'<div>'
                    f'<div style="font-size:14px;font-weight:900;color:{_sh_ci_col};">{_sh_ci:.0f}%</div>'
                    f'<div style="font-size:8px;color:#64748b;">CI%</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:14px;font-weight:900;color:{_sh_crash_col};">{_sh_crash*100:.0f}%</div>'
                    f'<div style="font-size:8px;color:#64748b;">crash</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:14px;font-weight:900;color:#94a3b8;">{_sh_exp_dd:+.1f}%</div>'
                    f'<div style="font-size:8px;color:#64748b;">exp dd</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #1e293b;'
                    f'font-size:10px;color:#64748b;line-height:1.6;">'
                    f'<span style="color:{_sh_col};font-weight:700;">Today: </span>'
                    f'SPX {_sh_ret:+.2f}% · {get_state_tips(_sh_s.state_label)}</div>'
                    f'<div style="margin-top:8px;padding:6px 10px;background:#0a0f1a;'
                    f'border-radius:4px;border:1px solid #1e293b;">'
                    f'<div style="font-size:8px;color:#1e293b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:4px;">WHY A SECOND BRAIN</div>'
                    f'<div style="font-size:8px;color:#1e3a5f;line-height:1.7;">'
                    f'The credit brain above sees spreads, yields, VIX — not price. '
                    f'The shadow brain sees only SPX log returns. '
                    f'<span style="color:#1e4a7a;">Agreement</span> = high-conviction regime call. '
                    f'<span style="color:#1e4a7a;">Disagreement</span> is itself the signal '
                    f'(credit calm + price panic = liquidity flush; credit stress + price calm = slow-burn late cycle).'
                    f'</div></div>'
                    f'</div>'
                )
            elif _sh_b is None:
                _shadow_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:5px;padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">SHADOW BRAIN</div>'
                    f'<div style="font-size:11px;color:#ef444466;">'
                    f'Not trained yet — run <span style="font-family:monospace;">python tools/train_hmm_shadow.py</span> '
                    f'then <span style="font-family:monospace;">python tools/backtest_shadow_ci.py</span></div>'
                    f'</div>'
                )
            else:
                _shadow_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:5px;padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">SHADOW BRAIN</div>'
                    f'<div style="font-size:11px;color:#f59e0b;">'
                    f'Brain loaded but no state scored yet — run '
                    f'<span style="font-family:monospace;">python -c "from services.hmm_shadow import score_current_shadow_state; score_current_shadow_state()"</span>'
                    f'</div></div>'
                )
        except Exception as _sh_exc:
            import traceback as _sh_tb
            _shadow_block = (
                f'<div style="background:#1a0000;border:1px solid #ef4444;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="font-size:11px;color:#ef4444;font-weight:700;">SHADOW BRAIN ERROR</div>'
                f'<pre style="font-size:9px;color:#f87171;white-space:pre-wrap;">'
                f'{"".join(_sh_tb.format_exception(type(_sh_exc), _sh_exc, _sh_exc.__traceback__))}'
                f'</pre></div>'
            )


        # ── Top Brain (macro drift / top detection) ───────────────────────────
        _top_block = ""
        try:
            from services.hmm_top import (
                get_top_signal_cached as _get_top_signal_cached,
                _LATE_LABELS as _top_late_labels,
                _LL_ROLL_THRESH as _top_roll_thresh,
                _LL_ROLL_WINDOW as _top_roll_window,
                _BT_HITS as _top_bt_hits,
                _BT_PEAKS as _top_bt_peaks,
                _BT_HIT_PCT as _top_bt_pct,
                _BT_FA as _top_bt_fa,
                _BT_AVG_LEAD as _top_bt_lead,
            )
            _top_sig = _get_top_signal_cached()
            if _top_sig is not None:
                _top_firing   = _top_sig["sig_and"]
                _top_roll     = _top_sig["ll_z_roll"]
                _top_llz      = _top_sig["ll_z"]
                _top_regime   = _top_sig["regime_label"]
                _top_days_on  = _top_sig["days_in_stress"]
                _top_fill     = min(_top_sig["roll_fill_pct"], 100.0)
                _top_anchor   = _top_sig["ci_anchor"]

                _top_gate_col  = "#ef4444" if _top_firing else "#22c55e"
                _top_reg_col   = "#ef4444" if _top_regime in _top_late_labels else "#22c55e"
                _top_meter_col = "#ef4444" if _top_fill >= 100 else "#f59e0b" if _top_fill >= 50 else "#22c55e"

                _top_days_html = (
                    f'<span style="font-size:8px;color:#f59e0b;margin-left:6px;">'
                    f'{_top_days_on}d active</span>'
                    if _top_firing and _top_days_on > 0 else ""
                )

                _top_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-left:3px solid {_top_gate_col};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<span style="font-size:13px;color:#475569;font-weight:700;letter-spacing:0.1em;">'
                    f'TOP BRAIN '
                    f'<span style="color:#64748b;font-size:10px;">(VIX·NFCI·BAA10Y·T10Y3M · macro drift)</span>'
                    f'</span>'
                    f'<span style="font-size:7px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'background:#0a0f1a;padding:1px 5px;border-radius:2px;">&#9650; TOP DETECTOR</span>'
                    f'<span style="background:{_top_gate_col}22;border:1px solid {_top_gate_col}55;'
                    f'border-radius:3px;padding:1px 7px;font-size:9px;font-weight:800;'
                    f'color:{_top_gate_col};letter-spacing:0.06em;">'
                    f'{"FIRING" if _top_firing else "QUIET"}</span>'
                    f'{_top_days_html}'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:6px;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">40-DAY LL ROLL</div>'
                    f'<div style="font-size:20px;font-weight:900;color:{_top_meter_col};">'
                    f'{_top_roll:.3f}</div>'
                    f'<div style="background:#1e293b;border-radius:2px;height:5px;'
                    f'position:relative;overflow:hidden;margin-top:3px;">'
                    f'<div style="height:5px;border-radius:2px;width:{_top_fill:.1f}%;'
                    f'background:{_top_meter_col};"></div>'
                    f'<div style="position:absolute;right:0;top:0;width:2px;height:5px;'
                    f'background:#ef444488;"></div>'
                    f'</div>'
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;">'
                    f'thresh {_top_roll_thresh} · today ll_z {_top_llz:+.3f}</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">REGIME</div>'
                    f'<div style="font-size:16px;font-weight:900;color:{_top_reg_col};">'
                    f'{_top_regime}</div>'
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;">'
                    f'Gate: Late Cycle / Stress / Crisis</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-size:8px;color:#64748b;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:2px;">ACCURACY</div>'
                    f'<div style="font-size:16px;font-weight:900;color:#94a3b8;">'
                    f'{_top_bt_hits}/{_top_bt_peaks}</div>'
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;">'
                    f'{_top_bt_pct}% hit · {_top_bt_fa} FA · {_top_bt_lead}d lead</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:8px;color:#334155;border-top:1px solid #1e293b;'
                    f'padding-top:5px;line-height:1.7;">'
                    f'Gate: regime ∈ {{Late Cycle, Stress}} AND '
                    f'{_top_roll_window}-day ll_z roll &lt; {_top_roll_thresh}'
                    f' &nbsp;·&nbsp; ci_anchor={_top_anchor:.3f}'
                    f'</div>'
                    f'<div style="margin-top:6px;padding:5px 8px;background:#0a0f1a;'
                    f'border-radius:4px;border:1px solid #1e293b;">'
                    f'<div style="font-size:7px;color:#2d3748;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:3px;">HOW TO USE</div>'
                    f'<div style="font-size:7px;color:#2d3748;line-height:1.8;">'
                    f'<span style="color:#3d5a80;">&#9650; FIRING</span> → trim longs, tighten stops, raise cash'
                    f' · 107d avg lead gives time to act&nbsp;&nbsp;'
                    f'<span style="color:#1e4060;">&#9632; CI% Z3 ≥40%</span> → crash underway, full defense&nbsp;&nbsp;'
                    f'<span style="font-style:italic;">Top Brain fires early — not the crisis gate.</span>'
                    f'</div></div>'
                    f'</div>'
                )
            else:
                _top_block = (
                    f'<div style="background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:5px;padding:8px 12px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">TOP BRAIN</div>'
                    f'<div style="font-size:11px;color:#ef444466;">'
                    f'Not trained — run <span style="font-family:monospace;">'
                    f'python tools/train_top_brain.py</span></div></div>'
                )
        except Exception as _top_exc:
            import traceback as _top_tb
            _top_block = (
                f'<div style="background:#1a0000;border:1px solid #ef4444;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<div style="font-size:11px;color:#ef4444;font-weight:700;">TOP BRAIN ERROR</div>'
                f'<pre style="font-size:9px;color:#f87171;white-space:pre-wrap;">'
                f'{"".join(_top_tb.format_exception(type(_top_exc), _top_exc, _top_exc.__traceback__))}'
                f'</pre></div>'
            )

        # ── Main + Shadow Combo Gate ───────────────────────────────────────────
        try:
            _combo_main_ci  = max(0.0, (abs(_tb_ll_z) / _ci_anchor() * 100.0) if _tb_ll_z < 0 else 0.0)
            _sh_obj_combo   = st.session_state.get("_shadow_state_obj")
            _combo_shad_ci  = float(getattr(_sh_obj_combo, "ci_pct", 0.0) or 0.0) if _sh_obj_combo else None

            _combo_m_z2 = _combo_main_ci >= 22
            _combo_m_z3 = _combo_main_ci >= 40
            _combo_s_z2 = (_combo_shad_ci is not None and _combo_shad_ci >= 22)
            _combo_s_z3 = (_combo_shad_ci is not None and _combo_shad_ci >= 40)

            # Evaluate all six strategies
            _combo_strategies = [
                ("OR — either Zone 3 fires",            _combo_m_z3 or  _combo_s_z3,  "7/8 (88%)", "5/1098 (0.5%)"),
                ("AND — both Zone 3 fire",              _combo_m_z3 and _combo_s_z3,  "5/8 (62%)", "0/1098 (0.0%)"),
                ("OR — either Zone 2 fires",            _combo_m_z2 or  _combo_s_z2,  "8/8 (100%)","59/1098 (5.4%)"),
                ("AND — both Zone 2 fire",              _combo_m_z2 and _combo_s_z2,  "7/8 (88%)", "0/1098 (0.0%) ★"),
                ("Main Z3 OR (Main Z2 AND Shadow Z2)",  _combo_m_z3 or (_combo_m_z2 and _combo_s_z2), "7/8 (88%)", "0/1098 (0.0%) ★"),
                ("Main Z3 OR (Shadow Z3 AND Main Z2)",  _combo_m_z3 or (_combo_s_z3 and _combo_m_z2), "7/8 (88%)", "0/1098 (0.0%) ★"),
            ]

            # How many strategies are firing?
            _n_firing = sum(1 for _, active, _, _ in _combo_strategies if active)
            _best_firing = [s for s, active, _, _ in _combo_strategies
                            if active and "0/1098 (0.0%) ★" in _]

            # Gate status color
            if _n_firing >= 4:
                _gate_color = "#ef4444"; _gate_label = "MULTIPLE GATES OPEN"
            elif _n_firing >= 2:
                _gate_color = "#f59e0b"; _gate_label = "STRESS CONFIRMED"
            elif _n_firing == 1:
                _gate_color = "#f59e0b"; _gate_label = "EARLY WARNING"
            else:
                _gate_color = "#22c55e"; _gate_label = "ALL QUIET"

            _shad_ci_str = f"{_combo_shad_ci:.1f}%" if _combo_shad_ci is not None else "—"
            _shad_missing = _combo_shad_ci is None

            # Build strategy table rows
            _strat_rows = ""
            for _sname, _active, _det, _fa in _combo_strategies:
                _row_bg   = "rgba(239,68,68,0.12)" if _active else "transparent"
                _dot_col  = "#ef4444" if _active else "#334155"
                _dot      = f'<span style="color:{_dot_col};font-size:10px;">&#9679;</span>'
                _star     = " ★" if "★" in _fa else ""
                _fa_clean = _fa.replace(" ★", "")
                _star_html = f'<span style="color:#f59e0b;">{_star}</span>' if _star else ""
                _strat_rows += (
                    f'<tr style="background:{_row_bg};">'
                    f'<td style="padding:3px 8px;color:#94a3b8;font-size:10px;">{_dot} {_sname}</td>'
                    f'<td style="padding:3px 8px;text-align:center;font-size:10px;'
                    f'color:{"#f87171" if _active else "#64748b"};">'
                    f'{"FIRING" if _active else "quiet"}</td>'
                    f'<td style="padding:3px 8px;text-align:center;font-size:10px;color:#64748b;">{_det}</td>'
                    f'<td style="padding:3px 8px;text-align:center;font-size:10px;color:#475569;">{_fa_clean}{_star_html}</td>'
                    f'</tr>'
                )

            _shad_note = (
                '<div style="color:#f59e0b;font-size:9px;margin-top:4px;">'
                '&#9888; Shadow brain not scored yet — score from QIR first</div>'
                if _shad_missing else ""
            )

            _combo_html = f"""
<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;
            padding:14px 16px;margin:12px 0;">
  <!-- Header row -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:0.12em;
                text-transform:uppercase;">MAIN + SHADOW COMBO GATE</div>
    <div style="background:{_gate_color}22;border:1px solid {_gate_color}55;
                border-radius:4px;padding:2px 10px;">
      <span style="font-size:11px;font-weight:800;color:{_gate_color};
                   letter-spacing:0.08em;">{_gate_label}</span>
    </div>
  </div>

  <!-- Dual CI bars -->
  <div style="display:flex;gap:12px;margin-bottom:10px;">
    <!-- Main brain bar -->
    <div style="flex:1;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:3px;">
        MAIN BRAIN CI%</div>
      <div style="background:#1e293b;border-radius:3px;height:8px;position:relative;">
        <div style="height:8px;border-radius:3px;width:{min(_combo_main_ci,100):.0f}%;
                    background:{"#ef4444" if _combo_main_ci>=40 else "#f59e0b" if _combo_main_ci>=22 else "#22c55e"};
                    transition:width 0.4s;"></div>
        <div style="position:absolute;top:0;left:22%;width:1px;height:8px;
                    background:#64748b55;"></div>
        <div style="position:absolute;top:0;left:40%;width:1px;height:8px;
                    background:#ef444488;"></div>
      </div>
      <div style="font-size:10px;font-weight:700;
                  color:{"#ef4444" if _combo_main_ci>=40 else "#f59e0b" if _combo_main_ci>=22 else "#22c55e"};
                  margin-top:2px;">{_combo_main_ci:.1f}%
        <span style="font-size:8px;color:#475569;font-weight:400;">
          {"Zone 3" if _combo_main_ci>=40 else "Zone 2" if _combo_main_ci>=22 else "Zone 1"}</span>
      </div>
    </div>
    <!-- Shadow brain bar -->
    <div style="flex:1;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:3px;">
        SHADOW BRAIN CI%</div>
      <div style="background:#1e293b;border-radius:3px;height:8px;position:relative;">
        <div style="height:8px;border-radius:3px;
                    width:{min(_combo_shad_ci or 0,100):.0f}%;
                    background:{"#ef4444" if (_combo_shad_ci or 0)>=40 else "#f59e0b" if (_combo_shad_ci or 0)>=22 else "#22c55e"};
                    transition:width 0.4s;"></div>
        <div style="position:absolute;top:0;left:22%;width:1px;height:8px;
                    background:#64748b55;"></div>
        <div style="position:absolute;top:0;left:40%;width:1px;height:8px;
                    background:#ef444488;"></div>
      </div>
      <div style="font-size:10px;font-weight:700;
                  color:{"#ef4444" if (_combo_shad_ci or 0)>=40 else "#f59e0b" if (_combo_shad_ci or 0)>=22 else "#22c55e"};
                  margin-top:2px;">{_shad_ci_str}
        <span style="font-size:8px;color:#475569;font-weight:400;">
          {"Zone 3" if (_combo_shad_ci or 0)>=40 else "Zone 2" if (_combo_shad_ci or 0)>=22 else "Zone 1"}</span>
      </div>
    </div>
    <!-- Firing count -->
    <div style="text-align:center;min-width:60px;">
      <div style="font-size:9px;color:#64748b;letter-spacing:0.08em;margin-bottom:3px;">
        GATES FIRING</div>
      <div style="font-size:26px;font-weight:800;line-height:1;
                  color:{_gate_color};">{_n_firing}</div>
      <div style="font-size:8px;color:#475569;">of 6</div>
    </div>
  </div>

  <!-- Strategy table -->
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="border-bottom:1px solid #1e293b;">
        <th style="padding:3px 8px;text-align:left;font-size:9px;
                   color:#334155;letter-spacing:0.08em;font-weight:600;">STRATEGY</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;
                   color:#334155;letter-spacing:0.08em;font-weight:600;">NOW</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;
                   color:#334155;letter-spacing:0.08em;font-weight:600;">DETECTION</th>
        <th style="padding:3px 8px;text-align:center;font-size:9px;
                   color:#334155;letter-spacing:0.08em;font-weight:600;">FALSE ALARMS</th>
      </tr>
    </thead>
    <tbody>{_strat_rows}</tbody>
  </table>
  {_shad_note}
  <div style="font-size:8px;color:#334155;margin-top:6px;line-height:1.6;">
    Backtest: 8 crashes 2012–2026 · 1,098 normal days · ★ = best tradeoff (88% detection, 0% FA)
  </div>
  <div style="font-size:8px;color:#475569;line-height:1.7;margin-top:3px;
              border-top:1px solid #1e293b;padding-top:5px;">
    The one crash missed in all 88% strategies is <span style="color:#64748b;">2022-01 Rate Shock</span> — the macro brain never registered it
    (z=0, bull regime throughout), and the shadow brain didn't spike either. That's a structural blind spot:
    slow Fed rate hikes don't create LL spikes in either model.
  </div>
</div>"""
            st.markdown(_combo_html, unsafe_allow_html=True)
        except Exception:
            pass

        # ── Velocity Cascade V2 (continuous conviction scores) ─────────────────
        _cascade_block = ""
        try:
            import math as _math
            import numpy as np

            # ── Raw data collection ────────────────────────────────────────────
            _of_ctx_vc  = st.session_state.get("_options_flow_context") or {}
            _of_score_vc = float(_of_ctx_vc.get("options_score", 50) or 50)
            _of_label_vc = _of_ctx_vc.get("label", "—") or "—"
            # VIX slope signal from options context signals list
            _of_sigs_vc = _of_ctx_vc.get("signals", []) or []
            _vix_slope_sig = next((s for s in _of_sigs_vc if "VIX Slope" in s.get("Signal", "")), None)
            _vix_slope_score_vc = float(_vix_slope_sig["Score"]) if _vix_slope_sig else 0.0
            _vix_slope_val_vc = _vix_slope_sig["Value"] if _vix_slope_sig else "—"

            _gex_vc     = st.session_state.get("_gex_profile_spx") or {}
            _gex_zone_vc = _gex_vc.get("zone", "—") or "—"
            _gex_flip_vc = _gex_vc.get("gamma_flip")
            _gex_spot_vc = float(_gex_vc.get("spot") or 0)
            _gex_total_vc = float(_gex_vc.get("total_gex") or 0)
            _flip_pct_vc = ((float(_gex_flip_vc) - _gex_spot_vc) / _gex_spot_vc * 100.0
                            if _gex_flip_vc and _gex_spot_vc else 0.0)
            _gex_score_vc = float(np.tanh(_gex_total_vc / 1500.0))  # [-1, +1]

            _sh_vc       = st.session_state.get("_shadow_state_obj")
            _sh_label_vc = getattr(_sh_vc, "state_label", "—") if _sh_vc else "—"
            _sh_ci_vc    = float(getattr(_sh_vc, "ci_pct", 0.0) or 0.0) if _sh_vc else 0.0
            _sh_crash_vc = float((getattr(_sh_vc, "crash_prob_10pct", 0.0) or 0.0)) if _sh_vc else 0.0

            _tac_vc      = st.session_state.get("_tactical_context") or {}
            _tac_score_vc = float(_tac_vc.get("tactical_score", 50) or 50)

            _hmm_crisis_vc = st.session_state.get("_ll_anchored_crisis") or {}
            _hmm_label_vc  = _hmm_crisis_vc.get("hmm_state", "—") or "—"
            _hmm_ci_vc     = float(_hmm_crisis_vc.get("ci_pct", 0.0) or 0.0)
            _hmm_crash_vc  = float(_hmm_crisis_vc.get("crash_prob", 0.0) or 0.0)

            _fg_vc = float(st.session_state.get("_fear_greed") or 50)

            # ── Continuous scores [-1, +1] per tier ───────────────────────────
            # FAST (0-1d): options, VIX slope, GEX zone
            _fast_of   = (_of_score_vc - 50.0) / 50.0
            _fast_score = float(np.clip(
                0.5 * _fast_of + 0.3 * _vix_slope_score_vc + 0.2 * _gex_score_vc,
                -1.0, 1.0
            ))

            # MEDIUM (1-5d): shadow brain state + CI + tactical
            _sh_state_map = {
                "Strong Bull": 1.0, "Mild Bull": 0.5, "Transition": 0.0,
                "Mild Bear": -0.5, "Strong Bear": -0.8, "Crisis": -1.0,
            }
            _sh_dir_score = _sh_state_map.get(_sh_label_vc, 0.0)
            _sh_ci_score  = -float(np.tanh(_sh_ci_vc / 100.0))  # high CI = bearish
            _tac_score_vc_norm = (_tac_score_vc - 50.0) / 50.0
            _med_score = float(np.clip(
                0.4 * _sh_dir_score + 0.3 * _sh_ci_score + 0.3 * _tac_score_vc_norm,
                -1.0, 1.0
            ))

            # SLOW (5-30d): primary HMM state + CI
            _hmm_state_map = {
                "Bull": 1.0, "Neutral": 0.0,
                "Late Cycle": -0.3, "Stress": -0.7, "Crisis": -1.0,
            }
            _hmm_dir_score = _hmm_state_map.get(_hmm_label_vc, 0.0)
            _hmm_ci_score  = -float(np.tanh(_hmm_ci_vc / 100.0))
            _slow_score = float(np.clip(
                0.5 * _hmm_dir_score + 0.5 * _hmm_ci_score,
                -1.0, 1.0
            ))

            # ── Composite conviction (fast-weighted for early detection) ───────
            _composite = float(np.clip(
                0.45 * _fast_score + 0.35 * _med_score + 0.20 * _slow_score,
                -1.0, 1.0
            ))
            _conv_pct = int(abs(_composite) * 100)
            if _composite > 0.15:
                _conv_label = f"{_conv_pct}% BULLISH"
                _conv_color = "#22c55e" if _composite > 0.5 else "#86efac"
            elif _composite < -0.15:
                _conv_label = f"{_conv_pct}% BEARISH"
                _conv_color = "#ef4444" if _composite < -0.5 else "#fca5a5"
            else:
                _conv_label = "NEUTRAL"
                _conv_color = "#64748b"

            # ── Divergence detection ──────────────────────────────────────────
            _div_fm = abs(_fast_score - _med_score)
            _div_fs = abs(_fast_score - _slow_score)
            _div_ms = abs(_med_score - _slow_score)
            _divergence = max(_div_fm, _div_fs, _div_ms)
            _opp_signs = (_fast_score * _slow_score < 0)

            if _fast_score < -0.3 and _slow_score > 0.0 and _divergence > 0.4:
                _div_label = "⚠ TOP FORMING"
                _div_color = "#ef4444"
            elif _fast_score > 0.3 and _slow_score < 0.0 and _divergence > 0.4:
                _div_label = "↗ BOTTOM FORMING"
                _div_color = "#22c55e"
            elif _divergence > 0.6 and _opp_signs:
                _div_label = "DIVERGING"
                _div_color = "#f59e0b"
            elif _divergence < 0.2 and abs(_composite) > 0.3:
                _div_label = "ALL ALIGNED"
                _div_color = _conv_color
            else:
                _div_label = ""
                _div_color = "#475569"

            # ── Score bar helper ──────────────────────────────────────────────
            def _score_bar(score: float, col: str) -> str:
                """Thin horizontal bar showing score position [-1, +1]."""
                _pct = int((score + 1.0) / 2.0 * 100)
                _mid = 50
                if _pct >= _mid:
                    _left = f"{_mid}%"; _width = f"{_pct - _mid}%"; _bgcol = col
                else:
                    _left = f"{_pct}%"; _width = f"{_mid - _pct}%"; _bgcol = col
                return (
                    f'<div style="position:relative;height:3px;background:#1e293b;border-radius:2px;margin-top:3px;">'
                    f'<div style="position:absolute;top:0;left:50%;width:1px;height:3px;background:#334155;"></div>'
                    f'<div style="position:absolute;top:0;left:{_left};width:{_width};height:3px;'
                    f'background:{_bgcol};border-radius:2px;"></div></div>'
                )

            # ── Confirmation progress bars ─────────────────────────────────────
            def _confirm_bar(current: float, gate: float, label: str, col: str) -> str:
                """Progress toward a gate threshold."""
                _pct = min(100, int(current / gate * 100)) if gate > 0 else 0
                return (
                    f'<div style="font-size:7px;color:#475569;margin-top:2px;">'
                    f'{label}: {current:.0f}% → {gate:.0f}% gate '
                    f'<span style="color:{col};font-weight:700;">({_pct}%)</span>'
                    f'<div style="height:2px;background:#1e293b;border-radius:1px;margin-top:1px;">'
                    f'<div style="height:2px;width:{_pct}%;background:{col};border-radius:1px;"></div>'
                    f'</div></div>'
                )

            # ── Tier row renderer ─────────────────────────────────────────────
            def _vc_row_v2(tier_label: str, speed: str, score: float,
                           metrics: list, confirm_html: str) -> str:
                _col = ("#22c55e" if score > 0.5 else
                        "#86efac" if score > 0.15 else
                        "#64748b" if abs(score) <= 0.15 else
                        "#fca5a5" if score > -0.5 else "#ef4444")
                _score_lbl = (f"+{score:.2f}" if score >= 0 else f"{score:.2f}")
                _cells = "".join(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:7px;color:#475569;font-weight:700;letter-spacing:0.04em;">{lbl}</div>'
                    f'<div style="font-size:11px;font-weight:800;color:{mc};">{val}</div>'
                    f'</div>'
                    for lbl, val, mc in metrics
                )
                return (
                    f'<div style="border-left:2px solid {_col};padding-left:8px;margin-bottom:6px;">'
                    f'<div style="display:grid;grid-template-columns:80px 1fr;gap:6px;align-items:start;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:{_col};font-weight:800;letter-spacing:0.08em;">{tier_label}</div>'
                    f'<div style="font-size:7px;color:#475569;">{speed}</div>'
                    f'<div style="font-size:9px;font-weight:800;color:{_col};margin-top:1px;">{_score_lbl}</div>'
                    f'{_score_bar(score, _col)}'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);gap:4px;">'
                    f'{_cells}</div></div>'
                    f'{confirm_html}'
                    f'</div>'
                )

            # Confirmation progress toward stress gates (only shown when fast is stressed)
            _fast_confirm_html = ""
            _fg_col = "#22c55e" if _fg_vc > 50 else "#ef4444"

            _med_confirm_html = ""
            if _fast_score < -0.3:  # fast bearish — show shadow progress to gate
                _med_confirm_html = _confirm_bar(_sh_ci_vc, 22.0, "Shadow CI → stress gate",
                                                 "#ef4444" if _sh_ci_vc > 22 else "#f59e0b")
            elif _fast_score > 0.3:  # fast bullish — show shadow state
                pass

            _slow_confirm_html = ""
            if _fast_score < -0.3:
                _slow_confirm_html = _confirm_bar(_hmm_ci_vc, 22.0, "Primary CI → stress gate",
                                                  "#ef4444" if _hmm_ci_vc > 22 else "#f59e0b")

            _fast_row = _vc_row_v2("FAST", "0-1 days", _fast_score, [
                ("OPTIONS", f"{_of_score_vc:.0f}/100",
                 "#22c55e" if _of_score_vc >= 65 else ("#ef4444" if _of_score_vc < 38 else "#94a3b8")),
                ("VIX SLOPE", _vix_slope_val_vc.split("·")[0].strip()[:12] if _vix_slope_val_vc != "—" else "—",
                 "#22c55e" if _vix_slope_score_vc > 0.1 else ("#ef4444" if _vix_slope_score_vc < -0.1 else "#94a3b8")),
                ("GEX ZONE", _gex_zone_vc.replace(" Gamma Zone", "").replace(" Zone", "")[:8],
                 "#22c55e" if _gex_score_vc > 0.1 else ("#ef4444" if _gex_score_vc < -0.1 else "#94a3b8")),
                ("F&G", f"{_fg_vc:.0f}", _fg_col),
            ], _fast_confirm_html)

            _sh_crash_col_vc = "#22c55e" if _sh_crash_vc < 0.05 else ("#f59e0b" if _sh_crash_vc < 0.10 else "#ef4444")
            _med_row = _vc_row_v2("MEDIUM", "1-5 days", _med_score, [
                ("SHADOW", _sh_label_vc[:10],
                 "#22c55e" if _sh_dir_score > 0 else ("#ef4444" if _sh_dir_score < 0 else "#94a3b8")),
                ("SHADOW CI", f"{_sh_ci_vc:.0f}%",
                 "#22c55e" if _sh_ci_vc < 22 else ("#f59e0b" if _sh_ci_vc < 67 else "#ef4444")),
                ("TACTICAL", f"{_tac_score_vc:.0f}/100",
                 "#22c55e" if _tac_score_vc >= 65 else ("#ef4444" if _tac_score_vc < 38 else "#94a3b8")),
                ("CRASH", f"{_sh_crash_vc*100:.0f}%", _sh_crash_col_vc),
            ], _med_confirm_html)

            _slow_row = _vc_row_v2("SLOW", "5-30 days", _slow_score, [
                ("PRIMARY", _hmm_label_vc[:10],
                 "#22c55e" if _hmm_dir_score > 0 else ("#ef4444" if _hmm_dir_score < 0 else "#94a3b8")),
                ("PRIMARY CI", f"{_hmm_ci_vc:.0f}%",
                 "#22c55e" if _hmm_ci_vc < 22 else ("#f59e0b" if _hmm_ci_vc < 67 else "#ef4444")),
                ("CRASH", f"{_hmm_crash_vc*100:.0f}%",
                 "#22c55e" if _hmm_crash_vc < 0.04 else ("#f59e0b" if _hmm_crash_vc < 0.08 else "#ef4444")),
            ], _slow_confirm_html)

            _div_badge = (
                f'<span style="font-size:8px;font-weight:800;color:{_div_color};'
                f'background:{_div_color}18;padding:1px 6px;border-radius:3px;'
                f'border:1px solid {_div_color}44;margin-left:6px;">{_div_label}</span>'
            ) if _div_label else ""

            _cascade_block = (
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                # Header row: title + composite conviction
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<span style="font-size:10px;color:#475569;font-weight:700;letter-spacing:0.1em;">⚡ VELOCITY CASCADE</span>'
                f'<div style="display:flex;align-items:center;">'
                f'<span style="font-size:13px;font-weight:900;color:{_conv_color};">{_conv_label}</span>'
                f'{_div_badge}'
                f'</div>'
                f'</div>'
                # Tier rows
                f'{_fast_row}{_med_row}{_slow_row}'
                # Footer
                f'<div style="font-size:7px;color:#334155;margin-top:4px;line-height:1.4;">'
                f'Composite = 45% Fast + 35% Med + 20% Slow · '
                f'Score bar shows [-1 bearish ← 0 → +1 bullish] · '
                f'Divergence = tiers pointing opposite directions</div>'
                f'</div>'
            )
        except Exception:
            pass

        # ── Bottom Watch card (always visible; greyed when CI% < 22) ───────────
        _bw_block = ""
        # _ci in outer scope (same formula as inside _build_ll_anchored_block)
        _ci = max(0.0, (abs(_tb_ll_z) / _ci_anchor() * 100.0) if _tb_ll_z < 0 else 0.0)
        try:
            _bw_live = _ci >= 22.0 and bool(_tb_bw)
            _bw_score = (_tb_bw.get("score", 0) if _tb_bw else 0) if _bw_live else 0
            _bw_active = (_tb_bw.get("active", False) if _tb_bw else False) if _bw_live else False
            _bw_sigs  = (_tb_bw.get("signals", {}) if _tb_bw else {}) if _bw_live else {}
            _bw_vals  = (_tb_bw.get("values", {}) if _tb_bw else {}) if _bw_live else {}
            _bw_note  = (_tb_bw.get("note", "") if _tb_bw else "") if _bw_live else ""

            if _bw_live:
                _bw_score = _tb_bw.get("score", 0)
                _bw_active = _tb_bw.get("active", False)
                _bw_sigs = _tb_bw.get("signals", {})
                _bw_vals = _tb_bw.get("values", {})
                _bw_note = _tb_bw.get("note", "")

                # Color by score: 4=emerald, 3=yellow, 1-2=slate, 0=slate
                if _bw_score == 4:
                    _bw_color = "#10b981"   # emerald — confirmed
                    _bw_label = "BOTTOM CONFIRMED"
                elif _bw_score == 3:
                    _bw_color = "#f59e0b"   # amber — watch
                    _bw_label = "BOTTOM WATCH"
                elif _bw_score >= 1:
                    _bw_color = "#64748b"   # slate — early
                    _bw_label = "EARLY SIGNALS"
                else:
                    _bw_color = "#334155"   # dark slate — no signal
                    _bw_label = "NO SIGNAL"

                # Signal pills row
                _sig_cfg = [
                    ("LL↑", "ll_recovery",  "LL Recovering"),
                    ("VIX", "vix_elevated", "VIX Normalized"),
                    ("HY↓", "hy_compress",  "HY Compressing"),
                    ("VVIX", "vvix_compress", "VVIX Compressing"),
                ]
                _pill_html = ""
                for _plabel, _pkey, _ptitle in _sig_cfg:
                    _pfiring = _bw_sigs.get(_pkey, False)
                    _pcol = "#10b981" if _pfiring else "#334155"
                    _ptxt = "#ecfdf5" if _pfiring else "#64748b"
                    _pill_html += (
                        f'<span title="{_ptitle}" style="display:inline-block;background:{_pcol}22;'
                        f'border:1px solid {_pcol};border-radius:3px;padding:1px 6px;margin-right:4px;'
                        f'font-size:9px;color:{_ptxt};font-weight:700;">'
                        f'{"✓ " if _pfiring else "○ "}{_plabel}</span>'
                    )

                # Values row
                _vix_now = _bw_vals.get("vix_now")
                _vix_peak = _bw_vals.get("vix_60d_peak")
                _hy_now = _bw_vals.get("hy_now")
                _vvix_now = _bw_vals.get("vvix_now")
                _vals_parts = []
                if _vix_now is not None:
                    _vals_parts.append(f"VIX {_vix_now:.1f} (pk {_vix_peak:.0f})")
                if _hy_now is not None:
                    _vals_parts.append(f"HY {_hy_now:.2f}%")
                if _vvix_now is not None:
                    _vals_parts.append(f"VVIX {_vvix_now:.0f}")
                _vals_str = " · ".join(_vals_parts) if _vals_parts else ""

                # Score bar (4 segments)
                _bar_html = ""
                for _i in range(4):
                    _seg_col = _bw_color if _i < _bw_score else "#1e293b"
                    _bar_html += (
                        f'<div style="flex:1;height:4px;background:{_seg_col};'
                        f'border-radius:2px;margin-right:{"0" if _i==3 else "2"}px;"></div>'
                    )

                _bw_block = (
                    f'<div style="background:#0f172a;border:1px solid {_bw_color}44;'
                    f'border-left:3px solid {_bw_color};border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:10px;">'

                    # Header
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<span style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.12em;">BOTTOM WATCH</span>'
                    f'<span style="font-size:10px;color:{_bw_color};font-weight:800;">{_bw_score}/4</span>'
                    f'</div>'

                    # Status label
                    f'<div style="font-size:13px;color:{_bw_color};font-weight:800;margin-bottom:5px;">{_bw_label}</div>'

                    # Score bar
                    f'<div style="display:flex;margin-bottom:8px;">{_bar_html}</div>'

                    # Signal pills
                    f'<div style="margin-bottom:7px;">{_pill_html}</div>'

                    # Values
                    + (f'<div style="font-size:9px;color:#475569;margin-bottom:5px;">{_vals_str}</div>' if _vals_str else "")

                    # Note
                    + f'<div style="border-top:1px solid #1e293b;padding-top:6px;font-size:9px;color:#64748b;">'
                    + f'<span style="color:{_bw_color};">▸</span> {_bw_note}</div>'

                    # Tips
                    + f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid #0f172a;font-size:8px;color:#334155;line-height:1.6;">'
                    + f'<span style="color:#1e3a5f;font-weight:700;letter-spacing:0.08em;">HOW TO READ</span><br>'
                    + f'4/4 = confirmed (9% crash prob = 3x baseline) · 3/4 = watch, wait for last signal<br>'
                    + f'Signal fires <em>after</em> price low (+8 to +66d) — confirms worst is over, not the exact tick<br>'
                    + f'Volmageddon (pure vol shock, no credit stress) never fires HY — that\'s by design<br>'
                    + f'VIX pill = 60d peak ≥ 28 AND now below 24 · VVIX = tail-risk proxy (P/C equivalent)'
                    + f'</div>'

                    # Signal count legend
                    + f'<div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;">'
                    + f'<div style="background:#33415511;border:1px solid #33415544;border-radius:3px;padding:5px 6px;text-align:center;">'
                    + f'<div style="font-size:10px;font-weight:800;color:#475569;">1/4</div>'
                    + f'<div style="font-size:7px;color:#334155;margin-top:2px;line-height:1.4;">Early<br>noise</div>'
                    + f'</div>'
                    + f'<div style="background:#1e3a5f22;border:1px solid #1e3a5f55;border-radius:3px;padding:5px 6px;text-align:center;">'
                    + f'<div style="font-size:10px;font-weight:800;color:#3b82f6;">2/4</div>'
                    + f'<div style="font-size:7px;color:#334155;margin-top:2px;line-height:1.4;">Tentative<br>watch</div>'
                    + f'</div>'
                    + f'<div style="background:#f59e0b11;border:1px solid #f59e0b44;border-radius:3px;padding:5px 6px;text-align:center;">'
                    + f'<div style="font-size:10px;font-weight:800;color:#f59e0b;">3/4</div>'
                    + f'<div style="font-size:7px;color:#334155;margin-top:2px;line-height:1.4;">High alert<br>wait last</div>'
                    + f'</div>'
                    + f'<div style="background:#10b98122;border:1px solid #10b98166;border-radius:3px;padding:5px 6px;text-align:center;">'
                    + f'<div style="font-size:10px;font-weight:800;color:#10b981;">4/4</div>'
                    + f'<div style="font-size:7px;color:#10b98199;margin-top:2px;line-height:1.4;">Confirmed<br>3x baseline risk</div>'
                    + f'</div>'
                    + f'</div>'

                    f'</div>'
                )
            else:
                # Greyed-out dormant state — always visible so you know the card exists
                _bw_block = (
                    f'<div style="background:#0a0f1a;border:1px solid #1e293b55;'
                    f'border-left:3px solid #1e293b;border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:10px;opacity:0.55;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<span style="font-size:8px;color:#334155;font-weight:700;letter-spacing:0.12em;">BOTTOM WATCH</span>'
                    f'<span style="font-size:8px;color:#1e3a5f;font-weight:700;">DORMANT · CI% &lt; 22</span>'
                    f'</div>'
                    f'<div style="font-size:12px;color:#334155;font-weight:700;margin-bottom:6px;">— INACTIVE —</div>'
                    f'<div style="display:flex;gap:4px;margin-bottom:8px;">'
                    + "".join(
                        f'<span style="display:inline-block;background:#1e293b22;border:1px solid #1e293b;'
                        f'border-radius:3px;padding:1px 6px;font-size:9px;color:#1e3a5f;font-weight:700;">'
                        f'○ {lbl}</span>'
                        for lbl in ["LL↑", "VIX", "HY↓", "VVIX"]
                    )
                    + f'</div>'
                    f'<div style="border-top:1px solid #1e293b33;padding-top:6px;font-size:8px;color:#1e3a5f;line-height:1.6;">'
                    f'Activates when CI% ≥ 22 (stress/crisis regime) · 4/4 signals = 9% crash prob (3x baseline)<br>'
                    f'Monitors: LL recovery · VIX normalization · HY compression · VVIX compression'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

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
    _slow_html = ""  # HMM and Bottom Watch rendered separately below

    # ── MEDIUM: structural Kelly sizing + contextual signals + entry card ────
    _medium_html = ""
    if _populated:
        _medium_parts = []
        if _conviction_block:
            _medium_parts.append(_conviction_block)

        # ── Kritzman contagion adjustment banner ──────────────────────────────
        if _kritzman_label and _conviction_score is not None:
            _adj_conviction = int(round(_conviction_score * _kritzman_mult))
            _k_banner = (
                f'<div style="background:#0a0512;border:1px solid #7c3aed44;'
                f'border-left:3px solid #7c3aed;border-radius:6px;'
                f'padding:8px 14px;margin-bottom:10px;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;">'
                f'<div>'
                f'<div style="font-size:9px;color:#7c3aed;font-weight:700;letter-spacing:0.1em;margin-bottom:3px;">'
                f'KRITZMAN TURBULENCE ADJ · SKULLS (FAJ 2010)</div>'
                f'<div style="font-size:10px;color:#94a3b8;">'
                f'Contagion {_k_score:.0f} → signal value ratio {_kritzman_mult:.2f}x</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<span style="font-size:20px;font-weight:900;color:#64748b;">{_conviction_score}</span>'
                f'<span style="font-size:14px;color:#a78bfa;margin:0 6px;">→</span>'
                f'<span style="font-size:20px;font-weight:900;color:#a78bfa;">{_adj_conviction}</span>'
                f'<div style="font-size:8px;color:#64748b;">adj conviction</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            _medium_parts.append(_k_banner)

        # ── Contagion block (confirmatory signal, after conviction) ───────────
        try:
            from services.market_data import fetch_correlation_matrix as _fcm
            import numpy as _np
            _corr_uni = ("SPY", "QQQ", "TLT", "GLD", "UUP", "HYG", "^VIX", "USO", "EEM")
            _corr_30 = _fcm(_corr_uni, period="1mo")
            _corr_90 = _fcm(_corr_uni, period="3mo")
            if _corr_30 is not None and not _corr_30.empty:
                _cmask = _np.ones(_corr_30.shape, dtype=bool)
                _np.fill_diagonal(_cmask, False)
                _c_score = min(100.0, _np.abs(_corr_30.values[_cmask]).mean() / 0.8 * 100.0)
                _c_color = ("#ff1744" if _c_score >= 80 else
                            "#ef4444" if _c_score >= 60 else
                            "#f59e0b" if _c_score >= 30 else "#00c853")
                _c_label = ("CRISIS" if _c_score >= 80 else
                            "CONTAGION RISK" if _c_score >= 60 else
                            "ELEVATED" if _c_score >= 30 else "HEALTHY")
                _c_note = ("Everything correlated — validate all QIR signals before acting" if _c_score >= 80 else
                           "Broad risk-off selling — cross-check QIR conviction" if _c_score >= 60 else
                           "Markets starting to move together — stay alert" if _c_score >= 30 else
                           "Assets behaving independently — normal conditions")
                # Top shifted pair
                _c_shift_html = ""
                if _corr_90 is not None and not _corr_90.empty:
                    _c_tks = list(_corr_30.columns)
                    _c_max_d, _c_max_pair = 0.0, ""
                    for _ci in range(len(_c_tks)):
                        for _cj in range(_ci + 1, len(_c_tks)):
                            _cd = abs(_corr_30.iloc[_ci, _cj] - _corr_90.iloc[_ci, _cj])
                            if _cd > _c_max_d:
                                _c_max_d = _cd
                                _c_max_pair = f"{_c_tks[_ci]}/{_c_tks[_cj]}"
                                _c_max_cur = _corr_30.iloc[_ci, _cj]
                    if _c_max_pair and _c_max_d >= 0.2:
                        _c_shift_html = (
                            f'<span style="font-size:8px;color:#64748b;margin-left:10px;">'
                            f'Biggest shift: <span style="color:#94a3b8;">{_c_max_pair}</span>'
                            f' Δ{_c_max_d:+.2f} (now {_c_max_cur:.2f})</span>'
                        )
                _medium_parts.append(
                    f'<div style="background:#0f172a;border:1px solid {_c_color}33;'
                    f'border-left:3px solid {_c_color};border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                    f'<div>'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:2px;">CONTAGION INDEX</div>'
                    f'<div style="font-size:22px;font-weight:900;color:{_c_color};line-height:1;">'
                    f'{_c_score:.0f}</div>'
                    f'</div>'
                    f'<div style="flex:1;">'
                    f'<span style="background:{_c_color}22;color:{_c_color};font-size:9px;'
                    f'font-weight:700;padding:2px 8px;border-radius:10px;'
                    f'border:1px solid {_c_color}44;">{_c_label}</span>'
                    f'<div style="font-size:9px;color:#64748b;margin-top:4px;">{_c_note}</div>'
                    f'{_c_shift_html}'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

        if _kelly_block:
            _medium_parts.append(f'<div style="margin-bottom:10px;">{_kelly_block}</div>')
        if _verdict_html:
            _medium_parts.append(_verdict_html)
        if _lean_card:
            _medium_parts.append(_lean_card)
        _medium_parts.append(_gex_block + _signal_breakdown_block)
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

    # ── Render the full dashboard — two tabs ─────────────────────────────────
    def _tf_divider(label: str, color: str = "#1e293b") -> str:
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;">'
            f'<div style="flex:1;height:1px;background:{color};"></div>'
            f'<span style="font-size:7px;color:#334155;font-weight:700;letter-spacing:0.12em;'
            f'white-space:nowrap;">{label}</span>'
            f'<div style="flex:1;height:1px;background:{color};"></div>'
            f'</div>'
        )

    _tab_macro, _tab_brain = st.tabs(["⚡  MACRO & TACTICAL", "🧠  BRAIN SIGNALS"])

    with _tab_macro:
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid {_border_color};border-radius:8px;'
            f'box-shadow:{_border_glow};padding:14px 16px;margin:8px 0 12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;'
            f'text-transform:uppercase;">QIR · Macro &amp; Tactical</div>'
            f'</div>'
            f'{_freshness_html}'
            f'{_crash_alert_html}'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
            f'<div>{_t1}</div><div>{_t2}</div><div>{_t3}</div>'
            f'</div>'
            f'{_velocity_block if _populated else ""}'
            f'{_tf_divider("⏱  SLOW — REGIME · WEEKS / MONTHS")}'
            f'{_entry_rec_html}'
            f'{_slow_html}'
            f'{_tf_divider("⏑  MEDIUM — SIZING · DAYS / WEEKS")}'
            f'{_medium_html}'
            f'{_cascade_block if _populated else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with _tab_brain:
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid {_border_color};border-radius:8px;'
            f'box-shadow:{_border_glow};padding:14px 16px;margin:8px 0 12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#475569;'
            f'text-transform:uppercase;">QIR · Brain Signals</div>'
            f'</div>'
            f'{_hmm_block if _populated else ""}'
            f'{_ll_anchored_block if _populated else ""}'
            f'{_shadow_block if _populated else ""}'
            f'{_top_block if _populated else ""}'
            f'{_bw_block if _populated else ""}'
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
        with _qc1:
            st.markdown(
                f'<div style="background:#080d14;border:1px solid #f59e0b22;border-left:2px solid #f59e0b;'
                f'padding:5px 10px;border-radius:3px;font-size:10px;color:#64748b;line-height:1.5;">'
                f'💡 <span style="color:#f59e0b;font-weight:700;">Daily habit:</span> '
                f'click <span style="color:#94a3b8;font-weight:700;">📌 Log Signal</span> every time you run QIR. '
                f'30+ logs → accuracy stats + pattern edge become meaningful.</div>',
                unsafe_allow_html=True,
            )
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

        # ── SPY Trade Journal quick-log ───────────────────────────────
        _default_dir = (
            "Long"  if _spy_prediction == "Buy"  else
            "Short" if _spy_prediction == "Sell" else
            "Long"
        )
        _ql_col, _ = st.columns([2, 3])
        with _ql_col:
            with st.popover("📒 Log SPY Trade to Journal", use_container_width=True):
                st.markdown(
                    f'<div style="color:#f59e0b;font-size:11px;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:8px;">📒 SPY TRADE JOURNAL</div>'
                    f'<div style="color:#94a3b8;font-size:10px;margin-bottom:10px;">'
                    f'Logs to trade_journal.json · auto-evaluates ATR stop/target on refresh</div>',
                    unsafe_allow_html=True,
                )
                import datetime as _dt
                # Fetch live SPY price
                _ql_spy_price = None
                try:
                    import yfinance as _yf
                    _spy_tick = _yf.Ticker("SPY")
                    _spy_hist = _spy_tick.history(period="1d", interval="1m")
                    if _spy_hist is not None and not _spy_hist.empty:
                        _ql_spy_price = round(float(_spy_hist["Close"].iloc[-1]), 2)
                except Exception:
                    pass

                if _ql_spy_price:
                    st.markdown(
                        f'<div style="background:#0a0f1a;border:1px solid #22c55e33;border-left:3px solid #22c55e;'
                        f'padding:6px 10px;border-radius:4px;margin-bottom:8px;display:flex;align-items:center;gap:10px;">'
                        f'<span style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;">SPY LIVE</span>'
                        f'<span style="font-size:18px;font-weight:900;color:#22c55e;">${_ql_spy_price:.2f}</span>'
                        f'<span style="font-size:9px;color:#475569;">click Use Price → fills entry below</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Use ${_ql_spy_price:.2f} as entry", key="ql_use_live_price", use_container_width=True):
                        st.session_state["ql_price"] = _ql_spy_price

                _ql_dir    = st.selectbox("Direction", ["Long", "Short"],
                                          index=0 if _default_dir == "Long" else 1,
                                          key="ql_dir")
                _ql_price  = st.number_input("Entry Price ($)", min_value=0.01, step=0.01,
                                             format="%.2f", key="ql_price",
                                             value=float(st.session_state.get("ql_price") or _ql_spy_price or 0.01))
                _ql_size   = st.number_input("Size (% of portfolio)",
                                             min_value=0.0, max_value=100.0, step=0.5,
                                             value=float(_kly_half) if _kly_viable and _kly_half else 0.0,
                                             key="ql_size")
                _ql_date   = st.date_input("Entry Date", value=_dt.date.today(), key="ql_date")
                _ql_pattern = _cls.get("pattern") or _verdict_label or ""
                st.caption(f"Pattern: {_ql_pattern}  ·  Kelly suggested: {_kly_half if _kly_viable else 0}%")
                _ql_notes = st.text_area("Comments", placeholder="e.g. tariff risk, earnings week, macro catalyst…", height=70, key="ql_notes")
                if st.button("📌 Log Trade", key="ql_log_btn", type="primary", use_container_width=True):
                    if _ql_price <= 0:
                        st.error("Enter a price.")
                    else:
                        import uuid as _uuid, json as _json, os as _os
                        from datetime import datetime as _datetime
                        _jpath = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "trade_journal.json")
                        try:
                            with open(_jpath, encoding="utf-8") as _f:
                                _trades = _json.load(_f)
                        except Exception:
                            _trades = []

                        # Fetch ATR levels
                        _atr_val = _stop_l = _tgt_l = None
                        try:
                            from services.forecast_tracker import _fetch_atr_and_history, _ATR_MULT_STOP, _ATR_MULT_TARGET
                            _ts_e = _datetime.fromisoformat(str(_ql_date) + "T00:00:00")
                            _ad   = _fetch_atr_and_history("SPY", _ts_e)
                            if _ad and _ad.get("atr", 0) > 0:
                                _a       = _ad["atr"]
                                _atr_val = round(_a, 4)
                                _is_s    = _ql_dir.lower() == "short"
                                _stop_l  = round(_ql_price + _ATR_MULT_STOP * _a if _is_s else _ql_price - _ATR_MULT_STOP * _a, 2)
                                _tgt_l   = round(_ql_price - _ATR_MULT_TARGET * _a if _is_s else _ql_price + _ATR_MULT_TARGET * _a, 2)
                        except Exception:
                            pass

                        _tid = str(_uuid.uuid4())[:8].upper()
                        _trades.append({
                            "id":               _tid,
                            "ticker":           "SPY",
                            "direction":        _ql_dir,
                            "entry_price":      round(_ql_price, 2),
                            "entry_date":       str(_ql_date),
                            "exit_price":       None,
                            "exit_reason":      None,
                            "status":           "open",
                            "position_size":    round(_ql_size, 2),
                            "kelly_suggested":  round(_kly_half, 1) if _kly_viable and _kly_half else None,
                            "pattern_at_entry": _ql_pattern or None,
                            "notes":            _ql_notes.strip() or None,
                            "logged_at":        _datetime.now().isoformat(),
                            "atr_at_log":       _atr_val,
                            "stop_at_log":      _stop_l,
                            "target_at_log":    _tgt_l,
                        })
                        with open(_jpath, "w", encoding="utf-8") as _f:
                            _json.dump(_trades, _f, indent=2, default=str)
                        _atr_note = f"  ·  Stop ${_stop_l:.2f} / Target ${_tgt_l:.2f}" if _stop_l else ""
                        st.success(f"✅ [{_tid}] SPY {_ql_dir} logged!{_atr_note}")


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

    # ── Brain health status banner ─────────────────────────────────────────────
    try:
        from modules.regime_chart import _brain_health, _render_status_banner, _MAIN_HISTORY, _SHADOW_HISTORY
        from services.hmm_regime import load_hmm_brain as _qir_load_main
        from services.hmm_shadow import load_shadow_brain as _qir_load_shadow
        _render_status_banner(
            _brain_health(_qir_load_main(),   _MAIN_HISTORY,   "Main"),
            _brain_health(_qir_load_shadow(), _SHADOW_HISTORY, "Shadow"),
        )
    except Exception:
        pass

    # ── Hidden AI context block — read by Gemini / browser AI sidebar ─────────
    # display:none but fully in DOM; contains structured signal state for AI
    _ai_ctx = st.session_state.get("_qir_ai_context") or {}
    if _ai_ctx:
        import json as _ctx_json
        _ctx_lines = "\n".join(f"  {k}: {v}" for k, v in _ai_ctx.items())
        st.markdown(
            f'<div id="regarded-terminals-ai-context" aria-hidden="true" '
            f'style="display:none;position:absolute;width:0;height:0;overflow:hidden;" '
            f'data-source="regarded-terminals-qir">'
            f'<pre id="rt-signal-state">\n{_ctx_lines}\n</pre>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">'
        f'<span style="font-size:13px;color:{_oc};font-weight:700;letter-spacing:0.1em;">⚡ QUICK INTEL RUN</span>'
        + (
            f'<span title="Structured signal state injected into DOM — Gemini/browser AI sidebar can read it now" '
            f'style="font-size:8px;font-weight:700;letter-spacing:0.1em;'
            f'background:#052e16;color:#4ade80;border:1px solid #166534;'
            f'padding:2px 7px;border-radius:3px;cursor:default;">'
            f'🤖 AI CONTEXT LIVE</span>'
            if _ai_ctx else
            f'<span title="Run QIR to populate AI context for Gemini" '
            f'style="font-size:8px;font-weight:700;letter-spacing:0.1em;'
            f'background:#1c1917;color:#57534e;border:1px solid #292524;'
            f'padding:2px 7px;border-radius:3px;cursor:default;">'
            f'🤖 AI CONTEXT —</span>'
        )
        + f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Runs Risk Regime + Fed Rate Path + Policy Transmission + Current Events + Doom Briefing + Whale Activity + Black Swans + Macro Synopsis + Portfolio Risk Snapshot in sequence. "
        "Navigate to Portfolio Intelligence when done."
    )
    st.markdown(
        '<div style="font-size:10px;color:#334155;font-family:\'JetBrains Mono\',Consolas,monospace;'
        'margin-top:-6px;margin-bottom:4px;">'
        '💡 Tip: Open <span style="color:#4285f4;font-weight:700;">Gemini</span> in the Chrome sidebar — '
        'after running QIR it already has your full signal state loaded.</div>',
        unsafe_allow_html=True,
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
        from services.market_data import fetch_ohlcv_single as _fetch_ohlcv
        @st.cache_data(ttl=120)
        def _fetch_vix_change():
            _h = _fetch_ohlcv("^VIX", period="2d", interval="1d")
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
        st.warning("👑 Highly Regarded uses Claude Haiku 4.5 — reserve for elevated volatility or high-conviction sessions.")
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

        # ── Auto-refresh HMM + Shadow brain state (once per day) ─────────────
        # Both brains' history files are append-once-per-day; if today's entry
        # is missing, run scoring so persistence/CI advance. Fails silently —
        # downstream load_current_*_state() falls back to last entry on error.
        try:
            from datetime import datetime as _ar_dt, timezone as _ar_tz
            from services.hmm_regime import (
                score_current_state as _ar_score_hmm,
                _load_history as _ar_hmm_hist,
            )
            from services.hmm_shadow import (
                score_current_shadow_state as _ar_score_sh,
                _load_history as _ar_sh_hist,
            )
            _ar_today = _ar_dt.now(_ar_tz.utc).strftime("%Y-%m-%d")
            _ar_hmm_h = _ar_hmm_hist()
            _ar_sh_h = _ar_sh_hist()
            _ar_hmm_due = not _ar_hmm_h or _ar_hmm_h[-1].get("date") != _ar_today
            _ar_sh_due = not _ar_sh_h or _ar_sh_h[-1].get("date") != _ar_today
            if _ar_hmm_due or _ar_sh_due:
                with st.spinner("🧠 Refreshing HMM + Shadow brain state for today..."):
                    with ThreadPoolExecutor(max_workers=2) as _ar_pool:
                        _ar_futs = []
                        if _ar_hmm_due:
                            _ar_futs.append(_ar_pool.submit(_ar_score_hmm, None, True))
                        if _ar_sh_due:
                            _ar_futs.append(_ar_pool.submit(_ar_score_sh, True))
                        for _ar_f in _ar_futs:
                            try:
                                _ar_f.result()
                            except Exception:
                                pass
        except Exception:
            pass

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

                # ── HMM + Shadow Brain context (explicitly include both brains) ──
                _llc = st.session_state.get("_ll_anchored_crisis") or {}
                _hmm_lbl = _llc.get("hmm_state", "") or (_rc.get("regime", "") if _rc else "")
                _hmm_conf = _llc.get("hmm_conf", None)
                _hmm_persist = _llc.get("hmm_persistence", None)
                _hmm_llz = _llc.get("ll_zscore", None)
                _hmm_ci = _llc.get("ci_pct", None)
                _hmm_confirms = _llc.get("confirmations", []) or []
                _shadow = st.session_state.get("_shadow_state_obj")
                _shadow_lbl = getattr(_shadow, "state_label", "") if _shadow else ""
                _shadow_llz = getattr(_shadow, "ll_zscore", None) if _shadow else None
                _shadow_ci = getattr(_shadow, "ci_pct", None) if _shadow else None
                _shadow_crash = getattr(_shadow, "crash_prob_10pct", None) if _shadow else None
                _hmm_line = f"HMM LEARN BRAIN: {_hmm_lbl or '?'}"
                if _hmm_conf is not None:
                    _hmm_line += f" | confidence {float(_hmm_conf) * 100:.0f}%"
                if _hmm_persist is not None:
                    _hmm_line += f" | persistence {int(_hmm_persist)}d"
                if _hmm_llz is not None:
                    _hmm_line += f" | ll_z {float(_hmm_llz):+.2f}"
                if _hmm_ci is not None:
                    _hmm_line += f" | CI {float(_hmm_ci):.1f}%"
                _sig_parts.append(_hmm_line)
                if _hmm_confirms:
                    _sig_parts.append("HMM CONFIRMATIONS: " + ", ".join(str(x) for x in _hmm_confirms[:4]))
                if _shadow_lbl or _shadow_llz is not None:
                    _sh_line = f"SHADOW BRAIN: {_shadow_lbl or '?'}"
                    if _shadow_llz is not None:
                        _sh_line += f" | ll_z {float(_shadow_llz):+.2f}"
                    if _shadow_ci is not None:
                        _sh_line += f" | CI {float(_shadow_ci):.1f}%"
                    if _shadow_crash is not None:
                        _sh_line += f" | 30d crash prob>{'{'}10%{'}'} {float(_shadow_crash) * 100:.0f}%"
                    _sig_parts.append(_sh_line)

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
                st.session_state["_qir_debate_signals_text"] = _signals_text_for_debate
                _synopsis = _gen_synopsis(_signals_text_for_debate, use_claude=_use_claude, model=_cl_model)
                _syn_tier = "👑 Highly Regarded Mode" if (_use_claude and _cl_model == "claude-haiku-4-5-20251001") \
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

        # ── Debate payload refresh: include full QIR context after all rounds ──
        try:
            _dbt_parts = []
            _snap_d = st.session_state.get("_portfolio_risk_snapshot") or {}
            _risk_i = st.session_state.get("_risk_matrix_interpretation") or {}
            if _snap_d:
                _dbt_parts.append(
                    "PORTFOLIO RISK SNAPSHOT: "
                    f"beta {_snap_d.get('beta', '?')} | "
                    f"delta {_snap_d.get('delta', '?')} | "
                    f"open positions {_snap_d.get('n_open', '?')}"
                )
            if _risk_i:
                _dbt_parts.append(
                    "RISK MATRIX INTERPRETATION: "
                    f"alert {_risk_i.get('alert_level', 'none')} | "
                    f"{(_risk_i.get('summary', '') or '')[:240]}"
                )
            _er_live = st.session_state.get("_qir_earnings_risk") or []
            if _er_live:
                _dbt_parts.append(
                    "EARNINGS RISK (live): " + ", ".join(
                        f"{e['ticker']} in {e['days_away']}d"
                        + (f" (±{e['expected_move_pct']:.1f}%)" if e.get("expected_move_pct") else "")
                        for e in _er_live[:6]
                    )
                )
            _llc2 = st.session_state.get("_ll_anchored_crisis") or {}
            _shadow2 = st.session_state.get("_shadow_state_obj")
            _dbt_parts.append(
                "CRISIS ENGINES: "
                f"HMM={_llc2.get('hmm_state', '?')} "
                f"(ll_z={float(_llc2.get('ll_zscore', 0.0)):+.2f}, CI={float(_llc2.get('ci_pct', 0.0)):.1f}%) | "
                f"Shadow={getattr(_shadow2, 'state_label', '?')} "
                f"(ll_z={float(getattr(_shadow2, 'll_zscore', 0.0)):+.2f}, "
                f"CI={float(getattr(_shadow2, 'ci_pct', 0.0)):.1f}%, "
                f"crash>{'{'}10%{'}'}={float(getattr(_shadow2, 'crash_prob_10pct', 0.0))*100:.0f}%)"
            )
            _base_dbt = st.session_state.get("_qir_debate_signals_text", "")
            _extra_dbt = "\n\n".join([p for p in _dbt_parts if p])
            if _extra_dbt:
                st.session_state["_qir_debate_signals_text"] = (
                    (_base_dbt + "\n\n" + _extra_dbt) if _base_dbt else _extra_dbt
                )[:12000]
        except Exception:
            pass

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

                # ── Build AI context snapshot (read by Gemini / browser AI sidebar) ──
                try:
                    import datetime as _ai_dt
                    _ai_crisis = st.session_state.get("_ll_anchored_crisis") or {}
                    _ai_ll_z   = _ai_crisis.get("ll_zscore", 0.0) or 0.0
                    _ai_ci     = round(max(0.0, abs(_ai_ll_z) / _ci_anchor() * 100.0) if _ai_ll_z < 0 else 0.0, 1)
                    _ai_zone   = (
                        "Zone 4 · Beyond Training Range (purple)" if _ai_ci > 100.0 else
                        "Zone 3 · Crisis Gate Open (75% historical detection rate, 0% false alarms)" if _ai_ci >= 40.0 else
                        "Zone 2 · Model Stress (context signals)" if _ai_ci >= 22.0 else
                        "Zone 1 · Normal (conviction signals suppressed)"
                    )
                    _prox = st.session_state.get("_top_bottom_proximity") or {}
                    _syn  = st.session_state.get("_macro_synopsis") or {}
                    st.session_state["_qir_ai_context"] = {
                        "app":            "Regarded Terminals — Narrative Investing Tool",
                        "generated_at":   _ai_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "SIGNAL_PATTERN": _new_pat_name.replace("_", " "),
                        "buy_tier":       _new_pat["buy_tier"],
                        "short_tier":     _new_pat["short_tier"],
                        "pattern_interpretation": _new_pat.get("interpretation", ""),
                        "REGIME_SCORE":   f"{_rc_f.get('score', '?')}/100",
                        "regime_label":   _rc_f.get("regime", "?"),
                        "regime_quadrant":_rc_f.get("quadrant", "?"),
                        "leading_subscore": _rc_f.get("leading_score", "?"),
                        "TACTICAL_SCORE": f"{_tac_f.get('tactical_score', '?')}/100",
                        "tactical_label": _tac_f.get("label", "?"),
                        "OPTIONS_FLOW_SCORE": f"{_of_f.get('options_score', '?')}/100",
                        "options_label":  _of_f.get("label", "?"),
                        "CI_PCT":         f"{_ai_ci}%",
                        "ci_zone":        _ai_zone,
                        "ll_zscore":      _ai_ll_z,
                        "crisis_status":  _ai_crisis.get("status_text", ""),
                        "HMM_BRAIN_STATE": _ai_crisis.get("hmm_state", _rc_f.get("regime", "?")),
                        "hmm_confirmations": _ai_crisis.get("confirmations", []),
                        "CONVICTION":     _syn.get("conviction", "?"),
                        "macro_narrative":_syn.get("narrative", ""),
                        "top_proximity_pct": _prox.get("top_pct", "?"),
                        "bottom_proximity_pct": _prox.get("bottom_pct", "?"),
                        "data_quality":   f"{_dq_f.get('score', '?')}/100",
                        "CI_FORMULA":     "CI% = abs(ll_zscore) / brain.ci_anchor × 100 · Gate opens at 40% CI",
                        "ZONE_GUIDE":     "Normal<22% | Stress 22-40% | Crisis>=40% (75% historical detection, 0% FP) | Beyond>100%",
                        "KELLY_GUIDE":    "Half-Kelly shown — reduce by 50% in Zone 2+, 75% in Zone 3+",
                        "SHADOW_BRAIN_STATE": getattr(st.session_state.get("_shadow_state_obj"), "state_label", "?"),
                        "SHADOW_CI_PCT": f"{getattr(st.session_state.get('_shadow_state_obj'), 'ci_pct', '?')}%",
                        "SHADOW_LL_ZSCORE": getattr(st.session_state.get("_shadow_state_obj"), "ll_zscore", "?"),
                        "SHADOW_CRASH_PROB_30D": f"{(getattr(st.session_state.get('_shadow_state_obj'), 'crash_prob_10pct', 0) or 0)*100:.0f}%",
                        "SHADOW_GUIDE": "Shadow CI anchor 1.194 · Zone 3 at z<-0.80 · 95% hit rate · confirmation signal only",
                        "BRAIN_AGREEMENT": (
                            "AGREE" if (
                                _ai_crisis.get("hmm_state", "").lower() in ("crisis", "stress", "late cycle") and
                                getattr(st.session_state.get("_shadow_state_obj"), "state_label", "").lower() in ("crisis", "strong bear")
                            ) or (
                                _ai_crisis.get("hmm_state", "").lower() in ("bull", "neutral") and
                                getattr(st.session_state.get("_shadow_state_obj"), "state_label", "").lower() in ("mild bull", "strong bull")
                            ) else "DISAGREE — investigate divergence"
                        ),
                    }

                    # ── Write per-module AI context blocks (read when navigating each module) ──
                    import datetime as _ai_dt2
                    _ai_now = _ai_dt2.datetime.now().strftime("%Y-%m-%d %H:%M")
                    _raw_sigs_ai = st.session_state.get("_regime_raw_signals") or {}

                    # Risk Regime context
                    st.session_state["_risk_regime_ai_context"] = {
                        "module": "Risk Regime — Macro Risk-On/Risk-Off Dashboard",
                        "generated_at": _ai_now,
                        "REGIME_SCORE": f"{_rc_f.get('score','?')}/100",
                        "regime_label": _rc_f.get("regime", "?"),
                        "regime_quadrant": _rc_f.get("quadrant", "?"),
                        "leading_subscore": _rc_f.get("leading_score", "?"),
                        "composite_5d_trend": _rc_f.get("composite_trend", "?"),
                        "leading_5d_trend": _rc_f.get("leading_trend", "?"),
                        "macro_score": _rc_f.get("macro_score", "?"),
                        "data_quality": f"{_dq_f.get('score','?')}/100 — {_dq_f.get('label','')}",
                        "INTERPRETATION": (
                            "Score>60=Risk-On · 40-60=Neutral · <40=Risk-Off · "
                            "Leading divergence +8=early bull signal · -8=early bear signal"
                        ),
                    }

                    # Options Flow context
                    st.session_state["_options_flow_ai_context"] = {
                        "module": "Options Activity — Flow Sentiment Dashboard",
                        "generated_at": _ai_now,
                        "OPTIONS_FLOW_SCORE": f"{_of_f.get('options_score','?')}/100",
                        "flow_label": _of_f.get("label", "?"),
                        "flow_bias": _of_f.get("bias", "?"),
                        "put_call_ratio": _of_f.get("put_call_ratio", "?"),
                        "unusual_activity": _of_f.get("unusual_summary", "?"),
                        "INTERPRETATION": "Score>60=bullish flow · <40=bearish flow · divergence from regime=contrarian signal",
                    }

                    # Tactical (Wyckoff/Elliott) context
                    _ai_wyk = st.session_state.get("_ll_anchored_crisis") or {}
                    st.session_state["_wyckoff_ai_context"] = {
                        "module": "Wyckoff Method Analysis",
                        "generated_at": _ai_now,
                        "TACTICAL_SCORE": f"{_tac_f.get('tactical_score','?')}/100",
                        "tactical_label": _tac_f.get("label", "?"),
                        "wyckoff_phase": _ai_wyk.get("wyckoff_phase", "?"),
                        "wyckoff_confidence": f"{_ai_wyk.get('wyckoff_conf', '?')}%",
                        "top_signals": _ai_wyk.get("top_signals", []),
                        "bottom_signals": _ai_wyk.get("bottom_signals", []),
                        "INTERPRETATION": (
                            "Accumulation=base building/buy zone · Markup=uptrend · "
                            "Distribution=top forming/reduce · Markdown=downtrend"
                        ),
                    }

                    # Stress / Crisis context
                    st.session_state["_stress_signals_ai_context"] = {
                        "module": "Stress Signals — Crisis Detection Dashboard",
                        "generated_at": _ai_now,
                        "CI_PCT": f"{_ai_ci}%",
                        "ci_zone": _ai_zone,
                        "ll_zscore": _ai_ll_z,
                        "HMM_BRAIN_STATE": _ai_wyk.get("hmm_state", "?"),
                        "crisis_status": _ai_wyk.get("status_text", "Normal"),
                        "gate_open": _ai_wyk.get("gate_open", False),
                        "confirmations": _ai_wyk.get("confirmations", []),
                        "CI_FORMULA": "CI% = abs(ll_zscore) / brain.ci_anchor × 100",
                        "ZONE_GUIDE": "Normal<22% | Stress 22-40% | Crisis>=40% (75% historical detection, 0% FP) | Beyond>100%",
                        "SHADOW_BRAIN_STATE": getattr(st.session_state.get("_shadow_state_obj"), "state_label", "?"),
                        "SHADOW_CI_PCT": f"{getattr(st.session_state.get('_shadow_state_obj'), 'ci_pct', '?')}%",
                        "SHADOW_LL_ZSCORE": getattr(st.session_state.get("_shadow_state_obj"), "ll_zscore", "?"),
                        "BRAIN_AGREEMENT": (
                            "AGREE" if (
                                _ai_crisis.get("hmm_state", "").lower() in ("crisis", "stress", "late cycle") and
                                getattr(st.session_state.get("_shadow_state_obj"), "state_label", "").lower() in ("crisis", "strong bear")
                            ) or (
                                _ai_crisis.get("hmm_state", "").lower() in ("bull", "neutral") and
                                getattr(st.session_state.get("_shadow_state_obj"), "state_label", "").lower() in ("mild bull", "strong bull")
                            ) else "DISAGREE — investigate divergence"
                        ),
                    }

                    # Whale / Activism context
                    _whale_sum = st.session_state.get("_whale_summary", "")
                    _activ_sum = st.session_state.get("_activism_digest", "")
                    st.session_state["_whale_ai_context"] = {
                        "module": "Whale Movement — 13F Institutional & Activism Tracker",
                        "generated_at": _ai_now,
                        "whale_summary": _whale_sum[:500] if _whale_sum else "No data",
                        "activism_digest": _activ_sum[:500] if _activ_sum else "No data",
                        "INTERPRETATION": "Large 13F changes signal institutional conviction. Activism=catalyst risk.",
                    }

                    # Tail Risk context
                    _swans = st.session_state.get("_custom_swans") or {}
                    _swan_lines = [
                        f"{k}: {v.get('probability_pct',0):.1f}% annual · severity={v.get('severity','?')}"
                        for k, v in list(_swans.items())[:5]
                    ] if _swans else ["No black swan scenarios loaded"]
                    st.session_state["_tail_risk_ai_context"] = {
                        "module": "Tail Risk Studio — Black Swan Scenario Analysis",
                        "generated_at": _ai_now,
                        "black_swans": _swan_lines,
                        "regime_for_tail": _rc_f.get("regime", "?"),
                        "ci_pct_for_tail": f"{_ai_ci}%",
                        "INTERPRETATION": "Scenarios with >5% annual probability warrant portfolio hedge review.",
                    }

                    # Portfolio / Trade Journal context
                    _port_snap = st.session_state.get("_portfolio_risk_snapshot") or {}
                    _risk_interp = st.session_state.get("_risk_matrix_interpretation") or {}
                    st.session_state["_trade_journal_ai_context"] = {
                        "module": "My Regarded Portfolio — Trade Journal & Risk Matrix",
                        "generated_at": _ai_now,
                        "portfolio_beta": _port_snap.get("beta", "?"),
                        "portfolio_delta": _port_snap.get("delta", "?"),
                        "open_positions": _port_snap.get("n_open", "?"),
                        "risk_alert_level": _risk_interp.get("alert_level", "none"),
                        "risk_summary": _risk_interp.get("summary", "")[:300],
                        "earnings_risk": [
                            f"{e['ticker']} in {e['days_away']}d"
                            for e in (st.session_state.get("_qir_earnings_risk") or [])[:5]
                        ],
                        "INTERPRETATION": "Beta>1.2=high market risk · Alert=CRITICAL means reduce exposure immediately",
                    }

                    # Discovery / Narrative context
                    _narratives = st.session_state.get("_trending_narratives") or []
                    _sector_dig = st.session_state.get("_sector_regime_digest", "")
                    _narr_lines = [
                        f"{n.get('name','?')}: {n.get('momentum','?')} momentum · {n.get('tickers',[][:3])}"
                        for n in _narratives[:6]
                    ] if _narratives else ["No narrative scan data — run QIR first"]
                    st.session_state["_discovery_ai_context"] = {
                        "module": "Narrative Discovery — Ticker & Theme Scanner",
                        "generated_at": _ai_now,
                        "regime_quadrant": _rc_f.get("quadrant", "?"),
                        "regime_label": _rc_f.get("regime", "?"),
                        "trending_narratives": _narr_lines,
                        "sector_regime_digest": _sector_dig[:400] if _sector_dig else "No sector data",
                        "dominant_rate_path": st.session_state.get("_dominant_rate_path", "?"),
                        "INTERPRETATION": (
                            "Narratives with strong momentum in a Risk-On/Goldilocks regime = highest conviction longs. "
                            "Fade narratives in Risk-Off regime."
                        ),
                    }

                except Exception:
                    pass

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

        # ── Build AI context snapshot — runs regardless of round success count ──
        # Write minimal context first (guarantees pill lights up), then enrich
        import datetime as _ai_dt_base
        _rc_f2  = st.session_state.get("_regime_context") or {}
        _tac_f2 = st.session_state.get("_tactical_context") or {}
        _dq_f2  = st.session_state.get("_data_quality") or {}
        _of_f2  = st.session_state.get("_options_flow_context") or {}
        _ai_crisis2 = st.session_state.get("_ll_anchored_crisis") or {}
        _ai_ll_z2   = float(_ai_crisis2.get("ll_zscore") or 0.0)
        _ai_ci2     = round(max(0.0, abs(_ai_ll_z2) / _ci_anchor() * 100.0) if _ai_ll_z2 < 0 else 0.0, 1)
        _ai_zone2   = (
            "Zone 4 · Beyond Training Range" if _ai_ci2 > 100.0 else
            "Zone 3 · Crisis Gate Open (75% historical detection rate, 0% false alarms)" if _ai_ci2 >= 40.0 else
            "Zone 2 · Model Stress" if _ai_ci2 >= 22.0 else
            "Zone 1 · Normal"
        )
        _ai_now2 = _ai_dt_base.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Guaranteed write — pill lights up from here
        st.session_state["_qir_ai_context"] = {
            "app": "Regarded Terminals — Narrative Investing Tool",
            "generated_at": _ai_now2,
            "REGIME_SCORE": f"{_rc_f2.get('score','?')}/100",
            "regime_label": _rc_f2.get("regime", "?"),
            "regime_quadrant": _rc_f2.get("quadrant", "?"),
            "TACTICAL_SCORE": f"{_tac_f2.get('tactical_score','?')}/100",
            "OPTIONS_FLOW_SCORE": f"{_of_f2.get('options_score','?')}/100",
            "CI_PCT": f"{_ai_ci2}%",
            "ci_zone": _ai_zone2,
            "HMM_BRAIN_STATE": _ai_crisis2.get("hmm_state", _rc_f2.get("regime", "?")),
            "data_quality": f"{_dq_f2.get('score','?')}/100",
            "CI_FORMULA": "CI% = abs(ll_zscore) / brain.ci_anchor × 100 · Gate opens at 40% CI",
            "ZONE_GUIDE": "Normal<22% | Stress 22-40% | Crisis>=40% (75% historical detection, 0% FP) | Beyond>100%",
        }

        # Enrich with pattern + synopsis (may fail on partial runs — safe to skip)
        try:
            _new_pat2 = _classify_signals(_rc_f2, _tac_f2, _of_f2)
            _new_pat_name2 = _new_pat2["pattern"]
            _syn2  = st.session_state.get("_macro_synopsis") or {}
            _prox2 = st.session_state.get("_top_bottom_proximity") or {}
            st.session_state["_qir_ai_context"].update({
                "SIGNAL_PATTERN": _new_pat_name2.replace("_", " "),
                "buy_tier":       _new_pat2["buy_tier"],
                "short_tier":     _new_pat2["short_tier"],
                "pattern_interpretation": _new_pat2.get("interpretation", ""),
                "CONVICTION":     _syn2.get("conviction", "?"),
                "macro_narrative":_syn2.get("narrative", ""),
                "top_proximity_pct": _prox2.get("top_pct", "?"),
                "bottom_proximity_pct": _prox2.get("bottom_pct", "?"),
                "KELLY_GUIDE": "Half-Kelly shown — reduce by 50% in Zone 2+, 75% in Zone 3+",
            })
        except Exception:
            pass

        # Per-module context blocks (each wrapped independently)
        try:
            st.session_state["_risk_regime_ai_context"] = {
                "module": "Risk Regime — Macro Risk-On/Risk-Off Dashboard",
                "generated_at": _ai_now2,
                "REGIME_SCORE": f"{_rc_f2.get('score','?')}/100",
                "regime_label": _rc_f2.get("regime", "?"),
                "regime_quadrant": _rc_f2.get("quadrant", "?"),
                "leading_subscore": _rc_f2.get("leading_score", "?"),
                "data_quality": f"{_dq_f2.get('score','?')}/100",
                "INTERPRETATION": "Score>60=Risk-On · 40-60=Neutral · <40=Risk-Off",
            }
        except Exception:
            pass
        try:
            st.session_state["_options_flow_ai_context"] = {
                "module": "Options Activity — Flow Sentiment Dashboard",
                "generated_at": _ai_now2,
                "OPTIONS_FLOW_SCORE": f"{_of_f2.get('options_score','?')}/100",
                "flow_label": _of_f2.get("label", "?"),
                "INTERPRETATION": "Score>60=bullish flow · <40=bearish flow",
            }
        except Exception:
            pass
        try:
            st.session_state["_wyckoff_ai_context"] = {
                "module": "Wyckoff Method Analysis",
                "generated_at": _ai_now2,
                "TACTICAL_SCORE": f"{_tac_f2.get('tactical_score','?')}/100",
                "tactical_label": _tac_f2.get("label", "?"),
                "wyckoff_phase": _ai_crisis2.get("wyckoff_phase", "?"),
                "top_signals": _ai_crisis2.get("top_signals", []),
                "bottom_signals": _ai_crisis2.get("bottom_signals", []),
                "INTERPRETATION": "Accumulation=buy zone · Distribution=top forming",
            }
        except Exception:
            pass
        try:
            st.session_state["_stress_signals_ai_context"] = {
                "module": "Stress Signals — Crisis Detection Dashboard",
                "generated_at": _ai_now2,
                "CI_PCT": f"{_ai_ci2}%",
                "ci_zone": _ai_zone2,
                "ll_zscore": _ai_ll_z2,
                "HMM_BRAIN_STATE": _ai_crisis2.get("hmm_state", "?"),
                "gate_open": _ai_crisis2.get("gate_open", False),
                "confirmations": _ai_crisis2.get("confirmations", []),
            }
        except Exception:
            pass
        try:
            _swans2 = st.session_state.get("_custom_swans") or {}
            _swan_lines2 = [
                f"{k}: {v.get('probability_pct',0):.1f}% annual"
                for k, v in list(_swans2.items())[:5]
            ] if _swans2 else ["No black swan scenarios loaded"]
            st.session_state["_tail_risk_ai_context"] = {
                "module": "Tail Risk Studio — Black Swan Scenario Analysis",
                "generated_at": _ai_now2,
                "black_swans": _swan_lines2,
                "regime_for_tail": _rc_f2.get("regime", "?"),
                "ci_pct_for_tail": f"{_ai_ci2}%",
            }
        except Exception:
            pass
        try:
            _port_snap2 = st.session_state.get("_portfolio_risk_snapshot") or {}
            _risk_interp2 = st.session_state.get("_risk_matrix_interpretation") or {}
            st.session_state["_trade_journal_ai_context"] = {
                "module": "My Regarded Portfolio — Trade Journal & Risk Matrix",
                "generated_at": _ai_now2,
                "portfolio_beta": _port_snap2.get("beta", "?"),
                "open_positions": _port_snap2.get("n_open", "?"),
                "risk_alert_level": _risk_interp2.get("alert_level", "none"),
                "earnings_risk": [
                    f"{e['ticker']} in {e['days_away']}d"
                    for e in (st.session_state.get("_qir_earnings_risk") or [])[:5]
                ],
            }
        except Exception:
            pass
        try:
            _whale_sum2 = st.session_state.get("_whale_summary", "")
            _activ_sum2 = st.session_state.get("_activism_digest", "")
            st.session_state["_whale_ai_context"] = {
                "module": "Whale Movement — 13F Institutional & Activism Tracker",
                "generated_at": _ai_now2,
                "whale_summary": (_whale_sum2[:500] if _whale_sum2 else "No data"),
                "activism_digest": (_activ_sum2[:500] if _activ_sum2 else "No data"),
            }
        except Exception:
            pass
        try:
            _narratives2 = st.session_state.get("_trending_narratives") or []
            _sector_dig2 = st.session_state.get("_sector_regime_digest", "")
            st.session_state["_discovery_ai_context"] = {
                "module": "Narrative Discovery — Ticker & Theme Scanner",
                "generated_at": _ai_now2,
                "regime_quadrant": _rc_f2.get("quadrant", "?"),
                "trending_narratives": [n.get("name","?") for n in _narratives2[:6]] or ["No data"],
                "sector_regime_digest": (_sector_dig2[:400] if _sector_dig2 else "No sector data"),
                "dominant_rate_path": st.session_state.get("_dominant_rate_path", "?"),
            }
        except Exception:
            pass

        # Rerun so the pill at the top of render() reads the newly written context
        st.rerun()

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
            "👑 Highly Regarded Mode": (True, "claude-haiku-4-5-20251001"),
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
        _dbt_sigs = st.session_state.get("_qir_debate_signals_text") or _build_dbt_sa()
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
            # ── Zone map + retrain tip ────────────────────────────────────
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                f'<div style="font-size:9px;color:#475569;font-weight:700;'
                f'letter-spacing:0.1em;margin-bottom:6px;">CI% ZONE MAP (anchor {_ci_anchor():.3f})</div>'
                f'<table style="border-collapse:collapse;width:100%;">'
                f'<tr>'
                f'<td style="padding:4px 8px;font-size:10px;color:#22c55e;font-weight:700;">Zone 1</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Normal</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &lt; 22%</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &gt; -0.10</td>'
                f'<td style="padding:4px 8px;font-size:9px;color:#475569;">~3% crash prob &middot; signals suppressed</td>'
                f'</tr>'
                f'<tr>'
                f'<td style="padding:4px 8px;font-size:10px;color:#f59e0b;font-weight:700;">Zone 2</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Model Stress</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI 22-40%</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z -0.10 to -0.30</td>'
                f'<td style="padding:4px 8px;font-size:9px;color:#475569;">~6% crash prob &middot; signals as context</td>'
                f'</tr>'
                f'<tr>'
                f'<td style="padding:4px 8px;font-size:10px;color:#ef4444;font-weight:700;">Zone 3</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Crisis Gate</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &ge; 40%</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &lt; -0.30</td>'
                f'<td style="padding:4px 8px;font-size:9px;color:#475569;">75% historical detection · 0% false alarms</td>'
                f'</tr>'
                f'<tr>'
                f'<td style="padding:4px 8px;font-size:10px;color:#a855f7;font-weight:700;">Zone 4</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Beyond Training</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &gt; 100%</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &lt; -{_ci_anchor():.3f}</td>'
                f'<td style="padding:4px 8px;font-size:9px;color:#475569;">post-training extremes (COVID = 100%)</td>'
                f'</tr>'
                f'</table>'
                f'<div style="font-size:9px;color:#334155;margin-top:6px;">'
                f'Recalibrated to brain.ci_anchor: Volmageddon = 34% &middot; Fed Panic = 41% &middot; '
                f'Tariff 2025 = 50% &middot; COVID = 100%</div>'
                f'</div>'
                f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:5px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                f'<div style="font-size:9px;color:#3b82f6;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:4px;">AFTER RETRAINING</div>'
                f'<div style="font-size:9px;color:#64748b;line-height:1.7;">'
                f'Nothing required — <span style="color:#94a3b8;font-weight:600;">brain.ci_anchor</span> '
                f'auto-calibrates at training time (currently {_ci_anchor():.3f}). All consumers read '
                f'it dynamically via <span style="font-family:monospace;color:#94a3b8;">'
                f'services.hmm_regime.get_ci_anchor()</span>.<br>'
                f'Optional diagnostic (read-only, makes no edits): '
                f'<span style="font-family:monospace;color:#94a3b8;">python ll_gate_backtest_live_brain.py</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        except ImportError:
            st.error("hmmlearn not installed. Run: pip install hmmlearn")
        except Exception as _hmm_e:
            st.error(f"HMM section error: {_hmm_e}")

    # ── Shadow Brain (SPX price) — maintenance section ────────────────────────
    with st.expander("🧠 Shadow Brain (SPX price)", expanded=False):
        try:
            from services.hmm_shadow import (
                load_shadow_brain as _load_sh_brain,
                train_shadow_hmm as _train_sh,
                score_current_shadow_state as _score_sh,
            )
            from services.hmm_regime import get_state_color as _sh_col
            _shb = _load_sh_brain()
            if _shb:
                _shb_trained = _shb.trained_at[:10]
                _shb_bic_txt = f"{_shb.ci_anchor:.3f}"
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:10px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:9px;color:#475569;font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">CURRENT BRAIN</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">STATES</div>'
                    f'<div style="font-size:18px;font-weight:900;color:#94a3b8;">{_shb.n_states}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">CI ANCHOR</div>'
                    f'<div style="font-size:18px;font-weight:900;color:#94a3b8;">{_shb_bic_txt}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">TRAINED</div>'
                    f'<div style="font-size:11px;font-weight:700;color:#94a3b8;margin-top:4px;">{_shb_trained}</div></div>'
                    f'<div><div style="font-size:8px;color:#64748b;font-weight:700;letter-spacing:0.08em;">WINDOW</div>'
                    f'<div style="font-size:9px;font-weight:700;color:#64748b;margin-top:4px;">{_shb.training_start}<br>→ {_shb.training_end}</div></div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#475569;">State labels: '
                    f'{" | ".join(_shb.state_labels)}</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:2px;">'
                    f'Features: ^GSPC daily log returns (single-series · {_shb.n_obs or "?"} obs)</div>'
                    f'<div style="font-size:9px;color:#334155;margin-top:2px;">'
                    f'Model: MarkovRegression · switching_variance=True · CI anchor {_shb.ci_anchor:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Transition matrix
                _shtm = _shb.transmat
                _shn  = _shb.n_states
                _sh_rows_html = ""
                for i in range(_shn):
                    def _sh_fmt_tm(v):
                        pct = v * 100
                        if pct >= 1:
                            return f"{pct:.1f}%"
                        elif pct > 0:
                            return "<1%"
                        return "0%"
                    _sh_row_cells = "".join(
                        f'<td style="padding:3px 8px;font-size:10px;color:#94a3b8;'
                        f'background:rgba(255,255,255,{float(_shtm[i][j])*0.15:.2f});'
                        f'text-align:center;">{_sh_fmt_tm(_shtm[i][j])}</td>'
                        for j in range(_shn)
                    )
                    _shs_col = _sh_col(_shb.state_labels[i])
                    _sh_rows_html += (
                        f'<tr><td style="padding:3px 8px;font-size:9px;color:{_shs_col};'
                        f'font-weight:700;white-space:nowrap;">{_shb.state_labels[i]}</td>'
                        f'{_sh_row_cells}</tr>'
                    )
                _sh_col_headers = "".join(
                    f'<th style="padding:3px 8px;font-size:8px;color:#475569;'
                    f'font-weight:700;text-align:center;">{lbl[:4]}</th>'
                    for lbl in _shb.state_labels
                )
                st.markdown(
                    f'<div style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:4px;">TRANSITION MATRIX</div>'
                    f'<table style="border-collapse:collapse;background:#0a0a14;'
                    f'border-radius:4px;overflow:hidden;">'
                    f'<thead><tr><th style="padding:3px 8px;"></th>{_sh_col_headers}</tr></thead>'
                    f'<tbody>{_sh_rows_html}</tbody></table>'
                    f'<div style="font-size:9px;color:#334155;margin-top:4px;">'
                    f'Row = current state → columns = next-day probability</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="color:#ef444466;font-size:12px;padding:8px 0;">'
                    'Shadow brain not trained yet. Click Retrain to build the SPX model.</div>',
                    unsafe_allow_html=True,
                )

            # ── Zone map tips ─────────────────────────────────────────────
            if _shb:
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:5px;'
                    f'padding:10px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:9px;color:#475569;font-weight:700;'
                    f'letter-spacing:0.1em;margin-bottom:6px;">ZONE MAP (anchor {_shb.ci_anchor:.3f})</div>'
                    f'<table style="border-collapse:collapse;width:100%;">'
                    f'<tr>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#22c55e;font-weight:700;">Zone 1</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Normal</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &lt; 22%</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &gt; -0.26</td>'
                    f'<td style="padding:4px 8px;font-size:9px;color:#475569;">conviction signals suppressed</td>'
                    f'</tr>'
                    f'<tr>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#f59e0b;font-weight:700;">Zone 2</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Stress</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI 22-40%</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z -0.26 to -0.80</td>'
                    f'<td style="padding:4px 8px;font-size:9px;color:#475569;">signals shown as context</td>'
                    f'</tr>'
                    f'<tr>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#ef4444;font-weight:700;">Zone 3</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Crisis</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &ge; 40%</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &lt; -0.80</td>'
                    f'<td style="padding:4px 8px;font-size:9px;color:#475569;">95% hit rate &middot; 16% days flagged</td>'
                    f'</tr>'
                    f'<tr>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#a855f7;font-weight:700;">Zone 4</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;">Beyond</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">CI &gt; 100%</td>'
                    f'<td style="padding:4px 8px;font-size:10px;color:#64748b;">z &lt; -1.19</td>'
                    f'<td style="padding:4px 8px;font-size:9px;color:#475569;">beyond training range</td>'
                    f'</tr>'
                    f'</table>'
                    f'<div style="margin-top:8px;padding:6px 10px;background:#0a0a14;'
                    f'border-radius:4px;border:1px solid #1e293b;">'
                    f'<div style="font-size:9px;color:#f59e0b;font-weight:700;margin-bottom:3px;">FALSE ALARM NOTE</div>'
                    f'<div style="font-size:9px;color:#64748b;line-height:1.6;">'
                    f'89% of Zone 3 calls are false alarms when used alone. '
                    f'The shadow brain is a <span style="color:#94a3b8;font-weight:600;">confirmation signal</span>, '
                    f'not standalone. Require primary brain (credit/yield) agreement before acting on crisis calls.</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Retrain due check ────────────────────────────────────────
            _sh_retrain_due = False
            _sh_days_since  = None
            if _shb:
                try:
                    from datetime import datetime, timezone as _tz
                    _sh_dt = datetime.fromisoformat(_shb.trained_at.replace("Z", "+00:00"))
                    _sh_days_since = (datetime.now(_tz.utc) - _sh_dt).days
                    _sh_retrain_due = _sh_days_since >= 90
                except Exception:
                    pass

            if _sh_retrain_due:
                st.markdown(
                    f'<div style="background:#1a0a00;border:1px solid #f59e0b;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:8px;display:flex;align-items:center;gap:10px;">'
                    f'<span style="font-size:18px;">⏰</span>'
                    f'<div>'
                    f'<div style="font-size:11px;font-weight:700;color:#f59e0b;">Quarterly retrain due</div>'
                    f'<div style="font-size:10px;color:#92400e;">'
                    f'Last trained {_sh_days_since} days ago — click Retrain Shadow to refresh</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            _sh_retrain_label = "🔄 Retrain Shadow" if not _sh_retrain_due else "🔄 Retrain Shadow ⚠️"
            _sh_retrain_help = (
                f"Quarterly retrain due — {_sh_days_since}d since last train"
                if _sh_retrain_due else
                "Fit MarkovRegression on ^GSPC 1960→now (~2–5 min), then recalibrate CI anchor + crash bins."
            )

            _shc1, _shc2 = st.columns([1, 2])
            with _shc1:
                if st.button(_sh_retrain_label, key="btn_shadow_retrain", use_container_width=True,
                             help=_sh_retrain_help, type="primary" if _sh_retrain_due else "secondary"):
                    try:
                        with st.spinner("Fitting MarkovRegression on ^GSPC 1960→now..."):
                            _sh_new_brain = _train_sh()
                        with st.spinner("Calibrating CI anchor + crash bins on full history..."):
                            import subprocess, sys as _sys
                            _cp = subprocess.run(
                                [_sys.executable, "tools/backtest_shadow_ci.py"],
                                capture_output=True, text=True,
                            )
                            if _cp.returncode != 0 and "calibration written" not in (_cp.stdout or ""):
                                st.warning(f"Calibration exit code {_cp.returncode} — check console. stderr: {_cp.stderr[:400]}")
                        with st.spinner("Scoring today's state..."):
                            _sh_new_state = _score_sh(log_to_history=True)
                        _sh_lbl = _sh_new_state.state_label if _sh_new_state else "unknown"
                        st.toast(
                            f"✅ Shadow retrained — {_sh_new_brain.n_states} regimes "
                            f"· CI anchor {_sh_new_brain.ci_anchor:.3f} · Today: {_sh_lbl}",
                            icon="🧠",
                        )
                        st.rerun()
                    except Exception as _se:
                        st.error(f"Shadow retrain failed: {_se}")
            with _shc2:
                if st.button("📍 Score Today (Shadow)", key="btn_shadow_score", use_container_width=True,
                             help="Extend the fitted model with today's return and log to history (no refit)."):
                    if _shb is None:
                        st.warning("Train the Shadow brain first (Retrain Shadow).")
                    else:
                        with st.spinner("Scoring today's Shadow state..."):
                            _sh_ts = _score_sh(log_to_history=True)
                            if _sh_ts:
                                st.toast(
                                    f"📍 {_sh_ts.state_label} · {int(_sh_ts.confidence*100)}% confidence "
                                    f"· {_sh_ts.persistence}d persistence",
                                    icon="🧠",
                                )
                                st.rerun()
                            else:
                                st.error("Shadow scoring failed — check console for details.")
            # ── Retrain tip ───────────────────────────────────────────────
            st.markdown(
                f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:5px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                f'<div style="font-size:9px;color:#3b82f6;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:4px;">AFTER RETRAINING</div>'
                f'<div style="font-size:9px;color:#64748b;line-height:1.7;">'
                f'Re-run the CI anchor sweep to verify the gate threshold:<br>'
                f'<span style="font-family:monospace;color:#94a3b8;font-size:9px;">'
                f'python tools/sweep_shadow_anchor.py</span><br>'
                f'Then update the <span style="color:#94a3b8;font-weight:600;">1.194</span> anchor in '
                f'services/hmm_shadow.py + data/hmm_shadow_brain.json if the optimal z-gate changed.<br>'
                f'Also re-run calibration: '
                f'<span style="font-family:monospace;color:#94a3b8;font-size:9px;">'
                f'python tools/backtest_shadow_ci.py</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        except ImportError:
            st.error("statsmodels not installed. Run: pip install statsmodels>=0.14.0")
        except Exception as _shm_e:
            st.error(f"Shadow section error: {_shm_e}")

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
