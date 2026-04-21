import json
import os
import requests
import streamlit as st

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
XAI_API_URL  = "https://api.x.ai/v1/chat/completions"
XAI_MODEL_REGARD = "grok-4-1-fast-reasoning"


def _is_xai_model(model: str | None) -> bool:
    """True when the model should be routed to xAI (api.x.ai)."""
    return bool(model and model.startswith("grok-"))


def _fmt_tactical_ctx(ctx: dict | None) -> str:
    """Format a _tactical_context dict as a one-line prompt injection string."""
    if not ctx:
        return ""
    score = ctx.get("tactical_score", "?")
    label = ctx.get("label", "?")
    bias  = ctx.get("action_bias", "")
    return f"Tactical Score: {score}/100 ({label}) — {bias}"


def _call_xai(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str | None = None,
    json_mode: bool = False,
) -> str:
    """Call xAI API (OpenAI-compatible). Raises on error."""
    key = os.getenv("XAI_API_KEY", "")
    if not key:
        raise ValueError("XAI_API_KEY not set")
    _is_reasoning = "reasoning" in model
    body: dict = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        "max_tokens": max_tokens,
    }
    if not _is_reasoning:
        body["temperature"] = temperature
    if json_mode and not _is_reasoning:
        body["response_format"] = {"type": "json_object"}
    # Reasoning models think internally — they need much more time than 30s
    _timeout = 120 if _is_reasoning else 45
    resp = requests.post(
        XAI_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=_timeout,
    )
    if not resp.ok:
        raise ValueError(f"xAI {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


@st.cache_data(ttl=3600)
def classify_narrative(topic: str) -> dict:
    """Classify a trending topic for market relevance via Groq LLM.

    Returns dict with keys: market_relevant, sector, thesis, suggested_tickers
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY environment variable not set.")
        return _empty_result()

    prompt = f"""Analyze this trending topic for stock market relevance.

Topic: "{topic}"

Return ONLY valid JSON (no markdown fences) with these keys:
- "market_relevant": boolean — true ONLY if this topic is directly about finance, economics, a public company, an industry/sector, trade policy, regulation, commodities, or monetary policy. Set false for celebrity news, sports, entertainment, social media trends, and pop culture even if a company is tangentially mentioned
- "sector": string — primary market sector affected (e.g. "Technology", "Healthcare", "Energy", "Consumer", "Finance", "Industrials", "N/A")
- "thesis": string — one sentence investment thesis explaining the market impact
- "suggested_tickers": list of strings — 2-5 US stock tickers most exposed to this theme

If not market relevant, set sector to "N/A", thesis to "", and suggested_tickers to []."""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return _empty_result()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _empty_result()


@st.cache_data(ttl=3600)
def describe_company(name: str, ticker: str, sic_description: str) -> dict:
    """Generate a brief company description and investment narrative via Groq.

    Returns dict with keys: description, narrative, sector
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"description": "", "narrative": "", "sector": sic_description}

    prompt = f"""You are a financial analyst. Given this company info, provide a brief overview.

Company: {name}
Ticker: {ticker}
Industry: {sic_description}

Return ONLY valid JSON (no markdown fences) with these keys:
- "description": string — 2-3 sentence description of what the company does, its market position, and key products/services
- "narrative": string — 1-2 sentence current investment narrative or theme (e.g. AI play, dividend aristocrat, turnaround story, growth compounder)
- "sector": string — primary sector (Technology, Healthcare, Energy, Consumer, Finance, Industrials, etc.)"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"description": "", "narrative": "", "sector": sic_description}

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"description": "", "narrative": "", "sector": sic_description}


@st.cache_data(ttl=86400)
def summarize_filing(filing_text: str, form_type: str, company: str, use_claude: bool = False, model: str = None) -> str:
    """Summarize a SEC filing's text content via Groq or Claude.

    Returns a markdown-formatted summary string.
    """
    prompt = f"""You are a financial analyst. Summarize this SEC {form_type} filing from {company}.

Focus on:
- Key announcements or material events (for 8-K)
- Financial highlights and performance (for 10-K/10-Q)
- Risk factors or notable disclosures
- Any forward-looking guidance

Keep the summary to 4-6 bullet points. Be specific with numbers and dates where available.

Filing text:
{filing_text}"""

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 600, 0.1)
        except Exception as e:
            return f"Error generating summary: {e}"
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=600,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate summary."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating summary: {e}"


@st.cache_data(ttl=86400)
def analyze_mda_sentiment(
    mda_text: str,
    company: str,
    use_claude: bool = False,
    model: str | None = None,
) -> dict:
    """Score management tone in a 10-K MD&A section.

    Returns dict with:
        tone            — "confident" | "cautious" | "defensive" | "neutral"
        tone_score      — 0-100 (100 = very confident, 0 = very defensive)
        forward_outlook — "positive" | "mixed" | "negative"
        bullish_phrases — list of up to 3 confident/bullish language samples
        bearish_phrases — list of up to 3 cautious/defensive language samples
        summary         — 2-sentence narrative of the overall management tone
    """
    import json as _json
    import re as _re

    if not mda_text or len(mda_text) < 200:
        return {"tone": "neutral", "tone_score": 50, "forward_outlook": "mixed",
                "bullish_phrases": [], "bearish_phrases": [], "summary": "Insufficient MD&A text."}

    prompt = f"""You are a financial NLP analyst. Analyze the management tone in this MD&A section from {company}'s 10-K filing.

CRITICAL: bullish_phrases and bearish_phrases must be EXACT verbatim quotes copied from the text below. Do not paraphrase or invent. If you cannot find a suitable quote, use an empty list.

Score the tone and return ONLY valid JSON (no markdown, no explanation):

{{
  "tone": "<confident|cautious|defensive|neutral>",
  "tone_score": <integer 0-100, 100=very confident, 0=very defensive/alarming>,
  "forward_outlook": "<positive|mixed|negative>",
  "bullish_phrases": ["<exact verbatim quote from text>", "<exact verbatim quote from text>"],
  "bearish_phrases": ["<exact verbatim quote from text>", "<exact verbatim quote from text>"],
  "summary": "<2 sentences summarizing the overall management tone and key tone signals>"
}}

MD&A text:
{mda_text[:8000]}
[Note: text may be truncated]"""

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."
    _cl_model = model or "grok-4-1-fast-reasoning"
    _fallback = {"tone": "neutral", "tone_score": 50, "forward_outlook": "mixed",
                 "bullish_phrases": [], "bearish_phrases": [], "summary": "Analysis unavailable."}

    raw = None
    try:
        if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai([{"role": "user", "content": prompt}], _cl_model, 600, 0.1,
                            system=_system, json_mode=True)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model or "claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
        else:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return _fallback
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600, "temperature": 0.1},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return _fallback

    if not raw:
        return _fallback

    # Strip markdown fences if present
    raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
    raw = _re.sub(r"\s*```$", "", raw, flags=_re.MULTILINE).strip()
    try:
        result = _json.loads(raw)
        # Ensure all expected keys exist
        result.setdefault("tone", "neutral")
        result.setdefault("tone_score", 50)
        result.setdefault("forward_outlook", "mixed")
        result.setdefault("bullish_phrases", [])
        result.setdefault("bearish_phrases", [])
        result.setdefault("summary", "")
        return result
    except Exception:
        return _fallback


def group_tickers_by_narrative(
    tickers_json: str,
    regime_context: str = "",
    use_claude: bool = False,
    model: str | None = None,
) -> list[dict]:
    """Group a list of trending tickers into narrative themes.

    Input: JSON string of [{symbol, name}, ...]
    Returns list of dicts: [{narrative, description, tickers, conviction, regime_alignment, rationale}, ...]
    """
    result = _group_tickers_cached(tickers_json, regime_context, use_claude, model or "")
    if not result:
        _group_tickers_cached.clear()
    return result


@st.cache_data(ttl=3600)
def _group_tickers_cached(
    tickers_json: str,
    regime_context: str = "",
    use_claude: bool = False,
    model: str = "",
) -> list[dict]:
    _regime_block = (
        f"\nCURRENT MACRO REGIME:\n{regime_context}\n"
        if regime_context else ""
    )

    prompt = f"""You are a financial analyst. Given these trending tickers, group them into 3-6 investment narrative themes.{_regime_block}
The list may include stocks, commodity futures (tickers ending in =F), bond ETFs, and currency ETFs. Create cross-asset themes where appropriate (e.g. "Inflation Hedge" grouping gold futures with TIPS ETFs, or "Risk-Off Flight" grouping treasuries with yen).

Tickers:
{tickers_json}

For each narrative group, pick a short punchy title (e.g. "AI Infrastructure Boom", "Rate Cut Beneficiaries", "Energy Transition", "Commodity Super-Cycle", "Safe Haven Rotation").

{"Also assess each group against the current macro regime: conviction level and whether it aligns with or contradicts the regime." if regime_context else ""}

Return ONLY valid JSON (no markdown fences) — a list of objects:
[
  {{
    "narrative": "Theme Title",
    "description": "One sentence explaining why these assets are grouped together",
    "tickers": ["SYM1", "SYM2"],
    "conviction": "HIGH" | "MEDIUM" | "LOW",
    "regime_alignment": "aligned" | "contrarian" | "neutral",
    "rationale": "One sentence on why this theme is or isn't consistent with current macro conditions"
  }}
]

Rules:
- Every ticker must appear in exactly one group
- If a ticker doesn't fit a clear theme, put it in a "Market Movers" catch-all group
- Sort groups so the strongest/most actionable narrative is first
- conviction and regime_alignment are required fields (use "MEDIUM" and "neutral" if macro context is unavailable)"""

    text = ""

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            text = _call_xai([{"role": "user", "content": prompt}], _cl_model, 1500, 0.3)
        except Exception as e:
            st.warning(f"xAI timed out — falling back to Groq ({type(e).__name__})")
            # text stays "" → falls through to Groq below
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
        except Exception as e:
            st.warning(f"Narrative grouping failed (Claude error): {e}")
            return []

    # Groq fallback: used in Freeloader Mode OR when xAI/Claude failed
    if not text:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.warning("GROQ_API_KEY not set — cannot group tickers by narrative.")
            return []
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.3,
                },
                timeout=20,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.warning(f"Narrative grouping failed (API error): {e}")
            return []

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.warning("Narrative grouping failed (malformed AI response). Showing flat list.")
        return []


def generate_tactical_analysis(
    signals: list[dict],
    tactical_score: int,
    label: str,
    macro_label: str = "",
    use_claude: bool = False,
    model: str | None = None,
) -> str:
    """Generate an AI narrative interpretation of the Tactical Regime signals.

    Returns a plain-text 2-3 paragraph analysis. Falls back to a template
    string on API failure so the UI always has something to display.
    """
    from datetime import date as _date
    _today = _date.today().isoformat()

    sig_lines = "\n".join(
        f"- {s['Signal']}: {s['Value']}  (score {s['Score']:+.3f} — {s['Direction']})"
        for s in signals
    )
    prompt = f"""You are a short-term market tactician. Today is {_today}.

TACTICAL REGIME: {label} (score {tactical_score}/100)
MACRO BACKDROP: {macro_label if macro_label else "Not specified"}

TACTICAL SIGNALS:
{sig_lines}

Write a concise 2-3 paragraph tactical analysis. Cover:
1. What the combined signal picture says about short-term risk appetite and entry conditions
2. Which specific signals are most dominant right now and why they matter
3. A concrete action implication: what a trader should do with this setup (add, hold, reduce, or hedge)

Be specific, clinical, and use the actual signal values. No vague generalities."""

    _cl_model = model or "grok-4-1-fast-reasoning"

    try:
        if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 600, 0.4)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model, max_tokens=600, temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        else:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return f"Tactical Regime: {label} ({tactical_score}/100). Add a GROQ_API_KEY to enable AI narrative."
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600, "temperature": 0.4},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Tactical Regime: {label} ({tactical_score}/100). AI analysis unavailable: {e}"


def generate_valuation(ticker: str, signals_text: str, use_claude: bool = False, model: str = None, current_events: str = "", tactical_context: str = "") -> dict | None:
    """Generate an AI valuation and recommendation for a ticker via Groq or Claude.

    Returns dict with keys: rating, confidence, time_horizon, summary,
    conviction_drivers, bullish_factors, bearish_factors, signal_conflicts,
    scenarios, key_levels, catalysts, recommendation

    Uses signal fingerprinting to return a cached verdict when underlying signals
    haven't changed, avoiding redundant AI calls.
    Groq (free tier) uses a lean classification prompt; Grok/Claude get the full prompt.
    """
    import re
    from utils.signal_block import build_macro_block, build_ticker_block, get_ticker_fingerprint

    # ── Fingerprint cache check ────────────────────────────────────────────────
    try:
        _fp = get_ticker_fingerprint(ticker)
        _vcache = st.session_state.get("_valuation_cache") or {}
        if _fp in _vcache:
            return _vcache[_fp]
    except Exception:
        _fp = None

    from datetime import date as _date
    _today = _date.today().isoformat()
    _ce_block = (
        f"\nCURRENT EVENTS & MARKET INTEL:\n{current_events[:800]}\n"
        if current_events and len(current_events.strip()) > 20
        else "\nCURRENT EVENTS: None provided — base analysis on signals only.\n"
    )

    _tac_block = (f"\nTACTICAL REGIME (days-to-weeks entry timing): {tactical_context}\n"
                  if tactical_context else "")

    # Ground-truth numeric blocks — injected first so AI sees raw numbers before narratives
    try:
        _macro_blk = build_macro_block()
    except Exception:
        _macro_blk = ""
    try:
        _ticker_blk = build_ticker_block(ticker)
    except Exception:
        _ticker_blk = ""

    _grounding = ""
    if _macro_blk:
        _grounding += f"\n{_macro_blk}\n"
    if _ticker_blk:
        _grounding += f"\n{_ticker_blk}\n"

    # Full prompt — used by Grok/Claude (reasoning models, full narration)
    prompt = f"""You are an expert equity research analyst. Analysis date: {_today}. Based on the following data snapshot for {ticker}, provide a comprehensive, multi-dimensional valuation and recommendation. You have access to cross-module signals including insider trades, 13F institutional flow, options activity, macro regime, sector rotation, and analyst revisions — use ALL of them to inform your analysis.
{_grounding}
DETAILED SIGNALS (narrative context — raw numbers above take precedence):
{signals_text}{_ce_block}{_tac_block}

Think carefully before responding. Identify the 2-3 signals most responsible for your rating. Flag any contradictions between signals. Build three concrete scenarios.

Return ONLY valid JSON (no markdown fences, no extra text) with EXACTLY these keys:
{{
  "rating": "Strong Buy" or "Buy" or "Hold" or "Sell" or "Strong Sell",
  "confidence": 0-100,
  "time_horizon": "short: <bearish|neutral|bullish> (1-4 wks), medium: <bearish|neutral|bullish> (1-3 mo), long: <bearish|neutral|bullish> (3-12 mo)",
  "summary": "2-3 sentence thesis integrating the most important cross-signal findings",
  "conviction_drivers": ["top signal driving rating", "second signal", "third signal"],
  "bullish_factors": ["specific bullish factor 1", "factor 2", "factor 3", "factor 4"],
  "bearish_factors": ["specific bearish factor 1", "factor 2", "factor 3"],
  "signal_conflicts": ["e.g. Options flow bullish but insiders net sold $3M", "second conflict if any"],
  "scenarios": {{
    "bull": {{"thesis": "1 sentence bull case", "target": 0.00, "probability": 35}},
    "base": {{"thesis": "1 sentence base case", "target": 0.00, "probability": 45}},
    "bear": {{"thesis": "1 sentence bear case", "target": 0.00, "probability": 20}}
  }},
  "key_levels": {{"support": 0.00, "resistance": 0.00, "stop_loss": 0.00, "target_1": 0.00, "target_2": 0.00}},
  "catalysts": [
    {{"event": "event name", "date": "YYYY-MM-DD or 'TBD'", "impact": "high|medium|low", "direction": "bullish|bearish|neutral"}}
  ],
  "recommendation": "2-3 sentence specific action guidance with entry/exit conditions"
}}

Rules:
- bull+base+bear probabilities must sum to 100
- conviction_drivers must reference actual signal values from the data (e.g. "RSI 68 + above all 3 SMAs")
- signal_conflicts: list [] if none exist
- catalysts: include earnings, FOMC, sector events — use exact dates from the data where available
- All price targets must be realistic dollar amounts based on the current price in the data"""

    # Lean classification prompt — used by Groq free tier (LLaMA)
    # ~half the tokens: drops verbose instructions, caps signal context at 1500 chars.
    # NOTE: signals are truncated — if signal context is >1500 chars, later signals may be cut.
    # The macro ground truth block (_grounding) is always included in full.
    _signals_truncated = len(signals_text) > 1500
    _groq_prompt = f"""Rate {ticker} as of {_today}. JSON only.
{_grounding}
SIGNALS{' (truncated — first 1500 chars)' if _signals_truncated else ''}: {signals_text[:1500]}{_ce_block}
Return ONLY this JSON (no fences):
{{
  "rating": "Strong Buy|Buy|Hold|Sell|Strong Sell",
  "confidence": 0-100,
  "time_horizon": "short: <bearish|neutral|bullish> (1-4 wks), medium: <bearish|neutral|bullish> (1-3 mo), long: <bearish|neutral|bullish> (3-12 mo)",
  "summary": "1-2 sentences max",
  "conviction_drivers": ["signal1", "signal2", "signal3"],
  "bullish_factors": ["factor1", "factor2", "factor3"],
  "bearish_factors": ["factor1", "factor2"],
  "signal_conflicts": [],
  "scenarios": {{
    "bull": {{"thesis": "1 sentence", "target": 0.00, "probability": 35}},
    "base": {{"thesis": "1 sentence", "target": 0.00, "probability": 45}},
    "bear": {{"thesis": "1 sentence", "target": 0.00, "probability": 20}}
  }},
  "key_levels": {{"support": 0.00, "resistance": 0.00, "stop_loss": 0.00, "target_1": 0.00, "target_2": 0.00}},
  "catalysts": [{{"event": "earnings", "date": "TBD", "impact": "high", "direction": "neutral"}}],
  "recommendation": "1-2 sentence action guidance"
}}
probabilities must sum to 100. Use realistic price targets."""

    def _parse_valuation_json(text: str) -> dict | None:
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                text = text[start:]
        if not text.endswith("}"):
            end = text.rfind("}")
            if end != -1:
                text = text[:end + 1]
        try:
            result = json.loads(text)
            return result if "rating" in result else None
        except json.JSONDecodeError:
            return None

    def _cache_and_return(result: dict | None) -> dict | None:
        """Write result to fingerprint cache before returning."""
        if result and _fp:
            _vc = st.session_state.get("_valuation_cache") or {}
            _vc[_fp] = result
            # Keep cache bounded — evict oldest entries beyond 20 tickers
            if len(_vc) > 20:
                oldest = next(iter(_vc))
                del _vc[oldest]
            st.session_state["_valuation_cache"] = _vc
        return result

    _cl_model = model or "grok-4-1-fast-reasoning"
    _val_system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _cache_and_return(_parse_valuation_json(_call_xai(
                [{"role": "user", "content": prompt}], _cl_model, 2000, 0.1, system=_val_system)))
        except Exception as e:
            st.error(f"xAI API error: {e}")
            return None
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=2000,
                temperature=0.1,
                system=_val_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _cache_and_return(_parse_valuation_json(message.content[0].text.strip()))
        except Exception as e:
            st.error(f"Claude API error: {e}")
            return None

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY environment variable not set.")
        return None

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": _val_system},
                    {"role": "user", "content": _groq_prompt},  # lean prompt for Groq
                ],
                "max_tokens": 1200,   # down from 2000 — lean prompt needs less
                "temperature": 0.1,
            },
            timeout=45,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

    return _cache_and_return(_parse_valuation_json(text))


def suggest_regime_plays(regime: str, score: float, signal_summary: str, use_claude: bool = False, model: str = None) -> dict:
    """Suggest sectors, stocks, and bonds based on the current risk regime via Groq or Claude.

    Returns dict with keys: sectors, stocks, bonds, rationale
    Uses signal fingerprinting — returns cached result if macro signals unchanged.
    """
    _empty = {"sectors": [], "stocks": [], "bonds": [], "rationale": ""}

    # ── Fingerprint cache check ────────────────────────────────────────────────
    try:
        from utils.signal_block import get_signal_fingerprint as _get_fp, build_macro_block as _build_mb
        _fp = _get_fp()
        _rpc = st.session_state.get("_regime_plays_cache") or {}
        if _fp in _rpc:
            return _rpc[_fp]
    except Exception:
        _fp = None
        _build_mb = None

    # Extract quadrant from signal_summary if present
    _quadrant = ""
    if "Dalio Quadrant:" in signal_summary:
        try:
            _quadrant = signal_summary.split("Dalio Quadrant:")[1].split(",")[0].strip()
        except Exception:
            pass

    _quadrant_guidance = {
        "Stagflation": "STAGFLATION regime (falling growth + rising inflation): prioritize real assets (GLD, TIP, XLE, UUP, MCD, PG). Avoid growth tech (QQQ, IWM) and HY bonds.",
        "Deflation": "DEFLATION regime (falling growth + falling inflation): prioritize long-duration bonds (TLT, IEF), gold (GLD), investment-grade bonds (LQD). Avoid energy (XLE) and EM (EEM).",
        "Reflation": "REFLATION regime (rising growth + rising inflation): prioritize commodities (XLE, CPER, GLD), cyclicals, emerging markets (EEM). Avoid long-duration bonds (TLT).",
        "Goldilocks": "GOLDILOCKS regime (rising growth + stable inflation): prioritize equities broadly (QQQ, SPY, IWM), growth sectors, EM. Avoid defensive gold and long bonds.",
    }.get(_quadrant, "")

    from datetime import date as _date
    _today = _date.today().isoformat()

    # Inject raw z-scores block alongside signal summary
    try:
        _raw_blk = _build_mb() if _build_mb else ""
    except Exception:
        _raw_blk = ""

    prompt = f"""You are a macro strategist. As of {_today}, the market risk regime is currently **{regime}** with an aggregate score of {score:+.2f} (scale: -1 risk-off to +1 risk-on).

{_raw_blk}

Key signal summary (AI-generated — raw numbers above take precedence):
{signal_summary}

{_quadrant_guidance}

Based on this regime and signals, suggest what to buy right now.

Return ONLY valid JSON (no markdown fences) with these keys:
- "sectors": list of 3-5 objects, each with "name" (sector name) and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "stocks": list of 4-6 objects, each with "ticker", "reason" (1 sentence), and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "bonds": list of 2-3 objects, each with "ticker", "reason", and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "rationale": 2-3 sentence macro rationale for these picks given the current regime and quadrant

Be selective with 3-star (strong buy) ratings — only give them to picks that are the best fit for this exact regime. Most picks should be 1 or 2 stars."""

    def _parse(text: str) -> dict:
        import re as _re
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            return _empty
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return _empty

    def _parse_and_cache(text: str) -> dict:
        result = _parse(text)
        if _fp and result and result.get("sectors"):
            _rpc2 = st.session_state.get("_regime_plays_cache") or {}
            _rpc2[_fp] = result
            st.session_state["_regime_plays_cache"] = _rpc2
        return result

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _parse_and_cache(_call_xai(
                [{"role": "user", "content": prompt}], _cl_model, 1200, 0.3, system=_system))
        except Exception as _e:
            st.error(f"xAI API error (Regime Plays): {_e}")
            return _empty
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1200,
                temperature=0.3,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_and_cache(message.content[0].text.strip())
        except Exception as _e:
            st.error(f"Claude API error (Regime Plays): {_e}")
            return _empty

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _empty

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": _system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1200,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return _parse_and_cache(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as _e:
        st.error(f"Groq API error (Regime Plays): {_e}")
        return _empty


def suggest_scenario_plays(
    scenario: str,
    regime: str,
    quadrant: str,
    signal_summary: str,
    use_claude: bool = False,
    model: str = None,
) -> dict:
    """Generate sector/stock/bond plays for a user-defined macro scenario.

    Returns dict with keys: sectors, stocks, bonds, rationale, avoid
    """
    _empty = {"sectors": [], "stocks": [], "bonds": [], "rationale": "", "avoid": []}

    _quadrant_hints = {
        "Stagflation": "Stagflation = falling growth + rising inflation. Prioritize: GLD, TIP, XLE, UUP, MCD, PG. Avoid: QQQ, IWM, HYG, growth tech.",
        "Deflation": "Deflation = falling growth + falling inflation. Prioritize: TLT, IEF, GLD, LQD, defensive equities. Avoid: XLE, HYG, EEM.",
        "Reflation": "Reflation = rising growth + rising inflation. Prioritize: XLE, CPER, EEM, cyclicals, commodities. Avoid: TLT, long-duration bonds.",
        "Goldilocks": "Goldilocks = rising growth + stable inflation. Prioritize: QQQ, SPY, IWM, EEM, growth tech. Avoid: GLD, TLT.",
    }
    _hint = _quadrant_hints.get(quadrant, "")

    try:
        from utils.signal_block import build_macro_block as _build_mb
        _raw_blk = _build_mb()
    except Exception:
        _raw_blk = ""
    _raw_header = f"RAW NUMERIC GROUND TRUTH (take precedence over all narrative):\n{_raw_blk}\n\n" if _raw_blk else ""

    prompt = f"""You are a macro strategist. A specific market scenario is unfolding:

{_raw_header}Scenario: {scenario}

Current macro backdrop:
- Regime: {regime}
- Dalio Quadrant: {quadrant}
- {signal_summary}

Quadrant context: {_hint}

Given this specific scenario occurring on top of the current macro backdrop, what should an investor do RIGHT NOW?

Return ONLY valid JSON (no markdown fences) with these keys:
- "sectors": list of 3-5 objects, each with "name" and "conviction" (integer 1-3, where 3 = strong buy)
- "stocks": list of 4-6 objects, each with "ticker", "reason" (1 sentence), and "conviction" (1-3)
- "bonds": list of 2-3 objects, each with "ticker", "reason", and "conviction" (1-3)
- "rationale": 2-3 sentence explanation of why this scenario drives these specific plays
- "avoid": list of 2-4 ticker strings to explicitly exit or avoid in this scenario

3 stars = immediate strong buy specifically because of this scenario. Include both "buy" and "avoid" lists."""

    def _parse(text: str) -> dict:
        import re as _re
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            return _empty
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return _empty

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _parse(_call_xai(
                [{"role": "user", "content": prompt}], _cl_model, 1200, 0.3, system=_system))
        except Exception as _e:
            st.error(f"xAI API error (Scenario Plays): {_e}")
            return _empty
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1200,
                temperature=0.3,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse(message.content[0].text.strip())
        except Exception as _e:
            st.error(f"Claude API error (Scenario Plays): {_e}")
            return _empty

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _empty

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": _system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1200,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return _parse(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as _e:
        st.error(f"Groq API error (Scenario Plays): {_e}")
        return _empty


def summarize_whale_activity(activity_text: str, use_claude: bool = False, model: str = None) -> str:
    """Summarize whale 13F activity into a narrative via Groq or Claude.

    Returns a markdown-formatted narrative string about big money themes this quarter.
    """
    from datetime import date as _date
    _today = _date.today().isoformat()

    prompt = f"""You are a top-tier institutional equity analyst. As of {_today}, analyze these quarterly 13F whale position changes and recent activism filings, then write a narrative summary.

Focus on:
- What themes or sectors are the biggest hedge funds and institutions converging on?
- Which names are seeing the most conviction (multiple whales buying)?
- Are there notable exits or position closures that signal a shift?
- What does the overall flow pattern suggest about institutional sentiment?
- If SC 13D activism filings are included: name the activist, the target company, and what campaign they likely intend.

Write in clear paragraphs with a blank line between each section. Be specific about names, dollar amounts, and the whales involved.

Whale activity data:
{activity_text}"""

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 1000, 0.3)
        except Exception as _e:
            st.error(f"xAI API error (Whale Summary): {_e}")
            return f"Error generating whale summary: {_e}"
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as _e:
            st.error(f"Claude API error (Whale Summary): {_e}")
            return f"Error generating whale summary: {_e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate whale activity summary."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating whale summary: {e}"

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def summarize_sector_regime(
    sector_data: list,
    regime_context: dict,
    use_claude: bool = False,
    model: str = None,
) -> str:
    """Merge sector rotation momentum with macro regime to produce a tactical digest.

    Args:
        sector_data: List of sector dicts from get_sector_momentum() — each has
                     ticker, name, ret_4w, ret_12w, ret_26w, rank_4w, rank_12w.
        regime_context: _regime_context dict with regime, score, quadrant, signal_summary.
    Returns:
        A single prose paragraph (≤250 words) for injection into downstream prompts.
    """
    import os as _os
    from services.sector_rotation import QUADRANT_ALIGNMENT
    from datetime import date as _date

    _today = _date.today().isoformat()

    quadrant = regime_context.get("quadrant", "Unknown")
    regime   = regime_context.get("regime", "Unknown")
    score    = regime_context.get("score", 0)
    aligned  = set(QUADRANT_ALIGNMENT.get(quadrant, []))

    leaders  = sector_data[:3]
    laggards = sector_data[-3:]

    def _fmt(s):
        r4 = f"{s['ret_4w']:+.1f}%" if s.get("ret_4w") is not None else "N/A"
        r12 = f"{s['ret_12w']:+.1f}%" if s.get("ret_12w") is not None else "N/A"
        flag = " ✓ALIGNED" if s["ticker"] in aligned else ""
        return f"{s['ticker']} ({s['name']}) 4W:{r4} 12W:{r12}{flag}"

    leaders_str  = " | ".join(_fmt(s) for s in leaders)
    laggards_str = " | ".join(_fmt(s) for s in laggards)
    aligned_leading = [s for s in leaders if s["ticker"] in aligned]
    confirm = "CONFIRMED" if len(aligned_leading) >= 2 else ("PARTIAL" if len(aligned_leading) == 1 else "DIVERGING")

    prompt = f"""You are a macro-tactical strategist. As of {_today}, synthesize sector rotation momentum with the macro regime.

MACRO REGIME: {regime} | Score: {score:+.2f} | Dalio Quadrant: {quadrant}
REGIME-ALIGNED SECTORS: {', '.join(aligned) or 'None'}

SECTOR MOMENTUM LEADERS (4W): {leaders_str}
SECTOR MOMENTUM LAGGARDS (4W): {laggards_str}
REGIME CONFIRMATION STATUS: {confirm} ({len(aligned_leading)}/3 leaders are regime-aligned)

Write a single tactical paragraph (max 200 words) covering:
1. Whether sector momentum CONFIRMS or DIVERGES from the macro regime
2. The 1-2 highest-conviction sector plays right now (specific ETF + rationale)
3. Any notable rotation signal (e.g. defensives leading in a Risk-On regime = warning)
4. A one-sentence trading implication

Be direct. No headers. No bullet points. Use specific ETF tickers."""

    text = ""
    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and _os.getenv("XAI_API_KEY"):
        try:
            text = _call_xai([{"role": "user", "content": prompt}], _cl_model, 600, 0.3)
        except Exception as _e:
            return f"Error generating sector regime digest: {_e}"

    elif use_claude and _os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=_os.getenv("ANTHROPIC_API_KEY", ""))
            msg = client.messages.create(
                model=_cl_model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
        except Exception as _e:
            return f"Error generating sector regime digest: {_e}"

    if not text:
        try:
            import requests as _req
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {_os.environ.get('GROQ_API_KEY','')}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0.3},
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating sector regime digest: {e}"

    return text


def summarize_activism_filings(filings_text: str, use_claude: bool = False, model: str = None) -> str:
    """Analyze SC 13D/13G activism filings and generate an investment intelligence summary.

    Returns a markdown-formatted analysis string covering activist intent,
    historical campaign patterns, and likely outcomes.
    """
    from datetime import date as _date
    _today = _date.today().isoformat()

    prompt = f"""You are a top-tier activist investing analyst. As of {_today}, analyze these SC 13D/13G filings and write an actionable intelligence summary.

For each significant 13D filing, identify:
- The activist's likely campaign objective (board seats, asset sale, spinoff, CEO change, buyback, merger opposition)
- The target's vulnerability (undervalued assets, weak governance, depressed valuation, insider entrenchment)
- Historical outcome pattern for this type of campaign
- Estimated probability of activist success and timeline

Then provide a cross-filing synthesis:
- Which sectors are attracting the most activism right now?
- Are there common themes (e.g., energy restructuring, biotech governance, tech buybacks)?
- What does this activism wave signal about broader market conditions?

Be specific, direct, and investment-actionable. Write in clear paragraphs.

SC 13D/13G filings data:
{filings_text}"""

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 1500, 0.3)
        except Exception as _e:
            st.error(f"xAI API error (Activism Analysis): {_e}")
            return f"Error generating activism analysis: {_e}"
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as _e:
            st.error(f"Claude API error (Activism Analysis): {_e}")
            return f"Error generating activism analysis: {_e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate activism analysis."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3,
            },
            timeout=45,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating activism analysis: {e}"

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def generate_doom_briefing(stress_data: str, use_claude: bool = False, model: str = None, current_events: str = "") -> str:
    """Generate an ominous risk intelligence briefing from stress signal data via Groq or Claude.

    Returns a markdown-formatted doom briefing string.
    """
    # Truncate stress_data to avoid Groq context limits (~6k chars is safe)
    from datetime import date as _date
    _today = _date.today().isoformat()
    _stress_truncated = stress_data[:6000] + "\n[...truncated]" if len(stress_data) > 6000 else stress_data
    if current_events and len(current_events.strip()) > 20:
        _stress_truncated += f"\n\nCURRENT EVENTS CONTEXT:\n{current_events[:1500]}"
    else:
        _stress_truncated += "\n\nCURRENT EVENTS CONTEXT: None provided — base analysis on signal data only."

    prompt = f"""You are a senior sell-side risk strategist writing for institutional allocators. Analysis date: {_today}.

Analyze the stress signal data below and produce a concise risk assessment briefing.

Rules:
- Use clinical, data-driven language — no hyperbole, no metaphors, no editorial flair.
- Every claim must cite a specific number from the data provided. Do not assert without evidence.
- Measured urgency is appropriate; dramatic language is not.
- Flag systemic concerns — contagion risks, interconnected failures, cascading defaults — but describe them in terms of observed data.
- Rate overall financial system stress from 1-10 (1 = calm, 10 = systemic crisis).
- Structure as: STRESS LEVEL rating, then 4-6 bullet points on the most significant signals, then a 2-3 sentence forward-looking assessment.
- If data is limited or unavailable, note what cannot be assessed and why.

Stress Signal Data:
{_stress_truncated}"""

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 1000, 0.4)
        except Exception as e:
            return f"Error generating doom briefing: {e}"
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1000,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating doom briefing: {e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate doom briefing."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.4,
            },
            timeout=30,
        )
        if not resp.ok:
            # Auto-fallback to Claude (Haiku) if Groq is restricted/unavailable
            if resp.status_code == 400 and os.getenv("ANTHROPIC_API_KEY"):
                try:
                    import anthropic as _ac
                    _client = _ac.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    _msg = _client.messages.create(
                        model="claude-haiku-4-5",
                        max_tokens=1000,
                        temperature=0.4,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return _msg.content[0].text.strip()
                except Exception as _ce:
                    return f"Error generating doom briefing: Groq restricted, Claude fallback also failed: {_ce}"
            return f"Error generating doom briefing: {resp.status_code} — {resp.text[:300]}"
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating doom briefing: {e}"

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def narrate_policy_transmission(chains_json: str, adj_probs_json: str, use_claude: bool = False, model: str = None) -> str:
    """Narrative interpretation of the Fed policy transmission path.

    chains_json and adj_probs_json are JSON strings (hashable for @st.cache_data).
    Returns 3-4 sentence narrative string.
    """
    from datetime import date as _date
    _today = _date.today().isoformat()
    prompt = (
        f"You are a macro policy analyst. As of {_today}, based on these Fed policy probability scenarios and "
        "causal transmission chains, write exactly 3-4 sentences explaining: "
        "(1) the most likely rate path and its probability, "
        "(2) how that path transmits through credit markets, housing, and employment, "
        "(3) which specific asset classes benefit and which face headwinds — name tickers or sectors. "
        "Be precise and specific. No vague generalities.\n\n"
        f"Rate path probabilities: {adj_probs_json}\n"
        f"Transmission chains: {chains_json}"
    )

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 400, 0.3)
        except Exception as e:
            return f"Error generating narration: {e}"
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model,
                max_tokens=400,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"Error generating narration: {e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set."
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating narration: {e}"


def generate_macro_synopsis(signals_text: str, use_claude: bool = False, model: str = None) -> dict:
    """Synthesize all QIR signals into a macro conviction assessment.

    Returns dict with keys:
      conviction: "BULLISH" | "BEARISH" | "MIXED" | "UNCERTAIN"
      summary: 2-3 sentence synthesis
      key_points: list of 3-4 supporting bullet strings
      contradictions: list of contradiction strings (may be empty)
    """
    try:
        from utils.signal_block import build_macro_block as _build_mb
        _raw_blk = _build_mb()
    except Exception:
        _raw_blk = ""

    _raw_header = f"RAW NUMERIC GROUND TRUTH:\n{_raw_blk}\n\n" if _raw_blk else ""

    prompt = (
        "You are a senior macro strategist synthesizing multiple independent signal sources. "
        "Based on the signals below, assess overall macro conviction.\n\n"
        "Return ONLY valid JSON (no markdown fences) with this exact structure:\n"
        '{"conviction": "BULLISH|BEARISH|MIXED|UNCERTAIN", '
        '"summary": "<2-3 sentence synthesis>", '
        '"key_points": ["<point 1>", "<point 2>", "<point 3>"], '
        '"contradictions": ["<contradiction if any>"]}\n\n'
        "Rules:\n"
        "- conviction = BULLISH if ≥3 signals align bullish, BEARISH if ≥3 align bearish, "
        "MIXED if signals conflict, UNCERTAIN if data is sparse\n"
        "- key_points: cite specific data from the signals (numbers, labels, names)\n"
        "- contradictions: list any signals that contradict the dominant conviction; empty list if none\n"
        "- Be clinical and specific — no vague generalities\n"
        "- Raw numbers above take precedence over narrative summaries\n\n"
        f"{_raw_header}SIGNALS (narrative context):\n{signals_text[:4000]}"
    )

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            import json as _json, re as _re
            _raw = _call_xai([{"role": "user", "content": prompt}], _cl_model, 600, 0.3, json_mode=True)
            _raw = _re.sub(r"^```(?:json)?\s*", "", _raw, flags=_re.MULTILINE)
            _raw = _re.sub(r"\s*```$", "", _raw, flags=_re.MULTILINE).strip()
            return _json.loads(_raw)
        except Exception:
            pass
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic, json as _json
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=_cl_model,
                max_tokens=600,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _json.loads(msg.content[0].text.strip())
        except Exception:
            pass

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"conviction": "UNCERTAIN", "summary": "GROQ_API_KEY not set.", "key_points": [], "contradictions": []}
    try:
        import json as _json
        import re as _re
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        _raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if model wrapped the JSON
        _raw = _re.sub(r"^```(?:json)?\s*", "", _raw, flags=_re.MULTILINE)
        _raw = _re.sub(r"\s*```$", "", _raw, flags=_re.MULTILINE).strip()
        return _json.loads(_raw)
    except Exception as e:
        return {"conviction": "UNCERTAIN", "summary": f"Error: {e}", "key_points": [], "contradictions": []}


def assess_macro_fit(
    ticker: str,
    company_name: str,
    sector: str,
    price_summary: str,
    regime_context: str,
    rate_path_context: str,
    black_swan_context: str,
    tactical_context: str = "",
    use_claude: bool = False,
    model: str | None = None,
) -> dict | None:
    """Evaluate how well a ticker fits the current macro regime environment."""
    _tac_line = f"\n- Tactical Regime (entry timing): {tactical_context}" if tactical_context else ""
    prompt = f"""You are a macro-regime portfolio analyst. Evaluate how well {ticker} ({company_name}, {sector}) fits the CURRENT macro environment.

CURRENT MACRO ENVIRONMENT:
- Regime Context: {regime_context}
- Fed Rate Path: {rate_path_context}
- Black Swan Tail Risks: {black_swan_context if black_swan_context else "None analyzed"}{_tac_line}
- Stock Technical: {price_summary}

Rate the macro fit from 1-5 and explain. Consider:
1. Does this sector historically outperform in this regime/quadrant?
2. Is this stock rate-sensitive? Does the dominant rate path help or hurt?
3. Does this stock have exposure to the analyzed black swan risks?
4. Any regime-specific catalysts or headwinds?

Respond ONLY with valid JSON:
{{
  "fit_stars": <1-5 integer>,
  "verdict": "<Strong Fit|Moderate Fit|Neutral|Caution|Avoid>",
  "rationale": "<2-3 sentences explaining the fit score>",
  "tailwinds": ["<macro tailwind for this ticker>"],
  "headwinds": ["<macro headwind for this ticker>"]
}}"""
    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."
    try:
        if use_claude and model and _is_xai_model(model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai([{"role": "user", "content": prompt}], model, 700, 0.2, system=_system)
        elif use_claude and model:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=model, max_tokens=700, temperature=0.2,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [
                          {"role": "system", "content": _system},
                          {"role": "user", "content": prompt},
                      ],
                      "max_tokens": 600, "temperature": 0.2},
                timeout=30,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
        import json as _json, re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        return _json.loads(m.group()) if m else {"_error": f"No JSON in response: {raw[:200]}"}
    except Exception as _e:
        return {"_error": str(_e)}


def analyze_portfolio(
    positions: list,
    upstream: dict,
    use_claude: bool = True,
    model: str | None = None,
) -> dict | None:
    """Analyze open positions against current macro conditions.

    Returns a dict with verdict, risk_score, narrative, per-position assessments,
    and priority_actions. Defaults to Sonnet (highest-stakes call).
    """
    from utils.signal_block import build_macro_block as _build_mb
    try:
        _macro_blk = _build_mb()
    except Exception:
        _macro_blk = "(macro ground truth unavailable — run QIR first)"

    regime = upstream.get("regime", "Unknown")
    score = upstream.get("score", 0.0)
    quadrant = upstream.get("quadrant", "Unknown")
    fed_funds_rate = upstream.get("fed_funds_rate", "Unknown")
    doom_briefing = (upstream.get("doom_briefing") or "")[:400]
    whale_summary = (upstream.get("whale_summary") or "")[:300]
    chain_narration = (upstream.get("chain_narration") or "")[:300]

    # Rate path
    dominant_rp = upstream.get("dominant_rate_path") or {}
    dominant_scenario = dominant_rp.get("scenario", "Unknown")
    prob_pct = dominant_rp.get("prob_pct", 0)

    # Regime plays sectors
    rp = upstream.get("regime_plays") or {}
    regime_plays_sectors = ", ".join(s.get("name", s) if isinstance(s, dict) else s for s in rp.get("sectors", [])) or "None"

    # Discovery plays
    dp = upstream.get("discovery_plays") or {}
    _dp_sectors = ", ".join(s.get("name", "") for s in dp.get("sectors", [])[:3])
    _dp_stocks = ", ".join(s.get("ticker", "") for s in dp.get("stocks", [])[:4])
    discovery_plays_str = (
        f"Sectors: {_dp_sectors} | Stocks: {_dp_stocks}"
        if (_dp_sectors or _dp_stocks) else "None"
    )

    # Black swan block
    custom_swans = upstream.get("custom_swans") or {}
    if custom_swans:
        swan_lines = []
        for label, data in custom_swans.items():
            prob = data.get("probability_pct", "?")
            impacts = data.get("asset_impacts", {})
            eq = impacts.get("equities", "?")
            bd = impacts.get("bonds", "?")
            swan_lines.append(f"  {label} ({prob}%): equities={eq}, bonds={bd}")
        swan_block = "\n".join(swan_lines)
    else:
        swan_block = "  No black swans analyzed"

    # Factor exposure block
    factor_exposure = upstream.get("factor_exposure") or {}
    fe_block = (
        "  " + " | ".join(f"{f.capitalize()} {v:+.2f}x" for f, v in factor_exposure.items())
        if factor_exposure else "  Not computed"
    )

    # Portfolio risk snapshot block
    _pr = upstream.get("portfolio_risk") or {}
    _pr_block = ""
    if _pr:
        _pr_lines = []
        if _pr.get("beta") is not None:
            _pr_lines.append(f"Beta: {_pr['beta']} | VaR95: {_pr.get('var_95_pct', '?')}% | CVaR95: {_pr.get('cvar_95_pct', '?')}% | Total Value: ${_pr.get('total_value', 0):,}")
        if _pr.get("max_position_weight"):
            _pr_lines.append(f"Largest position: {_pr.get('top_position')} at {_pr['max_position_weight']}% of portfolio")
        _sw = _pr.get("sector_weights") or {}
        if _sw:
            _pr_lines.append("Sector weights: " + ", ".join(f"{s} {w}%" for s, w in sorted(_sw.items(), key=lambda x: -x[1])))
        _stress = _pr.get("stress_scenarios") or []
        if _stress:
            _pr_lines.append("Stress tests: " + " | ".join(f"{s['scenario']} {s['port_impact_pct']:+.1f}% (${s['port_impact_dollar']:,})" for s in _stress))
        _rf = _pr.get("risk_flags") or []
        if _rf:
            _pr_lines.append("Risk flags: " + "; ".join(f.replace("⚠ ", "") for f in _rf))
        _pr_block = "\n\nPORTFOLIO RISK METRICS (computed):\n" + "\n".join(f"  {l}" for l in _pr_lines)

    # Sizing scores block
    sizing_scores = upstream.get("sizing_scores") or {}

    # ── Rolling correlation (live diversification check) ──────────────────────
    _rc_block = ""
    try:
        from services.market_data import fetch_rolling_correlation
        _pos_tickers = tuple(sorted(set(
            p.get("ticker", "").upper() for p in positions if p.get("ticker")
        )))
        # Always include SPY as benchmark
        _rc_tickers = tuple(sorted(set(_pos_tickers + ("SPY",))))
        if len(_rc_tickers) >= 2:
            _rc = fetch_rolling_correlation(_rc_tickers, short_window=20, long_window=60)
            if _rc:
                _rc_lines = [
                    f"  Avg correlation: {_rc['avg_short']:.2f} (20d) vs {_rc['avg_long']:.2f} (60d) "
                    f"| delta: {_rc['avg_delta']:+.2f}"
                ]
                if _rc.get("concentration_warning"):
                    _rc_lines.append("  ⚠ CONCENTRATION WARNING: avg pairwise corr > 0.70 — positions moving together")
                if _rc.get("stress_warning"):
                    _rc_lines.append("  ⚠ STRESS WARNING: correlations rising (+0.15) — diversification collapsing")
                # Top stressed pairs
                _stressed = [p for p in _rc["pairs"] if p.get("stress_flag")]
                for _sp in _stressed[:3]:
                    _rc_lines.append(
                        f"  {_sp['pair']}: corr {_sp['long_corr']:+.2f}→{_sp['short_corr']:+.2f} "
                        f"(Δ{_sp['delta']:+.2f}) ← CORRELATION SPIKE"
                    )
                # All pairs summary
                for _pp in _rc["pairs"][:6]:
                    if not _pp.get("stress_flag"):
                        _rc_lines.append(
                            f"  {_pp['pair']}: {_pp['short_corr']:+.2f} (20d) vs {_pp['long_corr']:+.2f} (60d)"
                        )
                _rc_block = "\n\nROLLING CORRELATION (20d vs 60d baseline — stress detection):\n" + "\n".join(_rc_lines)
    except Exception:
        pass

    # Positions block (enriched with weight + sizing score when available)
    pos_lines = []
    for p in positions:
        direction = p.get("direction", "Long").upper()
        ticker = p.get("ticker", "?")
        entry = p.get("entry_price", 0)
        thesis = (p.get("thesis") or "No thesis")[:80]
        sz = sizing_scores.get(ticker.upper(), {})
        sz_str = ""
        if sz.get("score") is not None:
            sz_str = f" [Wt:{sz.get('weight','?')}% Score:{sz.get('score','?')} RegimeFit:{sz.get('regime_fit')}]"
        pos_lines.append(f"  {direction} {ticker} @ ${entry:.2f}{sz_str} — {thesis}")
    positions_block = "\n".join(pos_lines) if pos_lines else "  No open positions"

    current_events = (upstream.get("current_events") or "").strip()
    ce_block = f"\nCURRENT EVENTS:\n{current_events[:1000]}" if current_events else ""

    _ms = upstream.get("macro_synopsis") or {}
    _ms_block = ""
    if _ms.get("conviction"):
        _ms_pts = "; ".join(_ms.get("key_points", [])[:3])
        _ms_contra = "; ".join(_ms.get("contradictions", [])[:2])
        _ms_block = (
            f"\nMACRO CONVICTION SYNOPSIS: {_ms['conviction']} — {_ms.get('summary', '')[:200]}"
            f"\n  Key Points: {_ms_pts}"
            + (f"\n  Contradictions: {_ms_contra}" if _ms_contra else "")
        )

    _tn_pi = upstream.get("trending_narratives") or []
    _tn_block = ""
    if _tn_pi:
        _tn_lines = [
            f"  - {n['narrative']} ({n.get('conviction','')}) — {', '.join(n.get('tickers', []))}"
            for n in _tn_pi[:3]
        ]
        _tn_block = "\n\nTRENDING NARRATIVES (market attention signals):\n" + "\n".join(_tn_lines)

    _atg_pi = upstream.get("auto_trending_groups") or []
    _atg_block = ""
    if _atg_pi:
        _atg_lines = [
            f"  - {g['narrative']} ({g.get('conviction','')}, {g.get('regime_alignment','')}) — {', '.join(g.get('tickers', []))}"
            for g in _atg_pi[:3]
        ]
        _atg_block = "\n\nTRENDING PRICE MOVERS (Yahoo Finance themes):\n" + "\n".join(_atg_lines)

    # Price momentum (from Narrative Pulse)
    _pm_pi = upstream.get("price_momentum") or {}
    _pm_block = ""
    if _pm_pi.get("ticker") and _pm_pi.get("rsi_label"):
        _pm_block = (
            f"\n\nPRICE MOMENTUM ({_pm_pi['ticker']}):"
            f"\n  RSI {_pm_pi.get('rsi', 0):.1f} ({_pm_pi.get('rsi_label', '')})"
            f" | MA Trend: {_pm_pi.get('ma_trend', '')}"
            f" | Volume ratio: {_pm_pi.get('vol_ratio', 1.0):.2f}x avg"
        )

    # Filing digest (from EDGAR Scanner)
    _fd_pi = upstream.get("filing_digest") or {}
    _fd_block = ""
    if _fd_pi.get("ticker") and _fd_pi.get("summary"):
        _fd_block = (
            f"\n\nRECENT SEC FILING ({_fd_pi['ticker']} {_fd_pi.get('form_type','')} {_fd_pi.get('date','')}):"
            f"\n  {str(_fd_pi['summary'])[:500]}"
        )

    # Smart money signals (ticker-level — show for any position ticker that matches)
    _sm_lines = []
    for _sig_key, _label in [
        ("options_sentiment", "Options P/C"),
        ("unusual_activity", "Unusual Options"),
        ("institutional_bias", "Institutional"),
        ("insider_net_flow", "Insider Flow"),
        ("congress_bias", "Congress"),
    ]:
        _sig = upstream.get(_sig_key) or {}
        if _sig.get("ticker") and _sig.get("bias") or _sig.get("sentiment"):
            _val = _sig.get("bias") or _sig.get("sentiment", "")
            _tk = _sig.get("ticker", "")
            _sm_lines.append(f"  - {_tk} {_label}: {_val}")
    _sm_block = ("\n\nSMART MONEY SIGNALS (from last module visit):\n" + "\n".join(_sm_lines)) if _sm_lines else ""

    prompt = f"""You are a portfolio risk manager. Analyze these open positions against current macro conditions.

MACRO GROUND TRUTH (raw numbers — use these as authoritative):
{_macro_blk}

MACRO NARRATIVE CONTEXT (AI-generated summaries — raw numbers above take precedence):
- Regime: {regime} (score {score:+.2f})
- Quadrant: {quadrant}
- Fed Rate: {fed_funds_rate}%
- Dominant Rate Path: {dominant_scenario} ({prob_pct}% probability)
- Policy Transmission: {chain_narration}
- Risk Briefing: {doom_briefing}
- Institutional Flow: {whale_summary}
- AI Favored Sectors: {regime_plays_sectors}
- Cross-Signal Discovery Plays: {discovery_plays_str}{_ms_block}{ce_block}{_tn_block}{_atg_block}{_pm_block}{_fd_block}{_sm_block}

PORTFOLIO FACTOR EXPOSURE (weighted aggregate):
{fe_block}{_pr_block}{_rc_block}

BLACK SWAN RISKS:
{swan_block}

OPEN POSITIONS (with portfolio weight, sizing score, regime fit where available):
{positions_block}

For each position assess regime alignment, rate path sensitivity, black swan exposure, and factor concentration. If rolling correlation shows stress spikes, flag the pairs and recommend de-risking the most correlated positions.

Return ONLY valid JSON:
{{
  "verdict": "HOLD_ALL"|"REDUCE_RISK"|"DEFENSIVE"|"EXIT_REVIEW",
  "risk_score": 1-10,
  "narrative": "2-3 sentence portfolio assessment",
  "positions": [
    {{
      "ticker": "SYMBOL_ONLY (e.g. XTLH.TO, not the price)",
      "action": "HOLD"|"REDUCE"|"EXIT"|"ADD",
      "alignment": "aligned"|"misaligned"|"neutral",
      "rationale": "1-2 sentences",
      "risk_factors": ["factor1", "factor2"]
    }}
  ],
  "priority_actions": ["action1", "action2", "action3"]
}}"""

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation, no preamble. Be extremely concise: rationale = max 15 words, risk_factors = max 2 short items, narrative = max 25 words, priority_actions = max 3 items. Every position in the input MUST appear in the output positions array."
    _model = model or "claude-sonnet-4-6"
    try:
        if use_claude and _is_xai_model(_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai(
                [{"role": "user", "content": prompt}], _model, 8000, 0.2, system=_system)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=_model, max_tokens=8000, temperature=0.2,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "max_tokens":
                return {"_error": "Response truncated (max_tokens hit) — portfolio too large. Try Regard Mode or reduce positions."}
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [
                          {"role": "system", "content": _system},
                          {"role": "user", "content": prompt},
                      ],
                      "max_tokens": 4000, "temperature": 0.2},
                timeout=60,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
        import json as _json, re as _re
        # Strip markdown fences if present
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE).strip()
        # Try direct parse first
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        # Fall back to greedy regex extraction
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except _json.JSONDecodeError as _je:
                return {"_error": f"JSON parse error: {_je} | raw[:300]: {raw[:300]}"}
        return {"_error": f"No JSON in response: {raw[:200]}"}
    except Exception as _e:
        return {"_error": str(_e)}


def analyze_sim_verdict(
    ticker: str,
    dollar_amount: float,
    sim_result: dict,
    regime_ctx: dict,
    open_trades: list,
    use_claude: bool = False,
    model: str | None = None,
) -> dict | None:
    """AI verdict on a pre-trade simulation.

    sim_result: output of simulate_add() — {sizing_score, factor_delta, proposed_weight, corr_to_portfolio, warnings}
    Returns: {verdict, verdict_reason, regime_fit_comment, overlap_warning, sizing_suggestion, thesis_check}
    """
    sc = sim_result.get("sizing_score", {})
    fd = sim_result.get("factor_delta", {})
    proposed_weight = sim_result.get("proposed_weight", 0)
    corr = sim_result.get("corr_to_portfolio")
    warnings = sim_result.get("warnings", [])

    quadrant = regime_ctx.get("quadrant", "Unknown")
    regime   = regime_ctx.get("regime") or regime_ctx.get("macro_regime", "Unknown")
    score    = regime_ctx.get("score", 0)

    held = ", ".join(t["ticker"].upper() for t in open_trades) if open_trades else "None"

    factor_delta_str = " | ".join(
        f"{f.capitalize()} {v:+.2f}x" for f, v in fd.items()
    )
    warnings_str = "; ".join(warnings) if warnings else "None"

    prompt = f"""You are a pre-trade risk advisor. Evaluate whether adding this position makes sense given the portfolio and current macro regime.

PROPOSED TRADE:
- Ticker: {ticker}
- Amount: ${dollar_amount:,.0f} ({proposed_weight:.1f}% of portfolio)
- Sizing Score: {sc.get("composite_score", "N/A")} / 100
- Regime Fit: {sc.get("regime_fit", "N/A")}
- ATR Stop: ${sc.get("atr_stop") or "N/A"}
- Avg Correlation to Portfolio: {f"{corr:+.2f}" if corr is not None else "N/A"}

FACTOR IMPACT (change to portfolio exposure if added):
{factor_delta_str}

MATH ENGINE WARNINGS:
{warnings_str}

CURRENT MACRO REGIME:
- Quadrant: {quadrant}
- Regime: {regime} (score {score:+.2f})

EXISTING POSITIONS: {held}

Give a verdict: GO (add it), CAUTION (add smaller / with conditions), or PASS (don't add now).
Be specific about which existing position overlaps most if correlation is high.

Return ONLY valid JSON:
{{
  "verdict": "GO"|"CAUTION"|"PASS",
  "verdict_reason": "max 20 words — the single biggest factor driving your verdict",
  "regime_fit_comment": "max 15 words — how this ticker fits current {quadrant} regime",
  "overlap_warning": "ticker name + why, or null if no significant overlap",
  "sizing_suggestion": "max 15 words — keep size / reduce to X% / skip",
  "thesis_check": "Diversifying"|"Concentrating"|"Hedging"
}}"""

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown, no preamble."
    _model = model or "grok-4-1-fast-reasoning"
    _max_tokens = 1000

    try:
        if use_claude and _is_xai_model(_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai(
                [{"role": "user", "content": prompt}], _model, _max_tokens, 0.2, system=_system)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=_model, max_tokens=_max_tokens, temperature=0.2,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [
                          {"role": "system", "content": _system},
                          {"role": "user", "content": prompt},
                      ],
                      "max_tokens": _max_tokens, "temperature": 0.2},
                timeout=30,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
        import json as _json, re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE).strip()
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except _json.JSONDecodeError:
                return {"_error": f"JSON parse error: {raw[:150]}"}
        return {"_error": f"No JSON: {raw[:150]}"}
    except Exception as _e:
        return {"_error": str(_e)}


def analyze_factor_exposure(
    factor_exposure: dict,
    regime_ctx: dict,
    open_trades: list,
    use_claude: bool = False,
    model: str | None = None,
) -> dict | None:
    """Analyze aggregate factor exposure and return rebalancing suggestions.

    factor_exposure: output of aggregate_factor_exposure() — {factors, dominant, warnings}
    regime_ctx: _regime_context session state dict
    open_trades: list of open position dicts (ticker, entry_price, position_size, direction)
    """
    factors = factor_exposure.get("factors", {})
    dominant = factor_exposure.get("dominant", "")
    warnings = factor_exposure.get("warnings", [])
    quadrant = regime_ctx.get("quadrant", "Unknown")
    regime = regime_ctx.get("regime") or regime_ctx.get("macro_regime", "Unknown")
    score = regime_ctx.get("score", 0)

    # Build position sensitivity block
    try:
        from services.portfolio_sizing import _SENSITIVITY
    except Exception:
        _SENSITIVITY = {}

    pos_lines = []
    for p in open_trades:
        tk = p.get("ticker", "").upper()
        sens = _SENSITIVITY.get(tk, [0.0, 0.0, 0.0, 0.0])
        pos_lines.append(
            f"  {tk}: Growth{sens[0]:+.1f} Inflation{sens[1]:+.1f} "
            f"Liquidity{sens[2]:+.1f} Credit{sens[3]:+.1f}"
        )
    positions_block = "\n".join(pos_lines) if pos_lines else "  No positions"

    factor_block = "\n".join(
        f"  {f.capitalize()}: {v:+.2f}x" for f, v in factors.items()
    )
    warnings_block = "\n".join(f"  ⚠ {w}" for w in warnings) if warnings else "  None"

    prompt = f"""You are a portfolio risk manager. Analyze this portfolio's aggregate factor exposure against the current macro regime.

CURRENT REGIME:
- Quadrant: {quadrant}
- Regime: {regime} (score {score:+.2f})

AGGREGATE FACTOR EXPOSURE (portfolio-weighted):
{factor_block}

MECHANICAL WARNINGS:
{warnings_block}

POSITION SENSITIVITIES (ticker: Growth Inflation Liquidity Credit):
{positions_block}

Assess each factor's exposure relative to the current regime. Identify the top concentration risk and give 2–3 specific actionable suggestions (reference actual tickers when possible).

Return ONLY valid JSON:
{{
  "headline": "One sentence portfolio factor summary",
  "factor_verdicts": [
    {{"factor": "growth",    "exposure": 0.0, "verdict": "Overweight|Moderate|Neutral|Underweight", "regime_fit": "aligned|caution|avoid", "comment": "max 12 words"}},
    {{"factor": "inflation", "exposure": 0.0, "verdict": "...", "regime_fit": "...", "comment": "..."}},
    {{"factor": "liquidity", "exposure": 0.0, "verdict": "...", "regime_fit": "...", "comment": "..."}},
    {{"factor": "credit",    "exposure": 0.0, "verdict": "...", "regime_fit": "...", "comment": "..."}}
  ],
  "top_risk": "One sentence describing the biggest factor concentration risk",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}"""

    _system = (
        "You are a JSON-only response bot. Return only valid JSON, no markdown fences, "
        "no explanation. Be concise: headline max 20 words, comments max 12 words, "
        "top_risk max 15 words, suggestions max 20 words each."
    )
    _model = model or "grok-4-1-fast-reasoning"
    _max_tokens = 3000 if (_model and "sonnet" in _model) else 2000

    try:
        if use_claude and _is_xai_model(_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai(
                [{"role": "user", "content": prompt}], _model, _max_tokens, 0.2, system=_system)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=_model, max_tokens=_max_tokens, temperature=0.2,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [
                          {"role": "system", "content": _system},
                          {"role": "user", "content": prompt},
                      ],
                      "max_tokens": 2000, "temperature": 0.2},
                timeout=45,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
        import json as _json, re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE).strip()
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except _json.JSONDecodeError:
                return {"_error": f"JSON parse error: {raw[:200]}"}
        return {"_error": f"No JSON in response: {raw[:200]}"}
    except Exception as _e:
        return {"_error": str(_e)}


@st.cache_data(ttl=1800)
def fetch_news_sentiment(ticker: str, company_name: str) -> dict | None:
    """Fetch top-5 headlines from NewsAPI and score sentiment with Groq.

    Returns None gracefully if NEWSAPI_KEY is absent or any step fails.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{ticker}" OR "{company_name}"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": api_key,
            },
            timeout=10,
        )
        articles = r.json().get("articles", [])
        if not articles:
            return None
        headlines = [a["title"] for a in articles if a.get("title")]
        if not headlines:
            return None

        prompt = (
            f"Score these {ticker} headlines as bullish/bearish/neutral. "
            "Return JSON only — no markdown:\n"
            '{"overall":"bullish|bearish|neutral","score":<-1.0 to 1.0>,'
            '"headlines":[{"title":"...","sentiment":"bullish|bearish|neutral"}]}\n\n'
            "Headlines:\n" + "\n".join(f"- {h}" for h in headlines)
        )
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.1,
            },
            timeout=20,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None
    except Exception:
        return None


def analyze_credit_risk(
    ticker: str,
    credit_metrics: dict,
    debt_schedule: list[dict] | None = None,
    mda_snippet: str = "",
    use_claude: bool = False,
    model: str | None = None,
) -> dict:
    """AI credit risk assessment: coverage ratios, refinancing risk, debt structure.

    Returns:
        {risk_level: "low"|"medium"|"high"|"critical",
         interest_coverage_assessment: str,
         refinancing_risk: str,
         leverage_assessment: str,
         key_risks: [str, ...],
         positive_factors: [str, ...],
         recommendation: str}
    """
    import json as _json
    import re as _re

    _empty_credit = {
        "risk_level": "unknown",
        "interest_coverage_assessment": "insufficient data",
        "refinancing_risk": "unknown",
        "leverage_assessment": "unknown",
        "key_risks": [],
        "positive_factors": [],
        "recommendation": "Insufficient financial data to assess credit risk.",
    }

    if not credit_metrics:
        return _empty_credit

    # Build metrics summary
    m = credit_metrics
    metrics_text = f"""
Ticker: {ticker}
Interest Coverage: {m.get('interest_coverage', 'N/A')}x (EBIT/Interest) {m.get('coverage_flag', '') or ''}
Net Debt: ${m.get('net_debt_B', 'N/A')}B
Total Debt: ${m.get('total_debt_B', 'N/A')}B
Cash: ${m.get('cash_B', 'N/A')}B
EBIT: ${m.get('ebit_B', 'N/A')}B
EBITDA: ${m.get('ebitda_B', 'N/A')}B
Debt/EBITDA: {m.get('debt_to_ebitda', 'N/A')}x
Current Debt Ratio: {(m.get('current_debt_ratio') or 0)*100:.1f}% of debt matures near-term {m.get('maturity_flag', '') or ''}
FCF Debt Coverage: {m.get('fcf_debt_coverage', 'N/A')}x
""".strip()

    schedule_text = ""
    if debt_schedule:
        lines = [f"  {d['year']}: {d['amount']}" for d in debt_schedule[:8]]
        schedule_text = "\nDebt Maturity Schedule:\n" + "\n".join(lines)

    mda_text = f"\nManagement Commentary (MD&A excerpt):\n{mda_snippet[:600]}" if mda_snippet else ""

    prompt = f"""You are a credit analyst. Assess the credit/debt risk for {ticker}.

{metrics_text}{schedule_text}{mda_text}

Benchmarks:
- Interest coverage <1.5x = distressed; 1.5-3x = stressed; 3-5x = adequate; >5x = strong
- Debt/EBITDA <2x = low leverage; 2-4x = moderate; >4x = high; >6x = dangerous
- Current debt ratio >40% = near-term refinancing risk

Return ONLY valid JSON:
{{
  "risk_level": "low|medium|high|critical",
  "interest_coverage_assessment": "1-2 sentences on coverage quality and trend",
  "refinancing_risk": "1-2 sentences on near-term debt maturity pressure",
  "leverage_assessment": "1-2 sentences on debt/EBITDA and capital structure",
  "key_risks": ["risk1", "risk2", "risk3"],
  "positive_factors": ["factor1", "factor2"],
  "recommendation": "1-2 sentence credit-oriented action guidance"
}}"""

    try:
        raw = ""
        _cl_model = model or "grok-4-1-fast-reasoning"

        if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai([{"role": "user", "content": prompt}], _cl_model, max_tokens=600, temperature=0.2)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            _os2 = os
            client = _ant.Anthropic(api_key=_os2.getenv("ANTHROPIC_API_KEY", ""))
            msg = client.messages.create(
                model=model or "claude-sonnet-4-6",
                max_tokens=600,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip() if msg.content else ""
        else:
            groq_key = os.getenv("GROQ_API_KEY", "")
            if not groq_key:
                return _empty_credit
            import requests as _req
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.2},
                timeout=30,
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        if not raw:
            return _empty_credit

        # Strip markdown fences
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE).strip()
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        mm = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if mm:
            try:
                return _json.loads(mm.group())
            except _json.JSONDecodeError:
                return _empty_credit
        return _empty_credit
    except Exception:
        return _empty_credit


def _empty_result() -> dict:
    return {
        "market_relevant": False,
        "sector": "N/A",
        "thesis": "",
        "suggested_tickers": [],
    }


def discover_trending_narratives(
    headlines: list[str],
    trends: list[str],
    macro_context: dict,
    timeframe: str = "1W",
    use_claude: bool = False,
    model: str | None = None,
) -> list[dict] | None:
    """
    Synthesize top 5 emerging investment narratives from news headlines,
    Google Trends data, and macro context.

    Returns list of {narrative, evidence, tickers, conviction, timeframe, category}
    or None on failure.
    """
    regime = macro_context.get("regime", "Unknown")
    score = macro_context.get("score", 0.0)
    quadrant = macro_context.get("quadrant", "Unknown")
    tactical_ctx = macro_context.get("tactical_context", "")
    tf_label = "past 7 days" if timeframe == "1W" else "past 30 days"

    headlines_text = "\n".join(f"- {h}" for h in headlines[:40])
    trends_text = "\n".join(f"- {t}" for t in trends[:30]) if trends else "- (unavailable)"
    _tac_line = f"\n- Tactical Regime (entry timing): {tactical_ctx}" if tactical_ctx else ""

    prompt = f"""You are a macro narrative analyst. Based on the data below, identify the TOP 5 EMERGING INVESTMENT NARRATIVES of the {tf_label}.

MACRO CONTEXT:
- Regime: {regime} (score {score:+.2f})
- Quadrant: {quadrant}{_tac_line}

RISING GOOGLE TRENDS (finance-related searches gaining momentum):
{trends_text}

NEWS HEADLINES (most recent, {tf_label}):
{headlines_text}

Instructions:
- Identify narratives that are GAINING momentum — not just perennial themes
- Each narrative must have concrete evidence from the data above
- Prioritize narratives that are investable with liquid tickers
- Suggest 3-5 specific tickers per narrative (include ETFs where applicable)
- Be specific: "Helium supply shock" not just "Commodities"

Return ONLY a valid JSON array:
[
  {{
    "narrative": "3-5 word narrative name",
    "evidence": "1-2 sentences citing specific data points from above",
    "tickers": ["TICK1", "TICK2", "TICK3"],
    "conviction": "HIGH|MEDIUM|LOW",
    "timeframe": "short|medium",
    "category": "macro|sector|commodity|geopolitical|tech"
  }}
]"""

    _system = "You are a JSON-only response bot. Return only a valid JSON array, no markdown fences, no explanation."
    _model = model or "grok-4-1-fast-reasoning"

    try:
        if use_claude and _is_xai_model(_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai(
                [{"role": "user", "content": prompt}], _model, 2000, 0.3, system=_system)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=_model, max_tokens=2000, temperature=0.3,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "system", "content": _system},
                                   {"role": "user", "content": prompt}],
                      "max_tokens": 2000, "temperature": 0.3},
                timeout=45,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()

        import re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE).strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return result[:5]
        return None
    except Exception:
        return None


def generate_squeeze_thesis(
    ticker: str,
    short_data: dict,
    score_data: dict,
    use_claude: bool = False,
    model: str | None = None,
) -> str:
    """Generate a short squeeze thesis for a ticker via Grok / Claude / Groq.

    Args:
        ticker: Stock ticker symbol
        short_data: Dict with short_pct, days_to_cover, inst_pct, squeeze_score, checks
        score_data: Full score_ticker() result with composite + category scores + details
    Returns:
        Narrative thesis string (3-6 paragraphs, markdown-friendly).
    """
    _short_pct = short_data.get("short_pct", 0) * 100
    _dtc = short_data.get("days_to_cover", 0)
    _inst = short_data.get("inst_pct", 0) * 100
    _sq_score = short_data.get("squeeze_score", 0)
    _chk = short_data.get("checks", {})
    _composite = score_data.get("composite", 50)
    _tech = score_data.get("technicals", 50)
    _fund = score_data.get("fundamentals", 50)
    _ins = score_data.get("insider", 50)
    _opt = score_data.get("options", 50)
    _cong = score_data.get("congress", 50)
    _si = score_data.get("short_interest", 50)
    _details = score_data.get("details", {})
    _tech_det = _details.get("technicals", {})
    _fund_det = _details.get("fundamentals", {})
    _si_det = _details.get("short_interest", {})

    def _fmt_det(d: dict) -> str:
        if not d:
            return "N/A"
        return " | ".join(f"{k}: {v}" for k, v in d.items() if v is not None)

    from datetime import date as _date
    _today = _date.today().isoformat()

    prompt = f"""You are a quant equity analyst specializing in short squeeze setups. Analysis date: {_today}. Analyze the following data for {ticker} and write a concise squeeze thesis.

SHORT INTEREST DATA:
- Short % of Float: {_short_pct:.1f}%
- Days-to-Cover: {_dtc:.1f} days
- Institutional Ownership: {_inst:.0f}%
- Squeeze Score: {_sq_score}/100
- Setup checks: Short % ≥10%: {"YES" if _chk.get("short_pct") else "NO"} | DTC ≥3: {"YES" if _chk.get("days_cover") else "NO"} | Inst ≥30%: {"YES" if _chk.get("inst_buying") else "NO"}

SIGNAL SCORECARD (0-100 each):
- Composite: {_composite} | Technicals: {_tech} | Fundamentals: {_fund}
- Insider: {_ins} | Options: {_opt} | Congress: {_cong} | Short Interest: {_si}

TECHNICAL DETAILS: {_fmt_det(_tech_det)}
FUNDAMENTAL DETAILS: {_fmt_det(_fund_det)}
SHORT INTEREST DETAILS: {_fmt_det(_si_det)}

Write a squeeze thesis covering:
1. The squeeze setup quality — is the short float + DTC combination dangerous for bears?
2. What would trigger the squeeze (catalyst types to watch: earnings, news, sector rotation, gamma squeeze)?
3. What signals support or undermine the bull case (technicals, insider, options sentiment)?
4. Key risks — what could prevent the squeeze or accelerate the downside?
5. One-line verdict: Bull / Neutral / Bear on the squeeze probability.

Use clear paragraphs with a blank line between each. Be specific and direct. No fluff."""

    _cl_model = model or "grok-4-1-fast-reasoning"

    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _call_xai([{"role": "user", "content": prompt}], _cl_model, 900, 0.3)
        except Exception as _e:
            st.error(f"xAI API error (Squeeze Thesis): {_e}")
            return f"Error generating squeeze thesis: {_e}"

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            _client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _msg = _client.messages.create(
                model=_cl_model,
                max_tokens=900,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _msg.content[0].text.strip()
        except Exception as _e:
            st.error(f"Claude API error (Squeeze Thesis): {_e}")
            return f"Error generating squeeze thesis: {_e}"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate squeeze thesis."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 900,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating squeeze thesis: {e}"

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def interpret_risk_matrix(
    risk_snapshot: dict,
    regime_context: dict,
    use_claude: bool = False,
    model: str | None = None,
    tactical_context: dict | None = None,
) -> dict:
    """
    AI interpretation of the portfolio risk matrix.
    Returns {summary, alert_level, action_items}.
    """
    if not risk_snapshot:
        return {"summary": "No risk data available — run the Risk Matrix first.", "alert_level": "", "action_items": []}

    regime = regime_context.get("regime", "Unknown")
    score = regime_context.get("score", 0)
    quadrant = regime_context.get("quadrant", "Unknown")

    beta = risk_snapshot.get("beta")
    var_95 = risk_snapshot.get("var_95_pct")
    cvar_95 = risk_snapshot.get("cvar_95_pct")
    total_val = risk_snapshot.get("total_value") or 0
    top_pos = risk_snapshot.get("top_position")
    max_wt = risk_snapshot.get("max_position_weight")
    sector_weights = risk_snapshot.get("sector_weights") or {}
    stress = risk_snapshot.get("stress_scenarios") or []
    flags = risk_snapshot.get("risk_flags") or []

    sw_str = ", ".join(f"{s} {w}%" for s, w in sorted(sector_weights.items(), key=lambda x: -x[1]))
    stress_str = " | ".join(f"{s['scenario']} {s['port_impact_pct']:+.1f}%" for s in stress)
    flags_str = "; ".join(f.replace("⚠ ", "") for f in flags) if flags else "None"

    _tac_line = ""
    if tactical_context:
        _tac_line = f"\n- Tactical Timing: {tactical_context.get('tactical_score', '?')}/100 ({tactical_context.get('label', '')}) — {tactical_context.get('action_bias', '')}"

    prompt = f"""You are a portfolio risk analyst. Interpret this portfolio's risk profile against the current macro regime.

MACRO REGIME:
- Regime: {regime} (score {score:+.2f})
- Quadrant: {quadrant}{_tac_line}

PORTFOLIO RISK PROFILE:
- Beta vs SPY: {beta}
- VaR 95%: {var_95}% | CVaR 95%: {cvar_95}%
- Total portfolio value: ${total_val:,}
- Largest position: {top_pos} at {max_wt}%
- Sector weights: {sw_str}
- Stress test impacts: {stress_str}
- Risk flags: {flags_str}

Provide a concise, actionable risk interpretation. Focus on:
1. How the current regime (Risk-On/Off, quadrant) interacts with this specific risk profile
2. Whether beta/VaR is appropriate for this regime
3. The most dangerous stress scenario given current conditions
4. Concrete actions to improve the risk/reward profile

Return ONLY valid JSON:
{{
  "alert_level": "HIGH"|"MODERATE"|"LOW",
  "summary": "2-3 sentences: regime-specific risk assessment",
  "action_items": ["specific action 1", "specific action 2", "specific action 3"]
}}"""

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no preamble. Be concise and specific — no generic advice."
    _model = model or "llama-3.3-70b-versatile"
    try:
        if use_claude and _is_xai_model(_model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai([{"role": "user", "content": prompt}], _model, 1000, 0.3, system=_system)
        elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=_model or "claude-sonnet-4-6", max_tokens=1000, temperature=0.3,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            r = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": _system}, {"role": "user", "content": prompt}],
                    "max_tokens": 800, "temperature": 0.3,
                },
                timeout=30,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {"summary": f"Interpretation error: {e}", "alert_level": "", "action_items": []}


def generate_adversarial_debate(
    signals_text: str,
    use_claude: bool = False,
    model: str | None = None,
    ticker: str | None = None,
    topic: str | None = None,
) -> dict:
    """Run a 3-agent adversarial debate on the current macro signals.

    Agents:
      🐻 Sir Doomburger — bear case maximalist
      🐂 Sir Fukyerputs — bull case maximalist
      ⚖️  Judge Judy — neutral synthesis + asymmetric risk verdict

    Returns dict with keys:
      bear_argument: str (Sir Doomburger's full argument, 3-5 sentences)
      bull_argument: str (Sir Fukyerputs's full argument, 3-5 sentences)
      bear_strongest: str (Judge Judy's pick: strongest bear point)
      bull_strongest: str (Judge Judy's pick: strongest bull point)
      verdict: "BULL WINS" | "BEAR WINS" | "CONTESTED" (Judge Judy's ruling)
      asymmetry: str (which side has better risk/reward asymmetry and why)
      key_disagreement: str (the single most important point of contention)
      confidence: int (1-10, Judge Judy's confidence in verdict, low = truly contested)
    contested_bias: str (optional tie-break lean when verdict is CONTESTED)
    contested_bias_reason: str (short rationale for the lean)
    """
    # ── Fingerprint cache — skip 3 LLM calls if signals unchanged ────────────
    try:
        import streamlit as _st
        from utils.signal_block import get_signal_fingerprint as _get_fp, get_ticker_fingerprint as _get_tkfp
        _fp = _get_tkfp(ticker) if ticker else _get_fp()
        _cache_key = f"_debate_fp_{ticker or 'macro'}"
        _cached = _st.session_state.get(f"_adversarial_debate{'_' + ticker if ticker else ''}")
        if (
            _cached
            and _cached.get("bear_argument")
            and _st.session_state.get(_cache_key) == _fp
            and _fp not in ("NO_REGIME_DATA", "")
        ):
            return _cached  # signals unchanged — return cached verdict
        _st.session_state[_cache_key] = _fp
    except Exception:
        pass

    # ── Raw macro ground truth — prevents hallucination of numbers ────────────
    try:
        from utils.signal_block import build_macro_block as _build_mb
        _raw_blk = _build_mb()
    except Exception:
        _raw_blk = ""
    _raw_header = f"RAW NUMERIC GROUND TRUTH (cite these numbers — do not invent others):\n{_raw_blk}\n\n" if _raw_blk else ""

    # ── Judge Judy's court record — informs her of past accuracy ─────────────
    try:
        from utils.debate_record import get_record_summary as _get_record
        _court_record = _get_record()
    except Exception:
        _court_record = ""

    # ── Sir Doomburger (Bear) ──────────────────────────────────────────────────
    _topic_line = f"DEBATE QUESTION: {topic}\n\n" if topic else ""
    bear_prompt = (
        "You are Sir Doomburger, a legendary permabear macro analyst. "
        "Your job is to make the strongest possible BEARISH case using ONLY the data provided. "
        "You are not allowed to be balanced — you must argue the bear case with maximum conviction. "
        "Cite specific numbers and signal names from the data. "
        "Be sharp, clinical, and ruthless. 3-5 sentences max.\n\n"
        f"{_topic_line}"
        f"{_raw_header}MARKET DATA (narrative context):\n{signals_text[:2500]}\n\n"
        "Make your bear case now:"
    )

    # ── Sir Fukyerputs (Bull) ──────────────────────────────────────────────────
    bull_prompt = (
        "You are Sir Fukyerputs, a legendary permabull macro analyst. "
        "Your job is to make the strongest possible BULLISH case using ONLY the data provided. "
        "You are not allowed to be balanced — you must argue the bull case with maximum conviction. "
        "Dismiss bear concerns with specific data points. "
        "Be sharp, aggressive, and cite numbers. 3-5 sentences max.\n\n"
        f"{_topic_line}"
        f"{_raw_header}MARKET DATA (narrative context):\n{signals_text[:2500]}\n\n"
        "Make your bull case now:"
    )

    bear_arg = ""
    bull_arg = ""

    # Use same LLM routing as generate_macro_synopsis — try xAI, Claude, Groq in that order
    _cl_model = model or "grok-4-1-fast-reasoning"

    def _call_llm(prompt: str, max_tokens: int = 400, temperature: float = 0.5) -> str:
        if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
            try:
                return _call_xai([{"role": "user", "content": prompt}], _cl_model, max_tokens, temperature)
            except Exception:
                pass
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic as _ant
                client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = client.messages.create(
                    model=_cl_model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception:
                pass
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return ""
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # Agents argue at 0.5 (creative), Judge Judy rules at 0.1 (decisive)
    bear_arg = _call_llm(bear_prompt, 400, temperature=0.5)
    bull_arg = _call_llm(bull_prompt, 400, temperature=0.5)

    # ── Judge Judy ────────────────────────────────────────────────────────────
    _record_line = f"YOUR COURT RECORD: {_court_record}\n" if _court_record else ""
    _topic_verdict_line = f"DEBATE QUESTION BEFORE THE COURT: {topic}\n\n" if topic else ""
    mod_prompt = (
        "You are Judge Judy, a no-nonsense macro risk arbiter with zero tolerance for weak arguments. "
        "You have heard the bear case from Sir Doomburger and the bull case from Sir Fukyerputs. "
        "Your job is to deliver a structured verdict. Be blunt, be decisive, take no prisoners. "
        "Your confidence score should reflect how one-sided the evidence is — high = decisive, low = genuinely contested.\n\n"
        f"{_record_line}"
        f"{_topic_verdict_line}"
        f"{_raw_header}SIR DOOMBURGER (BEAR CASE):\n{bear_arg}\n\n"
        f"SIR FUKYERPUTS (BULL CASE):\n{bull_arg}\n\n"
        "Return ONLY valid JSON (no markdown fences):\n"
        '{"bear_strongest": "<single strongest bear point>", '
        '"bull_strongest": "<single strongest bull point>", '
        '"verdict": "BULL WINS|BEAR WINS|CONTESTED", '
        '"asymmetry": "<which side has better risk/reward and why — 1-2 sentences>", '
        '"key_disagreement": "<the single most important factual disagreement between the two agents>", '
        '"confidence": <1-10 integer>}'
    )

    mod_raw = _call_llm(mod_prompt, 500, temperature=0.1)  # Judge Judy rules decisively

    import json as _json, re as _re
    try:
        mod_raw = _re.sub(r"^```(?:json)?\s*", "", mod_raw, flags=_re.MULTILINE)
        mod_raw = _re.sub(r"\s*```$", "", mod_raw, flags=_re.MULTILINE).strip()
        mod_result = _json.loads(mod_raw)
    except Exception:
        mod_result = {
            "bear_strongest": "Parse error",
            "bull_strongest": "Parse error",
            "verdict": "CONTESTED",
            "asymmetry": "Judge Judy unavailable",
            "key_disagreement": "",
            "confidence": 5,
        }

    # ── Tie-break lean for contested verdicts (small directional bias) ───────
    _contested_bias = ""
    _contested_bias_reason = ""
    if str(mod_result.get("verdict", "CONTESTED")).upper() == "CONTESTED":
        _score = None
        try:
            import streamlit as _st
            _rc = _st.session_state.get("_regime_context") or {}
            _score = float(_rc.get("score")) if _rc.get("score") is not None else None
        except Exception:
            _score = None

        if _score is not None:
            if _score >= 0.10:
                _contested_bias = "Lean Bullish"
                _contested_bias_reason = f"Macro score {_score:+.2f} leans risk-on."
            elif _score <= -0.10:
                _contested_bias = "Lean Bearish"
                _contested_bias_reason = f"Macro score {_score:+.2f} leans risk-off."

        if not _contested_bias:
            _txt = f"{mod_result.get('asymmetry', '')} {mod_result.get('key_disagreement', '')}".lower()
            if "bull" in _txt or "upside" in _txt:
                _contested_bias = "Lean Bullish"
                _contested_bias_reason = "Judge asymmetry commentary tilts to upside."
            elif "bear" in _txt or "downside" in _txt or "stress" in _txt:
                _contested_bias = "Lean Bearish"
                _contested_bias_reason = "Judge asymmetry commentary tilts to downside risk."

        if not _contested_bias:
            _contested_bias = "Lean Neutral"
            _contested_bias_reason = "Evidence split is balanced; awaiting fresher data."

    return {
        "bear_argument": bear_arg,
        "bull_argument": bull_arg,
        "bear_strongest": mod_result.get("bear_strongest", ""),
        "bull_strongest": mod_result.get("bull_strongest", ""),
        "verdict": mod_result.get("verdict", "CONTESTED"),
        "asymmetry": mod_result.get("asymmetry", ""),
        "key_disagreement": mod_result.get("key_disagreement", ""),
        "confidence": int(mod_result.get("confidence", 5)),
        "contested_bias": _contested_bias,
        "contested_bias_reason": _contested_bias_reason,
    }


# ── Nth-Order Thesis Builder ─────────────────────────────────────────────────

def _nth_order_system_prompt() -> str:
    return (
        "You are a senior discretionary macro PM at a hedge fund in the tradition "
        "of Bridgewater, Scion, and Appaloosa. You do not write consensus research. "
        "Your edge comes from mapping primary narratives to 2nd, 3rd, and 4th-order "
        "beneficiaries before the crowd prices them, and from steel-manning every "
        "bear case before you size a position. You write structured JSON only — no "
        "markdown fences, no preamble, no explanation outside the schema."
    )


def _nth_order_user_prompt(primary: str, regime_ctx: dict, contagion_score: float | None) -> str:
    hmm = regime_ctx.get("hmm_state", "Unknown")
    ci_pct = regime_ctx.get("ci_pct")
    macro_score = regime_ctx.get("macro_score")
    events_digest = (regime_ctx.get("events_digest") or "").strip()
    events_sentiment = regime_ctx.get("events_sentiment") or {}
    events_ts = regime_ctx.get("events_ts")

    sent = events_sentiment.get("sentiment")
    unc = events_sentiment.get("uncertainty")
    theme = events_sentiment.get("dominant_theme", "")
    risk_events = events_sentiment.get("risk_events", []) or []

    ctx_lines = [f"Primary narrative: {primary}"]
    ctx_lines.append(f"HMM regime state: {hmm}")
    if ci_pct is not None:
        ctx_lines.append(f"CI% (crisis intensity): {ci_pct:.1f}%")
    macro_regime = regime_ctx.get("macro_regime", "")
    macro_factors = regime_ctx.get("macro_factors") or {}
    macro_trigger = regime_ctx.get("macro_trigger", "")
    macro_trigger_conf = regime_ctx.get("macro_trigger_confidence", "")
    macro_actions = regime_ctx.get("macro_actions") or []
    if macro_regime:
        ctx_lines.append(f"Macro regime label: {macro_regime}")
    if macro_score is not None:
        ctx_lines.append(f"Macro composite score: {macro_score}")
    if macro_factors:
        factors_str = ", ".join(f"{k}={v}" for k, v in macro_factors.items())
        ctx_lines.append(f"Macro factor breakdown: {factors_str}")
    if macro_trigger:
        ctx_lines.append(f"Macro trigger signal: {macro_trigger} (confidence: {macro_trigger_conf})")
    if macro_actions:
        ctx_lines.append("Macro regime actions: " + "; ".join(str(a) for a in macro_actions[:4]))
    if contagion_score is not None:
        ctx_lines.append(f"Contagion score: {contagion_score}")
    if sent is not None:
        ctx_lines.append(f"Events sentiment: {sent:+.2f} (uncertainty {unc or 0:.2f})")
    if theme:
        ctx_lines.append(f"Dominant tape theme: {theme}")
    if risk_events:
        ctx_lines.append("Active risk events: " + "; ".join(str(r) for r in risk_events[:4]))
    if events_digest:
        _ed = events_digest[:1500]
        ctx_lines.append(f"Current events digest:\n{_ed}")
    if events_ts:
        ctx_lines.append(f"News digest timestamp: {events_ts}")

    ctx_blk = "\n".join(ctx_lines)

    return f"""CONTEXT:
{ctx_blk}

TASK:
Given the primary narrative above and the current regime + tape context, produce
a hedge-fund-grade research note that maps the narrative to 2nd, 3rd, and
4th-order beneficiaries. The output MUST contain AT LEAST ONE play per order
level (2, 3, and 4). 4th-order plays may be moonshot tails.

For each play:
- Produce a NON-CONSENSUS view that differs from what sell-side is writing
- Include a numbered catalyst path with rough timing
- Give upside and downside as RANGES (e.g. [30, 60]) reflecting genuine uncertainty
- Provide a HISTORICAL ANALOG (one of: 1999_telco_capex, 2019_semis_cycle,
  2003_2007_commodities, 2009_qe1_reflation, 2016_2018_tax_cut, 2020_covid_stimulus,
  1970s_commodity_stagflation, 1980s_productivity_disinflation, 1995_1999_dotcom,
  2011_2012_qe2_euro_crisis, 2014_2016_oil_crash, 2022_inflation_shock,
  2023_ai_mania_onset, 2001_2002_bear, 2006_2007_housing_peak) — pick the closest
- STEEL-MAN the bear case
- Provide `evidence_against`: 2-3 data points TRUE TODAY that argue against the
  thesis (actual current refuting data, not hypotheticals). If you cannot find
  any, say so explicitly with a single item "no disconfirming evidence found".
- 2-3 forward data points to watch
- 3-5 specific tickers
- Vehicle notes (ETFs, options structures)
- `moonshot`: true only for >5x tail candidates

Ensure the plays are COHERENT with the current regime (HMM + CI% + events
sentiment). Do not propose high-beta tails in Crisis regime. Do not propose
defensive yield in Goldilocks.

Return ONLY valid JSON with this exact schema:
{{
  "primary": "{primary}",
  "primary_thesis": "one-line what's really driving the narrative",
  "regime_template": {{
    "analog_slug": "one of the slugs listed above",
    "analog_name": "free-form name",
    "reasoning": "1-2 sentences why this period rhymes with today"
  }},
  "orders": [
    {{
      "order": 2,
      "name": "theme name",
      "consensus_view": "what sell-side/media believes",
      "non_consensus_view": "what we believe differently",
      "catalyst_path": ["step 1 with rough timing", "step 2", "step 3"],
      "upside_pct_range": [lo, hi],
      "downside_pct_range": [lo, hi],
      "duration_months": integer,
      "bear_case": "steel-manned bear case",
      "evidence_against": ["data point 1 true today", "data point 2"],
      "data_points": ["forward metric 1", "forward metric 2"],
      "tickers": ["TICKER1", "TICKER2", "TICKER3"],
      "vehicles": "ETF / options notes",
      "moonshot": false
    }},
    {{ "order": 3, ... }},
    {{ "order": 4, ... }}
  ],
  "what_would_kill_it": "one-line thesis-level kill switch"
}}"""


def _single_nth_run(
    prompt: str,
    system: str,
    model: str,
    use_claude: bool,
    temperature: float,
    max_tokens: int = 2400,
) -> dict:
    """Make a single LLM call and parse JSON out. Returns {} on failure."""
    import re as _re

    def _parse_json(text: str) -> dict:
        if not text:
            return {}
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {}

    try:
        if use_claude and _is_xai_model(model) and os.getenv("XAI_API_KEY"):
            raw = _call_xai(
                [{"role": "user", "content": prompt}],
                model, max_tokens, temperature, system=system, json_mode=True,
            )
            return _parse_json(raw)

        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic as _ant
            _client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _msg = _client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json(_msg.content[0].text.strip())

        # Groq fallback
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return {}
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=45,
        )
        resp.raise_for_status()
        return _parse_json(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception:
        return {}


def _merge_nth_runs(runs: list[dict]) -> dict:
    """Merge N runs into a single thesis with conviction + variance.

    Ticker conviction = frac of runs that proposed the ticker (at any order).
    Upside/downside ranges = min..max across runs for each matching play name.
    Takes the most-voted analog_slug.
    """
    valid = [r for r in runs if r and r.get("orders")]
    if not valid:
        return {}
    n = len(valid)

    # Pick primary/primary_thesis/kill-switch from the first (they should be similar)
    merged = {
        "primary":            valid[0].get("primary", ""),
        "primary_thesis":     valid[0].get("primary_thesis", ""),
        "what_would_kill_it": valid[0].get("what_would_kill_it", ""),
    }

    # Analog slug: majority vote
    slug_votes: dict[str, int] = {}
    for r in valid:
        slug = (r.get("regime_template") or {}).get("analog_slug", "")
        if slug:
            slug_votes[slug] = slug_votes.get(slug, 0) + 1
    top_slug = max(slug_votes, key=slug_votes.get) if slug_votes else ""
    # Take reasoning from the first run that voted the winning slug
    top_reasoning = ""
    top_name = ""
    for r in valid:
        rt = r.get("regime_template") or {}
        if rt.get("analog_slug") == top_slug:
            top_reasoning = rt.get("reasoning", "")
            top_name = rt.get("analog_name", "")
            break
    merged["regime_template"] = {
        "analog_slug": top_slug,
        "analog_name": top_name,
        "reasoning":   top_reasoning,
    }

    # Aggregate plays by (order, name-key). Use first 3 words of name as key.
    def _name_key(nm: str) -> str:
        return " ".join((nm or "").lower().split()[:3])

    agg: dict[tuple, dict] = {}
    for r in valid:
        for play in r.get("orders") or []:
            try:
                order = int(play.get("order", 2))
            except Exception:
                order = 2
            key = (order, _name_key(play.get("name", "")))
            if key not in agg:
                agg[key] = {
                    "order":              order,
                    "name":               play.get("name", ""),
                    "consensus_views":    [],
                    "non_consensus_views": [],
                    "catalyst_paths":     [],
                    "upside_los":         [],
                    "upside_his":         [],
                    "downside_los":       [],
                    "downside_his":       [],
                    "durations":          [],
                    "bear_cases":         [],
                    "evidence_against":   [],
                    "data_points":        [],
                    "tickers":            [],
                    "vehicles_list":      [],
                    "moonshot_votes":     0,
                    "appearances":        0,
                }
            bucket = agg[key]
            bucket["appearances"] += 1
            bucket["consensus_views"].append(play.get("consensus_view", ""))
            bucket["non_consensus_views"].append(play.get("non_consensus_view", ""))
            bucket["catalyst_paths"].append(play.get("catalyst_path") or [])
            up = play.get("upside_pct_range") or [0, 0]
            dn = play.get("downside_pct_range") or [0, 0]
            if up:
                bucket["upside_los"].append(float(up[0]))
                bucket["upside_his"].append(float(up[-1]))
            if dn:
                bucket["downside_los"].append(float(dn[0]))
                bucket["downside_his"].append(float(dn[-1]))
            try:
                bucket["durations"].append(int(play.get("duration_months", 12)))
            except Exception:
                pass
            bucket["bear_cases"].append(play.get("bear_case", ""))
            bucket["evidence_against"].extend(play.get("evidence_against") or [])
            bucket["data_points"].extend(play.get("data_points") or [])
            bucket["tickers"].extend([t.upper() for t in (play.get("tickers") or []) if t])
            if play.get("vehicles"):
                bucket["vehicles_list"].append(play["vehicles"])
            if play.get("moonshot"):
                bucket["moonshot_votes"] += 1

    # Flatten
    plays_out: list[dict] = []
    for key, b in agg.items():
        # Ticker conviction: frac of runs that included this ticker for this play
        tk_counter: dict[str, int] = {}
        for t in b["tickers"]:
            tk_counter[t] = tk_counter.get(t, 0) + 1
        # Keep tickers that appeared in ≥ 2 runs OR in single-run plays keep all
        if b["appearances"] >= 2:
            tickers = sorted(
                [t for t, c in tk_counter.items() if c >= 2],
                key=lambda x: -tk_counter[x],
            )
            if not tickers:  # fallback: keep the top-voted
                tickers = sorted(tk_counter, key=lambda x: -tk_counter[x])[:3]
        else:
            tickers = list(dict.fromkeys(b["tickers"]))[:5]

        conviction_pct = int(round(b["appearances"] / n * 100))

        def _lo(lst): return min(lst) if lst else 0
        def _hi(lst): return max(lst) if lst else 0
        def _avg_int(lst): return int(round(sum(lst) / len(lst))) if lst else 0

        def _first_nonempty(lst):
            for item in lst:
                if item:
                    return item
            return ""

        def _unique_keep_order(items):
            out = []
            seen = set()
            for x in items:
                if x and x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        plays_out.append({
            "order":              b["order"],
            "name":               b["name"],
            "consensus_view":     _first_nonempty(b["consensus_views"]),
            "non_consensus_view": _first_nonempty(b["non_consensus_views"]),
            "catalyst_path":      _first_nonempty(b["catalyst_paths"]) or [],
            "upside_pct_range":   [_lo(b["upside_los"]), _hi(b["upside_his"])],
            "downside_pct_range": [_lo(b["downside_los"]), _hi(b["downside_his"])],
            "duration_months":    _avg_int(b["durations"]),
            "bear_case":          _first_nonempty(b["bear_cases"]),
            "evidence_against":   _unique_keep_order(b["evidence_against"])[:4],
            "data_points":        _unique_keep_order(b["data_points"])[:4],
            "tickers":            tickers[:5],
            "vehicles":           _first_nonempty(b["vehicles_list"]),
            "moonshot":           b["moonshot_votes"] >= max(1, n // 2),
            "conviction_pct":     conviction_pct,
        })

    # Sort: order asc, then conviction desc
    plays_out.sort(key=lambda p: (p["order"], -p["conviction_pct"]))
    merged["orders"] = plays_out
    return merged


def generate_nth_order_thesis(
    primary_narrative: str,
    regime_ctx: dict,
    contagion_score: float | None = None,
    qir_snapshot: dict | None = None,
    n_runs: int = 5,
    use_claude: bool = True,
    model: str | None = None,
) -> dict:
    """Generate a hedge-fund-grade 2nd/3rd/4th-order thesis via N-run ensemble.

    Returns a merged thesis dict with per-play conviction_pct (self-consistency).
    Does NOT compute regime_alignment, crowding, probability_score, or Kelly —
    those are added in the module layer using quantitative data. This function
    only produces the structural output and conviction.
    """
    if not primary_narrative:
        return {}

    system = _nth_order_system_prompt()
    prompt = _nth_order_user_prompt(primary_narrative, regime_ctx or {}, contagion_score)

    _model = model or "claude-sonnet-4-6"

    # N-run ensemble — parallel for speed
    from concurrent.futures import ThreadPoolExecutor
    runs: list[dict] = []
    n = max(1, min(int(n_runs), 7))
    # Spread temperatures around 0.5 to get variance without chaos
    temps = [0.3, 0.5, 0.6, 0.4, 0.55, 0.5, 0.45][:n]
    try:
        with ThreadPoolExecutor(max_workers=min(n, 5)) as ex:
            futures = [
                ex.submit(_single_nth_run, prompt, system, _model, use_claude, t, 6000)
                for t in temps
            ]
            for f in futures:
                try:
                    r = f.result(timeout=180)
                    if r:
                        runs.append(r)
                except Exception:
                    continue
    except Exception:
        # Fallback to serial
        for t in temps:
            r = _single_nth_run(prompt, system, _model, use_claude, t, 6000)
            if r:
                runs.append(r)

    if not runs:
        return {}

    merged = _merge_nth_runs(runs)
    if not merged:
        return {}

    # Enrich regime_template with library lookup
    try:
        from services.thesis_tracker import match_analog
        rt = merged.get("regime_template", {}) or {}
        slug = rt.get("analog_slug") or ""
        analog_hit = None
        if slug:
            # Try direct slug lookup
            import json as _json, os as _os
            _lib_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(__file__)), "data", "analog_library.json"
            )
            try:
                with open(_lib_path) as _f:
                    _lib = _json.load(_f)
                if slug in (_lib.get("analogs") or {}):
                    analog_hit = dict(_lib["analogs"][slug])
                    analog_hit["slug"] = slug
            except Exception:
                analog_hit = None
        if not analog_hit:
            analog_hit = match_analog(rt.get("analog_name", "") or slug)
        if analog_hit:
            rt["analog_return_pct"]    = analog_hit.get("return_pct")
            rt["analog_duration_months"] = analog_hit.get("duration_months")
            rt["analog_max_drawdown"]  = analog_hit.get("max_drawdown")
            rt["analog_hit_rate"]      = analog_hit.get("hit_rate")
            rt["analog_note"]          = analog_hit.get("note")
            rt["analog_period"]        = analog_hit.get("period")
            if not rt.get("analog_name"):
                rt["analog_name"] = analog_hit.get("name", "")
        merged["regime_template"] = rt
    except Exception:
        pass

    from datetime import datetime as _dt
    merged["generated_at"]     = _dt.utcnow().isoformat()
    merged["regime_snapshot"]  = {
        "hmm_state":        (regime_ctx or {}).get("hmm_state"),
        "ci_pct":           (regime_ctx or {}).get("ci_pct"),
        "macro_score":      (regime_ctx or {}).get("macro_score"),
        "contagion_score":  contagion_score,
        "events_sentiment": ((regime_ctx or {}).get("events_sentiment") or {}).get("sentiment"),
        "dominant_theme":   ((regime_ctx or {}).get("events_sentiment") or {}).get("dominant_theme"),
    }
    merged["n_runs"]           = len(runs)
    return merged

