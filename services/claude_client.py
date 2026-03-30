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
    resp = requests.post(
        XAI_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
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
            st.warning(f"Narrative grouping failed (xAI error): {e}")
            return []
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
    else:
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

    Returns dict with keys: rating, confidence, summary, bullish_factors,
    bearish_factors, key_levels, recommendation
    """
    import re

    from datetime import date as _date
    _today = _date.today().isoformat()
    _ce_block = (
        f"\nCURRENT EVENTS & MARKET INTEL:\n{current_events[:800]}\n"
        if current_events and len(current_events.strip()) > 20
        else "\nCURRENT EVENTS: None provided — base analysis on signals only.\n"
    )

    _tac_block = (f"\nTACTICAL REGIME (days-to-weeks entry timing): {tactical_context}\n"
                  if tactical_context else "")

    prompt = f"""You are an expert equity research analyst. Analysis date: {_today}. Based on the following data snapshot for {ticker}, provide a comprehensive valuation and recommendation.

{signals_text}{_ce_block}{_tac_block}

Return ONLY valid JSON (no markdown fences, no extra text) with these exact keys:
{{
  "rating": "Strong Buy" or "Buy" or "Hold" or "Sell" or "Strong Sell",
  "confidence": 0-100,
  "summary": "2-3 sentence thesis",
  "bullish_factors": ["factor1", "factor2", "factor3"],
  "bearish_factors": ["factor1", "factor2", "factor3"],
  "key_levels": {{"support": 123.45, "resistance": 234.56}},
  "recommendation": "2-3 sentence guidance"
}}"""

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

    _cl_model = model or "grok-4-1-fast-reasoning"
    _val_system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _parse_valuation_json(_call_xai(
                [{"role": "user", "content": prompt}], _cl_model, 1000, 0.1, system=_val_system))
        except Exception as e:
            st.error(f"xAI API error: {e}")
            return None
    elif use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=_cl_model,
                max_tokens=1000,
                temperature=0.1,
                system=_val_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_valuation_json(message.content[0].text.strip())
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
                    {"role": "system", "content": "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

    return _parse_valuation_json(text)


def suggest_regime_plays(regime: str, score: float, signal_summary: str, use_claude: bool = False, model: str = None) -> dict:
    """Suggest sectors, stocks, and bonds based on the current risk regime via Groq or Claude.

    Returns dict with keys: sectors, stocks, bonds, rationale
    """
    _empty = {"sectors": [], "stocks": [], "bonds": [], "rationale": ""}

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

    prompt = f"""You are a macro strategist. As of {_today}, the market risk regime is currently **{regime}** with an aggregate score of {score:+.2f} (scale: -1 risk-off to +1 risk-on).

Key signal summary:
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

    _system = "You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation."

    _cl_model = model or "grok-4-1-fast-reasoning"
    if use_claude and _is_xai_model(_cl_model) and os.getenv("XAI_API_KEY"):
        try:
            return _parse(_call_xai(
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
            return _parse(message.content[0].text.strip())
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
        return _parse(resp.json()["choices"][0]["message"]["content"].strip())
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

    prompt = f"""You are a macro strategist. A specific market scenario is unfolding:

Scenario: {scenario}

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
            # Auto-fallback to Claude if Groq is restricted/unavailable
            if resp.status_code == 400 and os.getenv("ANTHROPIC_API_KEY"):
                try:
                    import anthropic as _ac
                    _client = _ac.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    _msg = _client.messages.create(
                        model="grok-4-1-fast-reasoning",
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
        "- Be clinical and specific — no vague generalities\n\n"
        f"SIGNALS:\n{signals_text[:4000]}"
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
            sz_str = f" [Wt:{sz['weight']}% Score:{sz['score']} RegimeFit:{sz.get('regime_fit')}]"
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

MACRO ENVIRONMENT:
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
{fe_block}{_pr_block}

BLACK SWAN RISKS:
{swan_block}

OPEN POSITIONS (with portfolio weight, sizing score, regime fit where available):
{positions_block}

For each position assess regime alignment, rate path sensitivity, black swan exposure, and factor concentration.

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
            return _json.loads(m.group())
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
            return _json.loads(m.group())
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
        return json.loads(m.group()) if m else None
    except Exception:
        return None


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
