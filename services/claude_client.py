import json
import os
import requests
import streamlit as st

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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


def group_tickers_by_narrative(tickers_json: str) -> list[dict]:
    """Group a list of trending tickers into narrative themes via Groq.

    Input: JSON string of [{symbol, name}, ...]
    Returns list of dicts: [{narrative, description, tickers: [{symbol, name}]}, ...]
    """
    result = _group_tickers_cached(tickers_json)
    if not result:
        # Clear only this function's cache so failures don't stick for 1hr
        _group_tickers_cached.clear()
    return result


@st.cache_data(ttl=3600)
def _group_tickers_cached(tickers_json: str) -> list[dict]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.warning("GROQ_API_KEY not set — cannot group tickers by narrative.")
        return []

    prompt = f"""You are a financial analyst. Given these trending tickers, group them into 3-6 investment narrative themes.

The list may include stocks, commodity futures (tickers ending in =F), bond ETFs, and currency ETFs. Create cross-asset themes where appropriate (e.g. "Inflation Hedge" grouping gold futures with TIPS ETFs, or "Risk-Off Flight" grouping treasuries with yen).

Tickers:
{tickers_json}

For each narrative group, pick a short punchy title (e.g. "AI Infrastructure Boom", "Rate Cut Beneficiaries", "Energy Transition", "Commodity Super-Cycle", "Safe Haven Rotation").

Return ONLY valid JSON (no markdown fences) — a list of objects:
[
  {{
    "narrative": "Theme Title",
    "description": "One sentence explaining why these assets are grouped together",
    "tickers": ["SYM1", "SYM2"]
  }}
]

Rules:
- Every ticker must appear in exactly one group
- If a ticker doesn't fit a clear theme, put it in a "Market Movers" catch-all group
- Sort groups so the strongest/most actionable narrative is first"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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


def generate_valuation(ticker: str, signals_text: str, use_claude: bool = False, model: str = None) -> dict | None:
    """Generate an AI valuation and recommendation for a ticker via Groq or Claude.

    Returns dict with keys: rating, confidence, summary, bullish_factors,
    bearish_factors, key_levels, recommendation
    """
    import re

    prompt = f"""You are an expert equity research analyst. Based on the following data snapshot for {ticker}, provide a comprehensive valuation and recommendation.

{signals_text}

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

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=1000,
                temperature=0.1,
                system="You are a JSON-only response bot. Return only valid JSON, no markdown fences, no explanation.",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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

    prompt = f"""You are a macro strategist. The market risk regime is currently **{regime}** with an aggregate score of {score:+.2f} (scale: -1 risk-off to +1 risk-on).

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

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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


@st.cache_data(ttl=3600)
def summarize_whale_activity(activity_text: str) -> str:
    """Summarize whale 13F activity into a narrative via Groq.

    Returns a markdown-formatted narrative string about big money themes this quarter.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate whale activity summary."

    prompt = f"""You are a top-tier institutional equity analyst. Analyze these quarterly 13F whale position changes and write a concise narrative summary.

Focus on:
- What themes or sectors are the biggest hedge funds and institutions converging on?
- Which names are seeing the most conviction (multiple whales buying)?
- Are there notable exits or position closures that signal a shift?
- What does the overall flow pattern suggest about institutional sentiment?

Keep the summary to 4-6 bullet points. Be specific about names, dollar amounts, and the whales involved.

Whale activity data:
{activity_text}"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating whale summary: {e}"

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def generate_doom_briefing(stress_data: str, use_claude: bool = False, model: str = None) -> str:
    """Generate an ominous risk intelligence briefing from stress signal data via Groq or Claude.

    Returns a markdown-formatted doom briefing string.
    """
    prompt = f"""You are a doomsday-focused financial risk analyst. Your job is to be the canary in the coal mine — flagging systemic risks that mainstream analysts ignore.

Analyze the following stress signal data and write a dramatic but data-driven risk assessment briefing.

Rules:
- Be direct about risks. Do NOT sugarcoat. If something looks bad, say it looks bad.
- Flag any systemic concerns — contagion risks, interconnected failures, cascading defaults.
- Rate overall financial system stress from 1-10 (1 = calm seas, 10 = 2008-level crisis).
- Use ominous, urgent language but remain factual and professional.
- Structure as: STRESS LEVEL rating, then 4-6 bullet points on the most concerning signals, then a 2-3 sentence closing assessment.
- If data is limited or unavailable, note what you cannot assess and why that itself is concerning.

Stress Signal Data:
{stress_data}"""

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.4,
            },
            timeout=30,
        )
        resp.raise_for_status()
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
    prompt = (
        "You are a macro policy analyst. Based on these Fed policy probability scenarios and "
        "causal transmission chains, write exactly 3-4 sentences explaining: "
        "(1) the most likely rate path and its probability, "
        "(2) how that path transmits through credit markets, housing, and employment, "
        "(3) which specific asset classes benefit and which face headwinds — name tickers or sectors. "
        "Be precise and specific. No vague generalities.\n\n"
        f"Rate path probabilities: {adj_probs_json}\n"
        f"Transmission chains: {chains_json}"
    )

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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


def assess_macro_fit(
    ticker: str,
    company_name: str,
    sector: str,
    price_summary: str,
    regime_context: str,
    rate_path_context: str,
    black_swan_context: str,
    use_claude: bool = False,
    model: str | None = None,
) -> dict | None:
    """Evaluate how well a ticker fits the current macro regime environment."""
    prompt = f"""You are a macro-regime portfolio analyst. Evaluate how well {ticker} ({company_name}, {sector}) fits the CURRENT macro environment.

CURRENT MACRO ENVIRONMENT:
- Regime Context: {regime_context}
- Fed Rate Path: {rate_path_context}
- Black Swan Tail Risks: {black_swan_context if black_swan_context else "None analyzed"}
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
        if use_claude and model:
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
