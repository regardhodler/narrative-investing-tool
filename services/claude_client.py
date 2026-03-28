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


def generate_valuation(ticker: str, signals_text: str, use_claude: bool = False, model: str = None, current_events: str = "") -> dict | None:
    """Generate an AI valuation and recommendation for a ticker via Groq or Claude.

    Returns dict with keys: rating, confidence, summary, bullish_factors,
    bearish_factors, key_levels, recommendation
    """
    import re

    _ce_block = f"\nCURRENT EVENTS & MARKET INTEL:\n{current_events[:800]}\n" if current_events else ""

    prompt = f"""You are an expert equity research analyst. Based on the following data snapshot for {ticker}, provide a comprehensive valuation and recommendation.

{signals_text}{_ce_block}

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


def summarize_whale_activity(activity_text: str, use_claude: bool = False, model: str = None) -> str:
    """Summarize whale 13F activity into a narrative via Groq or Claude.

    Returns a markdown-formatted narrative string about big money themes this quarter.
    """
    prompt = f"""You are a top-tier institutional equity analyst. Analyze these quarterly 13F whale position changes and write a concise narrative summary.

Focus on:
- What themes or sectors are the biggest hedge funds and institutions converging on?
- Which names are seeing the most conviction (multiple whales buying)?
- Are there notable exits or position closures that signal a shift?
- What does the overall flow pattern suggest about institutional sentiment?

Keep the summary to 4-6 bullet points. Be specific about names, dollar amounts, and the whales involved.

Whale activity data:
{activity_text}"""

    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=800,
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
    _stress_truncated = stress_data[:6000] + "\n[...truncated]" if len(stress_data) > 6000 else stress_data
    if current_events:
        _stress_truncated += f"\n\nCURRENT EVENTS CONTEXT:\n{current_events[:1500]}"

    prompt = f"""You are a senior sell-side risk strategist writing for institutional allocators.

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
        if not resp.ok:
            # Auto-fallback to Claude if Groq is restricted/unavailable
            if resp.status_code == 400 and os.getenv("ANTHROPIC_API_KEY"):
                try:
                    import anthropic as _ac
                    _client = _ac.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    _msg = _client.messages.create(
                        model="claude-haiku-4-5-20251001",
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
- Cross-Signal Discovery Plays: {discovery_plays_str}{ce_block}

PORTFOLIO FACTOR EXPOSURE (weighted aggregate):
{fe_block}

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
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
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
    _model = model or "claude-haiku-4-5-20251001"
    _max_tokens = 1000

    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
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
    _model = model or "claude-haiku-4-5-20251001"
    _max_tokens = 3000 if (_model and "sonnet" in _model) else 2000

    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
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
    tf_label = "past 7 days" if timeframe == "1W" else "past 30 days"

    headlines_text = "\n".join(f"- {h}" for h in headlines[:40])
    trends_text = "\n".join(f"- {t}" for t in trends[:30]) if trends else "- (unavailable)"

    prompt = f"""You are a macro narrative analyst. Based on the data below, identify the TOP 5 EMERGING INVESTMENT NARRATIVES of the {tf_label}.

MACRO CONTEXT:
- Regime: {regime} (score {score:+.2f})
- Quadrant: {quadrant}

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
    _model = model or "claude-haiku-4-5-20251001"

    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
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
