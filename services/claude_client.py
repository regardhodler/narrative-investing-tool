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


@st.cache_data(ttl=3600)
def summarize_filing(filing_text: str, form_type: str, company: str) -> str:
    """Summarize a SEC filing's text content via Groq.

    Returns a markdown-formatted summary string.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — cannot generate summary."

    prompt = f"""You are a financial analyst. Summarize this SEC {form_type} filing from {company}.

Focus on:
- Key announcements or material events (for 8-K)
- Financial highlights and performance (for 10-K/10-Q)
- Risk factors or notable disclosures
- Any forward-looking guidance

Keep the summary to 4-6 bullet points. Be specific with numbers and dates where available.

Filing text:
{filing_text}"""

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

Tickers:
{tickers_json}

For each narrative group, pick a short punchy title (e.g. "AI Infrastructure Boom", "Rate Cut Beneficiaries", "Energy Transition", "Momentum Leaders").

Return ONLY valid JSON (no markdown fences) — a list of objects:
[
  {{
    "narrative": "Theme Title",
    "description": "One sentence explaining why these stocks are grouped together",
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


@st.cache_data(ttl=3600)
def generate_valuation(ticker: str, signals_text: str) -> dict | None:
    """Generate an AI valuation and recommendation for a ticker via Groq.

    Returns dict with keys: rating, confidence, summary, bullish_factors,
    bearish_factors, key_levels, recommendation
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY environment variable not set.")
        return None

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

    # Strip markdown fences if present
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try to extract JSON object even if there's surrounding text
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
        # Validate required keys exist
        if "rating" not in result:
            return None
        return result
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=3600)
def suggest_regime_plays(regime: str, score: float, signal_summary: str) -> dict:
    """Suggest sectors, stocks, and bonds based on the current risk regime.

    Returns dict with keys: sectors, stocks, bonds, rationale
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"sectors": [], "stocks": [], "bonds": [], "rationale": ""}

    prompt = f"""You are a macro strategist. The market risk regime is currently **{regime}** with an aggregate score of {score:+.2f} (scale: -1 risk-off to +1 risk-on).

Key signal summary:
{signal_summary}

Based on this regime, suggest what to buy right now.

Return ONLY valid JSON (no markdown fences) with these keys:
- "sectors": list of 3-5 objects, each with "name" (sector name) and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "stocks": list of 4-6 objects, each with "ticker", "reason" (1 sentence), and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "bonds": list of 2-3 objects, each with "ticker", "reason", and "conviction" (integer 1-3, where 3 = strong buy, 2 = moderate buy, 1 = buy)
- "rationale": 2-3 sentence macro rationale for these picks given the current regime

Be selective with 3-star (strong buy) ratings — only give them to picks that are the best fit for this exact regime. Most picks should be 1 or 2 stars."""

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
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"sectors": [], "stocks": [], "bonds": [], "rationale": ""}

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"sectors": [], "stocks": [], "bonds": [], "rationale": ""}


def _empty_result() -> dict:
    return {
        "market_relevant": False,
        "sector": "N/A",
        "thesis": "",
        "suggested_tickers": [],
    }
