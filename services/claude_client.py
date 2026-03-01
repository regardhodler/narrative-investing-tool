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
- "market_relevant": boolean — true if this topic could meaningfully move stock prices or sectors
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


def _empty_result() -> dict:
    return {
        "market_relevant": False,
        "sector": "N/A",
        "thesis": "",
        "suggested_tickers": [],
    }
