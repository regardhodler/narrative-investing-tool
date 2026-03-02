import json
import re

import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=3600)
def get_congress_trades(ticker: str) -> tuple[pd.DataFrame, str]:
    """Scrape Quiver Quant for congress trading activity on a given ticker.

    Extracts embedded Plotly trace data which contains all historical
    congress trades with politician names, dates, types, sizes, and prices.

    Returns (DataFrame, error_message). DataFrame cols: politician, date, type, size, price
    """
    ticker_upper = ticker.strip().upper()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        resp = requests.get(
            f"https://www.quiverquant.com/congresstrading/stock/{ticker_upper}",
            headers=headers,
            timeout=20,
        )
        if resp.status_code == 403:
            return _empty_df(), "Quiver Quant blocked the request (403 Forbidden). This typically happens from cloud-hosted servers."
        if resp.status_code != 200:
            return _empty_df(), f"Quiver Quant returned HTTP {resp.status_code}."
    except requests.exceptions.ConnectionError:
        return _empty_df(), "Could not connect to Quiver Quant. The site may be blocking cloud IPs."
    except requests.exceptions.Timeout:
        return _empty_df(), "Quiver Quant request timed out."
    except Exception as e:
        return _empty_df(), f"Request failed: {e}"

    html = resp.text

    # Extract Plotly traces from Plotly.newPlot() call
    m = re.search(
        r'Plotly\.newPlot\(\s*["\x27][\w-]+["\x27],\s*(\[.+?\])\s*,\s*\{',
        html,
        re.DOTALL,
    )
    if not m:
        if "captcha" in html.lower() or "cloudflare" in html.lower():
            return _empty_df(), "Quiver Quant is showing a CAPTCHA/Cloudflare challenge — site blocks automated requests from cloud servers."
        return _empty_df(), "Could not find trade data in Quiver Quant page. Site format may have changed."

    try:
        traces = json.loads(m.group(1))
    except json.JSONDecodeError:
        return _empty_df(), "Failed to parse Quiver Quant trade data."

    trades = []
    for trace in traces:
        name = trace.get("name", "").lower()
        if "sale" in name:
            trade_type = "Sale"
        elif "purchase" in name:
            trade_type = "Purchase"
        else:
            continue

        dates = trace.get("x", [])
        prices = trace.get("y", [])
        # Politician names may be in customdata or parsed from text
        customdata = trace.get("customdata", [])
        text_list = trace.get("text", [])
        hovertext = trace.get("hovertext", [])

        for i in range(len(dates)):
            trade_date = str(dates[i]).split("T")[0] if i < len(dates) else ""
            price = prices[i] if i < len(prices) else ""

            # Get politician name — try customdata first, then parse from text
            politician = ""
            if customdata and i < len(customdata):
                p = customdata[i]
                if isinstance(p, list):
                    p = p[0] if p else ""
                politician = str(p).strip()

            if not politician and text_list and i < len(text_list):
                t = text_list[i]
                if isinstance(t, list):
                    t = t[0] if t else ""
                pm = re.search(r"Politician</b>:\s*(.+?)(?:<br>|<extra>)", str(t))
                if pm:
                    politician = pm.group(1).strip()

            # Get size from hovertext or text
            size = ""
            if hovertext and i < len(hovertext):
                size = str(hovertext[i]).strip()
            if not size and text_list and i < len(text_list):
                t = text_list[i]
                if isinstance(t, list):
                    t = t[0] if t else ""
                sm = re.search(r"Transaction Size</b>:\s*(.+?)(?:<br>|<extra>)", str(t))
                if sm:
                    size = sm.group(1).strip()

            if politician and trade_date:
                trades.append({
                    "politician": politician,
                    "date": trade_date,
                    "type": trade_type,
                    "size": size,
                    "price": f"${price:,.2f}" if isinstance(price, (int, float)) else str(price),
                })

    if not trades:
        return _empty_df(), ""

    df = pd.DataFrame(trades)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df, ""


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["politician", "date", "type", "size", "price"])
