import re
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


@st.cache_data(ttl=3600)
def get_congress_trades(ticker: str) -> pd.DataFrame:
    """Scrape Capitol Trades for congress trading activity on a given ticker.

    Returns DataFrame: politician, party, date, type, size, price
    """
    ticker_upper = ticker.strip().upper()
    all_trades = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for page in range(1, 21):
        try:
            resp = requests.get(
                f"https://www.capitoltrades.com/trades?page={page}",
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                break

            rows = table.find_all("tr")
            if len(rows) <= 1:
                break

            found_on_page = False
            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 6:
                    continue

                # Look for ticker pattern like "AAPL:US" in the row text
                row_text = row.get_text()
                ticker_match = re.search(r"([A-Z]{1,5}):US", row_text)
                if not ticker_match or ticker_match.group(1) != ticker_upper:
                    continue

                found_on_page = True

                # Extract fields from cells
                politician = cells[0].get_text(strip=True)
                party = ""
                party_el = cells[0].find(class_=re.compile(r"party|badge"))
                if party_el:
                    party = party_el.get_text(strip=True)
                if not party:
                    # Try to infer from class names
                    row_classes = " ".join(row.get("class", []))
                    if "democrat" in row_classes.lower():
                        party = "D"
                    elif "republican" in row_classes.lower():
                        party = "R"

                trade_date = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                trade_type = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                trade_size = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                trade_price = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                all_trades.append({
                    "politician": politician,
                    "party": party,
                    "date": trade_date,
                    "type": trade_type,
                    "size": trade_size,
                    "price": trade_price,
                })

            # Stop paginating if no more content
            if not rows[1:]:
                break

        except Exception:
            break

    if not all_trades:
        return pd.DataFrame(columns=["politician", "party", "date", "type", "size", "price"])

    return pd.DataFrame(all_trades)
