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

            # Dynamically map column headers to indices
            col_map = {}
            header_row = table.find("tr")
            if header_row:
                for idx, th in enumerate(header_row.find_all(["th", "td"])):
                    h = th.get_text(strip=True).lower()
                    if "politician" in h or "member" in h or "name" in h:
                        col_map["politician"] = idx
                    elif "date" in h and "published" not in h:
                        col_map["date"] = idx
                    elif "type" in h or "trade" in h or "transaction" in h:
                        col_map["type"] = idx
                    elif "size" in h or "amount" in h:
                        col_map["size"] = idx
                    elif "price" in h:
                        col_map["price"] = idx

            rows = table.find_all("tr")
            if len(rows) <= 1:
                break

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                # Look for ticker pattern like "AAPL:US" in the row text
                row_text = row.get_text()
                ticker_match = re.search(r"([A-Z]{1,5}):US", row_text)
                if not ticker_match or ticker_match.group(1) != ticker_upper:
                    continue

                # Extract politician from mapped or fallback index
                pol_idx = col_map.get("politician", 0)
                politician = cells[pol_idx].get_text(strip=True) if len(cells) > pol_idx else ""

                party = ""
                party_el = cells[pol_idx].find(class_=re.compile(r"party|badge")) if len(cells) > pol_idx else None
                if party_el:
                    party = party_el.get_text(strip=True)
                if not party:
                    row_classes = " ".join(row.get("class", []))
                    if "democrat" in row_classes.lower():
                        party = "D"
                    elif "republican" in row_classes.lower():
                        party = "R"

                # Use mapped columns with fallbacks
                date_idx = col_map.get("date", 2)
                type_idx = col_map.get("type", 3)
                size_idx = col_map.get("size", 4)
                price_idx = col_map.get("price", 5)

                trade_date = cells[date_idx].get_text(strip=True) if len(cells) > date_idx else ""
                trade_type = cells[type_idx].get_text(strip=True) if len(cells) > type_idx else ""
                trade_size = cells[size_idx].get_text(strip=True) if len(cells) > size_idx else ""
                trade_price = cells[price_idx].get_text(strip=True) if len(cells) > price_idx else ""

                # If type column didn't yield buy/sell, scan the full row text
                if trade_type and not re.search(r"(?i)buy|sell|purchase|sale|exchange", trade_type):
                    row_lower = row_text.lower()
                    if "buy" in row_lower or "purchase" in row_lower:
                        trade_type = "Buy"
                    elif "sell" in row_lower or "sale" in row_lower:
                        trade_type = "Sell"

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
