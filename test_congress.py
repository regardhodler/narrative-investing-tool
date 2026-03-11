import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# Scrape multiple pages and collect all trades, filter by ticker
all_trades = []
for page in range(1, 20):
    url = f'https://www.capitoltrades.com/trades?page={page}'
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code != 200:
        break

    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table')
    if not table:
        break

    rows = table.find_all('tr')[1:]  # Skip header
    if not rows:
        break

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 8:
            continue

        # Extract issuer cell which contains ticker
        issuer_cell = cells[1].get_text(strip=True)
        politician_cell = cells[0].get_text(strip=True)
        published = cells[2].get_text(strip=True)
        traded = cells[3].get_text(strip=True)
        owner = cells[5].get_text(strip=True)
        trade_type = cells[6].get_text(strip=True)
        size = cells[7].get_text(strip=True)
        price = cells[8].get_text(strip=True) if len(cells) > 8 else ''

        all_trades.append({
            'politician': politician_cell,
            'issuer': issuer_cell,
            'published': published,
            'traded': traded,
            'owner': owner,
            'type': trade_type,
            'size': size,
            'price': price,
        })

    print(f'Page {page}: {len(rows)} rows')

print(f'\nTotal trades scraped: {len(all_trades)}')

# Filter for AAPL
aapl_trades = [t for t in all_trades if 'AAPL' in t['issuer'].upper()]
print(f'AAPL trades found: {len(aapl_trades)}')
for t in aapl_trades[:10]:
    print(f"  {t['politician'][:30]} | {t['type']} | {t['size']} | {t['traded']} | {t['price']}")

# Also check TSLA
tsla_trades = [t for t in all_trades if 'TSLA' in t['issuer'].upper()]
print(f'\nTSLA trades found: {len(tsla_trades)}')
for t in tsla_trades[:10]:
    print(f"  {t['politician'][:30]} | {t['type']} | {t['size']} | {t['traded']} | {t['price']}")

# Show unique tickers
import re
tickers = set()
for t in all_trades:
    m = re.search(r'([A-Z]{1,5}):US', t['issuer'])
    if m:
        tickers.add(m.group(1))
print(f'\nUnique tickers in scraped data: {len(tickers)}')
print(sorted(tickers)[:30])
