import requests
import xml.etree.ElementTree as ET
import time

headers = {'User-Agent': 'NarrativeInvestingTool research@example.com'}

# Get AAPL submissions
r = requests.get('https://data.sec.gov/submissions/CIK0000320193.json', headers=headers, timeout=15)
data = r.json()
recent = data['filings']['recent']

count = 0
for i in range(len(recent['form'])):
    if recent['form'][i] != '4':
        continue

    acc = recent['accessionNumber'][i]
    doc = recent['primaryDocument'][i]

    # Strip the XSL prefix to get raw XML filename
    raw_doc = doc.split('/')[-1] if '/' in doc else doc
    acc_nd = acc.replace('-', '')

    # Get index to find raw XML
    idx_url = f'https://www.sec.gov/Archives/edgar/data/320193/{acc_nd}/index.json'
    time.sleep(0.15)
    r2 = requests.get(idx_url, headers=headers, timeout=15)
    if r2.status_code != 200:
        continue

    items = r2.json().get('directory', {}).get('item', [])
    xml_file = None
    for item in items:
        name = item['name']
        if name.endswith('.xml') and name != 'R1.htm':
            xml_file = name
            break

    if not xml_file:
        continue

    raw_url = f'https://www.sec.gov/Archives/edgar/data/320193/{acc_nd}/{xml_file}'
    time.sleep(0.15)
    r3 = requests.get(raw_url, headers=headers, timeout=15)

    root = ET.fromstring(r3.text)
    ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
    p = '{' + ns + '}' if ns else ''

    owner = root.find(f'.//{p}rptOwnerName')
    title = root.find(f'.//{p}officerTitle')

    txns = root.findall(f'.//{p}nonDerivativeTransaction')
    if not txns:
        continue

    print(f'=== {owner.text if owner is not None else "?"} ({title.text if title is not None else "Director/Other"}) ===')
    print(f'  Filing date: {recent["filingDate"][i]}')

    for txn in txns:
        date_el = txn.find(f'.//{p}transactionDate/{p}value')
        code_el = txn.find(f'.//{p}transactionCode')
        shares_el = txn.find(f'.//{p}transactionShares/{p}value')
        price_el = txn.find(f'.//{p}transactionPricePerShare/{p}value')
        acq_el = txn.find(f'.//{p}transactionAcquiredDisposedCode/{p}value')
        d = date_el.text if date_el is not None else "?"
        c = code_el.text if code_el is not None else "?"
        s = shares_el.text if shares_el is not None else "?"
        pr = price_el.text if price_el is not None else "?"
        a = acq_el.text if acq_el is not None else "?"
        action = "BUY" if a == "A" and c in ("P", "A") else "SELL" if a == "D" and c == "S" else f"{c}/{a}"
        print(f'  {d} | {action} | {float(s):,.0f} shares @ ${float(pr) if pr != "?" else 0:,.2f}')

    count += 1
    if count >= 5:
        break
