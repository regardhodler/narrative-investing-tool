# Narrative Investing Intelligence Tool

Spot market themes before they're priced in. Connects trending narratives → Google Trends momentum → SEC filing exposure → institutional ownership → options flow.

## Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run:
```bash
streamlit run app.py
```

## IBKR Setup (Module 5 — Options Activity)

1. **Install TWS or IB Gateway** from [interactivebrokers.com](https://www.interactivebrokers.com/en/trading/tws.php)
2. **Enable API access**: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Socket port: `7497` (paper) or `7496` (live)
   - Uncheck "Read-Only API"
3. **Login** to TWS/Gateway before connecting from the app
4. Select the matching port in the Options Activity module and click Connect

## Modules

| # | Module | Data Source |
|---|--------|-------------|
| 1 | Narrative Discovery | Google Trends + Claude classification |
| 2 | Narrative Pulse | Google Trends 90-day time series |
| 3 | EDGAR Scanner | SEC EDGAR full-text search |
| 4 | Smart Money Tracker | SEC 13F institutional filings |
| 5 | Options Activity | IBKR options chain via ib_insync |
