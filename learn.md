# Module 0: Macro Dashboard — Learning Guide

## What This Module Does

Module 0 is a **daily macro regime indicator**. It ingests 15 cross-asset signals, scores them on a -1 to +1 scale, and outputs:

- A **Risk-On / Neutral / Risk-Off** verdict
- A **Macro Score** (0–100)
- A **Ray Dalio quadrant** classification
- **Portfolio bias** recommendations
- **SPY gamma positioning** from options flow

The goal: answer "should I be adding risk or reducing risk today?" before looking at any individual stock.

---

## The 15 Indicators Explained

### 1. Yield Curve (10Y – 2Y Treasury Spread)
- **Source:** FRED `T10Y2Y`
- **Logic:** Positive spread = normal curve = growth expectations intact (Risk-On). Inverted = recession signal (Risk-Off).
- **Why it matters:** The yield curve has predicted every US recession since the 1960s. When it un-inverts after inversion, that's often when the recession actually starts.

### 2. Credit Spreads (High Yield vs Treasuries)
- **Source:** FRED `BAMLH0A0HYM2` (ICE BofA HY OAS)
- **Logic:** Tight spreads (<350bps) = credit market calm (Risk-On). Widening spreads = stress (Risk-Off).
- **Why it matters:** Credit markets are the "smart money" — they price default risk before equity markets react. Spread blowouts precede equity selloffs.

### 3. Commodity Trend (Oil + Copper)
- **Source:** yfinance `USO` (WTI Oil) and `CPER` (Copper)
- **Logic:** Rising oil + copper = real economy demand (Risk-On). Falling = slowdown (Risk-Off).
- **Why it matters:** "Dr. Copper" has a PhD in economics — copper demand tracks global industrial activity. Oil tracks both supply dynamics and growth expectations.

### 4. US Dollar Index (DXY)
- **Source:** yfinance `UUP` (USD Bull ETF as proxy)
- **Logic:** Dollar falling = global liquidity expanding, EM breathing room (Risk-On). Dollar rising = tightening financial conditions globally (Risk-Off).
- **Why it matters:** The dollar is the world's reserve currency. A strong dollar is a de facto tightening of global financial conditions because ~60% of global trade is dollar-denominated.

### 5. Global Liquidity (M2 Money Supply)
- **Source:** FRED `M2SL`
- **Logic:** M2 growing > 2% YoY = expanding liquidity (Risk-On). Contracting = quantitative tightening effect (Risk-Off).
- **Why it matters:** "Don't fight the Fed" — when central banks inject liquidity, asset prices tend to rise regardless of fundamentals. M2 is the broadest measure of money in the system.

### 6. Unemployment Trend (Sahm Rule)
- **Source:** FRED `SAHMREALTIME` or derived from `UNRATE`
- **Logic:** Sahm Rule < 0.4pp = labor market stable (Risk-On). Above 0.5pp = recession has likely begun (Risk-Off).
- **Why it matters:** Claudia Sahm's rule has a perfect track record: when the 3-month average unemployment rate rises 0.5pp above its 12-month low, a recession is underway.

### 7. Core Inflation (PCE)
- **Source:** FRED `PCEPILFE` (Core PCE Price Index)
- **Logic:** Core PCE near 2% = Fed can ease (Risk-On). Above 3%+ = Fed must tighten (Risk-Off).
- **Why it matters:** The Fed targets Core PCE, not CPI. It excludes food and energy volatility, giving a cleaner read on underlying price pressures that drive monetary policy.

### 8. Equity Trend (S&P 500, Nasdaq, Dow Jones)
- **Source:** yfinance `SPY`, `QQQ`, `^DJI`
- **Logic:** Average position above 200-day MA = secular bull intact (Risk-On). Below = trend broken (Risk-Off).
- **Why it matters:** Price is the ultimate arbiter. The 200-day moving average is the most widely followed institutional trend signal. When all three indices are above it, breadth confirms the uptrend.

### 9. Shiller CAPE Ratio
- **Source:** FRED `CAPE`
- **Logic:** CAPE < 20 = cheap (Risk-On). CAPE > 30 = expensive, lower forward returns (Risk-Off tilt).
- **Why it matters:** Robert Shiller's cyclically-adjusted P/E uses 10 years of earnings to smooth the cycle. It has strong predictive power for 10-year forward returns (high CAPE = low future returns), though it says nothing about timing.

### 10. Buffett Indicator (Market Cap / GDP)
- **Source:** FRED `WILL5000INDFC` / `GDP`
- **Logic:** Below 110% = reasonable (Risk-On). Above 150% = stretched (Risk-Off tilt).
- **Why it matters:** Warren Buffett called this "probably the best single measure of where valuations stand at any given moment." It compares total market capitalization to economic output.

### 11. Corporate CAPEX vs Global Liquidity
- **Source:** FRED `NCBDBIQ027S` (Nonfinancial Corporate CAPEX) vs `M2SL`
- **Logic:** CAPEX growing faster than M2 = real investment cycle (healthy). M2 growing faster than CAPEX = asset inflation without productive investment (fragile).
- **Why it matters:** This distinguishes between productive growth (companies investing in capacity) vs financial engineering (buybacks fueled by cheap money). The former is sustainable; the latter is fragile.

### 12. Gamma Exposure (Dealer Positioning)
- **Source:** yfinance options chain for SPY
- **Logic:** Positive gamma = dealers hedge by selling into rallies and buying dips (stabilizing, Risk-On). Negative gamma = dealers amplify moves (destabilizing, Risk-Off).
- **Why it matters:** Options dealers are forced hedgers. Their gamma positioning creates a mechanical force that either suppresses or amplifies volatility. This is the structural plumbing of the market.

### 13. Term Premium
- **Source:** FRED `THREEFYTP10`
- **Logic:** Positive term premium = investors demand compensation for duration risk (normal). Negative = yield curve distortion, often from central bank intervention.
- **Why it matters:** Term premium is what bond investors demand beyond expected short rates. When it's compressed or negative, it suggests either heavy central bank buying or extreme demand for safety.

### 14. ISM Manufacturing Index
- **Source:** FRED `NAPM`
- **Logic:** Above 50 = manufacturing expansion (Risk-On). Below 50 = contraction (Risk-Off).
- **Why it matters:** ISM is one of the oldest and most reliable leading indicators. It surveys purchasing managers who see orders before they hit revenue. Three consecutive months below 50 historically signals recession risk.

### 15. Financial Conditions Index (FCI)
- **Source:** FRED `NFCI` (Chicago Fed National Financial Conditions Index)
- **Logic:** Negative NFCI = loose conditions (Risk-On). Positive = tight conditions (Risk-Off).
- **Why it matters:** The NFCI aggregates 105 financial indicators across money markets, debt, equity, and the banking system. It's the Fed's own measure of whether financial conditions are restrictive or accommodative.

---

## Ray Dalio Macro Quadrant

The module maps the current environment to one of four quadrants based on growth and inflation direction:

| Quadrant | Growth | Inflation | What performs well |
|------------|---------|-----------|---------------------|
| **Goldilocks** | Rising | Falling | Equities (especially growth), credit, moderate duration bonds |
| **Reflation** | Rising | Rising | Commodities, value stocks, TIPS, EM equities, short duration |
| **Stagflation** | Falling | Rising | Gold, cash, real assets, defensive equity, commodity producers |
| **Deflation** | Falling | Falling | Long-duration Treasuries, quality equities, USD, cash |

**How it's computed:**
- Growth direction: average of yield curve, equity trend, ISM, and unemployment scores
- Inflation direction: 3-month change in Core PCE + commodity momentum

This is inspired by Ray Dalio's "All Weather" framework, which argues that asset performance is primarily driven by whether growth and inflation are above or below expectations.

---

## SPY Gamma Positioning

### What is Gamma?
Gamma measures how much an option's delta changes as the underlying price moves. For market makers (dealers), gamma exposure determines whether they stabilize or destabilize the market.

### Positive vs Negative Gamma
- **Positive Gamma Zone:** Dealers are long gamma. As SPY rises, they sell (hedging). As it falls, they buy. This creates a "pin" effect — low volatility, mean-reverting.
- **Negative Gamma Zone:** Dealers are short gamma. As SPY rises, they must buy more (chasing). As it falls, they must sell more. This amplifies moves — high volatility, trending.

### Key Levels
- **Gamma Flip:** The strike price where dealer gamma switches from positive to negative. Above this level = stability; below = turbulence.
- **Call Wall:** Strike with highest call open interest. Acts as magnetic resistance — market tends to get pulled toward it.
- **Put Wall:** Strike with highest put open interest. Acts as magnetic support — dealers hedge by buying here.

### How to use it
- If SPY is **above** the Gamma Flip: expect range-bound, low-vol behavior. Selling premium is favored.
- If SPY is **below** the Gamma Flip: expect directional moves. Buying protective puts or trend-following is favored.
- Between Put Wall and Call Wall: the market's "expected range" for the near term.

---

## Macro Score Interpretation

| Score | Regime | Implication |
|-------|--------|-------------|
| 70–100 | Strong Risk-On | Broad risk appetite, overweight growth assets |
| 60–69 | Mild Risk-On | Constructive but watch for divergences |
| 41–59 | Neutral | Mixed signals, stay balanced |
| 31–40 | Mild Risk-Off | Caution warranted, reduce high-beta exposure |
| 0–30 | Strong Risk-Off | Defensive posture: Treasuries, gold, cash |

The score is computed as: `(average_signal_score + 1.0) * 50`, mapping the -1 to +1 range onto 0–100.

---

## Cycle Stage (CAPEX vs Liquidity)

| Stage | CAPEX vs M2 | What it means |
|-------|-------------|---------------|
| **Capex-led expansion** | > +2pp | Companies investing in real capacity. Sustainable growth. |
| **Balanced mid-cycle** | -2 to +2pp | Neither over-investing nor under-investing. Normal. |
| **Liquidity-led / capex slowdown** | < -2pp | Money is growing but companies aren't investing. Asset inflation without real growth. Fragile. |

---

## Data Sources & Freshness

| Source | Update frequency | Staleness threshold |
|--------|-----------------|---------------------|
| yfinance (ETFs) | Daily | > 3 days old = stale |
| FRED macro series | Monthly/quarterly | Varies by series (14–120 days) |
| Truflation | Daily | Real-time |
| SPY options chain | Intraday | 3-hour cache TTL |

The confidence score (25–95%) for each indicator reflects data freshness. FRED series that haven't updated in weeks get lower confidence, which is surfaced in the dashboard.

---

## Architecture Overview

```
render()
  |
  +-- fetch_all_data()              # yfinance batch fetch (22 tickers, 1yr daily)
  |     cached 1hr
  |
  +-- _build_macro_dashboard()
        |
        +-- fetch_fred_series()     # 13 FRED series (cached 6hr, fallback to disk)
        |
        +-- _compute_spy_gamma_mode()  # Options chain snapshot (cached 3hr)
        |
        +-- Score each of 15 indicators on [-1, +1]
        |
        +-- Derive: macro_score, quadrant, cycle_stage, portfolio_bias, summary
```

All heavy data fetches are cached via `@st.cache_data`. FRED fetches have a two-tier retry with local CSV fallback for resilience.

---

## Key Mental Models for Daily Use

1. **Macro trumps micro.** A stock can have great fundamentals and still fall 30% in a Risk-Off environment. Check Module 0 before Module 7.

2. **Watch for divergences.** When the macro score is neutral but individual signals are at extremes in opposite directions, expect a regime change soon.

3. **FRED data lags.** ISM, unemployment, PCE — these are monthly/quarterly. The ETF-based signals (VIX, credit ETFs, equity momentum) update daily. Weight recent market-implied signals more heavily at turning points.

4. **Gamma is structural, not directional.** Positive gamma doesn't mean "market goes up." It means "moves will be dampened." Negative gamma doesn't mean "market goes down." It means "moves will be amplified."

5. **The quadrant is a compass, not a GPS.** It tells you the direction of travel (Goldilocks vs Stagflation), not precise positioning. Use it to tilt allocations, not to make binary bets.

---

## Engineering Notes

### Dual-Engine Legacy
The file currently contains two scoring engines:
- **Engine 1 (Legacy):** `SIGNAL_DEFS` + `compute_regime()` — declarative z-score system using ETF proxies only
- **Engine 2 (Current):** `_build_macro_dashboard()` — FRED + ETF hybrid with 15 indicators

Only Engine 2 is rendered in the dashboard. Engine 1 remains for the `get_current_regime()` public API consumed by other modules.

### Low Compute Mode
A toggle (default ON) that:
- Reduces options chain fetches to 1 expiry instead of 2
- Disables the gamma chart rendering
- Still shows textual gamma metrics (price, zone, flip, walls)

### Confidence Scoring
Each indicator carries a confidence % (25–95) based on:
- **ETF-based signals:** freshness of yfinance data (stale if >3 days old)
- **FRED-based signals:** age of last data point vs expected release cadence
- **Composite signals:** blended confidence from constituent sources
