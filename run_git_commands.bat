@echo off
cd /d "C:\Users\16476\claude projects\narrative-investing-tool"

echo === Git Status ===
git status
echo.

echo === Git Add ===
git add -A
echo.

echo === Git Commit ===
git commit -m "Add Elliott Wave & Wyckoff module upgrades

- Elliott Wave: intraday intervals, TradingView-style chart, EWF Fibonacci scoring
- Elliott Wave: ABC corrective wave detection and overlays
- Elliott Wave: Wave Forecast panel with zigzag projection and probability
- Elliott Wave: Fibonacci score bar with context, backtest engine
- Elliott Wave: title/legend layout fix, direction bug fix
- Wyckoff: ticker + interval selectors (any ticker, 1d/1wk/1mo)
- Wyckoff: sub-phase A-E detection per Accumulation/Distribution
- Wyckoff: Volume Spread Analysis (VSA) with chart annotations
- Wyckoff: Cause & Effect price targets on chart
- Wyckoff: demand/supply trendlines on chart
- Wyckoff: effort vs result divergence detection
- Wyckoff: trade setup panel (entry zone, stop, target, R:R)
- syntax_check.py: fixed UTF-8 encoding for file reads

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
echo.

echo === Git Push ===
git push
echo.

echo === Done ===
