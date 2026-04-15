# Comprehensive HMM Backtest Results: The Reality Behind the "88% Accuracy" Claim

## Executive Summary

**SHOCKING FINDING**: When tested against the full HMM training dataset (2012-2026) instead of 16 cherry-picked crash dates, the proximity count method shows **0% accuracy** across all time horizons, completely contradicting the claimed 88% accuracy.

## Key Findings

### 1. **Cherry-Picked vs Full Dataset Performance**

| Metric | Cherry-Picked (16 dates) | Full Dataset (2,355 days) | Reality Check |
|--------|-------------------------|----------------------------|---------------|
| **Accuracy** | 88% (claimed) | **0.0%** (actual) | **-88 percentage points** |
| **Our Replication** | 37.5% (6/16 hits) | **0.0%** | Even cherry-picked fails |
| **Sample Size** | 16 handpicked dates | 2,355 trading days | **147x larger dataset** |
| **Time Span** | Specific crash moments | 9 years daily (2017-2026) | **Comprehensive coverage** |

### 2. **Signal Fire Rate Analysis**

**MASSIVE OVER-FIRING**: The proximity count method fires signals on **100% of trading days**, making it essentially useless as a discriminating tool.

- **TOP signals fired**: 2,355 days (100% of days)
- **BOTTOM signals fired**: 1,414 days (60% of days)  
- **No signal days**: 0 (complete saturation)
- **Daily signal frequency**: 365 signals per year (constantly firing)

### 3. **Signal Accuracy Breakdown**

**ZERO PREDICTIVE POWER** across all signal strengths and time horizons:

| Signal Strength | Days Fired | 5-Day Accuracy | 10-Day Accuracy | 20-Day Accuracy |
|----------------|------------|----------------|-----------------|-----------------|
| **+1 Edge** (any signal) | TOP: 2,355, BOT: 1,414 | **0.0%** | **0.0%** | **0.0%** |
| **+2 Edge** (moderate) | TOP: 2,099, BOT: 71 | **0.0%** | **0.0%** | **0.0%** |
| **+3 Edge** (strong) | TOP: 1,225, BOT: 12 | **0.0%** | **0.0%** | **0.0%** |

### 4. **Most Common False Positive Signals**

**TOP Signals** (constantly misfiring):
1. **"Low conviction"**: 98.4% of days (2,317/2,355)
2. **"High regime entropy"**: 75.0% of days (1,766/2,355)
3. **"HMM Stress state"**: 37.3% of days (878/2,355)
4. **"Regime elevated"**: 15.8% of days (373/2,355)

**BOTTOM Signals** (frequently misfiring):
1. **"Regime deep negative"**: 44.7% of days (1,053/2,355)
2. **"Macro crushed"**: 10.3% of days (242/2,355)
3. **"HMM Crisis state"**: 7.8% of days (183/2,355)

### 5. **Systematic vs Cherry-Picked Turning Points**

**Systematic Algorithm Found**:
- **20 market peaks** (5% decline threshold)
- **33 market troughs** (5% rally threshold)
- **325 days** within 30 days of peaks
- **422 days** within 30 days of troughs

**Signal Performance at Turning Points**:
- Peak signal rate: **100%** (fired on every peak approach)
- Trough signal rate: **61.4%** (fired on most trough approaches)

**Problem**: Signals fire constantly, not just at turning points, making them useless.

### 6. **Time Period Analysis**

**Consistent failure across all years**:

| Year | Trading Days | TOP Accuracy | BOTTOM Accuracy | Avg Regime Score |
|------|-------------|-------------|----------------|------------------|
| 2017 | 195 | **0.0%** | **0.0%** | -0.384 |
| 2018 | 261 | **0.0%** | **0.0%** | -0.075 |
| 2019 | 261 | **0.0%** | **0.0%** | -0.123 |
| 2020 | 262 | **0.0%** | **0.0%** | 0.312 |
| 2021 | 261 | **0.0%** | **0.0%** | -0.238 |
| 2022 | 261 | **0.0%** | **0.0%** | 0.098 |
| 2023 | 261 | **0.0%** | **0.0%** | -0.149 |
| 2024 | 261 | **0.0%** | **0.0%** | -0.207 |
| 2025 | 261 | **0.0%** | **0.0%** | -0.256 |
| 2026 | 72 | **0.0%** | **0.0%** | -0.230 |

## Root Cause Analysis

### 1. **Over-Calibrated Thresholds**

The signal thresholds were calibrated on cherry-picked crash dates, resulting in:
- **Extremely low thresholds** that fire constantly
- **No discrimination** between normal and crisis periods
- **Signal saturation** making the method useless

### 2. **Cherry-Picking Bias**

The original 88% accuracy was achieved by:
- **Selecting only 16 known crash dates** from 2000-2024
- **Post-hoc optimization** of thresholds on these specific dates
- **Ignoring thousands of normal trading days** where signals would misfire

### 3. **Lack of Base Rate Consideration**

True market turning points are **extremely rare events**:
- **20 peaks in 9 years** = 0.8% of trading days
- **33 troughs in 9 years** = 1.4% of trading days
- **Combined turning points** = ~2.2% of all trading days

Any signal that fires on 100% of days will have **massive false positive rates**.

### 4. **Flawed Signal Logic**

The proximity count method fails because:
- **"Low conviction" signal** fires on 98% of days (threshold too low)
- **"High entropy" signal** fires on 75% of days (always noisy)
- **HMM state signals** fire constantly during normal market stress
- **No dynamic threshold adjustment** for changing market conditions

## Recommendations

### 1. **Abandon the Current Method**

The proximity count method, as currently implemented, should be **completely abandoned** due to:
- **Zero predictive power** when tested on full dataset
- **100% false positive rate** (fires constantly)
- **No statistical significance** over random guessing

### 2. **Require Proper Backtesting**

Any future signal development must:
- **Test on full historical datasets**, not cherry-picked dates
- **Use proper train/test splits** to avoid overfitting
- **Report out-of-sample performance** only
- **Include false positive rates** in all metrics

### 3. **Implement Dynamic Thresholds**

If rebuilding, consider:
- **Adaptive thresholds** based on rolling statistics
- **Multi-timeframe confirmation** before firing signals
- **Regime-specific thresholds** (different for bull/bear/crisis periods)
- **Signal combination rules** requiring multiple independent confirmations

### 4. **Focus on Base Rates**

Any signal system must account for:
- **Low base rates** of true turning points (~2% of days)
- **High cost of false positives** in live trading
- **Asymmetric risk** (missing rallies vs avoiding crashes)

## Conclusion

**The "88% accuracy" claim for the proximity count method is fundamentally false.** When tested against 2,355 trading days instead of 16 cherry-picked crash dates, the method shows **zero predictive power** and fires signals constantly, making it worse than useless.

This is a textbook example of **overfitting to a small, biased sample** and highlights the critical importance of proper backtesting on comprehensive datasets. The method should be completely redesigned or abandoned.

**Bottom Line**: Don't trust any investment signal that hasn't been tested on thousands of out-of-sample trading days.

---

*Analysis completed: April 14, 2026*  
*Dataset: 2,355 trading days (2017-2026)*  
*Methodology: Full HMM training period, systematic turning point identification*