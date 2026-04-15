# Validation Framework Methodology

## Quick Start Guide

### Running the Framework
```bash
python validation_framework.py
```

This will:
1. Load HMM daily data (2017-2026) 
2. Identify 16 known turning points
3. Validate all defined signals across 3 periods
4. Generate comprehensive report and baseline results

### Understanding the Output

#### Signal Quality Assessment
- **EXCELLENT**: >50% precision, <5% false positive rate → Deploy immediately
- **MODERATE**: 30-50% precision, 5-15% FPR → Tune thresholds  
- **POOR**: <30% precision, >15% FPR → Disable or redesign

#### Key Metrics to Watch
- **False Positive Rate (FPR)**: Most important for live trading
- **Precision**: When signal fires, how often is it right?
- **Crisis Specificity**: How much more likely to fire in crisis vs normal

### Adding New Signals

1. **Define the signal** in `signal_definitions`:
```python
"New_signal": {
    "description": "Description of what it measures",
    "signal_key": "keyword to find in top/bottom signals",
    "threshold_direction": "contains",  # or "greater", "less"
    "baseline_expected": 50.0  # expected normal market level
}
```

2. **Run validation** - the framework automatically tests it
3. **Review results** - check all metrics, not just accuracy
4. **Only deploy if EXCELLENT** - no exceptions

### Validation Periods

The framework uses 3 walk-forward periods:

1. **2019-2020_Crisis**: Train on 2012-2018, test on COVID period
2. **2021-2022_Bear**: Train through COVID, test on Fed tightening  
3. **2023-2024_Recent**: Train through 2022, test on recent period

**Holdout Period**: 2025-2026 reserved for final validation (never use until deployment)

### Crisis Labeling

- **±30 days** around each of 16 known turning points
- **12.7% crisis days**, 87.3% normal days (realistic base rate)
- Based on major crashes: Dotcom, GFC, EU Debt, China, COVID, etc.

### Signal Extraction

The framework extracts signals from the data structure:

- **Binary signals**: Checks if signal appears in `top_signals` or `bottom_signals`
- **Continuous signals**: Uses raw values (like `ll_zscore` for LL deteriorating)
- **Threshold optimization**: Uses ROC curves to find optimal firing points

### Threshold Optimization

- **Training data**: Optimize threshold using ROC analysis
- **Test data**: Apply optimized threshold, measure performance
- **Objective**: Defaults to F1 score, can be changed to precision/recall/specificity

### Preventing Overfitting

The framework prevents overfitting through:

1. **Temporal splits**: No future data in training
2. **Walk-forward testing**: Multiple periods prevent single-period overfitting
3. **Individual signal testing**: No ensemble masking of poor signals
4. **Proper base rates**: Realistic crisis frequency (not 100% crash data)
5. **Out-of-sample holdout**: Final test period never touched during development

### When to Deploy a Signal

Deploy ONLY if signal meets ALL criteria across ALL periods:

✅ **Precision >50%** across all test periods  
✅ **False positive rate <5%** consistently  
✅ **Crisis specificity >5x** (fires 5x more in crisis)  
✅ **Consistent performance** (not period-dependent)  

### Red Flags to Avoid

❌ **Period dependency**: Works in one period, fails in others  
❌ **High false positives**: >15% FPR is unacceptable for live trading  
❌ **Low precision**: <30% means signal is mostly noise  
❌ **Always firing**: >90% fire rate means broken signal  
❌ **Cherry-picked testing**: Only testing on known crash dates  

### Interpreting Results

#### Good Signal Example:
```
Precision: 0.75, Recall: 0.60, F1: 0.67
False Positive Rate: 0.03, Fire Rate: 0.15
Crisis Specificity: 20.0 (fires 20x more in crisis)
```

#### Bad Signal Example:
```  
Precision: 0.12, Recall: 1.00, F1: 0.22
False Positive Rate: 0.95, Fire Rate: 0.92
Crisis Specificity: 1.05 (barely different)
```

### Extending the Framework

#### Adding New Validation Periods
```python
ValidationPeriod(
    name="Custom_Period",
    train_start="YYYY-MM-DD", train_end="YYYY-MM-DD", 
    test_start="YYYY-MM-DD", test_end="YYYY-MM-DD",
    description="What makes this period unique"
)
```

#### Custom Metrics
Add business-specific metrics to `SignalMetrics` dataclass and update calculation methods.

#### Different Crisis Definitions  
Modify `create_crisis_labels()` to use different window sizes or turning point definitions.

### Best Practices

1. **Start simple**: Test individual signals before building ensembles
2. **Be conservative**: Better to miss some crises than cry wolf constantly  
3. **Monitor live performance**: Backtest ≠ real trading
4. **Update regularly**: Re-run validation as new data arrives
5. **Document everything**: Keep detailed records of all validation runs

### Common Mistakes

❌ **Testing on training data**: Always use separate test periods  
❌ **Optimizing on test data**: Only optimize thresholds on training data  
❌ **Ignoring false positives**: They kill live trading performance  
❌ **Single period validation**: One period is not enough  
❌ **Ensemble first**: Test individual signals before combining  

---

**Remember**: This framework is designed to prevent disasters. If a signal doesn't pass validation, it's not ready for deployment. No exceptions.

The goal is bulletproof signals that work consistently across different market environments, not cherry-picked performance on handpicked dates.