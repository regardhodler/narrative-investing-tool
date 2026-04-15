# Validation Pipeline Guide: Using the Framework for Signal Development

## Overview

This guide explains how to use the existing `validation_framework.py` to test each signal candidate systematically. The framework prevents overfitting through walk-forward validation and proper crisis labeling.

## Framework Architecture

### Core Components

**Location**: `validation_framework.py` (890 lines)
**Supporting Files**:
- `VALIDATION_FRAMEWORK_SUMMARY.md` - Framework overview
- `VALIDATION_METHODOLOGY.md` - Detailed methodology  
- `signal_validation_report.md` - Results from current (failed) signals

### Validation Periods

```python
VALIDATION_PERIODS = [
    ValidationPeriod(
        name="2019-2020_Crisis",
        train_start="2012-01-01", train_end="2018-12-31",
        test_start="2019-01-01", test_end="2020-12-31"
    ),
    ValidationPeriod(
        name="2021-2022_Bear",
        train_start="2012-01-01", train_end="2020-12-31",
        test_start="2021-01-01", test_end="2022-12-31"
    ),
    ValidationPeriod(
        name="2023-2024_Recent", 
        train_start="2012-01-01", train_end="2022-12-31",
        test_start="2023-01-01", test_end="2024-12-31"
    )
]

# NEVER TOUCH - Reserved for final validation only
HOLDOUT_PERIOD = ValidationPeriod(
    name="2025-2026_Holdout",
    train_start="2012-01-01", train_end="2024-12-31", 
    test_start="2025-01-01", test_end="2026-12-31"
)
```

### Crisis Labeling Method

**Crisis Windows**: ±30 days around 16 known turning points
**Crisis Rate**: 12.7% of all trading days (realistic imbalance)

```python
CRASH_SCENARIOS = [
    ("Dotcom Peak", "2000-03-24", "2000-04-23"),
    ("Dotcom Trough", "2002-09-09", "2002-11-08"),
    ("GFC Peak", "2007-09-09", "2007-11-08"), 
    ("GFC Trough", "2009-02-07", "2009-04-08"),
    ("EU Debt Peak", "2011-03-29", "2011-05-28"),
    ("EU Debt Trough", "2011-09-03", "2011-11-02"),
    ("China Peak", "2015-06-20", "2015-08-19"),
    ("China Trough", "2016-01-12", "2016-03-12"),
    ("Volmageddon Peak", "2018-01-26", "2018-02-25"),
    ("Volmageddon Trough", "2018-11-24", "2018-12-24"),
    ("COVID Peak", "2020-01-20", "2020-03-20"),
    ("COVID Trough", "2020-02-23", "2020-04-23"),
    ("Rate Shock Peak", "2022-01-03", "2022-02-02"), 
    ("Rate Shock Trough", "2022-09-12", "2022-11-11"),
    ("Carry Trade Peak", "2024-06-16", "2024-08-15"),
    ("Carry Trade Trough", "2024-07-05", "2024-09-03")
]
```

## Step-by-Step Signal Validation Process

### Step 1: Signal Definition

Create signal function that returns boolean array:

```python
def vix_persistent_elevation(data, threshold=25, window=10):
    """
    VIX >25 for 10+ consecutive days
    
    Args:
        data: DataFrame with VIX column
        threshold: VIX threshold (optimize on training data)
        window: Consecutive days requirement
        
    Returns:
        Boolean array indicating signal active
    """
    vix_above = data['VIX'] > threshold
    consecutive_count = vix_above.rolling(window=window).sum()
    return consecutive_count >= window
```

### Step 2: Parameter Optimization

Use training data to find optimal threshold:

```python
from validation_framework import ValidationFramework, optimize_signal_threshold

# Initialize framework
vf = ValidationFramework()

# Define signal function
def signal_func(data, threshold):
    return vix_persistent_elevation(data, threshold=threshold)

# Find optimal threshold for each period
for period in VALIDATION_PERIODS:
    train_data = vf.get_training_data(period)
    
    # ROC-based optimization
    optimal_threshold, metrics = optimize_signal_threshold(
        signal_func=signal_func,
        data=train_data,
        threshold_range=np.arange(20, 35, 1),  # Test VIX 20-35
        crisis_labels=vf.get_crisis_labels(train_data)
    )
    
    print(f"{period.name}: Optimal threshold = {optimal_threshold}")
    print(f"Training AUC = {metrics.auc_score:.3f}")
```

### Step 3: Out-of-Sample Validation

Test optimized signal on validation periods:

```python
# Test each period with threshold optimized on training data
results = []

for period in VALIDATION_PERIODS:
    # Get training data and optimize threshold
    train_data = vf.get_training_data(period)
    optimal_threshold = optimize_threshold(train_data)  # From step 2
    
    # Test on out-of-sample validation period  
    test_data = vf.get_test_data(period)
    test_signals = signal_func(test_data, threshold=optimal_threshold)
    test_crisis_labels = vf.get_crisis_labels(test_data)
    
    # Calculate performance metrics
    metrics = vf.calculate_signal_metrics(
        signals=test_signals,
        crisis_labels=test_crisis_labels,
        signal_name="VIX_persistent_elevation",
        period_name=period.name
    )
    
    results.append(metrics)
    
    # Check success criteria
    success = (
        metrics.precision > 0.30 and
        metrics.false_positive_rate < 0.15 and
        metrics.crisis_specificity > 3.0
    )
    
    print(f"\n{period.name} Results:")
    print(f"Precision: {metrics.precision:.1%}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"Crisis Specificity: {metrics.crisis_specificity:.1f}x")
    print(f"Signal Fire Rate: {metrics.signal_fire_rate:.1%}")
    print(f"SUCCESS: {success}")
```

### Step 4: Cross-Period Analysis

Evaluate signal stability across periods:

```python
# Aggregate results across periods
avg_precision = np.mean([r.precision for r in results])
avg_fpr = np.mean([r.false_positive_rate for r in results])
precision_std = np.std([r.precision for r in results])

print(f"\nCross-Period Performance:")
print(f"Average Precision: {avg_precision:.1%} ± {precision_std:.1%}")
print(f"Average FPR: {avg_fpr:.1%}")
print(f"Precision Range: {min(r.precision for r in results):.1%} - {max(r.precision for r in results):.1%}")

# Check temporal stability (precision shouldn't vary >15% across periods)
temporal_stability = precision_std < 0.15
print(f"Temporal Stability: {temporal_stability}")

# Overall signal acceptance
signal_passes = (
    avg_precision > 0.30 and
    avg_fpr < 0.15 and 
    temporal_stability and
    all(r.crisis_specificity > 3.0 for r in results)
)

print(f"\nOVERALL SIGNAL PASSES: {signal_passes}")
```

## Complete Example: Testing VIX Persistent Elevation

```python
import pandas as pd
import numpy as np
from validation_framework import ValidationFramework, ValidationPeriod

# Load data
vf = ValidationFramework()
data = vf.load_market_data()  # Includes VIX, dates, etc.

def vix_persistent_elevation_signal(data, threshold=25, window=10):
    """VIX >threshold for window+ consecutive days"""
    vix_above = data['VIX'] > threshold
    consecutive = vix_above.rolling(window=window).sum()
    return consecutive >= window

# Test signal across all validation periods
def validate_vix_signal():
    results = []
    
    for period in vf.validation_periods:
        print(f"\n=== Testing {period.name} ===")
        
        # Get training data
        train_data = vf.get_period_data(
            period.train_start, period.train_end
        )
        train_crisis = vf.get_crisis_labels(train_data)
        
        # Optimize threshold on training data
        best_threshold = None
        best_auc = 0
        
        for threshold in range(20, 36):  # Test VIX 20-35
            signals = vix_persistent_elevation_signal(
                train_data, threshold=threshold
            )
            auc = vf.calculate_auc(signals, train_crisis)
            
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold} (AUC: {best_auc:.3f})")
        
        # Test on out-of-sample period
        test_data = vf.get_period_data(
            period.test_start, period.test_end
        )
        test_crisis = vf.get_crisis_labels(test_data)
        
        test_signals = vix_persistent_elevation_signal(
            test_data, threshold=best_threshold
        )
        
        # Calculate metrics
        metrics = vf.calculate_signal_metrics(
            signals=test_signals,
            crisis_labels=test_crisis,
            signal_name="VIX_persistent_elevation",
            period_name=period.name
        )
        
        results.append(metrics)
        
        # Print results
        print(f"Test Results:")
        print(f"  Precision: {metrics.precision:.1%}")
        print(f"  Recall: {metrics.recall:.1%}")
        print(f"  FPR: {metrics.false_positive_rate:.1%}")
        print(f"  Fire Rate: {metrics.signal_fire_rate:.1%}")
        print(f"  Crisis Specificity: {metrics.crisis_specificity:.1f}x")
        
        # Check pass/fail
        passes = (
            metrics.precision > 0.30 and
            metrics.false_positive_rate < 0.15 and
            metrics.crisis_specificity > 3.0
        )
        print(f"  PASSES: {passes}")
    
    return results

# Run validation
signal_results = validate_vix_signal()

# Summary across all periods
avg_precision = np.mean([r.precision for r in signal_results])
avg_fpr = np.mean([r.false_positive_rate for r in signal_results])

print(f"\n=== FINAL ASSESSMENT ===")
print(f"Average Precision: {avg_precision:.1%}")
print(f"Average FPR: {avg_fpr:.1%}")
print(f"Signal Quality: {'GOOD' if avg_precision > 0.30 and avg_fpr < 0.15 else 'POOR'}")
```

## Signal Quality Assessment

### Success Criteria

**Minimum Requirements:**
- Precision >30% across all periods
- False positive rate <15% across all periods  
- Crisis specificity >3x (fires 3+ times more in crisis vs normal)
- Temporal stability (precision std dev <15%)

**Target Performance:**
- Precision >50%
- False positive rate <10%
- Crisis specificity >5x
- Economic interpretability

### Common Failure Modes

**Period Dependency**: Works in one period, fails in others
- **Symptom**: High precision in 2019-2020, poor in 2021-2024
- **Cause**: Overfitting to specific market regime
- **Solution**: Test longer training periods, different regimes

**Base Rate Neglect**: Optimized on crisis days only
- **Symptom**: High recall, terrible precision (current proximity problem)  
- **Cause**: Ignoring 87% normal market days in optimization
- **Solution**: Use balanced metrics (F1, AUC) not just recall

**False Positive Explosion**: Signals fire constantly
- **Symptom**: >20% fire rate, low precision
- **Cause**: Threshold too loose or signal fundamentally noisy
- **Solution**: Tighten thresholds or abandon signal

**Economic Nonsense**: Good statistics, bad logic
- **Symptom**: Passes metrics but makes no economic sense
- **Cause**: Data mining without economic foundation
- **Solution**: Require causal story before testing

## Integration with QIR System

### After Signal Validation

Once signal passes validation framework:

1. **Document Results**: Save validation metrics in `validated_signals/`
2. **Create Signal Class**: Implement standardized signal interface
3. **Add to QIR Pipeline**: Integrate in `modules/quick_run.py`
4. **Setup Monitoring**: Track live performance vs validation metrics
5. **Add Badges**: Show validation metrics in dashboard

### Signal Interface Standard

```python
class ValidatedSignal:
    def __init__(self, config):
        self.name = config.name
        self.threshold = config.validated_threshold
        self.validation_metrics = config.metrics
        
    def evaluate(self, market_data):
        raw_score = self._calculate(market_data)
        return {
            'signal_active': raw_score > self.threshold,
            'confidence': self._confidence_score(raw_score),
            'precision': self.validation_metrics.precision,
            'false_positive_rate': self.validation_metrics.false_positive_rate
        }
        
    def get_validation_badge(self):
        return f"✅ {self.validation_metrics.precision:.0%} precision, {self.validation_metrics.false_positive_rate:.0%} FPR"
```

## Best Practices

### Development Workflow

1. **Economic Hypothesis**: Start with economic reasoning for signal
2. **Single Signal Testing**: Test each signal individually first
3. **Parameter Optimization**: Use training data only
4. **Out-of-Sample Testing**: Test on validation periods
5. **Cross-Period Analysis**: Ensure temporal stability
6. **Documentation**: Record all tested signals (successes and failures)

### Avoiding Pitfalls

**DON'T:**
- Optimize on test/validation periods
- Combine signals before individual validation
- Ignore false positive rates
- Use holdout period (2025-2026) until final deployment
- Test dozens of similar signals (multiple testing problem)

**DO:**
- Test high-conviction signals first
- Require economic logic for each signal
- Document all failures to avoid retesting
- Use proper base rates (87% normal days)
- Maintain temporal train/test splits

This framework prevents the overfitting disaster that led to the current 0% accuracy proximity signals. Use it religiously for every signal candidate.