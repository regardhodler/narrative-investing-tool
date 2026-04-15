# Signal Rebuild Roadmap: Complete Top/Bottom Proximity Reconstruction

## Executive Summary

The current Top/Bottom Proximity signals have **0% real-world accuracy** due to catastrophic threshold miscalibration on cherry-picked data. This roadmap outlines a systematic, validation-first approach to rebuild genuinely predictive signals from the ground up.

## Current State Assessment

### Critical Findings
- **All signals FAIL validation**: 0-30% precision, 15-100% false positive rates
- **Massive threshold miscalibration**: Signals fire 3-99× more than optimal rates
- **Cherry-picked optimization**: Trained on 16 extreme dates (0.7% of data), fails on 99.3% of normal days
- **Broken signal examples**:
  - Low Conviction fires **99.3% of days** (should be 1-5%)
  - High Entropy fires **77.1% of days** (should be 5-10%)
  - Deep Regime fires **50.7% of days** (should be 3-8%)

### Root Cause
Count method optimized on extreme peaks/troughs without considering base rates or false positive costs. No validation framework used during development.

## Rebuild Strategy

### Phase 1: Immediate Cleanup (Week 1)
**Goal**: Stop the bleeding, prepare for reconstruction

#### 1.1 Disable Current Proximity Section
- [ ] Add feature flag `DISABLE_PROXIMITY_SIGNALS = True` in quick_run.py
- [ ] Return neutral values (top_count=0, bottom_count=0, scores=0) when disabled
- [ ] Update QIR pattern classification to handle absence of proximity signals
- [ ] Add warning banner in dashboard indicating proximity signals are under maintenance

#### 1.2 Archive Current Implementation
- [ ] Create `archive/broken_proximity/` folder
- [ ] Move current proximity logic to archive with timestamp
- [ ] Document failure analysis in `archive/broken_proximity/FAILURE_ANALYSIS.md`

### Phase 2: Signal Discovery (Weeks 2-3)
**Goal**: Identify 2-3 individual signals that pass validation

#### 2.1 Data Preparation
- [ ] Extract 12+ years of HMM training data (2012-2026)
- [ ] Prepare crisis labeling using validation framework (±30 days around 16 turning points)
- [ ] Create signal testing harness using validation_framework.py
- [ ] Set up walk-forward validation periods

#### 2.2 Systematic Signal Testing
Test each signal candidate individually using validation framework:

**VIX-Based Signals:**
- [ ] VIX >30 for 5+ consecutive days
- [ ] VIX >40 spike (daily)
- [ ] VIX persistent elevation (>25 for 10+ days)
- [ ] VIX term structure inversion (VIX9D > VIX)

**Credit Spread Signals:**
- [ ] HY spreads >500bp + expanding
- [ ] HY spreads >700bp (crisis threshold)
- [ ] HY spread momentum (5-day acceleration >50bp)
- [ ] IG-HY spread differential widening

**Yield Curve Signals:**
- [ ] 10Y-2Y inverted >6 months
- [ ] 10Y-3M inverted >3 months  
- [ ] Curve steepening after inversion (>50bp move)
- [ ] Real yields >3% (DFII10)

**Market Breadth Signals:**
- [ ] SPX <200MA + breadth <30%
- [ ] New 52w lows >new highs for 10+ days
- [ ] Sector rotation breakdown (utilities outperforming tech)
- [ ] Small cap underperformance (IWM vs SPY <0.8)

**Momentum Break Signals:**
- [ ] SPX breaks 50-day MA with volume
- [ ] 200-day MA slope turning negative
- [ ] Price momentum acceleration <-10%
- [ ] Cross-asset momentum breakdown

#### 2.3 Validation Requirements
Each signal must achieve:
- **Precision**: >30% minimum, >50% target
- **False Positive Rate**: <15% maximum, <10% target
- **Crisis Specificity**: Fires 3+ times more in crisis vs normal periods
- **Temporal Stability**: Works across multiple validation periods
- **Economic Logic**: Causally linked to market stress/recovery

### Phase 3: Signal Combination Testing (Week 4)
**Goal**: Test if validated individual signals can be combined effectively

#### 3.1 Combination Methods
Only proceed if 2+ individual signals pass validation:

**Ensemble Approaches:**
- [ ] Majority vote (2 of 3 signals)
- [ ] Weighted scoring (precision-weighted)
- [ ] Sequential logic (Signal A AND Signal B within N days)
- [ ] Probabilistic combination (Bayesian inference)

#### 3.2 Combination Validation
Test each combination method with same validation requirements. Combinations often perform worse than individual signals due to interaction effects.

### Phase 4: Integration & Deployment (Week 5-6)
**Goal**: Integrate validated signals into QIR system

#### 4.1 New Signal Architecture
Replace count method with validated approach:

```python
class ValidatedProximitySignal:
    def __init__(self, signal_config):
        self.name = signal_config.name
        self.threshold = signal_config.validated_threshold
        self.validation_metrics = signal_config.metrics
        
    def evaluate(self, market_data):
        raw_score = self._calculate_raw_score(market_data)
        return {
            'signal_active': raw_score > self.threshold,
            'confidence': self._calculate_confidence(raw_score),
            'validation_badge': self.validation_metrics.precision
        }
```

#### 4.2 QIR Integration Updates
- [ ] Update pattern classification to use validated signals
- [ ] Add validation badges showing signal precision/FPR
- [ ] Implement graceful degradation if signals unavailable
- [ ] Update conviction calculation weights

#### 4.3 Monitoring & Alerts
- [ ] Daily signal performance tracking
- [ ] Alert if signal fire rate drifts >2σ from expected
- [ ] Monthly validation refresh on recent data
- [ ] Automatic disable if performance degrades below thresholds

## Implementation Details

### Technology Stack
- **Validation Framework**: `validation_framework.py` (existing)
- **Data Sources**: FRED cache (25 series), yfinance (VIX, market data)
- **Testing Harness**: Jupyter notebooks for iterative signal development
- **Integration Point**: `modules/quick_run.py` proximity section

### Quality Gates

#### Signal Acceptance Criteria
1. **Statistical Requirements**:
   - Precision >30% across all validation periods
   - False positive rate <15%
   - AUC >0.65 on ROC curve

2. **Economic Requirements**:
   - Causal relationship to market stress identifiable
   - Signal timing economically actionable (not too late)
   - Robust to regime changes (works in different market environments)

3. **Technical Requirements**:
   - Reproducible calculation method
   - Real-time computable with available data
   - Stable parameter values (no frequent recalibration needed)

#### Combination Acceptance Criteria
- Combined signal outperforms best individual signal by >5% precision
- False positive rate doesn't increase vs best individual signal
- Economic interpretation remains clear

### Risk Management

#### Overfitting Prevention
- **No optimization on test periods**: Parameters tuned only on training data
- **Held-out validation**: 2025-2026 data reserved for final validation
- **Simple signals preferred**: Fewer parameters = less overfitting risk
- **Economic constraints**: Parameters must make economic sense

#### Performance Monitoring
- **Daily tracking**: Signal fire rates, market performance after signals
- **Weekly review**: Precision/recall trending, false positive monitoring
- **Monthly validation**: Refresh validation metrics with new data
- **Quarterly review**: Full signal performance assessment

## Success Metrics

### Phase Success Criteria

**Phase 1 Success**: Current proximity signals disabled without breaking QIR
**Phase 2 Success**: 2-3 individual signals pass full validation framework  
**Phase 3 Success**: Signal combination testing completed (may show combinations ineffective)
**Phase 4 Success**: New signals integrated with performance monitoring active

### Long-term KPIs
- **Signal Precision**: Maintain >30% over 6+ months
- **False Positive Rate**: Keep <15% in normal market conditions
- **Economic Value**: Signals provide actionable investment insight
- **User Trust**: Validation badges restore confidence in proximity signals

## Timeline

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Immediate Cleanup | Current signals disabled, failure analysis documented |
| 2-3 | Signal Discovery | 2-3 validated individual signals identified |
| 4 | Combination Testing | Ensemble methods tested (if applicable) |
| 5-6 | Integration | New signals deployed with monitoring |

## Resource Requirements

- **Development Time**: 40-50 hours across 6 weeks
- **Data Requirements**: 12+ years FRED/market data (already available)
- **Compute Resources**: Validation framework testing (existing infrastructure)
- **Review Requirements**: Weekly validation checkpoint reviews

## Alternative Outcomes

### If No Signals Pass Validation
**Option A**: Remove proximity signals entirely, rely on other QIR components  
**Option B**: Develop ensemble of weak signals with explicit uncertainty quantification  
**Option C**: Research alternative signal approaches (machine learning, alternative data)

### If Combinations Fail
Individual validated signals still provide value. Combinations often fail due to correlation and interaction effects. Better to have 2 excellent signals than 1 mediocre combination.

## Conclusion

This roadmap prioritizes **validation-first development** to prevent the overfitting disaster from recurring. The goal is fewer, genuinely predictive signals rather than many noisy ones. Success is measured by real-world precision and false positive rates, not cherry-picked accuracy metrics.

The rebuild will take 6 weeks but will establish a robust, trustworthy foundation for proximity signal analysis going forward.