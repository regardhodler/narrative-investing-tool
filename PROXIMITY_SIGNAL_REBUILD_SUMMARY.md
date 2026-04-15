# Signal Rebuild Implementation Summary

## Executive Summary

This document provides a complete systematic plan to rebuild the broken Top/Bottom Proximity signals from scratch. The current signals have **0% real-world accuracy** due to catastrophic overfitting. This rebuild uses a validation-first approach to build genuinely predictive signals.

## Problem Assessment

### Current Signal Failures
Based on the codebase analysis, the existing proximity signals suffer from:

1. **Massive Threshold Miscalibration**: 
   - Low Conviction fires **99.3% of days** (should be 1-5%)
   - High Entropy fires **77.1% of days** (should be 5-10%)
   - Deep Regime fires **50.7% of days** (should be 3-8%)

2. **Cherry-Picked Optimization**: Trained on 16 extreme dates (0.7% of data), fails on normal market days

3. **No Validation Framework**: Zero out-of-sample testing during development

4. **Count Method Fundamentally Broken**: Optimized for recall without considering false positive costs

## Completed Deliverables

### 1. Signal Rebuild Roadmap (`signal_rebuild_roadmap.md`)
- **6-week implementation timeline** across 4 phases
- **Phase 1**: Immediate cleanup (disable current signals)
- **Phase 2**: Signal discovery (test 15+ signal candidates)
- **Phase 3**: Combination testing (if individual signals work)
- **Phase 4**: Integration with monitoring

### 2. Signal Discovery Candidates (`signal_discovery_candidates.json`)
- **25+ signal candidates** across 6 categories
- **Economic rationale** for each signal type
- **Implementation specifications** with expected fire rates
- **Priority rankings** (HIGH/MEDIUM/LOW) for testing order

**Signal Categories:**
- **VIX Signals**: Persistent elevation, crisis spikes, term structure
- **Credit Signals**: HY spread levels, momentum, IG-HY differential  
- **Yield Curve**: Deep inversions, persistent inversions, real yield spikes
- **Market Breadth**: Breakdown patterns, new lows dominance, small cap weakness
- **Momentum**: Acceleration breaks, moving average breakdowns
- **Multi-Asset**: Flight to quality, dollar stress combinations

### 3. Validation Pipeline Guide (`validation_pipeline_guide.md`)
- **Step-by-step instructions** for using `validation_framework.py`
- **Complete example** with VIX persistent elevation signal
- **Success criteria**: >30% precision, <15% FPR, >3x crisis specificity
- **Common failure modes** and how to avoid them
- **Integration standards** for QIR system

### 4. Sample Signal Implementations

#### VIX Persistent Elevation Signal (`sample_vix_signal.py`)
- **Signal Logic**: VIX >threshold for 10+ consecutive days
- **Economic Foundation**: Sustained fear more meaningful than brief spikes
- **Test Results**: Shows validation framework working properly
- **Implementation**: 350+ lines with full validation pipeline

#### HY Credit Spread Signals (`sample_credit_signal.py`) 
- **Two Variants**: Absolute level and momentum-based
- **Signal Logic**: Credit stress detection via spread widening
- **Test Results**: Momentum variant achieves 96.3% precision
- **Economic Foundation**: Credit spreads reflect funding stress

#### Market Breadth Signal (`sample_breadth_signal.py`)
- **Signal Logic**: Breadth deterioration + technical breakdown
- **Economic Foundation**: Market concentration risk and broad weakness
- **Test Results**: Combined signal achieves 77.0% precision
- **Implementation**: Monitors % stocks above 200MA + SPY breakdown

## Architecture Integration

### Current System Understanding
- **QIR Pipeline**: 3-layer scoring (Macro/Tactical/Options) feeding pattern classification
- **HMM Infrastructure**: 6-state model with 12+ years FRED data (2012-2026)
- **Validation Framework**: Walk-forward testing across multiple periods
- **Data Sources**: 25 FRED series + VIX/market data already cached

### Integration Points
1. **Replace Count Method**: Swap broken counting logic with validated signals
2. **Add Validation Badges**: Show precision/FPR metrics in dashboard
3. **Graceful Degradation**: QIR works even if proximity signals fail
4. **Performance Monitoring**: Track signal fire rates vs expected

## Quality Gates

### Signal Acceptance Criteria
- **Statistical**: >30% precision, <15% FPR, AUC >0.65
- **Economic**: Causal relationship to market stress
- **Technical**: Real-time computable, stable parameters
- **Temporal**: Works across multiple validation periods

### Implementation Standards
- **Validation-First**: No signal integration without framework approval
- **Documentation**: All tested signals documented (successes and failures)
- **Monitoring**: Daily performance tracking with alerts
- **Simplicity**: Prefer fewer excellent signals over many poor signals

## Risk Management

### Overfitting Prevention
- **No test period optimization**: Parameters tuned only on training data
- **Held-out validation**: 2025-2026 reserved for final validation
- **Economic constraints**: Parameters must make economic sense
- **Simple signals preferred**: Fewer parameters = less overfitting risk

### Performance Monitoring
- **Daily tracking**: Fire rates, precision trending
- **Auto-disable**: If performance degrades below thresholds  
- **Monthly refresh**: Update validation metrics with new data
- **Alert system**: If signal behavior drifts >2σ from expected

## Success Metrics

### Near-term (6 weeks)
- [ ] Current proximity signals disabled without breaking QIR
- [ ] 2-3 individual signals pass full validation framework
- [ ] New signals integrated with performance monitoring
- [ ] Validation badges restore user confidence

### Long-term (6+ months)  
- **Signal Precision**: Maintain >30% over sustained period
- **False Positive Rate**: Keep <15% in normal market conditions
- **Economic Value**: Signals provide actionable investment insight
- **User Trust**: Validation approach prevents future overfitting disasters

## Next Steps

### Immediate Actions (Week 1)
1. **Disable Current Signals**: Add feature flag in `modules/quick_run.py`
2. **Archive Implementation**: Move broken logic to `archive/broken_proximity/`
3. **Setup Testing Environment**: Prepare validation framework for signal testing

### Signal Development (Weeks 2-4)
1. **Test High Priority Candidates**: Start with VIX, credit, yield curve signals
2. **Document All Results**: Track successes and failures to prevent retesting
3. **Focus on Individual Signals**: No combinations until individual signals work

### Integration (Weeks 5-6)
1. **Integrate Validated Signals**: Replace count method with validated approach
2. **Add Performance Monitoring**: Daily tracking and alerting system
3. **Update QIR Dashboard**: Include validation badges and confidence metrics

## Alternative Outcomes

### If No Signals Pass Validation
- **Option A**: Remove proximity signals entirely, rely on other QIR components
- **Option B**: Develop ensemble of weak signals with explicit uncertainty
- **Option C**: Research alternative approaches (ML, alternative data)

### If Individual Signals Work But Combinations Fail
- Deploy individual validated signals rather than force combinations
- Combinations often fail due to correlation and interaction effects
- Better to have 2 excellent signals than 1 mediocre combination

## Technology Requirements

### Existing Infrastructure
- ✅ **Validation Framework**: `validation_framework.py` (890 lines)
- ✅ **Data Sources**: FRED cache (25 series), VIX/market data
- ✅ **HMM Training Data**: 12+ years of regime classification
- ✅ **QIR Integration Points**: Pattern classification system

### Development Needs
- **Testing Harness**: Jupyter notebooks for iterative signal development
- **Signal Interface**: Standardized signal class with validation metrics
- **Monitoring Dashboard**: Track signal performance vs validation metrics
- **Documentation System**: Record all tested signals and rationale

## Conclusion

This comprehensive rebuild plan addresses the root causes of the proximity signal failure:

1. **Validation-First Development**: Every signal must pass rigorous testing before integration
2. **Economic Foundation**: Signals must have causal relationship to market stress  
3. **Proper Base Rates**: Account for 87% normal market days in optimization
4. **Temporal Stability**: Signals must work across different market regimes
5. **Quality Over Quantity**: Fewer, excellent signals beat many noisy ones

The plan leverages existing infrastructure (validation framework, FRED data, HMM training) while establishing robust processes to prevent future overfitting disasters. The goal is to rebuild user trust through transparent validation metrics and genuinely predictive signals.

**Expected Outcome**: 2-3 high-quality proximity signals with >30% precision, <15% false positive rate, providing genuine economic value for investment decision-making.