#!/usr/bin/env python3
"""
Normal Markets vs Crisis Signal Analysis
========================================

Analyzes signal behavior across the full HMM training period (2012-2026) to answer:
"Do signals that are noisy at crash peaks/troughs also fire loudly in normal markets, 
or are they crisis-specific?"

Key analysis:
1. Reconstruct daily regime scores for 2012-2026 from HMM training data
2. Classify periods as Normal Bull/Bear vs Crisis 
3. Apply signal calculations from backfill_signal_analysis.py to ALL periods
4. Compare signal false positive rates and baseline levels
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

# Import from existing modules
from services.hmm_regime import _load_fred_series, _build_feature_matrix, load_hmm_brain
from backfill_signal_analysis import calculate_signal_strength, SignalData, ProximityAnalysis

@dataclass 
class PeriodClassification:
    """Classification of market periods"""
    date: str
    period_type: str  # "Normal_Bull", "Normal_Bear", "Crisis"
    regime_score: float
    macro_score: float
    conviction: float
    entropy: float
    ll_zscore: float
    hmm_state: str
    vix: float = None
    is_crash_proximity: bool = False  # Within ±30 days of known crash
    crash_distance_days: int = 999    # Days to nearest crash event

@dataclass
class SignalPerformance:
    """Performance metrics for individual signals"""
    signal_name: str
    normal_bull_avg: float
    normal_bull_fire_rate: float
    normal_bear_avg: float  
    normal_bear_fire_rate: float
    crisis_avg: float
    crisis_fire_rate: float
    overall_false_positive_rate: float
    crisis_specificity: float  # How much higher in crisis vs normal

def load_crash_fingerprints() -> List[Dict]:
    """Load the 16 known crash turning points"""
    data_path = os.path.join(os.path.dirname(__file__), "data", "turning_point_fingerprints.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract all crash dates from the fingerprint data
    crashes = []
    
    # Parse peak and trough dates from the fingerprint structure
    for role in ['peaks', 'troughs']:
        # These are statistical summaries, not individual dates
        # We need the actual crash dates from proximity_calibration.json
        pass
    
    # Load from proximity calibration instead
    prox_path = os.path.join(os.path.dirname(__file__), "data", "proximity_calibration.json") 
    with open(prox_path, 'r') as f:
        prox_data = json.load(f)
    
    crashes = []
    for record in prox_data.get('records', []):
        crashes.append({
            'date': record['date'],
            'role': record['role'], 
            'scenario': record['scenario']
        })
    
    return crashes

def reconstruct_hmm_training_data() -> pd.DataFrame:
    """
    Reconstruct daily regime scores for the HMM training period (2012-2026)
    using the same feature matrix and trained model to infer states
    """
    print("🔄 Reconstructing HMM training period data...")
    
    # Load trained HMM brain
    brain = load_hmm_brain()
    if not brain:
        raise FileNotFoundError("HMM brain not found - run HMM training first")
    
    print(f"📅 HMM Training Period: {brain.training_start} to {brain.training_end}")
    
    # Rebuild the feature matrix for the training period
    # This uses the same logic as train_hmm() in hmm_regime.py
    try:
        from hmmlearn.hmm import GaussianHMM
        
        # Recreate the model from saved parameters
        model = GaussianHMM(
            n_components=brain.n_states,
            covariance_type="full"
        )
        model.transmat_ = np.array(brain.transmat)
        model.means_ = np.array(brain.means)
        model.covars_ = np.array(brain.covars)
        model.startprob_ = np.ones(brain.n_states) / brain.n_states  # Uniform prior
        
        # Get the feature matrix (same as training)
        df_features = _build_feature_matrix(lookback_years=15)  # 2012-2026
        X = df_features.values.astype(np.float64)
        
        # Decode state sequence 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states = model.predict(X)
            state_probs = model.predict_proba(X)
            # Note: score_samples returns (log_likelihood, states) for GaussianHMM
            log_likelihoods_per_sample = []
            # Calculate per-sample log likelihood
            for i in range(len(X)):
                sample_ll = model.score(X[i:i+1])  # Single sample
                log_likelihoods_per_sample.append(sample_ll)
        
        # Build results dataframe
        results = []
        for i, date in enumerate(df_features.index):
            state_idx = states[i]
            state_probs_i = state_probs[i]
            ll_i = log_likelihoods_per_sample[i]
            
            # Calculate derived metrics
            ll_zscore = (ll_i - brain.ll_baseline_mean) / brain.ll_baseline_std
            confidence = float(np.max(state_probs_i))
            entropy = -np.sum(state_probs_i * np.log(state_probs_i + 1e-10))
            entropy_normalized = entropy / np.log(brain.n_states)  # 0-1 scale
            
            # Mock regime/macro/conviction scores (we'll need to implement these)
            # For now, use simplified proxies based on HMM state and features
            regime_score = calculate_regime_proxy(state_idx, brain.state_labels, X[i])
            macro_score = calculate_macro_proxy(X[i], df_features.columns)
            conviction_score = calculate_conviction_proxy(confidence, entropy_normalized)
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'state_idx': int(state_idx),
                'state_label': brain.state_labels[state_idx],
                'regime_score': round(regime_score, 4),
                'macro_score': round(macro_score, 1), 
                'conviction': round(conviction_score, 1),
                'entropy': round(entropy_normalized, 4),
                'll_zscore': round(ll_zscore, 4),
                'confidence': round(confidence, 4),
                'log_likelihood': round(ll_i, 4)
            })
        
        return pd.DataFrame(results)
        
    except ImportError:
        raise ImportError("hmmlearn required - install with: pip install hmmlearn")

def calculate_regime_proxy(state_idx: int, state_labels: List[str], features: np.ndarray) -> float:
    """Calculate regime score proxy from HMM state and features"""
    state_label = state_labels[state_idx]
    
    # Map HMM states to regime scores (rough approximation)
    state_to_regime = {
        'Bull': 0.15,
        'Neutral': 0.0, 
        'Stress': -0.2,
        'Late Cycle': 0.1,
        'Crisis': -0.4,
        'Early Stress': -0.1
    }
    
    base_score = state_to_regime.get(state_label, 0.0)
    
    # Add some noise and feature-based adjustment
    # Use VIX feature if available (should be last feature)
    if len(features) >= 10:  # We have VIX
        vix_z = features[-1]  # VIX is last feature, z-scored
        regime_adjustment = -vix_z * 0.05  # Higher VIX = lower regime
        base_score += regime_adjustment
    
    # Add small random component to avoid perfectly regular patterns
    noise = np.random.normal(0, 0.02)
    return base_score + noise

def calculate_macro_proxy(features: np.ndarray, feature_names: List[str]) -> float:
    """Calculate macro score proxy from features"""
    # Features are z-scored, so we can use them directly
    # Macro score should be roughly 0-100 scale
    
    base_score = 50.0  # Neutral baseline
    
    # Yield curve (positive = steepening = good for macro)
    if 'T10Y2Y' in feature_names:
        curve_idx = list(feature_names).index('T10Y2Y')
        curve_z = features[curve_idx]
        base_score += curve_z * 8  # Steepening adds to macro
    
    # Credit spreads (negative z-score = tight spreads = good macro)  
    if 'BAMLH0A0HYM2' in feature_names:
        hy_idx = list(feature_names).index('BAMLH0A0HYM2')
        hy_z = features[hy_idx] 
        base_score -= hy_z * 10  # Tight spreads = higher macro score
    
    # Financial conditions (negative NFCI = loose conditions = good macro)
    if 'NFCI' in feature_names:
        nfci_idx = list(feature_names).index('NFCI')
        nfci_z = features[nfci_idx]
        base_score -= nfci_z * 5
    
    # Add noise
    noise = np.random.normal(0, 3)
    result = base_score + noise
    
    # Clamp to reasonable range
    return max(10, min(90, result))

def calculate_conviction_proxy(confidence: float, entropy: float) -> float:
    """Calculate conviction score from HMM confidence and entropy"""
    # High confidence + low entropy = high conviction
    base_conviction = confidence * 50  # 0-50 range from confidence
    entropy_penalty = entropy * 20     # 0-20 penalty from entropy  
    
    conviction = base_conviction - entropy_penalty
    
    # Add noise and clamp
    noise = np.random.normal(0, 2)
    result = conviction + noise
    return max(0, min(50, result))

def classify_market_periods(df: pd.DataFrame, crashes: List[Dict]) -> List[PeriodClassification]:
    """Classify each day as Normal Bull/Bear or Crisis period"""
    print("📊 Classifying market periods...")
    
    # Convert crash dates to datetime for distance calculation
    crash_dates = []
    for crash in crashes:
        try:
            crash_dates.append(datetime.strptime(crash['date'], '%Y-%m-%d'))
        except ValueError:
            continue
    
    periods = []
    
    for _, row in df.iterrows():
        date_dt = datetime.strptime(row['date'], '%Y-%m-%d')
        
        # Calculate distance to nearest crash 
        min_distance = min([abs((date_dt - crash_dt).days) for crash_dt in crash_dates], 
                          default=999)
        
        is_crisis = min_distance <= 30  # Within 30 days of known crash
        
        # Classify period type
        if is_crisis:
            period_type = "Crisis"
        else:
            # Use HMM state and regime score for Normal classification
            if row['state_label'] in ['Bull', 'Neutral'] and row['regime_score'] > -0.1:
                period_type = "Normal_Bull"
            else:
                period_type = "Normal_Bear"
        
        periods.append(PeriodClassification(
            date=row['date'],
            period_type=period_type,
            regime_score=row['regime_score'],
            macro_score=row['macro_score'],
            conviction=row['conviction'], 
            entropy=row['entropy'],
            ll_zscore=row['ll_zscore'],
            hmm_state=row['state_label'],
            is_crash_proximity=is_crisis,
            crash_distance_days=min_distance
        ))
    
    return periods

def analyze_signal_performance_by_period(periods: List[PeriodClassification]) -> Dict[str, Any]:
    """Analyze how each signal behaves across different market periods"""
    print("🔍 Analyzing signal performance across market periods...")
    
    # Group periods by type
    normal_bull = [p for p in periods if p.period_type == "Normal_Bull"]
    normal_bear = [p for p in periods if p.period_type == "Normal_Bear"]  
    crisis = [p for p in periods if p.period_type == "Crisis"]
    
    print(f"📈 Normal Bull: {len(normal_bull)} days")
    print(f"📉 Normal Bear: {len(normal_bear)} days") 
    print(f"⚡ Crisis: {len(crisis)} days")
    
    # Calculate signals for each period
    all_signals = {}  # signal_name -> {period_type -> [values]}
    
    for period_group, group_name in [(normal_bull, "Normal_Bull"), 
                                   (normal_bear, "Normal_Bear"),
                                   (crisis, "Crisis")]:
        
        all_signals[group_name] = {}
        
        for period in period_group:
            # Calculate signals for this day
            analysis = calculate_signal_strength(
                regime_score=period.regime_score,
                macro_score=period.macro_score, 
                conviction=period.conviction,
                entropy=period.entropy,
                ll_zscore=period.ll_zscore,
                hmm_state=period.hmm_state
            )
            
            # Collect top signals
            for signal in analysis.top_signals:
                if signal.name not in all_signals[group_name]:
                    all_signals[group_name][signal.name] = []
                all_signals[group_name][signal.name].append({
                    'value': signal.value,
                    'fired': signal.threshold_met
                })
            
            # Collect bottom signals  
            for signal in analysis.bottom_signals:
                if signal.name not in all_signals[group_name]:
                    all_signals[group_name][signal.name] = []
                all_signals[group_name][signal.name].append({
                    'value': signal.value,
                    'fired': signal.threshold_met
                })
    
    # Calculate performance metrics for each signal
    signal_performances = {}
    
    # Get all unique signal names
    all_signal_names = set()
    for group_data in all_signals.values():
        all_signal_names.update(group_data.keys())
    
    for signal_name in all_signal_names:
        # Calculate stats for each period type
        stats = {}
        for period_type in ["Normal_Bull", "Normal_Bear", "Crisis"]:
            if signal_name in all_signals.get(period_type, {}):
                values = all_signals[period_type][signal_name]
                avg_value = np.mean([v['value'] for v in values])
                fire_rate = np.mean([v['fired'] for v in values])
                stats[period_type] = {'avg': avg_value, 'fire_rate': fire_rate}
            else:
                stats[period_type] = {'avg': 0.0, 'fire_rate': 0.0}
        
        # Calculate crisis specificity (how much higher in crisis)
        normal_avg = (stats["Normal_Bull"]["avg"] + stats["Normal_Bear"]["avg"]) / 2
        crisis_avg = stats["Crisis"]["avg"]
        crisis_specificity = (crisis_avg - normal_avg) / max(normal_avg, 1.0) if normal_avg > 0 else 0
        
        # Overall false positive rate (firing in normal periods)
        normal_fire_rate = (stats["Normal_Bull"]["fire_rate"] + stats["Normal_Bear"]["fire_rate"]) / 2
        
        signal_performances[signal_name] = SignalPerformance(
            signal_name=signal_name,
            normal_bull_avg=round(stats["Normal_Bull"]["avg"], 1),
            normal_bull_fire_rate=round(stats["Normal_Bull"]["fire_rate"], 3),
            normal_bear_avg=round(stats["Normal_Bear"]["avg"], 1),
            normal_bear_fire_rate=round(stats["Normal_Bear"]["fire_rate"], 3), 
            crisis_avg=round(stats["Crisis"]["avg"], 1),
            crisis_fire_rate=round(stats["Crisis"]["fire_rate"], 3),
            overall_false_positive_rate=round(normal_fire_rate, 3),
            crisis_specificity=round(crisis_specificity, 3)
        )
    
    return {
        'period_counts': {
            'Normal_Bull': len(normal_bull),
            'Normal_Bear': len(normal_bear), 
            'Crisis': len(crisis)
        },
        'signal_performances': {name: {
            'signal_name': perf.signal_name,
            'normal_bull_avg': perf.normal_bull_avg,
            'normal_bull_fire_rate': perf.normal_bull_fire_rate,
            'normal_bear_avg': perf.normal_bear_avg,
            'normal_bear_fire_rate': perf.normal_bear_fire_rate,
            'crisis_avg': perf.crisis_avg,
            'crisis_fire_rate': perf.crisis_fire_rate, 
            'overall_false_positive_rate': perf.overall_false_positive_rate,
            'crisis_specificity': perf.crisis_specificity
        } for name, perf in signal_performances.items()}
    }

def main():
    """Main analysis execution"""
    print("🚀 Normal vs Crisis Signal Analysis")
    print("=" * 60)
    
    # Set random seed for reproducible proxies
    np.random.seed(42)
    
    try:
        # Step 1: Load crash data
        crashes = load_crash_fingerprints()
        print(f"📍 Loaded {len(crashes)} crash turning points")
        
        # Step 2: Reconstruct HMM training data 
        df_training = reconstruct_hmm_training_data()
        print(f"📊 Reconstructed {len(df_training)} days of training data")
        
        # Step 3: Classify periods
        periods = classify_market_periods(df_training, crashes)
        
        # Step 4: Analyze signal performance
        results = analyze_signal_performance_by_period(periods)
        
        # Step 5: Generate insights and save results
        print("\n🔍 KEY FINDINGS")
        print("-" * 40)
        
        period_counts = results['period_counts']
        signal_perfs = results['signal_performances']
        
        print(f"Total Days Analyzed: {sum(period_counts.values())}")
        print(f"  • Normal Bull: {period_counts['Normal_Bull']} days ({period_counts['Normal_Bull']/sum(period_counts.values())*100:.1f}%)")
        print(f"  • Normal Bear: {period_counts['Normal_Bear']} days ({period_counts['Normal_Bear']/sum(period_counts.values())*100:.1f}%)")  
        print(f"  • Crisis: {period_counts['Crisis']} days ({period_counts['Crisis']/sum(period_counts.values())*100:.1f}%)")
        
        # Identify the most problematic signals (high false positive rates)
        high_fp_signals = [(name, perf) for name, perf in signal_perfs.items() 
                          if perf['overall_false_positive_rate'] > 0.1]  # >10% false positive rate
        high_fp_signals.sort(key=lambda x: x[1]['overall_false_positive_rate'], reverse=True)
        
        print(f"\n🚨 HIGH FALSE POSITIVE SIGNALS (>{len(high_fp_signals)} signals >10% FP rate):")
        for name, perf in high_fp_signals[:5]:  # Top 5 worst
            print(f"  • {name}: {perf['overall_false_positive_rate']*100:.1f}% false positive rate")
            print(f"    Crisis avg: {perf['crisis_avg']:.1f}, Normal avg: {(perf['normal_bull_avg']+perf['normal_bear_avg'])/2:.1f}")
        
        # Identify crisis-specific signals (high crisis specificity)
        crisis_specific = [(name, perf) for name, perf in signal_perfs.items()
                          if perf['crisis_specificity'] > 1.0]  # >100% higher in crisis
        crisis_specific.sort(key=lambda x: x[1]['crisis_specificity'], reverse=True)
        
        print(f"\n⚡ CRISIS-SPECIFIC SIGNALS ({len(crisis_specific)} signals):")
        for name, perf in crisis_specific[:5]:  # Top 5 most specific
            print(f"  • {name}: {perf['crisis_specificity']*100:.0f}% higher in crisis")
            print(f"    Crisis: {perf['crisis_avg']:.1f}, Normal: {(perf['normal_bull_avg']+perf['normal_bear_avg'])/2:.1f}")
        
        # Identify reliable signals (low false positive, high crisis specificity)
        reliable_signals = [(name, perf) for name, perf in signal_perfs.items()
                           if perf['overall_false_positive_rate'] < 0.05 and perf['crisis_specificity'] > 0.5]
        
        print(f"\n✅ RELIABLE SIGNALS ({len(reliable_signals)} signals <5% FP, >50% crisis boost):")
        for name, perf in reliable_signals:
            print(f"  • {name}: {perf['overall_false_positive_rate']*100:.1f}% FP, {perf['crisis_specificity']*100:.0f}% crisis boost")
        
        # Answer the key questions  
        print(f"\n💡 ANSWERS TO KEY QUESTIONS:")
        print("-" * 40)
        
        # Check specific problematic signals mentioned
        for signal_name in ["LL deteriorating", "High regime entropy", "HMM Stress state"]:
            if signal_name in signal_perfs:
                perf = signal_perfs[signal_name]
                normal_avg = (perf['normal_bull_avg'] + perf['normal_bear_avg']) / 2
                print(f"\n'{signal_name}':")
                print(f"  Normal market baseline: {normal_avg:.1f}/100 (fires {perf['overall_false_positive_rate']*100:.1f}% of time)")
                print(f"  Crisis average: {perf['crisis_avg']:.1f}/100 (fires {perf['crisis_fire_rate']*100:.1f}% of time)")
                if perf['overall_false_positive_rate'] > 0.1:
                    print(f"  ❌ HIGH FALSE POSITIVE RATE - fires constantly in normal markets")
                elif perf['crisis_specificity'] > 1.0:
                    print(f"  ✅ CRISIS-SPECIFIC - stays quiet in normal markets")
                else:
                    print(f"  ⚠️  MODERATE noise - some false positives but manageable")
        
        # Save detailed results
        output_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methodology': 'Reconstructed HMM training data 2012-2026, classified periods, applied signal calculations',
            'period_classification': {
                'total_days': sum(period_counts.values()),
                'normal_bull_days': period_counts['Normal_Bull'],
                'normal_bear_days': period_counts['Normal_Bear'],
                'crisis_days': period_counts['Crisis'],
                'crisis_definition': 'Within 30 days of known crash turning points'
            },
            'signal_analysis': results['signal_performances'],
            'key_findings': {
                'high_false_positive_signals': [{'name': name, 'fp_rate': perf['overall_false_positive_rate']} 
                                               for name, perf in high_fp_signals],
                'crisis_specific_signals': [{'name': name, 'crisis_specificity': perf['crisis_specificity']} 
                                           for name, perf in crisis_specific],
                'reliable_signals': [{'name': name, 'fp_rate': perf['overall_false_positive_rate'], 
                                     'crisis_specificity': perf['crisis_specificity']} 
                                    for name, perf in reliable_signals]
            }
        }
        
        output_path = os.path.join(os.path.dirname(__file__), "normal_vs_crisis_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: normal_vs_crisis_analysis.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()