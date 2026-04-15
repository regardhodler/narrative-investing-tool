#!/usr/bin/env python3
"""
Sample Implementation: High-Yield Credit Spread Signal

This demonstrates implementation of a credit-based crisis signal that monitors
HY (high-yield) credit spreads for sustained widening indicating stress.

Signal Logic:
- Monitor BAMLH0A0HYM2 (HY OAS) for crisis-level spreads
- Two variants: absolute level and momentum-based
- Uses FRED data already cached in the system

Key Features:
- Leverages existing FRED data infrastructure
- Economic foundation in credit cycle theory
- Multiple signal variants for robustness testing
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class HYCreditSpreadSignal:
    """
    High-Yield Credit Spread Crisis Signal
    
    Economic Logic:
    - Credit spreads reflect market risk appetite and funding stress
    - HY bonds more sensitive to economic stress than investment grade
    - Widening spreads indicate flight to quality, funding pressure
    - Sustained wide spreads more meaningful than brief spikes
    
    Two variants:
    1. Absolute Level: HY OAS >threshold for sustained period
    2. Momentum: Rapid spread widening over short window
    """
    
    def __init__(self, signal_type="absolute"):
        self.signal_type = signal_type  # "absolute" or "momentum"
        self.name = f"HY_spread_{signal_type}"
        self.threshold = None  # Set during optimization
        self.window = 5 if signal_type == "momentum" else 10
        self.validation_metrics = None
        
    def calculate_absolute_signal(self, data, threshold=600, window=10):
        """
        HY spreads >threshold for window+ consecutive days
        
        Args:
            data: DataFrame with 'BAMLH0A0HYM2' column (HY OAS in basis points)
            threshold: Spread threshold in basis points (default 600bp = 6%)
            window: Consecutive days requirement
            
        Returns:
            Boolean Series indicating signal active
        """
        spreads_wide = data['BAMLH0A0HYM2'] > threshold
        consecutive_count = spreads_wide.rolling(window=window, min_periods=window).sum()
        signal = consecutive_count >= window
        return signal.fillna(False)
    
    def calculate_momentum_signal(self, data, threshold=100, window=5):
        """
        HY spreads widening >threshold basis points in window days
        
        Args:
            data: DataFrame with 'BAMLH0A0HYM2' column
            threshold: Widening threshold in basis points (default 100bp)
            window: Days over which to measure change
            
        Returns:
            Boolean Series indicating signal active
        """
        spread_change = data['BAMLH0A0HYM2'].rolling(window=window).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else np.nan
        )
        signal = spread_change > threshold
        return signal.fillna(False)
    
    def calculate_signal(self, data, threshold=None, window=None):
        """
        Calculate credit spread signal based on type
        """
        if threshold is None:
            threshold = 600 if self.signal_type == "absolute" else 100
        if window is None:
            window = self.window
            
        if self.signal_type == "absolute":
            return self.calculate_absolute_signal(data, threshold, window)
        else:
            return self.calculate_momentum_signal(data, threshold, window)
    
    def optimize_threshold(self, train_data, train_crisis_labels):
        """
        Optimize threshold for credit spread signal
        """
        if self.signal_type == "absolute":
            # Test spread levels from 400bp to 800bp
            threshold_range = np.arange(400, 801, 50)
        else:
            # Test momentum thresholds from 50bp to 200bp
            threshold_range = np.arange(50, 201, 25)
            
        results = []
        best_auc = 0
        best_threshold = threshold_range[0]
        
        for threshold in threshold_range:
            signals = self.calculate_signal(train_data, threshold=threshold)
            
            # Calculate performance metrics
            tp = np.sum(signals & train_crisis_labels)
            fp = np.sum(signals & ~train_crisis_labels)
            fn = np.sum(~signals & train_crisis_labels)
            tn = np.sum(~signals & ~train_crisis_labels)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Simple AUC approximation
            tpr = recall
            auc = 0.5 + (tpr - fpr) / 2
            
            results.append({
                'threshold': threshold,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'fire_rate': np.sum(signals) / len(signals)
            })
            
            if auc > best_auc and precision > 0.1:  # Minimum precision filter
                best_auc = auc
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_threshold, best_auc, results
    
    def calculate_metrics(self, signals, crisis_labels, period_name=""):
        """Calculate comprehensive performance metrics"""
        tp = np.sum(signals & crisis_labels)
        fp = np.sum(signals & ~crisis_labels)
        fn = np.sum(~signals & crisis_labels)
        tn = np.sum(~signals & ~crisis_labels)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        signal_fire_rate = np.sum(signals) / len(signals)
        crisis_fire_rate = np.sum(signals[crisis_labels]) / np.sum(crisis_labels) if np.sum(crisis_labels) > 0 else 0
        normal_fire_rate = np.sum(signals[~crisis_labels]) / np.sum(~crisis_labels) if np.sum(~crisis_labels) > 0 else 0
        crisis_specificity = crisis_fire_rate / normal_fire_rate if normal_fire_rate > 0 else float('inf')
        
        return {
            'signal_name': self.name,
            'signal_type': self.signal_type,
            'period_name': period_name,
            'threshold': self.threshold,
            'window': self.window,
            
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            
            'signal_fire_rate': signal_fire_rate,
            'crisis_specificity': crisis_specificity,
            
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            
            'passes_precision': precision > 0.30,
            'passes_fpr': fpr < 0.15,
            'passes_specificity': crisis_specificity > 3.0,
            'passes_all': precision > 0.30 and fpr < 0.15 and crisis_specificity > 3.0
        }


def generate_hy_spread_data():
    """
    Generate realistic HY credit spread data for demonstration
    
    In real implementation, this would load from:
    - data/fred_cache/BAMLH0A0HYM2.csv (existing cache)
    - services/free_data.py (FRED data fetcher)
    """
    np.random.seed(42)
    dates = pd.date_range('2012-01-01', '2024-12-31', freq='D')
    
    # Base HY spread around 400bp with some noise
    base_spread = 400 + 50 * np.random.randn(len(dates))
    
    # Add crisis periods with spread widening
    crisis_periods = [
        ('2020-02-01', '2020-05-01', 600),  # COVID crisis - spreads to 1000bp+
        ('2022-01-01', '2022-03-01', 200),  # Rate shock - moderate widening
        ('2024-07-01', '2024-08-15', 150)   # Recent stress - mild widening
    ]
    
    for start, end, additional_spread in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        # Add exponential spread widening during crisis
        crisis_spreads = np.random.exponential(additional_spread, np.sum(mask))
        base_spread[mask] += crisis_spreads
    
    # Ensure spreads stay realistic (150bp min, 1500bp max)
    spreads = np.maximum(base_spread, 150)
    spreads = np.minimum(spreads, 1500)
    
    # Add some mean reversion
    spreads = pd.Series(spreads, index=dates).ewm(span=10).mean()
    
    data = pd.DataFrame({
        'BAMLH0A0HYM2': spreads
    }, index=dates)
    
    return data


def create_crisis_labels(data):
    """Create crisis labels for credit spread validation"""
    crisis_periods = [
        ('2020-01-15', '2020-04-30'),  # COVID crisis
        ('2022-01-01', '2022-03-15'),  # Rate shock
        ('2024-06-15', '2024-08-15')   # Recent stress
    ]
    
    crisis_labels = pd.Series(False, index=data.index)
    
    for start, end in crisis_periods:
        mask = (data.index >= start) & (data.index <= end)
        crisis_labels.loc[mask] = True
    
    return crisis_labels


def validate_credit_signals():
    """
    Validate both absolute and momentum credit spread signals
    """
    print("=== HY Credit Spread Signal Validation ===\n")
    
    # Load data
    data = generate_hy_spread_data()
    crisis_labels = create_crisis_labels(data)
    
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"HY Spread range: {data['BAMLH0A0HYM2'].min():.0f}bp - {data['BAMLH0A0HYM2'].max():.0f}bp")
    print(f"Crisis days: {crisis_labels.sum()} ({crisis_labels.mean():.1%})\n")
    
    # Test both signal types
    signal_types = ["absolute", "momentum"]
    all_results = {}
    
    for signal_type in signal_types:
        print(f"=== TESTING {signal_type.upper()} CREDIT SIGNAL ===\n")
        
        signal = HYCreditSpreadSignal(signal_type=signal_type)
        results = []
        
        # Mock validation periods
        periods = [
            {'name': '2019-2020_Crisis', 'train_start': '2012-01-01', 'train_end': '2018-12-31',
             'test_start': '2019-01-01', 'test_end': '2020-12-31'},
            {'name': '2021-2022_Bear', 'train_start': '2012-01-01', 'train_end': '2020-12-31',
             'test_start': '2021-01-01', 'test_end': '2022-12-31'},
            {'name': '2023-2024_Recent', 'train_start': '2012-01-01', 'train_end': '2022-12-31',
             'test_start': '2023-01-01', 'test_end': '2024-12-31'}
        ]
        
        for period in periods:
            print(f"--- {period['name']} ---")
            
            # Training data
            train_mask = (data.index >= period['train_start']) & (data.index <= period['train_end'])
            train_data = data[train_mask]
            train_crisis = crisis_labels[train_mask]
            
            # Optimize threshold
            optimal_threshold, best_auc, _ = signal.optimize_threshold(train_data, train_crisis)
            print(f"Optimal {signal_type} threshold: {optimal_threshold:.0f}bp (AUC: {best_auc:.3f})")
            
            # Test data
            test_mask = (data.index >= period['test_start']) & (data.index <= period['test_end'])
            test_data = data[test_mask]
            test_crisis = crisis_labels[test_mask]
            
            # Calculate signals and metrics
            test_signals = signal.calculate_signal(test_data)
            metrics = signal.calculate_metrics(test_signals, test_crisis, period['name'])
            results.append(metrics)
            
            # Print results
            print(f"Precision: {metrics['precision']:.1%}")
            print(f"FPR: {metrics['false_positive_rate']:.1%}")
            print(f"Fire Rate: {metrics['signal_fire_rate']:.1%}")
            print(f"Crisis Specificity: {metrics['crisis_specificity']:.1f}x")
            print(f"PASS: {'✅' if metrics['passes_all'] else '❌'}")
            print()
        
        # Summary for this signal type
        avg_precision = np.mean([r['precision'] for r in results])
        avg_fpr = np.mean([r['false_positive_rate'] for r in results])
        avg_specificity = np.mean([r['crisis_specificity'] for r in results])
        
        print(f"{signal_type.upper()} SIGNAL SUMMARY:")
        print(f"Average Precision: {avg_precision:.1%}")
        print(f"Average FPR: {avg_fpr:.1%}")
        print(f"Average Specificity: {avg_specificity:.1f}x")
        
        overall_pass = avg_precision > 0.30 and avg_fpr < 0.15 and avg_specificity > 3.0
        print(f"Overall Assessment: {'PASS ✅' if overall_pass else 'FAIL ❌'}")
        print("-" * 60)
        
        all_results[signal_type] = {
            'results': results,
            'summary': {
                'avg_precision': avg_precision,
                'avg_fpr': avg_fpr,
                'avg_specificity': avg_specificity,
                'overall_pass': overall_pass
            }
        }
    
    # Compare signal types
    print("\n=== SIGNAL TYPE COMPARISON ===")
    
    abs_summary = all_results['absolute']['summary']
    mom_summary = all_results['momentum']['summary']
    
    print(f"Absolute Level Signal:")
    print(f"  Precision: {abs_summary['avg_precision']:.1%}")
    print(f"  FPR: {abs_summary['avg_fpr']:.1%}")
    print(f"  Pass: {'✅' if abs_summary['overall_pass'] else '❌'}")
    
    print(f"\nMomentum Signal:")
    print(f"  Precision: {mom_summary['avg_precision']:.1%}")  
    print(f"  FPR: {mom_summary['avg_fpr']:.1%}")
    print(f"  Pass: {'✅' if mom_summary['overall_pass'] else '❌'}")
    
    # Recommendation
    if abs_summary['overall_pass'] and mom_summary['overall_pass']:
        best = 'absolute' if abs_summary['avg_precision'] > mom_summary['avg_precision'] else 'momentum'
        print(f"\n🏆 RECOMMENDATION: Deploy {best} signal (higher precision)")
    elif abs_summary['overall_pass']:
        print(f"\n✅ RECOMMENDATION: Deploy absolute level signal only")
    elif mom_summary['overall_pass']:
        print(f"\n✅ RECOMMENDATION: Deploy momentum signal only")
    else:
        print(f"\n❌ RECOMMENDATION: Both signals fail validation")
    
    # Save results
    with open('hy_credit_signals_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: hy_credit_signals_validation.json")
    
    return all_results


if __name__ == "__main__":
    results = validate_credit_signals()