#!/usr/bin/env python3
"""
Sample Implementation: VIX Persistent Elevation Signal

This demonstrates how to implement and validate a single signal using the validation framework.
This signal tests the hypothesis that sustained VIX elevation (>25 for 10+ days) indicates market stress.

Key Features:
- Uses validation_framework.py for proper testing
- Walk-forward validation across 3 periods  
- ROC-based threshold optimization
- Comprehensive metrics tracking
- Economic interpretability

Usage:
    python sample_vix_signal.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import validation framework (assuming it exists)
try:
    from validation_framework import ValidationFramework, ValidationPeriod
except ImportError:
    print("Warning: validation_framework.py not found. Using mock implementation.")
    # Mock implementation for demonstration
    class ValidationFramework:
        def __init__(self):
            self.validation_periods = [
                type('Period', (), {
                    'name': '2019-2020_Crisis',
                    'train_start': '2012-01-01',
                    'train_end': '2018-12-31', 
                    'test_start': '2019-01-01',
                    'test_end': '2020-12-31'
                })(),
                type('Period', (), {
                    'name': '2021-2022_Bear',
                    'train_start': '2012-01-01',
                    'train_end': '2020-12-31',
                    'test_start': '2021-01-01', 
                    'test_end': '2022-12-31'
                })(),
                type('Period', (), {
                    'name': '2023-2024_Recent',
                    'train_start': '2012-01-01',
                    'train_end': '2022-12-31',
                    'test_start': '2023-01-01',
                    'test_end': '2024-12-31'
                })()
            ]


class VIXPersistentElevationSignal:
    """
    VIX Persistent Elevation Signal
    
    Hypothesis: Sustained VIX >threshold for consecutive days indicates market stress
    
    Economic Logic:
    - VIX represents implied volatility/fear in market
    - Brief spikes common, sustained elevation rarer and more meaningful
    - Persistent fear indicates structural concerns, not just noise
    - Historical crises often involve extended volatility periods
    """
    
    def __init__(self, threshold=25, window=10):
        self.name = "VIX_persistent_elevation"
        self.threshold = threshold
        self.window = window
        self.validation_metrics = None
        
    def calculate_signal(self, data, threshold=None, window=None):
        """
        Calculate VIX persistent elevation signal
        
        Args:
            data: DataFrame with 'VIX' column and date index
            threshold: VIX threshold (default: self.threshold)
            window: Consecutive days requirement (default: self.window)
            
        Returns:
            Boolean Series indicating signal active
        """
        if threshold is None:
            threshold = self.threshold
        if window is None:
            window = self.window
            
        # VIX above threshold
        vix_above = data['VIX'] > threshold
        
        # Count consecutive days above threshold
        consecutive_count = vix_above.rolling(window=window, min_periods=window).sum()
        
        # Signal active when we have 'window' consecutive days above threshold
        signal = consecutive_count >= window
        
        return signal.fillna(False)
    
    def optimize_threshold(self, train_data, train_crisis_labels, threshold_range=None):
        """
        Optimize threshold using ROC analysis on training data
        
        Args:
            train_data: Training DataFrame with VIX data
            train_crisis_labels: Boolean array of crisis days
            threshold_range: Range of thresholds to test
            
        Returns:
            optimal_threshold, best_auc, threshold_results
        """
        if threshold_range is None:
            threshold_range = np.arange(20, 36, 1)  # Test VIX 20-35
            
        results = []
        best_auc = 0
        best_threshold = threshold_range[0]
        
        for threshold in threshold_range:
            signals = self.calculate_signal(train_data, threshold=threshold)
            
            # Calculate AUC (area under ROC curve)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(train_crisis_labels, signals.astype(int))
            except:
                # Manual AUC calculation if sklearn not available
                tp_rate = np.sum(signals & train_crisis_labels) / np.sum(train_crisis_labels)
                fp_rate = np.sum(signals & ~train_crisis_labels) / np.sum(~train_crisis_labels)
                auc = 0.5 + (tp_rate - fp_rate) / 2  # Simplified AUC approximation
            
            # Calculate basic metrics
            tp = np.sum(signals & train_crisis_labels)
            fp = np.sum(signals & ~train_crisis_labels)
            fn = np.sum(~signals & train_crisis_labels)
            tn = np.sum(~signals & ~train_crisis_labels)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fire_rate = np.sum(signals) / len(signals)
            
            results.append({
                'threshold': threshold,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'fire_rate': fire_rate,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            })
            
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold
        
        return best_threshold, best_auc, results
    
    def calculate_metrics(self, signals, crisis_labels, period_name=""):
        """
        Calculate comprehensive signal performance metrics
        
        Args:
            signals: Boolean array of signal activations
            crisis_labels: Boolean array of crisis days
            period_name: Name of validation period
            
        Returns:
            Dictionary of performance metrics
        """
        tp = np.sum(signals & crisis_labels)
        fp = np.sum(signals & ~crisis_labels)
        fn = np.sum(~signals & crisis_labels)
        tn = np.sum(~signals & ~crisis_labels)
        
        # Core metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Business metrics
        signal_fire_rate = np.sum(signals) / len(signals)
        crisis_fire_rate = np.sum(signals[crisis_labels]) / np.sum(crisis_labels) if np.sum(crisis_labels) > 0 else 0
        normal_fire_rate = np.sum(signals[~crisis_labels]) / np.sum(~crisis_labels) if np.sum(~crisis_labels) > 0 else 0
        crisis_specificity = crisis_fire_rate / normal_fire_rate if normal_fire_rate > 0 else float('inf')
        
        return {
            'signal_name': self.name,
            'period_name': period_name,
            'threshold': self.threshold,
            'window': self.window,
            
            # Core metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_negative_rate': tnr,
            
            # Business metrics
            'signal_fire_rate': signal_fire_rate,
            'crisis_fire_rate': crisis_fire_rate,
            'normal_fire_rate': normal_fire_rate,
            'crisis_specificity': crisis_specificity,
            
            # Raw counts
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_signals': int(tp + fp),
            'total_crisis_days': int(tp + fn),
            'total_normal_days': int(tn + fp),
            
            # Success criteria
            'passes_precision': precision > 0.30,
            'passes_fpr': fpr < 0.15,
            'passes_specificity': crisis_specificity > 3.0,
            'passes_all': precision > 0.30 and fpr < 0.15 and crisis_specificity > 3.0
        }


def load_sample_data():
    """
    Load or generate sample VIX data for demonstration
    
    In real implementation, this would load from:
    - yfinance for VIX data
    - FRED cache for macro data
    - HMM state history
    """
    # Generate sample VIX data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2012-01-01', '2024-12-31', freq='D')
    
    # Simulate VIX with occasional spikes
    base_vix = 15 + 5 * np.random.randn(len(dates))
    
    # Add crisis periods with elevated VIX
    crisis_periods = [
        ('2020-02-15', '2020-04-15'),  # COVID
        ('2022-01-01', '2022-03-01'),  # Rate shock
        ('2024-07-01', '2024-08-15')   # Recent stress
    ]
    
    for start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        base_vix[mask] += np.random.exponential(15, np.sum(mask))
    
    # Ensure VIX stays positive and realistic
    vix_values = np.maximum(base_vix, 9)  # VIX floor of 9
    vix_values = np.minimum(vix_values, 80)  # VIX cap of 80
    
    data = pd.DataFrame({
        'VIX': vix_values
    }, index=dates)
    
    return data


def create_crisis_labels(data):
    """
    Create crisis labels based on known turning points
    
    In real implementation, this would use the crisis periods from validation_framework.py
    """
    crisis_periods = [
        ('2020-01-20', '2020-04-20'),  # COVID crisis ±30 days
        ('2022-01-01', '2022-03-01'),  # Rate shock  
        ('2024-06-15', '2024-08-15')   # Recent stress
    ]
    
    crisis_labels = pd.Series(False, index=data.index)
    
    for start, end in crisis_periods:
        mask = (data.index >= start) & (data.index <= end)
        crisis_labels.loc[mask] = True
    
    return crisis_labels


def validate_vix_signal():
    """
    Complete validation of VIX persistent elevation signal
    """
    print("=== VIX Persistent Elevation Signal Validation ===\n")
    
    # Load data
    print("Loading data...")
    data = load_sample_data()
    crisis_labels = create_crisis_labels(data)
    
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Total trading days: {len(data)}")
    print(f"Crisis days: {crisis_labels.sum()} ({crisis_labels.mean():.1%})")
    print(f"Normal days: {(~crisis_labels).sum()} ({(~crisis_labels).mean():.1%})\n")
    
    # Initialize signal
    signal = VIXPersistentElevationSignal()
    
    # Validation framework
    vf = ValidationFramework()
    results = []
    
    for period in vf.validation_periods:
        print(f"=== {period.name} ===")
        
        # Get training data
        train_mask = (data.index >= period.train_start) & (data.index <= period.train_end)
        train_data = data[train_mask]
        train_crisis = crisis_labels[train_mask]
        
        print(f"Training: {period.train_start} to {period.train_end}")
        print(f"Training days: {len(train_data)} ({train_crisis.sum()} crisis)")
        
        # Optimize threshold on training data
        print("Optimizing threshold...")
        optimal_threshold, best_auc, threshold_results = signal.optimize_threshold(
            train_data, train_crisis, threshold_range=np.arange(20, 36, 2)
        )
        
        print(f"Optimal threshold: VIX >{optimal_threshold} (AUC: {best_auc:.3f})")
        signal.threshold = optimal_threshold
        
        # Test on validation period
        test_mask = (data.index >= period.test_start) & (data.index <= period.test_end)
        test_data = data[test_mask]
        test_crisis = crisis_labels[test_mask]
        
        print(f"Testing: {period.test_start} to {period.test_end}")
        print(f"Test days: {len(test_data)} ({test_crisis.sum()} crisis)")
        
        # Calculate test signals
        test_signals = signal.calculate_signal(test_data)
        
        # Calculate metrics
        metrics = signal.calculate_metrics(test_signals, test_crisis, period.name)
        results.append(metrics)
        
        # Print results
        print(f"\nTest Results:")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.1%}")
        print(f"  Signal Fire Rate: {metrics['signal_fire_rate']:.1%}")
        print(f"  Crisis Specificity: {metrics['crisis_specificity']:.1f}x")
        
        # Success criteria
        print(f"\nSuccess Criteria:")
        print(f"  Precision >30%: {'✅' if metrics['passes_precision'] else '❌'}")
        print(f"  FPR <15%: {'✅' if metrics['passes_fpr'] else '❌'}")  
        print(f"  Crisis Specificity >3x: {'✅' if metrics['passes_specificity'] else '❌'}")
        print(f"  OVERALL PASS: {'✅' if metrics['passes_all'] else '❌'}")
        print("-" * 50)
    
    # Cross-period analysis
    print(f"\n=== CROSS-PERIOD ANALYSIS ===")
    
    avg_precision = np.mean([r['precision'] for r in results])
    avg_fpr = np.mean([r['false_positive_rate'] for r in results])
    avg_specificity = np.mean([r['crisis_specificity'] for r in results])
    
    precision_std = np.std([r['precision'] for r in results])
    temporal_stability = precision_std < 0.15
    
    print(f"Average Precision: {avg_precision:.1%} ± {precision_std:.1%}")
    print(f"Average FPR: {avg_fpr:.1%}")
    print(f"Average Crisis Specificity: {avg_specificity:.1f}x")
    print(f"Temporal Stability (std <15%): {'✅' if temporal_stability else '❌'}")
    
    # Overall assessment
    overall_success = (
        avg_precision > 0.30 and
        avg_fpr < 0.15 and
        temporal_stability and
        all(r['crisis_specificity'] > 3.0 for r in results)
    )
    
    print(f"\n=== FINAL ASSESSMENT ===")
    print(f"Signal Quality: {'EXCELLENT' if avg_precision > 0.50 and avg_fpr < 0.10 else 'GOOD' if overall_success else 'POOR'}")
    print(f"Recommendation: {'DEPLOY' if overall_success else 'REJECT'}")
    
    if overall_success:
        print(f"\n✅ Signal PASSES validation framework")
        print(f"Ready for integration into QIR system")
        print(f"Validated threshold: VIX >{signal.threshold} for {signal.window}+ days")
    else:
        print(f"\n❌ Signal FAILS validation framework")
        print(f"Requires further development or rejection")
    
    # Save results
    with open('vix_signal_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: vix_signal_validation_results.json")
    
    return results, overall_success


if __name__ == "__main__":
    results, success = validate_vix_signal()