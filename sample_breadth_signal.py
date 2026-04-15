#!/usr/bin/env python3
"""
Sample Implementation: Market Breadth Breakdown Signal

This demonstrates a market structure signal that monitors the percentage of S&P 500
stocks trading above their 200-day moving average combined with index technical breakdown.

Signal Logic:
- Monitors deterioration in market breadth (% stocks above 200MA)
- Combines with SPY breaking below 200MA for confirmation
- Captures market concentration risk and broad-based weakness

Economic Foundation:
- Market can be held up by few large stocks while majority weakens
- Breadth deterioration often precedes major market declines
- Concentration in mega-caps creates vulnerability to broad selloffs
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MarketBreadthSignal:
    """
    Market Breadth Breakdown Signal
    
    Economic Logic:
    - Market breadth reflects participation across stocks
    - Narrow leadership (few stocks up) indicates vulnerability
    - Combined with technical breakdown shows broad-based weakness
    - Leading indicator of broader market stress
    
    Signal Components:
    1. Breadth deterioration: <X% of SPX stocks above 200MA
    2. Technical breakdown: SPY below its 200MA
    3. Persistence: Condition sustained for multiple days
    """
    
    def __init__(self):
        self.name = "market_breadth_breakdown"
        self.breadth_threshold = 30  # % of stocks above 200MA
        self.persistence_days = 3    # Days condition must persist
        self.validation_metrics = None
        
    def calculate_signal(self, data, breadth_threshold=None, persistence_days=None):
        """
        Calculate market breadth breakdown signal
        
        Args:
            data: DataFrame with columns:
                - 'SPY': SPY price data
                - 'SPY_200MA': SPY 200-day moving average
                - 'breadth_pct': % of SPX stocks above 200MA (0-100)
            breadth_threshold: Breadth threshold (default 30%)
            persistence_days: Days condition must persist (default 3)
            
        Returns:
            Boolean Series indicating signal active
        """
        if breadth_threshold is None:
            breadth_threshold = self.breadth_threshold
        if persistence_days is None:
            persistence_days = self.persistence_days
            
        # Component 1: Breadth deterioration
        breadth_weak = data['breadth_pct'] < breadth_threshold
        
        # Component 2: Technical breakdown
        technical_breakdown = data['SPY'] < data['SPY_200MA']
        
        # Combined condition
        combined_condition = breadth_weak & technical_breakdown
        
        # Persistence requirement
        if persistence_days > 1:
            persistent_condition = combined_condition.rolling(
                window=persistence_days, min_periods=persistence_days
            ).sum() >= persistence_days
            signal = persistent_condition.fillna(False)
        else:
            signal = combined_condition.fillna(False)
            
        return signal
    
    def calculate_breadth_only_signal(self, data, breadth_threshold=20, persistence_days=5):
        """
        Alternative signal using breadth deterioration only (more sensitive)
        """
        breadth_weak = data['breadth_pct'] < breadth_threshold
        
        if persistence_days > 1:
            signal = breadth_weak.rolling(
                window=persistence_days, min_periods=persistence_days
            ).sum() >= persistence_days
            return signal.fillna(False)
        else:
            return breadth_weak.fillna(False)
    
    def optimize_thresholds(self, train_data, train_crisis_labels, signal_type="combined"):
        """
        Optimize breadth threshold and persistence parameters
        
        Args:
            signal_type: "combined" or "breadth_only"
        """
        results = []
        best_auc = 0
        best_params = {}
        
        if signal_type == "combined":
            breadth_range = np.arange(20, 41, 5)  # Test 20%, 25%, 30%, 35%, 40%
            persistence_range = [1, 3, 5]
            
            for breadth_thresh in breadth_range:
                for persist_days in persistence_range:
                    signals = self.calculate_signal(
                        train_data, 
                        breadth_threshold=breadth_thresh,
                        persistence_days=persist_days
                    )
                    
                    # Calculate metrics
                    tp = np.sum(signals & train_crisis_labels)
                    fp = np.sum(signals & ~train_crisis_labels)
                    fn = np.sum(~signals & train_crisis_labels)
                    tn = np.sum(~signals & ~train_crisis_labels)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    # AUC approximation
                    auc = 0.5 + (recall - fpr) / 2
                    
                    results.append({
                        'breadth_threshold': breadth_thresh,
                        'persistence_days': persist_days,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'fpr': fpr
                    })
                    
                    if auc > best_auc and precision > 0.1:
                        best_auc = auc
                        best_params = {
                            'breadth_threshold': breadth_thresh,
                            'persistence_days': persist_days
                        }
        
        else:  # breadth_only
            breadth_range = np.arange(15, 31, 5)  # Test 15%, 20%, 25%, 30%
            persistence_range = [3, 5, 10]
            
            for breadth_thresh in breadth_range:
                for persist_days in persistence_range:
                    signals = self.calculate_breadth_only_signal(
                        train_data,
                        breadth_threshold=breadth_thresh,
                        persistence_days=persist_days
                    )
                    
                    # Calculate metrics (same as above)
                    tp = np.sum(signals & train_crisis_labels)
                    fp = np.sum(signals & ~train_crisis_labels)
                    fn = np.sum(~signals & train_crisis_labels)
                    tn = np.sum(~signals & ~train_crisis_labels)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    auc = 0.5 + (recall - fpr) / 2
                    
                    results.append({
                        'breadth_threshold': breadth_thresh,
                        'persistence_days': persist_days,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'fpr': fpr
                    })
                    
                    if auc > best_auc and precision > 0.1:
                        best_auc = auc
                        best_params = {
                            'breadth_threshold': breadth_thresh,
                            'persistence_days': persist_days
                        }
        
        # Update signal parameters
        self.breadth_threshold = best_params.get('breadth_threshold', self.breadth_threshold)
        self.persistence_days = best_params.get('persistence_days', self.persistence_days)
        
        return best_params, best_auc, results
    
    def calculate_metrics(self, signals, crisis_labels, period_name="", signal_type="combined"):
        """Calculate performance metrics for breadth signal"""
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
            'signal_type': signal_type,
            'period_name': period_name,
            'breadth_threshold': self.breadth_threshold,
            'persistence_days': self.persistence_days,
            
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


def generate_breadth_data():
    """
    Generate market breadth and SPY data for demonstration
    
    In real implementation, this would load from:
    - services/market_data.py (fetch_breadth_pct function)
    - yfinance SPY data with calculated 200MA
    """
    np.random.seed(42)
    dates = pd.date_range('2012-01-01', '2024-12-31', freq='D')
    n_days = len(dates)
    
    # Generate SPY prices with trend and volatility
    spy_returns = 0.0003 + 0.01 * np.random.randn(n_days)  # ~8% annual return, 16% vol
    spy_price = 100 * np.exp(np.cumsum(spy_returns))
    
    # Calculate 200-day MA
    spy_200ma = pd.Series(spy_price).rolling(window=200, min_periods=200).mean()
    
    # Generate market breadth (% stocks above 200MA)
    # Base breadth correlated with market performance but with additional noise
    base_breadth = 50 + 30 * np.tanh(spy_returns * 50)  # Range roughly 20-80%
    breadth_noise = 10 * np.random.randn(n_days)
    breadth_pct = np.clip(base_breadth + breadth_noise, 5, 95)
    
    # Add crisis periods with deteriorating breadth
    crisis_periods = [
        ('2020-02-15', '2020-04-15'),  # COVID crisis - breadth collapse
        ('2022-01-01', '2022-06-01'),  # Rate shock - gradual deterioration
        ('2024-07-01', '2024-08-15')   # Recent stress - mild deterioration
    ]
    
    for start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        if np.sum(mask) > 0:
            # During crisis: breadth deteriorates, prices decline
            crisis_days = np.sum(mask)
            breadth_decline = np.linspace(0, -40, crisis_days)  # Up to 40% decline in breadth
            breadth_pct[mask] = np.maximum(breadth_pct[mask] + breadth_decline, 5)
            
            # SPY also declines during crisis
            spy_decline = np.linspace(0, -0.20, crisis_days)  # Up to 20% decline
            spy_price[mask] *= (1 + spy_decline)
    
    # Smooth the data
    breadth_pct = pd.Series(breadth_pct, index=dates).ewm(span=5).mean()
    spy_price = pd.Series(spy_price, index=dates).ewm(span=2).mean()
    spy_200ma = spy_price.rolling(window=200, min_periods=200).mean()
    
    data = pd.DataFrame({
        'SPY': spy_price,
        'SPY_200MA': spy_200ma,
        'breadth_pct': breadth_pct
    }, index=dates)
    
    return data


def create_crisis_labels(data):
    """Create crisis labels for breadth signal validation"""
    crisis_periods = [
        ('2020-01-20', '2020-04-30'),  # COVID crisis
        ('2022-01-15', '2022-06-15'),  # Rate shock / bear market
        ('2024-07-01', '2024-08-15')   # Recent stress
    ]
    
    crisis_labels = pd.Series(False, index=data.index)
    
    for start, end in crisis_periods:
        mask = (data.index >= start) & (data.index <= end)
        crisis_labels.loc[mask] = True
    
    return crisis_labels


def validate_breadth_signals():
    """
    Validate both combined and breadth-only market signals
    """
    print("=== Market Breadth Breakdown Signal Validation ===\n")
    
    # Load data
    data = generate_breadth_data()
    crisis_labels = create_crisis_labels(data)
    
    # Remove NaN values (from 200MA calculation)
    valid_mask = ~data.isna().any(axis=1)
    data = data[valid_mask]
    crisis_labels = crisis_labels[valid_mask]
    
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Valid trading days: {len(data)}")
    print(f"Breadth range: {data['breadth_pct'].min():.1f}% - {data['breadth_pct'].max():.1f}%")
    print(f"Crisis days: {crisis_labels.sum()} ({crisis_labels.mean():.1%})\n")
    
    # Test both signal variants
    signal_types = ["combined", "breadth_only"]
    all_results = {}
    
    for signal_type in signal_types:
        print(f"=== TESTING {signal_type.upper().replace('_', ' ')} SIGNAL ===\n")
        
        signal = MarketBreadthSignal()
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
            
            if len(train_data) == 0:
                print("No training data available for this period")
                continue
                
            # Optimize parameters
            best_params, best_auc, _ = signal.optimize_thresholds(
                train_data, train_crisis, signal_type=signal_type
            )
            
            print(f"Optimal params: breadth<{signal.breadth_threshold}%, {signal.persistence_days}d persist (AUC: {best_auc:.3f})")
            
            # Test data
            test_mask = (data.index >= period['test_start']) & (data.index <= period['test_end'])
            test_data = data[test_mask]
            test_crisis = crisis_labels[test_mask]
            
            if len(test_data) == 0:
                print("No test data available for this period")
                continue
            
            # Calculate signals based on type
            if signal_type == "combined":
                test_signals = signal.calculate_signal(test_data)
            else:
                test_signals = signal.calculate_breadth_only_signal(test_data)
                
            metrics = signal.calculate_metrics(test_signals, test_crisis, period['name'], signal_type)
            results.append(metrics)
            
            # Print results
            print(f"Precision: {metrics['precision']:.1%}")
            print(f"FPR: {metrics['false_positive_rate']:.1%}")
            print(f"Fire Rate: {metrics['signal_fire_rate']:.1%}")
            print(f"Crisis Specificity: {metrics['crisis_specificity']:.1f}x")
            print(f"PASS: {'✅' if metrics['passes_all'] else '❌'}")
            print()
        
        if len(results) == 0:
            print("No valid results for this signal type")
            continue
            
        # Summary for this signal type
        avg_precision = np.mean([r['precision'] for r in results])
        avg_fpr = np.mean([r['false_positive_rate'] for r in results])
        avg_specificity = np.mean([r['crisis_specificity'] for r in results])
        avg_fire_rate = np.mean([r['signal_fire_rate'] for r in results])
        
        print(f"{signal_type.upper().replace('_', ' ')} SIGNAL SUMMARY:")
        print(f"Average Precision: {avg_precision:.1%}")
        print(f"Average FPR: {avg_fpr:.1%}")
        print(f"Average Fire Rate: {avg_fire_rate:.1%}")
        print(f"Average Specificity: {avg_specificity:.1f}x")
        
        overall_pass = avg_precision > 0.30 and avg_fpr < 0.15 and avg_specificity > 3.0
        print(f"Overall Assessment: {'PASS ✅' if overall_pass else 'FAIL ❌'}")
        print("-" * 60)
        
        all_results[signal_type] = {
            'results': results,
            'summary': {
                'avg_precision': avg_precision,
                'avg_fpr': avg_fpr,
                'avg_fire_rate': avg_fire_rate,
                'avg_specificity': avg_specificity,
                'overall_pass': overall_pass
            }
        }
    
    # Compare signal variants
    print("\n=== SIGNAL VARIANT COMPARISON ===")
    
    if 'combined' in all_results and 'breadth_only' in all_results:
        combined_summary = all_results['combined']['summary']
        breadth_summary = all_results['breadth_only']['summary']
        
        print(f"Combined Signal (Breadth + Technical):")
        print(f"  Precision: {combined_summary['avg_precision']:.1%}")
        print(f"  FPR: {combined_summary['avg_fpr']:.1%}")
        print(f"  Fire Rate: {combined_summary['avg_fire_rate']:.1%}")
        print(f"  Pass: {'✅' if combined_summary['overall_pass'] else '❌'}")
        
        print(f"\nBreadth Only Signal:")
        print(f"  Precision: {breadth_summary['avg_precision']:.1%}")
        print(f"  FPR: {breadth_summary['avg_fpr']:.1%}")
        print(f"  Fire Rate: {breadth_summary['avg_fire_rate']:.1%}")
        print(f"  Pass: {'✅' if breadth_summary['overall_pass'] else '❌'}")
        
        # Recommendation
        if combined_summary['overall_pass'] and breadth_summary['overall_pass']:
            if combined_summary['avg_precision'] > breadth_summary['avg_precision']:
                print(f"\n🏆 RECOMMENDATION: Deploy combined signal (higher precision, lower noise)")
            else:
                print(f"\n🏆 RECOMMENDATION: Deploy breadth-only signal (higher precision)")
        elif combined_summary['overall_pass']:
            print(f"\n✅ RECOMMENDATION: Deploy combined signal only")
        elif breadth_summary['overall_pass']:
            print(f"\n✅ RECOMMENDATION: Deploy breadth-only signal only")
        else:
            print(f"\n❌ RECOMMENDATION: Both signals fail validation")
    
    # Save results
    with open('breadth_signals_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: breadth_signals_validation.json")
    
    return all_results


if __name__ == "__main__":
    results = validate_breadth_signals()