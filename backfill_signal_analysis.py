#!/usr/bin/env python3
"""
Signal Strength Historical Backfill Analysis

Re-runs the top/bottom proximity signal calculations against historical crash data
to enable threshold analysis and signal predictive power assessment.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class SignalData:
    """Container for individual signal calculations"""
    name: str
    value: float
    threshold_met: bool

@dataclass
class ProximityAnalysis:
    """Container for complete top/bottom proximity analysis"""
    top_signals: List[SignalData]
    bottom_signals: List[SignalData] 
    top_score: float
    bottom_score: float
    top_count: int
    bottom_count: int

def calculate_signal_strength(regime_score: float, macro_score: float, conviction: float, 
                            entropy: float, ll_zscore: float, hmm_state: str,
                            wyckoff_phase: str = None, wyckoff_conf: float = 0,
                            wyckoff_sub: str = None) -> ProximityAnalysis:
    """
    Recalculate top/bottom proximity signals using historical data
    
    Based on modules/quick_run.py lines 2244-2367
    """
    
    # Input parameters (mapped from historical data)
    _tb_regime = float(regime_score or 0)
    _tb_macro = float(macro_score or 50)  
    _tb_conv = float(conviction or 0)
    _tb_entropy = float(entropy or 0)
    _tb_ll_z = float(ll_zscore or 0)
    _tb_hmm_label = str(hmm_state or "")
    
    # Velocity calculation (simplified - using 0 for historical data)
    _tb_vel = 0.0
    
    # Wyckoff parameters
    _tb_wyckoff = {
        "phase": wyckoff_phase,
        "confidence": wyckoff_conf,
        "sub_phase": wyckoff_sub,
        "resistance": None,  # Not available in historical data
        "support": None,     # Not available in historical data
        "cause_target": None,
        "spy_last": None
    } if wyckoff_phase else None
    
    # For historical data, we don't have HY spreads, AAII sentiment, or breadth
    # These would require additional data sources
    _tb_hy = None
    _tb_aaii = {}
    _tb_breadth = None
    
    _top_signals = []
    _bottom_signals = []
    
    # ──────────────────────────────────────────────────────────────────────
    # TOP SIGNALS — calibrated thresholds (empirical avg at 8 known peaks)
    # ──────────────────────────────────────────────────────────────────────
    
    # regime avg=+0.14 (88% hit), entropy avg=0.71 (75%), conviction avg=17 (75%)
    # ll_z avg=-6.0 — tightened from -0.5 to -3.0 to reduce false fires
    if _tb_regime > 0.05:
        value = min(100, _tb_regime * 180)
        _top_signals.append(SignalData("Regime elevated", value, value >= 50))
        
    if _tb_vel < -3:
        value = min(100, abs(_tb_vel) * 5)
        _top_signals.append(SignalData("Velocity turning negative", value, value >= 50))
        
    if _tb_entropy > 0.68:
        value = min(100, (_tb_entropy - 0.45) * 200)
        _top_signals.append(SignalData("High regime entropy", value, value >= 50))
        
    if _tb_conv < 22:
        value = min(100, (22 - _tb_conv) * 5)
        _top_signals.append(SignalData("Low conviction", value, value >= 50))
        
    if _tb_ll_z < -3.0:
        value = min(100, abs(_tb_ll_z) * 8)
        _top_signals.append(SignalData("LL deteriorating", value, value >= 50))
        
    # Late Cycle → top only (not bottom) — empirically fires at peaks
    if _tb_hmm_label in ("Late Cycle", "Stress", "Early Stress"):
        value = 65
        _top_signals.append(SignalData(f"HMM {_tb_hmm_label} state", value, value >= 50))
    
    # Wyckoff top signals — only Distribution is reliable (38% hit at peaks)
    if _tb_wyckoff and wyckoff_phase:
        _wk_phase = _tb_wyckoff.get("phase", "")
        _wk_conf = _tb_wyckoff.get("confidence", 0)
        _wk_sub = _tb_wyckoff.get("sub_phase", "")
        
        if _wk_phase == "Distribution":
            value = min(100, _wk_conf)
            _top_signals.append(SignalData(f"Wyckoff Distribution {_wk_sub} ({_wk_conf}% conf)", value, value >= 50))
            
        if _wk_phase == "Markup" and _wk_sub in ("D", "E"):
            value = min(80, _wk_conf)
            _top_signals.append(SignalData(f"Wyckoff Markup late phase {_wk_sub}", value, value >= 50))
    
    # HY Credit Spread signals (placeholder - data not available historically)
    # AAII sentiment signals (placeholder - data not available historically)  
    # Market breadth signals (placeholder - data not available historically)
    
    # ──────────────────────────────────────────────────────────────────────
    # BOTTOM SIGNALS — calibrated thresholds (empirical avg at 8 known troughs)
    # ──────────────────────────────────────────────────────────────────────
    
    # regime avg=-0.35 (100% hit), macro avg=32.8 (88%), conviction avg=34.4 (88%)
    # ll_z avg=-20.9 — tightened from -5 to -8 to reduce noise
    if _tb_regime < -0.17:
        value = min(100, abs(_tb_regime) * 220)
        _bottom_signals.append(SignalData("Regime deep negative", value, value >= 50))
        
    if _tb_vel > 3:
        value = min(100, _tb_vel * 5)
        _bottom_signals.append(SignalData("Velocity turning positive", value, value >= 50))
        
    if _tb_macro < 37:
        value = min(100, (37 - _tb_macro) * 6)
        _bottom_signals.append(SignalData("Macro crushed", value, value >= 50))
        
    if _tb_conv > 24:
        value = min(100, _tb_conv * 2)
        _bottom_signals.append(SignalData("Conviction building", value, value >= 50))
        
    if _tb_ll_z < -8:
        value = min(100, abs(_tb_ll_z) * 3)
        _bottom_signals.append(SignalData("Extreme LL stress", value, value >= 50))
        
    # Crisis only for bottom HMM (not Late Cycle — it's a top indicator)
    if _tb_hmm_label in ("Crisis",):
        value = 75
        _bottom_signals.append(SignalData("HMM Crisis state", value, value >= 50))
    
    # Wyckoff bottom signals
    if _tb_wyckoff and wyckoff_phase:
        _wk_phase = _tb_wyckoff.get("phase", "")
        _wk_conf = _tb_wyckoff.get("confidence", 0)
        _wk_sub = _tb_wyckoff.get("sub_phase", "")
        
        if _wk_phase == "Accumulation":
            value = min(100, _wk_conf)
            _bottom_signals.append(SignalData(f"Wyckoff Accumulation {_wk_sub} ({_wk_conf}% conf)", value, value >= 50))
            
        if _wk_phase == "Markdown" and _wk_sub in ("D", "E"):
            value = min(80, _wk_conf)
            _bottom_signals.append(SignalData(f"Wyckoff Markdown exhaustion {_wk_sub}", value, value >= 50))
    
    # HY Credit, AAII, and breadth signals (placeholders for future enhancement)
    
    # Calculate final scores
    _top_score = round(sum(s.value for s in _top_signals) / max(1, len(_top_signals))) if _top_signals else 0
    _bot_score = round(sum(s.value for s in _bottom_signals) / max(1, len(_bottom_signals))) if _bottom_signals else 0
    
    # Count firing signals (≥50 threshold)  
    _top_count = sum(1 for s in _top_signals if s.threshold_met)
    _bot_count = sum(1 for s in _bottom_signals if s.threshold_met)
    
    return ProximityAnalysis(
        top_signals=_top_signals,
        bottom_signals=_bottom_signals,
        top_score=_top_score,
        bottom_score=_bot_score,
        top_count=_top_count,
        bottom_count=_bot_count
    )

def load_historical_data(file_path: str) -> Dict[str, Any]:
    """Load the proximity calibration data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_signal_performance(backfilled_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze signal performance across historical data
    
    Returns comprehensive analysis including:
    - Signal strength distributions by role (PEAK vs TROUGH)
    - Individual signal predictive power
    - Optimal firing thresholds
    - False positive/negative rates
    """
    
    # Separate peaks and troughs
    peaks = [record for record in backfilled_data if record['role'] == 'PEAK']
    troughs = [record for record in backfilled_data if record['role'] == 'TROUGH']
    
    # Collect all individual signals
    all_top_signals = {}
    all_bottom_signals = {}
    
    # Extract signal data from peaks and troughs
    for record in backfilled_data:
        analysis = record['signal_analysis']
        role = record['role']
        
        # Process top signals
        for signal in analysis['top_signals']:
            name = signal['name']
            if name not in all_top_signals:
                all_top_signals[name] = {'peak_values': [], 'trough_values': []}
            
            if role == 'PEAK':
                all_top_signals[name]['peak_values'].append(signal['value'])
            else:
                all_top_signals[name]['trough_values'].append(signal['value'])
        
        # Process bottom signals  
        for signal in analysis['bottom_signals']:
            name = signal['name']
            if name not in all_bottom_signals:
                all_bottom_signals[name] = {'peak_values': [], 'trough_values': []}
                
            if role == 'TROUGH':
                all_bottom_signals[name]['trough_values'].append(signal['value'])
            else:
                all_bottom_signals[name]['peak_values'].append(signal['value'])
    
    # Calculate signal effectiveness scores
    def calculate_signal_effectiveness(signal_data: Dict, expected_role: str) -> Dict[str, float]:
        """Calculate how well a signal performs for its intended role"""
        if expected_role == 'top':
            good_values = signal_data.get('peak_values', [])
            bad_values = signal_data.get('trough_values', [])
        else:  # bottom
            good_values = signal_data.get('trough_values', [])
            bad_values = signal_data.get('peak_values', [])
        
        if not good_values and not bad_values:
            return {'avg_good': 0, 'avg_bad': 0, 'effectiveness': 0, 'count_good': 0, 'count_bad': 0}
        
        avg_good = np.mean(good_values) if good_values else 0
        avg_bad = np.mean(bad_values) if bad_values else 0
        effectiveness = (avg_good - avg_bad) / 100 if (avg_good + avg_bad) > 0 else 0
        
        return {
            'avg_good': round(avg_good, 1),
            'avg_bad': round(avg_bad, 1), 
            'effectiveness': round(effectiveness, 3),
            'count_good': len(good_values),
            'count_bad': len(bad_values)
        }
    
    # Analyze top signals
    top_signal_analysis = {}
    for signal_name, signal_data in all_top_signals.items():
        top_signal_analysis[signal_name] = calculate_signal_effectiveness(signal_data, 'top')
    
    # Analyze bottom signals
    bottom_signal_analysis = {}
    for signal_name, signal_data in all_bottom_signals.items():
        bottom_signal_analysis[signal_name] = calculate_signal_effectiveness(signal_data, 'bottom')
    
    # Overall score distributions
    peak_top_scores = [record['signal_analysis']['top_score'] for record in peaks]
    peak_bottom_scores = [record['signal_analysis']['bottom_score'] for record in peaks]
    trough_top_scores = [record['signal_analysis']['top_score'] for record in troughs]
    trough_bottom_scores = [record['signal_analysis']['bottom_score'] for record in troughs]
    
    # Threshold analysis
    def analyze_thresholds(scores: List[float], label: str) -> Dict[str, Any]:
        """Analyze various threshold effectiveness"""
        thresholds = [40, 45, 50, 55, 60, 65, 70]
        threshold_stats = {}
        
        for threshold in thresholds:
            hits = sum(1 for score in scores if score >= threshold)
            hit_rate = hits / len(scores) if scores else 0
            threshold_stats[threshold] = {
                'hits': hits,
                'total': len(scores),
                'hit_rate': round(hit_rate, 3)
            }
        
        return {
            'label': label,
            'scores': scores,
            'avg_score': float(round(np.mean(scores), 1)) if scores else 0,
            'max_score': float(round(np.max(scores), 1)) if scores else 0,
            'min_score': float(round(np.min(scores), 1)) if scores else 0,
            'threshold_analysis': threshold_stats
        }
    
    return {
        'summary': {
            'total_records': len(backfilled_data),
            'peaks': len(peaks),
            'troughs': len(troughs)
        },
        'top_signals': {
            'individual_analysis': top_signal_analysis,
            'peak_performance': analyze_thresholds(peak_top_scores, 'Top Signals at Market Peaks'),
            'trough_performance': analyze_thresholds(trough_top_scores, 'Top Signals at Market Troughs (false positives)')
        },
        'bottom_signals': {
            'individual_analysis': bottom_signal_analysis,
            'trough_performance': analyze_thresholds(trough_bottom_scores, 'Bottom Signals at Market Troughs'),
            'peak_performance': analyze_thresholds(peak_bottom_scores, 'Bottom Signals at Market Peaks (false positives)')
        },
        'overall_effectiveness': {
            'top_signal_discrimination': {
                'avg_at_peaks': float(round(np.mean(peak_top_scores), 1)) if peak_top_scores else 0,
                'avg_at_troughs': float(round(np.mean(trough_top_scores), 1)) if trough_top_scores else 0,
                'discrimination_power': float(round(np.mean(peak_top_scores) - np.mean(trough_top_scores), 1)) if peak_top_scores and trough_top_scores else 0
            },
            'bottom_signal_discrimination': {
                'avg_at_troughs': float(round(np.mean(trough_bottom_scores), 1)) if trough_bottom_scores else 0,
                'avg_at_peaks': float(round(np.mean(peak_bottom_scores), 1)) if peak_bottom_scores else 0,
                'discrimination_power': float(round(np.mean(trough_bottom_scores) - np.mean(peak_bottom_scores), 1)) if trough_bottom_scores and peak_bottom_scores else 0
            }
        }
    }

def main():
    """Main execution function"""
    
    print("📊 Signal Strength Historical Backfill Analysis")
    print("=" * 60)
    
    # Load historical data
    data_path = os.path.join(os.path.dirname(__file__), "data", "proximity_calibration.json")
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found")
        return
    
    historical_data = load_historical_data(data_path)
    records = historical_data.get("records", [])
    
    if not records:
        print("❌ Error: No records found in proximity_calibration.json")
        return
    
    print(f"📈 Processing {len(records)} historical records...")
    
    # Backfill signal calculations
    backfilled_data = []
    
    for record in records:
        print(f"  Processing: {record['scenario']} - {record['role']} ({record['date']})")
        
        # Calculate signals using historical data
        analysis = calculate_signal_strength(
            regime_score=record.get('regime_score', 0),
            macro_score=record.get('macro_score', 50),
            conviction=record.get('conviction', 0),
            entropy=record.get('entropy', 0),
            ll_zscore=record.get('ll_zscore', 0),
            hmm_state=record.get('hmm_state', ''),
            wyckoff_phase=record.get('wyckoff_phase'),
            wyckoff_conf=record.get('wyckoff_conf', 0),
            wyckoff_sub=record.get('wyckoff_sub')
        )
        
        # Convert to serializable format
        backfilled_record = {
            **record,  # Include all original data
            'signal_analysis': {
                'top_signals': [
                    {'name': s.name, 'value': s.value, 'threshold_met': s.threshold_met} 
                    for s in analysis.top_signals
                ],
                'bottom_signals': [
                    {'name': s.name, 'value': s.value, 'threshold_met': s.threshold_met}
                    for s in analysis.bottom_signals
                ],
                'top_score': analysis.top_score,
                'bottom_score': analysis.bottom_score,
                'top_count': analysis.top_count,
                'bottom_count': analysis.bottom_count
            }
        }
        
        backfilled_data.append(backfilled_record)
    
    # Perform comprehensive analysis
    print("\n🔍 Analyzing signal performance...")
    performance_analysis = analyze_signal_performance(backfilled_data)
    
    # Combine all results
    final_results = {
        'generated': historical_data.get('generated'),
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methodology': {
            'source_function': '_calculate_top_bottom_proximity from modules/quick_run.py lines 2244-2367',
            'threshold_definition': 'Signals with strength >= 50 are considered "firing"',
            'score_calculation': 'Average of all signal strengths for top/bottom separately',
            'limitations': [
                'HY Credit spread data not available historically',
                'AAII sentiment data not available historically', 
                'Market breadth data not available historically',
                'Velocity calculations simplified (no 5-day history)',
                'Wyckoff resistance/support levels not available'
            ]
        },
        'historical_records': backfilled_data,
        'performance_analysis': performance_analysis
    }
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "data", "signal_strength_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")
    
    # Print key findings
    print("\n📋 KEY FINDINGS")
    print("-" * 40)
    
    overall = performance_analysis['overall_effectiveness']
    print(f"Top Signal Discrimination: {overall['top_signal_discrimination']['discrimination_power']} pts")
    print(f"  • Avg at peaks: {overall['top_signal_discrimination']['avg_at_peaks']}/100")
    print(f"  • Avg at troughs: {overall['top_signal_discrimination']['avg_at_troughs']}/100")
    
    print(f"\nBottom Signal Discrimination: {overall['bottom_signal_discrimination']['discrimination_power']} pts")
    print(f"  • Avg at troughs: {overall['bottom_signal_discrimination']['avg_at_troughs']}/100")
    print(f"  • Avg at peaks: {overall['bottom_signal_discrimination']['avg_at_peaks']}/100")
    
    # Show best performing individual signals
    top_signals = performance_analysis['top_signals']['individual_analysis']
    bottom_signals = performance_analysis['bottom_signals']['individual_analysis']
    
    if top_signals:
        best_top = max(top_signals.items(), key=lambda x: x[1]['effectiveness'])
        print(f"\nBest Top Signal: {best_top[0]}")
        print(f"  • Effectiveness: {best_top[1]['effectiveness']:.3f}")
        print(f"  • Avg at peaks: {best_top[1]['avg_good']}, at troughs: {best_top[1]['avg_bad']}")
    
    if bottom_signals:
        best_bottom = max(bottom_signals.items(), key=lambda x: x[1]['effectiveness'])
        print(f"\nBest Bottom Signal: {best_bottom[0]}")
        print(f"  • Effectiveness: {best_bottom[1]['effectiveness']:.3f}")
        print(f"  • Avg at troughs: {best_bottom[1]['avg_good']}, at peaks: {best_bottom[1]['avg_bad']}")
    
    print(f"\n📄 Full analysis available in: {output_path}")

if __name__ == "__main__":
    main()