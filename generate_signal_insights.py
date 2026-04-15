#!/usr/bin/env python3
"""
Signal Strength Analysis Report Generator

Creates actionable insights from the historical backfill data.
"""

import json
import os

def generate_insights_report():
    """Generate comprehensive insights and recommendations"""
    
    # Load analysis data
    with open('data/signal_strength_analysis.json', 'r') as f:
        data = json.load(f)
    
    performance = data['performance_analysis']
    
    print("=" * 80)
    print("🎯 SIGNAL STRENGTH ANALYSIS - ACTIONABLE INSIGHTS REPORT")
    print("=" * 80)
    print(f"Analysis Date: {data['analysis_date']}")
    print(f"Historical Records: {performance['summary']['total_records']} ({performance['summary']['peaks']} peaks, {performance['summary']['troughs']} troughs)")
    
    # Key Finding: Bottom signals work, top signals don't
    overall = performance['overall_effectiveness']
    
    print(f"\n📊 EXECUTIVE SUMMARY")
    print(f"-" * 40)
    print(f"✅ BOTTOM signals are HIGHLY EFFECTIVE:")
    print(f"   • Discrimination power: +{overall['bottom_signal_discrimination']['discrimination_power']:.1f} points")
    print(f"   • Fire at {overall['bottom_signal_discrimination']['avg_at_troughs']:.1f}/100 at troughs vs {overall['bottom_signal_discrimination']['avg_at_peaks']:.1f}/100 at peaks")
    
    print(f"\n❌ TOP signals are PROBLEMATIC:")
    print(f"   • Discrimination power: {overall['top_signal_discrimination']['discrimination_power']:.1f} points") 
    print(f"   • Fire at {overall['top_signal_discrimination']['avg_at_peaks']:.1f}/100 at peaks vs {overall['top_signal_discrimination']['avg_at_troughs']:.1f}/100 at troughs")
    print(f"   • FALSE POSITIVE PROBLEM: Top signals fire more at troughs than peaks!")
    
    # Signal rankings
    print(f"\n🏆 TIER 1: PERFECT SIGNALS (Effectiveness >= 0.7)")
    print(f"-" * 50)
    
    # Best bottom signals
    bottom_signals = performance['bottom_signals']['individual_analysis']
    top_signals = performance['top_signals']['individual_analysis']
    
    tier1_bottom = {k: v for k, v in bottom_signals.items() if v['effectiveness'] >= 0.7}
    tier1_top = {k: v for k, v in top_signals.items() if v['effectiveness'] >= 0.7}
    
    print("Bottom Signals (Buy opportunities):")
    for signal, stats in sorted(tier1_bottom.items(), key=lambda x: x[1]['effectiveness'], reverse=True):
        print(f"  • {signal}: {stats['effectiveness']:+.3f} ({stats['avg_good']:.0f} at troughs, {stats['avg_bad']:.0f} at peaks)")
    
    print("Top Signals (Sell opportunities):")
    for signal, stats in sorted(tier1_top.items(), key=lambda x: x[1]['effectiveness'], reverse=True):
        print(f"  • {signal}: {stats['effectiveness']:+.3f} ({stats['avg_good']:.0f} at peaks, {stats['avg_bad']:.0f} at troughs)")
    
    # Tier 2 signals
    print(f"\n🥈 TIER 2: GOOD SIGNALS (Effectiveness 0.3-0.7)")
    print(f"-" * 50)
    
    tier2_bottom = {k: v for k, v in bottom_signals.items() if 0.3 <= v['effectiveness'] < 0.7}
    tier2_top = {k: v for k, v in top_signals.items() if 0.3 <= v['effectiveness'] < 0.7}
    
    print("Bottom Signals:")
    for signal, stats in sorted(tier2_bottom.items(), key=lambda x: x[1]['effectiveness'], reverse=True):
        print(f"  • {signal}: {stats['effectiveness']:+.3f} ({stats['count_good']} troughs, {stats['count_bad']} peaks)")
    
    print("Top Signals:")
    for signal, stats in sorted(tier2_top.items(), key=lambda x: x[1]['effectiveness'], reverse=True):
        print(f"  • {signal}: {stats['effectiveness']:+.3f} ({stats['count_good']} peaks, {stats['count_bad']} troughs)")
    
    # Threshold recommendations
    print(f"\n🎯 OPTIMAL THRESHOLD RECOMMENDATIONS")
    print(f"-" * 50)
    
    # Bottom signal thresholds
    trough_thresholds = performance['bottom_signals']['trough_performance']['threshold_analysis']
    print("Bottom Signals at Market Troughs:")
    for threshold in [40, 50, 60, 70]:
        stats = trough_thresholds[str(threshold)]
        hit_rate = stats['hit_rate']
        color = "✅" if hit_rate >= 0.75 else ("⚠️" if hit_rate >= 0.5 else "❌")
        print(f"  {color} >={threshold}: {stats['hits']}/{stats['total']} hits ({hit_rate:.1%})")
    
    # Top signal thresholds  
    peak_thresholds = performance['top_signals']['peak_performance']['threshold_analysis']
    print("\nTop Signals at Market Peaks:")
    for threshold in [40, 50, 60, 70]:
        stats = peak_thresholds[str(threshold)]
        hit_rate = stats['hit_rate']
        color = "✅" if hit_rate >= 0.75 else ("⚠️" if hit_rate >= 0.5 else "❌")
        print(f"  {color} >={threshold}: {stats['hits']}/{stats['total']} hits ({hit_rate:.1%})")
    
    # Actionable recommendations
    print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS")
    print(f"-" * 50)
    
    print("1. PRIORITIZE BOTTOM SIGNALS:")
    print("   • Focus development effort on bottom/buy signals - they work!")
    print("   • Use >=40 threshold for bottom signals (100% historical hit rate)")
    print("   • Regime deep negative is the most reliable (72.0 avg strength)")
    
    print("\n2. FIX TOP SIGNALS:")
    print("   • Current top signals have severe false positive issues")
    print("   • Consider raising threshold to >=60 (37.5% hit rate but fewer false alarms)")
    print("   • Investigate why top signals fire more at troughs than peaks")
    
    print("\n3. WYCKOFF SIGNALS ARE GOLD:")
    print("   • Highest effectiveness when available")
    print("   • Distribution B (80% conf) = Perfect sell signal")  
    print("   • Accumulation B (80% conf) = Perfect buy signal")
    print("   • Prioritize improving Wyckoff detection accuracy")
    
    print("\n4. SIGNAL HIERARCHY FOR LIVE SYSTEM:")
    print("   • Tier 1: Wyckoff Distribution/Accumulation (when available)")
    print("   • Tier 2: Regime-based signals (regime negative for bottoms)")
    print("   • Tier 3: Conviction + Macro signals (supporting evidence)")
    print("   • AVOID: Current entropy and HMM stress signals (poor discrimination)")
    
    print(f"\n📈 MATHEMATICAL FORMULAS EXTRACTED:")
    print(f"-" * 50)
    print("Regime deep negative: min(100, abs(regime_score) * 220) when regime < -0.17")
    print("Macro crushed: min(100, (37 - macro_score) * 6) when macro < 37")
    print("Conviction building: min(100, conviction * 2) when conviction > 24")
    print("Regime elevated: min(100, regime_score * 180) when regime > 0.05")
    print("Low conviction: min(100, (22 - conviction) * 5) when conviction < 22")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"-" * 50)
    print("1. Implement >=40 threshold for bottom signals in live system")
    print("2. Investigate why top signals have false positive problem")
    print("3. Add missing data sources (HY spreads, AAII sentiment, breadth)")
    print("4. Focus engineering effort on improving Wyckoff signal accuracy")
    print("5. Consider machine learning approach to optimize signal combinations")
    
    print(f"\n💾 Data available in: signal_strength_analysis.json")
    print("=" * 80)

if __name__ == "__main__":
    generate_insights_report()