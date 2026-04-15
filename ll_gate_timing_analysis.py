#!/usr/bin/env python3
"""
LL Gate Opening Lead Time Analysis - FIND MOST RECENT LL FIRE
Looking for the most recent LL deterioration before each crash, not the earliest.
"""
import json
from datetime import datetime

# Major crash bottoms for reference
CRASH_BOTTOMS = {
    "2020-03-23": "COVID Bottom (-34%)",
    "2018-12-24": "Dec 2018 Bottom (-20%)",
    "2022-10-12": "2022 Bear Bottom (-25%)",
}

# Use actual LL threshold from validation data
LL_THRESHOLD = -0.35

with open('hmm_daily_backtest_data.json') as f:
    data = json.load(f)

# Sort by date
data.sort(key=lambda x: x.get('date', ''))

print("LL GATE OPENING LEAD TIME ANALYSIS")
print("=" * 50)
print(f"Using LL threshold: {LL_THRESHOLD}")
print("Looking for MOST RECENT LL fire before each crash")
print()

lead_times = []
details = []

for crash_date, crash_desc in CRASH_BOTTOMS.items():
    try:
        crash_dt = datetime.strptime(crash_date, "%Y-%m-%d")
        
        # Look backwards from crash date to find MOST RECENT LL fire
        ll_most_recent = None
        
        # Go through data backwards (most recent first)
        for day in reversed(data):
            day_date_str = day.get('date', '')
            if not day_date_str:
                continue
                
            try:
                day_dt = datetime.strptime(day_date_str, "%Y-%m-%d")
            except:
                continue
            
            # Only look at days before the crash
            if day_dt >= crash_dt:
                continue
                
            ll_z = day.get('ll_zscore', 0)
            
            # If LL fired, this is the most recent one before crash
            if ll_z < LL_THRESHOLD:
                ll_most_recent = (day_dt, ll_z)
                break  # Found most recent, stop looking
        
        if ll_most_recent:
            fire_date, fire_ll = ll_most_recent
            days_before = (crash_dt - fire_date).days
            lead_times.append(days_before)
            weeks_before = days_before / 7
            
            print(f"{crash_date} | {days_before:3d} days   | {fire_date.strftime('%Y-%m-%d')} (LL:{fire_ll:.3f}) | {crash_desc}")
            details.append((crash_date, days_before, weeks_before, crash_desc, fire_ll))
        else:
            print(f"{crash_date} | No signal  | None detected              | {crash_desc}")
            
    except Exception as e:
        print(f"{crash_date} | Error      | {str(e)[:10]}               | {crash_desc}")

if lead_times:
    avg_days = sum(lead_times) / len(lead_times)
    min_days = min(lead_times) 
    max_days = max(lead_times)
    
    print()
    print("📊 LL GATE TIMING STATISTICS:")
    print(f"Average lead time: {avg_days:.1f} days ({avg_days/7:.1f} weeks)")
    print(f"Minimum lead time: {min_days} days ({min_days/7:.1f} weeks)")
    print(f"Maximum lead time: {max_days} days ({max_days/7:.1f} weeks)")
    print(f"Sample size: {len(lead_times)} crashes analyzed")
    
    print()
    print("🎯 PRACTICAL WARNING WINDOW:")
    if avg_days < 30:
        print(f"• LL gate opens ~{avg_days:.0f} days ({avg_days/7:.1f} weeks) before crash")
        print(f"• Short-term warning - time to hedge/reduce risk immediately")
    elif avg_days < 90:
        print(f"• LL gate opens ~{avg_days:.0f} days ({avg_days/7:.1f} weeks) before crash")  
        print(f"• Medium-term warning - time to adjust portfolio positioning")
    else:
        print(f"• LL gate opens ~{avg_days:.0f} days ({avg_days/7:.1f} weeks) before crash")
        print(f"• Long-term warning - structural regime change signal")
    
    print()
    print("📈 CRASH-BY-CRASH BREAKDOWN:")
    for crash_date, days, weeks, desc, ll_val in details:
        timing_desc = "immediate" if days < 7 else f"{weeks:.1f} weeks"
        print(f"• {desc}: {timing_desc} warning (LL hit {ll_val:.3f})")
        
else:
    print()
    print("❌ No valid lead times found")
    
    # Check what LL values exist around crashes
    print()
    print("DEBUGGING - LL around crashes:")
    for crash_date in list(CRASH_BOTTOMS.keys())[:1]:  # Just one crash
        crash_dt = datetime.strptime(crash_date, "%Y-%m-%d")
        print(f"\nAround {crash_date}:")
        
        crash_window = []
        for day in data:
            if not day.get('date'):
                continue
            day_dt = datetime.strptime(day.get('date'), "%Y-%m-%d")
            days_diff = (crash_dt - day_dt).days  # Positive = before crash
            
            if -5 <= days_diff <= 90:  # 90 days before to 5 days after
                ll_z = day.get('ll_zscore', 0)
                crash_window.append((days_diff, day.get('date'), ll_z))
        
        # Sort by days before crash
        crash_window.sort(key=lambda x: x[0], reverse=True)
        
        print(f"LL values 90 days before crash:")
        for days_diff, date_str, ll_z in crash_window[:10]:
            if days_diff >= 0:
                indicator = "🔥" if ll_z < LL_THRESHOLD else "  "
                print(f"  T-{days_diff:2d}: {date_str} LL={ll_z:.3f} {indicator}")

print()
print("💡 INTERPRETATION:")
print("LL deteriorating = HMM model breakdown detected")
print("Lead time shows how early the model senses trouble")