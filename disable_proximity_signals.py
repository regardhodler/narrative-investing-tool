#!/usr/bin/env python3
"""
PHASE 1 IMPLEMENTATION: Disable Current Proximity Signals

This script implements the immediate cleanup actions to disable the broken
proximity signals without breaking the QIR system.

Actions:
1. Add feature flag to disable proximity signals
2. Return neutral values when disabled
3. Update pattern classification to handle absence
4. Add maintenance warning to dashboard

Usage:
    python disable_proximity_signals.py
"""

import os
import shutil
from datetime import datetime


def backup_current_implementation():
    """
    Archive the current broken proximity signal implementation
    """
    print("=== BACKING UP CURRENT IMPLEMENTATION ===")
    
    # Create archive directory
    archive_dir = "archive/broken_proximity"
    os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Files to archive
    files_to_archive = [
        "modules/quick_run.py",  # Contains proximity signal logic
        "backfill_signal_analysis.py",  # Historical backfill
        "hmm_full_backtest.py",  # Daily backtest
        "tools/calibrate_proximity.py",  # Calibration tool
        "data/top_bottom_history.json",  # Signal history
        "data/proximity_calibration.json",  # Calibration data
    ]
    
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            # Create backup with timestamp
            backup_name = f"{os.path.basename(file_path)}_{timestamp}_BROKEN"
            backup_path = os.path.join(archive_dir, backup_name)
            shutil.copy2(file_path, backup_path)
            print(f"✅ Archived: {file_path} → {backup_path}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Create failure analysis document
    failure_analysis_path = os.path.join(archive_dir, f"FAILURE_ANALYSIS_{timestamp}.md")
    
    failure_analysis = f"""# Proximity Signal Failure Analysis
    
## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary of Failure

The Top/Bottom Proximity signals showed **0% real-world accuracy** when properly validated, despite claims of 88% accuracy.

### Key Problems Identified

1. **Catastrophic Threshold Miscalibration**
   - Low Conviction: fires 99.3% of days (should be 1-5%)
   - High Entropy: fires 77.1% of days (should be 5-10%) 
   - Deep Regime: fires 50.7% of days (should be 3-8%)

2. **Cherry-Picked Training Data**
   - Optimized on 16 extreme dates (0.7% of 2,355 total days)
   - Ignored 99.3% of normal market behavior
   - No out-of-sample validation

3. **Count Method Fundamentally Flawed**
   - Focused on recall without considering false positive costs
   - No economic foundation for threshold selection
   - Signals fire constantly in normal market conditions

### Validation Results

All 5 tested signals **FAILED** validation framework:

| Signal | Avg Precision | Avg FPR | Assessment |
|--------|---------------|---------|------------|
| LL_deteriorating | 23.2% | 2.6% | **POOR** |
| Low_conviction | 13.7% | **98.7%** | **BROKEN** |
| Regime_deep_negative | 10.8% | 35.7% | **POOR** |
| HMM_Crisis_state | 7.1% | 11.0% | **POOR** |
| High_regime_entropy | 19.6% | 60.2% | **POOR** |

### Root Cause

**Overfitting without validation** - signals were optimized on extreme market events without considering base rates or false positive costs.

### Action Taken

Proximity signals disabled {timestamp} and archived to prevent future use.
Rebuilding with validation-first approach documented in signal_rebuild_roadmap.md.

### Files Archived

{chr(10).join([f"- {f}" for f in files_to_archive if os.path.exists(f)])}

## Next Steps

1. Follow signal_rebuild_roadmap.md for systematic rebuilding
2. Use validation_framework.py for all new signal development
3. Never deploy signals without proper out-of-sample testing
"""

    with open(failure_analysis_path, 'w') as f:
        f.write(failure_analysis)
    
    print(f"✅ Created failure analysis: {failure_analysis_path}")
    print()


def create_feature_flag_patch():
    """
    Create patch to add DISABLE_PROXIMITY_SIGNALS feature flag
    """
    print("=== CREATING FEATURE FLAG PATCH ===")
    
    patch_content = '''# PROXIMITY SIGNAL DISABLE PATCH
# Add this to the top of modules/quick_run.py

# Feature flag to disable broken proximity signals
DISABLE_PROXIMITY_SIGNALS = True

# Add this function to replace proximity signal calculation:

def get_disabled_proximity_signals():
    """
    Return neutral proximity signal values when disabled
    """
    return {
        'top_count': 0,
        'bottom_count': 0,
        'top_score': 0,
        'bottom_score': 0,
        'net_lean': 0,
        'proximity_status': 'DISABLED',
        'proximity_message': '⚠️ Proximity signals under maintenance - rebuilding with validation framework',
        'last_updated': datetime.now().isoformat()
    }

# Replace proximity signal calculation section (around lines 2244-2410) with:

def calculate_proximity_signals(market_data, hmm_state, regime_score, macro_score):
    """
    Calculate proximity signals with disable flag support
    """
    if DISABLE_PROXIMITY_SIGNALS:
        return get_disabled_proximity_signals()
    
    # Original proximity calculation code would go here
    # (but don't use it - it's broken!)
    
    # For now, return disabled values
    return get_disabled_proximity_signals()

# Update pattern classification to handle disabled proximity signals:

def classify_pattern_with_disabled_proximity(macro_score, tactical_score, options_score, proximity_data):
    """
    Pattern classification that works even when proximity signals are disabled
    """
    if proximity_data.get('proximity_status') == 'DISABLED':
        # Rely on other signal layers when proximity disabled
        if macro_score > 70 and tactical_score > 70:
            return "BULLISH_CONFIRMATION"
        elif macro_score < 30 and tactical_score < 30:
            return "BEARISH_CONFIRMATION"
        else:
            return "MIXED_NO_PROXIMITY"
    
    # Original pattern classification logic
    # (update to use this function instead of relying on proximity counts)
'''
    
    patch_file = "proximity_disable_patch.py"
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    print(f"✅ Created patch file: {patch_file}")
    print("📝 Manual step required: Apply this patch to modules/quick_run.py")
    print()


def create_dashboard_warning():
    """
    Create warning banner for dashboard
    """
    print("=== CREATING DASHBOARD WARNING ===")
    
    warning_content = '''# Dashboard Warning for Proximity Signals

Add this warning banner to the Streamlit dashboard when proximity signals are disabled.

```python
# Add to app.py or modules/quick_run.py dashboard rendering

if DISABLE_PROXIMITY_SIGNALS:
    st.warning("""
    🚧 **Proximity Signals Under Maintenance** 🚧
    
    Top/Bottom Proximity signals are temporarily disabled while being rebuilt 
    with proper validation framework. 
    
    **Reason**: Current signals showed 0% accuracy when properly tested.
    
    **Timeline**: Rebuilt signals expected in 4-6 weeks.
    
    **Impact**: QIR analysis continues using Macro + Tactical + Options layers.
    """)
```

# Alternative compact warning:
st.error("⚠️ Proximity signals disabled - rebuilding with validation framework")
'''
    
    warning_file = "dashboard_warning.md"
    with open(warning_file, 'w') as f:
        f.write(warning_content)
    
    print(f"✅ Created warning template: {warning_file}")
    print()


def verify_qir_compatibility():
    """
    Verify that QIR system can handle disabled proximity signals
    """
    print("=== QIR COMPATIBILITY CHECK ===")
    
    compatibility_notes = """
QIR System Components That Need Updates:

1. **Pattern Classification** (modules/quick_run.py)
   - Update to handle proximity_status = 'DISABLED'
   - Fallback to 2-layer analysis (Macro + Tactical + Options)
   - Add new pattern: 'MIXED_NO_PROXIMITY'

2. **Conviction Calculation**
   - Remove proximity signal weight from conviction score
   - Rebalance weights across remaining signal layers
   - Document temporary weight adjustments

3. **Dashboard Display**
   - Show warning banner when proximity disabled
   - Hide proximity signal charts/metrics
   - Update legend to indicate 2-layer vs 3-layer analysis

4. **History Tracking** (services/qir_history.py)
   - Log proximity_status in QIR run history
   - Track performance during disabled period
   - Maintain continuity for other signal layers

5. **Signal Validation Badges**
   - Prepare framework for validation metrics display
   - Design space for precision/FPR badges
   - Plan integration with new validated signals

Testing Checklist:
- [ ] QIR runs without errors when proximity disabled
- [ ] Pattern classification produces valid results  
- [ ] Dashboard displays correctly with warning
- [ ] History tracking continues to work
- [ ] Other signal layers (Macro/Tactical/Options) unaffected
"""
    
    print(compatibility_notes)
    
    with open("qir_compatibility_checklist.md", 'w') as f:
        f.write(compatibility_notes)
    
    print("✅ Created compatibility checklist: qir_compatibility_checklist.md")
    print()


def main():
    """
    Execute Phase 1 implementation steps
    """
    print("🚧 PROXIMITY SIGNAL REBUILD - PHASE 1 IMPLEMENTATION 🚧")
    print("=" * 70)
    print()
    
    # Step 1: Backup current implementation
    backup_current_implementation()
    
    # Step 2: Create feature flag patch
    create_feature_flag_patch()
    
    # Step 3: Create dashboard warning
    create_dashboard_warning()
    
    # Step 4: Verify QIR compatibility
    verify_qir_compatibility()
    
    print("=" * 70)
    print("✅ PHASE 1 PREPARATION COMPLETE")
    print()
    print("📋 NEXT MANUAL STEPS:")
    print("1. Apply proximity_disable_patch.py to modules/quick_run.py")
    print("2. Add dashboard warning to app.py")
    print("3. Test QIR system with disabled proximity signals")
    print("4. Begin Phase 2: Signal Discovery (see signal_rebuild_roadmap.md)")
    print()
    print("🎯 GOAL: Zero broken signals in production, clean foundation for rebuild")


if __name__ == "__main__":
    main()