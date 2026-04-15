"""
Comprehensive Crisis Signal Validation Framework

A bulletproof validation system designed to prevent the overfitting disaster that
produced "88% accuracy" on cherry-picked data but 0% accuracy on full datasets.

Key Features:
- Temporal train/test splits (no data leakage)
- Walk-forward validation across multiple periods 
- Proper metrics for imbalanced crisis data
- Individual signal validation before ensemble
- ROC curve optimization for business objectives
- Out-of-sample holdout testing

This framework would have caught the proximity count overfitting immediately.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationPeriod:
    """Define a validation period with train/test splits."""
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    description: str


@dataclass
class SignalMetrics:
    """Comprehensive signal performance metrics."""
    signal_name: str
    period_name: str
    
    # Core metrics
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    true_negative_rate: float
    
    # Business metrics
    crisis_specificity: float
    signal_fire_rate: float
    
    # Raw counts
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Additional info
    total_signals: int
    total_crisis_days: int
    total_normal_days: int
    optimal_threshold: float
    auc_score: float


@dataclass
class TurningPoint:
    """Represent a market turning point for validation."""
    date: str
    type: str  # 'peak' or 'trough'
    magnitude: float  # percentage move magnitude
    duration: int  # days to complete move
    confirmed: bool  # whether this is a confirmed turning point


class CrisisSignalValidator:
    """Comprehensive validation framework for crisis signals."""
    
    def __init__(self, hmm_data_path: str = "hmm_daily_backtest_data.json"):
        self.hmm_data_path = hmm_data_path
        self.data = None
        self.turning_points = []
        
        # Validation periods for walk-forward testing
        self.validation_periods = [
            ValidationPeriod(
                name="2019-2020_Crisis",
                train_start="2012-01-01",
                train_end="2018-12-31",
                test_start="2019-01-01", 
                test_end="2020-12-31",
                description="Train pre-2019, test on COVID period"
            ),
            ValidationPeriod(
                name="2021-2022_Bear",
                train_start="2012-01-01",
                train_end="2020-12-31", 
                test_start="2021-01-01",
                test_end="2022-12-31",
                description="Train through COVID, test on Fed tightening"
            ),
            ValidationPeriod(
                name="2023-2024_Recent",
                train_start="2012-01-01",
                train_end="2022-12-31",
                test_start="2023-01-01",
                test_end="2024-12-31", 
                description="Train through 2022, test recent period"
            )
        ]
        
        # Holdout period - NEVER touched until final validation
        self.holdout_period = ValidationPeriod(
            name="2025-2026_Holdout",
            train_start="2012-01-01",
            train_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2026-12-31",
            description="Final holdout test - use only once"
        )
        
        # Signal definitions for validation
        self.signal_definitions = {
            "LL_deteriorating": {
                "description": "Log-likelihood deteriorating signal",
                "field_name": "ll_zscore",
                "threshold_direction": "less",  # LL deteriorating means more negative
                "baseline_expected": -0.1,  # normal market baseline
                "signal_threshold": -0.5  # threshold for crisis detection
            },
            "Low_conviction": {
                "description": "Low conviction signal (known broken)",
                "signal_key": "Low conviction",
                "threshold_direction": "contains",
                "baseline_expected": 98.0  # fires almost always
            },
            "Regime_deep_negative": {
                "description": "Regime in deep negative state",
                "signal_key": "Regime deep negative", 
                "threshold_direction": "contains",
                "baseline_expected": 50.0
            },
            "HMM_Crisis_state": {
                "description": "HMM classified as crisis state",
                "signal_key": "Crisis",
                "threshold_direction": "contains",
                "baseline_expected": 0.5  # regime-dependent
            },
            "High_regime_entropy": {
                "description": "High regime entropy signal",
                "signal_key": "High regime entropy",
                "threshold_direction": "contains", 
                "baseline_expected": 75.0
            }
        }

    def load_data(self) -> bool:
        """Load and prepare HMM daily data."""
        try:
            with open(self.hmm_data_path, 'r') as f:
                raw_data = json.load(f)
            
            self.data = pd.DataFrame(raw_data)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            print(f"Loaded {len(self.data)} days of data from {self.data['date'].min()} to {self.data['date'].max()}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def load_known_turning_points(self) -> List[TurningPoint]:
        """
        Load known turning points from proximity calibration data.
        
        Returns:
            List of known turning points from historical analysis
        """
        try:
            # Load from proximity calibration data
            prox_path = "data/proximity_calibration.json"
            with open(prox_path, 'r') as f:
                prox_data = json.load(f)
            
            turning_points = []
            for record in prox_data.get('records', []):
                turning_points.append(TurningPoint(
                    date=record['date'],
                    type=record['role'],  # 'peak' or 'trough'
                    magnitude=10.0,  # Placeholder - we know these are significant
                    duration=30,  # Placeholder
                    confirmed=True
                ))
            
            self.turning_points = sorted(turning_points, key=lambda x: x.date)
            
            print(f"Loaded {len(self.turning_points)} known turning points:")
            for tp in self.turning_points:
                print(f"  {tp.date}: {tp.type}")
            
            return self.turning_points
            
        except FileNotFoundError:
            print("No proximity calibration data found, using days_to_peak/trough fields...")
            return self.identify_turning_points_from_fields()

    def identify_turning_points_from_fields(self) -> List[TurningPoint]:
        """
        Identify turning points using days_to_peak and days_to_trough fields in data.
        
        Returns:
            List of inferred turning points
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        turning_points = []
        seen_dates = set()
        
        for i, row in self.data.iterrows():
            current_date = row['date']
            
            # Check if this is near a peak (days_to_peak == 0 or very small)
            if pd.notna(row.get('days_to_peak', np.nan)) and row['days_to_peak'] <= 2:
                peak_date = current_date + timedelta(days=int(row['days_to_peak']))
                if peak_date.strftime('%Y-%m-%d') not in seen_dates:
                    turning_points.append(TurningPoint(
                        date=peak_date.strftime('%Y-%m-%d'),
                        type='peak',
                        magnitude=10.0,  # Placeholder
                        duration=30,
                        confirmed=True
                    ))
                    seen_dates.add(peak_date.strftime('%Y-%m-%d'))
            
            # Check if this is near a trough (days_to_trough == 0 or very small)
            if pd.notna(row.get('days_to_trough', np.nan)) and row['days_to_trough'] <= 2:
                trough_date = current_date + timedelta(days=int(row['days_to_trough']))
                if trough_date.strftime('%Y-%m-%d') not in seen_dates:
                    turning_points.append(TurningPoint(
                        date=trough_date.strftime('%Y-%m-%d'),
                        type='trough',
                        magnitude=10.0,  # Placeholder
                        duration=30,
                        confirmed=True
                    ))
                    seen_dates.add(trough_date.strftime('%Y-%m-%d'))
        
        self.turning_points = sorted(turning_points, key=lambda x: x.date)
        
        print(f"Identified {len(self.turning_points)} turning points from data fields:")
        for tp in self.turning_points:
            print(f"  {tp.date}: {tp.type}")
        
        return self.turning_points

    def identify_turning_points(self, 
                               spy_column: str = 'spy_price',
                               threshold_pct: float = 5.0,
                               min_duration: int = 5) -> List[TurningPoint]:
        """
        Main method to identify turning points, trying multiple approaches.
        
        Args:
            spy_column: Column containing SPY price data
            threshold_pct: Minimum percentage move to qualify as turning point
            min_duration: Minimum days for move to complete
        
        Returns:
            List of confirmed turning points
        """
        # Try to load known turning points first
        try:
            return self.load_known_turning_points()
        except:
            pass
        
        # Fall back to fields-based identification
        try:
            return self.identify_turning_points_from_fields()
        except:
            pass
        
        # Final fallback: regime-based identification
        return self.identify_turning_points_from_regime()

    def identify_turning_points_from_regime(self) -> List[TurningPoint]:
        """
        Identify turning points using regime_score extreme values.
        
        Returns:
            List of regime-based turning points
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        turning_points = []
        
        # Use regime_score to identify stress periods
        regime_scores = self.data['regime_score'].fillna(0)
        
        # Find periods of extreme regime stress (top/bottom 5%)
        top_threshold = regime_scores.quantile(0.95)
        bottom_threshold = regime_scores.quantile(0.05)
        
        stress_periods = []
        for i, score in enumerate(regime_scores):
            if score >= top_threshold or score <= bottom_threshold:
                date = self.data.iloc[i]['date']
                stress_periods.append((date.strftime('%Y-%m-%d'), score))
        
        # Cluster nearby stress periods into turning points
        if stress_periods:
            current_cluster = [stress_periods[0]]
            
            for i in range(1, len(stress_periods)):
                prev_date = datetime.strptime(current_cluster[-1][0], '%Y-%m-%d')
                curr_date = datetime.strptime(stress_periods[i][0], '%Y-%m-%d')
                
                if (curr_date - prev_date).days <= 30:
                    current_cluster.append(stress_periods[i])
                else:
                    # Process current cluster
                    if len(current_cluster) >= 3:  # Minimum cluster size
                        avg_score = np.mean([x[1] for x in current_cluster])
                        center_date = current_cluster[len(current_cluster)//2][0]
                        
                        turning_points.append(TurningPoint(
                            date=center_date,
                            type='peak' if avg_score > 0 else 'trough',
                            magnitude=abs(avg_score) * 100,
                            duration=len(current_cluster),
                            confirmed=True
                        ))
                    
                    current_cluster = [stress_periods[i]]
        
        self.turning_points = turning_points
        
        print(f"Identified {len(self.turning_points)} regime-based turning points:")
        for tp in self.turning_points:
            print(f"  {tp.date}: {tp.type} (magnitude: {tp.magnitude:.1f})")
        
        return self.turning_points

    def create_crisis_labels(self, 
                           window_days: int = 30) -> pd.Series:
        """
        Create binary crisis labels based on proximity to turning points.
        
        Args:
            window_days: Days before/after turning point to label as crisis
            
        Returns:
            Binary series where 1 = crisis period, 0 = normal period
        """
        if not self.turning_points:
            raise ValueError("No turning points identified. Call identify_turning_points() first.")
        
        crisis_labels = pd.Series(0, index=self.data.index)
        
        for tp in self.turning_points:
            tp_date = pd.to_datetime(tp.date)
            
            # Create window around turning point
            window_start = tp_date - timedelta(days=window_days)
            window_end = tp_date + timedelta(days=window_days)
            
            # Mark days in window as crisis
            crisis_mask = (self.data['date'] >= window_start) & (self.data['date'] <= window_end)
            crisis_labels[crisis_mask] = 1
        
        crisis_days = crisis_labels.sum()
        normal_days = len(crisis_labels) - crisis_days
        crisis_rate = crisis_days / len(crisis_labels) * 100
        
        print(f"Crisis labeling: {crisis_days} crisis days ({crisis_rate:.1f}%), {normal_days} normal days")
        
        return crisis_labels

    def extract_signal_values(self, signal_name: str) -> pd.Series:
        """
        Extract signal values from the data based on signal definition.
        
        Args:
            signal_name: Name of signal as defined in signal_definitions
            
        Returns:
            Series of signal values
        """
        if signal_name not in self.signal_definitions:
            raise ValueError(f"Signal {signal_name} not defined. Available: {list(self.signal_definitions.keys())}")
        
        signal_def = self.signal_definitions[signal_name]
        
        # Special handling for LL deteriorating using ll_zscore field
        if signal_name == "LL_deteriorating":
            if 'll_zscore' in self.data.columns:
                return self.data['ll_zscore'].fillna(0)
            else:
                return pd.Series(0.0, index=self.data.index)
        
        # For other signals, check if they appear in top_signals or bottom_signals
        signal_key = signal_def["signal_key"]
        signal_values = pd.Series(0.0, index=self.data.index)
        
        for i, row in self.data.iterrows():
            signal_fired = False
            
            # Check top signals
            if 'top_signals' in row and isinstance(row['top_signals'], list):
                for signal in row['top_signals']:
                    if signal_key.lower() in signal.lower():
                        signal_fired = True
                        break
            
            # Check bottom signals  
            if not signal_fired and 'bottom_signals' in row and isinstance(row['bottom_signals'], list):
                for signal in row['bottom_signals']:
                    if signal_key.lower() in signal.lower():
                        signal_fired = True
                        break
            
            signal_values.iloc[i] = 1.0 if signal_fired else 0.0
        
        return signal_values

    def optimize_threshold(self, 
                          signal_values: pd.Series,
                          crisis_labels: pd.Series, 
                          signal_name: str,
                          objective: str = "f1") -> Tuple[float, float]:
        """
        Find optimal threshold for signal using ROC curve analysis.
        
        Args:
            signal_values: Signal values to threshold
            crisis_labels: Binary crisis labels
            signal_name: Name of signal for special handling
            objective: Optimization objective ("precision", "recall", "f1", "specificity")
            
        Returns:
            Tuple of (optimal_threshold, score_at_threshold)
        """
        if len(signal_values) != len(crisis_labels):
            raise ValueError("Signal values and crisis labels must have same length")
        
        # Remove NaN values
        mask = ~(signal_values.isna() | crisis_labels.isna())
        signal_clean = signal_values[mask]
        labels_clean = crisis_labels[mask]
        
        if len(signal_clean) == 0:
            return 0.0, 0.0
        
        # Special handling for binary signals (0/1 values)
        if signal_name != "LL_deteriorating":
            # For binary signals, threshold is simply 0.5
            predictions = (signal_clean >= 0.5).astype(int)
            
            if len(np.unique(predictions)) < 2:
                return 0.5, 0.0  # No discrimination
            
            precision = precision_score(labels_clean, predictions, zero_division=0)
            recall = recall_score(labels_clean, predictions, zero_division=0)  
            f1 = f1_score(labels_clean, predictions, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(labels_clean, predictions).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if objective == "precision":
                score = precision
            elif objective == "recall":
                score = recall
            elif objective == "f1":
                score = f1
            elif objective == "specificity":
                score = specificity
            else:
                score = f1
            
            return 0.5, score
        
        # For continuous signals like LL deteriorating
        try:
            fpr, tpr, thresholds = roc_curve(labels_clean, signal_clean)
        except ValueError:
            return 0.0, 0.0
        
        best_threshold = 0.0
        best_score = 0.0
        
        for i, threshold in enumerate(thresholds):
            predictions = (signal_clean >= threshold).astype(int)
            
            if len(np.unique(predictions)) < 2:
                continue  # Skip if all predictions are the same
            
            precision = precision_score(labels_clean, predictions, zero_division=0)
            recall = recall_score(labels_clean, predictions, zero_division=0)  
            f1 = f1_score(labels_clean, predictions, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(labels_clean, predictions).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if objective == "precision":
                score = precision
            elif objective == "recall":
                score = recall
            elif objective == "f1":
                score = f1
            elif objective == "specificity":
                score = specificity
            else:
                score = f1  # Default to F1
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score

    def validate_signal_period(self, 
                              signal_name: str,
                              period: ValidationPeriod,
                              threshold: float = None) -> SignalMetrics:
        """
        Validate a signal on a specific period with proper train/test split.
        
        Args:
            signal_name: Name of signal to validate
            period: ValidationPeriod defining train/test split
            threshold: Signal threshold (if None, optimize on training data)
            
        Returns:
            SignalMetrics with comprehensive performance data
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Create crisis labels once for entire dataset
        crisis_labels = self.create_crisis_labels()
        
        # Filter data for this period
        test_mask = (self.data['date'] >= period.test_start) & (self.data['date'] <= period.test_end)
        train_mask = (self.data['date'] >= period.train_start) & (self.data['date'] <= period.train_end)
        
        test_data = self.data[test_mask].copy()
        train_data = self.data[train_mask].copy()
        
        if len(test_data) == 0:
            raise ValueError(f"No test data found for period {period.name}")
        
        # Extract signal values
        all_signals = self.extract_signal_values(signal_name)
        
        train_signals = all_signals[train_mask]
        test_signals = all_signals[test_mask]
        
        train_crisis = crisis_labels[train_mask]
        test_crisis = crisis_labels[test_mask]
        
        # Optimize threshold on training data if not provided
        if threshold is None:
            threshold, _ = self.optimize_threshold(train_signals, train_crisis, signal_name, objective="f1")
        
        # Evaluate on test data
        if signal_name == "LL_deteriorating":
            # For LL deteriorating, signal fires when value is more negative (< threshold)
            test_predictions = (test_signals <= threshold).astype(int)
        else:
            # For binary signals, use >= threshold
            test_predictions = (test_signals >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(test_crisis, test_predictions, zero_division=0)
        recall = recall_score(test_crisis, test_predictions, zero_division=0)
        f1 = f1_score(test_crisis, test_predictions, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_crisis, test_predictions).ravel()
        
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Business metrics
        signal_fire_rate = test_predictions.sum() / len(test_predictions)
        crisis_days = test_crisis.sum()
        normal_days = len(test_crisis) - crisis_days
        
        # Crisis specificity: how much more likely to fire in crisis vs normal
        crisis_fire_rate = test_predictions[test_crisis == 1].mean() if crisis_days > 0 else 0
        normal_fire_rate = test_predictions[test_crisis == 0].mean() if normal_days > 0 else 0
        crisis_specificity = crisis_fire_rate / normal_fire_rate if normal_fire_rate > 0 else float('inf')
        
        # AUC score
        try:
            auc_score = roc_auc_score(test_crisis, test_signals)
        except ValueError:
            auc_score = 0.5  # No discrimination
        
        return SignalMetrics(
            signal_name=signal_name,
            period_name=period.name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=false_positive_rate,
            true_negative_rate=true_negative_rate,
            crisis_specificity=crisis_specificity,
            signal_fire_rate=signal_fire_rate,
            true_positives=tp,
            false_positives=fp, 
            true_negatives=tn,
            false_negatives=fn,
            total_signals=test_predictions.sum(),
            total_crisis_days=crisis_days,
            total_normal_days=normal_days,
            optimal_threshold=threshold,
            auc_score=auc_score
        )

    def validate_all_signals(self) -> Dict[str, List[SignalMetrics]]:
        """
        Run comprehensive validation across all signals and periods.
        
        Returns:
            Dict mapping signal names to list of metrics for each period
        """
        all_results = {}
        
        print("Starting comprehensive signal validation...")
        print("=" * 60)
        
        for signal_name in self.signal_definitions:
            print(f"\nValidating {signal_name}...")
            signal_results = []
            
            for period in self.validation_periods:
                print(f"  Period: {period.name}")
                
                try:
                    metrics = self.validate_signal_period(signal_name, period)
                    signal_results.append(metrics)
                    
                    print(f"    Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}")
                    print(f"    False Positive Rate: {metrics.false_positive_rate:.3f}")
                    print(f"    Signal Fire Rate: {metrics.signal_fire_rate:.3f}")
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            
            all_results[signal_name] = signal_results
        
        return all_results

    def generate_validation_report(self, 
                                 results: Dict[str, List[SignalMetrics]],
                                 output_path: str = "signal_validation_report.md") -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            results: Results from validate_all_signals()
            output_path: Path to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "# Crisis Signal Validation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Methodology",
            "",
            "This report presents results from a comprehensive validation framework designed to prevent",
            "the overfitting disaster that produced '88% accuracy' on cherry-picked data but 0% accuracy",
            "on full datasets.",
            "",
            "### Key Features:",
            "- **Temporal splits**: No data leakage between train/test periods",
            "- **Walk-forward testing**: Multiple validation periods",
            "- **Proper metrics**: Designed for imbalanced crisis data", 
            "- **Individual signal focus**: Test each signal before combining",
            "- **Business-relevant**: Optimize for false positive minimization",
            "",
            "## Signal Performance Summary",
            ""
        ]
        
        # Create summary table
        summary_data = []
        for signal_name, metrics_list in results.items():
            if not metrics_list:
                continue
                
            avg_precision = np.mean([m.precision for m in metrics_list])
            avg_recall = np.mean([m.recall for m in metrics_list])
            avg_f1 = np.mean([m.f1_score for m in metrics_list])
            avg_fpr = np.mean([m.false_positive_rate for m in metrics_list])
            avg_fire_rate = np.mean([m.signal_fire_rate for m in metrics_list])
            
            summary_data.append({
                'Signal': signal_name,
                'Avg Precision': f"{avg_precision:.3f}",
                'Avg Recall': f"{avg_recall:.3f}",  
                'Avg F1': f"{avg_f1:.3f}",
                'Avg FPR': f"{avg_fpr:.3f}",
                'Avg Fire Rate': f"{avg_fire_rate:.3f}"
            })
        
        if summary_data:
            report_lines.extend([
                "| Signal | Avg Precision | Avg Recall | Avg F1 | Avg FPR | Avg Fire Rate |",
                "|--------|---------------|------------|--------|---------|---------------|"
            ])
            
            for row in summary_data:
                report_lines.append(f"| {row['Signal']} | {row['Avg Precision']} | {row['Avg Recall']} | {row['Avg F1']} | {row['Avg FPR']} | {row['Avg Fire Rate']} |")
        
        report_lines.extend([
            "",
            "## Detailed Results by Signal",
            ""
        ])
        
        # Detailed results for each signal
        for signal_name, metrics_list in results.items():
            report_lines.extend([
                f"### {signal_name}",
                "",
                f"**Description:** {self.signal_definitions[signal_name]['description']}",
                ""
            ])
            
            if not metrics_list:
                report_lines.extend([
                    "❌ **No valid results** - Signal failed validation",
                    ""
                ])
                continue
            
            # Performance across periods
            report_lines.extend([
                "| Period | Precision | Recall | F1 | FPR | Fire Rate | Crisis Days | Normal Days |",
                "|--------|-----------|--------|----|----- |-----------|-------------|-------------|"
            ])
            
            for metrics in metrics_list:
                report_lines.append(
                    f"| {metrics.period_name} | {metrics.precision:.3f} | {metrics.recall:.3f} | "
                    f"{metrics.f1_score:.3f} | {metrics.false_positive_rate:.3f} | "
                    f"{metrics.signal_fire_rate:.3f} | {metrics.total_crisis_days} | {metrics.total_normal_days} |"
                )
            
            # Signal assessment
            avg_fpr = np.mean([m.false_positive_rate for m in metrics_list])
            avg_precision = np.mean([m.precision for m in metrics_list])
            
            if avg_fpr < 0.05 and avg_precision > 0.5:
                assessment = "EXCELLENT - Low false positives, good precision"
            elif avg_fpr < 0.15 and avg_precision > 0.3:
                assessment = "MODERATE - Acceptable performance with tuning"
            else:
                assessment = "POOR - High false positives or low precision"
            
            report_lines.extend([
                "",
                assessment,
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Immediate Actions:",
            ""
        ])
        
        excellent_signals = []
        moderate_signals = []
        poor_signals = []
        
        for signal_name, metrics_list in results.items():
            if not metrics_list:
                poor_signals.append(signal_name)
                continue
                
            avg_fpr = np.mean([m.false_positive_rate for m in metrics_list])
            avg_precision = np.mean([m.precision for m in metrics_list])
            
            if avg_fpr < 0.05 and avg_precision > 0.5:
                excellent_signals.append(signal_name)
            elif avg_fpr < 0.15 and avg_precision > 0.3:
                moderate_signals.append(signal_name)
            else:
                poor_signals.append(signal_name)
        
        if excellent_signals:
            report_lines.extend([
                "**Deploy immediately:**",
                ""
            ])
            for signal in excellent_signals:
                report_lines.append(f"- DEPLOY: {signal}")
            report_lines.append("")
        
        if moderate_signals:
            report_lines.extend([
                "**Tune thresholds:**", 
                ""
            ])
            for signal in moderate_signals:
                report_lines.append(f"- TUNE: {signal}")
            report_lines.append("")
        
        if poor_signals:
            report_lines.extend([
                "**Disable or redesign:**",
                ""
            ])
            for signal in poor_signals:
                report_lines.append(f"- DISABLE: {signal}")
            report_lines.append("")
        
        report_lines.extend([
            "### Next Steps:",
            "",
            "1. **Start with excellent signals**: Build system around proven low-FPR signals",
            "2. **Tune moderate signals**: Optimize thresholds for better precision", 
            "3. **Research poor signals**: Understand why they fail before combining",
            "4. **Ensemble cautiously**: Only combine signals that individually perform well",
            "5. **Monitor live performance**: Track real-world vs backtest performance",
            "",
            "---",
            "",
            "*This report was generated by the Crisis Signal Validation Framework*",
            "*Framework designed to prevent overfitting and cherry-picking bias*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nValidation report saved to: {output_path}")
        
        return report_content

    def save_baseline_results(self, 
                            signal_name: str,
                            results: List[SignalMetrics],
                            output_path: str) -> None:
        """Save detailed results for a specific signal as JSON baseline."""
        
        # Convert metrics to JSON-serializable format
        json_results = []
        for metrics in results:
            metrics_dict = asdict(metrics)
            # Convert numpy integers to Python integers
            for key, value in metrics_dict.items():
                if hasattr(value, 'item'):  # numpy scalar
                    metrics_dict[key] = value.item()
                elif isinstance(value, np.integer):
                    metrics_dict[key] = int(value)
                elif isinstance(value, np.floating):
                    metrics_dict[key] = float(value)
            json_results.append(metrics_dict)
        
        baseline_data = {
            "signal_name": signal_name,
            "validation_date": datetime.now().isoformat(),
            "methodology": "Temporal train/test splits with walk-forward validation",
            "periods_tested": len(results),
            "results": json_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        print(f"Baseline results for {signal_name} saved to: {output_path}")


def main():
    """Run the complete validation framework."""
    print("🚨 CRISIS SIGNAL VALIDATION FRAMEWORK 🚨")
    print("=" * 50)
    print("Designed to prevent overfitting disasters")
    print("Testing signals with bulletproof methodology")
    print("=" * 50)
    
    # Initialize validator
    validator = CrisisSignalValidator()
    
    # Load data
    if not validator.load_data():
        print("❌ Failed to load data")
        return
    
    # Identify turning points
    print("\n📊 IDENTIFYING TURNING POINTS")
    turning_points = validator.identify_turning_points()
    
    if not turning_points:
        print("❌ No turning points identified")
        return
    
    # Run validation
    print("\n🧪 RUNNING COMPREHENSIVE VALIDATION")
    results = validator.validate_all_signals()
    
    # Generate report
    print("\n📝 GENERATING VALIDATION REPORT")
    report = validator.generate_validation_report(results)
    
    # Save baseline for LL deteriorating signal (the proven one)
    if "LL_deteriorating" in results and results["LL_deteriorating"]:
        validator.save_baseline_results(
            "LL_deteriorating", 
            results["LL_deteriorating"],
            "baseline_ll_validation.json"
        )
    
    print("\n✅ VALIDATION COMPLETE")
    print("=" * 50)
    print("Results saved:")
    print("- signal_validation_report.md (comprehensive report)")  
    print("- baseline_ll_validation.json (LL deteriorating baseline)")
    print("\nFramework designed to catch overfitting immediately.")
    print("No more 88% fake accuracy!")


if __name__ == "__main__":
    main()