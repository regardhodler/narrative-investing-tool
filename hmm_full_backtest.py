#!/usr/bin/env python3
"""
Comprehensive HMM Full Backtest: Top/Bottom Proximity Count Method

Tests the "88% accuracy" claim against the full HMM training dataset (2012-2026)
instead of just 16 cherry-picked crash dates. Reconstructs daily market data,
applies signal calculation formulas daily, and measures performance across 
~3,660 trading days.

OBJECTIVE: Answer "What is the TRUE accuracy when tested on 12 years of daily 
data instead of 16 handpicked crash dates?"
"""

import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@dataclass
class DailyBacktestRow:
    """Container for single day's backtest data"""
    date: str
    regime_score: float
    macro_score: float
    conviction: float
    entropy: float
    ll_zscore: float
    hmm_state: str
    hmm_state_idx: int
    velocity: float
    spy_price: float
    vix: float
    
    # Calculated signals
    top_signals: List[str]
    bottom_signals: List[str]
    top_count: int
    bottom_count: int
    top_score: float
    bottom_score: float
    net_lean: int
    
    # Performance labels  
    days_to_peak: Optional[int] = None
    days_to_trough: Optional[int] = None
    next_5pct_decline_days: Optional[int] = None
    next_5pct_rally_days: Optional[int] = None

class HMMBacktester:
    """Comprehensive backtester for top/bottom proximity count method"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.hmm_brain = None
        self.fred_data = {}
        self.market_data = {}
        self.daily_backtest_data = []
        
        # HMM state labels (from hmm_brain.json)
        self.state_labels = ["Early Stress", "Bull", "Late Cycle", "Crisis", "Stress", "Neutral"]
        
        # FRED features used in HMM training
        self.fred_features = [
            "BAMLH0A0HYM2",  # HY spreads
            "BAMLC0A0CM",    # IG spreads  
            "T10Y2Y",        # 10yr-2yr yield curve
            "T10Y3M",        # 10yr-3mo yield curve
            "DGS10",         # 10yr yield
            "DGS2",          # 2yr yield
            "DFII10",        # Real 10yr yield
            "NFCI",          # Financial Conditions
            "ICSA"           # Weekly jobless claims
        ]
    
    def load_hmm_brain(self) -> Dict[str, Any]:
        """Load the trained HMM brain"""
        brain_path = os.path.join(self.data_dir, "hmm_brain.json")
        if not os.path.exists(brain_path):
            raise FileNotFoundError(f"HMM brain not found at {brain_path}")
        
        with open(brain_path, 'r') as f:
            self.hmm_brain = json.load(f)
        
        logger.info(f"Loaded HMM brain: {self.hmm_brain['n_states']} states, "
                   f"trained {self.hmm_brain['training_start']} to {self.hmm_brain['training_end']}")
        return self.hmm_brain
    
    def load_fred_data(self) -> Dict[str, pd.DataFrame]:
        """Load cached FRED data from CSV files"""
        fred_cache_dir = os.path.join(self.data_dir, "fred_cache")
        
        for feature in self.fred_features:
            csv_path = os.path.join(fred_cache_dir, f"{feature}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                self.fred_data[feature] = df
                logger.info(f"Loaded {feature}: {len(df)} records")
            else:
                logger.warning(f"FRED data not found: {csv_path}")
        
        return self.fred_data
    
    def fetch_market_data(self, start_date: str = "2012-01-01", end_date: str = "2026-12-31") -> Dict[str, pd.DataFrame]:
        """Fetch SPY and VIX data for the backtest period"""
        logger.info(f"Fetching market data from {start_date} to {end_date}")
        
        # Fetch SPY and VIX
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        self.market_data = {
            "SPY": spy[["Close"]].rename(columns={"Close": "spy_close"}),
            "VIX": vix[["Close"]].rename(columns={"Close": "vix"})
        }
        
        logger.info(f"Fetched SPY: {len(spy)} records, VIX: {len(vix)} records")
        return self.market_data
    
    def build_feature_matrix(self, start_date: str = "2012-04-02", end_date: str = "2026-04-10") -> pd.DataFrame:
        """
        Reconstruct the feature matrix used for HMM training
        
        This replicates the preprocessing pipeline:
        1. Raw FRED data + VIX
        2. EWMA smoothing (10-day span)
        3. Rolling z-score (5-year window)  
        4. Cap at ±3σ
        """
        logger.info(f"Building feature matrix from {start_date} to {end_date}")
        
        # Create business day index
        date_range = pd.bdate_range(start=start_date, end=end_date)
        feature_df = pd.DataFrame(index=date_range)
        
        # Add FRED features
        for feature in self.fred_features:
            if feature in self.fred_data:
                # Forward fill and align to business days
                series = self.fred_data[feature].iloc[:, 0]  # First column
                feature_df[feature] = series.reindex(date_range, method='ffill')
            else:
                # Fill with neutral values if missing
                feature_df[feature] = 0
                logger.warning(f"Missing FRED feature {feature}, filling with zeros")
        
        # Add VIX
        if "VIX" in self.market_data:
            vix_series = self.market_data["VIX"]["vix"]
            feature_df["VIX"] = vix_series.reindex(date_range, method='ffill')
        else:
            feature_df["VIX"] = 15  # Neutral VIX level
            logger.warning("Missing VIX data, filling with 15")
        
        # Apply preprocessing pipeline
        logger.info("Applying EWMA smoothing (10-day span)")
        smoothed_df = feature_df.ewm(span=10, adjust=False).mean()
        
        logger.info("Applying rolling z-score (5-year window)")
        zscore_df = smoothed_df.rolling(window=5*252, min_periods=252).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        logger.info("Capping at ±3σ")
        capped_df = zscore_df.clip(-3, 3)
        
        # Drop initial warmup period (5 years)
        warmup_cutoff = pd.Timestamp(start_date) + pd.DateOffset(years=5)
        final_df = capped_df[capped_df.index >= warmup_cutoff].dropna()
        
        logger.info(f"Final feature matrix: {len(final_df)} rows, {len(final_df.columns)} features")
        return final_df
    
    def simulate_hmm_inference(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate HMM inference using the trained model parameters
        
        Note: This is a simplified simulation since we don't have the actual
        HMM model object. We'll use the trained means/covars to classify states.
        """
        logger.info("Running simulated HMM inference")
        
        if not self.hmm_brain:
            raise ValueError("HMM brain not loaded")
        
        means = np.array(self.hmm_brain['means'])
        n_states = self.hmm_brain['n_states']
        
        # Simple classification: assign each day to closest state by Euclidean distance
        results = []
        
        for date, row in feature_matrix.iterrows():
            features = row.values
            
            # Calculate distance to each state's mean
            distances = []
            for state_idx in range(n_states):
                state_mean = means[state_idx]
                # Handle case where we have different number of features
                min_len = min(len(features), len(state_mean))
                dist = np.linalg.norm(features[:min_len] - state_mean[:min_len])
                distances.append(dist)
            
            # Assign to closest state
            state_idx = np.argmin(distances)
            state_label = self.state_labels[state_idx]
            
            # Simulate state probabilities (softmax of negative distances)
            probs = np.exp(-np.array(distances))
            probs = probs / probs.sum()
            
            # Calculate entropy and ll_zscore
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            confidence = probs.max()
            
            # Simple ll_zscore simulation (use distance to assigned state)
            ll_zscore = -distances[state_idx] / 10  # Normalize roughly
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'state_idx': int(state_idx),
                'state_label': state_label,
                'confidence': float(confidence),
                'entropy': float(entropy),
                'll_zscore': float(ll_zscore),
                'state_probabilities': probs.tolist()
            })
        
        hmm_df = pd.DataFrame(results)
        hmm_df['date'] = pd.to_datetime(hmm_df['date'])
        hmm_df.set_index('date', inplace=True)
        
        logger.info(f"Completed HMM inference: {len(hmm_df)} predictions")
        return hmm_df
    
    def calculate_regime_scores(self, feature_matrix: pd.DataFrame, hmm_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime scores and macro scores from feature data
        
        This replicates the regime calculation logic from the modules
        """
        logger.info("Calculating daily regime and macro scores")
        
        regime_df = hmm_results.copy()
        
        # Simple regime score: average of key risk signals (HY spreads, VIX, NFCI)
        key_features = ["BAMLH0A0HYM2", "VIX", "NFCI"]
        available_features = [f for f in key_features if f in feature_matrix.columns]
        
        if available_features:
            regime_raw = feature_matrix[available_features].mean(axis=1)
            # Normalize to [-1, +1] range  
            regime_df['regime_score'] = np.tanh(regime_raw / 2)
        else:
            regime_df['regime_score'] = 0
            logger.warning("No key regime features available, using zeros")
        
        # Macro score: inverse of financial stress (0-100 scale)
        if "NFCI" in feature_matrix.columns:
            nfci_norm = feature_matrix["NFCI"]
            macro_raw = 50 - (nfci_norm * 15)  # Convert z-score to 0-100
            regime_df['macro_score'] = np.clip(macro_raw, 0, 100)
        else:
            regime_df['macro_score'] = 50
            logger.warning("No NFCI data available, using neutral macro score")
        
        # Conviction: simplified calculation based on trend consistency
        if len(regime_df) > 10:
            regime_df['velocity'] = regime_df['macro_score'].diff(6).fillna(0)
            regime_df['conviction'] = np.abs(regime_df['velocity']) * 2
        else:
            regime_df['velocity'] = 0
            regime_df['conviction'] = 0
        
        return regime_df
    
    def calculate_signal_strength(self, regime_score: float, macro_score: float, conviction: float,
                                entropy: float, ll_zscore: float, hmm_state: str,
                                velocity: float = 0.0, wyckoff_phase: str = None, wyckoff_conf: float = 0) -> ProximityAnalysis:
        """
        Calculate top/bottom proximity signals using the same logic as backfill_signal_analysis.py
        """
        # Input parameters
        _tb_regime = float(regime_score or 0)
        _tb_macro = float(macro_score or 50)
        _tb_conv = float(conviction or 0)
        _tb_entropy = float(entropy or 0)
        _tb_ll_z = float(ll_zscore or 0)
        _tb_hmm_label = str(hmm_state or "")
        _tb_vel = float(velocity or 0)
        
        _top_signals = []
        _bottom_signals = []
        
        # ──────────────────────────────────────────────────────────────────────
        # TOP SIGNALS — calibrated thresholds (empirical avg at 8 known peaks)
        # ──────────────────────────────────────────────────────────────────────
        
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
        
        # HMM state-based top signals
        if _tb_hmm_label in ("Late Cycle", "Stress", "Early Stress"):
            value = 65
            _top_signals.append(SignalData(f"HMM {_tb_hmm_label} state", value, value >= 50))
        
        # Wyckoff top signals
        if wyckoff_phase == "Distribution":
            value = min(100, wyckoff_conf)
            _top_signals.append(SignalData(f"Wyckoff Distribution ({wyckoff_conf}% conf)", value, value >= 50))
        
        # ──────────────────────────────────────────────────────────────────────
        # BOTTOM SIGNALS — calibrated thresholds (empirical avg at 8 known troughs)
        # ──────────────────────────────────────────────────────────────────────
        
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
        
        # HMM state-based bottom signals
        if _tb_hmm_label == "Crisis":
            value = 75
            _bottom_signals.append(SignalData("HMM Crisis state", value, value >= 50))
        
        # Wyckoff bottom signals
        if wyckoff_phase == "Accumulation":
            value = min(100, wyckoff_conf)
            _bottom_signals.append(SignalData(f"Wyckoff Accumulation ({wyckoff_conf}% conf)", value, value >= 50))
        
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
    
    def identify_turning_points(self, market_data: pd.DataFrame, lookback: int = 30, threshold: float = 0.05) -> Dict[str, List[str]]:
        """
        Identify systematic turning points in market data
        
        Returns:
        - peaks: Local maxima followed by ≥5% decline within 30 days
        - troughs: Local minima followed by ≥5% rally within 30 days
        """
        logger.info(f"Identifying turning points: {threshold*100}% threshold, {lookback} day window")
        
        spy_data = market_data["SPY"]["spy_close"] if "SPY" in market_data else pd.Series()
        
        if spy_data.empty:
            logger.warning("No SPY data available for turning point identification")
            return {"peaks": [], "troughs": []}
        
        # Convert to numpy for faster processing
        prices = spy_data.values
        dates = spy_data.index
        n = len(prices)
        
        peaks = []
        troughs = []
        
        for i in range(lookback, n - lookback):
            current_date = dates[i]
            current_price = prices[i]
            
            # Check for peak: local maximum
            is_local_max = True
            for j in range(max(0, i - lookback), min(n, i + lookback + 1)):
                if j != i and prices[j] >= current_price:
                    is_local_max = False
                    break
            
            if is_local_max:
                # Check if followed by significant decline
                end_idx = min(n, i + lookback + 1)
                future_prices = prices[i:end_idx]
                if len(future_prices) > 1:
                    min_future = np.min(future_prices)
                    decline = (current_price - min_future) / current_price
                    
                    if decline >= threshold:
                        peaks.append(current_date.strftime('%Y-%m-%d'))
            
            # Check for trough: local minimum
            is_local_min = True
            for j in range(max(0, i - lookback), min(n, i + lookback + 1)):
                if j != i and prices[j] <= current_price:
                    is_local_min = False
                    break
            
            if is_local_min:
                # Check if followed by significant rally
                end_idx = min(n, i + lookback + 1)
                future_prices = prices[i:end_idx]
                if len(future_prices) > 1:
                    max_future = np.max(future_prices)
                    rally = (max_future - current_price) / current_price
                    
                    if rally >= threshold:
                        troughs.append(current_date.strftime('%Y-%m-%d'))
        
        logger.info(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
        return {"peaks": peaks, "troughs": troughs}
    
    def calculate_performance_labels(self, regime_df: pd.DataFrame, turning_points: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Calculate forward-looking performance labels for each trading day
        """
        logger.info("Calculating performance labels")
        
        result_df = regime_df.copy()
        result_df['days_to_peak'] = None
        result_df['days_to_trough'] = None
        
        # Convert turning points to timestamps
        peak_dates = [pd.Timestamp(date) for date in turning_points['peaks']]
        trough_dates = [pd.Timestamp(date) for date in turning_points['troughs']]
        
        for i, (date, row) in enumerate(result_df.iterrows()):
            # Days to next peak
            future_peaks = [peak for peak in peak_dates if peak > date]
            if future_peaks:
                days_to_peak = (min(future_peaks) - date).days
                result_df.loc[date, 'days_to_peak'] = days_to_peak
            
            # Days to next trough
            future_troughs = [trough for trough in trough_dates if trough > date]
            if future_troughs:
                days_to_trough = (min(future_troughs) - date).days
                result_df.loc[date, 'days_to_trough'] = days_to_trough
        
        return result_df
    
    def run_daily_backtest(self) -> List[DailyBacktestRow]:
        """
        Run the complete daily backtest:
        1. Reconstruct daily market data for 2012-2026
        2. Apply signal calculations to each day  
        3. Return comprehensive results
        """
        logger.info("Starting comprehensive HMM backtest")
        
        # Step 1: Load all required data
        self.load_hmm_brain()
        self.load_fred_data()
        self.fetch_market_data()
        
        # Step 2: Reconstruct the HMM training feature matrix
        feature_matrix = self.build_feature_matrix()
        
        # Step 3: Run HMM inference on full period
        hmm_results = self.simulate_hmm_inference(feature_matrix)
        
        # Step 4: Calculate regime scores and macro data
        regime_df = self.calculate_regime_scores(feature_matrix, hmm_results)
        
        # Step 5: Merge with market data
        spy_data = self.market_data["SPY"]["spy_close"] if "SPY" in self.market_data else pd.Series()
        vix_data = self.market_data["VIX"]["vix"] if "VIX" in self.market_data else pd.Series()
        
        for date in regime_df.index:
            regime_df.loc[date, 'spy_price'] = spy_data.get(date, np.nan)
            regime_df.loc[date, 'vix'] = vix_data.get(date, np.nan)
        
        # Step 6: Identify turning points
        turning_points = self.identify_turning_points(self.market_data)
        regime_df = self.calculate_performance_labels(regime_df, turning_points)
        
        # Step 7: Run signal calculations for each day
        logger.info("Calculating proximity signals for each trading day")
        
        backtest_results = []
        for date, row in regime_df.iterrows():
            # Calculate proximity signals
            proximity = self.calculate_signal_strength(
                regime_score=row['regime_score'],
                macro_score=row['macro_score'],
                conviction=row['conviction'],
                entropy=row['entropy'],
                ll_zscore=row['ll_zscore'],
                hmm_state=row['state_label'],
                velocity=row.get('velocity', 0)
            )
            
            # Create backtest row
            backtest_row = DailyBacktestRow(
                date=date.strftime('%Y-%m-%d'),
                regime_score=row['regime_score'],
                macro_score=row['macro_score'],
                conviction=row['conviction'],
                entropy=row['entropy'],
                ll_zscore=row['ll_zscore'],
                hmm_state=row['state_label'],
                hmm_state_idx=row['state_idx'],
                velocity=row.get('velocity', 0),
                spy_price=row.get('spy_price', np.nan),
                vix=row.get('vix', np.nan),
                
                top_signals=[s.name for s in proximity.top_signals if s.threshold_met],
                bottom_signals=[s.name for s in proximity.bottom_signals if s.threshold_met],
                top_count=proximity.top_count,
                bottom_count=proximity.bottom_count,
                top_score=proximity.top_score,
                bottom_score=proximity.bottom_score,
                net_lean=proximity.top_count - proximity.bottom_count,
                
                days_to_peak=row.get('days_to_peak'),
                days_to_trough=row.get('days_to_trough')
            )
            
            backtest_results.append(backtest_row)
        
        self.daily_backtest_data = backtest_results
        logger.info(f"Completed backtest: {len(backtest_results)} trading days analyzed")
        
        return backtest_results
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of the proximity count method across all trading days
        
        Returns comprehensive performance metrics including:
        - Overall accuracy vs cherry-picked 88%
        - False positive/negative rates
        - Performance by signal strength (+1/+2/+3 edge)
        - Breakdown by time periods and market regimes
        """
        if not self.daily_backtest_data:
            raise ValueError("No backtest data available. Run run_daily_backtest() first.")
        
        logger.info("Analyzing proximity count method performance")
        
        results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "methodology": "Full HMM training dataset 2012-2026, proximity count method, systematic turning points",
            "dataset_summary": {},
            "cherry_picked_comparison": {},
            "overall_performance": {},
            "signal_strength_analysis": {},
            "time_period_breakdown": {},
            "false_positive_analysis": {},
            "turning_point_performance": {}
        }
        
        df = pd.DataFrame([asdict(row) for row in self.daily_backtest_data])
        df['date'] = pd.to_datetime(df['date'])
        
        # Dataset summary
        results["dataset_summary"] = {
            "total_trading_days": len(df),
            "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            "years_covered": round((df['date'].max() - df['date'].min()).days / 365.25, 1),
            "top_signal_fire_days": len(df[df['top_count'] > 0]),
            "bottom_signal_fire_days": len(df[df['bottom_count'] > 0]),
            "no_signal_days": len(df[(df['top_count'] == 0) & (df['bottom_count'] == 0)]),
            "conflicting_signal_days": len(df[(df['top_count'] > 0) & (df['bottom_count'] > 0)])
        }
        
        # Overall performance metrics
        # Define prediction accuracy
        df['top_prediction'] = df['top_count'] > 0
        df['bottom_prediction'] = df['bottom_count'] > 0
        
        # Define "correct" predictions based on forward returns
        df['forward_5d_return'] = df['spy_price'].pct_change(5).shift(-5)
        df['forward_10d_return'] = df['spy_price'].pct_change(10).shift(-10)
        df['forward_20d_return'] = df['spy_price'].pct_change(20).shift(-20)
        
        # Correct top prediction = negative forward return
        # Correct bottom prediction = positive forward return  
        performance_windows = [5, 10, 20]
        overall_perf = {}
        
        for window in performance_windows:
            forward_ret_col = f'forward_{window}d_return'
            
            if forward_ret_col in df.columns:
                # TOP signal accuracy
                top_pred_days = df[df['top_prediction']]
                if len(top_pred_days) > 0:
                    top_correct = len(top_pred_days[top_pred_days[forward_ret_col] < 0])
                    top_accuracy = top_correct / len(top_pred_days)
                else:
                    top_accuracy = 0
                
                # BOTTOM signal accuracy  
                bottom_pred_days = df[df['bottom_prediction']]
                if len(bottom_pred_days) > 0:
                    bottom_correct = len(bottom_pred_days[bottom_pred_days[forward_ret_col] > 0])
                    bottom_accuracy = bottom_correct / len(bottom_pred_days)
                else:
                    bottom_accuracy = 0
                
                # Combined accuracy
                total_signals = len(top_pred_days) + len(bottom_pred_days)
                total_correct = top_correct + (bottom_correct if len(bottom_pred_days) > 0 else 0)
                combined_accuracy = total_correct / total_signals if total_signals > 0 else 0
                
                overall_perf[f"{window}_day"] = {
                    "top_signal_accuracy": round(top_accuracy * 100, 1),
                    "bottom_signal_accuracy": round(bottom_accuracy * 100, 1), 
                    "combined_accuracy": round(combined_accuracy * 100, 1),
                    "top_signals_fired": len(top_pred_days),
                    "bottom_signals_fired": len(bottom_pred_days),
                    "total_signals": total_signals
                }
        
        results["overall_performance"] = overall_perf
        
        # Signal strength analysis (+1/+2/+3 edge)
        signal_strength = {}
        for strength in [1, 2, 3]:
            strong_top = df[df['top_count'] >= strength]
            strong_bottom = df[df['bottom_count'] >= strength]
            
            strength_stats = {
                "top_days": len(strong_top),
                "bottom_days": len(strong_bottom),
                "top_accuracy_5d": 0,
                "bottom_accuracy_5d": 0
            }
            
            if len(strong_top) > 0 and 'forward_5d_return' in df.columns:
                top_correct = len(strong_top[strong_top['forward_5d_return'] < 0])
                strength_stats["top_accuracy_5d"] = round(top_correct / len(strong_top) * 100, 1)
            
            if len(strong_bottom) > 0 and 'forward_5d_return' in df.columns:
                bottom_correct = len(strong_bottom[strong_bottom['forward_5d_return'] > 0])
                strength_stats["bottom_accuracy_5d"] = round(bottom_correct / len(strong_bottom) * 100, 1)
            
            signal_strength[f"plus_{strength}_edge"] = strength_stats
        
        results["signal_strength_analysis"] = signal_strength
        
        # Cherry-picked comparison
        # Load the original 16 crash dates for comparison
        cherry_picked_path = os.path.join(self.data_dir, "proximity_calibration.json")
        if os.path.exists(cherry_picked_path):
            with open(cherry_picked_path, 'r') as f:
                cherry_data = json.load(f)
            
            cherry_dates = [record['date'] for record in cherry_data.get('records', [])]
            cherry_accuracy_count = 0
            cherry_total = len(cherry_dates)
            
            for record in cherry_data.get('records', []):
                date_str = record['date']
                role = record['role']
                
                # Find corresponding day in backtest
                matching_days = df[df['date'] == pd.Timestamp(date_str)]
                if len(matching_days) > 0:
                    day = matching_days.iloc[0]
                    if (role == "PEAK" and day['top_count'] > 0) or (role == "TROUGH" and day['bottom_count'] > 0):
                        cherry_accuracy_count += 1
            
            cherry_picked_accuracy = round(cherry_accuracy_count / cherry_total * 100, 1) if cherry_total > 0 else 0
            
            results["cherry_picked_comparison"] = {
                "original_claim": "88% accuracy on 16 handpicked crash dates",
                "our_replication": f"{cherry_picked_accuracy}% accuracy on {cherry_total} dates",
                "cherry_picked_dates_tested": cherry_total,
                "cherry_picked_hits": cherry_accuracy_count,
                "full_dataset_days": len(df),
                "accuracy_difference": f"Cherry-picked: {cherry_picked_accuracy}% vs Full dataset: {overall_perf.get('5_day', {}).get('combined_accuracy', 0)}% (5-day)"
            }
        
        # Time period breakdown
        df['year'] = df['date'].dt.year
        yearly_performance = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            if len(year_data) > 0 and 'forward_5d_return' in df.columns:
                top_days = year_data[year_data['top_prediction']]
                bottom_days = year_data[year_data['bottom_prediction']]
                
                year_stats = {
                    "trading_days": len(year_data),
                    "top_signals": len(top_days),
                    "bottom_signals": len(bottom_days),
                    "avg_regime_score": round(year_data['regime_score'].mean(), 3),
                    "avg_macro_score": round(year_data['macro_score'].mean(), 1)
                }
                
                if len(top_days) > 0:
                    top_correct = len(top_days[top_days['forward_5d_return'] < 0])
                    year_stats["top_accuracy"] = round(top_correct / len(top_days) * 100, 1)
                
                if len(bottom_days) > 0:
                    bottom_correct = len(bottom_days[bottom_days['forward_5d_return'] > 0])
                    year_stats["bottom_accuracy"] = round(bottom_correct / len(bottom_days) * 100, 1)
                
                yearly_performance[str(year)] = year_stats
        
        results["time_period_breakdown"] = yearly_performance
        
        # False positive analysis
        no_signal_days = df[(df['top_count'] == 0) & (df['bottom_count'] == 0)]
        signal_days = df[(df['top_count'] > 0) | (df['bottom_count'] > 0)]
        
        results["false_positive_analysis"] = {
            "signal_fire_rate": round(len(signal_days) / len(df) * 100, 1),
            "no_signal_days": len(no_signal_days),
            "signal_days": len(signal_days),
            "daily_signal_frequency": round(len(signal_days) / len(df) * 365.25, 1),  # signals per year
            "most_common_top_signals": self._get_most_common_signals(df, 'top_signals'),
            "most_common_bottom_signals": self._get_most_common_signals(df, 'bottom_signals')
        }
        
        # Turning point performance (systematic vs cherry-picked)
        turning_point_dates = []
        with open(cherry_picked_path, 'r') as f:
            cherry_data = json.load(f)
            for record in cherry_data.get('records', []):
                turning_point_dates.append(record['date'])
        
        systematic_peaks = df[~df['days_to_peak'].isna() & (df['days_to_peak'] <= 30)]
        systematic_troughs = df[~df['days_to_trough'].isna() & (df['days_to_trough'] <= 30)]
        
        results["turning_point_performance"] = {
            "systematic_peaks_identified": len(systematic_peaks),
            "systematic_troughs_identified": len(systematic_troughs), 
            "systematic_peak_signal_rate": round(len(systematic_peaks[systematic_peaks['top_count'] > 0]) / max(1, len(systematic_peaks)) * 100, 1),
            "systematic_trough_signal_rate": round(len(systematic_troughs[systematic_troughs['bottom_count'] > 0]) / max(1, len(systematic_troughs)) * 100, 1),
            "cherry_picked_dates": len(turning_point_dates)
        }
        
        return results
    
    def _get_most_common_signals(self, df: pd.DataFrame, signal_col: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently firing signals"""
        all_signals = []
        for signals_list in df[signal_col]:
            if isinstance(signals_list, list):
                all_signals.extend(signals_list)
        
        if not all_signals:
            return []
        
        from collections import Counter
        signal_counts = Counter(all_signals)
        
        return [
            {"signal_name": signal, "fire_count": count, "fire_rate": round(count / len(df) * 100, 2)}
            for signal, count in signal_counts.most_common(top_n)
        ]
    
    def save_results(self, performance_results: Dict[str, Any], filename: str = "hmm_full_backtest_results.json"):
        """Save the comprehensive backtest results to JSON"""
        output_path = os.path.join(os.getcwd(), filename)
        
        # Also save the daily backtest data
        daily_data_path = os.path.join(os.getcwd(), "hmm_daily_backtest_data.json")
        daily_data = [asdict(row) for row in self.daily_backtest_data]
        
        with open(output_path, 'w') as f:
            json.dump(performance_results, f, indent=2, default=str)
        
        with open(daily_data_path, 'w') as f:
            json.dump(daily_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Daily data saved to {daily_data_path}")
        return output_path

def main():
    """Run the comprehensive HMM backtest"""
    print("🚀 Starting Comprehensive HMM Backtest")
    print("=" * 60)
    print("OBJECTIVE: Test '88% accuracy' claim against 12+ years of daily data")
    print("SCOPE: Full HMM training dataset (2012-2026), not cherry-picked dates")
    print("METHOD: Top/bottom proximity count method with systematic turning points")
    print("=" * 60)
    
    try:
        # Initialize backtester
        backtester = HMMBacktester()
        
        # Run comprehensive backtest
        print("\n📊 Running daily backtest...")
        backtest_data = backtester.run_daily_backtest()
        
        # Analyze performance
        print("\n📈 Analyzing performance...")
        performance_results = backtester.analyze_performance()
        
        # Save results
        print("\n💾 Saving results...")
        output_file = backtester.save_results(performance_results)
        
        # Print key findings
        print("\n" + "=" * 60)
        print("🎯 KEY FINDINGS")
        print("=" * 60)
        
        dataset_summary = performance_results["dataset_summary"]
        overall_perf = performance_results["overall_performance"]
        cherry_comparison = performance_results["cherry_picked_comparison"]
        
        print(f"📅 Dataset: {dataset_summary['total_trading_days']:,} trading days ({dataset_summary['date_range']})")
        print(f"🔥 Signal frequency: {performance_results['false_positive_analysis']['signal_fire_rate']}% of days")
        
        if "5_day" in overall_perf:
            perf_5d = overall_perf["5_day"]
            print(f"📊 5-day accuracy: {perf_5d['combined_accuracy']}% (vs 88% cherry-picked claim)")
            print(f"   • TOP signals: {perf_5d['top_signal_accuracy']}% ({perf_5d['top_signals_fired']} days)")
            print(f"   • BOTTOM signals: {perf_5d['bottom_signal_accuracy']}% ({perf_5d['bottom_signals_fired']} days)")
        
        if cherry_comparison:
            print(f"🍒 Cherry-picked replication: {cherry_comparison.get('our_replication', 'N/A')}")
        
        signal_strength = performance_results["signal_strength_analysis"]
        for edge, stats in signal_strength.items():
            if edge == "plus_3_edge":
                print(f"💪 Strong signals (+3): TOP {stats['top_accuracy_5d']}% ({stats['top_days']} days), "
                      f"BOTTOM {stats['bottom_accuracy_5d']}% ({stats['bottom_days']} days)")
        
        print(f"\n✅ Full results saved to: {output_file}")
        print(f"✅ Daily data saved to: hmm_daily_backtest_data.json")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()