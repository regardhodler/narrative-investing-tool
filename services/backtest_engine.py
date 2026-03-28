"""
Backtesting engine for signal strategies.

Backtestable signals:
- SMA Crossovers (golden/death cross)
- VIX Spikes (VIX > threshold → buy SPY)
- Regime Flips (Risk-Off → Risk-On from regime_history.json)
- Insider Clusters (3+ buys in 30 days on same stock)
"""

import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    signal_name: str
    ticker: str
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


def _compute_metrics(trades: list[dict], initial_capital: float = 10000) -> BacktestResult:
    """Compute backtest metrics from a list of trades."""
    result = BacktestResult(signal_name="", ticker="")
    if not trades:
        return result

    result.trades = trades
    result.num_trades = len(trades)

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    result.win_rate = len(wins) / len(returns) * 100 if returns else 0
    result.avg_return = np.mean(returns) if returns else 0
    result.total_return = sum(returns)

    # Equity curve
    equity = [initial_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))
    result.equity_curve = equity

    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown = round(max_dd, 2)

    return result


@st.cache_data(ttl=3600)
def backtest_sma_crossover(ticker: str = "SPY", short_window: int = 50, long_window: int = 200,
                           holding_days: int = 20, lookback_years: int = 5) -> BacktestResult:
    """Backtest SMA crossover (golden cross = buy signal)."""
    df = yf.download(ticker, period=f"{lookback_years}y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return BacktestResult(signal_name="SMA Crossover", ticker=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)

    close = df["Close"].dropna()
    if len(close) < long_window + holding_days:
        return BacktestResult(signal_name="SMA Crossover", ticker=ticker)

    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()

    # Detect golden crosses (short crosses above long)
    cross = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    signal_dates = cross[cross].index

    trades = []
    for entry_date in signal_dates:
        idx = close.index.get_loc(entry_date)
        exit_idx = min(idx + holding_days, len(close) - 1)
        entry_price = float(close.iloc[idx])
        exit_price = float(close.iloc[exit_idx])
        ret = (exit_price / entry_price - 1) * 100
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(close.index[exit_idx].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret, 2),
        })

    result = _compute_metrics(trades)
    result.signal_name = f"SMA {short_window}/{long_window} Crossover"
    result.ticker = ticker
    return result


@st.cache_data(ttl=3600)
def backtest_vix_spike(vix_threshold: float = 25, holding_days: int = 20,
                       lookback_years: int = 5) -> BacktestResult:
    """Backtest buying SPY when VIX spikes above threshold."""
    vix = yf.download("^VIX", period=f"{lookback_years}y", interval="1d", progress=False, auto_adjust=True)
    spy = yf.download("SPY", period=f"{lookback_years}y", interval="1d", progress=False, auto_adjust=True)

    if vix is None or spy is None or vix.empty or spy.empty:
        return BacktestResult(signal_name="VIX Spike", ticker="SPY")
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix.droplevel("Ticker", axis=1)
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel("Ticker", axis=1)

    vix_close = vix["Close"].dropna()
    spy_close = spy["Close"].dropna()

    # Find VIX spike entries (crosses above threshold, wasn't above yesterday)
    above = vix_close > vix_threshold
    cross_above = above & (~above.shift(1).fillna(False))
    signal_dates = cross_above[cross_above].index

    trades = []
    for entry_date in signal_dates:
        # Find matching SPY date
        spy_dates = spy_close.index
        valid = spy_dates[spy_dates >= entry_date]
        if len(valid) < 2:
            continue
        entry_idx = spy_close.index.get_loc(valid[0])
        exit_idx = min(entry_idx + holding_days, len(spy_close) - 1)
        entry_price = float(spy_close.iloc[entry_idx])
        exit_price = float(spy_close.iloc[exit_idx])
        ret = (exit_price / entry_price - 1) * 100
        trades.append({
            "entry_date": str(valid[0].date()),
            "exit_date": str(spy_close.index[exit_idx].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret, 2),
        })

    result = _compute_metrics(trades)
    result.signal_name = f"VIX Spike (>{vix_threshold})"
    result.ticker = "SPY"
    return result


@st.cache_data(ttl=3600)
def backtest_regime_flip(holding_days: int = 20) -> BacktestResult:
    """Backtest buying SPY on Risk-Off → Risk-On regime flips."""
    history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json")
    if not os.path.exists(history_file):
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    with open(history_file) as f:
        history = json.load(f)

    if len(history) < 2:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    # Find Risk-Off → Risk-On transitions
    flip_dates = []
    entries = sorted(history, key=lambda x: x.get("date", ""))
    for i in range(1, len(entries)):
        prev_regime = entries[i - 1].get("regime", "")
        curr_regime = entries[i].get("regime", "")
        if "Off" in prev_regime and "On" in curr_regime:
            flip_dates.append(entries[i]["date"])

    if not flip_dates:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    # Get SPY data covering the date range
    min_date = min(flip_dates)
    spy = yf.download("SPY", start=min_date, progress=False, auto_adjust=True)
    if spy is None or spy.empty:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel("Ticker", axis=1)
    spy_close = spy["Close"].dropna()

    trades = []
    for date_str in flip_dates:
        try:
            target = pd.Timestamp(date_str)
            valid = spy_close.index[spy_close.index >= target]
            if len(valid) < 2:
                continue
            entry_idx = spy_close.index.get_loc(valid[0])
            exit_idx = min(entry_idx + holding_days, len(spy_close) - 1)
            entry_price = float(spy_close.iloc[entry_idx])
            exit_price = float(spy_close.iloc[exit_idx])
            ret = (exit_price / entry_price - 1) * 100
            trades.append({
                "entry_date": str(valid[0].date()),
                "exit_date": str(spy_close.index[exit_idx].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret, 2),
            })
        except Exception:
            continue

    result = _compute_metrics(trades)
    result.signal_name = "Regime Flip (Off→On)"
    result.ticker = "SPY"
    return result


@st.cache_data(ttl=3600)
def backtest_insider_cluster(ticker: str = "AAPL", min_buys: int = 3,
                             cluster_days: int = 30, holding_days: int = 20) -> BacktestResult:
    """Backtest buying when insider buy clusters detected."""
    from services.sec_client import get_insider_trades

    df = get_insider_trades(ticker)
    if df.empty:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    buys = df[df["type"] == "Purchase"].copy()
    if buys.empty:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
    buys = buys.dropna(subset=["date"]).sort_values("date")

    # Find clusters: rolling window of cluster_days with min_buys+ purchases
    cluster_dates = []
    dates = buys["date"].tolist()
    i = 0
    while i < len(dates):
        window_end = dates[i] + pd.Timedelta(days=cluster_days)
        count = sum(1 for d in dates[i:] if d <= window_end)
        if count >= min_buys:
            cluster_dates.append(dates[i])
            # Skip past this cluster
            i += count
        else:
            i += 1

    if not cluster_dates:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    # Get price data
    price_data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if price_data is None or price_data.empty:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data.droplevel("Ticker", axis=1)
    close = price_data["Close"].dropna()

    trades = []
    for entry_date in cluster_dates:
        valid = close.index[close.index >= entry_date]
        if len(valid) < 2:
            continue
        entry_idx = close.index.get_loc(valid[0])
        exit_idx = min(entry_idx + holding_days, len(close) - 1)
        entry_price = float(close.iloc[entry_idx])
        exit_price = float(close.iloc[exit_idx])
        ret = (exit_price / entry_price - 1) * 100
        trades.append({
            "entry_date": str(valid[0].date()),
            "exit_date": str(close.index[exit_idx].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret, 2),
        })

    result = _compute_metrics(trades)
    result.signal_name = f"Insider Cluster ({min_buys}+ buys/{cluster_days}d)"
    result.ticker = ticker
    return result


# ── Walk-Forward Validation ────────────────────────────────────────────────────

def _window_metrics(trades: list[dict]) -> dict:
    """Compute win rate and avg return for a slice of trades."""
    if not trades:
        return {"win_rate": 0.0, "avg_return": 0.0, "num_trades": 0}
    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    return {
        "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else 0.0,
        "avg_return": round(float(np.mean(returns)), 2) if returns else 0.0,
        "num_trades": len(returns),
    }


def _oos_confidence(oos_win_rate: float, oos_avg_return: float, oos_trades: int) -> str:
    if oos_trades < 5:
        return "INSUFFICIENT DATA"
    if oos_win_rate >= 55 and oos_avg_return > 0:
        return "HIGH"
    if oos_win_rate >= 50 or oos_avg_return > 0:
        return "MODERATE"
    return "LOW"


@st.cache_data(ttl=3600)
def walk_forward_sma(
    ticker: str,
    short_w: int,
    long_w: int,
    hold_days: int,
    train_months: int,
    test_months: int,
    lookback_years: int,
) -> dict:
    """Walk-forward validation for SMA crossover strategy.

    Slides a (train_months + test_months) window across lookback_years of data,
    stepping by test_months each time. Measures out-of-sample performance per window.

    Returns dict with: windows, oos_equity, oos_win_rate, oos_avg_return,
                       oos_total_trades, in_sample_win_rate, in_sample_avg_return,
                       signal_name, ticker, confidence.
    """
    total_years = lookback_years + train_months / 12
    raw = yf.download(ticker, period=f"{int(total_years) + 1}y", interval="1d",
                      progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel("Ticker", axis=1)

    close = raw["Close"].dropna()
    if len(close) < long_w + hold_days + 20:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}

    # Generate all crossover signals on the full series (no lookahead — SMAs are backward-looking)
    sma_s = close.rolling(short_w).mean()
    sma_l = close.rolling(long_w).mean()
    cross = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    signal_dates = cross[cross].index

    # Build all trades over the full history
    all_trades = []
    for entry_date in signal_dates:
        idx = close.index.get_loc(entry_date)
        exit_idx = min(idx + hold_days, len(close) - 1)
        entry_price = float(close.iloc[idx])
        exit_price  = float(close.iloc[exit_idx])
        ret = (exit_price / entry_price - 1) * 100
        all_trades.append({
            "entry_date": entry_date,
            "exit_date":  close.index[exit_idx],
            "entry_price": round(entry_price, 2),
            "exit_price":  round(exit_price, 2),
            "return_pct":  round(ret, 2),
        })

    if not all_trades:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}

    # Slide the window
    start = close.index[0]
    end   = close.index[-1]
    td_train = pd.DateOffset(months=train_months)
    td_test  = pd.DateOffset(months=test_months)

    windows = []
    win_num = 1
    cursor = start

    while True:
        train_start = cursor
        train_end   = cursor + td_train
        test_start  = train_end
        test_end    = test_start + td_test
        if test_end > end:
            break

        train_trades = [t for t in all_trades if train_start <= t["entry_date"] < train_end]
        test_trades  = [t for t in all_trades if test_start  <= t["entry_date"] < test_end]

        win_m = _window_metrics(train_trades)
        oos_m = _window_metrics(test_trades)
        windows.append({
            "n":                  win_num,
            "train_start":        train_start.strftime("%Y-%m-%d"),
            "train_end":          train_end.strftime("%Y-%m-%d"),
            "test_start":         test_start.strftime("%Y-%m-%d"),
            "test_end":           test_end.strftime("%Y-%m-%d"),
            "train_win_rate":     win_m["win_rate"],
            "train_avg_return":   win_m["avg_return"],
            "train_trades":       win_m["num_trades"],
            "test_win_rate":      oos_m["win_rate"],
            "test_avg_return":    oos_m["avg_return"],
            "test_trades":        oos_m["num_trades"],
            "test_trade_list":    [{"return_pct": t["return_pct"]} for t in test_trades],
        })

        cursor += td_test
        win_num += 1

    # Build concatenated OOS equity curve
    oos_equity = [10000.0]
    for w in windows:
        for t in w.get("test_trade_list", []):
            oos_equity.append(oos_equity[-1] * (1 + t["return_pct"] / 100))

    # Aggregate OOS stats
    all_oos = [t for w in windows for t in w.get("test_trade_list", [])]
    agg = _window_metrics([{"return_pct": t["return_pct"]} for t in all_oos])
    all_is = [{"win_rate": w["train_win_rate"], "avg_return": w["train_avg_return"]} for w in windows if w["train_trades"] > 0]
    is_win  = round(float(np.mean([x["win_rate"] for x in all_is])), 1) if all_is else 0.0
    is_ret  = round(float(np.mean([x["avg_return"] for x in all_is])), 2) if all_is else 0.0

    return {
        "windows":             windows,
        "oos_equity":          oos_equity,
        "oos_win_rate":        agg["win_rate"],
        "oos_avg_return":      agg["avg_return"],
        "oos_total_trades":    agg["num_trades"],
        "in_sample_win_rate":  is_win,
        "in_sample_avg_return": is_ret,
        "signal_name":         f"SMA {short_w}/{long_w} Walk-Forward",
        "ticker":              ticker,
        "confidence":          _oos_confidence(agg["win_rate"], agg["avg_return"], agg["num_trades"]),
    }


@st.cache_data(ttl=3600)
def walk_forward_vix(
    vix_threshold: float,
    hold_days: int,
    train_months: int,
    test_months: int,
    lookback_years: int,
) -> dict:
    """Walk-forward validation for VIX spike strategy."""
    total_years = lookback_years + train_months / 12
    vix_raw = yf.download("^VIX", period=f"{int(total_years) + 1}y", interval="1d",
                          progress=False, auto_adjust=True)
    spy_raw = yf.download("SPY",  period=f"{int(total_years) + 1}y", interval="1d",
                          progress=False, auto_adjust=True)
    if vix_raw is None or spy_raw is None or vix_raw.empty or spy_raw.empty:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}
    for _r in (vix_raw, spy_raw):
        if isinstance(_r.columns, pd.MultiIndex):
            _r.columns = _r.columns.droplevel("Ticker")

    vix_c = vix_raw["Close"].dropna()
    spy_c = spy_raw["Close"].dropna()

    above = vix_c > vix_threshold
    cross = above & (~above.shift(1).fillna(False))
    signal_dates = cross[cross].index

    all_trades = []
    for entry_date in signal_dates:
        valid = spy_c.index[spy_c.index >= entry_date]
        if len(valid) < 2:
            continue
        entry_idx = spy_c.index.get_loc(valid[0])
        exit_idx  = min(entry_idx + hold_days, len(spy_c) - 1)
        entry_price = float(spy_c.iloc[entry_idx])
        exit_price  = float(spy_c.iloc[exit_idx])
        ret = (exit_price / entry_price - 1) * 100
        all_trades.append({
            "entry_date":  valid[0],
            "return_pct":  round(ret, 2),
        })

    if not all_trades:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}

    start = vix_c.index[0]
    end   = vix_c.index[-1]
    td_train = pd.DateOffset(months=train_months)
    td_test  = pd.DateOffset(months=test_months)

    windows = []
    win_num = 1
    cursor  = start

    while True:
        train_start = cursor
        train_end   = cursor + td_train
        test_start  = train_end
        test_end    = test_start + td_test
        if test_end > end:
            break

        train_trades = [t for t in all_trades if train_start <= t["entry_date"] < train_end]
        test_trades  = [t for t in all_trades if test_start  <= t["entry_date"] < test_end]

        win_m = _window_metrics(train_trades)
        oos_m = _window_metrics(test_trades)
        windows.append({
            "n": win_num,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "test_start":  test_start.strftime("%Y-%m-%d"),
            "test_end":    test_end.strftime("%Y-%m-%d"),
            "train_win_rate":   win_m["win_rate"],
            "train_avg_return": win_m["avg_return"],
            "train_trades":     win_m["num_trades"],
            "test_win_rate":    oos_m["win_rate"],
            "test_avg_return":  oos_m["avg_return"],
            "test_trades":      oos_m["num_trades"],
            "test_trade_list":  [{"return_pct": t["return_pct"]} for t in test_trades],
        })
        cursor += td_test
        win_num += 1

    oos_equity = [10000.0]
    for w in windows:
        for t in w.get("test_trade_list", []):
            oos_equity.append(oos_equity[-1] * (1 + t["return_pct"] / 100))

    all_oos = [t for w in windows for t in w.get("test_trade_list", [])]
    agg = _window_metrics([{"return_pct": t["return_pct"]} for t in all_oos])
    all_is = [{"win_rate": w["train_win_rate"], "avg_return": w["train_avg_return"]} for w in windows if w["train_trades"] > 0]
    is_win = round(float(np.mean([x["win_rate"] for x in all_is])), 1) if all_is else 0.0
    is_ret = round(float(np.mean([x["avg_return"] for x in all_is])), 2) if all_is else 0.0

    return {
        "windows":              windows,
        "oos_equity":           oos_equity,
        "oos_win_rate":         agg["win_rate"],
        "oos_avg_return":       agg["avg_return"],
        "oos_total_trades":     agg["num_trades"],
        "in_sample_win_rate":   is_win,
        "in_sample_avg_return": is_ret,
        "signal_name":          f"VIX Spike (>{vix_threshold}) Walk-Forward",
        "ticker":               "SPY",
        "confidence":           _oos_confidence(agg["win_rate"], agg["avg_return"], agg["num_trades"]),
    }
