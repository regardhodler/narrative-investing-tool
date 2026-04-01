"""
Backtesting engine for signal strategies.

Backtestable signals:
- SMA Crossovers (golden/death cross)
- VIX Spikes (VIX > threshold → buy SPY)
- Regime Flips (Risk-Off → Risk-On from regime_history.json)
- Insider Clusters (3+ buys in 30 days on same stock)

All strategies use ATR-based exits (trailing stop + profit target) instead of
fixed hold days. Each trade stays open until 2×ATR trailing stop or 3×ATR
profit target fires, giving a natural 1.5:1 R:R on every signal.
"""

import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dataclasses import dataclass, field

_ATR_PERIOD      = 14
_DEFAULT_STOP    = 2.0   # multiplier — trailing stop
_DEFAULT_TARGET  = 3.0   # multiplier — profit target


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
    exit_reason_counts: dict = field(default_factory=dict)


# ── ATR exit engine ────────────────────────────────────────────────────────────

def _atr_exit_trade(
    df: pd.DataFrame,
    entry_idx: int,
    is_short: bool = False,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> dict:
    """Walk OHLC forward from entry_idx using ATR trailing stop + profit target.

    Returns trade dict with entry_date, exit_date, entry_price, exit_price,
    return_pct, exit_reason, atr.
    """
    # ATR(14) at entry — use pre-entry window
    pre = df.iloc[max(0, entry_idx - _ATR_PERIOD * 2): entry_idx + 1]
    high_pre  = pre["High"]
    low_pre   = pre["Low"]
    cp        = pre["Close"].shift(1)
    tr = pd.concat([high_pre - low_pre, (high_pre - cp).abs(), (low_pre - cp).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(_ATR_PERIOD).mean().iloc[-1]) if len(tr) >= _ATR_PERIOD else float(tr.mean())
    if np.isnan(atr) or atr <= 0:
        atr = float(df["Close"].iloc[entry_idx]) * 0.02  # fallback: 2% of price

    entry_price = float(df["Close"].iloc[entry_idx])
    entry_date  = str(df.index[entry_idx].date())
    stop_dist   = atr * atr_stop_mult
    target_dist = atr * atr_target_mult

    if is_short:
        stop_level   = entry_price + stop_dist
        target_level = entry_price - target_dist
    else:
        stop_level   = entry_price - stop_dist
        target_level = entry_price + target_dist

    watermark     = entry_price
    trailing_stop = stop_level

    # Walk forward
    for i in range(entry_idx + 1, len(df)):
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        d = str(df.index[i].date())

        if is_short:
            if l < watermark:
                watermark     = l
                trailing_stop = watermark + stop_dist
            if l <= target_level:
                ret = round((entry_price - target_level) / entry_price * 100, 2)
                return {"entry_date": entry_date, "exit_date": d, "entry_price": round(entry_price, 2),
                        "exit_price": round(target_level, 2), "return_pct": ret,
                        "exit_reason": "profit_target", "atr": round(atr, 2)}
            if h >= trailing_stop:
                ret = round((entry_price - trailing_stop) / entry_price * 100, 2)
                return {"entry_date": entry_date, "exit_date": d, "entry_price": round(entry_price, 2),
                        "exit_price": round(trailing_stop, 2), "return_pct": ret,
                        "exit_reason": "trailing_stop", "atr": round(atr, 2)}
        else:
            if h > watermark:
                watermark     = h
                trailing_stop = watermark - stop_dist
            if h >= target_level:
                ret = round((target_level - entry_price) / entry_price * 100, 2)
                return {"entry_date": entry_date, "exit_date": d, "entry_price": round(entry_price, 2),
                        "exit_price": round(target_level, 2), "return_pct": ret,
                        "exit_reason": "profit_target", "atr": round(atr, 2)}
            if l <= trailing_stop:
                ret = round((trailing_stop - entry_price) / entry_price * 100, 2)
                return {"entry_date": entry_date, "exit_date": d, "entry_price": round(entry_price, 2),
                        "exit_price": round(trailing_stop, 2), "return_pct": ret,
                        "exit_reason": "trailing_stop", "atr": round(atr, 2)}

    # Reached end of data — close at last price
    last_price = float(df["Close"].iloc[-1])
    last_date  = str(df.index[-1].date())
    ret = round(((entry_price - last_price) if is_short else (last_price - entry_price)) / entry_price * 100, 2)
    return {"entry_date": entry_date, "exit_date": last_date, "entry_price": round(entry_price, 2),
            "exit_price": round(last_price, 2), "return_pct": ret,
            "exit_reason": "data_end", "atr": round(atr, 2)}


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

    # Exit reason breakdown
    for t in trades:
        r = t.get("exit_reason", "unknown")
        result.exit_reason_counts[r] = result.exit_reason_counts.get(r, 0) + 1

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


def _fetch_ohlc(ticker: str, lookback_years: int) -> pd.DataFrame | None:
    """Fetch adjusted OHLC with High/Low columns for ATR computation."""
    df = yf.download(ticker, period=f"{lookback_years}y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df if len(df) > _ATR_PERIOD + 5 else None


@st.cache_data(ttl=3600)
def backtest_sma_crossover(
    ticker: str = "SPY",
    short_window: int = 50,
    long_window: int = 200,
    lookback_years: int = 5,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> BacktestResult:
    """Backtest SMA crossover (golden cross = buy) with ATR trailing stop/target."""
    df = _fetch_ohlc(ticker, lookback_years)
    if df is None or len(df) < long_window + _ATR_PERIOD:
        return BacktestResult(signal_name="SMA Crossover", ticker=ticker)

    close     = df["Close"]
    sma_short = close.rolling(short_window).mean()
    sma_long  = close.rolling(long_window).mean()
    cross     = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    signal_dates = cross[cross].index

    trades = []
    in_trade_until = None
    for entry_date in signal_dates:
        if in_trade_until and entry_date <= in_trade_until:
            continue  # skip overlapping signals
        idx = df.index.get_loc(entry_date)
        trade = _atr_exit_trade(df, idx, is_short=False,
                                atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
        trades.append(trade)
        in_trade_until = pd.Timestamp(trade["exit_date"])

    result = _compute_metrics(trades)
    result.signal_name = f"SMA {short_window}/{long_window} Crossover"
    result.ticker = ticker
    return result


@st.cache_data(ttl=3600)
def backtest_vix_spike(
    vix_threshold: float = 25,
    lookback_years: int = 5,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> BacktestResult:
    """Backtest buying SPY when VIX spikes above threshold — ATR trailing stop/target exit."""
    vix_raw = yf.download("^VIX", period=f"{lookback_years}y", interval="1d", progress=False, auto_adjust=True)
    spy_df  = _fetch_ohlc("SPY", lookback_years)

    if vix_raw is None or vix_raw.empty or spy_df is None:
        return BacktestResult(signal_name="VIX Spike", ticker="SPY")
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw = vix_raw.droplevel("Ticker", axis=1)

    vix_close   = vix_raw["Close"].dropna()
    above       = vix_close > vix_threshold
    cross_above = above & (~above.shift(1).fillna(False))
    signal_dates = cross_above[cross_above].index

    trades = []
    in_trade_until = None
    for entry_date in signal_dates:
        if in_trade_until and entry_date <= in_trade_until:
            continue
        valid = spy_df.index[spy_df.index >= entry_date]
        if len(valid) < 2:
            continue
        idx = spy_df.index.get_loc(valid[0])
        trade = _atr_exit_trade(spy_df, idx, is_short=False,
                                atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
        trades.append(trade)
        in_trade_until = pd.Timestamp(trade["exit_date"])

    result = _compute_metrics(trades)
    result.signal_name = f"VIX Spike (>{vix_threshold})"
    result.ticker = "SPY"
    return result


@st.cache_data(ttl=3600)
def backtest_regime_flip(
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> BacktestResult:
    """Backtest buying SPY on Risk-Off → Risk-On regime flips — ATR trailing stop/target exit."""
    history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json")
    if not os.path.exists(history_file):
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    with open(history_file) as f:
        history = json.load(f)
    if len(history) < 2:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    entries = sorted(history, key=lambda x: x.get("date", ""))
    flip_dates = [
        entries[i]["date"]
        for i in range(1, len(entries))
        if "Off" in entries[i - 1].get("regime", "") and "On" in entries[i].get("regime", "")
    ]
    if not flip_dates:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")

    min_date = min(flip_dates)
    spy_df = yf.download("SPY", start=min_date, progress=False, auto_adjust=True)
    if spy_df is None or spy_df.empty:
        return BacktestResult(signal_name="Regime Flip", ticker="SPY")
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df = spy_df.droplevel("Ticker", axis=1)
    spy_df = spy_df[["Open", "High", "Low", "Close"]].dropna()

    trades = []
    in_trade_until = None
    for date_str in flip_dates:
        try:
            target = pd.Timestamp(date_str)
            if in_trade_until and target <= in_trade_until:
                continue
            valid = spy_df.index[spy_df.index >= target]
            if len(valid) < 2:
                continue
            idx = spy_df.index.get_loc(valid[0])
            trade = _atr_exit_trade(spy_df, idx, is_short=False,
                                    atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
            trades.append(trade)
            in_trade_until = pd.Timestamp(trade["exit_date"])
        except Exception:
            continue

    result = _compute_metrics(trades)
    result.signal_name = "Regime Flip (Off→On)"
    result.ticker = "SPY"
    return result


@st.cache_data(ttl=3600)
def backtest_insider_cluster(
    ticker: str = "AAPL",
    min_buys: int = 3,
    cluster_days: int = 30,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> BacktestResult:
    """Backtest buying on insider buy clusters — ATR trailing stop/target exit."""
    from services.sec_client import get_insider_trades

    df_ins = get_insider_trades(ticker)
    if df_ins.empty:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    buys = df_ins[df_ins["type"] == "Purchase"].copy()
    if buys.empty:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
    buys = buys.dropna(subset=["date"]).sort_values("date")

    dates = buys["date"].tolist()
    cluster_dates = []
    i = 0
    while i < len(dates):
        window_end = dates[i] + pd.Timedelta(days=cluster_days)
        count = sum(1 for d in dates[i:] if d <= window_end)
        if count >= min_buys:
            cluster_dates.append(dates[i])
            i += count
        else:
            i += 1

    if not cluster_dates:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    price_df = _fetch_ohlc(ticker, lookback_years=2)
    if price_df is None:
        return BacktestResult(signal_name="Insider Cluster", ticker=ticker)

    trades = []
    in_trade_until = None
    for entry_date in cluster_dates:
        if in_trade_until and pd.Timestamp(entry_date) <= in_trade_until:
            continue
        valid = price_df.index[price_df.index >= pd.Timestamp(entry_date)]
        if len(valid) < 2:
            continue
        idx = price_df.index.get_loc(valid[0])
        trade = _atr_exit_trade(price_df, idx, is_short=False,
                                atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
        trades.append(trade)
        in_trade_until = pd.Timestamp(trade["exit_date"])

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
    train_months: int,
    test_months: int,
    lookback_years: int,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> dict:
    """Walk-forward validation for SMA crossover with ATR exits."""
    total_years = lookback_years + train_months / 12
    raw = yf.download(ticker, period=f"{int(total_years) + 1}y", interval="1d",
                      progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel("Ticker", axis=1)

    df = raw[["Open", "High", "Low", "Close"]].dropna()
    if len(df) < long_w + _ATR_PERIOD + 20:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}

    close = df["Close"]
    sma_s = close.rolling(short_w).mean()
    sma_l = close.rolling(long_w).mean()
    cross = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    signal_dates = cross[cross].index

    # Build all trades over full history (ATR exit)
    all_trades = []
    in_trade_until = None
    for entry_date in signal_dates:
        if in_trade_until and entry_date <= in_trade_until:
            continue
        idx = df.index.get_loc(entry_date)
        trade = _atr_exit_trade(df, idx, is_short=False,
                                atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
        trade["entry_date"] = entry_date  # keep as Timestamp for window slicing
        trade["exit_date"]  = pd.Timestamp(trade["exit_date"])
        all_trades.append(trade)
        in_trade_until = trade["exit_date"]

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
    train_months: int,
    test_months: int,
    lookback_years: int,
    atr_stop_mult: float = _DEFAULT_STOP,
    atr_target_mult: float = _DEFAULT_TARGET,
) -> dict:
    """Walk-forward validation for VIX spike strategy with ATR exits."""
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

    vix_c  = vix_raw["Close"].dropna()
    spy_df = spy_raw[["Open", "High", "Low", "Close"]].dropna()

    above  = vix_c > vix_threshold
    cross  = above & (~above.shift(1).fillna(False))
    signal_dates = cross[cross].index

    all_trades = []
    in_trade_until = None
    for entry_date in signal_dates:
        if in_trade_until and entry_date <= in_trade_until:
            continue
        valid = spy_df.index[spy_df.index >= entry_date]
        if len(valid) < 2:
            continue
        idx = spy_df.index.get_loc(valid[0])
        trade = _atr_exit_trade(spy_df, idx, is_short=False,
                                atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult)
        trade["entry_date"] = valid[0]
        all_trades.append(trade)
        in_trade_until = pd.Timestamp(trade["exit_date"])

    if not all_trades:
        return {"windows": [], "oos_equity": [], "confidence": "INSUFFICIENT DATA"}

    start    = vix_c.index[0]
    end      = vix_c.index[-1]
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
    agg     = _window_metrics([{"return_pct": t["return_pct"]} for t in all_oos])
    all_is  = [{"win_rate": w["train_win_rate"], "avg_return": w["train_avg_return"]} for w in windows if w["train_trades"] > 0]
    is_win  = round(float(np.mean([x["win_rate"] for x in all_is])), 1) if all_is else 0.0
    is_ret  = round(float(np.mean([x["avg_return"] for x in all_is])), 2) if all_is else 0.0

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
