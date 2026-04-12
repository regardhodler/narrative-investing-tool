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


# ══════════════════════════════════════════════════════════════════════════════
# CRASH STRESS TEST SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

CRASH_SCENARIOS = {
    "dotcom_2000": {
        "name": "Dot-com Bubble (2000–2002)",
        "warmup_start": "1999-01-01",
        "sim_start": "2000-01-01",
        "peak": "2000-03-24",
        "trough": "2002-10-09",
        "recovery_end": "2003-06-01",
        "context": "Tech mania, P/E >30, Fed hiking, Y2K liquidity drain",
    },
    "gfc_2008": {
        "name": "Global Financial Crisis (2007–2009)",
        "warmup_start": "2006-01-01",
        "sim_start": "2007-06-01",
        "peak": "2007-10-09",
        "trough": "2009-03-09",
        "recovery_end": "2009-09-01",
        "context": "Subprime, credit spreads blew out, yield curve inverted 2006",
    },
    "eu_debt_2011": {
        "name": "EU Debt Crisis (2011)",
        "warmup_start": "2010-01-01",
        "sim_start": "2011-03-01",
        "peak": "2011-04-29",
        "trough": "2011-10-03",
        "recovery_end": "2012-03-01",
        "context": "Greek/EU sovereign debt, S&P US downgrade",
    },
    "china_2015": {
        "name": "China Devaluation (2015–2016)",
        "warmup_start": "2014-01-01",
        "sim_start": "2015-05-01",
        "peak": "2015-07-20",
        "trough": "2016-02-11",
        "recovery_end": "2016-07-01",
        "context": "Yuan devaluation, EM contagion, oil crash, VIX spike",
    },
    "volmageddon_2018": {
        "name": "Volmageddon + Fed Selloff (2018)",
        "warmup_start": "2017-01-01",
        "sim_start": "2018-01-01",
        "peak": "2018-01-26",
        "trough": "2018-12-24",
        "recovery_end": "2019-05-01",
        "context": "Feb VIX explosion, Dec Fed overshoot, trade war",
    },
    "covid_2020": {
        "name": "COVID Crash (2020)",
        "warmup_start": "2019-01-01",
        "sim_start": "2020-01-01",
        "peak": "2020-02-19",
        "trough": "2020-03-23",
        "recovery_end": "2020-08-01",
        "context": "Pandemic, fastest bear market in history (33 days)",
    },
    "rate_shock_2022": {
        "name": "Rate Shock Bear Market (2022)",
        "warmup_start": "2021-01-01",
        "sim_start": "2021-11-01",
        "peak": "2022-01-03",
        "trough": "2022-10-12",
        "recovery_end": "2023-03-01",
        "context": "Inflation surge, aggressive Fed hiking, yield curve inversion",
    },
    "carry_unwind_2024": {
        "name": "Carry Trade Unwind (2024)",
        "warmup_start": "2023-06-01",
        "sim_start": "2024-06-01",
        "peak": "2024-07-16",
        "trough": "2024-08-05",
        "recovery_end": "2024-11-01",
        "context": "Yen carry unwind, AI rotation, VIX spike to 65",
    },
}

# FRED series used for historical regime reconstruction
_HIST_FRED_SERIES = {
    "yield_curve":    "T10Y2Y",
    "yield_curve_3m": "T10Y3M",
    "credit_hy":      "BAMLH0A0HYM2",
    "credit_ig":      "BAMLC0A0CM",
    "fci":            "NFCI",
    "icsa":           "ICSA",
    "fedfunds":       "FEDFUNDS",
    "real_yield":     "DFII10",
    "indpro":         "INDPRO",
    "umcsent":        "UMCSENT",
    "permit":         "PERMIT",
    "totbkcr":        "TOTBKCR",
}

# Weights mirror the live SIGNAL_WEIGHTS tiers from risk_regime.py
_HIST_SIGNAL_WEIGHTS = {
    "yield_curve":    2.0,
    "yield_curve_3m": 2.0,
    "credit_hy":      2.0,
    "credit_ig":      1.5,
    "fci":            2.0,
    "icsa":           1.5,
    "vix":            2.0,
    "spy_trend":      1.5,
    "real_yield":     2.0,
    "indpro":         1.0,
    "umcsent":        1.0,
    "permit":         0.5,
    "credit_impulse": 2.0,
}

# Which signals are inverted (higher = risk-off)
_HIST_INVERT = {"credit_hy", "credit_ig", "fci", "icsa", "vix"}


def _hist_zscore(series: pd.Series, date: pd.Timestamp, lookback: int = 252, invert: bool = False) -> float:
    """Compute z-score of a series at a specific historical date, mapped to [-1, +1]."""
    if series is None or series.empty:
        return 0.0
    sliced = series[series.index <= date].dropna()
    n = min(lookback, len(sliced))
    if n < 20:
        return 0.0
    window = sliced.iloc[-n:]
    std = float(window.std())
    if std < 1e-9:
        return 0.0
    z = (float(sliced.iloc[-1]) - float(window.mean())) / std
    if invert:
        z = -z
    return max(-1.0, min(1.0, z / 2.0))


def _hist_credit_impulse(series: pd.Series, date: pd.Timestamp) -> float:
    """Credit impulse = YoY acceleration of bank credit, z-scored."""
    if series is None or series.empty:
        return 0.0
    sliced = series[series.index <= date].dropna()
    if len(sliced) < 60:
        return 0.0
    yoy = sliced.pct_change(52).dropna()  # weekly data → 52 = 1 year
    if len(yoy) < 52:
        yoy = sliced.pct_change(12).dropna()  # monthly fallback
    if len(yoy) < 20:
        return 0.0
    std = float(yoy.iloc[-252:].std()) if len(yoy) >= 252 else float(yoy.std())
    if std < 1e-9:
        return 0.0
    z = (float(yoy.iloc[-1]) - float(yoy.iloc[-252:].mean() if len(yoy) >= 252 else yoy.mean())) / std
    return max(-1.0, min(1.0, z / 2.0))


@st.cache_data(ttl=86400, show_spinner=False)
def _load_all_historical_data() -> dict:
    """Load all FRED series + SPY + VIX for the full available history. Cached 24h."""
    from services.market_data import fetch_fred_series_safe

    data = {}
    for key, sid in _HIST_FRED_SERIES.items():
        data[key] = fetch_fred_series_safe(sid)

    # SPY and VIX from yfinance
    for ticker, label in [("SPY", "spy"), ("^VIX", "vix")]:
        try:
            raw = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw.droplevel("Ticker", axis=1)
                data[label] = raw["Close"].dropna()
                if label == "spy":
                    data["spy_ohlc"] = raw[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            pass

    return data


def reconstruct_regime_at_date(date: str, data: dict) -> dict:
    """Reconstruct REGARD regime score at a specific historical date.

    Args:
        date: ISO date string e.g. "2008-09-15"
        data: pre-loaded historical data dict from _load_all_historical_data()

    Returns dict with regime_score, quadrant, signal_details, spy_price, vix, etc.
    """
    dt = pd.Timestamp(date)
    signals = {}
    details = []

    # FRED-based z-scores
    for key in _HIST_FRED_SERIES:
        series = data.get(key)
        if series is None:
            continue
        invert = key in _HIST_INVERT
        z = _hist_zscore(series, dt, invert=invert)
        signals[key] = z
        # Get the raw value at date
        sliced = series[series.index <= dt].dropna()
        raw_val = float(sliced.iloc[-1]) if len(sliced) > 0 else None
        details.append({"name": key, "z_score": round(z, 3), "value": raw_val, "weight": _HIST_SIGNAL_WEIGHTS.get(key, 1.0)})

    # VIX z-score
    vix_series = data.get("vix")
    vix_val = None
    if vix_series is not None:
        vix_z = _hist_zscore(vix_series, dt, invert=True)
        signals["vix"] = vix_z
        sliced = vix_series[vix_series.index <= dt].dropna()
        vix_val = float(sliced.iloc[-1]) if len(sliced) > 0 else None
        # VIX percentile
        vix_pct = None
        if len(sliced) >= 252:
            window = sliced.iloc[-252:]
            vix_pct = round(float((window < vix_val).sum()) / len(window) * 100, 1)
        details.append({"name": "vix", "z_score": round(vix_z, 3), "value": vix_val, "weight": 2.0, "percentile": vix_pct})

    # SPY trend (50/200 SMA signal)
    spy_series = data.get("spy")
    spy_price = None
    spy_sma50 = None
    spy_sma200 = None
    spy_rsi = None
    if spy_series is not None:
        sliced = spy_series[spy_series.index <= dt].dropna()
        if len(sliced) >= 200:
            spy_price = float(sliced.iloc[-1])
            spy_sma50 = float(sliced.iloc[-50:].mean())
            spy_sma200 = float(sliced.iloc[-200:].mean())
            # Trend score: above both SMAs = +1, below both = -1, mixed = 0
            above_50 = spy_price > spy_sma50
            above_200 = spy_price > spy_sma200
            if above_50 and above_200:
                trend_z = 0.5
            elif not above_50 and not above_200:
                trend_z = -0.5
            elif above_200 and not above_50:
                trend_z = -0.15  # weakening
            else:
                trend_z = 0.15  # recovering
            # RSI(14)
            delta = sliced.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = 100 - (100 / (1 + rs))
            spy_rsi = float(rsi.iloc[-1]) if not rsi.empty else None
            # RSI modifier
            if spy_rsi is not None:
                if spy_rsi < 30:
                    trend_z -= 0.2  # oversold = more bearish regime signal
                elif spy_rsi > 70:
                    trend_z += 0.1  # overbought doesn't necessarily mean risk-off
            signals["spy_trend"] = max(-1.0, min(1.0, trend_z))
            details.append({"name": "spy_trend", "z_score": round(trend_z, 3), "value": spy_price, "weight": 1.5,
                            "sma50": round(spy_sma50, 2), "sma200": round(spy_sma200, 2), "rsi": round(spy_rsi, 1) if spy_rsi else None})

    # Credit impulse (bank credit acceleration)
    totbkcr = data.get("totbkcr")
    if totbkcr is not None:
        ci = _hist_credit_impulse(totbkcr, dt)
        signals["credit_impulse"] = ci
        details.append({"name": "credit_impulse", "z_score": round(ci, 3), "value": None, "weight": 2.0})

    # Weighted composite score
    if not signals:
        return {"date": date, "regime_score": 0.0, "quadrant": "Unknown", "signal_details": [], "spy_price": spy_price, "vix": vix_val}

    weighted_sum = 0.0
    weight_sum = 0.0
    for key, z in signals.items():
        w = _HIST_SIGNAL_WEIGHTS.get(key, 1.0)
        weighted_sum += z * w
        weight_sum += w

    regime_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    regime_score = max(-1.0, min(1.0, regime_score))

    # Map to 0-100 macro score
    macro_score = round((regime_score + 1) / 2 * 100, 1)

    # Quadrant classification (simplified — no inflation axis in historical mode)
    if regime_score > 0.15:
        quadrant = "Goldilocks" if regime_score > 0.4 else "Reflation"
        regime_label = "Risk-On"
    elif regime_score < -0.15:
        quadrant = "Stagflation" if regime_score < -0.4 else "Deflation"
        regime_label = "Risk-Off"
    else:
        quadrant = "Neutral"
        regime_label = "Neutral"

    return {
        "date": date,
        "regime_score": round(regime_score, 4),
        "regime_label": regime_label,
        "macro_score": macro_score,
        "quadrant": quadrant,
        "signal_details": details,
        "spy_price": spy_price,
        "spy_sma50": spy_sma50,
        "spy_sma200": spy_sma200,
        "spy_rsi": spy_rsi,
        "vix": vix_val,
        "vix_percentile": details[-2].get("percentile") if len(details) >= 2 else None,
    }


@st.cache_data(ttl=3600, show_spinner="Running crash simulation...")
def run_crash_simulation(crash_key: str) -> dict:
    """Run full historical crash simulation.

    Reconstructs REGARD's regime score for every trading day in the crash window,
    identifies the early warning date, and measures avoided drawdown + dip buy return.
    """
    scenario = CRASH_SCENARIOS.get(crash_key)
    if not scenario:
        return {"error": f"Unknown crash: {crash_key}"}

    data = _load_all_historical_data()
    spy_series = data.get("spy")
    if spy_series is None or spy_series.empty:
        return {"error": "Failed to load SPY data"}

    sim_start = pd.Timestamp(scenario["sim_start"])
    recovery_end = pd.Timestamp(scenario["recovery_end"])
    peak_date = pd.Timestamp(scenario["peak"])
    trough_date = pd.Timestamp(scenario["trough"])

    # Get trading days in the simulation window
    spy_window = spy_series[(spy_series.index >= sim_start) & (spy_series.index <= recovery_end)]
    if spy_window.empty:
        return {"error": "No SPY data for this period"}

    trading_days = [str(d)[:10] for d in spy_window.index]

    # Reconstruct regime at each trading day (sample every N days for speed)
    # Full daily for short crashes, weekly for long ones
    total_days = len(trading_days)
    step = 1 if total_days <= 200 else max(1, total_days // 200)
    sampled_days = trading_days[::step]
    if trading_days[-1] not in sampled_days:
        sampled_days.append(trading_days[-1])

    snapshots = []
    for d in sampled_days:
        snap = reconstruct_regime_at_date(d, data)
        snapshots.append(snap)

    # Find actual peak and trough SPY prices
    spy_at_peak = None
    spy_at_trough = None
    peak_slice = spy_series[spy_series.index <= peak_date].dropna()
    trough_slice = spy_series[spy_series.index <= trough_date].dropna()
    if len(peak_slice) > 0:
        spy_at_peak = float(peak_slice.iloc[-1])
    if len(trough_slice) > 0:
        spy_at_trough = float(trough_slice.iloc[-1])

    max_drawdown = round((spy_at_trough - spy_at_peak) / spy_at_peak * 100, 2) if spy_at_peak and spy_at_trough else None

    # Find early warning: first date with 3+ consecutive days of regime_score < -0.15
    warning_date = None
    warning_spy = None
    consec_risk_off = 0
    for snap in snapshots:
        if snap["regime_score"] < -0.15:
            consec_risk_off += 1
            if consec_risk_off >= 3 and warning_date is None:
                warning_date = snap["date"]
                warning_spy = snap["spy_price"]
        else:
            consec_risk_off = 0

    # Lead time and avoided drawdown
    warning_lead_days = None
    avoided_drawdown = None
    if warning_date and spy_at_trough and warning_spy:
        warning_lead_days = (trough_date - pd.Timestamp(warning_date)).days
        avoided_drawdown = round((spy_at_trough - warning_spy) / warning_spy * 100, 2)

    # Find dip buy signal: first date regime_score crosses back above 0 AFTER the trough
    dip_buy_date = None
    dip_buy_spy = None
    dip_buy_return_20d = None
    dip_buy_return_60d = None
    for snap in snapshots:
        snap_dt = pd.Timestamp(snap["date"])
        if snap_dt > trough_date and snap["regime_score"] > 0.0 and dip_buy_date is None:
            dip_buy_date = snap["date"]
            dip_buy_spy = snap["spy_price"]
            # Compute forward returns
            if dip_buy_spy:
                future_20 = spy_series[spy_series.index > snap_dt]
                if len(future_20) >= 20:
                    dip_buy_return_20d = round((float(future_20.iloc[19]) - dip_buy_spy) / dip_buy_spy * 100, 2)
                if len(future_20) >= 60:
                    dip_buy_return_60d = round((float(future_20.iloc[59]) - dip_buy_spy) / dip_buy_spy * 100, 2)
            break

    # Signal breakdown: which individual signal went negative first (before peak)
    signal_first_warnings = {}
    for snap in snapshots:
        snap_dt = pd.Timestamp(snap["date"])
        if snap_dt > peak_date:
            break  # only care about pre-peak warnings
        for detail in snap.get("signal_details", []):
            name = detail["name"]
            if name in signal_first_warnings:
                continue
            if detail["z_score"] < -0.3:  # signal is clearly negative
                signal_first_warnings[name] = {
                    "date": snap["date"],
                    "z_score": detail["z_score"],
                    "value": detail.get("value"),
                    "lead_days": (peak_date - snap_dt).days,
                }

    # Also check post-peak for signals that only fired during the crash
    for snap in snapshots:
        snap_dt = pd.Timestamp(snap["date"])
        if snap_dt <= peak_date:
            continue
        for detail in snap.get("signal_details", []):
            name = detail["name"]
            if name in signal_first_warnings:
                continue
            if detail["z_score"] < -0.3:
                signal_first_warnings[name] = {
                    "date": snap["date"],
                    "z_score": detail["z_score"],
                    "value": detail.get("value"),
                    "lead_days": -(snap_dt - peak_date).days,  # negative = after peak
                }

    return {
        "crash_key": crash_key,
        "crash_name": scenario["name"],
        "context": scenario["context"],
        "peak_date": scenario["peak"],
        "trough_date": scenario["trough"],
        "spy_at_peak": spy_at_peak,
        "spy_at_trough": spy_at_trough,
        "max_drawdown": max_drawdown,
        "warning_date": warning_date,
        "warning_spy": warning_spy,
        "warning_lead_days": warning_lead_days,
        "avoided_drawdown": avoided_drawdown,
        "dip_buy_date": dip_buy_date,
        "dip_buy_spy": dip_buy_spy,
        "dip_buy_return_20d": dip_buy_return_20d,
        "dip_buy_return_60d": dip_buy_return_60d,
        "signal_first_warnings": signal_first_warnings,
        "snapshots": snapshots,
        "total_sim_days": len(sampled_days),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def _build_hmm_historical_inference() -> dict | None:
    """Build HMM feature matrix and run inference over full history.

    Returns dict with:
        dates:       list of date strings
        states:      list of state_label per date
        state_idxs:  list of state_idx per date
        probs:       list of probability vectors per date
        labels:      state label list from brain
        transmat:    transition matrix
        ll_per_obs:  list of per-date log-likelihood
        ll_baseline_mean/std: from brain
        n_states:    int
    Or None if HMM brain not available.
    """
    try:
        from services.hmm_regime import load_hmm_brain, _build_feature_matrix
        from hmmlearn.hmm import GaussianHMM

        brain = load_hmm_brain()
        if brain is None:
            return None

        n = brain.n_states
        model = GaussianHMM(n_components=n, covariance_type="full")
        model.n_features = len(brain.feature_names)
        model.startprob_ = np.ones(n) / n
        model.transmat_ = np.array(brain.transmat)
        model.means_ = np.array(brain.means)
        model.covars_ = np.array(brain.covars)

        df = _build_feature_matrix(lookback_years=30)
        for col in brain.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna()
        X = df[brain.feature_names].values.astype(np.float64)

        posteriors = model.predict_proba(X)
        state_idxs = np.argmax(posteriors, axis=1).tolist()

        # Compute distance-based soft probabilities per state
        # Raw HMM posteriors are always 1.0/0.0 (states too well-separated in 10D space)
        # Instead, use Mahalanobis distance → inverse-distance weighting for meaningful
        # confidence/entropy values
        softened_list = []
        for i in range(len(X)):
            dists = []
            for k in range(n):
                diff = X[i] - model.means_[k]
                try:
                    inv_cov = np.linalg.inv(model.covars_[k])
                    maha = float(np.sqrt(diff @ inv_cov @ diff))
                except Exception:
                    maha = 1e6
                dists.append(maha)
            dists = np.array(dists)
            # Inverse-distance weighting with scaling
            inv_dists = 1.0 / (dists + 1e-6)
            probs_soft = inv_dists / inv_dists.sum()
            softened_list.append(probs_soft.tolist())
        softened = np.array(softened_list)

        # Per-date windowed log-likelihood (60-day rolling window)
        _LL_WINDOW = 60
        ll_per_date = []
        for i in range(len(X)):
            start = max(0, i - _LL_WINDOW + 1)
            window = X[start:i + 1]
            try:
                ll_val = float(model.score(window)) / len(window)
            except Exception:
                ll_val = 0.0
            ll_per_date.append(ll_val)

        dates = [str(d)[:10] for d in df.index]

        return {
            "dates": dates,
            "states": [brain.state_labels[i] for i in state_idxs],
            "state_idxs": state_idxs,
            "probs": [softened[i].tolist() for i in range(len(softened))],
            "labels": brain.state_labels,
            "transmat": brain.transmat,
            "ll_per_date": ll_per_date,
            "ll_baseline_mean": brain.ll_baseline_mean,
            "ll_baseline_std": brain.ll_baseline_std,
            "n_states": n,
        }
    except Exception:
        return None


def reconstruct_hmm_at_date(date: str, hmm_data: dict | None) -> dict | None:
    """Extract HMM state at a specific date from pre-computed inference.

    Returns dict with state_label, confidence, entropy, persistence,
    forecast_1m, forecast_2m, transition_risk_1m/2m, ll_zscore, probs.
    """
    if hmm_data is None:
        return None

    dates = hmm_data["dates"]
    # Find closest date <= target
    target = date
    idx = None
    for i, d in enumerate(dates):
        if d <= target:
            idx = i
        elif d > target:
            break
    if idx is None:
        return None

    probs = hmm_data["probs"][idx]
    state_idx = hmm_data["state_idxs"][idx]
    state_label = hmm_data["states"][idx]
    confidence = round(max(probs), 3)

    # Persistence
    persistence = 1
    for j in range(idx - 1, -1, -1):
        if hmm_data["state_idxs"][j] == state_idx:
            persistence += 1
        else:
            break

    # Entropy
    from scipy.stats import entropy as _shannon_entropy
    raw_entropy = float(_shannon_entropy(probs))
    max_entropy = float(np.log(hmm_data["n_states"]))
    normalized_entropy = round(raw_entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    # LL z-score (per-date windowed)
    ll_per_date = hmm_data.get("ll_per_date")
    if ll_per_date and idx < len(ll_per_date):
        _ll_val = ll_per_date[idx]
    else:
        _ll_val = hmm_data.get("ll_per_obs", 0)  # fallback to global
    ll_zscore = round(
        (_ll_val - hmm_data["ll_baseline_mean"]) /
        max(hmm_data["ll_baseline_std"], 1e-6), 3
    )

    # Transition forecasts
    transmat = np.array(hmm_data["transmat"])
    prob_vec = np.array(probs)
    forecast_1m = (prob_vec @ np.linalg.matrix_power(transmat, 21)).tolist()
    forecast_2m = (prob_vec @ np.linalg.matrix_power(transmat, 42)).tolist()
    transition_risk_1m = round(1.0 - forecast_1m[state_idx], 4)
    transition_risk_2m = round(1.0 - forecast_2m[state_idx], 4)

    return {
        "state_label": state_label,
        "state_idx": state_idx,
        "confidence": confidence,
        "persistence": persistence,
        "entropy": normalized_entropy,
        "ll_zscore": ll_zscore,
        "probs": [round(p, 4) for p in probs],
        "labels": hmm_data["labels"],
        "forecast_1m": [round(p, 4) for p in forecast_1m],
        "forecast_2m": [round(p, 4) for p in forecast_2m],
        "transition_risk_1m": transition_risk_1m,
        "transition_risk_2m": transition_risk_2m,
    }


def build_qir_snapshot(date: str, data: dict, hmm_data: dict | None, prev_regime_score: float | None = None) -> dict:
    """Build a full simulated QIR snapshot for a specific date.

    Combines regime reconstruction + HMM + entry signal into a dict
    matching the QIR dashboard's display fields.

    prev_regime_score: regime score from ~5-10 days prior, used to compute velocity.
    """
    regime = reconstruct_regime_at_date(date, data)
    hmm = reconstruct_hmm_at_date(date, hmm_data)

    # Domain scores (historical mode — only macro + tech available)
    macro_score = regime.get("macro_score", 50)
    # Technical = SPY trend mapped to 0-100
    spy_trend_z = 0.0
    for d in regime.get("signal_details", []):
        if d["name"] == "spy_trend":
            spy_trend_z = d["z_score"]
    tech_score = round((spy_trend_z + 1) / 2 * 100, 1)

    # Entry signal (use macro + tech, options/sentiment default to 50)
    from modules.quick_run import _classify_entry_recommendation
    try:
        entry = _classify_entry_recommendation(
            leading_score=int(macro_score),  # best proxy in historical mode
            macro_score=int(macro_score),
            tactical_score=int(tech_score),
            options_score=50,  # unavailable historically
            divergence_label="Aligned",
            divergence_pts=0,
        )
    except Exception:
        entry = {"verdict": "N/A"}

    # Lean direction
    if macro_score >= 55 and tech_score >= 50:
        lean = "BULLISH"
        lean_pct = round((macro_score + tech_score) / 2, 0)
    elif macro_score <= 45 and tech_score <= 50:
        lean = "BEARISH"
        lean_pct = round(100 - (macro_score + tech_score) / 2, 0)
    else:
        lean = "BEARISH" if macro_score < 50 else "BULLISH"
        lean_pct = 52

    # Conviction score (simplified)
    conviction = round(abs(macro_score - 50) * 2, 0)
    conviction = min(100, max(0, conviction))

    # Kelly multiplier from HMM
    _KELLY_MULT = {"Bull": 1.10, "Neutral": 1.00, "Early Stress": 0.90,
                   "Stress": 0.85, "Late Cycle": 0.75, "Crisis": 0.60}
    hmm_mult = _KELLY_MULT.get(hmm["state_label"], 1.0) if hmm else 1.0

    # Regime velocity: change in regime score vs prior snapshot
    regime_velocity = None
    if prev_regime_score is not None:
        regime_velocity = round(regime["regime_score"] - prev_regime_score, 4)

    return {
        "date": date,
        "regime": regime,
        "hmm": hmm,
        "entry_signal": entry.get("verdict", "N/A"),
        "lean": lean,
        "lean_pct": lean_pct,
        "macro_score": macro_score,
        "tech_score": tech_score,
        "opts_score": "N/A",
        "sent_score": "N/A",
        "event_score": "N/A",
        "conviction": conviction,
        "regime_velocity": regime_velocity,
        "hmm_kelly_mult": hmm_mult,
        "spy_price": regime.get("spy_price"),
        "vix": regime.get("vix"),
    }


@st.cache_data(ttl=3600, show_spinner="Running all crash simulations...")
def run_all_crash_simulations() -> list[dict]:
    """Run all crash simulations and return comparison data."""
    results = []
    for key in CRASH_SCENARIOS:
        result = run_crash_simulation(key)
        results.append(result)
    return results
