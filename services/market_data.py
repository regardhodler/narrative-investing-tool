"""
Shared market data fetching layer.

Uses yfinance as primary provider with caching.
Other modules should import from here instead of calling yfinance directly.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
import os
from io import StringIO
from dataclasses import dataclass, field


class _FredFetchError(Exception):
    """Raised when a FRED series fetch fails, preventing st.cache_data from caching None."""
    pass


class _BatchFetchError(Exception):
    """Raised when all tickers fail to fetch, preventing st.cache_data from caching empty snapshots."""
    pass


class _OptionsFetchError(Exception):
    """Raised when options chain fetch fails, preventing st.cache_data from caching None."""
    pass


@dataclass
class AssetSnapshot:
    """Standardized snapshot for a single asset."""
    ticker: str
    label: str
    latest_price: float | None = None
    pct_change_1d: float | None = None
    pct_change_5d: float | None = None
    pct_change_30d: float | None = None
    pct_change_ytd: float | None = None
    series: pd.Series | None = None  # close price series for sparklines
    stale: bool = False  # True if data may be outdated (weekend/holiday)


def _fetch_single(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV via yfinance for a single ticker."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            # yfinance returns MultiIndex columns for single ticker downloads
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def get_yf_info_safe(ticker: str) -> dict:
    """Cached yfinance .info fetch. Single source of truth for ticker metadata.

    Avoids redundant yfinance calls when the same ticker is referenced by
    multiple modules in the same Streamlit session. Returns {} on failure.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=86400)
def fetch_earnings_date(ticker: str) -> dict | None:
    """Return next earnings date and days away for a ticker. None if unavailable.

    Returns: {"date": "Apr 25", "days_away": 32, "full_date": "2026-04-25"}
    """
    try:
        from datetime import date as _date
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        # yfinance may return a DataFrame or a dict depending on version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].values
                if len(raw) == 0:
                    return None
                ts = raw[0]
            else:
                return None
        elif isinstance(cal, dict):
            raw_list = cal.get("Earnings Date") or []
            if not raw_list:
                return None
            ts = raw_list[0]
        else:
            return None
        # Convert timestamp/datetime to date
        if hasattr(ts, "date"):
            earn_date = ts.date()
        else:
            earn_date = pd.Timestamp(ts).date()
        today = _date.today()
        days_away = (earn_date - today).days
        if days_away < -7:  # already passed more than a week ago — not useful
            return None
        return {
            "date": earn_date.strftime("%b %d"),
            "full_date": earn_date.isoformat(),
            "days_away": days_away,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_earnings_intelligence(ticker: str) -> dict:
    """
    Fetch comprehensive earnings intelligence for a ticker.

    Returns a dict with:
      next_earnings:   {date, days_away, full_date} or None
      eps_history:     list of {period, estimate, actual, surprise_pct, beat} — last 4
      analyst:         {buy, hold, sell, strong_buy, strong_sell, mean_target, current_price, upside_pct}
      expected_move:   {pct, dollar, expiry, dte} or None — from nearest-expiry ATM IV
    """
    result = {
        "next_earnings": None,
        "eps_history":   [],
        "analyst":       {},
        "expected_move": None,
    }
    tk = yf.Ticker(ticker)

    # ── Next earnings date ────────────────────────────────────────────────────
    try:
        from datetime import date as _date
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            ts = cal.loc["Earnings Date"].values[0]
        elif isinstance(cal, dict):
            raw = (cal.get("Earnings Date") or [])
            ts = raw[0] if raw else None
        else:
            ts = None
        if ts is not None:
            earn_date = pd.Timestamp(ts).date()
            days = (earn_date - _date.today()).days
            if days >= -7:
                result["next_earnings"] = {
                    "date": earn_date.strftime("%b %d, %Y"),
                    "days_away": days,
                    "full_date": earn_date.isoformat(),
                }
    except Exception:
        pass

    # ── EPS history (last 4 quarters) ─────────────────────────────────────────
    try:
        earn_df = tk.earnings_dates
        if earn_df is not None and not earn_df.empty:
            # earnings_dates has "EPS Estimate" and "Reported EPS" columns
            earn_df = earn_df.dropna(subset=["Reported EPS"]).head(4)
            for idx, row in earn_df.iterrows():
                est    = row.get("EPS Estimate")
                actual = row.get("Reported EPS")
                if pd.isna(actual):
                    continue
                surprise = None
                beat = None
                if est is not None and not pd.isna(est) and est != 0:
                    surprise = round((actual - est) / abs(est) * 100, 1)
                    beat = actual >= est
                elif est is not None and not pd.isna(est) and est == 0:
                    beat = actual > 0
                result["eps_history"].append({
                    "period":       pd.Timestamp(idx).strftime("%b %Y") if hasattr(idx, "strftime") else str(idx),
                    "estimate":     round(float(est), 2) if est is not None and not pd.isna(est) else None,
                    "actual":       round(float(actual), 2),
                    "surprise_pct": surprise,
                    "beat":         beat,
                })
    except Exception:
        pass

    # ── Analyst consensus ─────────────────────────────────────────────────────
    try:
        info = tk.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        mean_target   = info.get("targetMeanPrice")
        upside = None
        if mean_target and current_price and current_price > 0:
            upside = round((mean_target / current_price - 1) * 100, 1)
        rec = tk.recommendations_summary
        buy = hold = sell = strong_buy = strong_sell = 0
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            buy          = int(latest.get("buy", 0) or 0)
            hold         = int(latest.get("hold", 0) or 0)
            sell         = int(latest.get("sell", 0) or 0)
            strong_buy   = int(latest.get("strongBuy", 0) or 0)
            strong_sell  = int(latest.get("strongSell", 0) or 0)
        result["analyst"] = {
            "strong_buy":   strong_buy,
            "buy":          buy,
            "hold":         hold,
            "sell":         sell,
            "strong_sell":  strong_sell,
            "total":        strong_buy + buy + hold + sell + strong_sell,
            "mean_target":  round(float(mean_target), 2) if mean_target else None,
            "current_price": round(float(current_price), 2) if current_price else None,
            "upside_pct":   upside,
        }
    except Exception:
        pass

    # ── Expected move from nearest-expiry ATM IV ──────────────────────────────
    try:
        from datetime import date as _date2
        expirations = tk.options
        if expirations:
            exp = expirations[0]
            chain = tk.option_chain(exp)
            calls = chain.calls
            # Get current price
            info2 = tk.fast_info
            spot = float(info2.get("lastPrice", 0) or 0) if hasattr(info2, "get") else 0
            if spot == 0 and not calls.empty and "lastPrice" in calls.columns:
                spot = float(calls["strike"].median())
            if spot > 0 and not calls.empty:
                # ATM = strike closest to spot
                calls = calls[calls["impliedVolatility"] > 0].copy()
                if not calls.empty:
                    calls["dist"] = abs(calls["strike"] - spot)
                    atm = calls.nsmallest(3, "dist")
                    iv = float(atm["impliedVolatility"].mean())
                    exp_date = _date2.fromisoformat(exp)
                    dte = max(1, (exp_date - _date2.today()).days)
                    em_pct = round(iv * (dte / 365) ** 0.5 * 100, 1)
                    em_dollar = round(spot * em_pct / 100, 2)
                    result["expected_move"] = {
                        "pct":    em_pct,
                        "dollar": em_dollar,
                        "expiry": exp,
                        "dte":    dte,
                        "iv":     round(iv * 100, 1),
                    }
    except Exception:
        pass

    return result


@st.cache_data(ttl=3600)
def fetch_ohlcv_single(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV DataFrame for a single ticker."""
    return _fetch_single(ticker, period, interval)


@st.cache_data(ttl=3600)
def fetch_correlation_matrix(tickers: tuple[str, ...], period: str = "6mo") -> pd.DataFrame | None:
    """Compute pairwise Pearson correlation of daily returns for a set of tickers.

    Args:
        tickers: tuple of ticker symbols (tuple so it's hashable for cache)
        period: yfinance period string, default 6 months

    Returns:
        DataFrame of pairwise correlations (tickers × tickers), or None on failure.
    """
    if not tickers or len(tickers) < 2:
        return None
    try:
        raw = yf.download(
            list(tickers), period=period, interval="1d",
            progress=False, auto_adjust=True, threads=True
        )
        if raw.empty:
            return None
        # Handle both single and multi-ticker yfinance output
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
        closes = closes.dropna(how="all")
        returns = closes.pct_change().dropna(how="all")
        corr = returns.corr()
        # Rename columns/index to uppercase tickers
        corr.columns = [str(c).upper() for c in corr.columns]
        corr.index = [str(i).upper() for i in corr.index]
        return corr
    except Exception:
        return None


@st.cache_data(ttl=14400)
def fetch_batch(tickers: dict[str, str], period: str = "1y", interval: str = "1d") -> dict[str, AssetSnapshot]:
    """
    Fetch multiple tickers concurrently.

    Args:
        tickers: {ticker_symbol: human_label} e.g. {"SPY": "S&P 500"}
        period: yfinance period string
        interval: yfinance interval string

    Returns:
        {ticker: AssetSnapshot}
    """
    results: dict[str, AssetSnapshot] = {}
    ticker_list = list(tickers.keys())

    try:
        raw = yf.download(ticker_list, period=period, interval=interval,
                          progress=False, auto_adjust=True, threads=True)
    except Exception:
        raw = pd.DataFrame()

    for ticker, label in tickers.items():
        snap = AssetSnapshot(ticker=ticker, label=label)

        try:
            if raw is not None and not raw.empty:
                # Single ticker: flat columns; multiple tickers: MultiIndex (Price, Ticker)
                if len(ticker_list) == 1:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw["Close"].iloc[:, 0].dropna()
                    else:
                        close = raw["Close"].dropna()
                else:
                    close = raw["Close"][ticker].dropna()

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                if len(close) > 0:
                    snap.latest_price = float(close.iloc[-1])
                    snap.series = close.copy()

                    if len(close) >= 2:
                        snap.pct_change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
                    if len(close) >= 6:
                        snap.pct_change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
                    if len(close) >= 22:
                        snap.pct_change_30d = float((close.iloc[-1] / close.iloc[-22] - 1) * 100)

                    year_start = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
                    ytd_data = close[close.index >= year_start]
                    if len(ytd_data) >= 2:
                        snap.pct_change_ytd = float((close.iloc[-1] / ytd_data.iloc[0] - 1) * 100)

                    last_date = close.index[-1]
                    days_old = (pd.Timestamp.now(tz=last_date.tzinfo if hasattr(last_date, 'tzinfo') and last_date.tzinfo else None) - last_date).days
                    snap.stale = days_old > 3
        except Exception:
            pass

        results[ticker] = snap

    if not any(s.latest_price is not None for s in results.values()):
        raise _BatchFetchError("All tickers failed to fetch")
    return results


def fetch_batch_safe(tickers: dict[str, str], period: str = "1y", interval: str = "1d") -> dict[str, AssetSnapshot]:
    """Wrapper around fetch_batch that returns empty snapshots on total failure instead of raising.

    Use this at call sites so that total failures aren't cached by st.cache_data.
    """
    try:
        return fetch_batch(tickers, period, interval)
    except _BatchFetchError:
        return {t: AssetSnapshot(ticker=t, label=label) for t, label in tickers.items()}


def zscore(series: pd.Series, lookback: int = 252) -> float | None:
    """Z-score of latest value vs trailing lookback window."""
    if series is None or len(series) < max(lookback, 20):
        return None
    window = series.iloc[-lookback:]
    std = window.std()
    if std == 0:
        return 0.0
    return float((series.iloc[-1] - window.mean()) / std)


def ratio_latest(snaps: dict[str, AssetSnapshot], t1: str, t2: str) -> float | None:
    """Ratio of latest prices for two tickers."""
    s1, s2 = snaps.get(t1), snaps.get(t2)
    if s1 and s2 and s1.latest_price and s2.latest_price and s2.latest_price != 0:
        return s1.latest_price / s2.latest_price
    return None


@st.cache_data(ttl=3600)
def fetch_truflation() -> dict | None:
    """Fetch Truflation current inflation from their public API."""
    try:
        r = requests.get(
            "https://api.truflation.com/inflation",
            timeout=8,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=43200)
def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a single FRED series as a pandas Series indexed by date."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fred_cache")
    cache_file = os.path.join(cache_dir, f"{series_id}.csv")

    def _parse_csv_text(csv_text: str) -> pd.Series | None:
        df = pd.read_csv(StringIO(csv_text))
        # FRED CSV uses "observation_date" (newer) or "DATE" (legacy) as date column
        date_col = None
        for candidate in ("DATE", "observation_date"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None or series_id not in df.columns:
            return None

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=[date_col, series_id])
        if df.empty:
            return None

        series = pd.Series(df[series_id].values, index=df[date_col], name=series_id)
        return series.sort_index()

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    for timeout in (5, 10):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "NarrativeInvestingTool/1.0"})
            resp.raise_for_status()
            parsed = _parse_csv_text(resp.text)
            if parsed is not None:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                return parsed
        except Exception:
            continue

    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_text = f.read()
            result = _parse_csv_text(cached_text)
            if result is not None:
                return result
    except Exception:
        pass

    raise _FredFetchError(f"Failed to fetch FRED series {series_id}")


def fetch_fred_series_safe(series_id: str) -> pd.Series | None:
    """Wrapper around fetch_fred_series that returns None on failure instead of raising.

    Use this at call sites so that transient failures aren't cached by st.cache_data.
    """
    try:
        return fetch_fred_series(series_id)
    except _FredFetchError:
        return None


def compute_data_quality_score(snaps: dict, fred_data: dict) -> dict:
    """Score data freshness 0-100. Pure function — safe in background threads.

    Market signals (50 pts): SPY, ^VIX, TLT, GLD, DXY — 10 pts each.
    FRED signals (50 pts): DGS10, BAMLH0A0HYM2, T10Y2Y, SAHMREALTIME, PCEPILFE — 10 pts each.
    Returns {"score", "label", "stale_market", "stale_fred", "total_signals", "fresh_signals"}.
    """
    _market_keys = ["SPY", "^VIX", "TLT", "GLD", "DXY"]
    _fred_keys   = ["dgs10", "credit_spread", "yield_curve", "sahm", "core_pce"]

    stale_market, stale_fred = [], []
    market_pts = 0
    for tk in _market_keys:
        snap = snaps.get(tk)
        if snap and snap.latest_price is not None and not snap.stale:
            market_pts += 10
        else:
            stale_market.append(tk)

    fred_pts = 0
    _cutoff = pd.Timestamp.now(tz=None).normalize() - pd.offsets.BDay(5)
    for fk in _fred_keys:
        s = fred_data.get(fk)
        if s is not None and len(s) > 0:
            _last = s.index[-1]
            if hasattr(_last, "tz_localize"):
                _last = _last.tz_localize(None) if _last.tzinfo is not None else _last
            if _last >= _cutoff:
                fred_pts += 10
            else:
                stale_fred.append(fk)
        else:
            stale_fred.append(fk)

    score = market_pts + fred_pts
    if score >= 80:
        label = "High Confidence"
    elif score >= 60:
        label = "Moderate — some signals stale"
    else:
        label = "Low — AI reasoning may be unreliable"

    fresh = (len(_market_keys) - len(stale_market)) + (len(_fred_keys) - len(stale_fred))
    return {
        "score": score,
        "label": label,
        "stale_market": stale_market,
        "stale_fred": stale_fred,
        "total_signals": len(_market_keys) + len(_fred_keys),
        "fresh_signals": fresh,
    }


_DEFAULT_FRED_IDS = [
    "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
    "PCEPILFE", "PNFI", "THREEFYTP10", "INDPRO", "NFCI", "DGS10",
    "ICSA", "USSLIND", "UMCSENT", "PERMIT", "FEDFUNDS", "DFII10", "MANEMP", "TOTBKCR", "DGS2",
]


def warm_fred_cache(series_ids: list[str] | None = None):
    """Pre-fetch FRED series in staggered mini-batches to warm disk cache.

    Uses batches of 4 with 0.5s delay between batches to avoid FRED rate limits.
    Call at app startup so the dashboard build doesn't serialize FRED fetches.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    if series_ids is None:
        series_ids = _DEFAULT_FRED_IDS
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fred_cache")
    os.makedirs(cache_dir, exist_ok=True)
    batch_size = 4
    for i in range(0, len(series_ids), batch_size):
        batch = series_ids[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            list(executor.map(fetch_fred_series_safe, batch))
        if i + batch_size < len(series_ids):
            time.sleep(0.5)


@st.cache_data(ttl=14400)
def fetch_options_chain_snapshot(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """
    Fetch options chain snapshot and build strike-level aggregates.

    Returns:
        {
            "ticker": str,
            "price": float,
            "asof": str,
            "strikes": [float],
            "call_oi": [float],
            "put_oi": [float],
            "net_gamma_proxy": [float],
            "expiries": [str],
        }
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise _OptionsFetchError(f"No price history for {ticker}")

        price = float(hist["Close"].iloc[-1])
        expiries = list(tk.options or [])
        if not expiries:
            raise _OptionsFetchError(f"No options expiries for {ticker}")

        selected = expiries[:max(1, min(max_expiries, len(expiries)))]
        strike_map: dict[float, dict[str, float]] = {}

        for exp in selected:
            chain = tk.option_chain(exp)
            for side, sign in ((chain.calls, 1.0), (chain.puts, -1.0)):
                if side is None or side.empty:
                    continue
                subset = side[["strike", "openInterest", "impliedVolatility"]].copy()
                subset["openInterest"] = pd.to_numeric(subset["openInterest"], errors="coerce").fillna(0)
                subset["impliedVolatility"] = pd.to_numeric(subset["impliedVolatility"], errors="coerce").fillna(0.2)

                for _, row in subset.iterrows():
                    strike = float(row["strike"])
                    oi = float(row["openInterest"])
                    iv = float(row["impliedVolatility"])

                    if strike not in strike_map:
                        strike_map[strike] = {"call_oi": 0.0, "put_oi": 0.0, "net_gamma_proxy": 0.0}

                    if sign > 0:
                        strike_map[strike]["call_oi"] += oi
                    else:
                        strike_map[strike]["put_oi"] += oi

                    distance = abs(strike - price) / max(price, 1e-6)
                    weight = np.exp(-((distance / 0.1) ** 2))
                    gamma_proxy = sign * oi * iv * weight
                    strike_map[strike]["net_gamma_proxy"] += gamma_proxy

        if not strike_map:
            raise _OptionsFetchError(f"No strike data for {ticker}")

        strikes = sorted(strike_map.keys())
        return {
            "ticker": ticker,
            "price": price,
            "asof": str(pd.Timestamp.utcnow()),
            "strikes": strikes,
            "call_oi": [float(strike_map[s]["call_oi"]) for s in strikes],
            "put_oi": [float(strike_map[s]["put_oi"]) for s in strikes],
            "net_gamma_proxy": [float(strike_map[s]["net_gamma_proxy"]) for s in strikes],
            "expiries": selected,
        }
    except _OptionsFetchError:
        raise
    except Exception as e:
        raise _OptionsFetchError(f"Options fetch failed for {ticker}: {e}")


def fetch_options_chain_snapshot_safe(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """Wrapper around fetch_options_chain_snapshot that returns None on failure instead of raising.

    Use this at call sites so that transient failures aren't cached by st.cache_data.
    """
    try:
        return fetch_options_chain_snapshot(ticker, max_expiries)
    except _OptionsFetchError:
        return None
