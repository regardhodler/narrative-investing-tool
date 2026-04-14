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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wyckoff_spy(period: str = "1y") -> dict | None:
    """Cached Wyckoff analysis for SPY. Returns compact dict of key signals.

    Extracted fields:
      phase         - "Accumulation" | "Distribution" | "Markup" | "Markdown"
      sub_phase     - "A"-"E" or ""
      confidence    - 0-100
      support       - float price level
      resistance    - float price level
      cause_target  - float price target or None
    """
    try:
        from services.wyckoff_engine import analyze_wyckoff
        df = _fetch_single("SPY", period=period, interval="1d")
        if df is None or len(df) < 60:
            return None
        result = analyze_wyckoff(df["Close"], df["High"], df["Low"], df["Volume"], interval="1d")
        if result is None:
            return None
        cp = result.current_phase
        return {
            "phase":        cp.phase,
            "sub_phase":    cp.sub_phase or "",
            "confidence":   cp.confidence,
            "support":      cp.key_levels.get("support"),
            "resistance":   cp.key_levels.get("resistance"),
            "cause_target": cp.cause_target,
            "spy_last":     float(df["Close"].iloc[-1]),
        }
    except Exception:
        return None


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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gex_profile(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """Compute Gamma Exposure (GEX) profile and dealer positioning for a ticker.

    GEX Model: Assumes dealers are short options sold to market participants.
      - Call OI: dealers short calls → long gamma → positive GEX
      - Put OI:  dealers short puts → short gamma → negative GEX
    
    GEX notional per strike = OI × IV × spot × contract_multiplier × gaussian_weight
    (True Black-Scholes gamma requires T, rate — we use IV × gaussian as proxy)

    Returns dict with:
      strikes, call_gex, put_gex, net_gex (arrays)
      total_gex (float, $ millions) 
      gamma_flip (float, strike where cumulative net_gex crosses zero)
      call_wall (float, strike with max call OI)
      put_wall (float, strike with max put OI)
      dealer_net_delta (float, rough directional bias -1 to +1)
      zone ("Positive Gamma Zone" or "Negative Gamma Zone")
      zone_detail (one-line description of what the zone means)
      spot (float)
      asof (str)
    """
    import numpy as np
    import pandas as pd

    try:
        ticker_obj = yf.Ticker(ticker)
        exps = ticker_obj.options
        if not exps:
            return None

        price_info = ticker_obj.fast_info
        spot = getattr(price_info, "last_price", None) or getattr(price_info, "previous_close", None)
        if not spot or spot <= 0:
            return None

        exps_to_use = list(exps[:max_expiries])

        call_oi_map: dict[float, float] = {}
        put_oi_map: dict[float, float] = {}
        call_iv_map: dict[float, float] = {}
        put_iv_map: dict[float, float] = {}

        for exp in exps_to_use:
            try:
                chain = ticker_obj.option_chain(exp)
                calls = chain.calls[["strike", "openInterest", "impliedVolatility"]].dropna()
                puts  = chain.puts[["strike", "openInterest", "impliedVolatility"]].dropna()
                for _, row in calls.iterrows():
                    k = float(row["strike"])
                    call_oi_map[k] = call_oi_map.get(k, 0) + float(row["openInterest"] or 0)
                    call_iv_map[k] = max(call_iv_map.get(k, 0), float(row["impliedVolatility"] or 0))
                for _, row in puts.iterrows():
                    k = float(row["strike"])
                    put_oi_map[k] = put_oi_map.get(k, 0) + float(row["openInterest"] or 0)
                    put_iv_map[k] = max(put_iv_map.get(k, 0), float(row["impliedVolatility"] or 0))
            except Exception:
                continue

        all_strikes = sorted(set(call_oi_map) | set(put_oi_map))
        if not all_strikes:
            return None

        # Filter to ±20% of spot to avoid noise
        all_strikes = [k for k in all_strikes if abs(k - spot) / spot <= 0.20]
        if not all_strikes:
            return None

        strikes = np.array(all_strikes)
        call_gex = np.zeros(len(strikes))
        put_gex  = np.zeros(len(strikes))

        for i, k in enumerate(all_strikes):
            distance = abs(k - spot) / spot
            # Gaussian weight — peaks at ATM, decays with distance
            weight = float(np.exp(-((distance / 0.08) ** 2)))
            c_oi = call_oi_map.get(k, 0)
            p_oi = put_oi_map.get(k, 0)
            c_iv = call_iv_map.get(k, 0.3)
            p_iv = put_iv_map.get(k, 0.3)
            # GEX notional: OI × IV × spot × 100 (contract multiplier)
            call_gex[i] = +c_oi * c_iv * spot * 100 * weight
            put_gex[i]  = -p_oi * p_iv * spot * 100 * weight

        net_gex = call_gex + put_gex

        # Gamma flip: cumsum crosses zero
        cum = np.cumsum(net_gex)
        gamma_flip = float(spot)  # default to spot if no flip found
        for i in range(1, len(cum)):
            if cum[i - 1] * cum[i] <= 0:  # sign change
                gamma_flip = float(strikes[i])
                break

        # Walls: strike with highest absolute OI
        call_wall_idx = np.argmax([call_oi_map.get(k, 0) for k in all_strikes])
        put_wall_idx  = np.argmax([put_oi_map.get(k, 0) for k in all_strikes])
        call_wall = float(strikes[call_wall_idx])
        put_wall  = float(strikes[put_wall_idx])

        # Dealer net delta: rough directional proxy
        # Positive = dealers net long delta (tend to sell into strength)
        # Negative = dealers net short delta (tend to buy into weakness)
        total_call_oi = sum(call_oi_map.get(k, 0) for k in all_strikes)
        total_put_oi  = sum(put_oi_map.get(k, 0) for k in all_strikes)
        dealer_net_delta = 0.0
        if total_call_oi + total_put_oi > 0:
            dealer_net_delta = round((total_call_oi - total_put_oi) / (total_call_oi + total_put_oi), 3)

        total_gex = float(np.sum(net_gex)) / 1e6  # in $millions

        # Zone classification
        spot_idx = int(np.argmin(np.abs(strikes - spot)))
        spot_gex = float(net_gex[spot_idx])
        if spot_gex >= 0:
            zone = "Positive Gamma Zone"
            zone_detail = "Dealers are net long gamma — they sell into rallies and buy dips, suppressing volatility."
        else:
            zone = "Negative Gamma Zone"
            zone_detail = "Dealers are net short gamma — they chase moves directionally, amplifying volatility."

        import datetime as _dt
        return {
            "ticker":           ticker,
            "spot":             round(spot, 2),
            "strikes":          strikes.tolist(),
            "call_gex":         call_gex.tolist(),
            "put_gex":          put_gex.tolist(),
            "net_gex":          net_gex.tolist(),
            "total_gex":        round(total_gex, 1),
            "gamma_flip":       round(gamma_flip, 2),
            "call_wall":        round(call_wall, 2),
            "put_wall":         round(put_wall, 2),
            "dealer_net_delta": dealer_net_delta,
            "zone":             zone,
            "zone_detail":      zone_detail,
            "asof":             _dt.datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_rolling_correlation(tickers: tuple[str, ...], short_window: int = 20, long_window: int = 60) -> dict | None:
    """Compute rolling pairwise correlations at two windows to detect stress-driven spikes.

    Args:
        tickers: tuple of ticker symbols (hashable for cache). Include benchmark (e.g. "SPY").
        short_window: recent window in trading days (default 20 = ~1 month)
        long_window:  baseline window in trading days (default 60 = ~3 months)

    Returns dict with:
        pairs: list of {pair, short_corr, long_corr, delta, stress_flag}
            - delta = short_corr - long_corr  (positive = correlations rising = less diversification)
            - stress_flag = True when delta > +0.25 (correlations spiking toward 1 in a selloff)
        avg_short: portfolio average pairwise correlation over short window
        avg_long:  portfolio average pairwise correlation over long window
        avg_delta: avg_short - avg_long
        concentration_warning: True if avg_short > 0.70 (positions moving together)
        stress_warning: True if avg_delta > 0.15 (correlations rising — diversification collapsing)
        as_of: ISO date string
    """
    import datetime as _dt
    if not tickers or len(tickers) < 2:
        return None
    needed_days = long_window + 10
    try:
        raw = yf.download(
            list(tickers), period=f"{needed_days * 2}d", interval="1d",
            progress=False, auto_adjust=True, threads=True
        )
        if raw is None or raw.empty:
            return None
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        closes = closes.dropna(how="all")
        if len(closes) < long_window + 5:
            return None
        returns = closes.pct_change().dropna(how="all")
        cols = [c for c in returns.columns if str(c).upper() in [t.upper() for t in tickers]]
        returns = returns[cols]
        returns.columns = [str(c).upper() for c in returns.columns]

        recent = returns.iloc[-short_window:]
        baseline = returns.iloc[-long_window:]

        short_corr = recent.corr()
        long_corr = baseline.corr()

        pairs = []
        tickers_clean = list(returns.columns)
        for i in range(len(tickers_clean)):
            for j in range(i + 1, len(tickers_clean)):
                a, b = tickers_clean[i], tickers_clean[j]
                sc = float(short_corr.loc[a, b]) if a in short_corr.index and b in short_corr.columns else None
                lc = float(long_corr.loc[a, b]) if a in long_corr.index and b in long_corr.columns else None
                if sc is None or lc is None:
                    continue
                delta = sc - lc
                pairs.append({
                    "pair": f"{a}/{b}",
                    "short_corr": round(sc, 3),
                    "long_corr": round(lc, 3),
                    "delta": round(delta, 3),
                    "stress_flag": delta > 0.25,
                })

        if not pairs:
            return None

        avg_short = sum(p["short_corr"] for p in pairs) / len(pairs)
        avg_long  = sum(p["long_corr"]  for p in pairs) / len(pairs)
        avg_delta = avg_short - avg_long

        return {
            "pairs": sorted(pairs, key=lambda x: abs(x["delta"]), reverse=True),
            "avg_short": round(avg_short, 3),
            "avg_long":  round(avg_long,  3),
            "avg_delta": round(avg_delta, 3),
            "concentration_warning": avg_short > 0.70,
            "stress_warning":        avg_delta > 0.15,
            "short_window": short_window,
            "long_window":  long_window,
            "as_of": _dt.date.today().isoformat(),
        }
    except Exception:
        return None


@st.cache_data(ttl=14400)
def fetch_ticker_fundamentals(ticker: str) -> dict:
    """Fetch raw fundamental and positioning metrics for a ticker.

    Returns a flat dict of numeric fields — all from yfinance .info and
    .recommendations. Safe to call in any module; cached 4 hours.

    Fields returned (all None if unavailable):
      Valuation multiples:
        pe_trailing, pe_forward, peg, ps_ratio, pb_ratio, ev_ebitda
      Returns & quality:
        roe, roa, profit_margin, revenue_growth_yoy, earnings_growth_yoy
      Balance sheet:
        debt_to_equity, current_ratio, quick_ratio, total_cash_per_share
      Cash flow:
        fcf_yield (FCF / market_cap), operating_cashflow, levered_fcf
      Dividends:
        div_yield, payout_ratio
      Short interest:
        short_pct_float, short_ratio (days to cover)
      Analyst consensus:
        analyst_score (1=strong buy, 5=strong sell), analyst_count,
        target_mean, target_median, target_high, target_low,
        revision_score (net upgrades - downgrades last 30d, from .recommendations)
    """
    result: dict = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return result

    def _safe(key: str, transform=None):
        v = info.get(key)
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return None
        try:
            return transform(v) if transform else v
        except Exception:
            return None

    # ── Valuation multiples ────────────────────────────────────────────────────
    result["pe_trailing"]  = _safe("trailingPE",    float)
    result["pe_forward"]   = _safe("forwardPE",     float)
    result["peg"]          = _safe("pegRatio",      float)
    result["ps_ratio"]     = _safe("priceToSalesTrailing12Months", float)
    result["pb_ratio"]     = _safe("priceToBook",   float)
    result["ev_ebitda"]    = _safe("enterpriseToEbitda", float)

    # ── Returns & quality ─────────────────────────────────────────────────────
    result["roe"]               = _safe("returnOnEquity",   float)
    result["roa"]               = _safe("returnOnAssets",   float)
    result["profit_margin"]     = _safe("profitMargins",    float)
    result["revenue_growth_yoy"]  = _safe("revenueGrowth",   float)
    result["earnings_growth_yoy"] = _safe("earningsGrowth",  float)

    # ── Balance sheet ─────────────────────────────────────────────────────────
    result["debt_to_equity"]     = _safe("debtToEquity",    float)
    result["current_ratio"]      = _safe("currentRatio",    float)
    result["quick_ratio"]        = _safe("quickRatio",      float)
    result["total_cash_per_share"] = _safe("totalCashPerShare", float)

    # ── Cash flow ─────────────────────────────────────────────────────────────
    result["operating_cashflow"] = _safe("operatingCashflow", float)
    result["levered_fcf"]        = _safe("freeCashflow",     float)
    mktcap = _safe("marketCap", float)
    fcf    = result.get("levered_fcf")
    result["fcf_yield"] = (fcf / mktcap) if (fcf and mktcap and mktcap > 0) else None

    # ── Dividends ─────────────────────────────────────────────────────────────
    result["div_yield"]    = _safe("dividendYield",  float)
    result["payout_ratio"] = _safe("payoutRatio",    float)

    # ── Short interest ────────────────────────────────────────────────────────
    result["short_pct_float"] = _safe("shortPercentOfFloat", float)
    result["short_ratio"]     = _safe("shortRatio",          float)

    # ── Analyst consensus ─────────────────────────────────────────────────────
    result["analyst_score"]  = _safe("recommendationMean",       float)  # 1=strong buy, 5=strong sell
    result["analyst_count"]  = _safe("numberOfAnalystOpinions",  int)
    result["target_mean"]    = _safe("targetMeanPrice",          float)
    result["target_median"]  = _safe("targetMedianPrice",        float)
    result["target_high"]    = _safe("targetHighPrice",          float)
    result["target_low"]     = _safe("targetLowPrice",           float)

    # ── Analyst revisions (net upgrades - downgrades, last ~30d) ─────────────
    try:
        rec_df = yf.Ticker(ticker).recommendations
        if rec_df is not None and not rec_df.empty:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
            rec_df.index = pd.to_datetime(rec_df.index, utc=True)
            recent = rec_df[rec_df.index >= cutoff]
            if not recent.empty:
                _upgrades   = recent["To Grade"].str.lower().str.contains("buy|outperform|overweight").sum()
                _downgrades = recent["To Grade"].str.lower().str.contains("sell|underperform|underweight").sum()
                result["revision_score"] = int(_upgrades) - int(_downgrades)
            else:
                result["revision_score"] = None
        else:
            result["revision_score"] = None
    except Exception:
        result["revision_score"] = None

    return result


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_credit_metrics(ticker: str) -> dict | None:
    """Fetch fixed income & credit risk metrics for a ticker.

    Uses yfinance financial statements (income statement + balance sheet) to compute:
    - Interest coverage ratio (EBIT / Interest Expense) — key solvency signal
    - Net debt = total_debt - cash
    - Debt-to-EBITDA (leverage ratio)
    - Current debt / total debt ratio (near-term refinancing pressure)
    - FCF debt coverage (levered FCF / total debt)
    - Debt maturity risk flag

    Returns dict or None on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # ── Balance sheet ──────────────────────────────────────────────────────
        bs = tk.balance_sheet
        total_debt = None
        current_debt = None
        cash = None
        total_assets = None

        if bs is not None and not bs.empty:
            def _get_bs(keys):
                for k in keys:
                    for col in bs.columns:
                        if k in bs.index:
                            val = bs.loc[k, col]
                            if val is not None and str(val) not in ("nan", "None"):
                                try:
                                    return float(val)
                                except Exception:
                                    pass
                return None

            total_debt   = _get_bs(["Total Debt", "Long Term Debt And Capital Lease Obligation"])
            current_debt = _get_bs(["Current Debt", "Current Debt And Capital Lease Obligation", "Current Portion Of Long Term Debt"])
            cash         = _get_bs(["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
            total_assets = _get_bs(["Total Assets"])

        # ── Income statement ───────────────────────────────────────────────────
        inc = tk.income_stmt
        ebit = None
        ebitda = None
        interest_expense = None

        if inc is not None and not inc.empty:
            def _get_inc(keys):
                for k in keys:
                    for col in inc.columns:
                        if k in inc.index:
                            val = inc.loc[k, col]
                            if val is not None and str(val) not in ("nan", "None"):
                                try:
                                    return float(val)
                                except Exception:
                                    pass
                return None

            ebit             = _get_inc(["EBIT", "Operating Income"])
            ebitda           = _get_inc(["EBITDA"])
            interest_expense = _get_inc(["Interest Expense", "Interest Expense Non Operating"])
            if interest_expense and interest_expense < 0:
                interest_expense = abs(interest_expense)  # normalize sign

        # ── Derived metrics ────────────────────────────────────────────────────
        interest_coverage = None
        if ebit and interest_expense and interest_expense > 0:
            interest_coverage = round(ebit / interest_expense, 2)

        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash

        debt_to_ebitda = None
        if total_debt and ebitda and ebitda > 0:
            debt_to_ebitda = round(total_debt / ebitda, 2)

        current_debt_ratio = None
        if current_debt and total_debt and total_debt > 0:
            current_debt_ratio = round(current_debt / total_debt, 3)

        # FCF from yfinance info
        levered_fcf = info.get("freeCashflow")
        fcf_debt_coverage = None
        if levered_fcf and total_debt and total_debt > 0:
            fcf_debt_coverage = round(levered_fcf / total_debt, 3)

        # ── Risk classification ────────────────────────────────────────────────
        coverage_flag = None
        if interest_coverage is not None:
            if interest_coverage < 1.5:
                coverage_flag = "CRITICAL — coverage below 1.5x, default risk elevated"
            elif interest_coverage < 3.0:
                coverage_flag = "WARNING — coverage below 3.0x, limited cushion"
            elif interest_coverage >= 5.0:
                coverage_flag = "STRONG — coverage above 5x"

        maturity_flag = None
        if current_debt_ratio is not None:
            if current_debt_ratio > 0.40:
                maturity_flag = "HIGH REFINANCING RISK — >40% of debt matures within 12 months"
            elif current_debt_ratio > 0.20:
                maturity_flag = "ELEVATED REFINANCING RISK — >20% of debt matures near-term"

        result = {
            "ticker": ticker.upper(),
            "total_debt_B": round(total_debt / 1e9, 2) if total_debt else None,
            "current_debt_B": round(current_debt / 1e9, 2) if current_debt else None,
            "cash_B": round(cash / 1e9, 2) if cash else None,
            "net_debt_B": round(net_debt / 1e9, 2) if net_debt else None,
            "ebit_B": round(ebit / 1e9, 2) if ebit else None,
            "ebitda_B": round(ebitda / 1e9, 2) if ebitda else None,
            "interest_expense_B": round(interest_expense / 1e9, 2) if interest_expense else None,
            "interest_coverage": interest_coverage,
            "debt_to_ebitda": debt_to_ebitda,
            "current_debt_ratio": current_debt_ratio,
            "fcf_debt_coverage": fcf_debt_coverage,
            "coverage_flag": coverage_flag,
            "maturity_flag": maturity_flag,
        }
        # Remove None-only result
        if all(v is None for k, v in result.items() if k != "ticker"):
            return None
        return result
    except Exception:
        return None

