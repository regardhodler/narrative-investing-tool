import asyncio
import random
import streamlit as st


def _ensure_event_loop():
    """Ensure an asyncio event loop exists in the current thread."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def connect_ibkr(port: int = 7497) -> bool:
    """Connect to IBKR TWS/Gateway. Stores IB object in session state.

    Args:
        port: 7497 for paper trading, 7496 for live.

    Returns True if connected successfully.
    """
    _ensure_event_loop()
    from ib_insync import IB

    # Reuse existing connection if still alive
    ib = st.session_state.get("ib_connection")
    if ib is not None:
        try:
            if ib.isConnected():
                return True
        except Exception:
            pass

    ib = IB()
    client_id = random.randint(1, 9999)
    try:
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=10)
        st.session_state["ib_connection"] = ib
        return True
    except Exception as e:
        st.session_state.pop("ib_connection", None)
        st.error(f"IBKR connection failed: {e}")
        return False


def disconnect_ibkr():
    ib = st.session_state.get("ib_connection")
    if ib is not None:
        try:
            ib.disconnect()
        except Exception:
            pass
        st.session_state.pop("ib_connection", None)


def get_options_chain(ticker: str) -> list[dict]:
    """Fetch options chain for ticker: nearest 3 expirations, strikes ±10% ATM.

    Returns list of dicts with contract + greeks data.
    """
    _ensure_event_loop()
    from ib_insync import Stock, util

    ib = st.session_state.get("ib_connection")
    if ib is None or not ib.isConnected():
        return []

    stock = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stock)

    [ticker_data] = ib.reqTickers(stock)
    ib.sleep(1)
    price = ticker_data.marketPrice()
    if price != price:  # NaN check
        price = ticker_data.close
    if not price or price != price:
        return []

    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        return []

    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    expirations = sorted(chain.expirations)[:3]

    strike_low = price * 0.9
    strike_high = price * 1.1
    strikes = [s for s in chain.strikes if strike_low <= s <= strike_high]

    results = []
    for exp in expirations:
        for right in ["C", "P"]:
            for strike in strikes:
                from ib_insync import Option

                contract = Option(ticker, exp, strike, right, "SMART")
                try:
                    ib.qualifyContracts(contract)
                except Exception:
                    continue

                [opt_ticker] = ib.reqTickers(contract)
                ib.sleep(0.2)

                greeks = opt_ticker.modelGreeks or opt_ticker.lastGreeks
                vol = (
                    opt_ticker.volume
                    if opt_ticker.volume and opt_ticker.volume != -1
                    else 0
                )
                oi = (
                    opt_ticker.openInterest
                    if hasattr(opt_ticker, "openInterest")
                    and opt_ticker.openInterest
                    and opt_ticker.openInterest != -1
                    else 0
                )

                results.append(
                    {
                        "expiration": exp,
                        "strike": strike,
                        "right": right,
                        "volume": vol,
                        "open_interest": oi,
                        "iv": greeks.impliedVol if greeks else None,
                        "delta": greeks.delta if greeks else None,
                        "gamma": greeks.gamma if greeks else None,
                        "theta": greeks.theta if greeks else None,
                        "vega": greeks.vega if greeks else None,
                        "vol_oi_ratio": (vol / oi) if oi > 0 else None,
                    }
                )

    return results


def fetch_market_depth(ticker: str = "SPY", num_rows: int = 10) -> dict | None:
    """Fetch Level 2 market depth from IBKR.

    Returns dict with bid/ask ladder, imbalance ratio, and large order flags.
    Returns None if IBKR is not connected.
    """
    _ensure_event_loop()
    from ib_insync import Stock

    ib = st.session_state.get("ib_connection")
    if ib is None or not ib.isConnected():
        return None

    stock = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stock)

    # Request market depth (L2)
    depth = ib.reqMktDepth(stock, numRows=num_rows)
    ib.sleep(2)  # wait for depth data to populate

    try:
        ticker_obj = depth if hasattr(depth, "domBids") else None
        # ib_insync stores depth on the contract's ticker object
        tickers = ib.reqTickers(stock)
        ib.sleep(0.5)
        if not tickers:
            ib.cancelMktDepth(stock)
            return None

        t = tickers[0]
        dom_bids = getattr(t, "domBids", []) or []
        dom_asks = getattr(t, "domAsks", []) or []

        bid_levels = [{"price": b.price, "size": b.size} for b in dom_bids if b.price > 0]
        ask_levels = [{"price": a.price, "size": a.size} for a in dom_asks if a.price > 0]

        sum_bid = sum(b["size"] for b in bid_levels) or 1
        sum_ask = sum(a["size"] for a in ask_levels) or 1

        # Bid/Ask imbalance: +1 = heavy bids (bullish), -1 = heavy asks (bearish)
        imbalance = (sum_bid - sum_ask) / (sum_bid + sum_ask)

        # Spot price for 1% depth filter
        spot = t.marketPrice()
        if not spot or spot != spot:
            spot = t.close or 0

        # Depth within 1% of spot
        bid_1pct = sum(b["size"] for b in bid_levels if spot > 0 and b["price"] >= spot * 0.99)
        ask_1pct = sum(a["size"] for a in ask_levels if spot > 0 and a["price"] <= spot * 1.01)
        depth_ratio_1pct = ((bid_1pct - ask_1pct) / (bid_1pct + ask_1pct)
                            if (bid_1pct + ask_1pct) > 0 else 0.0)

        # Large order detection: any level > 3σ of average
        import numpy as np
        all_sizes = [b["size"] for b in bid_levels] + [a["size"] for a in ask_levels]
        if len(all_sizes) >= 3:
            _mean = np.mean(all_sizes)
            _std = np.std(all_sizes)
            _threshold = _mean + 3 * _std if _std > 0 else _mean * 3
            large_bids = [b for b in bid_levels if b["size"] > _threshold]
            large_asks = [a for a in ask_levels if a["size"] > _threshold]
        else:
            large_bids, large_asks = [], []

        ib.cancelMktDepth(stock)

        return {
            "ticker": ticker,
            "spot": float(spot) if spot else 0,
            "bid_levels": bid_levels[:num_rows],
            "ask_levels": ask_levels[:num_rows],
            "sum_bid": int(sum_bid),
            "sum_ask": int(sum_ask),
            "imbalance": round(float(imbalance), 4),
            "depth_ratio_1pct": round(float(depth_ratio_1pct), 4),
            "large_bids": large_bids,
            "large_asks": large_asks,
            "n_large_orders": len(large_bids) + len(large_asks),
        }
    except Exception:
        try:
            ib.cancelMktDepth(stock)
        except Exception:
            pass
        return None
