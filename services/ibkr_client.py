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
