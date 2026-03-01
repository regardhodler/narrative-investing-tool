import streamlit as st


def get_narrative() -> str | None:
    return st.session_state.get("active_narrative")


def set_narrative(narrative: str):
    st.session_state["active_narrative"] = narrative


def get_ticker() -> str | None:
    return st.session_state.get("active_ticker")


def set_ticker(ticker: str):
    st.session_state["active_ticker"] = ticker


def is_ibkr_connected() -> bool:
    ib = st.session_state.get("ib_connection")
    if ib is None:
        return False
    try:
        return ib.isConnected()
    except Exception:
        return False


def get_ib():
    return st.session_state.get("ib_connection")
