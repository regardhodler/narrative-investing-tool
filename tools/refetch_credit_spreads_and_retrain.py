"""Refetch BAMLH0A0HYM2 + BAMLC0A0CM with full FRED history (requires FRED_API_KEY),
then retrain the main HMM brain on the refreshed feature window.

Usage:
    set FRED_API_KEY=your_key   (Windows)
    export FRED_API_KEY=your_key  (Mac/Linux)
    python tools/refetch_credit_spreads_and_retrain.py
"""
from __future__ import annotations

import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from services.hmm_regime import save_hmm_brain, train_hmm
from services.hmm_shadow import train_shadow_hmm, save_shadow_brain
from services.hmm_top import train_top_brain, save_top_brain

_TARGETS = ["BAA10Y", "AAA10Y"]
_FRED_API = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fred_cache")


def _fetch_full_series(series_id: str, api_key: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "1990-01-01",
        "sort_order": "asc",
    }
    r = requests.get(_FRED_API, params=params, timeout=60)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value", ".")
        if v == ".":
            continue
        try:
            rows.append({"observation_date": o["date"], series_id: float(v)})
        except (ValueError, KeyError):
            continue
    return pd.DataFrame(rows)


def main() -> int:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print("ERROR: FRED_API_KEY env var is not set.")
        print("  Get a free key at https://fredaccount.stlouisfed.org/apikey")
        return 1

    cache_dir = os.path.abspath(_CACHE_DIR)
    print(f"Cache dir: {cache_dir}")

    for sid in _TARGETS:
        print(f"\n-> Fetching {sid} ...")
        df = _fetch_full_series(sid, api_key)
        if df.empty:
            print(f"  [!] Empty response for {sid} — skipping.")
            continue
        out_path = os.path.join(cache_dir, f"{sid}.csv")
        df.to_csv(out_path, index=False)
        first = df["observation_date"].iloc[0]
        last = df["observation_date"].iloc[-1]
        print(f"  [OK] Wrote {out_path} — {first} -> {last}  ({len(df)} rows)")

    print("\n-> Retraining Main Brain on refreshed feature window ...")
    brain = train_hmm(lookback_years=15)
    save_hmm_brain(brain)
    print(f"  [OK] Main Brain saved.")
    print(f"     n_states     = {brain.n_states}")
    print(f"     training     = {brain.training_start} -> {brain.training_end}")
    print(f"     state_labels = {brain.state_labels}")
    print(f"     BIC          = {brain.bic:.0f}")

    print("\n-> Retraining Shadow Brain (SPX log returns + VIX) ...")
    shadow_brain = train_shadow_hmm()
    save_shadow_brain(shadow_brain)
    print(f"  [OK] Shadow Brain saved.")
    print(f"     n_states     = {shadow_brain.n_states}")
    print(f"     ci_anchor    = {shadow_brain.ci_anchor:.4f}")
    print(f"     state_labels = {shadow_brain.state_labels}")

    print("\n-> Retraining Top Brain (VIX+NFCI+BAA10Y+T10Y3M) ...")
    top_brain = train_top_brain(lookback_years=15)
    save_top_brain(top_brain)
    print(f"  [OK] Top Brain saved.")
    print(f"     n_states     = {top_brain.n_states}")
    print(f"     ci_anchor    = {top_brain.ci_anchor:.4f}")
    print(f"     state_labels = {top_brain.state_labels}")

    print("\nDone. Refresh your Streamlit app to see the new chart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
