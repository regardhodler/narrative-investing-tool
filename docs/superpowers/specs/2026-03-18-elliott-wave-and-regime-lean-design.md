# Design Spec: Elliott Wave Module + Risk Regime Neutral Leaning

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Two independent features for the narrative-investing-tool Streamlit app

---

## Feature 1: Elliott Wave Module (SPY only)

### Goal
Add a new "Elliott Wave" sidebar module that automatically counts Elliott waves on SPY at the Primary degree, scores the count against hard EW rules and Fibonacci ratios, and generates a Groq LLaMA narrative interpretation of the current wave position.

This module is **SPY-only** by design. The scope is a single instrument with a clean long-history dataset suitable for primary-degree counting.

---

### New File: `services/elliott_wave_engine.py`

Pure detection logic. No Streamlit imports. Fully testable in isolation.

#### Data Models

```python
@dataclass
class Pivot:
    date: pd.Timestamp
    price: float           # closing price at pivot bar
    type: str              # "high" | "low"

@dataclass
class WaveSequence:
    pivots: list[Pivot]    # 6 pivots for impulse (0,1,2,3,4,5), 4 for corrective (0,A,B,C)
    wave_type: str         # "impulse" | "corrective"
    labels: list[str]      # ["0","1","2","3","4","5"] or ["0","A","B","C"]

@dataclass
class BestCount:
    sequence: WaveSequence
    confidence: int             # 0-100
    wave_position: str          # e.g. "In Wave 3 of Primary Impulse (incomplete)"
    current_wave_label: str     # e.g. "3" or "C" — the active wave
    invalidation_level: float   # price that breaks the count (see derivation below)
    fibonacci_hits: list[str]   # format: "Wave 3 = 1.618 × Wave 1 (+20pts)"
```

#### Function: `detect_pivots(series, atr_multiplier=1.0)`

ATR(14)-filtered zigzag pivot detection using **closing prices**.

Algorithm:
1. Compute ATR(14) on the input series
2. Track a running candidate pivot (starting direction determined by first move)
3. A candidate **high pivot** is confirmed when price subsequently closes ≥ `atr_multiplier × ATR(14)` *below* the candidate high's close
4. A candidate **low pivot** is confirmed when price subsequently closes ≥ `atr_multiplier × ATR(14)` *above* the candidate low's close
5. Returns a list of alternating `Pivot` objects (high, low, high, low…)

Default `atr_multiplier=1.0` (one full ATR range) produces approximately 20–40 pivots on 252 trading days of SPY data, which is the target density for primary-degree counting.

#### Function: `find_wave_sequences(pivots)`

Generates candidate wave sequences using a **sliding window** over the pivot list.

- Impulse (5-wave): window of 6 consecutive pivots. Must start on a low pivot (0 = low).
- Corrective (3-wave): window of 4 consecutive pivots. Must start on a high pivot (0 = high) for post-impulse correction.
- Sliding step: 1 pivot at a time
- **Search bound:** last 30 pivots only (covers ~6–18 months at primary degree density). Older pivots are excluded to keep the count current and computation bounded.
- Returns all candidate sequences before filtering.

#### Function: `score_sequence(seq) -> tuple[bool, int, list[str]]`

Returns `(is_valid, confidence_score, fibonacci_hits)`.

**Hard rules (any violation → `is_valid=False`, sequence discarded):**
- Wave 3 is not the shortest among waves 1, 3, 5 (measured close-to-close)
- Wave 4 close does not overlap wave 1 close (wave 4 low must not exceed wave 1 high in an upward impulse)
- Wave 2 does not retrace beyond wave 0 (wave 2 low must not exceed wave 0 low in an upward impulse)

**Fibonacci scoring (soft, tolerance ±10%):**
| Condition | Points |
|-----------|--------|
| Wave 3 length = 1.618 × Wave 1 length | +20 |
| Wave 2 retraces 0.618 of Wave 1 | +15 |
| Wave 4 retraces 0.382 of Wave 3 | +15 |
| Wave 5 length = Wave 1 length | +10 |
| Wave 3 length = 2.618 × Wave 1 length | +15 |
| Wave 4 retraces 0.500 of Wave 3 | +10 |

Maximum raw Fibonacci score: 85 pts. Normalized to 0–100 range. A sequence with zero Fibonacci hits scores 0; all hits scores 100.

`fibonacci_hits` list entries use format: `"Wave 3 = 1.618 × Wave 1 (+20pts)"`

#### Function: `get_best_wave_count(series) -> BestCount | None`

Orchestrates the full pipeline:
1. `detect_pivots(series)` → pivot list
2. Return `None` if fewer than 6 pivots found
3. `find_wave_sequences(pivots)` → candidates
4. `score_sequence(seq)` on each → keep only valid sequences
5. Return `None` if no valid sequences
6. Select the sequence with the highest confidence score
7. Determine `wave_position` and `current_wave_label`:
   - If the last pivot in the sequence equals the most recent pivot overall → sequence is complete ("Wave 5 of Primary Impulse complete")
   - Otherwise → sequence is in progress ("In Wave 3 of Primary Impulse (incomplete)")
8. Derive `invalidation_level`:
   - **Impulse:** price of pivot 0 (the wave 0 origin). Any close below this level invalidates the count.
   - **Corrective:** price of pivot 0 (the wave A start). Any close above this level invalidates a bearish correction.
9. Return `BestCount`

---

### New File: `modules/elliott_wave.py`

UI layer only. Follows existing module pattern.

#### `_fetch_spy_data() -> AssetSnapshot`
```python
@st.cache_data(ttl=3600)
def _fetch_spy_data():
    snaps = fetch_batch_safe({"SPY": "S&P 500"}, period="1y", interval="1d")
    return snaps.get("SPY")
```

#### `_build_groq_narrative(wave_position, current_wave_label, confidence, invalidation_level, fibonacci_hits) -> str`
```python
@st.cache_data(ttl=3600)  # matches price data TTL — avoids stale narrative vs chart
def _build_groq_narrative(...):
```
Calls Groq LLaMA via existing `services/claude_client.py` pattern.

Prompt includes:
- Current wave label and position description
- Confidence score
- Invalidation level
- List of Fibonacci hits
- Instruction: describe current position, what EW theory predicts next, and the invalidation scenario in 3–5 sentences

Note: `BestCount` dataclass is **not** passed directly to this cached function to avoid serialization issues. Only primitive/string fields are passed as arguments.

#### `_make_wave_chart(series, best_count) -> go.Figure`
- SPY daily candlestick
- Wave lines connecting pivot points: green for impulse waves, orange for corrective
- Wave labels (0–5 or A–B–C) annotated at each pivot point
- Horizontal dashed red line at `invalidation_level` with label "Invalidation"
- Uses `apply_dark_layout()` from `utils/theme.py`
- If `best_count` is `None`, returns a plain price chart with no wave overlay

#### `render()`
Layout:
1. **Chart** (full width) — `_make_wave_chart()`
2. **Warning banner** (conditional):
   - If `best_count is None`: `st.info("No clean Elliott Wave count detected...")`
   - If `best_count.confidence < 40`: `st.warning("Low-confidence count — treat as speculative")`
   - These banners render *above* the metrics row, not replacing the chart
3. **Metrics row** (3 columns):
   - Col 1: `bloomberg_metric("Wave Position", best_count.wave_position)`
   - Col 2: `bloomberg_metric("Confidence", f"{best_count.confidence}/100")`
   - Col 3: `bloomberg_metric("Invalidation", f"${best_count.invalidation_level:.2f}")`
4. **Fibonacci Hits** (if any): bulleted list under metrics
5. **AI Narrative** expander (default collapsed):
   - Groq narrative text
   - If Groq failed: `st.warning("AI narrative unavailable")` + "Retry" button that calls `st.cache_data.clear()` then `st.rerun()`
   - Retry clears the full cache (same pattern as existing modules), re-running both the wave count and the narrative

---

### Modified Files

#### `app.py`
Add `"Elliott Wave"` to the top-level sidebar radio list, positioned after `"Risk Regime"`:
```python
top_level = st.radio(
    "Module",
    ["Discovery", "Risk Regime", "Elliott Wave", "Whale Movement", ...],
    ...
)
```
Add routing block:
```python
elif top_level == "Elliott Wave":
    from modules.elliott_wave import render
    render()
```

---

## Feature 2: Risk Regime Neutral Leaning

### Goal
When macro score is in the neutral zone (41–59), display a nuanced three-tier label with clear boundary ownership.

### Label Logic (exclusive boundaries)

```
score ≥ 60   →  "Risk-On"
score 53–59  →  "Neutral — Leaning Risk-On"
score 48–52  →  "True Neutral"
score 41–47  →  "Neutral — Leaning Risk-Off"
score ≤ 40   →  "Risk-Off"
```

Boundaries are exclusive: score 60 is "Risk-On", score 40 is "Risk-Off", no collision.

### Changes to `modules/risk_regime.py`

#### 1. New helper `_neutral_lean_label(score: int) -> str`
```python
def _neutral_lean_label(score: int) -> str:
    if score >= 53:
        return "Neutral — Leaning Risk-On"
    elif score <= 47:
        return "Neutral — Leaning Risk-Off"
    return "True Neutral"
```

#### 2. Updated `_score_to_bucket(score: float) -> tuple[str, str]`
Emoji stays 🟡 for all neutral tiers. Label uses `_neutral_lean_label`:
```python
def _score_to_bucket(score: float) -> tuple[str, str]:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "🟢", "Risk-On"
    if macro_score <= 40:
        return "🔴", "Risk-Off"
    return "🟡", _neutral_lean_label(macro_score)
```

#### 3. Updated `_label_from_score(score: float) -> str`
Used for history signal change comparisons. Must match `_score_to_bucket` logic:
```python
def _label_from_score(score: float) -> str:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "Risk-On"
    if macro_score <= 40:
        return "Risk-Off"
    return _neutral_lean_label(macro_score)
```

#### 4. Updated `_build_macro_dashboard()` — `macro_regime` field
Replace the existing neutral assignment:
```python
# Before:
macro_regime = "Neutral"
# After:
macro_regime = _neutral_lean_label(macro_score)
```
This propagates the nuanced label to history snapshots and `get_current_regime()` automatically.

#### 5. Updated `render()` — Regime metric display
The `bloomberg_metric("Regime", regime, regime_color)` call requires no change to its signature since `regime` now holds the nuanced string. The `regime_color` logic stays the same — yellow for any non-Risk-On, non-Risk-Off value:
```python
regime_color = (
    COLORS["green"] if regime == "Risk-On"
    else COLORS["red"] if regime == "Risk-Off"
    else COLORS["yellow"]
)
```
The gauge title in `_make_gauge()` also receives the nuanced label via `regime` parameter — no change needed there.

---

## Out of Scope
- Multiple wave degrees (Intermediate, Minor) — Primary only
- Interactive wave count annotation by user
- Backtesting wave count accuracy
- Real-time intraday wave counting
- Multi-ticker EW analysis
