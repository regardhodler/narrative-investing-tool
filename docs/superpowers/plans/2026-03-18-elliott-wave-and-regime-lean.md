# Elliott Wave Module + Risk Regime Neutral Leaning — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SPY Elliott Wave counting module with Groq AI narrative, and improve the Risk Regime "Neutral" label to show directional lean.

**Architecture:** Two independent features. Feature 1 adds `services/elliott_wave_engine.py` (pure detection logic) and `modules/elliott_wave.py` (Streamlit UI), wired into `app.py`. Feature 2 makes three targeted edits to `modules/risk_regime.py` — a new helper function and updates to four existing functions.

**Tech Stack:** Python 3.11+, Streamlit, Plotly, pandas, numpy, yfinance (via existing `services/market_data.py`), Groq LLaMA API (via existing `services/claude_client.py` pattern)

---

## Chunk 1: Risk Regime Neutral Leaning

**Files:**
- Modify: `modules/risk_regime.py`

---

### Task 1: Add `_neutral_lean_label` helper and update `_score_to_bucket`

- [ ] **Step 1: Write the failing test**

Create `tests/test_risk_regime_labels.py`:

```python
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import only the pure helper functions — no Streamlit needed
from modules.risk_regime import _neutral_lean_label, _score_to_bucket, _label_from_score


class TestNeutralLeanLabel:
    def test_leaning_risk_on(self):
        assert _neutral_lean_label(59) == "Neutral — Leaning Risk-On"
        assert _neutral_lean_label(53) == "Neutral — Leaning Risk-On"

    def test_true_neutral(self):
        assert _neutral_lean_label(52) == "True Neutral"
        assert _neutral_lean_label(50) == "True Neutral"
        assert _neutral_lean_label(48) == "True Neutral"

    def test_leaning_risk_off(self):
        assert _neutral_lean_label(47) == "Neutral — Leaning Risk-Off"
        assert _neutral_lean_label(41) == "Neutral — Leaning Risk-Off"


class TestScoreToBucket:
    def test_risk_on(self):
        emoji, label = _score_to_bucket(0.5)   # maps to macro_score ~75
        assert emoji == "🟢"
        assert label == "Risk-On"

    def test_risk_off(self):
        emoji, label = _score_to_bucket(-0.5)  # maps to macro_score ~25
        assert emoji == "🔴"
        assert label == "Risk-Off"

    def test_neutral_leaning_on(self):
        # score=0.16 → macro_score = int(round((0.16+1)*50)) = 58
        emoji, label = _score_to_bucket(0.16)
        assert emoji == "🟡"
        assert label == "Neutral — Leaning Risk-On"

    def test_true_neutral(self):
        emoji, label = _score_to_bucket(0.0)   # maps to macro_score 50
        assert emoji == "🟡"
        assert label == "True Neutral"

    def test_neutral_leaning_off(self):
        # score=-0.1 → macro_score = int(round((-0.1+1)*50)) = 45
        emoji, label = _score_to_bucket(-0.1)
        assert emoji == "🟡"
        assert label == "Neutral — Leaning Risk-Off"


class TestLabelFromScore:
    def test_risk_on(self):
        assert _label_from_score(0.5) == "Risk-On"

    def test_risk_off(self):
        assert _label_from_score(-0.5) == "Risk-Off"

    def test_neutral_boundaries(self):
        # score=0.16 → macro_score=58 → "Neutral — Leaning Risk-On"
        assert _label_from_score(0.16) == "Neutral — Leaning Risk-On"
        assert _label_from_score(0.0) == "True Neutral"
        assert _label_from_score(-0.1) == "Neutral — Leaning Risk-Off"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_risk_regime_labels.py -v 2>&1 | head -40
```

Expected: ImportError or AttributeError — `_neutral_lean_label` does not exist yet.

- [ ] **Step 3: Add `_neutral_lean_label` to `modules/risk_regime.py`**

Insert this new function after the existing `_confidence_label` helper (around line 216):

```python
def _neutral_lean_label(score: int) -> str:
    """Three-tier neutral label based on macro score (0-100 scale)."""
    if score >= 53:
        return "Neutral — Leaning Risk-On"
    elif score <= 47:
        return "Neutral — Leaning Risk-Off"
    return "True Neutral"
```

- [ ] **Step 4: Update `_score_to_bucket` in `modules/risk_regime.py`**

Replace the existing `_score_to_bucket` function (around line 127):

```python
def _score_to_bucket(score: float) -> tuple[str, str]:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "🟢", "Risk-On"
    if macro_score <= 40:
        return "🔴", "Risk-Off"
    return "🟡", _neutral_lean_label(macro_score)
```

- [ ] **Step 5: Update `_label_from_score` in `modules/risk_regime.py`**

Replace the existing `_label_from_score` function (around line 119):

```python
def _label_from_score(score: float) -> str:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "Risk-On"
    elif macro_score <= 40:
        return "Risk-Off"
    return _neutral_lean_label(macro_score)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_risk_regime_labels.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add modules/risk_regime.py tests/test_risk_regime_labels.py
git commit -m "feat: add three-tier neutral lean label to risk regime"
```

---

### Task 2: Propagate nuanced label into `_build_macro_dashboard` and `render`

**Depends on:** Task 1 must be committed first (Task 2 uses `_neutral_lean_label` which Task 1 creates).

- [ ] **Step 1: Append boundary tests to `tests/test_risk_regime_labels.py`**

**Append** (do not replace) to the existing `tests/test_risk_regime_labels.py` file:

```python
class TestNeutralLeanBoundaries:
    """Verify exact boundary values to prevent off-by-one regressions."""

    def test_boundary_60_is_risk_on(self):
        assert _label_from_score(0.2) == "Risk-On"   # macro_score=60

    def test_boundary_40_is_risk_off(self):
        assert _label_from_score(-0.2) == "Risk-Off"  # macro_score=40

    def test_boundary_59_is_leaning_on(self):
        # score=0.18 → macro_score=int(round(1.18*50))=59
        assert _label_from_score(0.18) == "Neutral — Leaning Risk-On"

    def test_boundary_41_is_leaning_off(self):
        # score=-0.18 → macro_score=int(round(0.82*50))=41
        assert _label_from_score(-0.18) == "Neutral — Leaning Risk-Off"
```

- [ ] **Step 2: Run test to verify it passes (these should pass already)**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_risk_regime_labels.py::TestNeutralLeanBoundaries -v
```

Expected: All 4 boundary tests PASS without any code changes.

- [ ] **Step 3: Update `_build_macro_dashboard` neutral assignment**

In `modules/risk_regime.py`, find the block around line 922–927:

```python
# BEFORE:
if macro_score >= 60:
    macro_regime = "Risk-On"
elif macro_score <= 40:
    macro_regime = "Risk-Off"
else:
    macro_regime = "Neutral"

# AFTER:
if macro_score >= 60:
    macro_regime = "Risk-On"
elif macro_score <= 40:
    macro_regime = "Risk-Off"
else:
    macro_regime = _neutral_lean_label(macro_score)
```

- [ ] **Step 4: Verify `regime_color` in `render()` already handles nuanced neutral strings**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "
import ast, sys
src = open('modules/risk_regime.py').read()
assert 'COLORS[\"yellow\"]' in src or \"COLORS['yellow']\" in src, 'yellow fallback missing'
# Confirm the ternary does not use == \"Neutral\" (hardcoded match that would break)
assert '== \"Neutral\"' not in src and \"== 'Neutral'\" not in src, 'hardcoded Neutral check found — must be removed'
print('regime_color fallthrough OK — no hardcoded Neutral string detected')
"
```

Expected: `regime_color fallthrough OK — no hardcoded Neutral string detected`

If the assertion fails with "hardcoded Neutral check found", locate and remove any `== "Neutral"` comparison in the `regime_color` line in `render()`, replacing it with the else-fallthrough form shown below:

```python
regime_color = (
    COLORS["green"] if regime == "Risk-On"
    else COLORS["red"] if regime == "Risk-Off"
    else COLORS["yellow"]
)
```

- [ ] **Step 4b: Confirm no other callers of `_score_to_bucket` exist outside `risk_regime.py`**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
grep -rn "_score_to_bucket" --include="*.py" .
```

Expected: Only matches inside `modules/risk_regime.py`. If matches appear elsewhere, those call sites must also be updated to handle the new nuanced neutral strings.

- [ ] **Step 5: Smoke-test the module imports cleanly**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "from modules.risk_regime import _neutral_lean_label, _score_to_bucket, _label_from_score; print('OK')"
```

Expected output: `OK`

- [ ] **Step 6: Run full label test suite**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_risk_regime_labels.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add modules/risk_regime.py tests/test_risk_regime_labels.py
git commit -m "feat: propagate neutral lean label into macro dashboard and render"
```

---

## Chunk 2: Elliott Wave Module

**Files:**
- Create: `services/elliott_wave_engine.py`
- Create: `modules/elliott_wave.py`
- Create: `tests/test_elliott_wave_engine.py`
- Modify: `app.py`

---

### Task 3: Elliott Wave Engine — data models and `detect_pivots`

- [ ] **Step 1: Write failing tests for `detect_pivots`**

Create `tests/test_elliott_wave_engine.py`:

```python
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.elliott_wave_engine import Pivot, detect_pivots


def _make_series(prices: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


class TestDetectPivots:
    def test_returns_list_of_pivots(self):
        prices = [100, 110, 105, 115, 108, 120, 112]
        result = detect_pivots(_make_series(prices))
        assert isinstance(result, list)
        assert all(isinstance(p, Pivot) for p in result)

    def test_alternates_high_low(self):
        # Up-down-up-down pattern should give alternating high/low
        prices = [100, 115, 105, 120, 108, 130, 115]
        result = detect_pivots(_make_series(prices))
        for i in range(len(result) - 1):
            assert result[i].type != result[i + 1].type

    def test_insufficient_data_returns_empty(self):
        prices = [100, 101, 99]
        result = detect_pivots(_make_series(prices))
        assert result == []

    def test_returns_empty_for_flat_data(self):
        prices = [100.0] * 50
        result = detect_pivots(_make_series(prices))
        assert result == []

    def test_pivot_has_required_fields(self):
        prices = [100, 110, 90, 115, 85, 120, 95, 125]
        result = detect_pivots(_make_series(prices))
        if result:
            p = result[0]
            assert hasattr(p, "date")
            assert hasattr(p, "price")
            assert p.type in ("high", "low")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError — `services.elliott_wave_engine` does not exist.

- [ ] **Step 3: Create `services/elliott_wave_engine.py` with data models and `detect_pivots`**

```python
"""
Elliott Wave Engine — SPY Primary Degree Wave Counting

Pipeline:
  detect_pivots()         → ATR-filtered swing highs/lows
  find_wave_sequences()   → candidate 5-wave impulse + 3-wave corrective windows
  score_sequence()        → hard EW rules + Fibonacci ratio scoring
  get_best_wave_count()   → highest-confidence valid count
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Pivot:
    date: pd.Timestamp
    price: float      # closing price at pivot bar
    type: str         # "high" | "low"


@dataclass
class WaveSequence:
    pivots: list[Pivot]   # 6 for impulse, 4 for corrective
    wave_type: str        # "impulse" | "corrective"
    labels: list[str]     # ["0","1","2","3","4","5"] or ["0","A","B","C"]


@dataclass
class BestCount:
    sequence: WaveSequence
    confidence: int              # 0-100
    wave_position: str           # e.g. "In Wave 3 of Primary Impulse (incomplete)"
    current_wave_label: str      # e.g. "3" or "C"
    invalidation_level: float    # price that breaks this count
    fibonacci_hits: list[str] = field(default_factory=list)


def _atr14(series: pd.Series) -> pd.Series:
    """Approximate ATR(14) using daily close-to-close range."""
    diff = series.diff().abs()
    return diff.rolling(14).mean()


def detect_pivots(series: pd.Series, atr_multiplier: float = 1.0) -> list[Pivot]:
    """
    ATR(14)-filtered zigzag pivot detection using closing prices.

    A high pivot is confirmed when price subsequently closes >= atr_multiplier * ATR(14)
    below the candidate high's close.
    A low pivot is confirmed when price subsequently closes >= atr_multiplier * ATR(14)
    above the candidate low's close.

    Returns alternating list of Pivot objects (high, low, high, low, ...).
    Target density: 20-40 pivots on 252 trading days with default atr_multiplier=1.0.
    """
    if len(series) < 20:
        return []

    atr = _atr14(series)
    closes = series.values
    dates = series.index
    n = len(closes)

    pivots: list[Pivot] = []
    # Track current candidate
    candidate_idx: int | None = None
    candidate_type: str | None = None  # "high" | "low"

    # Determine initial direction from first meaningful move
    for i in range(1, min(n, 30)):
        atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        if atr_val == 0:
            continue
        if closes[i] > closes[0] + atr_val:
            candidate_idx = 0
            candidate_type = "low"
            break
        if closes[i] < closes[0] - atr_val:
            candidate_idx = 0
            candidate_type = "high"
            break

    if candidate_idx is None:
        return []

    for i in range(1, n):
        atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
        if atr_val == 0:
            continue

        threshold = atr_multiplier * atr_val

        if candidate_type == "high":
            if closes[i] > closes[candidate_idx]:
                # Extend the candidate higher
                candidate_idx = i
            elif closes[i] <= closes[candidate_idx] - threshold:
                # Confirm the high pivot
                pivots.append(Pivot(
                    date=pd.Timestamp(dates[candidate_idx]),
                    price=float(closes[candidate_idx]),
                    type="high",
                ))
                # New candidate is a low starting from current bar
                candidate_idx = i
                candidate_type = "low"

        else:  # candidate_type == "low"
            if closes[i] < closes[candidate_idx]:
                # Extend the candidate lower
                candidate_idx = i
            elif closes[i] >= closes[candidate_idx] + threshold:
                # Confirm the low pivot
                pivots.append(Pivot(
                    date=pd.Timestamp(dates[candidate_idx]),
                    price=float(closes[candidate_idx]),
                    type="low",
                ))
                # New candidate is a high starting from current bar
                candidate_idx = i
                candidate_type = "high"

    return pivots
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py::TestDetectPivots -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add services/elliott_wave_engine.py tests/test_elliott_wave_engine.py
git commit -m "feat: add elliott wave engine data models and detect_pivots"
```

---

### Task 4: Elliott Wave Engine — `find_wave_sequences` and `score_sequence`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_elliott_wave_engine.py`:

```python
from services.elliott_wave_engine import WaveSequence, find_wave_sequences, score_sequence


def _make_pivots(prices: list[float], types: list[str]) -> list[Pivot]:
    """Build a pivot list from prices and types."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return [Pivot(date=d, price=p, type=t) for d, p, t in zip(dates, prices, types)]


class TestFindWaveSequences:
    def test_returns_list_of_wave_sequences(self):
        # 10 alternating pivots: plenty for sliding window
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        assert isinstance(result, list)
        assert all(isinstance(s, WaveSequence) for s in result)

    def test_impulse_has_6_pivots(self):
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        impulses = [s for s in result if s.wave_type == "impulse"]
        assert all(len(s.pivots) == 6 for s in impulses)

    def test_corrective_has_4_pivots(self):
        prices = [100, 120, 105, 135, 110, 150, 125, 160, 140, 175]
        types  = ["low","high","low","high","low","high","low","high","low","high"]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        correctives = [s for s in result if s.wave_type == "corrective"]
        assert all(len(s.pivots) == 4 for s in correctives)

    def test_insufficient_pivots_returns_empty(self):
        pivots = _make_pivots([100, 110, 95], ["low", "high", "low"])
        assert find_wave_sequences(pivots) == []

    def test_search_bound_last_30_pivots(self):
        # Build 40 pivots — sequences should only use the last 30
        prices = [100 + i * 3 if i % 2 == 0 else 100 + i * 3 - 5 for i in range(40)]
        types = ["low" if i % 2 == 0 else "high" for i in range(40)]
        pivots = _make_pivots(prices, types)
        result = find_wave_sequences(pivots)
        # All sequences should use pivots from the last 30
        cutoff_date = pivots[-30].date
        for seq in result:
            assert seq.pivots[0].date >= cutoff_date


class TestScoreSequence:
    def _bullish_impulse(self) -> WaveSequence:
        """Well-formed bullish impulse with near-perfect Fibonacci ratios."""
        # Wave lengths: W1=10, W2 retraces 0.618*10=6.18, W3=16.18 (1.618*W1)
        # W4 retraces 0.382*W3=6.18, W5=10 (=W1)
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100.00, "low"),   # 0
            Pivot(pd.Timestamp("2024-02-01"), 110.00, "high"),  # 1: W1=10
            Pivot(pd.Timestamp("2024-03-01"), 103.82, "low"),   # 2: retraces 0.618 of W1
            Pivot(pd.Timestamp("2024-04-01"), 120.00, "high"),  # 3: W3=16.18 ≈ 1.618*W1
            Pivot(pd.Timestamp("2024-05-01"), 113.82, "low"),   # 4: retraces 0.382 of W3
            Pivot(pd.Timestamp("2024-06-01"), 123.82, "high"),  # 5: W5≈10
        ]
        return WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])

    def test_valid_impulse_is_valid(self):
        seq = self._bullish_impulse()
        is_valid, confidence, hits = score_sequence(seq)
        assert is_valid is True

    def test_valid_impulse_has_positive_confidence(self):
        seq = self._bullish_impulse()
        is_valid, confidence, hits = score_sequence(seq)
        assert confidence > 0

    def test_wave3_shortest_rule_rejects(self):
        # Make wave3 shorter than wave1 and wave5
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),  # W1=20
            Pivot(pd.Timestamp("2024-03-01"), 112, "low"),
            Pivot(pd.Timestamp("2024-04-01"), 117, "high"),  # W3=5 (shortest — INVALID)
            Pivot(pd.Timestamp("2024-05-01"), 110, "low"),
            Pivot(pd.Timestamp("2024-06-01"), 135, "high"),  # W5=25
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_wave4_overlap_rule_rejects(self):
        # Wave4 low dips below wave1 high
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),  # W1 high=120
            Pivot(pd.Timestamp("2024-03-01"), 108, "low"),
            Pivot(pd.Timestamp("2024-04-01"), 140, "high"),
            Pivot(pd.Timestamp("2024-05-01"), 115, "low"),   # W4 low=115 < W1 high=120 — INVALID
            Pivot(pd.Timestamp("2024-06-01"), 150, "high"),
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_wave2_beyond_wave0_rejects(self):
        # Wave2 retraces below wave0
        p = [
            Pivot(pd.Timestamp("2024-01-01"), 100, "low"),   # wave 0 low=100
            Pivot(pd.Timestamp("2024-02-01"), 120, "high"),
            Pivot(pd.Timestamp("2024-03-01"), 95, "low"),    # wave2 low=95 < wave0 low=100 — INVALID
            Pivot(pd.Timestamp("2024-04-01"), 140, "high"),
            Pivot(pd.Timestamp("2024-05-01"), 125, "low"),
            Pivot(pd.Timestamp("2024-06-01"), 160, "high"),
        ]
        seq = WaveSequence(pivots=p, wave_type="impulse", labels=["0","1","2","3","4","5"])
        is_valid, _, _ = score_sequence(seq)
        assert is_valid is False

    def test_fibonacci_hits_format(self):
        seq = self._bullish_impulse()
        _, _, hits = score_sequence(seq)
        # Each hit must contain "pts" and "×" or "retraces"
        for hit in hits:
            assert "pts" in hit

    def test_returns_tuple_of_three(self):
        seq = self._bullish_impulse()
        result = score_sequence(seq)
        assert len(result) == 3
        is_valid, confidence, hits = result
        assert isinstance(is_valid, bool)
        assert isinstance(confidence, int)
        assert isinstance(hits, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py::TestFindWaveSequences tests/test_elliott_wave_engine.py::TestScoreSequence -v 2>&1 | head -30
```

Expected: ImportError — `find_wave_sequences` and `score_sequence` not defined yet.

- [ ] **Step 3: Implement `find_wave_sequences` in `services/elliott_wave_engine.py`**

Append to the file:

```python
def find_wave_sequences(pivots: list[Pivot]) -> list[WaveSequence]:
    """
    Sliding window over last 30 pivots to generate candidate wave sequences.

    Impulse (5-wave): window of 6 consecutive pivots starting on a low pivot.
    Corrective (3-wave): window of 4 consecutive pivots starting on a high pivot.
    """
    if len(pivots) < 4:
        return []

    # Restrict to last 30 pivots
    window = pivots[-30:]
    sequences: list[WaveSequence] = []

    for i in range(len(window) - 5):
        # Impulse candidate: 6 pivots starting on a low
        chunk6 = window[i:i + 6]
        if chunk6[0].type == "low":
            sequences.append(WaveSequence(
                pivots=chunk6,
                wave_type="impulse",
                labels=["0", "1", "2", "3", "4", "5"],
            ))

    for i in range(len(window) - 3):
        # Corrective candidate: 4 pivots starting on a high
        chunk4 = window[i:i + 4]
        if chunk4[0].type == "high":
            sequences.append(WaveSequence(
                pivots=chunk4,
                wave_type="corrective",
                labels=["0", "A", "B", "C"],
            ))

    return sequences
```

- [ ] **Step 4: Implement `score_sequence` in `services/elliott_wave_engine.py`**

Append to the file:

```python
_FIB_TOLERANCE = 0.10  # ±10%


def _fib_close(ratio: float, target: float) -> bool:
    """Return True if ratio is within FIB_TOLERANCE of target."""
    return abs(ratio - target) / max(target, 1e-9) <= _FIB_TOLERANCE


def score_sequence(seq: WaveSequence) -> tuple[bool, int, list[str]]:
    """
    Validate a wave sequence against EW hard rules and score Fibonacci ratios.

    Returns:
        (is_valid, confidence_0_to_100, fibonacci_hits)
    """
    p = seq.pivots

    if seq.wave_type == "impulse":
        # Wave lengths (absolute price moves, close-to-close)
        w1 = abs(p[1].price - p[0].price)
        w2 = abs(p[2].price - p[1].price)
        w3 = abs(p[3].price - p[2].price)
        w4 = abs(p[4].price - p[3].price)
        w5 = abs(p[5].price - p[4].price)

        # ── Hard Rules ──
        # 1. Wave 3 is not the shortest impulse wave
        if w3 <= w1 and w3 <= w5:
            return False, 0, []

        # 2. Wave 4 does not overlap wave 1 (upward impulse: w4 low >= w1 high)
        #    p[0]=0, p[1]=1(high), p[4]=4(low) in upward impulse
        if p[0].type == "low":
            if p[4].price < p[1].price:
                return False, 0, []

        # 3. Wave 2 never retraces beyond wave 0
        #    In upward impulse: w2 low must be >= w0 low
        if p[0].type == "low":
            if p[2].price < p[0].price:
                return False, 0, []

        # ── Fibonacci Scoring ──
        raw_score = 0
        hits: list[str] = []

        if w1 > 0:
            r31 = w3 / w1
            if _fib_close(r31, 1.618):
                raw_score += 20
                hits.append(f"Wave 3 = 1.618 × Wave 1 (+20pts)")
            elif _fib_close(r31, 2.618):
                raw_score += 15
                hits.append(f"Wave 3 = 2.618 × Wave 1 (+15pts)")

        w1_retrace = w2 / w1 if w1 > 0 else 0
        if _fib_close(w1_retrace, 0.618):
            raw_score += 15
            hits.append(f"Wave 2 retraces 0.618 of Wave 1 (+15pts)")

        if w3 > 0:
            w3_retrace = w4 / w3
            if _fib_close(w3_retrace, 0.382):
                raw_score += 15
                hits.append(f"Wave 4 retraces 0.382 of Wave 3 (+15pts)")
            elif _fib_close(w3_retrace, 0.500):
                raw_score += 10
                hits.append(f"Wave 4 retraces 0.500 of Wave 3 (+10pts)")

        if w1 > 0 and _fib_close(w5 / w1, 1.0):
            raw_score += 10
            hits.append(f"Wave 5 = Wave 1 (+10pts)")

        # Normalize: max possible raw score = 85
        confidence = int(round(raw_score / 85 * 100))
        return True, confidence, hits

    else:  # corrective
        # Basic corrective validation: A and C move in same direction
        wa = abs(p[1].price - p[0].price)
        wb = abs(p[2].price - p[1].price)
        wc = abs(p[3].price - p[2].price)

        # C should be in same direction as A
        a_dir = 1 if p[1].price < p[0].price else -1
        c_dir = 1 if p[3].price < p[2].price else -1
        if a_dir != c_dir:
            return False, 0, []

        raw_score = 0
        hits: list[str] = []

        # Common corrective Fibonacci: C = A (equality)
        if wa > 0 and _fib_close(wc / wa, 1.0):
            raw_score += 30
            hits.append(f"Wave C = Wave A (+30pts)")
        if wa > 0 and _fib_close(wc / wa, 0.618):
            raw_score += 20
            hits.append(f"Wave C = 0.618 × Wave A (+20pts)")

        confidence = int(round(min(raw_score / 50 * 100, 100)))
        return True, confidence, hits
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add services/elliott_wave_engine.py tests/test_elliott_wave_engine.py
git commit -m "feat: add find_wave_sequences and score_sequence to elliott wave engine"
```

---

### Task 5: Elliott Wave Engine — `get_best_wave_count`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_elliott_wave_engine.py`:

```python
from services.elliott_wave_engine import BestCount, get_best_wave_count


class TestGetBestWaveCount:
    def _spy_like_series(self) -> pd.Series:
        """Synthetic SPY-like uptrend with clear wave structure over 252 bars."""
        np.random.seed(42)
        # Build a roughly trending series with enough volatility for ATR
        prices = [400.0]
        for i in range(251):
            change = np.random.normal(0.05, 1.5)
            prices.append(max(300.0, prices[-1] + change))
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        return pd.Series(prices, index=dates)

    def test_returns_best_count_or_none(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        assert result is None or isinstance(result, BestCount)

    def test_confidence_in_range(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert 0 <= result.confidence <= 100

    def test_invalidation_level_is_float(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.invalidation_level, float)

    def test_wave_position_is_string(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.wave_position, str)
            assert len(result.wave_position) > 0

    def test_too_few_bars_returns_none(self):
        short_series = pd.Series(
            [100 + i for i in range(30)],
            index=pd.date_range("2024-01-01", periods=30, freq="B")
        )
        result = get_best_wave_count(short_series)
        assert result is None

    def test_fibonacci_hits_is_list_of_strings(self):
        series = self._spy_like_series()
        result = get_best_wave_count(series)
        if result is not None:
            assert isinstance(result.fibonacci_hits, list)
            for hit in result.fibonacci_hits:
                assert isinstance(hit, str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py::TestGetBestWaveCount -v 2>&1 | head -20
```

Expected: ImportError — `get_best_wave_count` not defined yet.

- [ ] **Step 3: Implement `get_best_wave_count` in `services/elliott_wave_engine.py`**

Append to the file:

```python
def get_best_wave_count(series: pd.Series) -> BestCount | None:
    """
    Full pipeline: detect pivots → find sequences → score → return best valid count.

    Returns None if:
    - Fewer than 60 bars in series
    - Fewer than 6 pivots detected
    - No sequence passes hard EW rules
    """
    if len(series) < 60:
        return None

    pivots = detect_pivots(series)
    if len(pivots) < 6:
        return None

    sequences = find_wave_sequences(pivots)
    if not sequences:
        return None

    best_seq: WaveSequence | None = None
    best_confidence: int = -1
    best_hits: list[str] = []

    for seq in sequences:
        is_valid, confidence, hits = score_sequence(seq)
        if is_valid and confidence > best_confidence:
            best_seq = seq
            best_confidence = confidence
            best_hits = hits

    if best_seq is None:
        return None

    # Determine wave position
    most_recent_pivot = pivots[-1]
    last_seq_pivot = best_seq.pivots[-1]
    is_complete = (last_seq_pivot.date == most_recent_pivot.date)

    current_wave_label = best_seq.labels[-1]
    wave_type_name = "Primary Impulse" if best_seq.wave_type == "impulse" else "Primary Correction"

    if is_complete:
        wave_position = f"Wave {current_wave_label} of {wave_type_name} complete"
    else:
        wave_position = f"In Wave {current_wave_label} of {wave_type_name} (incomplete)"

    # Derive invalidation level
    # Impulse: origin is pivot 0 (wave 0); corrective: pivot 0 (wave A start)
    invalidation_level = float(best_seq.pivots[0].price)

    return BestCount(
        sequence=best_seq,
        confidence=best_confidence,
        wave_position=wave_position,
        current_wave_label=current_wave_label,
        invalidation_level=invalidation_level,
        fibonacci_hits=best_hits,
    )
```

- [ ] **Step 4: Run full engine test suite**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/test_elliott_wave_engine.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add services/elliott_wave_engine.py tests/test_elliott_wave_engine.py
git commit -m "feat: complete elliott wave engine with get_best_wave_count"
```

---

### Task 6: Elliott Wave UI module

- [ ] **Step 1: Verify Groq client pattern before implementing**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GROQ_API_KEY set:', bool(os.getenv('GROQ_API_KEY')))"
```

Expected: `GROQ_API_KEY set: True` (confirms env var is available)

- [ ] **Step 2: Create `modules/elliott_wave.py`**

```python
"""
Module: Elliott Wave Analysis (SPY)

Counts primary-degree Elliott Waves on SPY using the elliott_wave_engine,
then generates a Groq LLaMA narrative interpretation.

Layout:
  - SPY candlestick chart with wave overlay
  - Warning banner (if no count or low confidence)
  - Metrics row: Wave Position | Confidence | Invalidation Level
  - Fibonacci Hits list
  - AI Narrative expander
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from services.market_data import fetch_batch_safe
from services.elliott_wave_engine import get_best_wave_count, BestCount
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@st.cache_data(ttl=3600)
def _fetch_spy_data() -> pd.Series | None:
    """Fetch 1 year of SPY daily closes. Returns close price Series or None."""
    snaps = fetch_batch_safe({"SPY": "S&P 500"}, period="1y", interval="1d")
    snap = snaps.get("SPY")
    if snap is None or snap.series is None or snap.series.empty:
        return None
    return snap.series.dropna()


@st.cache_data(ttl=3600)
def _build_groq_narrative(
    wave_position: str,
    current_wave_label: str,
    confidence: int,
    invalidation_level: float,
    fibonacci_hits: tuple[str, ...],
) -> str:
    """
    Call Groq LLaMA to generate a 3-5 sentence Elliott Wave narrative.
    fibonacci_hits is a tuple (not list) so it is hashable for @st.cache_data.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY not set — narrative unavailable."

    fib_text = "\n".join(f"  - {h}" for h in fibonacci_hits) if fibonacci_hits else "  None detected"

    prompt = f"""You are an expert Elliott Wave analyst. Interpret the following automated wave count for SPY (S&P 500 ETF) and write a concise 3-5 sentence market commentary.

Current Wave Count:
- Position: {wave_position}
- Active Wave: {current_wave_label}
- Count Confidence: {confidence}/100
- Invalidation Level: ${invalidation_level:.2f}
- Fibonacci Confirmations:
{fib_text}

Write your commentary covering:
1. What the current wave position means for near-term price action
2. What Elliott Wave theory predicts should happen next
3. The key invalidation level and what a break below/above it would signal

Be direct and specific. Do not hedge excessively. Do not repeat the input data verbatim."""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"_Narrative generation failed: {e}_"


def _make_wave_chart(series: pd.Series, best_count: BestCount | None) -> go.Figure:
    """SPY candlestick with optional wave overlay."""
    fig = go.Figure()

    # Candlestick approximation using close prices (no OHLC in AssetSnapshot.series)
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="SPY",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    if best_count is not None:
        pivots = best_count.sequence.pivots
        labels = best_count.sequence.labels
        wave_color = COLORS["green"] if best_count.sequence.wave_type == "impulse" else COLORS["bloomberg_orange"]

        # Wave lines
        fig.add_trace(go.Scatter(
            x=[p.date for p in pivots],
            y=[p.price for p in pivots],
            mode="lines+markers+text",
            line=dict(color=wave_color, width=2, dash="dot"),
            marker=dict(size=8, color=wave_color),
            text=labels,
            textposition="top center",
            textfont=dict(color=wave_color, size=13, family="JetBrains Mono"),
            name="Wave Count",
            hovertemplate="Wave %{text}<br>$%{y:.2f}<extra></extra>",
        ))

        # Invalidation level
        fig.add_hline(
            y=best_count.invalidation_level,
            line_dash="dash",
            line_color=COLORS["red"],
            line_width=1.5,
            annotation_text=f"Invalidation ${best_count.invalidation_level:.2f}",
            annotation_font_color=COLORS["red"],
            annotation_font_size=11,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title="Price ($)"),
    )
    apply_dark_layout(fig, title="SPY — Elliott Wave Primary Count")
    return fig


def render():
    st.title("Elliott Wave Analysis")
    st.caption("SPY primary-degree wave count · ATR pivot detection · Rule-engine validation · Groq AI narrative")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching SPY data and computing wave count..."):
        series = _fetch_spy_data()

    if series is None:
        st.error("SPY price data unavailable. Please try again later.")
        return

    if len(series) < 60:
        st.warning("Insufficient price history for Elliott Wave analysis (need ≥ 60 bars).")
        return

    best_count = get_best_wave_count(series)

    # ── Chart ──
    fig = _make_wave_chart(series, best_count)
    st.plotly_chart(fig, use_container_width=True)

    # ── Warning Banner ──
    if best_count is None:
        st.info(
            "No clean Elliott Wave count detected in the current data window. "
            "Market may be in a complex correction or transitional phase."
        )
        return

    if best_count.confidence < 40:
        st.warning(
            f"Low-confidence wave count ({best_count.confidence}/100) — "
            "structural rules pass but Fibonacci confirmations are weak. Treat as speculative."
        )

    # ── Metrics Row ──
    m1, m2, m3 = st.columns(3)
    m1.markdown(bloomberg_metric("Wave Position", best_count.wave_position), unsafe_allow_html=True)
    m2.markdown(bloomberg_metric("Confidence", f"{best_count.confidence}/100"), unsafe_allow_html=True)
    m3.markdown(bloomberg_metric("Invalidation", f"${best_count.invalidation_level:.2f}", COLORS["red"]), unsafe_allow_html=True)

    # ── Fibonacci Hits ──
    if best_count.fibonacci_hits:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:8px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Fibonacci Confirmations</div>',
            unsafe_allow_html=True,
        )
        for hit in best_count.fibonacci_hits:
            st.markdown(f"- {hit}")
    else:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_dim"]};">No Fibonacci ratios confirmed</div>',
            unsafe_allow_html=True,
        )

    # ── AI Narrative ──
    with st.expander("Elliott Wave AI Narrative", expanded=False):
        narrative_key = (
            best_count.wave_position,
            best_count.current_wave_label,
            best_count.confidence,
            best_count.invalidation_level,
            tuple(best_count.fibonacci_hits),
        )
        try:
            narrative = _build_groq_narrative(*narrative_key)
            if narrative.startswith("_Narrative generation failed"):
                st.warning("AI narrative unavailable.")
                if st.button("Retry Narrative", key="retry_narrative"):
                    _build_groq_narrative.clear()
                    st.rerun()
            else:
                st.markdown(narrative)
                st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Model: {GROQ_MODEL}")
        except Exception:
            st.warning("AI narrative unavailable.")
            if st.button("Retry Narrative", key="retry_narrative_err"):
                st.cache_data.clear()
                st.rerun()
```

- [ ] **Step 3: Smoke-test the module imports cleanly**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "from modules.elliott_wave import render; print('OK')"
```

Expected: `OK` (no import errors)

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add modules/elliott_wave.py
git commit -m "feat: add Elliott Wave UI module"
```

---

### Task 7: Wire Elliott Wave into `app.py`

- [ ] **Step 1: Add "Elliott Wave" to sidebar radio in `app.py`**

In `app.py`, find the `st.radio` call for top-level navigation (around line 257):

```python
# BEFORE:
top_level = st.radio(
    "Module",
    ["Discovery", "Risk Regime", "Whale Movement", "Stress Signals",
     "Signal Scorecard", "Backtesting", "Trade Journal", "Alerts"],
    key="top_module",
)

# AFTER:
top_level = st.radio(
    "Module",
    ["Discovery", "Risk Regime", "Elliott Wave", "Whale Movement", "Stress Signals",
     "Signal Scorecard", "Backtesting", "Trade Journal", "Alerts"],
    key="top_module",
)
```

- [ ] **Step 2: Add routing block in `app.py`**

After the `elif top_level == "Risk Regime":` block (around line 291), add:

```python
elif top_level == "Elliott Wave":
    from modules.elliott_wave import render
    render()
```

- [ ] **Step 3: Smoke-test app.py parses without errors**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "import ast; ast.parse(open('app.py').read()); print('app.py syntax OK')"
```

Expected: `app.py syntax OK`

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git add app.py
git commit -m "feat: wire Elliott Wave module into app sidebar"
```

---

### Task 8: Integration smoke test

- [ ] **Step 1: Run all tests**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -m pytest tests/ -v
```

Expected: All tests PASS. Note any warnings but do not fail on them.

- [ ] **Step 2: Verify app loads**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
python -c "
import ast
for f in ['app.py', 'modules/risk_regime.py', 'modules/elliott_wave.py', 'services/elliott_wave_engine.py']:
    ast.parse(open(f).read())
    print(f'{f} — OK')
"
```

Expected: All four files print `OK`.

- [ ] **Step 3: Final commit tag**

```bash
cd "C:\Users\16476\claude projects\narrative-investing-tool"
git log --oneline -8
```

Expected: 8 commits from this feature visible in log.
