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
