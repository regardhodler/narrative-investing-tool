import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the helper directly — avoids triggering Streamlit on import
from modules.quick_run import _classify_signals


def _rc(score, regime=""):
    return {"score": score, "regime": regime, "quadrant": "Goldilocks"}

def _tac(score):
    return {"tactical_score": score, "label": "test", "action_bias": "test"}

def _of(score):
    return {"options_score": score, "label": "test", "action_bias": "test"}


class TestClassifySignals:

    def test_all_bullish_returns_bullish_confirmation(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        assert r["pattern"] == "BULLISH_CONFIRMATION"
        assert r["color"] == "#22c55e"
        assert r["buy_tier"] == "STRONG"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_all_bearish_returns_bearish_confirmation(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(30))
        assert r["pattern"] == "BEARISH_CONFIRMATION"
        assert r["color"] == "#ef4444"
        assert r["buy_tier"] == "NOT A BUYING ENV"
        assert r["short_tier"] == "STRONG"

    def test_regime_up_tac_down_of_up_is_pullback(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(30), _of(70))
        assert r["pattern"] == "PULLBACK_IN_UPTREND"
        assert r["buy_tier"] == "STRONG"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_regime_up_tac_up_of_down_is_options_divergence(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(30))
        assert r["pattern"] == "OPTIONS_FLOW_DIVERGENCE"
        assert r["buy_tier"] == "MODERATE"

    def test_regime_down_tac_up_of_up_is_bear_bounce(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(70), _of(70))
        assert r["pattern"] == "BEAR_MARKET_BOUNCE"
        assert r["buy_tier"] == "SELECTIVE"
        assert r["short_tier"] == "MODERATE"

    def test_regime_down_tac_down_of_up_is_late_cycle_squeeze(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(70))
        assert r["pattern"] == "LATE_CYCLE_SQUEEZE"
        assert r["short_tier"] == "STRONG"

    def test_all_neutral_is_genuine_uncertainty(self):
        r = _classify_signals(_rc(0.0), _tac(50), _of(50))
        assert r["pattern"] == "GENUINE_UNCERTAINTY"
        assert r["color"] == "#475569"
        assert r["buy_tier"] == "NOT A BUYING ENV"
        assert r["short_tier"] == "NOT A SHORTING ENV"

    def test_empty_contexts_return_uncertainty(self):
        r = _classify_signals({}, {}, {})
        assert r["pattern"] == "GENUINE_UNCERTAINTY"

    def test_regime_label_risk_on_overrides_score(self):
        # "Risk-On" in label should classify as bullish even with score=0
        r = _classify_signals(_rc(0.0, "Risk-On — Goldilocks"), _tac(70), _of(70))
        assert r["pattern"] == "BULLISH_CONFIRMATION"

    def test_result_has_all_required_keys(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        for key in ("pattern", "label", "color", "interpretation",
                    "buy_tier", "short_tier", "instruments_buy",
                    "instruments_short", "entry_buy", "entry_short"):
            assert key in r, f"Missing key: {key}"

    def test_instruments_buy_is_list(self):
        r = _classify_signals(_rc(0.5, "Risk-On"), _tac(70), _of(70))
        assert isinstance(r["instruments_buy"], list)
        assert len(r["instruments_buy"]) > 0

    def test_instruments_short_is_list(self):
        r = _classify_signals(_rc(-0.5, "Risk-Off"), _tac(30), _of(30))
        assert isinstance(r["instruments_short"], list)
        assert len(r["instruments_short"]) > 0
