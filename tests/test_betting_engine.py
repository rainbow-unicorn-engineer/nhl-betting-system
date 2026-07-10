"""
Tests for betting/engine.py — every number here is hand-computed.
"""
import pytest

from betting.engine import (
    BetDecision, EDGE_MIN_ML, KELLY_FRACTION, MAX_STAKE_PCT, decimal_odds,
    evaluate_moneyline, kelly_fraction, no_vig_probs, settle,
)


class TestOddsMath:
    def test_decimal_odds(self):
        assert decimal_odds(-150) == pytest.approx(1 + 100 / 150)
        assert decimal_odds(130) == pytest.approx(2.30)
        assert decimal_odds(-100) == pytest.approx(2.00)

    def test_no_vig_probs(self):
        # -110/-110: symmetric vig -> 50/50 fair
        ph, pa = no_vig_probs(-110, -110)
        assert ph == pytest.approx(0.5) and pa == pytest.approx(0.5)
        # -150/+130: imp = .600/.4348, sum 1.0348
        ph, pa = no_vig_probs(-150, 130)
        assert ph == pytest.approx(0.600 / 1.03478, abs=1e-4)
        assert ph + pa == pytest.approx(1.0)

    def test_kelly_hand_computed(self):
        # p=.55 at +100: b=1, f* = (1*.55 - .45)/1 = .10
        assert kelly_fraction(0.55, 100) == pytest.approx(0.10)
        # p=.60 at -150: b=2/3, f* = (2/3*.6 - .4)/(2/3) = 0.0
        assert kelly_fraction(0.60, -150) == pytest.approx(0.0)
        # negative edge clamps to zero
        assert kelly_fraction(0.40, -110) == 0.0


class TestEvaluate:
    def test_no_bet_when_edge_below_threshold(self):
        # fair 50/50 line, model at 50% + under threshold
        assert evaluate_moneyline(0.5 + EDGE_MIN_ML - 0.001, -110, -110) is None

    def test_home_bet_with_correct_stake(self):
        # -110/-110 fair 0.5; model 58% home -> edge .08
        d = evaluate_moneyline(0.58, -110, -110)
        assert d.side == "HOME" and d.edge == pytest.approx(0.08)
        # kelly: b=10/11, f*=(b*.58-.42)/b = .118; quarter = .0295 -> cap .02
        assert d.kelly == pytest.approx((10 / 11 * 0.58 - 0.42) / (10 / 11))
        assert d.stake_pct == MAX_STAKE_PCT

    def test_away_side_and_uncapped_stake(self):
        # model 55% AWAY at -110/-110 -> edge .05
        d = evaluate_moneyline(0.45, -110, -110)
        assert d.side == "AWAY"
        expected_kelly = (10 / 11 * 0.55 - 0.45) / (10 / 11)
        assert d.stake_pct == pytest.approx(expected_kelly * KELLY_FRACTION)
        assert d.stake_pct < MAX_STAKE_PCT

    def test_edge_vs_novig_but_negative_ev_vs_vig_is_skipped(self):
        # Heavy vig: -125/-125 -> fair .5/.5. Model .53: edge .03 >= .025,
        # but Kelly at -125 with p=.53: b=.8, f*=(.8*.53-.47)/.8 < 0 -> skip
        assert evaluate_moneyline(0.53, -125, -125) is None


class TestSettle:
    def test_payouts(self):
        d = BetDecision("HOME", -150, 0.62, 0.58, 0.04, 0.05, 0.0125)
        assert settle(d, home_won=True, stake=3.0) == pytest.approx(2.0)
        assert settle(d, home_won=False, stake=3.0) == -3.0
        a = BetDecision("AWAY", 130, 0.48, 0.43, 0.05, 0.04, 0.01)
        assert settle(a, home_won=False, stake=2.0) == pytest.approx(2.6)
        assert settle(a, home_won=True, stake=2.0) == -2.0
