"""
Tests for betting/engine.py — every number here is hand-computed.
"""
import pytest

from betting.engine import (
    BetDecision, DEFAULT_MAX_BETS_PER_GAME, DEFAULT_MAX_DAILY_PCT,
    DEFAULT_MAX_GAME_STAKE_PCT, DEFAULT_MAX_STAKE_PCT, EDGE_MIN_ML,
    KELLY_FRACTION, MAX_STAKE_PCT, cap_warnings, decimal_odds,
    evaluate_moneyline, game_cap_reason, kelly_fraction, no_vig_probs, settle,
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


class TestGameCaps:
    """Per-game limits (§7: max 3 correlated bets per game), any market.
    Bankroll 1000: 4% = 40 across every bet on one game."""

    def test_locked_defaults(self):
        assert DEFAULT_MAX_BETS_PER_GAME == 3
        assert DEFAULT_MAX_GAME_STAKE_PCT == pytest.approx(0.04)
        assert DEFAULT_MAX_STAKE_PCT == pytest.approx(0.02)
        assert DEFAULT_MAX_DAILY_PCT == pytest.approx(0.10)

    def test_fits(self):
        assert game_cap_reason(10.0, 0, 0.0, 1000) is None
        # two bets and 20 already on the game: a third of 20 lands exactly
        # on the 40 limit and still fits
        assert game_cap_reason(20.0, 2, 20.0, 1000) is None

    def test_fourth_bet_is_refused(self):
        why = game_cap_reason(1.0, 3, 3.0, 1000)
        assert why is not None and "max 3" in why

    def test_stake_past_the_game_limit_is_refused(self):
        why = game_cap_reason(20.01, 1, 20.0, 1000)
        assert why is not None and "40.00" in why

    def test_limits_are_parameters(self):
        assert game_cap_reason(5.0, 1, 0.0, 1000, max_bets=1) is not None
        assert game_cap_reason(25.0, 0, 0.0, 1000,
                               max_game_stake_pct=0.02) is not None
        assert game_cap_reason(25.0, 0, 0.0, 1000,
                               max_game_stake_pct=0.05) is None


class TestLimitSettings:
    """MAX_STAKE_PCT, MAX_DAILY_PCT and MAX_GAME_STAKE_PCT from the
    environment (a fresh interpreter each, since they are read at import):
    blank = the locked default; a share of bankroll above 0 and at most 1
    is used; anything else logs an error naming the setting and falls
    back, so a typo can't break the import (and the daily chain)."""

    NAMES = ("MAX_STAKE_PCT", "MAX_DAILY_PCT", "MAX_GAME_STAKE_PCT")

    def _read(self, **env):
        import os
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).parent.parent
        out = subprocess.run(
            [sys.executable, "-c",
             "import betting.engine as e; "
             "print(e.MAX_STAKE_PCT, e.MAX_DAILY_PCT, e.MAX_GAME_STAKE_PCT)"],
            cwd=root, capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": str(root), **env})
        vals = out.stdout.strip().splitlines()[-1].split()
        return tuple(float(v) for v in vals), out.stderr

    def test_blank_means_the_defaults(self):
        vals, err = self._read(**{n: "" for n in self.NAMES})
        assert vals == (0.02, 0.10, 0.04) and "is not" not in err

    def test_valid_values(self):
        vals, err = self._read(MAX_STAKE_PCT=" 0.25 ", MAX_DAILY_PCT="1",
                               MAX_GAME_STAKE_PCT="0.5")
        assert vals == (0.25, 1.0, 0.5) and "is not" not in err

    @pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "nan", "inf", "2%",
                                     "abc"])
    def test_bad_values_fall_back_with_an_error(self, bad):
        vals, err = self._read(**{n: bad for n in self.NAMES})
        assert vals == (0.02, 0.10, 0.04)
        for n in self.NAMES:
            assert f"{n}={bad!r} is not a fraction of bankroll" in err

    def test_contradicting_limits_warn(self):
        _, err = self._read(MAX_STAKE_PCT="0.3", MAX_DAILY_PCT="0.2",
                            MAX_GAME_STAKE_PCT="0.5")
        assert "MAX_STAKE_PCT (30%) is above MAX_DAILY_PCT (20%)" in err


class TestCapWarnings:
    def test_consistent_limits(self):
        assert cap_warnings(0.02, 0.10, 0.04) == []
        assert cap_warnings(0.25, 1.0, 0.5) == []

    def test_per_bet_above_day_and_game(self):
        w = cap_warnings(0.05, 0.04, 0.03)
        assert len(w) == 2
        assert "MAX_DAILY_PCT (4%)" in w[0] and "skipped" in w[0]
        assert "MAX_GAME_STAKE_PCT (3%)" in w[1]

    def test_fractional_limits_are_not_rounded_away(self):
        from betting.engine import game_cap_reason, pct_text
        assert [pct_text(x) for x in (0.02, 0.025, 0.1, 0.07, 1 / 3, 1.0)] == [
            "2%", "2.5%", "10%", "7%", "33.33%", "100%"]
        w = cap_warnings(0.025, 0.02, 0.015)
        assert "MAX_STAKE_PCT (2.5%)" in w[0] and "MAX_GAME_STAKE_PCT (1.5%)" in w[1]
        why = game_cap_reason(30, 0, 0, 1000, max_bets=3, max_game_stake_pct=0.025)
        assert "(2.5% of bankroll)" in why
