"""
Tests for betting/backtest.py that need no database: the backtest keeps
the locked caps (2% a bet, 10% a day) whatever .env sets for the live
picks, and reports the caps it used. The priced games are synthetic.
"""
from datetime import date

import pandas as pd
import pytest

from betting import backtest, engine


@pytest.fixture
def games(monkeypatch):
    """Six even-money games on one day; the model gives the home side 70%,
    so quarter-Kelly asks for 10% of the bankroll on each."""
    frame = pd.DataFrame({"game_id": range(1, 7), "date": [date(2026, 1, 10)] * 6,
                          "home_ml": [100] * 6, "away_ml": [100] * 6,
                          "home_won": [True, False] * 3})
    monkeypatch.setattr(backtest, "load_priced_games", lambda: frame.copy())
    return pd.DataFrame({"game_id": range(1, 7), "prob_home": [0.70] * 6})


def test_locked_caps_whatever_the_env_says(monkeypatch, games):
    monkeypatch.setattr(engine, "MAX_STAKE_PCT", 0.25)   # as if .env asked for 25%
    r = backtest.run_backtest(games)
    assert (r.max_stake_pct, r.max_daily_pct) == (0.02, 0.10)
    before = backtest.START_BANKROLL
    for bet in r.bets.itertuples():
        assert bet.stake == pytest.approx(0.02 * before, abs=1e-3)
        before = bet.bankroll
    assert r.n_bets == 4                     # a fifth 2% bet would pass 10% a day


def test_other_caps_on_request(games):
    r = backtest.run_backtest(games, max_stake_pct=0.05, max_daily_pct=0.15)
    assert (r.max_stake_pct, r.max_daily_pct) == (0.05, 0.15)
    assert r.bets["stake"].iloc[0] == pytest.approx(5.0)
    assert r.n_bets == 2                     # a third 5% bet would pass 15% a day


def test_evaluate_market_takes_its_own_cap(monkeypatch):
    monkeypatch.setattr(engine, "MAX_STAKE_PCT", 0.25)
    assert engine.evaluate_moneyline(0.70, 100, 100).stake_pct == pytest.approx(0.10)
    assert engine.evaluate_moneyline(0.70, 100, 100,
                                     max_stake_pct=0.02).stake_pct == pytest.approx(0.02)
    assert engine.evaluate_market(0.70, 0.5, 100, 100,
                                  max_stake_pct=0.03).stake_pct == pytest.approx(0.03)
