"""
Tests for features/power_play.py (no database).

Point-in-time: rewriting or deleting every row dated on or after a game's
date (the game itself included) must leave that game's features alone.
"""
import numpy as np
import pandas as pd
import pytest

from features import power_play as P

S = 20242025


def _base(n=6, team="T", season=S, start="2025-01-01", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n, freq="2D")
    return pd.DataFrame({
        "game_id": np.arange(n) + (1000 if team == "T" else 2000),
        "season": season, "date": days, "team": team,
        "toi": 3600.0, "pp_toi": rng.uniform(200, 500, n),
        "pk_toi": rng.uniform(200, 500, n),
        "pp_goals": rng.integers(0, 3, n).astype(float),
        "ppga": rng.integers(0, 3, n).astype(float),
        "pp_xgf": rng.uniform(0, 2, n), "pk_xga": rng.uniform(0, 2, n)})


def test_first_game_of_season_is_nan():
    r = P.compute_pp_rolling(_base())
    first = r[r["game_id"] == 1000]
    assert first.drop(columns=["game_id", "team"]).isna().all(axis=None)


def test_ratio_of_sums_over_prior_games_only():
    b = _base()
    r = P.compute_pp_rolling(b).set_index("game_id")
    # game 1003 uses games 1000-1002
    prior = b[b["game_id"] < 1003]
    assert r.loc[1003, "pp_toi_share_w20"] == pytest.approx(
        prior["pp_toi"].sum() / prior["toi"].sum())
    assert r.loc[1003, "pp_gf_per60_w82"] == pytest.approx(
        3600 * prior["pp_goals"].sum() / prior["pp_toi"].sum())
    assert r.loc[1003, "pk_xga_per60_w20"] == pytest.approx(
        3600 * prior["pk_xga"].sum() / prior["pk_toi"].sum())


def test_window_limits_history():
    b = _base(n=6)
    r = P.compute_pp_rolling(b, windows=(2,)).set_index("game_id")
    prior = b[b["game_id"].isin([1003, 1004])]
    assert r.loc[1005, "pk_ga_per60_w2"] == pytest.approx(
        3600 * prior["ppga"].sum() / prior["pk_toi"].sum())


def test_windows_do_not_cross_seasons():
    b = pd.concat([_base(n=4), _base(n=3, season=S + 10001, start="2025-10-10")
                   .assign(game_id=lambda d: d["game_id"] + 50)])
    r = P.compute_pp_rolling(b).set_index("game_id")
    assert np.isnan(r.loc[1050, "pp_toi_share_w82"])


@pytest.mark.parametrize("target", [1002, 1004])
def test_point_in_time_rewrite_and_delete(target):
    b = _base(n=6)
    ref = P.compute_pp_rolling(b).set_index("game_id").loc[target]
    day = b.loc[b["game_id"] == target, "date"].iloc[0]
    later = b["date"] >= day
    noisy = b.copy()
    noisy.loc[later & (noisy["game_id"] != target), ["pp_toi", "pp_goals", "pk_xga"]] = 9e3
    noisy.loc[noisy["game_id"] == target, ["pp_toi", "ppga", "toi"]] = 1.0
    got = P.compute_pp_rolling(noisy).set_index("game_id").loc[target]
    pd.testing.assert_series_equal(ref, got)
    cut = b[(~later) | (b["game_id"] == target)]
    got2 = P.compute_pp_rolling(cut).set_index("game_id").loc[target]
    pd.testing.assert_series_equal(ref, got2)


def test_zero_pp_time_gives_nan_rate_not_inf():
    b = _base(n=3)
    b["pp_toi"] = 0.0
    r = P.compute_pp_rolling(b).set_index("game_id")
    assert np.isnan(r.loc[1002, "pp_gf_per60_w20"])
    assert r.loc[1002, "pp_toi_share_w20"] == 0.0


def test_pp_diffs_home_minus_away():
    b = pd.concat([_base(n=3, team="T"), _base(n=3, team="U", seed=1)
                   .assign(game_id=lambda d: d["game_id"] - 1000)])
    # make both teams play the same games (ids 1000-1002)
    b["game_id"] = np.tile(np.arange(1000, 1003), 2)
    roll = P.compute_pp_rolling(b)
    games = pd.DataFrame({"game_id": [1002], "home_team": ["T"], "away_team": ["U"]})
    d = P.pp_diffs(games, roll)
    assert list(d.columns) == P.pp_feature_names()
    rt = roll[(roll["team"] == "T") & (roll["game_id"] == 1002)].iloc[0]
    ru = roll[(roll["team"] == "U") & (roll["game_id"] == 1002)].iloc[0]
    assert d["pp_xgf_per60_diff_w20"].iloc[0] == pytest.approx(
        rt["pp_xgf_per60_w20"] - ru["pp_xgf_per60_w20"])
