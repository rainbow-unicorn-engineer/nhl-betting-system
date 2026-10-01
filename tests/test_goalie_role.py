"""
Tests for features/goalie_role.py and the totals experiment variants.

No database: appearances are small hand-built frames. The point-in-time
tests rewrite every row dated on or after a query's date (including the
query's own game) and require every feature of that query to stay put.
"""
import numpy as np
import pandas as pd
import pytest

import models.totals as T
from features import goalie_role as R
from features.goalie_role import ROLE_FEATURES, compute_role_features


def _app(rows):
    """rows: (goalie_id, game_id, team, date, season, is_starter, saves,
    shots_against)."""
    df = pd.DataFrame(rows, columns=["goalie_id", "game_id", "team", "date",
                                     "season", "is_starter", "saves",
                                     "shots_against"])
    df["date"] = pd.to_datetime(df["date"])
    df["toi_seconds"] = np.where(df["is_starter"], 3600, 600)
    return df


S1, S2 = 20202021, 20212022


def _history():
    """Team T: goalie 1 starts games 1-4 and 6, goalie 2 starts game 5,
    all in season S1 on consecutive days from 2021-01-01; game 7 opens S2
    on 2021-10-10 (goalie 2). Opponent goalie 9 for team U each game."""
    rows = []
    starters = [1, 1, 1, 1, 2, 1]
    for k, g in enumerate(starters):
        day = pd.Timestamp("2021-01-01") + pd.Timedelta(days=k)
        rows.append((g, 100 + k, "T", day, S1, True, 27, 30))
        rows.append((9, 100 + k, "U", day, S1, True, 25, 28))
    rows.append((2, 106, "T", pd.Timestamp("2021-10-10"), S2, True, 30, 30))
    rows.append((9, 106, "U", pd.Timestamp("2021-10-10"), S2, True, 20, 20))
    return _app(rows)


def _q(goalie, date, season, team="T", game_id=999):
    return pd.DataFrame({"game_id": [game_id], "team": [team],
                         "goalie_id": [goalie], "date": [pd.Timestamp(date)],
                         "season": [season]})


class TestHandComputed:
    def test_shares_primary_rest(self):
        app = _history()
        # query: goalie 2 on 2021-01-07 (after the 6 S1 games)
        f = compute_role_features(app, _q(2, "2021-01-07", S1)).iloc[0]
        assert f["role_start_share_10"] == pytest.approx(1 / 6)
        assert f["role_start_share_season"] == pytest.approx(1 / 6)
        assert f["role_is_primary"] == 0.0
        assert f["role_rest_days"] == 2.0          # last played 2021-01-05
        assert f["role_started_yesterday"] == 0.0
        g1 = compute_role_features(app, _q(1, "2021-01-07", S1)).iloc[0]
        assert g1["role_is_primary"] == 1.0
        assert g1["role_started_yesterday"] == 1.0
        assert g1["role_rest_days"] == 1.0

    def test_share_10_needs_three_prior_games(self):
        app = _history()
        f = compute_role_features(app, _q(1, "2021-01-03", S1)).iloc[0]
        assert np.isnan(f["role_start_share_10"])      # only 2 prior games
        assert f["role_start_share_season"] == 1.0
        f = compute_role_features(app, _q(1, "2021-01-04", S1)).iloc[0]
        assert f["role_start_share_10"] == 1.0

    def test_season_start_uses_previous_seasons_leader(self):
        app = _history()
        # S2's first game, 2021-10-10: no S2 games yet before it
        f1 = compute_role_features(app, _q(1, "2021-10-10", S2)).iloc[0]
        f2 = compute_role_features(app, _q(2, "2021-10-10", S2)).iloc[0]
        assert np.isnan(f1["role_start_share_season"])
        assert f1["role_is_primary"] == 1.0           # S1 leader (5 starts)
        assert f2["role_is_primary"] == 0.0
        # share_10 crosses seasons: the 6 S1 games
        assert f2["role_start_share_10"] == pytest.approx(1 / 6)
        assert f2["role_rest_days"] == 10.0            # capped
        # no previous season in the data -> NaN
        f0 = compute_role_features(app, _q(1, "2021-01-01", S1)).iloc[0]
        assert np.isnan(f0["role_is_primary"])
        assert np.isnan(f0["role_rest_days"])
        assert f0["role_started_yesterday"] == 0.0

    def test_carry_sv_formula_and_alt(self):
        app = _history()
        d = pd.Timestamp("2021-01-07")
        prior = app[app["date"] < d]
        p = prior["saves"].sum() / prior["shots_against"].sum()
        s_bar = prior["shots_against"].mean()
        k = R.CARRY_K * s_bar
        g1 = prior[prior["goalie_id"] == 1]
        g2 = prior[prior["goalie_id"] == 2]
        c1 = (g1["saves"].sum() + k * p) / (g1["shots_against"].sum() + k)
        c2 = (g2["saves"].sum() + k * p) / (g2["shots_against"].sum() + k)
        f1 = compute_role_features(app, _q(1, d, S1)).iloc[0]
        f2 = compute_role_features(app, _q(2, d, S1)).iloc[0]
        assert f1["role_carry_sv"] == pytest.approx(c1)
        assert f1["role_starter_minus_alt"] == pytest.approx(c1 - c2)
        assert f2["role_starter_minus_alt"] == pytest.approx(c2 - c1)

    def test_no_other_goalie_gives_zero_alt_and_unknown_goalie_league_prior(self):
        app = _history()
        f = compute_role_features(app, _q(1, "2021-01-05", S1)).iloc[0]
        assert f["role_starter_minus_alt"] == 0.0     # only goalie 1 so far
        d = pd.Timestamp("2021-01-07")
        prior = app[app["date"] < d]
        p = prior["saves"].sum() / prior["shots_against"].sum()
        f = compute_role_features(app, _q(77, d, S1)).iloc[0]
        assert f["role_carry_sv"] == pytest.approx(p)
        assert f["role_start_share_season"] == 0.0
        assert f["role_is_primary"] == 0.0

    def test_league_window_is_365_days_and_fallback(self):
        app = _history()
        f = compute_role_features(app, _q(1, "2021-01-01", S1)).iloc[0]
        assert f["role_carry_sv"] == pytest.approx(R.LEAGUE_SV_FALLBACK)
        # 2022-01-06 is 366+ days after every S1 game: only the S2 game counts
        f = compute_role_features(app, _q(5, "2022-01-08", S2)).iloc[0]
        assert f["role_carry_sv"] == pytest.approx(50 / 50)

    def test_carry_sv_uses_at_most_60_appearances(self):
        rows = []
        for k in range(70):
            day = pd.Timestamp("2021-01-01") + pd.Timedelta(days=k)
            # first 10 appearances: 0 saves of 30; last 60: 30 of 30
            saves = 0 if k < 10 else 30
            rows.append((1, k, "T", day, S1, True, saves, 30))
        app = _app(rows)
        d = pd.Timestamp("2021-01-01") + pd.Timedelta(days=70)
        prior = app[app["date"] < d]
        p = prior["saves"].sum() / prior["shots_against"].sum()
        k = R.CARRY_K * 30.0
        want = (60 * 30 + k * p) / (60 * 30 + k)
        f = compute_role_features(app, _q(1, d, S1)).iloc[0]
        assert f["role_carry_sv"] == pytest.approx(want)


class TestPointInTime:
    def _random_app(self, seed=0):
        rng = np.random.default_rng(seed)
        rows, gid = [], 0
        for season, start in ((S1, "2021-01-01"), (S2, "2021-10-10")):
            for k in range(40):
                day = pd.Timestamp(start) + pd.Timedelta(days=int(k * 2))
                for team, pool in (("T", [1, 2, 3]), ("U", [4, 5])):
                    g = int(rng.choice(pool, p=None))
                    sh = int(rng.integers(15, 40))
                    rows.append((g, gid, team, day, season, True,
                                 int(sh - rng.integers(0, 6)), sh))
                gid += 1
        return _app(rows)

    def test_no_feature_uses_its_own_game_or_later(self):
        app = self._random_app()
        # one query per (team-game): the actual starters
        q = app.rename(columns={})[["game_id", "team", "goalie_id",
                                    "date", "season"]].reset_index(drop=True)
        base = compute_role_features(app, q)
        rng = np.random.default_rng(1)
        for i in rng.choice(len(q), 25, replace=False):
            d = q.loc[i, "date"]
            mutated = app.copy()
            later = (mutated["date"] >= d).to_numpy()
            # rewrite own game + same-day + later rows: starters swapped,
            # stats changed
            mutated.loc[later, "saves"] = 0
            mutated.loc[later, "shots_against"] = 50
            mutated.loc[later, "goalie_id"] = np.where(
                mutated.loc[later, "team"] == "T", 2, 5)
            got = compute_role_features(mutated, q.loc[[i]]).iloc[0]
            np.testing.assert_allclose(got.to_numpy(float),
                                       base.loc[i].to_numpy(float),
                                       equal_nan=True)

    def test_dropping_own_and_later_rows_changes_nothing(self):
        app = self._random_app(seed=3)
        q = app[["game_id", "team", "goalie_id", "date",
                 "season"]].reset_index(drop=True)
        base = compute_role_features(app, q)
        for i in (5, 40, 90, 150):
            d = q.loc[i, "date"]
            got = compute_role_features(app[app["date"] < d], q.loc[[i]])
            np.testing.assert_allclose(got.iloc[0].to_numpy(float),
                                       base.loc[i].to_numpy(float),
                                       equal_nan=True)

    def test_defending_frame_maps_starters_to_the_opposite_attack(self):
        app = _history()
        games = pd.DataFrame({"game_id": [999], "date": ["2021-01-07"],
                              "season": [S1], "home_team": ["T"],
                              "away_team": ["U"], "home_starter_id": [2.0],
                              "away_starter_id": [9.0]})
        home_def, away_def = R.defending_role_frame(games, app)
        # the home attack faces the AWAY goalie (9, U's only starter)
        assert home_def.loc[0, "role_start_share_season"] == 1.0
        # the away attack faces the HOME goalie (2, the T backup)
        assert away_def.loc[0, "role_is_primary"] == 0.0


class TestVariants:
    def test_names(self):
        a = T.variant_feature_names("A")
        assert a == T.ATTACK_FEATURES
        b = T.variant_feature_names("B")
        assert not any(n.startswith("goalie_") for n in b)
        assert len(b) == len(a) - sum(n.startswith("goalie_") for n in a)
        c = T.variant_feature_names("C")
        assert c == a + ROLE_FEATURES
        d = T.variant_feature_names("D")
        assert d == b + ROLE_FEATURES
        with pytest.raises(ValueError):
            T.variant_feature_names("E")

    def test_apply_variant_keeps_columns_aligned(self):
        n = 4
        Xh = np.tile(np.arange(len(T.ATTACK_FEATURES), dtype=float), (n, 1))
        role = pd.DataFrame(np.tile(100 + np.arange(len(ROLE_FEATURES),
                                                    dtype=float), (n, 1)),
                            columns=ROLE_FEATURES)
        for v in T.VARIANTS:
            xh, xa, names = T.apply_variant(Xh, Xh + 0.5, role, role + 0.5, v)
            assert xh.shape == (n, len(names))
            for j, name in enumerate(names):
                want = (100 + ROLE_FEATURES.index(name) if name in ROLE_FEATURES
                        else T.ATTACK_FEATURES.index(name))
                assert xh[0, j] == want and xa[0, j] == want + 0.5
        xh, _, _ = T.apply_variant(Xh, Xh, role, role, "A")
        np.testing.assert_array_equal(xh, Xh)

    def test_experiment_variants_cannot_register(self):
        for v in ("B", "C", "D"):
            with pytest.raises(ValueError, match="experiment"):
                T.run_totals(register=True, variant=v)
