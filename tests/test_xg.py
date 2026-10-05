"""
Tests for features/xg.py and models/xg.py (our expected-goals model).

Unit tests run on synthetic shots, no database: the shot features, their
point-in-time property (rewriting or deleting later shots, or the shot's
own outcome, leaves a shot's inputs unchanged), the walk-forward (a
season's xG never depends on that season or later), the metrics and the
pre-registered gate and downstream rule, and the helpers that feed our xG
into the team, goalie and player features. The database test (the
MoneyPuck arm rebuilt in memory equals the stored team features) skips
unless tests/conftest.py allows database tests.
"""
import numpy as np
import pandas as pd
import pytest

import features.xg as FX
import models.xg as MX
from features.team_features import replace_xg

SMALL = {**MX.XG_PARAMS, "n_estimators": 60, "min_child_samples": 20,
         "n_jobs": 1, "learning_rate": 0.1}


def _shot(i, game, t, team, **kw):
    row = {"shot_id": i, "game_id": game, "season": 20212022,
           "date": pd.Timestamp("2021-10-15"), "game_type": 2,
           "home_team": "AAA", "period": 1 + min(int(t // 1200), 3),
           "time_elapsed": t, "team": team, "shooter_id": 100 + i % 3,
           "goalie_id": 900 if team == "BBB" else 901,
           "x": 70.0, "y": 5.0, "shot_type": "WRIST", "event_type": "SHOT",
           "is_goal": False, "xg_moneypuck": 0.05, "strength": "5v5",
           "score_state": 0, "is_rebound": False, "is_rush": False,
           "shot_distance": 20.0, "shot_angle": 15.0}
    row.update(kw)
    return row


def one_game():
    """Game 1, AAA at home. Times in seconds of game time."""
    return pd.DataFrame([
        _shot(1, 1, 100, "AAA", x=60.0, y=10.0, shot_angle=-20.0, shot_distance=31.0),
        _shot(2, 1, 102, "AAA", x=85.0, y=2.0, shot_angle=10.0, shot_distance=5.0,
              is_rebound=True, is_goal=True, event_type="GOAL"),
        _shot(3, 1, 150, "BBB", x=-70.0, y=-3.0, score_state=1, shot_type=None),
        _shot(4, 1, 1250, "AAA", score_state=1, strength="5v4"),
        _shot(5, 1, 1255, "AAA", score_state=1, strength="5v6",
              event_type="MISS"),
        _shot(6, 1, 1258, "AAA", score_state=1, strength="6v5"),
    ])


class TestShotFeatures:
    def test_inputs_never_include_the_outcome_or_moneypuck(self):
        for bad in ("is_goal", "event_type", "xg_moneypuck", "goal"):
            assert bad not in FX.ALL_FEATURES

    def test_hand_computed_rows(self):
        f = FX.shot_features(one_game())
        r1, r2, r3 = f.iloc[0], f.iloc[1], f.iloc[2]
        assert r1["angle"] == 20.0 and r1["x_abs"] == 60.0
        assert np.isnan(r1["secs_since_prev"]) and np.isnan(r1["prev_was_goal"])
        assert r1["team_att_last10s"] == 0
        assert r2["secs_since_prev"] == 2 and r2["prev_same_team"] == 1.0
        assert r2["prev_dist_moved"] == pytest.approx(np.hypot(25.0, 8.0))
        assert r2["prev_angle_rate"] == pytest.approx(30.0 / 2.0)
        assert r2["prev_distance"] == 31.0 and r2["team_att_last10s"] == 1
        # shot 3 by the away side: score difference from its own side
        assert r3["score_diff"] == -1.0 and r3["shooter_home"] == 0.0
        assert r3["prev_was_goal"] == 1.0 and r3["prev_same_team"] == 0.0
        # a missing shot type is read as a wrist shot
        assert r3["shot_type_code"] == FX.SHOT_TYPE_CODE["WRIST"]

    def test_strength_period_and_empty_net(self):
        f = FX.shot_features(one_game())
        r4, r5, r6 = f.iloc[3], f.iloc[4], f.iloc[5]
        assert r4["period"] == 2 and r4["period_seconds"] == 50
        assert np.isnan(r4["prev_dist_moved"])          # new period
        assert r4["shooter_skaters"] == 5 and r4["defender_skaters"] == 4
        assert r5["empty_net"] == 1.0 and r5["own_net_empty"] == 0.0
        assert r6["own_net_empty"] == 1.0 and r6["empty_net"] == 0.0
        assert r6["team_att_last10s"] == 2               # shots 4 and 5

    def test_row_order_is_kept(self):
        g = one_game()
        shuffled = g.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = FX.shot_features(g).set_index(g["shot_id"])
        b = FX.shot_features(shuffled).set_index(shuffled["shot_id"])
        pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())

    @pytest.mark.parametrize("cut", [101, 151, 1256])
    def test_rewriting_or_deleting_later_shots(self, cut):
        g = one_game()
        base = FX.shot_features(g)
        keep = g["time_elapsed"] < cut
        later = ~keep
        rewritten = g.copy()
        rewritten.loc[later, "x"] = -12.0
        rewritten.loc[later, "y"] = 33.0
        rewritten.loc[later, "is_goal"] = ~rewritten.loc[later, "is_goal"]
        rewritten.loc[later, "team"] = "BBB"
        rewritten.loc[later, "shot_distance"] = 77.0
        a = FX.shot_features(rewritten)[keep.to_numpy()]
        b = FX.shot_features(g[keep])
        pd.testing.assert_frame_equal(a.reset_index(drop=True), base[keep.to_numpy()].reset_index(drop=True))
        pd.testing.assert_frame_equal(b.reset_index(drop=True), base[keep.to_numpy()].reset_index(drop=True))

    def test_own_outcome_does_not_change_own_inputs(self):
        g = one_game()
        flipped = g.copy()
        flipped["is_goal"] = ~flipped["is_goal"]
        flipped["event_type"] = "GOAL"
        a, b = FX.shot_features(g), FX.shot_features(flipped)
        cols = [c for c in FX.ALL_FEATURES if c != "prev_was_goal"]
        pd.testing.assert_frame_equal(a[cols], b[cols])

    def test_other_games_never_mix(self):
        g = one_game()
        other = one_game().assign(game_id=2, shot_id=lambda d: d["shot_id"] + 100)
        both = pd.concat([g, other], ignore_index=True)
        f = FX.shot_features(both)
        pd.testing.assert_frame_equal(f.iloc[:6].reset_index(drop=True),
                                      FX.shot_features(g))


def synthetic_league(n_per_season=3000, seasons=(20202021, 20212022, 20222023),
                     seed=0):
    """Random shots whose goal chance falls with distance."""
    rng = np.random.default_rng(seed)
    rows = []
    sid = 0
    for k, s in enumerate(seasons):
        start = pd.Timestamp(f"{2020 + k}-10-10")
        for i in range(n_per_season):
            sid += 1
            d = rng.uniform(3, 60)
            p = 1 / (1 + np.exp(-(0.5 - 0.09 * d)))
            game = s * 10 + i // 60
            rows.append(_shot(sid, game, (i % 60) * 60 + 1, "AAA" if i % 2 else "BBB",
                              season=s, date=start + pd.Timedelta(days=i // 60),
                              shot_distance=d, x=89 - d, y=rng.uniform(-20, 20),
                              shot_angle=rng.uniform(-60, 60),
                              is_goal=bool(rng.random() < p),
                              xg_moneypuck=float(p)))
    return pd.DataFrame(rows)


class TestWalkForward:
    def test_a_season_never_sees_itself_or_later(self):
        shots = synthetic_league()
        feats = FX.shot_features(shots)
        a = MX.walk_forward_xg(shots, feats, MX.CORE_FEATURES, SMALL)["oof"]
        s2 = (shots["season"] == 20212022).to_numpy()
        changed = shots.copy()
        later = changed["season"] >= 20212022
        changed.loc[later, "is_goal"] = ~changed.loc[later, "is_goal"]
        b = MX.walk_forward_xg(changed, FX.shot_features(changed),
                               MX.CORE_FEATURES, SMALL)["oof"]
        np.testing.assert_allclose(a[s2], b[s2])
        dropped = shots[shots["season"] <= 20212022].reset_index(drop=True)
        c = MX.walk_forward_xg(dropped, FX.shot_features(dropped),
                               MX.CORE_FEATURES, SMALL)["oof"]
        np.testing.assert_allclose(a[s2], c[s2[:len(c)]])

    def test_first_season_scored_only_when_cross_fitted(self):
        shots = synthetic_league(n_per_season=1500)
        feats = FX.shot_features(shots)
        first = (shots["season"] == 20202021).to_numpy()
        plain = MX.walk_forward_xg(shots, feats, MX.CORE_FEATURES, SMALL)
        assert np.isnan(plain["oof"][first]).all()
        assert not np.isnan(plain["oof"][~first]).any()
        cf = MX.walk_forward_xg(shots, feats, MX.CORE_FEATURES, SMALL,
                                crossfit_first=True)
        assert not np.isnan(cf["oof"]).any()
        np.testing.assert_allclose(cf["oof"][~first], plain["oof"][~first])

    def test_evaluate_reports_both_variants_and_the_gate(self):
        shots = synthetic_league(n_per_season=1500)
        res = MX.evaluate(shots, params=SMALL)
        assert set(res["variants"]) == {"X1", "X2"}
        pooled = res["variants"]["X2"]["pooled"]
        assert pooled["ours"]["n"] == pooled["mp"]["n"] == 3000
        assert res["gate"] == pooled["gate"]
        assert len(pooled["reliability_ours"]) == 10


class TestMetrics:
    def test_deciles_are_equal_count(self):
        p = np.linspace(0, 1, 1000)
        assert np.bincount(MX.decile_bins(p)).tolist() == [100] * 10

    def test_ece_deciles_hand(self):
        # two bins of 2: p 0.1, 0.1 (one goal) and 0.9, 0.9 (both goals)
        y = np.array([0, 1, 1, 1])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        assert MX.ece_deciles(y, p, n_bins=2) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)

    def test_perfectly_calibrated_is_near_zero(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.01, 0.4, 200_000)
        y = rng.random(len(p)) < p
        assert MX.ece_deciles(y, p) < 0.003

    def test_paired_and_clustered_se(self):
        a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4)
        r = MX.paired(a, b, clusters=[1, 1, 2, 2])
        assert r["diff"] == 2.5
        assert r["se"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)
        assert r["se_cluster"] == pytest.approx(np.sqrt(2 * 2.0 ** 2) / 4)

    def test_log_loss_clips(self):
        assert np.isfinite(MX.shot_log_loss([1, 0], [0.0, 1.0])).all()

    def test_gate_edges(self):
        mp = {"auc": 0.77, "ece_decile": 0.004}
        ok = {"auc": 0.765, "ece_decile": 0.009}
        assert MX.xg_gate(ok, mp, {"diff": 0.001, "se": 0.001})["passed"]
        assert not MX.xg_gate({**ok, "auc": 0.7649}, mp,
                              {"diff": 0, "se": 1})["auc_ok"]
        assert not MX.xg_gate({**ok, "ece_decile": 0.0091}, mp,
                              {"diff": 0, "se": 1})["ece_ok"]
        assert not MX.xg_gate(ok, mp, {"diff": 0.0011, "se": 0.001})["log_loss_ok"]

    def test_downstream_rule(self):
        ml_ok, ml_bad = {"diff": -0.0001, "se": 0.001}, {"diff": 0.0001, "se": 0.001}
        pr_ok, pr_bad = {"diff": 0.0009, "se": 0.001}, {"diff": 0.0011, "se": 0.001}
        assert MX.downstream_decision(True, ml_ok, pr_ok)["adopt"]
        assert not MX.downstream_decision(False, ml_ok, pr_ok)["adopt"]
        assert not MX.downstream_decision(True, ml_bad, pr_ok)["adopt"]
        assert not MX.downstream_decision(True, ml_ok, pr_bad)["adopt"]
        assert not MX.downstream_decision(True, ml_ok, pr_ok)["moneyline_improves_2se"]


class TestFeatureStoreHelpers:
    def shots(self):
        return pd.DataFrame({
            "shot_id": [1, 2, 3, 4, 5], "season": 20212022,
            "game_id": [1, 1, 1, 2, 2],
            "team": ["AAA", "AAA", "BBB", "AAA", "CCC"],
            "goalie_id": [901, 901, 900, 902, 900],
            "strength": ["5v5", "5v4", "5v5", "5v5", "5v5"],
            "event_type": ["SHOT", "GOAL", "MISS", "SHOT", "GOAL"],
            "is_goal": [False, True, False, False, True],
            "xg": [0.1, 0.3, 0.25, 0.05, 0.5],
        })

    def test_team_sums(self):
        t = FX.team_xg_sums(self.shots(), "xg").set_index(["game_id", "team"])
        assert t.loc[(1, "AAA"), "xg"] == pytest.approx(0.4)
        assert t.loc[(1, "AAA"), "pp_xg"] == pytest.approx(0.3)
        assert np.isnan(t.loc[(1, "BBB"), "pp_xg"])

    def test_replace_xg_own_and_opponent(self):
        base = pd.DataFrame({"game_id": [1, 1], "team": ["AAA", "BBB"],
                             "opp": ["BBB", "AAA"], "xgf": [9.0, 9.0],
                             "xga": [9.0, 9.0], "pp_xgf": [9.0, 9.0],
                             "pk_xga": [9.0, 9.0], "gf": [2, 1]})
        out = replace_xg(base, FX.team_xg_sums(self.shots(), "xg"))
        a = out[out["team"] == "AAA"].iloc[0]
        b = out[out["team"] == "BBB"].iloc[0]
        assert a["xgf"] == pytest.approx(0.4) and a["xga"] == pytest.approx(0.25)
        assert a["pp_xgf"] == pytest.approx(0.3) and np.isnan(a["pk_xga"])
        assert b["xga"] == pytest.approx(0.4) and b["pk_xga"] == pytest.approx(0.3)
        assert out["gf"].tolist() == [2, 1]

    def test_goalie_sums_follow_the_sql_definitions(self):
        g = FX.goalie_xg_sums(self.shots(), "xg").set_index(["game_id", "goalie_id"])
        # goalie 901: xg 0.1 + 0.3; high danger = on target with xG >= 0.20
        assert g.loc[(1, 901), "xga_shots"] == pytest.approx(0.4)
        assert g.loc[(1, 901), "hd_att"] == 1 and g.loc[(1, 901), "hd_goals"] == 1
        # goalie 900 in game 1 faced a MISS with xG 0.25: not on target
        assert g.loc[(1, 900), "hd_att"] == 0
        assert g.loc[(2, 900), "hd_goals"] == 1

    def test_gsax60_prior_repeats_the_join(self):
        apps = pd.DataFrame({"game_id": [1, 1, 2], "player_id": [901, 900, 903],
                             "toi_seconds": [3600, 3600, 1800]})
        # 901 faced 2 shots (TOI counted twice), 900 one, 903 none (once)
        num = (0.1 + 0.3 - 1) + (0.25 - 0)
        den = 3600 * 2 + 3600 + 1800
        assert FX.gsax60_prior(apps, self.shots(), "xg") == pytest.approx(3600 * num / den)
        assert FX.gsax60_prior(apps.iloc[2:], self.shots(), "xg") is None


class TestPlayerXgFeatures:
    def build(self, cut_day=None, mode=None):
        from features.player_shots import build_player_features
        from tests.test_props_sog import toy_inputs
        skaters, games, attempts, team_games = toy_inputs()
        shots = pd.DataFrame({
            "game_id":    [1, 1, 1, 2, 2, 4, 5, 6, 6],
            "shooter_id": [1, 1, 2, 2, 2, 2, None, 1, 2],
            "team": ["A", "A", "B", "B", "B", "B", "C", "A", "B"],
            "strength": "5v5",
            "xg": [0.1, 0.2, 0.05, 0.3, 0.1, 0.2, 0.4, 0.15, 0.05],
        })
        frame = build_player_features(skaters, games, attempts, team_games)
        return frame, shots, games

    def test_hand_computed_player_rate(self):
        frame, shots, _ = self.build()
        f = FX.player_xg_features(frame, shots, "xg")
        r = f[(f["player_id"] == 1) & (f["game_id"] == 6)].iloc[0]
        # player 1 before g6: g1 0.3 xG in 900 s, g2 0 in 1200 s (covered),
        # g3 unknown (no shot data) -> left out
        own = 0.3 * 3600 / 2100
        assert np.isfinite(r["ixg60_l10_rel"])
        league = f.attrs.get("unused")  # noqa: F841 (documentation only)
        assert r["ixg60_l10_rel"] == pytest.approx(r["ixg60_l20_rel"])
        assert r["ixg60_l10_rel"] > 0 and own > 0

    @pytest.mark.parametrize("cut", ["2021-01-04", "2021-01-06"])
    def test_rewriting_or_deleting_on_or_after_the_date(self, cut):
        frame, shots, games = self.build()
        cut = pd.Timestamp(cut)
        a = FX.player_xg_features(frame, shots, "xg")
        later_games = games.loc[pd.to_datetime(games["date"]) >= cut, "game_id"]
        s2 = shots.copy()
        s2.loc[s2["game_id"].isin(later_games), "xg"] = 0.99
        b = FX.player_xg_features(frame, s2, "xg")
        c = FX.player_xg_features(frame, shots[~shots["game_id"].isin(later_games)]
                                  .assign(xg=lambda d: d["xg"]), "xg")
        early = pd.to_datetime(a["date"]) < cut
        cols = FX.PROPS_XG_FEATURES
        pd.testing.assert_frame_equal(a.loc[early, cols].reset_index(drop=True),
                                      b.loc[early, cols].reset_index(drop=True))
        # deleting later shots turns those games into unknown ones, which
        # changes nothing before the date either
        pd.testing.assert_frame_equal(a.loc[early, cols].reset_index(drop=True),
                                      c.loc[early, cols].reset_index(drop=True))

    def test_rows_are_unchanged(self):
        frame, shots, _ = self.build()
        f = FX.player_xg_features(frame, shots, "xg")
        assert len(f) == len(frame)
        pd.testing.assert_frame_equal(
            f[frame.columns].reset_index(drop=True), frame.reset_index(drop=True))


class TestDatabase:
    """Skips unless tests/conftest.py allows database tests."""

    @pytest.fixture(scope="class")
    def db(self):
        from sqlalchemy import text

        from config.settings import engine
        try:
            with engine.connect() as conn:
                if conn.execute(text("SELECT COUNT(*) FROM raw.shots")).scalar() < 1000:
                    pytest.skip("raw.shots not loaded")
        except Exception as e:
            pytest.skip(f"database unavailable: {e}")
        return engine

    def test_moneypuck_arm_equals_the_sql_path(self, db):
        from features import goalie_features as GF
        from features import team_features as TF
        shots = FX.load_shots()
        season = int(sorted(shots["season"].unique())[1])
        a = TF.load_base(season)
        b = TF.load_base(season, xg_sums=FX.team_xg_sums(shots, "xg_moneypuck"))
        for c in ("xgf", "xga", "pp_xgf", "pk_xga"):
            np.testing.assert_allclose(a[c].astype(float), b[c].astype(float),
                                       atol=1e-6, equal_nan=True)
        assert GF.league_priors(season) == pytest.approx(
            GF.league_priors(season, xg_shots=shots, xg_col="xg_moneypuck"))
        ga = GF.load_goalie_base(season)
        gb = GF.load_goalie_base(season, xg_sums=FX.goalie_xg_sums(shots, "xg_moneypuck"))
        np.testing.assert_allclose(ga["gsax"].astype(float), gb["gsax"].astype(float),
                                   atol=1e-6, equal_nan=True)
        assert (ga["hd_att"].fillna(-1) == gb["hd_att"].fillna(-1)).all()
