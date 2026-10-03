"""
Tests for features/player_shots.py and models/props_sog.py.

Every number in the toy-league tests is computed by hand in the comments.
The point-in-time test rewrites the outcomes of every row ON a date and
deletes every row AFTER it, and checks that no feature of that date's
rows (or earlier rows) moves. No database is used.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

import features.player_shots as PS
import models.props_sog as P
from features.player_shots import FEATURES, build_player_features, trailing_sums
from models.props_sog import drift_ratio, exposure_baseline, fit_nb_alpha, nb_logpmf, prob_over, train_window_sog60

SEASON = 20202021


# ── Toy league, hand-computable ────────────────────────────────────
#
# g1 d1 A(h) v B   sog A 30 B 25
# g2 d2 B(h) v A   sog B 28 A 32
# g3 d4 A(h) v B   sog A 20 B 22   (no MoneyPuck shot data for g3)
# g4 d5 B(h) v C   sog B 31 C 27
# g5 d6 C(h) v A   sog C 24 A 33
# g6 d7 A(h) v B   sog A 29 B 26
# Player 1 (A, centre): g1 g2 g3 g6 (misses g5); shots 2 0 3 1,
#   TOI 900 1200 600 1500 s; attempts g1 4, g2 0 (covered game), g6 2.
# Player 2 (B, defence): g1 g2 g3 g4 g6; shots 1 1 0 2 1; TOI 1200 each.

def _day(d):
    return pd.Timestamp("2021-01-01") + pd.Timedelta(days=d - 1)


def toy_inputs():
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6],
        "season": SEASON,
        "date": [_day(d) for d in (1, 2, 4, 5, 6, 7)],
        "game_type": 2,
        "home_team": ["A", "B", "A", "B", "C", "A"],
        "away_team": ["B", "A", "B", "C", "A", "B"],
        "home_score": [3, 2, 1, 4, 2, 2],
        "away_score": [1, 4, 2, 1, 3, 1],
        "game_state": "FINAL",
    })
    skaters = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "game_id":   [1, 2, 3, 6, 1, 2, 3, 4, 6],
        "team":      ["A"] * 4 + ["B"] * 5,
        "position":  ["C"] * 4 + ["D"] * 5,
        "toi_seconds": [900, 1200, 600, 1500] + [1200] * 5,
        "shots":     [2, 0, 3, 1, 1, 1, 0, 2, 1],
    })
    attempts = pd.DataFrame({
        "game_id":    [1, 6, 1, 2, 4, 6],
        "shooter_id": [1, 1, 2, 2, 2, 2],
        "attempts":   [4, 2, 1, 3, 2, 1],
    })
    team_games = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "team": ["A", "B", "B", "A", "A", "B", "B", "C", "C", "A", "A", "B"],
        "sog": [30, 25, 28, 32, 20, 22, 31, 27, 24, 33, 29, 26],
    })
    return skaters, games, attempts, team_games


@pytest.fixture(scope="module")
def toy():
    s, g, a, t = toy_inputs()
    return build_player_features(s, g, a, t)


def row(df, player, game):
    r = df[(df["player_id"] == player) & (df["game_id"] == game)]
    assert len(r) == 1
    return r.iloc[0]


def f_league(shots, toi):
    """League F SOG/60 with its pseudo-TOI prior (only player 1 is F)."""
    k = PS.LEAGUE_PRIOR_SECONDS
    return (shots + k * PS.LEAGUE_PRIOR_SOG60["F"] / 3600) / (toi + k) * 3600


class TestTrailingSums:
    def test_hand_computed_window_excludes_same_day(self):
        dates = pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-03",
                                "2021-01-10", "2021-01-03"])
        groups = ["x", "x", "x", "x", "y"]
        out = trailing_sums(dates, groups, {"v": [1, 2, 4, 8, 16]},
                            window_days=7)
        # x on 01-01: nothing earlier -> 0 (same-day rows never count)
        # x on 01-03: 1 + 2 = 3; x on 01-10: window [01-03, 01-10) -> 4
        # y on 01-03: own group empty before -> 0
        np.testing.assert_allclose(out["v"], [0, 0, 3, 4, 0])


class TestToyFeatures:
    def test_counts_and_schedule(self, toy):
        r = row(toy, 1, 6)
        assert r["n_prior"] == 3
        assert r["days_since_last"] == 3              # d4 -> d7
        assert r["team_games_missed"] == 1            # A's g5 on d6
        assert r["b2b"] == 1.0                        # A played d6
        assert row(toy, 1, 2)["b2b"] == 1.0           # A played d1
        assert row(toy, 1, 3)["b2b"] == 0.0           # nothing on d3
        assert row(toy, 2, 4)["b2b"] == 1.0           # B played d4
        assert np.isnan(row(toy, 1, 1)["days_since_last"])
        assert row(toy, 1, 6)["is_home"] == 1.0 and row(toy, 2, 6)["is_home"] == 0.0
        assert row(toy, 2, 6)["is_d"] == 1.0 and row(toy, 1, 6)["is_d"] == 0.0

    def test_player_rates(self, toy):
        r = row(toy, 1, 6)
        # league F before d7: player 1's three games, 5 SOG in 2700 s
        lg = f_league(5, 2700)
        assert r["league_pos_sog60"] == pytest.approx(lg)
        # last-5 SOG/60: 5 x 3600 / 2700 = 6.6667, relative to the league
        assert r["sog60_l5_rel"] * lg == pytest.approx(5 * 3600 / 2700)
        # attempts known in g1 (4, 900 s) and g2 (0, 1200 s); g3 unknown
        assert r["att60_l5_rel"] * lg == pytest.approx(4 * 3600 / 2100)
        assert r["sog60_season_rel"] * lg == pytest.approx(5 * 3600 / 2700)
        # TOI: mean of 900, 1200, 600 = 900 s = 15 min; std = 300 s = 5 min
        assert r["toi_mean_l5"] == pytest.approx(15.0)
        assert r["toi_std_l5"] == pytest.approx(5.0)
        # first appearance has no history at all
        first = row(toy, 1, 1)
        assert np.isnan(first["sog60_l5_rel"]) and np.isnan(first["toi_mean_l5"])
        # g2: att60 from g1 only = 4 x 3600 / 900 = 16
        lg2 = f_league(2, 900)
        assert row(toy, 1, 2)["att60_l5_rel"] * lg2 == pytest.approx(16.0)
        # g3 is not covered by shot data, so g6's att60 ignores its TOI too
        # (checked above); g3's own att is unknown
        assert np.isnan(row(toy, 1, 3)["att"])

    def test_baseline_ingredients(self, toy):
        r = row(toy, 1, 6)
        lg = f_league(5, 2700)
        k = PS.SHRINK_TOI_SECONDS
        assert r["shrunk_sog60"] == pytest.approx(
            (5 + k * lg / 3600) / (2700 + k) * 3600)
        assert r["exp_toi"] == pytest.approx(900.0)
        # fewer than 5 prior games: B0 is the league F SOG per game, with
        # its pseudo-games prior: (5 + 2000 x 1.9) / (3 + 2000)
        assert r["b0_mean"] == pytest.approx(
            (5 + PS.LEAGUE_PRIOR_GAMES * 1.9) / (3 + PS.LEAGUE_PRIOR_GAMES))
        # all skaters before d7: 5 + 4 = 9 SOG in 2700 + 4 x 1200 = 7500 s
        kk = PS.LEAGUE_PRIOR_SECONDS
        assert r["league_sog60"] == pytest.approx(
            (9 + kk * 6.2 / 3600) / (7500 + kk) * 3600)

    def test_team_features(self, toy):
        # league shots per team game before d7: g1..g5 = 272 over 10 rows
        lg = (272 + 100 * 30.0) / (10 + 100)
        p1, p2 = row(toy, 1, 6), row(toy, 2, 6)
        # A's shots for before g6: 30 32 20 33 -> 28.75
        assert p1["team_sf_l10_rel"] * lg == pytest.approx(28.75)
        # B's shots against before g6: 30 32 20 27 -> 27.25
        assert p1["opp_sa_l10_rel"] * lg == pytest.approx(27.25)
        # B's shots for: 25 28 22 31 -> 26.5; A's shots against: 25 28 22 24
        assert p2["team_sf_l10_rel"] * lg == pytest.approx(26.5)
        assert p2["opp_sa_l10_rel"] * lg == pytest.approx(24.75)

    def test_elo_is_pre_game(self, toy):
        # Everyone starts at 1500: the first game's difference is 0 even
        # though A won it
        assert row(toy, 1, 1)["elo_diff"] == 0.0
        assert row(toy, 2, 1)["elo_diff"] == 0.0
        # after g1 (A won at home): A gained exactly what B lost
        a = row(toy, 1, 2)["elo_diff"]
        assert a > 0 and row(toy, 2, 2)["elo_diff"] == pytest.approx(-a)

    def test_b0_after_five_games(self, toy):
        # Player 2 has only 4 prior games at g6 (league fallback), so give
        # him a 6th appearance (g7, d9): 5 prior games, SOG 1 1 0 2 1
        s, g, a, t = toy_inputs()
        g = pd.concat([g, pd.DataFrame({
            "game_id": [7], "season": SEASON, "date": [_day(9)],
            "game_type": 2, "home_team": ["B"], "away_team": ["C"],
            "home_score": [1], "away_score": [0], "game_state": "FINAL"})])
        s = pd.concat([s, pd.DataFrame({
            "player_id": [2], "game_id": [7], "team": ["B"], "position": ["D"],
            "toi_seconds": [1200], "shots": [3]})])
        df = build_player_features(s, g, a, t)
        r = row(df, 2, 7)
        assert r["n_prior"] == 5
        assert r["b0_mean"] == pytest.approx((1 + 1 + 0 + 2 + 1) / 5)


# ── Point-in-time: rewriting / deleting the future changes nothing ──

def random_league(seed=0, seasons=(SEASON,), days=40, n_teams=6, roster=6):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    games, skaters, attempts, team_games = [], [], [], []
    gid = 0
    pid_rate = {}
    for si, season in enumerate(seasons):
        start = pd.Timestamp(f"{2020 + si}-10-10")
        for d in range(days):
            if rng.random() < 0.25:
                continue
            order = rng.permutation(teams)
            for h, a in zip(order[0::2], order[1::2]):
                gid += 1
                date = start + pd.Timedelta(days=d)
                games.append((gid, season, date, 2, h, a,
                              int(rng.integers(0, 6)), int(rng.integers(0, 6)),
                              "FINAL"))
                covered = rng.random() > 0.15
                for team in (h, a):
                    team_games.append((gid, team, int(rng.integers(18, 40))))
                    for j in range(roster):
                        pid = int(team[1:]) * 100 + j
                        if rng.random() < 0.1:
                            continue                     # scratched
                        rate = pid_rate.setdefault(pid, rng.uniform(3, 10))
                        toi = int(rng.integers(600, 1500))
                        shots = int(rng.poisson(rate * toi / 3600))
                        skaters.append((pid, gid, team, "D" if j < 2 else "C",
                                        toi, shots))
                        if covered:
                            att = shots + int(rng.poisson(1.5))
                            if att:
                                attempts.append((gid, pid, att))
                if covered and not any(x[0] == gid for x in attempts):
                    attempts.append((gid, 999999, 1))     # keeps game covered
    return (pd.DataFrame(skaters, columns=["player_id", "game_id", "team",
                                           "position", "toi_seconds", "shots"]),
            pd.DataFrame(games, columns=["game_id", "season", "date",
                                         "game_type", "home_team", "away_team",
                                         "home_score", "away_score",
                                         "game_state"]),
            pd.DataFrame(attempts, columns=["game_id", "shooter_id", "attempts"]),
            pd.DataFrame(team_games, columns=["game_id", "team", "sog"]))


PIT_COLS = FEATURES + ["n_prior", "league_pos_sog60", "league_sog60",
                       "league_pos_sog_pg", "shrunk_sog60", "exp_toi", "b0_mean"]


class TestPointInTime:
    @pytest.mark.parametrize("cut_day", [12, 25, 33])
    def test_rewriting_or_removing_rows_on_or_after_the_date(self, cut_day):
        s, g, a, t = random_league(seed=cut_day)
        full = build_player_features(s, g, a, t)
        dates = sorted(g["date"].unique())
        cut = pd.Timestamp(dates[min(cut_day, len(dates) - 1)])

        rng = np.random.default_rng(99)
        on = g.loc[g["date"] == cut, "game_id"]
        after = g.loc[g["date"] > cut, "game_id"]
        g2 = g[~g["game_id"].isin(after)].copy()
        sel = g2["game_id"].isin(on)
        g2.loc[sel, "home_score"] = rng.integers(0, 9, sel.sum())
        g2.loc[sel, "away_score"] = rng.integers(0, 9, sel.sum())
        s2 = s[~s["game_id"].isin(after)].copy()
        sel = s2["game_id"].isin(on)
        s2.loc[sel, "shots"] = rng.integers(0, 12, sel.sum())
        s2.loc[sel, "toi_seconds"] = rng.integers(60, 2400, sel.sum())
        a2 = a[~a["game_id"].isin(after)].copy()
        sel = a2["game_id"].isin(on)
        a2.loc[sel, "attempts"] = rng.integers(1, 20, sel.sum())
        t2 = t[~t["game_id"].isin(after)].copy()
        sel = t2["game_id"].isin(on)
        t2.loc[sel, "sog"] = rng.integers(5, 60, sel.sum())
        part = build_player_features(s2, g2, a2, t2)

        key = ["player_id", "game_id"]
        lhs = full[full["date"] <= cut].set_index(key).sort_index()
        rhs = part.set_index(key).sort_index()
        assert len(rhs) == len(lhs) and (rhs["date"] == cut).any()
        pd.testing.assert_frame_equal(lhs[PIT_COLS], rhs[PIT_COLS],
                                      check_exact=False, rtol=1e-12)
        # ... and the rewritten outcomes did change on the cut date
        assert not np.array_equal(lhs.loc[lhs["date"] == cut, "sog"],
                                  rhs.loc[rhs["date"] == cut, "sog"])


# ── Negative binomial ──────────────────────────────────────────────

class TestNegativeBinomial:
    def test_matches_scipy(self):
        y = np.array([0, 1, 2, 5, 9])
        mu = np.array([0.5, 2.0, 2.5, 3.0, 4.0])
        for alpha in (0.05, 0.3, 1.2):
            r = 1 / alpha
            np.testing.assert_allclose(nb_logpmf(y, mu, alpha),
                                       nbinom.logpmf(y, r, r / (r + mu)))

    def test_hand_computed(self):
        # alpha = 1 -> r = 1: geometric, P(y) = (1/(1+mu)) (mu/(1+mu))^y
        assert np.exp(nb_logpmf(np.array([2]), np.array([2.0]), 1.0))[0] \
            == pytest.approx((1 / 3) * (2 / 3) ** 2)

    def test_alpha_zero_is_poisson(self):
        y, mu = np.array([0, 3, 7]), np.array([1.0, 2.5, 4.0])
        np.testing.assert_allclose(nb_logpmf(y, mu, 0.0), poisson.logpmf(y, mu))
        np.testing.assert_allclose(nb_logpmf(y, mu, 1e-7),
                                   poisson.logpmf(y, mu), rtol=1e-5)

    def test_pmf_sums_to_one_and_mean(self):
        pmf = P.count_pmf(np.array([2.3]), 0.4, kmax=200)[0]
        assert pmf.sum() == pytest.approx(1.0)
        assert pmf @ np.arange(201) == pytest.approx(2.3)
        assert pmf @ np.arange(201) ** 2 - 2.3 ** 2 == pytest.approx(
            2.3 + 0.4 * 2.3 ** 2)

    def test_prob_over_is_tail_of_pmf(self):
        mu = np.array([0.7, 2.4])
        for alpha in (0.0, 0.25):
            pmf = P.count_pmf(mu, alpha, kmax=100)
            for line in P.LINES:
                k = int(np.floor(line))
                np.testing.assert_allclose(prob_over(mu, alpha, line),
                                           pmf[:, k + 1:].sum(axis=1))

    def test_alpha_mle_recovers_dispersion(self):
        rng = np.random.default_rng(5)
        mu = rng.uniform(0.5, 4.0, 60000)
        r = 1 / 0.3
        y = rng.negative_binomial(r, r / (r + mu))
        assert fit_nb_alpha(y, mu) == pytest.approx(0.3, abs=0.03)

    def test_alpha_mle_on_poisson_data_is_near_zero(self):
        rng = np.random.default_rng(6)
        mu = rng.uniform(0.5, 4.0, 60000)
        assert fit_nb_alpha(rng.poisson(mu), mu) < 0.01

    def test_alpha_is_zero_for_underdispersed_data(self):
        mu = np.full(1000, 2.0)
        y = np.full(1000, 2)          # variance 0 < mean: Poisson wins
        assert fit_nb_alpha(y, mu) == 0.0


# ── Baselines and drift ────────────────────────────────────────────

class TestBaselines:
    def test_exposure_baseline_hand(self):
        # 6.0 SOG/60 x 1080 s (18 min) / 3600 x 0.95 = 1.71
        assert exposure_baseline([6.0], [1080.0], [0.95])[0] == pytest.approx(1.71)

    def test_drift_ratio(self):
        np.testing.assert_allclose(drift_ratio([5.61, 6.39], 6.0),
                                   [0.935, 1.065])
        with pytest.raises(ValueError):
            drift_ratio([6.0], 0.0)
        with pytest.raises(ValueError):
            drift_ratio([6.0], float("nan"))

    def test_train_window_sog60(self):
        # 3 SOG in 1800 s = 6 per 60
        assert train_window_sog60([1, 2, 0], [600, 600, 600]) == pytest.approx(6.0)

    def test_registration_is_disabled(self):
        with pytest.raises(RuntimeError, match="registration is disabled"):
            P.run_props(register=True, frame=pd.DataFrame())

    def test_cli_without_evaluate_runs_nothing(self, monkeypatch):
        monkeypatch.setattr(P, "run_props",
                            lambda **k: pytest.fail("must not run"))
        assert P.main([]) is None


class TestWalkForwardWiring:
    def test_runs_end_to_end_on_a_synthetic_league(self):
        seasons = (20202021, 20212022, 20222023)
        s, g, a, t = random_league(seed=1, seasons=seasons, days=50,
                                   n_teams=8, roster=8)
        frame = build_player_features(s, g, a, t)
        params = dict(P.LGBM_PARAMS, n_estimators=30, min_child_samples=50)
        res = P.run_props(register=False, frame=frame, params=params)
        assert [f["val_season"] for f in res["folds"]] == list(seasons[1:])
        pooled = res["pooled"]
        for k in ("M", "B1", "B0"):
            assert np.isfinite(pooled[f"nll_{k}"])
        oof = res["oof"]
        # scored rows: validation seasons only, >= 5 prior appearances
        assert set(oof["season"]) == set(seasons[1:])
        elig = frame[(frame["n_prior"] >= 5) & frame["season"].isin(seasons[1:])]
        assert len(oof) == len(elig) == pooled["n"]
        assert "by_position" in res["folds"][-1]
        assert set(pooled["gate"]) >= {"passed", "folds_won_vs_B1"}
        # B1 is exactly the clipped exposure baseline, with the drift ratio
        # taken against the fold's own training rows
        f1 = res["folds"][0]
        assert f1["alpha_B1"] >= 0 and f1["alpha_M"] >= 0
        data = P.load_props_dataset(frame)
        first_val = data.loc[data["season"] == seasons[1], "date"].min()
        tr = data[(data["season"] == seasons[0])
                  & (data["date"] < first_val - pd.Timedelta(days=7))]
        mean60 = tr["sog"].sum() * 3600 / tr["toi_seconds"].sum()
        assert f1["train_sog60"] == pytest.approx(mean60, abs=1e-4)
        v = oof[oof["season"] == seasons[1]].merge(
            data, on=["player_id", "game_id"], suffixes=("", "_d"))
        want = np.clip(v["shrunk_sog60"] * v["exp_toi"] / 3600
                       * v["league_sog60"] / mean60, *P.MU_CLIP)
        np.testing.assert_allclose(v["mu_B1"], want, rtol=1e-9)
        np.testing.assert_allclose(v["mu_B0"],
                                   np.clip(v["b0_mean"], *P.MU_CLIP))
