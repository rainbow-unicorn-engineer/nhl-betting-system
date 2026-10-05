"""
Tests for the props shots-on-goal v3 additions: the power-play (PP →
his team has an extra skater after an opponent's penalty) and usage
features, the fixed exposure baseline B3, the variants and the adoption
rule (features/player_shots.py, models/props_sog.py). No database.

Every toy number is worked out by hand in the comments. The point-in-
time test rewrites every input row ON a date (including PP / PK time,
faceoffs and shot strengths) and deletes every row AFTER it, and checks
that no v3 feature of that date's rows (or earlier rows) moves.
"""
import numpy as np
import pandas as pd
import pytest

import features.player_shots as PS
import models.props_sog as P
from features.player_shots import (FEATURES, FEATURES_PP, FEATURES_USAGE,
                                   build_player_features, decayed_sums,
                                   is_pp_strength, season_env)

SEASON = 20202021


def _day(d):
    return pd.Timestamp("2021-01-01") + pd.Timedelta(days=d - 1)


# ── Toy league ─────────────────────────────────────────────────────
#
# g1 d1 A(h) v B, g2 d2 B(h) v A, g3 d4 A(h) v B.
# Team A: player 1 (C), player 2 (D). Team B: player 3 (C).
#            toi   pp   sh  fow fol shots
# g1  p1    1000  100   50   5   3   2      strengths: 5v4 x1, 5v5 x1
#     p2    1200   50  100   0   0   1      5v5 x1
#     p3     900    0  150   3   5   1      4v5 x1
# g2  p1    1100  200    0   6   6   3      5v5 x2, 6v5 x1
#     p2    1300  (stats never filled: unknown, not zero)  0
#     p3    1000   60   30   4   2   2      5v4 x2
# g3  p1    1200   90   60   7   1   1
#     p2    1250   40   80   0   0   2
#     p3     950   30   20   2   2   0

def toy_inputs():
    games = pd.DataFrame({
        "game_id": [1, 2, 3], "season": SEASON,
        "date": [_day(1), _day(2), _day(4)], "game_type": 2,
        "home_team": ["A", "B", "A"], "away_team": ["B", "A", "B"],
        "home_score": [3, 2, 1], "away_score": [1, 4, 2],
        "game_state": "FINAL"})
    skaters = pd.DataFrame({
        "player_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "game_id":   [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "team":      ["A", "A", "B"] * 3,
        "position":  ["C", "D", "C"] * 3,
        "toi_seconds": [1000, 1200, 900, 1100, 1300, 1000, 1200, 1250, 950],
        "shots":     [2, 1, 1, 3, 0, 2, 1, 2, 0],
        "pp_toi_seconds": [100, 50, 0, 200, 0, 60, 90, 40, 30],
        "sh_toi_seconds": [50, 100, 150, 0, 0, 30, 60, 80, 20],
        "fow": [5, 0, 3, 6, 0, 4, 7, 0, 2],
        "fol": [3, 0, 5, 6, 0, 2, 1, 0, 2],
        "stats_known": [True, True, True, True, False, True, True, True, True],
    })
    attempts = pd.DataFrame({"game_id": [1, 1, 1, 2, 2, 3],
                             "shooter_id": [1, 2, 3, 1, 3, 2],
                             "attempts": [3, 1, 1, 4, 2, 2]})
    team_games = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3], "team": ["A", "B", "B", "A", "A", "B"],
        "sog": [30, 25, 28, 32, 20, 22]})
    strength = pd.DataFrame({
        "game_id":    [1, 1, 1, 1, 2, 2, 2],
        "shooter_id": [1, 1, 2, 3, 1, 1, 3],
        "strength":   ["5v4", "5v5", "5v5", "4v5", "5v5", "6v5", "5v4"],
        "n":          [1, 1, 1, 1, 2, 1, 2]})
    return skaters, games, attempts, team_games, strength


@pytest.fixture(scope="module")
def toy():
    return build_player_features(*toy_inputs())


def row(df, player, game):
    r = df[(df["player_id"] == player) & (df["game_id"] == game)]
    assert len(r) == 1
    return r.iloc[0]


class TestHelpers:
    def test_pp_strength(self):
        got = is_pp_strength(["5v4", "5v3", "4v3", "6v4", "6v3", "5v5",
                              "6v5", "4v5", "4v4", "3v3", "7v5", "bad",
                              None])
        assert got.tolist() == [True] * 5 + [False] * 8

    def test_season_env_hand(self):
        # season S, group F: d1 two rows (2 SOG / 3600 s, 1 / 3600),
        # d2 one row (3 / 1800); last year's rate on the first date 6.0,
        # prior 3600 s. d1: nothing earlier -> 6.0. d2: (3 + 3600 x 6 /
        # 3600) / (7200 + 3600) x 3600 = 9 / 10800 x 3600 = 3.0.
        # Season S2 restarts at its own prior (4.0); group D separate.
        out = season_env(
            dates=[_day(1), _day(1), _day(2), _day(3), _day(1)],
            seasons=["S", "S", "S", "S2", "S"],
            groups=["F", "F", "F", "F", "D"],
            sog=[2, 1, 3, 5, 9], toi=[3600, 3600, 1800, 3600, 3600],
            prev_rate=[6.0, 99.0, 99.0, 4.0, 2.0], prior_seconds=3600)
        np.testing.assert_allclose(out, [6.0, 6.0, 3.0, 4.0, 2.0])

    def test_decayed_sums_hand(self):
        # one key, a year apart: row 0 -> 0; row 1 -> 1 x 0.5;
        # row 2 -> 1 x 0.25 + 2 x 0.5 = 1.25. A new key restarts at 0.
        d = [_day(1), _day(366), _day(731), _day(5)]
        out = decayed_sums(["x", "x", "x", "y"], d,
                           {"v": [1.0, 2.0, 4.0, 8.0]}, half_life_days=365)
        np.testing.assert_allclose(out["v"], [0.0, 0.5, 1.25, 0.0])

    def test_decayed_sums_rejects_same_day_rows(self):
        with pytest.raises(ValueError):
            decayed_sums(["x", "x"], [_day(1), _day(1)], {"v": [1, 2]})


class TestToyV3Features:
    def test_pp_pk_minutes_skip_unknown_games(self, toy):
        r1, r2 = row(toy, 1, 3), row(toy, 2, 3)
        assert r1["pp_toi_l5"] == pytest.approx((100 + 200) / 2 / 60)
        assert r1["pk_toi_l10"] == pytest.approx((50 + 0) / 2 / 60)
        # player 2's g2 is unknown: the mean is over g1 only (not (50+0)/2)
        assert r2["pp_toi_l5"] == pytest.approx(50 / 60)
        assert r2["es_toi_l10"] == pytest.approx((1200 - 50 - 100) / 60)
        assert np.isnan(row(toy, 1, 1)["pp_toi_l5"])     # no history

    def test_pp_share_drops_team_games_with_an_unknown_skater(self, toy):
        # team A's PP seconds: g1 (100 + 50) / 5 = 30; g2 unknown (player
        # 2's stats were never filled), so g2 counts in neither part
        assert row(toy, 1, 3)["pp_share_l5"] == pytest.approx(100 / 30)
        # team B: g1 0 / 5 = 0 (share 0/0 -> NaN at g2), g2 60 / 5 = 12
        assert np.isnan(row(toy, 3, 2)["pp_share_l5"])
        assert row(toy, 3, 3)["pp_share_l5"] == pytest.approx(60 / 12)

    def test_team_pp_and_opponent_pk(self, toy):
        # A's PP minutes over its earlier games: g1 0.5, g2 unknown -> 0.5
        assert row(toy, 1, 3)["team_pp_l10"] == pytest.approx(30 / 60)
        # B's PK seconds = its skaters' PK / 4: g1 150/4, g2 30/4
        want = (150 / 4 + 30 / 4) / 2 / 60
        assert row(toy, 1, 3)["opp_pk_l10"] == pytest.approx(want)
        assert row(toy, 2, 3)["opp_pk_l10"] == pytest.approx(want)

    def test_faceoffs(self, toy):
        r = row(toy, 1, 3)
        assert r["fo_l20"] == pytest.approx((8 + 12) / 2)
        assert r["fo_win_l20"] == pytest.approx((5 + 6) / (8 + 12))
        assert np.isnan(row(toy, 2, 3)["fo_win_l20"])   # 0 faceoffs: 0/0

    def test_pp_sog_rates(self, toy):
        r = row(toy, 1, 3)
        rel = r["league_pos_sog60"]
        # PP: g1 1 SOG in 100 s, g2 0 in 200 s -> 1 x 3600 / 300 = 12/h
        assert r["pp_sog60_l20_rel"] == pytest.approx(12.0 / rel)
        assert r["pp_sog60_season_rel"] == pytest.approx(12.0 / rel)
        # the rest: (2 - 1) + (3 - 0) = 4 SOG in 900 + 900 s -> 8/h
        assert r["nonpp_sog60_l20_rel"] == pytest.approx(8.0 / rel)
        # player 2: g1 0 PP SOG in 50 PP s (his shot was 5v5); g2 unknown
        r2 = row(toy, 2, 3)
        assert r2["pp_sog60_l20_rel"] == pytest.approx(0.0)
        # player 3 at g2: g1 PP time 0 -> no PP rate
        assert np.isnan(row(toy, 3, 2)["pp_sog60_l20_rel"])

    def test_lineup_ranks(self, toy):
        r1, r2 = row(toy, 1, 3), row(toy, 2, 3)
        assert (r1["pp_rank"], r2["pp_rank"]) == (1.0, 2.0)
        # one forward and one defenceman dressed: each alone in its group
        assert r1["es_toi_rank_pct"] == 0.0 and r2["toi_rank_pct"] == 0.0

    def test_b3_formula(self, toy):
        r = row(toy, 1, 3)
        e1, e2, e3 = (row(toy, 1, g)["env_pos_sog60"] for g in (1, 2, 3))
        w1, w2 = 0.5 ** (3 / 365), 0.5 ** (2 / 365)       # d4 - d1, d4 - d2
        kp = PS.SHRINK_TOI_SECONDS * e3 / 3600
        idx = ((w1 * 2 + w2 * 3 + kp)
               / (w1 * 1000 * e1 / 3600 + w2 * 1100 * e2 / 3600 + kp))
        assert r["decay_index"] == pytest.approx(idx)
        assert r["b3_mean"] == pytest.approx(idx * e3 * r["exp_toi"] / 3600)
        # no history: the index is exactly 1 (the league rate)
        assert row(toy, 1, 1)["decay_index"] == pytest.approx(1.0)

    def test_env_is_the_season_to_date_forward_rate(self, toy):
        # forwards (players 1 and 3) before d4: SOG 2+1+3+2 = 8 in
        # 1000+900+1100+1000 = 4000 s, shrunk toward L_prev (first date)
        lp = row(toy, 1, 1)["league_pos_sog60"]
        k = PS.ENV_PRIOR_SECONDS
        want = (8 + k * lp / 3600) / (4000 + k) * 3600
        assert row(toy, 1, 3)["env_pos_sog60"] == pytest.approx(want)
        assert row(toy, 1, 1)["env_pos_sog60"] == pytest.approx(lp)

    def test_without_v3_inputs_the_v2_columns_are_unchanged(self, toy):
        s, g, a, t, ss = toy_inputs()
        old = build_player_features(s[["player_id", "game_id", "team",
                                       "position", "toi_seconds", "shots"]],
                                    g, a, t)
        for c in FEATURES_PP + FEATURES_USAGE:
            if c in ("team_pp_l10", "opp_pk_l10", "pp_rank",
                     "es_toi_rank_pct", "toi_rank_pct"):
                continue
            assert old[c].isna().all(), c
        cols = FEATURES + ["b0_mean", "shrunk_sog60", "exp_toi", "b3_mean"]
        pd.testing.assert_frame_equal(
            old[cols].reset_index(drop=True), toy[cols].reset_index(drop=True))


# ── Point-in-time ──────────────────────────────────────────────────

def random_league_v3(seed=0, seasons=(SEASON,), days=40, n_teams=6, roster=6):
    from tests.test_props_sog import random_league
    s, g, a, t = random_league(seed=seed, seasons=seasons, days=days,
                               n_teams=n_teams, roster=roster)
    rng = np.random.default_rng(seed + 1000)
    s = s.copy()
    s["pp_toi_seconds"] = rng.integers(0, 240, len(s))
    s["sh_toi_seconds"] = rng.integers(0, 180, len(s))
    s["fow"] = np.where(s["position"] == "C", rng.integers(0, 12, len(s)), 0)
    s["fol"] = np.where(s["position"] == "C", rng.integers(0, 12, len(s)), 0)
    s["stats_known"] = rng.random(len(s)) > 0.03
    covered = set(a["game_id"])
    rows = []
    for r in s[s["game_id"].isin(covered)].itertuples():
        for st in ("5v5", "5v4", "4v5"):
            n = int(rng.poisson(0.6 if st != "4v5" else 0.1))
            if n:
                rows.append((r.game_id, r.player_id, st, n))
    ss = pd.DataFrame(rows, columns=["game_id", "shooter_id", "strength", "n"])
    return s, g, a, t, ss


V3_PIT_COLS = (FEATURES_PP + FEATURES_USAGE
               + ["env_pos_sog60", "decay_index", "b3_mean"])


class TestPointInTimeV3:
    @pytest.mark.parametrize("cut_day", [12, 25, 33])
    def test_rewriting_or_removing_rows_on_or_after_the_date(self, cut_day):
        s, g, a, t, ss = random_league_v3(seed=cut_day,
                                          seasons=(SEASON, 20212022))
        full = build_player_features(s, g, a, t, ss)
        dates = sorted(g["date"].unique())
        cut = pd.Timestamp(dates[min(cut_day + 25, len(dates) - 1)])
        rng = np.random.default_rng(7)
        on = set(g.loc[g["date"] == cut, "game_id"])
        after = set(g.loc[g["date"] > cut, "game_id"])

        def cut_rows(df):
            return df[~df["game_id"].isin(after)].copy()

        g2, s2, a2, t2, ss2 = map(cut_rows, (g, s, a, t, ss))
        sel = s2["game_id"].isin(on)
        k = int(sel.sum())
        s2.loc[sel, "shots"] = rng.integers(0, 12, k)
        s2.loc[sel, "toi_seconds"] = rng.integers(60, 2400, k)
        s2.loc[sel, "pp_toi_seconds"] = rng.integers(0, 600, k)
        s2.loc[sel, "sh_toi_seconds"] = rng.integers(0, 600, k)
        s2.loc[sel, "fow"] = rng.integers(0, 30, k)
        s2.loc[sel, "fol"] = rng.integers(0, 30, k)
        s2.loc[sel, "stats_known"] = rng.random(k) > 0.5
        sel = ss2["game_id"].isin(on)
        ss2.loc[sel, "n"] = rng.integers(1, 9, int(sel.sum()))
        ss2.loc[sel, "strength"] = rng.choice(["5v4", "5v5", "5v3"],
                                              int(sel.sum()))
        part = build_player_features(s2, g2, a2, t2, ss2)

        key = ["player_id", "game_id"]
        lhs = full[full["date"] <= cut].set_index(key).sort_index()
        rhs = part.set_index(key).sort_index()
        assert len(rhs) == len(lhs) and (rhs["date"] == cut).any()
        assert lhs["season"].nunique() == 2           # crosses a season
        pd.testing.assert_frame_equal(lhs[V3_PIT_COLS], rhs[V3_PIT_COLS],
                                      check_exact=False, rtol=1e-12)
        for c in ("pp_toi_l5", "decay_index", "pp_rank", "env_pos_sog60"):
            assert lhs[c].notna().any(), c


# ── Variants and the adoption rule ─────────────────────────────────

class TestVariants:
    def test_variant_specs(self):
        assert P.VARIANTS["P0"] == {"features": list(FEATURES), "offset": "B1"}
        assert P.VARIANTS["P1"]["features"] == list(FEATURES) + FEATURES_PP
        assert (P.VARIANTS["P2"]["features"] == P.VARIANTS["P3"]["features"]
                == list(FEATURES) + FEATURES_PP + FEATURES_USAGE)
        assert P.VARIANTS["P3"]["offset"] == "B3"
        assert P.DEFAULT_VARIANT in P.VARIANTS
        with pytest.raises(ValueError):
            P.run_props(frame=pd.DataFrame(), variant="P9")

    def test_p3_starts_from_b3_and_is_gated_against_b1(self):
        seasons = (20202021, 20212022, 20222023)
        s, g, a, t, ss = random_league_v3(seed=3, seasons=seasons, days=40,
                                          n_teams=6, roster=6)
        frame = build_player_features(s, g, a, t, ss)
        params = dict(P.LGBM_PARAMS, n_estimators=20, min_child_samples=50)
        res = P.run_props(frame=frame, params=params, variant="P3")
        oof = res["oof"]
        data = P.load_props_dataset(frame).set_index(["player_id", "game_id"])
        b3 = np.clip(data.loc[pd.MultiIndex.from_frame(
            oof[["player_id", "game_id"]]), "b3_mean"].to_numpy(), *P.MU_CLIP)
        np.testing.assert_allclose(oof["mu_B3"], b3)
        # the drift correction is taken against the variant's own offset
        log_c = P.drift_shift(oof["season"], oof["date"],
                              P.booster_adjustment(oof["mu_M1"], oof["mu_B3"]))
        np.testing.assert_allclose(oof["log_c"], log_c, rtol=1e-9, atol=1e-15)
        assert res["pooled"]["variant"] == "P3"
        assert res["pooled"]["gate"]["folds_won_vs_B1"] == sum(
            f["nll_M"] < f["nll_B1"] for f in res["folds"])
        assert {"diff_B3_B1", "se_B3_B1"} <= set(res["pooled"])


def _fake(nll_by_fold: dict, rows_nll: np.ndarray, gate: bool) -> dict:
    """A run_props-shaped result with given per-row NLLs (via a Poisson
    alpha of 0: mu chosen so that -log pmf(y=0) = mu)."""
    n = len(rows_nll)
    oof = pd.DataFrame({"player_id": np.arange(n), "game_id": 1,
                        "sog": 0.0, "mu_M": rows_nll, "alpha_M": 0.0})
    return {"oof": oof,
            "folds": [{"val_season": s, "nll_M": v} for s, v in nll_by_fold.items()],
            "pooled": {"gate": {"passed": gate}}}


class TestAdoption:
    def test_row_nll_is_the_nb_nll(self):
        r = _fake({1: 0.0}, np.array([0.5, 1.0, 2.0]), True)
        np.testing.assert_allclose(P.row_nll(r).to_numpy(), [0.5, 1.0, 2.0])

    def test_rule(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(1, 2, 4000)
        folds = {s: 1.5 for s in range(5)}
        p0 = _fake(folds, base, True)
        # P1: clearly better on every row, wins 3 of 5 folds -> adopted
        p1 = _fake({0: 1.4, 1: 1.4, 2: 1.4, 3: 1.6, 4: 1.6}, base - 0.01, True)
        # P2: 0.01 better on average but +-0.9 per row: -0.01 / (0.9 /
        # sqrt(4000)) = -0.7 SE, fails the 2 SE test -> not adopted
        swing = 0.9 * np.where(np.arange(4000) % 2 == 0, 1.0, -1.0)
        p2 = _fake({s: 1.3 for s in range(5)}, base - 0.01 + swing, True)
        # P3: much better but fails the gate -> not adopted
        p3 = _fake({s: 1.0 for s in range(5)}, base - 0.5, False)
        d = P.adoption({"P0": p0, "P1": p1, "P2": p2, "P3": p3})
        assert d["adopted"] == "P1"
        st = {x["variant"]: x for x in d["steps"]}
        assert st["P1"]["adopted"] and st["P1"]["cmp_folds_won"] == 3
        assert st["P2"]["against"] == "P1" and not st["P2"]["nll_by_2se"]
        assert not st["P3"]["gate_passed"] and not st["P3"]["adopted"]

    def test_too_few_folds_blocks_adoption(self):
        base = np.linspace(1, 2, 1000)
        p0 = _fake({s: 1.5 for s in range(5)}, base, True)
        p1 = _fake({0: 1.4, 1: 1.4, 2: 1.6, 3: 1.6, 4: 1.6}, base - 0.01, True)
        d = P.adoption({"P0": p0, "P1": p1})
        assert d["adopted"] == "P0" and not d["steps"][0]["folds_ok"]

    def test_different_rows_raise(self):
        a = _fake({0: 1.0}, np.ones(3), True)
        b = _fake({0: 1.0}, np.ones(4), True)
        with pytest.raises(ValueError):
            P.compare_variants(a, b)
