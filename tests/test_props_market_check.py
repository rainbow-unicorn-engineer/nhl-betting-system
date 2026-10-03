"""
Tests for models/props_market_check.py: the no-vig math, push handling on
integer lines, how stored over/under rows become a pre-game pair, the
game-clustered SE on a hand example, the betting rule at its boundary,
and the guard that only out-of-fold predictions are scored. No database
is used (every input is passed in).
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

import models.props_market_check as C
import models.props_sog as P
from features.player_shots import build_player_features
from tests.test_props_sog import random_league

T0 = pd.Timestamp("2026-01-15 00:00", tz="UTC")


# ── No-vig math ────────────────────────────────────────────────────

class TestNoVig:
    def test_american_to_decimal(self):
        np.testing.assert_allclose(C.american_to_decimal([-120, 110, 100, -100, -250]),
                                   [1 + 100 / 120, 2.10, 2.0, 2.0, 1.4])

    def test_even_market_is_one_half(self):
        d = C.american_to_decimal([-110, -110])
        assert C.no_vig_over(d[0], d[1]) == pytest.approx(0.5)

    def test_hand_computed(self):
        # over -120 -> 1/d = 120/220 = 0.545454..., under +100 -> 0.5
        # no-vig over = 0.545454 / 1.045454 = 0.521739...
        d_o, d_u = C.american_to_decimal([-120, 100])
        assert C.no_vig_over(d_o, d_u) == pytest.approx((120 / 220) / (120 / 220 + 0.5))
        assert C.no_vig_over(d_o, d_u) == pytest.approx(0.5217391, abs=1e-7)
        # the two no-vig sides add up to one
        assert C.no_vig_over(d_o, d_u) + C.no_vig_over(d_u, d_o) == pytest.approx(1.0)


# ── Model side probabilities and pushes ────────────────────────────

class TestSideProbabilities:
    def test_half_line_has_no_push_and_matches_prob_over(self):
        mu = np.array([0.8, 2.4, 3.9])
        for alpha in (0.0, 0.05):
            po, pu, push = C.model_side_probs(mu, alpha, 2.5)
            np.testing.assert_allclose(push, 0.0)
            np.testing.assert_allclose(po + pu, 1.0)
            np.testing.assert_allclose(po, P.prob_over(mu, alpha, 2.5))

    def test_integer_line_poisson_hand_computed(self):
        # Poisson mean 2: P(0) = e^-2, P(1) = 2e^-2, P(2) = 2e^-2
        e = math.exp(-2)
        po, pu, push = C.model_side_probs(np.array([2.0]), 0.0, 2.0)
        assert push[0] == pytest.approx(2 * e)
        assert po[0] == pytest.approx((1 - 5 * e) / (1 - 2 * e))
        assert pu[0] == pytest.approx(3 * e / (1 - 2 * e))
        assert po[0] + pu[0] == pytest.approx(1.0)

    def test_integer_line_negative_binomial(self):
        mu, alpha, line = 2.7, 0.2, 3.0
        r = 1 / alpha
        q = r / (r + mu)
        push = nbinom.pmf(3, r, q)
        po, pu, pp = C.model_side_probs(np.array([mu]), alpha, line)
        assert pp[0] == pytest.approx(push)
        assert po[0] == pytest.approx(nbinom.sf(3, r, q) / (1 - push))
        assert pu[0] == pytest.approx(nbinom.cdf(2, r, q) / (1 - push))

    def test_per_row_alpha_and_line(self):
        po, _, _ = C.model_side_probs(np.array([2.0, 2.0]), np.array([0.0, 0.1]),
                                       np.array([1.5, 2.5]))
        assert po[0] == pytest.approx(P.prob_over([2.0], 0.0, 1.5)[0])
        assert po[1] == pytest.approx(P.prob_over([2.0], 0.1, 2.5)[0])

    def test_push_rows_are_dropped_from_the_outcomes(self):
        matched = pd.DataFrame({
            "game_id": [1, 1, 2], "player_id": [10.0, 11.0, 10.0],
            "line": [2.0, 2.0, 1.5], "sog": [2, 3, 2],
            "mu_M": [2.0, 2.0, 2.0], "alpha_M": [0.0, 0.0, 0.0]})
        outcomes = pd.DataFrame({"game_id": [1, 1, 2], "player_id": [10, 11, 10],
                                 "shots": [2, 3, 2]})
        df, n_push = C.attach_outcomes(matched, outcomes)
        assert n_push == 1                              # 2 shots on line 2
        assert list(df["player_id"]) == [11.0, 10.0]
        assert list(df["over_hit"]) == [1.0, 1.0]
        e = math.exp(-2)
        assert df["p_model"].iloc[0] == pytest.approx((1 - 5 * e) / (1 - 2 * e))

    def test_outcome_disagreement_raises(self):
        matched = pd.DataFrame({"game_id": [1], "player_id": [10.0], "line": [1.5],
                                "sog": [2], "mu_M": [2.0], "alpha_M": [0.0]})
        with pytest.raises(ValueError, match="disagree"):
            C.attach_outcomes(matched, pd.DataFrame(
                {"game_id": [1], "player_id": [10], "shots": [3]}))


# ── Pairing the stored over/under rows ─────────────────────────────

def _row(**kw):
    base = {"game_id": 1, "book": "DraftKings", "market": C.MARKET,
            "player_id": 10, "line": 2.5, "over_price": None, "under_price": None,
            "over_price_open": None, "under_price_open": None,
            "last_updated": T0 - pd.Timedelta(hours=2), "event_start": T0}
    base.update(kw)
    return base


class TestPairing:
    def test_current_pair_before_puck_drop(self):
        r = _row(over_price=-120, under_price=100, over_price_open=-110,
                 under_price_open=-110)
        assert C.pregame_pair(r) == (-120, 100, "current")

    def test_current_pair_after_puck_drop_is_never_used(self):
        r = _row(over_price=-300, under_price=220, over_price_open=-110,
                 under_price_open=-110, last_updated=T0 + pd.Timedelta(minutes=5))
        assert C.pregame_pair(r) == (-110, -110, "open")
        # at puck drop exactly counts as in play too
        assert C.pregame_pair({**r, "last_updated": T0}) == (-110, -110, "open")

    def test_opening_pair_when_updated_in_play(self):
        r = _row(over_price_open=105, under_price_open=-135,
                 last_updated=T0 + pd.Timedelta(hours=1))
        assert C.pregame_pair(r) == (105, -135, "open")

    def test_one_sided_rows_have_no_pair(self):
        assert C.pregame_pair(_row(over_price=-120))[2] == "no_two_sided_pregame_pair"
        assert C.pregame_pair(_row(under_price_open=-120,
                                   last_updated=T0 + pd.Timedelta(hours=1))
                              )[0] is None
        # one-sided current, complete opening pair: the opening pair
        assert C.pregame_pair(_row(over_price=-120, over_price_open=-115,
                                   under_price_open=-105)) == (-115, -105, "open")

    def test_unknown_timestamps_do_not_trust_the_current_price(self):
        r = _row(over_price=-120, under_price=100, last_updated=None)
        assert C.pregame_pair(r)[2] == "no_two_sided_pregame_pair"

    def test_pair_prices_columns_and_counts(self):
        rows = pd.DataFrame([
            _row(player_id=10, over_price=-120, under_price=100),
            _row(player_id=11, over_price=-300, under_price=220,
                 over_price_open=110, under_price_open=-140,
                 last_updated=T0 + pd.Timedelta(minutes=1)),
            _row(player_id=12, over_price=-120),
        ])
        rows = C._object_rows(rows)
        props, counts = C.pair_prices(rows)
        assert counts == {"rows": 3, "current": 1, "open": 1,
                          "no_two_sided_pregame_pair": 1,
                          "current_ignored_after_puck_drop": 1}
        assert list(props["player_id"]) == [10, 11]
        assert list(props["over_am"]) == [-120, 110]
        assert list(props["under_am"]) == [100, -140]
        # p_mkt is the OVER side's no-vig probability
        assert props["p_mkt"].iloc[0] == pytest.approx(0.5217391, abs=1e-7)
        assert props["p_mkt"].iloc[1] < 0.5


# ── Clustered SE ───────────────────────────────────────────────────

class TestClusteredSE:
    def test_hand_example(self):
        # d = 1 2 3 6, mean 3, residuals -2 -1 0 3; games a a b c
        # S_a = -3, S_b = 0, S_c = 3; sum S^2 = 18; G = 3
        # SE = sqrt(3/2 x 18) / 4 = sqrt(27) / 4
        mean, se = C.clustered_mean_se([1, 2, 3, 6], ["a", "a", "b", "c"])
        assert mean == 3.0
        assert se == pytest.approx(math.sqrt(27) / 4)

    def test_singleton_clusters_give_the_usual_se(self):
        d = np.array([0.3, -0.1, 0.4, 0.0, 0.2])
        _, se = C.clustered_mean_se(d, np.arange(5))
        assert se == pytest.approx(d.std(ddof=1) / math.sqrt(5))

    def test_one_cluster_has_no_se(self):
        assert math.isnan(C.clustered_mean_se([1.0, 2.0], [7, 7])[1])

    def test_log_loss(self):
        np.testing.assert_allclose(C.log_loss([0.8, 0.8], [1, 0]),
                                   [-math.log(0.8), -math.log(0.2)])


class TestPrimaryGate:
    def _props(self, n, p_model, games=None):
        rng = np.random.default_rng(0)
        y = (rng.random(n) < 0.7).astype(float)
        return pd.DataFrame({"over_hit": y, "p_model": p_model, "p_mkt": 0.4,
                             "game_id": np.arange(n) // 3 if games is None else games})

    def test_needs_300_props(self):
        assert not C.primary_block(self._props(299, 0.7))["beats_market"]
        r = C.primary_block(self._props(300, 0.7))
        assert r["n"] == 300 and r["diff"] < 0 and r["beats_market"]

    def test_worse_model_does_not_beat(self):
        r = C.primary_block(self._props(600, 0.3))
        assert r["diff"] > 0 and not r["beats_market"]


# ── Betting rule ───────────────────────────────────────────────────

class TestBettingRule:
    def test_boundary(self):
        # over at even money: implied 0.5; edge exactly 0.04 bets at T=0.04
        side = C.bet_side([0.54, 0.5399, 0.56], [0.46, 0.4601, 0.44],
                          [2.0, 2.0, 2.0], [1.8, 1.8, 1.8], 0.04)
        assert list(side) == [1, 0, 1]
        assert list(C.bet_side([0.54, 0.56], [0.46, 0.44], [2.0, 2.0],
                               [1.8, 1.8], 0.06)) == [0, 1]

    def test_under_side(self):
        # under at 2.5 -> implied 0.4; P(under) 0.44 is an edge of 0.04
        assert list(C.bet_side([0.56, 0.5601], [0.44, 0.4399], [1.6, 1.6],
                               [2.5, 2.5], 0.04)) == [-1, 0]

    def test_at_most_one_side(self):
        # only possible without a margin: the larger edge is taken
        side = C.bet_side([0.6, 0.45], [0.4, 0.55], [2.5, 2.5], [2.5, 2.5], 0.04)
        assert list(side) == [1, -1]

    def test_profits_at_quoted_prices(self):
        df = pd.DataFrame({"game_id": [1, 1, 2, 3],
                           "p_model": [0.6, 0.6, 0.3, 0.5],
                           "p_model_under": [0.4, 0.4, 0.7, 0.5],
                           "d_over": [2.0, 2.0, 2.0, 1.9],
                           "d_under": [1.8, 1.8, 1.8, 1.9],
                           "over_hit": [1.0, 0.0, 0.0, 1.0]})
        b = C.bet_profits(df, 0.04)
        assert list(b["side"]) == ["over", "over", "under"]
        np.testing.assert_allclose(b["profit"], [1.0, -1.0, 0.8])
        blk = C.betting_block(df, 0.04)
        assert blk["bets"] == 3 and blk["hit_rate"] == pytest.approx(2 / 3)
        assert blk["roi"] == pytest.approx(0.8 / 3)
        assert blk["roi_lo"] <= blk["roi"] <= blk["roi_hi"]

    def test_bootstrap_is_seeded_and_clustered(self):
        prof = [1.0, -1.0, 0.9, -1.0, -1.0, 1.2]
        games = [1, 1, 2, 3, 3, 4]
        assert C.bootstrap_roi(prof, games) == C.bootstrap_roi(prof, games)
        # one game only: every resample is that game
        lo, hi = C.bootstrap_roi([1.0, -1.0, 1.0], [5, 5, 5])
        assert lo == pytest.approx(1 / 3) and hi == pytest.approx(1 / 3)
        assert all(math.isnan(x) for x in C.bootstrap_roi([], []))

    def test_line_groups(self):
        assert list(C.line_group([0.5, 1.5, 2.5, 3.5, 4.5])) == [
            "0.5", "1.5", "2.5", "3.5+", "3.5+"]


# ── Out-of-fold guard and an end-to-end run on a synthetic league ──

SEASONS = (20202021, 20212022, 20222023)
PARAMS = dict(P.LGBM_PARAMS, n_estimators=20, min_child_samples=50)


@pytest.fixture(scope="module")
def league():
    s, g, a, t = random_league(seed=3, seasons=SEASONS, days=40, n_teams=6, roster=6)
    frame = build_player_features(s, g, a, t)
    res = P.run_props(register=False, frame=frame, params=PARAMS)
    return frame, res


class TestOutOfFoldGuard:
    def test_training_only_season_has_no_predictions(self, league):
        _, res = league
        with pytest.raises(ValueError, match="not a walk-forward validation"):
            C.validation_predictions(res, SEASONS[0])
        with pytest.raises(ValueError, match="not a walk-forward validation"):
            C.validation_predictions(res, 20302031)

    def test_rows_are_the_folds_own(self, league):
        _, res = league
        v = C.validation_predictions(res, SEASONS[-1])
        fold = res["folds"][-1]
        assert len(v) == fold["n"] and set(v["season"]) == {SEASONS[-1]}
        broken = {**res, "oof": res["oof"].assign(
            alpha_M=res["oof"]["alpha_M"] + 0.5)}
        with pytest.raises(ValueError, match="alpha"):
            C.validation_predictions(broken, SEASONS[-1])
        oof = res["oof"]
        first_last = oof.index[oof["season"] == SEASONS[-1]][0]
        short = {**res, "oof": oof.drop(index=first_last)}
        with pytest.raises(ValueError, match="out-of-fold rows"):
            C.validation_predictions(short, SEASONS[-1])

    def test_predictions_never_see_the_seasons_outcomes(self, league):
        # Rewriting the validation season's targets (features unchanged)
        # must not move its predictions; rewriting a training season's
        # targets must (so the check is sensitive).
        frame, res = league
        v = C.validation_predictions(res, SEASONS[-1])
        rng = np.random.default_rng(0)
        last = frame["season"] == SEASONS[-1]
        f2 = frame.copy()
        f2.loc[last, "sog"] = rng.permutation(f2.loc[last, "sog"].to_numpy())
        v2 = C.validation_predictions(
            P.run_props(register=False, frame=f2, params=PARAMS), SEASONS[-1])
        np.testing.assert_array_equal(v["mu_M"].to_numpy(), v2["mu_M"].to_numpy())
        np.testing.assert_array_equal(v["alpha_M"].to_numpy(), v2["alpha_M"].to_numpy())
        mid = frame["season"] == SEASONS[1]
        f3 = frame.copy()
        f3.loc[mid, "sog"] = rng.permutation(f3.loc[mid, "sog"].to_numpy())
        v3 = C.validation_predictions(
            P.run_props(register=False, frame=f3, params=PARAMS), SEASONS[-1])
        assert not np.allclose(v["mu_M"].to_numpy(), v3["mu_M"].to_numpy())

    def test_run_market_check_end_to_end(self, league, monkeypatch):
        frame, res = league
        oof = C.validation_predictions(res, SEASONS[-1])
        sample = oof.iloc[:40]
        rows = [_row(game_id=int(r.game_id), player_id=int(r.player_id), line=1.5,
                     over_price=-120, under_price=100) for r in sample.itertuples()]
        g0 = int(sample["game_id"].iloc[0])
        # (a player-game with < 5 prior appearances; the synthetic league
        # only has them in its first season)
        inel = frame[frame["n_prior"] < 5].iloc[-1]
        rows += [
            _row(game_id=g0, player_id=None, espn_athlete_id=1, over_price=-110,
                 under_price=-110),
            _row(game_id=g0, player_id=987654, over_price=-110, under_price=-110),
            _row(game_id=int(inel["game_id"]), player_id=int(inel["player_id"]),
                 over_price=-110, under_price=-110),
            _row(game_id=g0, player_id=int(sample["player_id"].iloc[0]),
                 market=C.ALT_MARKET, line=2.5, over_price=150),
        ]
        rows = pd.DataFrame(rows).assign(date=pd.Timestamp("2023-01-01"))
        outcomes = frame.rename(columns={"sog": "shots"})[
            ["game_id", "player_id", "shots", "toi_seconds"]]

        def no_refit(*a, **k):
            raise AssertionError("run_market_check refitted the model")
        monkeypatch.setattr(P, "run_props", no_refit)
        out = C.run_market_check(season=SEASONS[-1], frame=frame, res=res,
                                 rows=rows, outcomes=outcomes)
        um = out["unmatched"]["DraftKings"]
        assert um == {"no_player_id": 1, "did_not_play": 1, "not_eligible": 1,
                      "other": 0, "matched": 40}
        prim = out["results"]["DraftKings"]["primary"]
        assert prim["n"] == 40 and not prim["beats_market"]    # n < 300
        assert out["results"][C.POOLED]["primary"]["n"] == 40
        assert out["alternate"]["DraftKings"]["n"] == 1
        scored = out["scored"]
        np.testing.assert_allclose(
            scored["p_model"], P.prob_over(scored["mu_M"], scored["alpha_M"].iloc[0], 1.5))
        assert "PRIMARY" in C.format_report(out)

    def test_cli_never_registers(self, monkeypatch):
        seen = {}

        def fake_run(register=False, frame=None, params=None, **k):
            seen["register"] = register
            raise RuntimeError("stop")
        monkeypatch.setattr(P, "run_props", fake_run)
        with pytest.raises(RuntimeError, match="stop"):
            C.run_market_check(frame=pd.DataFrame())
        assert seen == {"register": False}
