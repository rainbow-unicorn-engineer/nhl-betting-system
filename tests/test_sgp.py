"""
Tests for betting/sgp.py: the score grid, the tilts (each moves one
number and keeps the other), overtime settlement, leg pricing, the
validation's market builders, the OT fit, and point-in-time guards.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

import betting.sgp as sgp
from betting.sgp import (SgpLeg, base_grid, clustered_se, consensus_from_quotes,
                         dk_market, final_scores, fit_ot_beta, four_way,
                         four_way_independent, home_win_prob, joint_grid,
                         leg_masks, ot_beta_before, ot_home_prob, outcome_cell,
                         price_legs, settlement_totals, tilt_margin, tilt_total)
from config.settings import check_db_connection, engine
from models.totals import poisson_pmf, prob_over, total_pmf

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def _pmfs(lh=3.1, la=2.8):
    return poisson_pmf(np.array([lh]))[0], poisson_pmf(np.array([la]))[0]


def _total_dist(grid):
    t = settlement_totals(grid.shape[0])
    return np.bincount(t.ravel(), weights=grid.ravel(), minlength=2 * grid.shape[0])


class TestGeometry:
    def test_settlement_total_adds_the_ot_goal_on_ties(self):
        t = settlement_totals(5)
        assert t[2, 2] == 5          # 2-2 in regulation settles 3-2: total 5
        assert t[3, 1] == 4 and t[0, 0] == 1

    def test_final_scores_split_ties_by_ot_chance(self):
        g = np.zeros((3, 3))
        g[1, 1], g[2, 0] = 0.4, 0.6
        f = final_scores(g, 0.7)
        assert f.sum() == pytest.approx(1.0)
        assert f[2, 1] == pytest.approx(0.28) and f[1, 2] == pytest.approx(0.12)
        assert f[2, 0] == pytest.approx(0.6)

    def test_ot_home_prob(self):
        assert ot_home_prob(0.5, 0.4) == pytest.approx(0.5)
        assert ot_home_prob(0.7, 0.5) == pytest.approx(0.6)
        assert ot_home_prob(0.99, 10.0) == pytest.approx(0.98)   # clipped

    def test_base_grid_matches_totals_model_total(self):
        ph, pa = _pmfs()
        g = base_grid(ph, pa)
        tp = total_pmf(ph[None], pa[None])[0]
        assert _total_dist(g) == pytest.approx(tp, abs=1e-12)


class TestTilts:
    def test_margin_tilt_hits_target_and_keeps_totals(self):
        ph, pa = _pmfs()
        g = base_grid(ph, pa)
        for target in (0.25, 0.5, 0.62, 0.8):
            q = float(ot_home_prob(target, 0.3))
            gt = tilt_margin(g, target, q)
            assert home_win_prob(gt, q) == pytest.approx(target, abs=1e-8)
            assert _total_dist(gt) == pytest.approx(_total_dist(g), abs=1e-12)
            assert gt.sum() == pytest.approx(1.0)

    def test_total_tilt_hits_target_and_keeps_margin_shares(self):
        ph, pa = _pmfs()
        g = base_grid(ph, pa)
        gt = tilt_total(g, 0.58, 6.5)
        p_over, _ = prob_over(_total_dist(gt)[None], [6.5])
        assert p_over[0] == pytest.approx(0.58, abs=1e-8)
        # within one settlement total the cells keep their ratios
        t = settlement_totals(g.shape[0])
        sel = t == 6
        assert (gt[sel] / gt[sel].sum()) == pytest.approx(g[sel] / g[sel].sum())

    def test_total_tilt_integer_line_matches_given_no_push(self):
        g = base_grid(*_pmfs())
        gt = tilt_total(g, 0.55, 6.0)
        d = _total_dist(gt)
        assert d[7:].sum() / (1 - d[6]) == pytest.approx(0.55, abs=1e-8)

    def test_joint_grid_matches_both_marginals(self):
        ph, pa = _pmfs()
        g, q = joint_grid(ph, pa, p_home=0.64, p_over=0.47, line=6.5, beta=0.4)
        f = final_scores(g, q)
        n = f.shape[0]
        assert f[leg_masks(SgpLeg("ml", "HOME"), n)[0]].sum() == \
            pytest.approx(0.64, abs=1e-8)
        assert f[leg_masks(SgpLeg("total", "OVER", 6.5), n)[0]].sum() == \
            pytest.approx(0.47, abs=1e-8)
        assert q == pytest.approx(0.5 + 0.4 * 0.14)

    def test_unreachable_target_clips_instead_of_failing(self):
        g = base_grid(*_pmfs())
        gt = tilt_margin(g, 0.9999999, 0.5)
        assert np.isfinite(gt).all() and gt.sum() == pytest.approx(1.0)

    def test_joint_without_target_uses_grid_strength_for_ot(self):
        ph, pa = _pmfs(3.5, 2.5)
        g, q = joint_grid(ph, pa, beta=1.0)
        assert q == pytest.approx(home_win_prob(base_grid(ph, pa), 0.5))


class TestLegs:
    def _f(self):
        g, q = joint_grid(*_pmfs(), p_home=0.6, beta=0.3)
        return final_scores(g, q)

    def test_masks(self):
        n = 6
        w, p = leg_masks(SgpLeg("spread", "HOME", -1.5), n)
        assert w[3, 1] and not w[2, 1] and not p.any()
        w, p = leg_masks(SgpLeg("spread", "AWAY", 1.0), n)
        assert w[1, 1]
        assert p[2, 1] and not w[2, 1]
        w, p = leg_masks(SgpLeg("team_total", "OVER", 2.5, team="AWAY"), n)
        assert w[0, 3] and not w[3, 2]
        w, p = leg_masks(SgpLeg("total", "UNDER", 5.0), n)
        assert w[2, 2] and p[3, 2] and not w[3, 3]
        with pytest.raises(ValueError):
            leg_masks(SgpLeg("team_total", "OVER", 2.5), n)
        with pytest.raises(ValueError):
            leg_masks(SgpLeg("ml", "OVER"), n)
        with pytest.raises(ValueError):
            leg_masks(SgpLeg("total", "OVER"), n)

    def test_ml_plus_total_is_the_cell_sum(self):
        f = self._f()
        r = price_legs(f, [SgpLeg("ml", "HOME"), SgpLeg("total", "OVER", 5.5)])
        n = f.shape[0]
        hf, af = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        assert r["p_all_win"] == pytest.approx(f[(hf > af) & (hf + af > 5.5)].sum())
        assert r["p_win"][0] == pytest.approx(0.6, abs=1e-8)
        assert r["p_independent"] == pytest.approx(r["p_win"][0] * r["p_win"][1])

    def test_two_total_lines_is_a_window(self):
        f = self._f()
        r = price_legs(f, [SgpLeg("total", "OVER", 5.5),
                           SgpLeg("total", "UNDER", 6.5)])
        n = f.shape[0]
        hf, af = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        assert r["p_all_win"] == pytest.approx(f[hf + af == 6].sum())

    def test_exp_multiplier_with_a_push_hand_computed(self):
        f = np.zeros((4, 4))
        f[2, 1], f[1, 2], f[3, 0] = 0.5, 0.3, 0.2     # totals 3, 3, 3
        r = price_legs(f, [SgpLeg("ml", "HOME"), SgpLeg("total", "OVER", 3.0)],
                       decimals=[1.8, 1.9])
        # every total pushes: home wins (0.7) pays 1.8 * 1, else 0
        assert r["exp_multiplier"] == pytest.approx(0.7 * 1.8)
        assert r["p_all_win"] == 0.0 and r["p_push"][1] == pytest.approx(1.0)

    def test_favourite_and_over_correlate_positively(self):
        g, q = joint_grid(*_pmfs(), p_home=0.65, p_over=0.5, line=6.5, beta=0.3)
        f = final_scores(g, q)
        assert four_way(f, True, 6.5)[0] > four_way_independent(f, True, 6.5)[0]

    def test_four_way_sums_to_one_and_shares_marginals(self):
        g, q = joint_grid(*_pmfs(), p_home=0.42, p_over=0.55, line=5.5, beta=0.3)
        f = final_scores(g, q)
        pj, pi = four_way(f, False, 5.5), four_way_independent(f, False, 5.5)
        assert pj.sum() == pytest.approx(1.0) and pi.sum() == pytest.approx(1.0)
        assert pj[0] + pj[1] == pytest.approx(pi[0] + pi[1])    # P(fav)
        assert pj[0] + pj[2] == pytest.approx(pi[0] + pi[2])    # P(over)

    def test_outcome_cell(self):
        assert outcome_cell(4, 3, True, 6.5) == 0        # fav, over
        assert outcome_cell(2, 1, True, 5.5) == 1        # fav, under
        assert outcome_cell(2, 1, False, 5.5) == 3       # dog, under
        assert outcome_cell(3, 4, True, 6.5) == 2        # dog, over


class TestMarkets:
    def test_consensus_median_and_main_line(self):
        rows = []
        for book, (h, a), (pt, o, u) in [
                ("b1", (-150, 130), (6.5, 110, -130)),
                ("b2", (-140, 120), (6.5, 105, -125)),
                ("b3", (-160, 140), (5.5, -150, 130)),
                ("b4", (-145, 125), (6.0, -110, -110))]:
            rows += [(1, book, "h2h", "home", h, None),
                     (1, book, "h2h", "away", a, None),
                     (1, book, "totals", "over", o, pt),
                     (1, book, "totals", "under", u, pt)]
        q = pd.DataFrame(rows, columns=["game_id", "book", "market", "side",
                                        "price", "point"])
        c = consensus_from_quotes(q).iloc[0]
        assert c["n_ml_books"] == 4
        nv = [sgp._novig([h], [a])[0] for h, a in
              [(-150, 130), (-140, 120), (-160, 140), (-145, 125)]]
        assert c["p_home"] == pytest.approx(np.median(nv))
        assert c["line"] == 6.5 and c["n_ou_books"] == 2   # integer 6.0 ignored
        assert c["p_over"] == pytest.approx(np.median(
            [sgp._novig([110], [-130])[0], sgp._novig([105], [-125])[0]]))

    def test_consensus_drops_games_without_half_point_line(self):
        q = pd.DataFrame([(1, "b", "h2h", "home", -110, None),
                          (1, "b", "h2h", "away", -110, None),
                          (1, "b", "totals", "over", -110, 6.0),
                          (1, "b", "totals", "under", -110, 6.0)],
                         columns=["game_id", "book", "market", "side",
                                  "price", "point"])
        assert consensus_from_quotes(q).empty

    def test_consensus_line_tie_goes_to_most_balanced(self):
        rows = [(1, "b1", "h2h", "home", -110, None),
                (1, "b1", "h2h", "away", -110, None),
                (1, "b1", "totals", "over", 140, 6.5),
                (1, "b1", "totals", "under", -160, 6.5),
                (1, "b2", "totals", "over", -105, 5.5),
                (1, "b2", "totals", "under", -115, 5.5)]
        q = pd.DataFrame(rows, columns=["game_id", "book", "market", "side",
                                        "price", "point"])
        assert consensus_from_quotes(q).iloc[0]["line"] == 5.5

    def test_dk_market_filters(self):
        rows = pd.DataFrame({
            "game_id": [1, 2, 3, 4],
            "home_ml": [-130, -1200, -120, -110],
            "away_ml": [110, 800, 100, -110],
            "line": [6.5, 6.5, 6.0, 5.5],
            "over_price": [-110, -110, -110, None],
            "under_price": [-110, -110, -110, -110]})
        out = dk_market(rows)
        assert out["game_id"].tolist() == [1]
        assert out["p_over"].iloc[0] == pytest.approx(0.5)


class TestOvertimeFit:
    def test_fit_recovers_beta(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.3, 0.75, 20000)
        y = rng.random(20000) < 0.5 + 0.4 * (p - 0.5)
        assert fit_ot_beta(p, y) == pytest.approx(0.4, abs=0.1)

    def _ot_frame(self, n=3000, seed=1):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        he = rng.normal(1500, 60, n)
        ae = rng.normal(1500, 60, n)
        home_won = rng.random(n) < 0.5
        return pd.DataFrame({
            "game_id": np.arange(n), "season": 2021, "date": dates,
            "home_score": np.where(home_won, 3, 2),
            "away_score": np.where(home_won, 2, 3),
            "is_ot": rng.random(n) < 0.3, "home_elo": he, "away_elo": ae})

    def test_beta_uses_only_games_before_the_cutoff(self):
        """Point in time: rewriting or deleting games on or after the
        cutoff leaves the fitted overtime lean unchanged."""
        ot = self._ot_frame()
        cut = pd.Timestamp("2024-06-01")
        before = ot_beta_before(ot, cut)
        later = pd.to_datetime(ot["date"]) >= cut
        flipped = ot.copy()
        flipped.loc[later, ["home_score", "away_score"]] = \
            flipped.loc[later, ["away_score", "home_score"]].to_numpy()
        flipped.loc[later, "is_ot"] = True
        assert ot_beta_before(flipped, cut) == before
        assert ot_beta_before(ot[~later], cut) == before


class TestScoring:
    def test_clustered_se_equals_paired_se_for_singletons(self):
        rng = np.random.default_rng(3)
        d = rng.normal(0, 1, 500)
        assert clustered_se(d, np.arange(500)) == pytest.approx(
            d.std(ddof=1) / np.sqrt(500), rel=1e-9)

    def test_clustered_se_widens_with_duplicated_clusters(self):
        rng = np.random.default_rng(4)
        d = rng.normal(0, 1, 300)
        dup = np.repeat(d, 2)
        assert clustered_se(dup, np.repeat(np.arange(300), 2)) > \
            dup.std(ddof=1) / np.sqrt(600)

    def test_score_and_summarize_on_synthetic_games(self):
        data = pd.DataFrame({
            "game_id": [1, 2, 3], "season": [20242025] * 3,
            "p_home": [0.6, 0.45, 0.52], "line": [6.5, 5.5, 6.5],
            "p_over": [0.5, 0.48, 0.55], "home_score": [4, 1, 3],
            "away_score": [3, 2, 3],
            "lam_h": [3.1, 2.8, 3.0], "lam_a": [2.8, 3.0, 2.9],
            "env_h": [3.0] * 3, "env_a": [2.8] * 3,
            "w0": [1.15] * 3, "w1": [0.51] * 3, "w2": [0.72] * 3,
            "w3": [1.34] * 3, "w4": [1.0] * 3, "beta": [0.3] * 3,
            "p_lgbm": [0.58, 0.47, 0.5]})
        for v in sgp.VARIANTS:
            s = sgp.score_games(data, v)
            assert len(s) == 3 and np.isfinite(s["ll_joint"]).all()
            if v in ("A", "C", "D", "E"):
                assert s["p_over_grid"].to_numpy() == pytest.approx(
                    data["p_over"].to_numpy(), abs=1e-7)
            src = data["p_lgbm"] if v == "F" else data["p_home"]
            assert s["p_home_grid"].to_numpy() == pytest.approx(
                src.to_numpy(), abs=1e-7)
        summ = sgp.summarize(sgp.score_games(data, "A"))
        assert summ["n"] == 3 and 20242025 in summ["by_season"]


def test_power_if_true_is_small_for_a_weak_link():
    data = pd.DataFrame({
        "game_id": range(200), "season": 20242025,
        "p_home": np.linspace(0.35, 0.7, 200), "line": 6.5, "p_over": 0.5,
        "lam_h": 3.0, "lam_a": 2.9, "env_h": 3.0, "env_a": 2.8,
        "w0": 1.15, "w1": 0.51, "w2": 0.72, "w3": 1.34, "w4": 1.0,
        "beta": 0.3})
    p = sgp.power_if_true(data, "A")
    assert p["n"] == 200 and p["expected_diff"] < 0      # joint helps if true
    assert 0 < p["power"] < 0.5 and p["games_needed"] > 200


def test_failed_gate_keeps_the_checker_withholding():
    """The pre-registered result (STATUS): the joint pricer is not used."""
    assert sgp.GATE_PASSED is False


def test_cli_without_evaluate_runs_nothing(capsys):
    assert sgp.main([]) is None
    assert "--evaluate" in capsys.readouterr().out


@requires_db
class TestClosingQuotesPointInTime:
    """A snapshot taken at or after puck drop never enters the closing
    consensus (inside a rolled-back transaction)."""

    def test_post_start_snapshot_is_ignored(self):
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                g = conn.execute(text("""
                    SELECT o.game_id, g.start_time_utc AT TIME ZONE 'UTC' AS st
                    FROM raw.odds_history o JOIN raw.games g USING (game_id)
                    WHERE g.season = :s AND g.game_state IN ('FINAL','OFF')
                    LIMIT 1"""), {"s": sgp.VALIDATION_SEASONS[0]}).fetchone()
                if g is None:
                    pytest.skip("no 2024-25 odds history in this database")
                before = sgp.load_validation_market(conn)
                row0 = before.set_index("game_id").loc[g.game_id]
                late = g.st + dt.timedelta(minutes=1)
                for side, price in (("home", 900), ("away", -2000)):
                    conn.execute(text("""
                        INSERT INTO raw.odds_history (snapshot_ts, requested_ts,
                            event_id, game_id, book, market, side, price)
                        VALUES (:ts, :ts, 'dbtest-sgp', :g, 'pinnacle', 'h2h',
                                :side, :p)"""),
                        {"ts": late, "g": g.game_id, "side": side, "p": price})
                after = sgp.load_validation_market(conn)
                row1 = after.set_index("game_id").loc[g.game_id]
                assert row1["p_home"] == pytest.approx(row0["p_home"])
            finally:
                trans.rollback()
