"""
Tests for betting/checker.py — hand-computed EVs, push semantics, parlay
math, and a DB round-trip against synthetic prediction rows.
"""
import datetime as dt

import numpy as np
import pytest
from sqlalchemy import text

import betting.checker as checker
from betting.checker import (Leg, evaluate_leg, evaluate_parlay, parse_leg)
from config.settings import check_db_connection, engine

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


class TestParse:
    def test_ml_leg(self):
        l = parse_leg("bos@tor ml away -125")
        assert (l.away, l.home, l.market, l.side, l.price) == \
            ("BOS", "TOR", "ml", "AWAY", -125)

    def test_total_leg(self):
        l = parse_leg("BOS@TOR total over 6.5 +105")
        assert l.market == "total" and l.side == "OVER"
        assert l.line == 6.5 and l.price == 105

    def test_bad_specs(self):
        with pytest.raises(ValueError):
            parse_leg("BOS@TOR ml -125")            # missing side
        with pytest.raises(ValueError):
            parse_leg("BOS@TOR spread home -110")   # unknown market


class TestLegMath:
    @pytest.fixture
    def patched(self, monkeypatch):
        """Route probability lookup to a canned value; game always found."""
        monkeypatch.setattr(checker, "_resolve_game", lambda conn, leg: 1)
        canned = {}
        monkeypatch.setattr(checker, "_model_probability",
                            lambda conn, leg: canned.get("p"))
        return canned

    def test_positive_ev_ml_bet(self, patched):
        patched["p"] = (0.55, 0.0)
        l = evaluate_leg(parse_leg("A@B ml home +100"))
        assert l.ev == pytest.approx(0.55 * 1.0 - 0.45)   # +0.10/unit
        assert l.edge == pytest.approx(0.05)
        assert l.verdict == "BET"

    def test_negative_ev_pass(self, patched):
        patched["p"] = (0.50, 0.0)
        l = evaluate_leg(parse_leg("A@B ml home -110"))
        assert l.ev < 0 and l.verdict == "PASS"

    def test_thin_edge(self, patched):
        patched["p"] = (0.51, 0.0)     # +EV at +100 but edge 1% < 2.5%
        l = evaluate_leg(parse_leg("A@B ml home +100"))
        assert l.ev > 0 and l.verdict == "THIN"

    def test_push_counts_as_stake_returned(self, patched):
        # integer total line: win .40, push .15, lose .45 at +100
        patched["p"] = (0.40, 0.15)
        l = evaluate_leg(parse_leg("A@B total over 6 +100"))
        assert l.ev == pytest.approx(0.40 * 1.0 - 0.45)
        # conditional-on-action probability vs implied 0.5
        assert l.edge == pytest.approx(0.40 / 0.85 - 0.5)

    def test_totals_never_earn_full_bet_verdict(self, patched):
        # totals gate not passed: even a huge edge stays THIN
        patched["p"] = (0.60, 0.0)
        l = evaluate_leg(parse_leg("A@B total over 6.5 +100"))
        assert l.edge > checker.EDGE_MIN_TOTAL
        assert l.verdict == "THIN"


class TestParlayMath:
    @pytest.fixture
    def patched(self, monkeypatch):
        games = iter([1, 2, 3])
        monkeypatch.setattr(checker, "_resolve_game",
                            lambda conn, leg: next(games))
        probs = {}
        monkeypatch.setattr(checker, "_model_probability",
                            lambda conn, leg: probs[(leg.side, leg.market)])
        return probs

    def test_two_leg_parlay_hand_computed(self, patched):
        patched[("HOME", "ml")] = (0.6, 0.0)
        patched[("AWAY", "ml")] = (0.5, 0.0)
        r = evaluate_parlay([parse_leg("A@B ml home +100"),
                             parse_leg("C@D ml away +100")])
        assert r["p_win_all"] == pytest.approx(0.30)
        assert r["decimal_fair_product"] == pytest.approx(4.0)
        # E[mult] = (0.6*2) * (0.5*2) = 1.2 -> EV +0.2
        assert r["ev_per_unit"] == pytest.approx(0.20)
        # kelly = (3*0.3 - 0.7)/3
        assert r["kelly"] == pytest.approx((3 * 0.3 - 0.7) / 3)
        assert r["stake_pct"] == pytest.approx(r["kelly"] / 4)

    def test_push_leg_multiplier(self, patched):
        patched[("HOME", "ml")] = (0.6, 0.0)
        patched[("OVER", "total")] = (0.5, 0.1)
        r = evaluate_parlay([parse_leg("A@B ml home +100"),
                             parse_leg("C@D total over 6 +100")])
        # E[mult] = (0.6*2) * (0.5*2 + 0.1) = 1.2 * 1.1 = 1.32
        assert r["ev_per_unit"] == pytest.approx(0.32)

    def test_boosted_price_scales_multiplier(self, patched):
        patched[("HOME", "ml")] = (0.6, 0.0)
        patched[("AWAY", "ml")] = (0.5, 0.0)
        r = evaluate_parlay([parse_leg("A@B ml home +100"),
                             parse_leg("C@D ml away +100")],
                            combined_price=350)   # dec 4.5 vs fair 4.0
        assert r["ev_per_unit"] == pytest.approx(1.2 * 4.5 / 4.0 - 1.0)

    def test_same_game_legs_withhold_verdict(self, monkeypatch):
        monkeypatch.setattr(checker, "_resolve_game", lambda conn, leg: 7)
        monkeypatch.setattr(checker, "_model_probability",
                            lambda conn, leg: (0.6, 0.0))
        r = evaluate_parlay([parse_leg("A@B ml home +100"),
                             parse_leg("A@B total over 6.5 +100")])
        assert r["correlated"] is True
        assert r["verdict"] == "NO-MODEL"   # withheld
        assert any("same-game" in n for n in r["notes"])


@requires_db
class TestDatabaseRoundTrip:
    """Synthetic prediction rows -> checker reads them back correctly."""

    @pytest.fixture()
    def seeded_game(self):
        from models.lgbm import MODEL_NAME as ML_NAME
        from models.totals import (MODEL_NAME as T_NAME, poisson_pmf)
        with engine.begin() as conn:
            g = conn.execute(text("""
                SELECT game_id, date, home_team, away_team FROM raw.games
                WHERE season = 20202021 AND game_state IN ('FINAL','OFF')
                ORDER BY game_id LIMIT 1""")).fetchone()
            ml_id = conn.execute(text("""
                SELECT model_id FROM models.model_registry
                WHERE model_name = :n ORDER BY model_id DESC LIMIT 1
            """), {"n": ML_NAME}).scalar()
            t_id = conn.execute(text("""
                SELECT model_id FROM models.model_registry
                WHERE model_name = :n ORDER BY model_id DESC LIMIT 1
            """), {"n": T_NAME}).scalar()
            pmf_h = [float(x) for x in poisson_pmf(np.array([3.0]))[0]]
            pmf_a = [float(x) for x in poisson_pmf(np.array([2.7]))[0]]
            conn.execute(text("""
                INSERT INTO models.predictions
                    (game_id, model_id, market_type, home_win_prob, away_win_prob)
                VALUES (:g, :m, 'ml', 0.62, 0.38)
                ON CONFLICT (game_id, model_id, market_type) DO UPDATE
                    SET home_win_prob = 0.62, away_win_prob = 0.38
            """), {"g": g.game_id, "m": ml_id})
            conn.execute(text("""
                INSERT INTO models.predictions
                    (game_id, model_id, market_type, home_goals_pmf, away_goals_pmf)
                VALUES (:g, :m, 'total', :ph, :pa)
                ON CONFLICT (game_id, model_id, market_type) DO UPDATE
                    SET home_goals_pmf = :ph, away_goals_pmf = :pa
            """), {"g": g.game_id, "m": t_id, "ph": pmf_h, "pa": pmf_a})
        yield g
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM models.predictions
                WHERE game_id = :g AND model_id IN (:m1, :m2)
            """), {"g": g.game_id, "m1": ml_id, "m2": t_id})

    def test_ml_leg_reads_stored_probability(self, seeded_game):
        g = seeded_game
        l = evaluate_leg(Leg(away=g.away_team, home=g.home_team, market="ml",
                             side="HOME", price=-110, date=g.date))
        assert l.game_id == g.game_id
        assert l.p_win == pytest.approx(0.62)
        assert l.verdict == "BET"

    def test_total_leg_prices_any_line_from_pmfs(self, seeded_game):
        from models.totals import poisson_pmf, prob_over, total_pmf
        g = seeded_game
        l = evaluate_leg(Leg(away=g.away_team, home=g.home_team,
                             market="total", side="OVER", line=6.5,
                             price=100, date=g.date))
        tp = total_pmf(poisson_pmf(np.array([3.0])), poisson_pmf(np.array([2.7])))
        expect, _ = prob_over(tp, [6.5])
        assert l.p_win == pytest.approx(float(expect[0]), abs=1e-9)
