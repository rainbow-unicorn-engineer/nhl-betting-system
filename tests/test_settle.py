"""
Tests for betting/settle.py — synthetic paper bets on real historical
games; every P&L and CLV number hand-computed. All inserts cleaned up.
"""
import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


@requires_db
class TestPaperSettlement:
    @pytest.fixture()
    def scenario(self):
        """Two FINAL 2020-21 games: rec on the winner (+110) and rec on
        the loser (-110), plus closing snapshots (same book -125 for the
        first bet's side; none for the second -> consensus fallback)."""
        with engine.begin() as conn:
            games = conn.execute(text("""
                SELECT game_id, date, home_team, away_team,
                       (home_score > away_score) AS home_won
                FROM raw.games
                WHERE season = 20202021 AND game_state IN ('FINAL','OFF')
                ORDER BY game_id LIMIT 2
            """)).fetchall()
            g1, g2 = games
            side1 = "HOME" if g1.home_won else "AWAY"      # winning side
            side2 = "HOME" if g2.home_won else "AWAY"
            side2 = "AWAY" if side2 == "HOME" else "HOME"  # losing side
            rec_ids = []
            for g, side, price, stake in ((g1, side1, 110, 10.0),
                                          (g2, side2, -110, 11.0)):
                rec_ids.append(conn.execute(text("""
                    INSERT INTO betting.recommendations
                        (game_id, market_type, side, model_prob, best_book,
                         best_price, implied_prob_novig, edge_pct,
                         kelly_fraction, recommended_stake, status)
                    VALUES (:g, 'ml', :side, 0.55, 'testbook', :price,
                            0.50, 0.05, 0.10, :stake, 'PENDING')
                    RETURNING rec_id
                """), {"g": g.game_id, "side": side, "price": price,
                       "stake": stake}).scalar())
            # closing snapshot for game 1, same book: side1 at -125
            h1 = -125 if side1 == "HOME" else 150
            a1 = 150 if side1 == "HOME" else -125
            conn.execute(text("""
                INSERT INTO raw.odds_snapshots
                    (game_id, captured_at, book_name, market_type,
                     home_price, away_price)
                VALUES (:g, :t, 'testbook', 'ml', :h, :a)
            """), {"g": g1.game_id, "t": dt.datetime(2021, 1, 1, 12),
                   "h": h1, "a": a1})
        yield {"g1": g1, "g2": g2, "side1": side1, "side2": side2,
               "rec_ids": rec_ids}
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM betting.placed_bets "
                              "WHERE rec_id = ANY(:r)"), {"r": rec_ids})
            conn.execute(text("DELETE FROM betting.recommendations "
                              "WHERE rec_id = ANY(:r)"), {"r": rec_ids})
            conn.execute(text("DELETE FROM raw.odds_snapshots "
                              "WHERE book_name = 'testbook'"))
            conn.execute(text("DELETE FROM betting.bankroll_log"))

    def test_settles_results_pnl_and_clv(self, scenario):
        from betting.settle import settle_paper
        n = settle_paper()
        assert n == 2

        with engine.connect() as conn:
            bets = pd.read_sql(text("""
                SELECT * FROM betting.placed_bets
                WHERE rec_id = ANY(:r) ORDER BY rec_id
            """), conn, params={"r": scenario["rec_ids"]})
            recs = pd.read_sql(text("""
                SELECT status FROM betting.recommendations
                WHERE rec_id = ANY(:r)
            """), conn, params={"r": scenario["rec_ids"]})

        b1, b2 = bets.iloc[0], bets.iloc[1]
        # bet 1: winner at +110, stake 10 -> +11.00
        assert b1["result"] == "WIN"
        assert float(b1["pnl"]) == pytest.approx(11.00)
        # same-book closing -125: clv = 0.5556 - implied(+110)=0.4762
        assert int(b1["closing_line"]) == -125
        assert float(b1["clv"]) == pytest.approx(
            125 / 225 - 100 / 210, abs=1e-3)
        # bet 2: loser at -110, stake 11 -> -11.00; no snapshots -> no clv
        assert b2["result"] == "LOSS"
        assert float(b2["pnl"]) == pytest.approx(-11.00)
        assert b2["clv"] is None or pd.isna(b2["clv"])
        assert (recs["status"] == "SETTLED").all()

    def test_idempotent_and_bankroll_ledger(self, scenario):
        from betting.recommend import BANKROLL
        from betting.settle import settle_paper
        settle_paper()
        assert settle_paper() == 0     # second run settles nothing new

        with engine.connect() as conn:
            n_bets = conn.execute(text("""
                SELECT COUNT(*) FROM betting.placed_bets
                WHERE rec_id = ANY(:r)"""),
                {"r": scenario["rec_ids"]}).scalar()
            log = pd.read_sql(text("""
                SELECT * FROM betting.bankroll_log ORDER BY date
            """), conn)
        assert n_bets == 2             # no duplicates
        assert not log.empty
        assert float(log.iloc[0]["opening_balance"]) == pytest.approx(BANKROLL)
        # net pnl across the ledger = +11 - 11 = 0
        assert float(log["gross_pnl"].sum()) == pytest.approx(0.0)
        assert float(log.iloc[-1]["closing_balance"]) == pytest.approx(BANKROLL)
        assert int(log["total_bets"].sum()) == 2

    def test_clv_report_buckets(self, scenario):
        from betting.settle import clv_report, settle_paper
        settle_paper()
        report = clv_report()
        assert report is not None
        assert int(report.loc["ALL", "bets"]) == 2
        # both recs claimed 5% edge -> the 4-6% bucket holds both
        assert int(report.loc["4-6%", "bets"]) == 2
