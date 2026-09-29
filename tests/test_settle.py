"""
Tests for betting/settle.py — the void rule and CLV arithmetic (pure),
then synthetic paper bets on real historical games (DB); every P&L and
CLV number hand-computed. Inserts are cleaned up and the bankroll ledger
is rebuilt, not emptied, afterwards.
"""
import datetime as dt
from contextlib import contextmanager

import pandas as pd
import pytest
from sqlalchemy import text

from betting.settle import compute_clv, void_reason
from config.settings import check_db_connection, engine

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")

UTC = dt.timezone.utc


def _rebuild_ledger():
    from betting.settle import rebuild_bankroll_log
    rebuild_bankroll_log()


class TestVoidReason:
    START = dt.datetime(2026, 11, 3, 0, 0, tzinfo=UTC)     # aware, like start_time_utc

    def test_postponed_and_cancelled_void(self):
        assert void_reason("PPD", self.START, None, None, is_final=False) == "postponed"
        assert void_reason("CNCL", None, None, None, is_final=False) == "cancelled"

    def test_ok_and_suspended_do_not_void_on_state(self):
        priced = dt.datetime(2026, 11, 2, 15, 0)            # naive UTC, 9h before
        assert void_reason("OK", self.START, priced, None) is None
        assert void_reason("SUSP", self.START, priced, None) is None
        assert void_reason(None, self.START, priced, None) is None

    def test_game_played_more_than_36h_after_pricing_voids(self):
        priced = dt.datetime(2026, 11, 1, 11, 0)            # 37h before
        assert "rescheduled" in void_reason("OK", self.START, priced, None)
        at_36h = dt.datetime(2026, 11, 1, 12, 0)            # exactly 36h: kept
        assert void_reason("OK", self.START, at_36h, None) is None

    def test_created_at_stands_in_for_a_missing_priced_at(self):
        created = pd.Timestamp("2026-10-30 12:00")          # days before
        assert void_reason("OK", self.START, None, created) is not None
        assert void_reason("OK", self.START, pd.NaT, created) is not None

    def test_offsets_are_compared_as_instants(self):
        # 20:00 at UTC-5 on Nov 2 is 01:00 UTC Nov 3: 30h after pricing
        est = dt.timezone(dt.timedelta(hours=-5))
        start = dt.datetime(2026, 11, 2, 20, 0, tzinfo=est)
        priced = dt.datetime(2026, 11, 1, 19, 0)
        assert void_reason("OK", start, priced, None) is None

    def test_no_start_time_means_no_move_check(self):
        assert void_reason("OK", None, dt.datetime(2020, 1, 1), None) is None
        assert void_reason("OK", None, None, None,
                           scheduled_start=self.START) is None


class TestVoidReasonScheduledStart:
    """With the start stored at issue (scheduled_start), a moved game is
    one whose puck drop is more than 3 hours away from it, however early
    the pick was priced."""
    START = dt.datetime(2026, 11, 3, 0, 0, tzinfo=UTC)
    PRICED = dt.datetime(2026, 11, 2, 14, 0)               # naive UTC, 10h before

    def test_moved_game_voids(self):
        # postponed from Nov 3 to Dec 9 and played there
        moved = dt.datetime(2026, 12, 9, 0, 30, tzinfo=UTC)
        reason = void_reason("OK", moved, self.PRICED, None,
                             scheduled_start=self.START)
        assert "rescheduled" in reason
        # moved earlier counts too
        earlier = self.START - dt.timedelta(hours=4)
        assert void_reason("OK", earlier, self.PRICED, None,
                           scheduled_start=self.START) is not None

    def test_a_time_change_within_3_hours_is_not_a_move(self):
        for shift in (dt.timedelta(0), dt.timedelta(minutes=30),
                      dt.timedelta(hours=3), -dt.timedelta(hours=3)):
            assert void_reason("OK", self.START + shift, self.PRICED, None,
                               scheduled_start=self.START) is None, shift
        assert void_reason("OK", self.START + dt.timedelta(hours=3, minutes=1),
                           self.PRICED, None,
                           scheduled_start=self.START) is not None

    def test_unmoved_day_ahead_pick_is_not_voided(self):
        # betting.recommend --date <tomorrow>: priced 40h before puck drop
        priced = dt.datetime(2026, 11, 1, 8, 0)
        assert void_reason("OK", self.START, priced, None,
                           scheduled_start=self.START) is None
        # the 36-hour rule alone would have voided it
        assert void_reason("OK", self.START, priced, None) is not None

    def test_null_scheduled_start_falls_back_to_the_36_hour_rule(self):
        priced_37h = dt.datetime(2026, 11, 1, 11, 0)
        priced_36h = dt.datetime(2026, 11, 1, 12, 0)
        for missing in (None, pd.NaT, float("nan")):
            assert "rescheduled" in void_reason(
                "OK", self.START, priced_37h, None, scheduled_start=missing)
            assert void_reason("OK", self.START, priced_36h, None,
                               scheduled_start=missing) is None

    def test_scheduled_start_offsets_are_compared_as_instants(self):
        # 19:00 at UTC-5 on Nov 2 is the same instant as 00:00 UTC Nov 3
        est = dt.timezone(dt.timedelta(hours=-5))
        scheduled = dt.datetime(2026, 11, 2, 19, 0, tzinfo=est)
        assert void_reason("OK", self.START, self.PRICED, None,
                           scheduled_start=scheduled) is None
        assert void_reason("OK", self.START, self.PRICED, None,
                           scheduled_start=pd.Timestamp(scheduled)) is None

    def test_state_still_decides_first_and_unfinished_games_never_move(self):
        moved = dt.datetime(2026, 12, 9, 0, 30, tzinfo=UTC)
        assert void_reason("PPD", self.START, self.PRICED, None,
                           scheduled_start=self.START,
                           is_final=False) == "postponed"
        assert void_reason("OK", moved, self.PRICED, None,
                           scheduled_start=self.START, is_final=False) is None


class TestComputeClv:
    def test_same_book_close_uses_both_vig_prices(self):
        # close -125 against placed +110
        assert compute_clv(-125, 125 / 225, 110, 0.47) == pytest.approx(
            round(125 / 225 - 100 / 210, 4))

    def test_consensus_close_is_compared_with_the_pick_no_vig_prob(self):
        # the consensus close is no-vig, so subtract the no-vig fair
        # probability of the pick, not the vig-inclusive placed price
        assert compute_clv(None, 0.52, -110, 0.50) == pytest.approx(0.02)
        vig_biased = 0.52 - 110 / 210
        assert compute_clv(None, 0.52, -110, 0.50) != pytest.approx(vig_biased)

    def test_unknown(self):
        assert compute_clv(None, None, 110, 0.5) is None
        assert compute_clv(None, 0.52, 110, None) is None
        assert compute_clv(None, 0.52, 110, float("nan")) is None


def test_clv_report_upgrades_the_schema_first(monkeypatch):
    """clv_report reads placed_bets.is_paper, a column an old database
    lacks until ensure_schema adds it."""
    from betting import settle
    order = []

    class _Engine:
        @contextmanager
        def connect(self):
            order.append("query")
            yield None

    monkeypatch.setattr(settle, "ensure_schema", lambda: order.append("schema"))
    monkeypatch.setattr(settle, "db", _Engine())
    monkeypatch.setattr(settle.pd, "read_sql", lambda *a, **k: pd.DataFrame(
        columns=["pnl", "stake_amount", "clv", "edge_pct"]))
    assert settle.clv_report() is None
    assert order == ["schema", "query"]


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
        _rebuild_ledger()

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
        # settled_at is naive UTC, like captured_at and priced_at
        utc_now = dt.datetime.now(UTC).replace(tzinfo=None)
        for settled_at in bets["settled_at"]:
            assert abs(pd.Timestamp(settled_at).to_pydatetime() - utc_now) \
                < dt.timedelta(minutes=10)

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
        # only bet 1 had a closing snapshot
        assert int(report.loc["ALL", "with_clv"]) == 1


@requires_db
class TestClosingWindow:
    """The close is the same book's last snapshot strictly after the pick
    was priced and strictly before puck drop."""

    START = dt.datetime(2021, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
    T_PICK = dt.datetime(2021, 1, 14, 15, 0)      # naive UTC, like captured_at
    T_CLOSE = dt.datetime(2021, 1, 14, 23, 30)
    T_LIVE = dt.datetime(2021, 1, 15, 1, 0)       # after puck drop

    @pytest.fixture()
    def game(self):
        from config.migrate import ensure_schema
        ensure_schema()
        with engine.begin() as conn:
            g = conn.execute(text("""
                SELECT game_id, start_time_utc,
                       (home_score > away_score) AS home_won
                FROM raw.games
                WHERE season = 20202021 AND game_state IN ('FINAL','OFF')
                ORDER BY game_id LIMIT 1
            """)).one()
            conn.execute(text("UPDATE raw.games SET start_time_utc = :s "
                              "WHERE game_id = :g"),
                         {"s": self.START, "g": g.game_id})
            for t, h, a in ((self.T_PICK, 110, -130),      # the pick's own quote
                            (self.T_CLOSE, -125, 105),     # pre-game close
                            (self.T_LIVE, -400, 300)):     # in-play: never the close
                conn.execute(text("""
                    INSERT INTO raw.odds_snapshots
                        (game_id, captured_at, book_name, market_type,
                         home_price, away_price)
                    VALUES (:g, :t, 'testbook', 'ml', :h, :a)
                """), {"g": g.game_id, "t": t, "h": h, "a": a})
        yield g
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.odds_snapshots "
                              "WHERE book_name = 'testbook'"))
            conn.execute(text("UPDATE raw.games SET start_time_utc = :s "
                              "WHERE game_id = :g"),
                         {"s": g.start_time_utc, "g": g.game_id})

    def test_close_is_after_pick_and_before_puck_drop(self, game):
        from betting.settle import closing_quote
        price, implied = closing_quote(game.game_id, "testbook", "HOME",
                                       priced_at=self.T_PICK)
        assert price == -125
        assert implied == pytest.approx(125 / 225)

    def test_pick_never_grades_itself(self, game):
        from betting.settle import closing_quote
        # priced from the last pre-game quote: nothing qualifies -> unknown
        assert closing_quote(game.game_id, "testbook", "HOME",
                             priced_at=self.T_CLOSE) == (None, None)

    def test_consensus_fallback_uses_the_same_window(self, game):
        """A book with no close of its own falls back to the consensus
        no-vig close in the same window, and its CLV is like for like:
        that no-vig close minus the no-vig fair probability stored with
        the pick (implied_prob_novig), with closing_line left NULL."""
        from betting.settle import closing_quote, settle_paper
        price, implied = closing_quote(game.game_id, "otherbook", "AWAY",
                                       priced_at=self.T_PICK)
        ph, pa = 125 / 225, 100 / 205                 # the testbook close only
        consensus_away = 1 - ph / (ph + pa)
        assert price is None
        assert implied == pytest.approx(consensus_away)

        with engine.begin() as conn:
            rec_id = conn.execute(text("""
                INSERT INTO betting.recommendations
                    (game_id, market_type, side, model_prob, best_book,
                     best_price, implied_prob_novig, edge_pct,
                     kelly_fraction, recommended_stake, status, priced_at)
                VALUES (:g, 'ml', 'AWAY', 0.50, 'otherbook', 105, 0.4400,
                        0.06, 0.10, 10.0, 'PENDING', :t)
                RETURNING rec_id
            """), {"g": game.game_id, "t": self.T_PICK}).scalar()
        try:
            settle_paper()
            with engine.connect() as conn:
                bet = conn.execute(text("""
                    SELECT closing_line, clv FROM betting.placed_bets
                    WHERE rec_id = :r"""), {"r": rec_id}).one()
            assert bet.closing_line is None
            # 0.44 is the no-vig side probability stored at issue; the old
            # formula (minus implied(+105) = 0.4878) read about 0.05 lower
            assert float(bet.clv) == pytest.approx(consensus_away - 0.44, abs=1e-3)
        finally:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM betting.placed_bets "
                                  "WHERE rec_id = :r"), {"r": rec_id})
                conn.execute(text("DELETE FROM betting.recommendations "
                                  "WHERE rec_id = :r"), {"r": rec_id})
            _rebuild_ledger()

    def test_frozen_pick_settles_with_null_clv_not_zero(self, game):
        """A pick priced from the only pre-game snapshot has no later quote:
        CLV must be NULL (unknown), and placed_at is the priced_at."""
        from betting.settle import settle_paper
        side = "HOME" if game.home_won else "AWAY"
        price = 110 if side == "HOME" else -130
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.odds_snapshots WHERE "
                              "book_name = 'testbook' AND captured_at <> :t"),
                         {"t": self.T_PICK})
            rec_id = conn.execute(text("""
                INSERT INTO betting.recommendations
                    (game_id, market_type, side, model_prob, best_book,
                     best_price, implied_prob_novig, edge_pct,
                     kelly_fraction, recommended_stake, status, priced_at)
                VALUES (:g, 'ml', :side, 0.55, 'testbook', :price, 0.50,
                        0.05, 0.10, 10.0, 'PENDING', :t)
                RETURNING rec_id
            """), {"g": game.game_id, "side": side, "price": price,
                   "t": self.T_PICK}).scalar()
        try:
            settle_paper()
            with engine.connect() as conn:
                bet = conn.execute(text("""
                    SELECT placed_at, closing_line, clv FROM betting.placed_bets
                    WHERE rec_id = :r"""), {"r": rec_id}).one()
            assert bet.clv is None and bet.closing_line is None
            assert bet.placed_at == self.T_PICK
        finally:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM betting.placed_bets "
                                  "WHERE rec_id = :r"), {"r": rec_id})
                conn.execute(text("DELETE FROM betting.recommendations "
                                  "WHERE rec_id = :r"), {"r": rec_id})
            _rebuild_ledger()


@requires_db
class TestVoids:
    """Picks on a postponed game, on a game played more than 3 hours away
    from the start stored with the pick, or (no start stored) more than
    36 hours after the pick was priced, settle VOID: pnl 0, no CLV, and
    they count as no bet anywhere (ledger, CLV report, the day budget).
    A pick made a day ahead on a game that didn't move settles normally."""

    START = dt.datetime(2021, 2, 1, 0, 0, tzinfo=UTC)
    ORIGINAL = dt.datetime(2021, 1, 27, 0, 0, tzinfo=UTC)   # before the move
    NEW_START = dt.datetime(2021, 2, 20, 0, 30, tzinfo=UTC)  # the replay date
    NAMES = ("ppd", "moved", "legacy", "ahead", "ok")

    @pytest.fixture()
    def voids(self):
        from config.migrate import ensure_schema
        ensure_schema()
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT game_id, date, start_time_utc, schedule_state,
                       game_state, home_score, away_score,
                       (home_score > away_score) AS home_won
                FROM raw.games
                WHERE season = 20202021 AND game_state IN ('FINAL','OFF')
                ORDER BY game_id OFFSET 10 LIMIT 5
            """)).fetchall()
            games = dict(zip(self.NAMES, rows))
            # The postponed game is not played: upcoming, no score, so
            # only the PPD/CNCL branch of the settlement query can pick it
            conn.execute(text("""
                UPDATE raw.games SET schedule_state = 'PPD', game_state = 'FUT',
                       home_score = NULL, away_score = NULL, start_time_utc = :s
                WHERE game_id = :g"""),
                         {"s": self.START, "g": games["ppd"].game_id})
            for name in ("moved", "legacy", "ahead"):
                conn.execute(text("UPDATE raw.games SET start_time_utc = :s "
                                  "WHERE game_id = :g"),
                             {"s": self.START, "g": games[name].game_id})
            forty_h_before = dt.datetime(2021, 1, 30, 8, 0)     # naive UTC
            rec_ids = {}
            for name, priced_at, scheduled in (
                    ("ppd", None, None),
                    # written for Jan 27, played Feb 1: moved (> 3h)
                    ("moved", dt.datetime(2021, 1, 26, 15, 0), self.ORIGINAL),
                    # no start stored: the 36-hour fallback voids it
                    ("legacy", forty_h_before, None),
                    # made a day ahead for the Feb 1 start, which held
                    ("ahead", forty_h_before, self.START),
                    ("ok", None, None)):
                rec_ids[name] = conn.execute(text("""
                    INSERT INTO betting.recommendations
                        (game_id, market_type, side, model_prob, best_book,
                         best_price, implied_prob_novig, edge_pct,
                         kelly_fraction, recommended_stake, status, priced_at,
                         scheduled_start)
                    VALUES (:g, 'ml', 'HOME', 0.55, 'voidbook', 110, 0.50,
                            0.05, 0.10, 12.0, 'PENDING', :t, :s)
                    RETURNING rec_id
                """), {"g": games[name].game_id, "t": priced_at,
                       "s": scheduled}).scalar()
        yield {"games": games, "rec_ids": rec_ids}
        with engine.begin() as conn:
            # every pick on the synthetic book, including any a test added
            conn.execute(text("""
                DELETE FROM betting.placed_bets WHERE rec_id IN (
                    SELECT rec_id FROM betting.recommendations
                    WHERE best_book = 'voidbook')"""))
            conn.execute(text("DELETE FROM betting.recommendations "
                              "WHERE best_book = 'voidbook'"))
            for g in rows:
                conn.execute(text("""
                    UPDATE raw.games SET schedule_state = :ss, game_state = :gs,
                           home_score = :hs, away_score = :aws,
                           start_time_utc = :s
                    WHERE game_id = :g"""),
                             {"ss": g.schedule_state, "gs": g.game_state,
                              "hs": g.home_score, "aws": g.away_score,
                              "s": g.start_time_utc, "g": g.game_id})
        _rebuild_ledger()

    def _bets(self, rec_ids):
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT p.rec_id, p.result, p.pnl, p.clv, p.closing_line,
                       r.status
                FROM betting.placed_bets p
                JOIN betting.recommendations r USING (rec_id)
                WHERE p.rec_id = ANY(:r)
            """), conn, params={"r": list(rec_ids.values())}).set_index("rec_id")

    def test_postponed_and_moved_games_void(self, voids):
        from betting.settle import settle_paper
        settle_paper()
        bets = self._bets(voids["rec_ids"])
        assert len(bets) == 5
        assert (bets["status"] == "SETTLED").all()
        for name in ("ppd", "moved", "legacy"):
            b = bets.loc[voids["rec_ids"][name]]
            assert b["result"] == "VOID", name
            assert float(b["pnl"]) == 0.0
            assert pd.isna(b["clv"]) and pd.isna(b["closing_line"])
        for name in ("ahead", "ok"):
            b = bets.loc[voids["rec_ids"][name]]
            assert b["result"] == ("WIN" if voids["games"][name].home_won
                                   else "LOSS"), name

    def test_voids_are_not_bets_in_the_ledger_or_report(self, voids):
        from betting.settle import clv_report, settle_paper
        settle_paper()
        with engine.connect() as conn:
            real_bets = conn.execute(text("""
                SELECT COUNT(*) FROM betting.placed_bets
                WHERE is_paper AND result IS NOT NULL AND result <> 'VOID'
            """)).scalar()
            ledger = conn.execute(text("""
                SELECT COALESCE(SUM(total_bets), 0),
                       COALESCE(SUM(wins + losses), 0)
                FROM betting.bankroll_log
            """)).one()
        assert int(ledger[0]) == real_bets
        assert int(ledger[1]) == real_bets
        report = clv_report()
        assert int(report.loc["ALL", "bets"]) == real_bets

    def test_voided_pick_commits_no_stake(self, voids):
        from betting.recommend import committed_stake, load_issued_picks
        from betting.settle import settle_paper
        g_ppd = voids["games"]["ppd"]
        before = committed_stake(load_issued_picks(g_ppd.date))
        settle_paper()
        issued = load_issued_picks(g_ppd.date)
        mine = issued[issued["game_id"] == g_ppd.game_id]
        assert mine["voided"].astype(bool).all()
        # every void pick leaves the budget when it shares the date
        voided_that_day = [n for n in ("ppd", "moved", "legacy")
                           if voids["games"][n].date == g_ppd.date]
        assert committed_stake(issued) == pytest.approx(
            before - 12.0 * len(voided_that_day))

    def test_voided_pick_leaves_the_game_open_for_a_new_pick(self, voids):
        """The postponed game is played on a new date: its voided pick no
        longer counts as the game's pick, so a new one is issued, storing
        the new start; and that one is then frozen like any other."""
        from betting.recommend import (frozen_games, load_issued_picks,
                                       write_recommendations)
        from betting.settle import settle_paper
        g = voids["games"]["ppd"]
        assert g.game_id in frozen_games(load_issued_picks(g.date))
        settle_paper()                                 # voids the pick
        assert g.game_id not in frozen_games(load_issued_picks(g.date))
        with engine.begin() as conn:                   # rescheduled
            conn.execute(text("UPDATE raw.games SET schedule_state = 'OK', "
                              "start_time_utc = :s WHERE game_id = :g"),
                         {"s": self.NEW_START, "g": g.game_id})
        new = {"game_id": int(g.game_id), "prediction_id": None,
               "side": "AWAY", "model_prob": 0.55, "best_book": "voidbook",
               "best_price": 120, "implied_prob_novig": 0.50,
               "edge_pct": 0.05, "kelly_fraction": 0.10,
               "recommended_stake": 5.0, "priced_at": None}
        assert write_recommendations([new], [g.game_id], slate_date=g.date) == 1
        assert write_recommendations([dict(new, side="HOME")], [g.game_id],
                                     slate_date=g.date) == 0
        with engine.connect() as conn:
            recs = conn.execute(text("""
                SELECT r.side, r.status, r.scheduled_start, p.result
                FROM betting.recommendations r
                LEFT JOIN betting.placed_bets p USING (rec_id)
                WHERE r.game_id = :g AND r.best_book = 'voidbook'
                ORDER BY r.rec_id"""), {"g": g.game_id}).fetchall()
        assert [(r.side, r.status, r.result) for r in recs] == [
            ("HOME", "SETTLED", "VOID"), ("AWAY", "PENDING", None)]
        assert recs[1].scheduled_start == self.NEW_START
