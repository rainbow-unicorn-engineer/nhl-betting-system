"""
Tests for betting/ledger.py: the settlement math (moneyline, puck line,
totals with pushes, shots-on-goal props, parlays with pushed and void
legs, bonus bets), balances and running P/L (pure), then the database
path on synthetic games: the in-place migration and its idempotence,
recording, settling, hand settlement and balances. Every number is
worked out by hand in the comments.

The database tests follow tests/conftest.py: they skip unless pointed at
a disposable database, and they remove every row they write.
"""
import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import event, text

from betting import ledger
from betting.ledger import (LegInput, Outcome, american_from_decimal,
                            balance_table, combined_price, describe_leg,
                            leg_result, running_pl, slip_outcome, slip_pnl,
                            validate_slip)
from config.settings import check_db_connection, engine

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


@pytest.fixture(autouse=True)
def _bettors(monkeypatch):
    monkeypatch.setenv("BETTORS", "bettor 1,bettor 2")


# ── Odds arithmetic ────────────────────────────────────────────────

class TestOdds:
    def test_american_from_decimal(self):
        assert american_from_decimal(2.5) == 150
        assert american_from_decimal(2.0) == 100
        assert american_from_decimal(1.6667) == -150
        assert american_from_decimal(1.5) == -200
        with pytest.raises(ValueError):
            american_from_decimal(1.0)

    def test_combined_price_multiplies_decimal_odds(self):
        # 1.9091 x 1.9091 = 3.6446 -> +264
        assert combined_price([-110, -110]) == 264
        # 1.6667 x 1.9091 x 2.3 = 7.318 -> +632
        assert combined_price([-150, -110, 130]) == 632
        with pytest.raises(ValueError):
            combined_price([-110, None])

    def test_bettors_come_from_env(self, monkeypatch):
        monkeypatch.setenv("BETTORS", " bettor A , bettor B,bettor A,, ")
        assert ledger.bettors() == ["bettor A", "bettor B"]
        monkeypatch.delenv("BETTORS")
        assert ledger.bettors() == ["bettor 1", "bettor 2"]
        monkeypatch.setenv("BETTORS", " , ")
        assert ledger.bettors() == ["bettor 1", "bettor 2"]


# ── Leg results ────────────────────────────────────────────────────

class TestLegResult:
    def test_moneyline_counts_overtime_and_shootout(self):
        # a 3-2 shootout win: the NHL final already holds the extra goal
        assert leg_result("ml", "HOME", None, 3, 2, True) == "WIN"
        assert leg_result("ml", "AWAY", None, 3, 2, True) == "LOSS"
        assert leg_result("ml", "away", None, 2, 5, True) == "WIN"

    def test_moneyline_tie_or_unfinished_is_undecided(self):
        assert leg_result("ml", "HOME", None, 2, 2, True) is None
        assert leg_result("ml", "HOME", None, None, None, False) is None
        assert leg_result("ml", "HOME", None, 3, 2, False) is None

    def test_postponed_or_cancelled_voids_every_market(self):
        for market, side, line in (("ml", "HOME", None), ("pl", "AWAY", 1.5),
                                   ("total", "OVER", 6.5), ("prop_sog", "OVER", 2.5)):
            assert leg_result(market, side, line, None, None, False, "PPD") == "VOID"
            assert leg_result(market, side, line, None, None, False, "CNCL") == "VOID"
        assert leg_result("ml", "HOME", None, None, None, False, "SUSP") is None

    def test_puck_line(self):
        # home wins 4-2: home -1.5 covers (+0.5), away +1.5 loses (-0.5)
        assert leg_result("pl", "HOME", -1.5, 4, 2, True) == "WIN"
        assert leg_result("pl", "AWAY", 1.5, 4, 2, True) == "LOSS"
        # home wins 3-2 (one goal): home -1.5 loses, away +1.5 wins
        assert leg_result("pl", "HOME", -1.5, 3, 2, True) == "LOSS"
        assert leg_result("pl", "AWAY", 1.5, 3, 2, True) == "WIN"
        # whole-number handicap landing on it pushes
        assert leg_result("pl", "HOME", -1.0, 3, 2, True) == "PUSH"

    def test_totals_with_pushes(self):
        assert leg_result("total", "OVER", 5.5, 4, 2, True) == "WIN"
        assert leg_result("total", "UNDER", 5.5, 4, 2, True) == "LOSS"
        assert leg_result("total", "UNDER", 6.5, 4, 2, True) == "WIN"
        assert leg_result("total", "OVER", 6.0, 4, 2, True) == "PUSH"
        assert leg_result("total", "UNDER", 6.0, 4, 2, True) == "PUSH"

    def test_shots_on_goal_props(self):
        assert leg_result("prop_sog", "OVER", 2.5, 3, 2, True, None, 3, True) == "WIN"
        assert leg_result("prop_sog", "UNDER", 2.5, 3, 2, True, None, 3, True) == "LOSS"
        assert leg_result("prop_sog", "UNDER", 2.5, 3, 2, True, None, 0, True) == "WIN"
        assert leg_result("prop_sog", "OVER", 3.0, 3, 2, True, None, 3, True) == "PUSH"

    def test_prop_waits_for_the_box_score_and_voids_a_player_who_sat(self):
        assert leg_result("prop_sog", "OVER", 2.5, 3, 2, True, None, None, False) is None
        assert leg_result("prop_sog", "OVER", 2.5, 3, 2, True, None, None, True) == "VOID"

    def test_other_is_settled_by_hand(self):
        assert leg_result("other", "TOR to win the Cup", None, 3, 2, True) is None


# ── Slip results ───────────────────────────────────────────────────

class TestSlipOutcome:
    def test_single_bets(self):
        # $15 at -150 wins $10: payout 25
        assert slip_outcome(15, -150, [("WIN", -150)]) == Outcome("WON", 25.0)
        assert slip_outcome(15, -150, [("LOSS", -150)]) == Outcome("LOST", 0.0)
        assert slip_outcome(11, -110, [("PUSH", -110)]) == Outcome("PUSH", 11.0)
        assert slip_outcome(5, -115, [("VOID", -115)]) == Outcome("VOID", 5.0)
        assert slip_outcome(5, -115, [(None, -115)]) == Outcome("OPEN", None)
        # a single's own leg price may be missing: the slip price pays it
        assert slip_outcome(10, 120, [("WIN", None)]).payout == 22.0
        assert slip_outcome(10, 120, [("PUSH", None)]) == Outcome("PUSH", 10.0)

    def test_parlay_all_win_pays_the_exact_product(self):
        # -110/-110 shows as +264 but pays 1.90909^2 = 3.64463 -> $36.45
        assert slip_outcome(10, 264, [("WIN", -110), ("WIN", -110)]).payout == 36.45

    def test_a_losing_leg_loses_the_parlay_at_once(self):
        out = slip_outcome(4, 264, [("LOSS", -105), (None, -120)])
        assert out == Outcome("LOST", 0.0)

    def test_parlay_waits_for_every_leg(self):
        assert slip_outcome(6, 264, [("WIN", -150), (None, 110)]).status == "OPEN"

    def test_pushed_and_void_legs_count_as_a_factor_of_one(self):
        # legs -150 WIN, -110 PUSH, +130 VOID: only the -150 leg is left,
        # 10 x 1.66667 = 16.67
        legs = [("WIN", -150), ("PUSH", -110), ("VOID", 130)]
        out = slip_outcome(10, combined_price([-150, -110, 130]), legs)
        assert out.status == "WON"
        assert out.payout == 16.67
        assert "dropped out" in out.note

    def test_boosted_price_with_a_push_is_reduced_in_proportion(self):
        # boosted +300 (4.0) on two -110 legs, one pushes:
        # 10 x 4.0 / 1.90909 = 20.95
        out = slip_outcome(10, 300, [("WIN", -110), ("PUSH", -110)])
        assert out.payout == 20.95

    def test_no_winning_leg_returns_the_stake(self):
        assert slip_outcome(10, 264, [("PUSH", -110), ("VOID", -110)]) == Outcome("PUSH", 10.0)
        assert slip_outcome(10, 264, [("VOID", -110), ("VOID", -110)]) == Outcome("VOID", 10.0)

    def test_dropped_leg_without_its_own_odds_needs_a_hand_settlement(self):
        out = slip_outcome(10, 300, [("WIN", -110), ("PUSH", None)])
        assert out.status == "OPEN" and out.payout is None
        assert "by hand" in out.note

    def test_bonus_bet_pays_the_profit_only(self):
        # $10 bonus bet at +200: wins $20, the $10 credit is not returned
        assert slip_outcome(10, 200, [("WIN", 200)], is_bonus_bet=True).payout == 20.0
        assert slip_outcome(10, 200, [("LOSS", 200)], is_bonus_bet=True).payout == 0.0
        assert slip_outcome(10, 200, [("PUSH", 200)], is_bonus_bet=True) == Outcome("PUSH", 0.0)

    def test_slip_pnl(self):
        assert slip_pnl("WON", 15, 25) == 10.0
        assert slip_pnl("LOST", 4, 0) == -4.0
        assert slip_pnl("PUSH", 11, 11) == 0.0
        assert slip_pnl("CASHED_OUT", 6, 8.5) == 2.5
        assert slip_pnl("WON", 10, 20, is_bonus_bet=True) == 20.0
        assert slip_pnl("LOST", 10, 0, is_bonus_bet=True) == 0.0
        assert slip_pnl("OPEN", 6, None) is None


# ── Balances and running P/L ───────────────────────────────────────

def _slips(rows):
    cols = ["slip_id", "bettor", "platform", "stake", "status", "payout",
            "is_bonus_bet", "is_paper", "settled_at"]
    return pd.DataFrame(rows, columns=cols)


class TestBalances:
    TXNS = pd.DataFrame([
        ("bettor 1", "book A", "DEPOSIT", 100.0),
        ("bettor 1", "book A", "WITHDRAWAL", 30.0),
        ("bettor 1", "book A", "BONUS", 5.0),
        ("bettor 1", "book A", "ADJUSTMENT", -1.5),
        ("bettor 2", "exchange B", "DEPOSIT", 50.0),
    ], columns=["bettor", "platform", "kind", "amount"])
    T0 = dt.datetime(2026, 10, 10, 3, 0)
    SLIPS = _slips([
        (1, "bettor 1", "book A", 15.0, "WON", 25.0, False, False, T0),
        (2, "bettor 1", "book A", 4.0, "LOST", 0.0, False, False, T0 + dt.timedelta(hours=1)),
        (3, "bettor 1", "book A", 11.0, "PUSH", 11.0, False, False, T0),
        (4, "bettor 1", "book A", 6.0, "OPEN", None, False, False, None),
        (5, "bettor 1", "book A", 10.0, "WON", 20.0, True, False, T0),     # bonus bet
        (6, "bettor 1", "book A", 10.0, "OPEN", None, True, False, None),  # bonus, open
        (7, "bettor 1", "book A", 50.0, "WON", 100.0, False, True, T0),    # paper
        (8, "bettor 2", "exchange B", 20.0, "LOST", 0.0, False, False, T0),
    ])

    def test_balance_per_bettor_and_platform(self):
        b = balance_table(self.TXNS, self.SLIPS).set_index(["bettor", "platform"])
        one = b.loc[("bettor 1", "book A")]
        # settled P/L: +10 - 4 + 0 + 20 (bonus win) = 26; paper left out
        assert one["settled_pl"] == 26.0
        # only the cash bet's stake is riding; the open bonus bet is credit
        assert one["open_stakes"] == 6.0
        # 100 - 30 + 5 - 1.5 + 26 - 6 = 93.5
        assert one["balance"] == 93.5
        assert (one["bets"], one["won"], one["lost"]) == (6, 2, 1)
        # ROI over cash bets that won or lost: (10 - 4) / (15 + 4)
        assert one["roi"] == round(6 / 19, 4)
        two = b.loc[("bettor 2", "exchange B")]
        assert (two["balance"], two["settled_pl"], two["roi"]) == (30.0, -20.0, -1.0)

    def test_platform_with_only_bets_or_only_money(self):
        b = balance_table(self.TXNS.iloc[:0], self.SLIPS.iloc[[0]])
        assert b.loc[0, "balance"] == 10.0
        b = balance_table(self.TXNS.iloc[[4]], self.SLIPS.iloc[:0])
        assert list(b["balance"]) == [50.0] and b.loc[0, "roi"] is None
        assert balance_table(self.TXNS.iloc[:0], self.SLIPS.iloc[:0]).empty
        assert balance_table(None, None).empty

    def test_running_pl_in_settlement_order(self):
        r = running_pl(self.SLIPS)
        # paper and open slips left out; ties on settled_at go by slip_id
        assert list(r["slip_id"]) == [1, 3, 5, 8, 2]
        assert list(r["pnl"]) == [10.0, 0.0, 20.0, -20.0, -4.0]
        assert list(r["cum_pl"]) == [10.0, 10.0, 30.0, -20.0, 26.0]
        assert list(r["cum_pl_all"]) == [10.0, 10.0, 30.0, 10.0, 6.0]
        assert running_pl(self.SLIPS.iloc[:0]).empty


# ── Validation and descriptions ────────────────────────────────────

class TestValidate:
    def test_parlay_price_is_the_legs_multiplied(self):
        legs = [LegInput("ml", "HOME", 1, price_american=-110),
                LegInput("total", "OVER", 2, line=6.5, price_american=-110)]
        assert validate_slip("bettor 1", "book A", 10, legs) == (264, [])
        assert validate_slip("bettor 1", "book A", 10, legs, 300) == (300, [])

    def test_single_takes_its_leg_price(self):
        legs = [LegInput("ml", "AWAY", 1, price_american=125)]
        assert validate_slip("bettor 2", "book A", 10, legs) == (125, [])
        price, problems = validate_slip("bettor 2", "book A", 10, legs, 130)
        assert price is None and "differ" in problems[0]

    def test_problems_in_plain_english(self):
        legs = [LegInput("total", "OVER", 1, line=None, price_american=-110),
                LegInput("prop_sog", "OVER", 1, line=2.5, price_american=50),
                LegInput("ml", "DRAW", None),
                LegInput("other", "  ")]
        price, problems = validate_slip("someone", " ", 0, legs)
        text_ = " ".join(problems)
        assert price is None
        for bit in ("Unknown bettor", "platform", "stake", "Leg 1: this bet needs a line",
                    "Leg 2: pick the player", "Leg 2: American odds",
                    "Leg 3: pick the game", "Leg 3: side must be",
                    "Leg 4: describe the bet"):
            assert bit in text_, bit

    def test_parlay_without_odds_needs_the_combined_price(self):
        legs = [LegInput("ml", "HOME", 1), LegInput("ml", "AWAY", 2)]
        price, problems = validate_slip("bettor 1", "book A", 10, legs)
        assert price is None and "combined odds" in problems[0]
        assert validate_slip("bettor 1", "book A", 10, legs, 250) == (250, [])

    def test_record_slip_refuses_before_touching_the_database(self, monkeypatch):
        monkeypatch.setattr(ledger, "ensure_schema",
                            lambda: pytest.fail("validation runs first"))
        with pytest.raises(ValueError, match="Unknown bettor"):
            ledger.record_slip("a real name", "book A", 10,
                               [LegInput("ml", "HOME", 1, price_american=-110)])

    def test_describe_leg(self):
        assert describe_leg("ml", "HOME", None, "BOS", "TOR") == "BOS @ TOR: TOR win"
        assert describe_leg("pl", "AWAY", 1.5, "BOS", "TOR") == "BOS @ TOR: BOS +1.5 (puck line)"
        assert describe_leg("total", "UNDER", 6.0, "BOS", "TOR") == "BOS @ TOR: Under 6 goals"
        assert (describe_leg("prop_sog", "OVER", 2.5, "BOS", "TOR", "D. Pastrnak")
                == "BOS @ TOR: D. Pastrnak over 2.5 shots on goal")
        assert describe_leg("other", "TOR to win the Cup", None) == "TOR to win the Cup"


# ── Database: migration, recording, settlement, balances ───────────

G_BASE = 9_990_000_100          # synthetic game ids, far from real ones
PLAYER_A, PLAYER_B, PLAYER_C = 99_999_901, 99_999_902, 99_999_903
PLATFORM = "zz-ledger-test"


def _cleanup(conn):
    conn.execute(text("DELETE FROM betting.slips WHERE platform = :p"), {"p": PLATFORM})
    conn.execute(text("DELETE FROM betting.bankroll_txns WHERE platform = :p"),
                 {"p": PLATFORM})
    conn.execute(text("DELETE FROM raw.skater_games WHERE game_id BETWEEN :a AND :b"),
                 {"a": G_BASE, "b": G_BASE + 9})
    conn.execute(text("DELETE FROM raw.games WHERE game_id BETWEEN :a AND :b"),
                 {"a": G_BASE, "b": G_BASE + 9})
    conn.execute(text("DELETE FROM raw.players WHERE player_id = ANY(:ids)"),
                 {"ids": [PLAYER_A, PLAYER_B, PLAYER_C]})


@requires_db
class TestMigration:
    def test_ledger_tables_match_schema_sql_and_rerun_runs_no_ddl(self, monkeypatch):
        from config import migrate
        monkeypatch.setattr(migrate, "_done", False)
        migrate.ensure_schema()          # creates the tables on an old database
        with engine.connect() as conn:
            cols = {(r.table_name, r.column_name) for r in conn.execute(text("""
                SELECT table_name, column_name FROM information_schema.columns
                WHERE table_schema = 'betting'
                  AND table_name IN ('slips', 'slip_legs', 'bankroll_txns')"""))}
        assert {("slips", c) for c in ("slip_id", "bettor", "platform", "placed_at",
                                       "stake", "price_american", "is_parlay",
                                       "is_bonus_bet", "status", "payout",
                                       "settled_at", "notes", "is_paper",
                                       "created_at")} <= cols
        assert {("slip_legs", c) for c in ("slip_id", "leg_no", "game_id", "market",
                                           "side", "line", "price_american",
                                           "player_id", "rec_id", "result")} <= cols
        assert {("bankroll_txns", c) for c in ("bettor", "platform", "ts", "kind",
                                               "amount", "note")} <= cols

        ddl = []

        def spy(conn, cursor, statement, *args):
            if statement.lstrip().upper().startswith(("CREATE", "ALTER")):
                ddl.append(statement)
        event.listen(engine, "before_cursor_execute", spy)
        try:
            monkeypatch.setattr(migrate, "_done", False)
            migrate.ensure_schema()      # second run: nothing to do
        finally:
            event.remove(engine, "before_cursor_execute", spy)
        assert ddl == []

    def test_constraints_reject_bad_rows(self):
        from sqlalchemy.exc import IntegrityError
        ledger.ensure_schema()
        for sql in (
            "INSERT INTO betting.slips (bettor, platform, placed_at, stake, price_american) "
            "VALUES ('bettor 1', 'zz-ledger-test', now(), 0, -110)",
            "INSERT INTO betting.slips (bettor, platform, placed_at, stake, price_american, status) "
            "VALUES ('bettor 1', 'zz-ledger-test', now(), 5, -110, 'MAYBE')",
            "INSERT INTO betting.bankroll_txns (bettor, platform, ts, kind, amount) "
            "VALUES ('bettor 1', 'zz-ledger-test', now(), 'GIFT', 5)",
        ):
            with pytest.raises(IntegrityError):
                with engine.begin() as conn:
                    conn.execute(text(sql))


@requires_db
class TestLedgerOnTheDatabase:
    @pytest.fixture()
    def games(self):
        """G1 final 3-2 (shootout), G2 final 4-2, G3 postponed, G4 not
        played yet. G1's box score: player A 3 shots, player C 0 shots;
        player B has no row (scratched)."""
        ledger.ensure_schema()
        g1, g2, g3, g4 = (G_BASE + i for i in range(1, 5))
        with engine.begin() as conn:
            _cleanup(conn)
            for gid, hs, as_, state, sched, so in (
                    (g1, 3, 2, "OFF", "OK", True), (g2, 4, 2, "FINAL", "OK", False),
                    (g3, None, None, "FUT", "PPD", False),
                    (g4, None, None, "FUT", "OK", False)):
                conn.execute(text("""
                    INSERT INTO raw.games (game_id, season, game_type, date,
                        home_team, away_team, home_score, away_score,
                        game_state, schedule_state, is_so)
                    VALUES (:g, 20262027, 2, DATE '2026-10-10', 'TOR', 'BOS',
                            :hs, :as_, :st, :sc, :so)
                """), {"g": gid, "hs": hs, "as_": as_, "st": state, "sc": sched, "so": so})
            for pid, name in ((PLAYER_A, "Z. Testshooter"), (PLAYER_B, "Z. Scratched"),
                              (PLAYER_C, "Z. Quiet")):
                conn.execute(text("""
                    INSERT INTO raw.players (player_id, full_name, position)
                    VALUES (:p, :n, 'C')"""), {"p": pid, "n": name})
            for pid, shots in ((PLAYER_A, 3), (PLAYER_C, 0)):
                conn.execute(text("""
                    INSERT INTO raw.skater_games (player_id, game_id, team, shots)
                    VALUES (:p, :g, 'TOR', :s)"""), {"p": pid, "g": g1, "s": shots})
        yield g1, g2, g3, g4
        with engine.begin() as conn:
            _cleanup(conn)

    def test_record_settle_and_balance(self, games):
        g1, g2, g3, g4 = games
        rec = ledger.record_slip
        L = LegInput
        b1 = "bettor 1"
        ids = {
            # single moneyline, home won the shootout: $15 at -150 -> 25
            "ml": rec(b1, PLATFORM, 15, [L("ml", "HOME", g1, price_american=-150)]),
            # total 6 on a 4-2 game: push, stake back
            "push": rec(b1, PLATFORM, 11, [L("total", "UNDER", g2, 6.0)], -110),
            # player A over 2.5 shots with 3: $10 at +120 -> 22
            "prop": rec(b1, PLATFORM, 10, [L("prop_sog", "OVER", g1, 2.5, 120,
                                             player_id=PLAYER_A)]),
            # scratched player: void
            "sat": rec(b1, PLATFORM, 5, [L("prop_sog", "OVER", g1, 1.5, -115,
                                           player_id=PLAYER_B)]),
            # parlay WIN -150 + PUSH -110 + VOID +130 (postponed): 10 x 1.6667
            "parlay": rec(b1, PLATFORM, 10, [L("ml", "HOME", g1, price_american=-150),
                                             L("total", "OVER", g2, 6.0, -110),
                                             L("ml", "AWAY", g3, price_american=130)]),
            # parlay with a loser (6 goals > 5.5) and an unplayed leg: LOST now
            "lost": rec(b1, PLATFORM, 4, [L("total", "UNDER", g2, 5.5, -105),
                                          L("ml", "HOME", g4, price_american=-120)]),
            # parlay still waiting on G4
            "open": rec(b1, PLATFORM, 6, [L("ml", "HOME", g1, price_american=-150),
                                          L("ml", "AWAY", g4, price_american=110)]),
            # practice bet: settles, but stays out of the balance
            "paper": rec(b1, PLATFORM, 50, [L("ml", "HOME", g1, price_american=100)],
                         is_paper=True),
            # $10 bonus bet at +200 that wins: pays the $20 profit only
            "bonus": rec(b1, PLATFORM, 10, [L("ml", "HOME", g1, price_american=200)],
                         is_bonus_bet=True),
            # a futures bet the system can't grade
            "other": rec(b1, PLATFORM, 3, [L("other", "TOR to win the Cup")], 1000),
        }
        ledger.record_txn(b1, PLATFORM, "DEPOSIT", 100, "first deposit")
        ledger.record_txn(b1, PLATFORM, "WITHDRAWAL", 30)
        ledger.record_txn(b1, PLATFORM, "BONUS", 5, "sign-up bonus")
        with pytest.raises(ValueError):
            ledger.record_txn(b1, PLATFORM, "WITHDRAWAL", -30)

        counts = ledger.settle_slips()
        # legs decided: ml, push, prop, sat, parlay x3, lost's first leg,
        # open's first leg, paper, bonus = 11
        assert counts["legs"] == 11
        assert (counts["WON"], counts["LOST"], counts["PUSH"], counts["VOID"]) == (5, 1, 1, 1)

        slips = ledger.load_slips().set_index("slip_id")
        got = {k: (slips.loc[v, "status"], slips.loc[v, "payout"]) for k, v in ids.items()}
        assert got["ml"] == ("WON", 25.0)
        assert got["push"] == ("PUSH", 11.0)
        assert got["prop"] == ("WON", 22.0)
        assert got["sat"] == ("VOID", 5.0)
        assert got["parlay"] == ("WON", 16.67)
        assert got["lost"] == ("LOST", 0.0)
        assert got["open"][0] == "OPEN" and pd.isna(got["open"][1])
        assert got["paper"] == ("WON", 100.0)
        assert got["bonus"] == ("WON", 20.0)
        assert got["other"][0] == "OPEN"
        assert slips.loc[ids["parlay"], "price_american"] == 632
        assert bool(slips.loc[ids["parlay"], "is_parlay"])

        legs = ledger.load_legs([ids["parlay"], ids["lost"]])
        assert list(legs["result"].fillna("-")) == ["WIN", "PUSH", "VOID", "LOSS", "-"]
        assert legs["bet"].iloc[0] == "BOS @ TOR: TOR win"

        # running again decides nothing new
        again = ledger.settle_slips()
        assert again["legs"] == 0 and sum(v for k, v in again.items() if k != "legs") == 0

        def balance():
            b = ledger.balances()
            return b[b["platform"] == PLATFORM].iloc[0]
        b = balance()
        # settled P/L: +10 + 0 + 12 + 0 + 6.67 - 4 + 20 (bonus) = 44.67
        assert b["settled_pl"] == 44.67
        # riding: the open parlay 6 + the futures bet 3
        assert b["open_stakes"] == 9.0
        # 100 - 30 + 5 + 44.67 - 9
        assert b["balance"] == 110.67

        # the futures bet won (set by hand); the open parlay is cashed out
        ledger.set_leg_result(ids["other"], 1, "WIN")
        ledger.settle_by_hand(ids["open"], "CASHED_OUT", 8.5)
        b = balance()
        # + 30 (3 at +1000 pays 33) + 2.5 (cash-out 8.5 on 6)
        assert b["settled_pl"] == 77.17
        assert b["open_stakes"] == 0.0
        assert b["balance"] == 152.17

        # a correction re-works the slip out from its legs
        ledger.set_leg_result(ids["ml"], 1, "LOSS")
        assert ledger.load_slips().set_index("slip_id").loc[ids["ml"], "status"] == "LOST"
        # a cash-out stays a cash-out
        ledger.set_leg_result(ids["open"], 2, "LOSS")
        assert ledger.load_slips().set_index("slip_id").loc[ids["open"], "status"] == "CASHED_OUT"

        pl = ledger.running_pl(ledger.load_slips())
        assert ids["paper"] not in set(pl["slip_id"])

        assert ledger.delete_slip(ids["paper"])
        assert ids["paper"] not in set(ledger.load_slips()["slip_id"])
        assert PLATFORM in ledger.known_platforms()

    def test_settle_by_hand_rules(self, games):
        g1 = games[0]
        sid = ledger.record_slip("bettor 2", PLATFORM, 10,
                                 [LegInput("ml", "HOME", g1, price_american=-110)])
        with pytest.raises(ValueError, match="needs the payout"):
            ledger.settle_by_hand(sid, "WON")
        ledger.settle_by_hand(sid, "VOID")
        s = ledger.load_slips().set_index("slip_id").loc[sid]
        assert (s["status"], s["payout"]) == ("VOID", 10.0)
        ledger.settle_by_hand(sid, "OPEN")
        s = ledger.load_slips().set_index("slip_id").loc[sid]
        assert s["status"] == "OPEN" and pd.isna(s["payout"])
        with pytest.raises(ValueError):
            ledger.settle_by_hand(sid, "MAYBE")
