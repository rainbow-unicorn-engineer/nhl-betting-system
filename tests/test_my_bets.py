"""
Tests for dashboard/my_bets.py: the legs table -> ledger legs, labels and
time conversion (pure), then the whole dashboard rendered by Streamlit's
AppTest on a disposable database: every tab renders, a parlay saved from
the My bets form lands in the ledger, and each parlay shows as its own
group. The database test follows tests/conftest.py (skips without a
disposable database) and removes what it writes.
"""
import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from dashboard import my_bets
from dashboard.my_bets import (LEG_COLUMNS, checker_rows, editor_legs, empty_legs,
                               game_label, player_labels)

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")

GAMES = {"Sat Oct 10  BOS @ TOR": 11, "Sat Oct 10  MTL @ OTT": 12}
PLAYERS = {"Z. Shooter (TOR)": 901}


def _rows(*rows):
    return pd.DataFrame([dict(zip(LEG_COLUMNS, r)) for r in rows], columns=LEG_COLUMNS)


class TestEditorLegs:
    def test_every_bet_type(self):
        rows = _rows(
            ("Sat Oct 10  BOS @ TOR", "Home win", None, -150, None, None, 7),
            ("Sat Oct 10  BOS @ TOR", "Away puck line", 1.5, -200, None, None, None),
            ("Sat Oct 10  MTL @ OTT", "Under", 6.5, -110, None, None, None),
            ("Sat Oct 10  BOS @ TOR", "Player shots over", 2.5, 120, "Z. Shooter (TOR)",
             None, None),
            (None, "Other (describe it)", None, 1000, None, "TOR to win the Cup", None))
        legs, problems = editor_legs(rows, GAMES, PLAYERS)
        assert problems == []
        got = [(l.market, l.side, l.game_id, l.line, l.price_american, l.player_id, l.rec_id)
               for l in legs]
        assert got == [("ml", "HOME", 11, None, -150, None, 7),
                       ("pl", "AWAY", 11, 1.5, -200, None, None),
                       ("total", "UNDER", 12, 6.5, -110, None, None),
                       ("prop_sog", "OVER", 11, 2.5, 120, 901, None),
                       ("other", "TOR to win the Cup", None, None, 1000, None, None)]

    def test_blank_rows_are_skipped_and_odds_may_be_missing(self):
        rows = pd.concat([empty_legs(2),
                          _rows(("Sat Oct 10  BOS @ TOR", "Over", 6.0, None, None, None,
                                 None))], ignore_index=True)
        legs, problems = editor_legs(rows, GAMES, PLAYERS)
        assert problems == [] and len(legs) == 1 and legs[0].price_american is None

    def test_problems(self):
        rows = _rows(
            (None, "Home win", None, -110, None, None, None),
            ("Sat Oct 10  BOS @ TOR", "Player shots over", 2.5, 120, None, None, None),
            (None, "Other (describe it)", None, 500, None, "  ", None),
            ("Mon Oct 12  NYR @ NJD", "Home win", None, -110, None, None, None))
        legs, problems = editor_legs(rows, GAMES, PLAYERS)
        assert legs == []
        assert problems == [
            "Row 1: pick the game.", "Row 2: pick the player.",
            "Row 3: describe the bet in the Description column.",
            "Row 4: that game is no longer in the list; pick it again."]
        assert editor_legs(empty_legs(), GAMES, PLAYERS)[1] == [
            "Add at least one leg: pick a game, a bet and the odds."]


def test_checker_rows_copy_over_unchanged():
    slip = pd.DataFrame({"Game": ["Sat Oct 10  BOS @ TOR", "Sat Oct 10  MTL @ OTT"],
                         "Bet": ["Away win", "Over"], "Line": [None, 6.5],
                         "Odds": [130, -105]})
    rows = checker_rows(slip)
    assert list(rows.columns) == LEG_COLUMNS
    legs, problems = editor_legs(rows, GAMES, PLAYERS)
    assert problems == []
    assert [(l.market, l.side, l.line, l.price_american) for l in legs] == [
        ("ml", "AWAY", None, 130), ("total", "OVER", 6.5, -105)]
    assert checker_rows(None).empty


def test_labels():
    assert game_label(dt.date(2026, 10, 10), "BOS", "TOR") == "Sat Oct 10  BOS @ TOR"
    players = pd.DataFrame({"player_id": [1, 2, 3], "full_name": ["A. One", "B. Two", "B. Two"],
                            "team": ["TOR", "BOS", "BOS"]})
    assert player_labels(players) == {"A. One (TOR)": 1, "B. Two (BOS) #2": 2,
                                      "B. Two (BOS) #3": 3}


def test_local_time_to_utc(monkeypatch):
    from zoneinfo import ZoneInfo
    monkeypatch.setattr(my_bets, "LOCAL_TZ", ZoneInfo("America/Chicago"))
    # 7:05 pm Central (CDT, UTC-5) on Oct 10 = 00:05 UTC Oct 11
    assert my_bets.local_to_utc(dt.date(2026, 10, 10), dt.time(19, 5)) == \
        dt.datetime(2026, 10, 11, 0, 5)


def test_money():
    assert my_bets.money(-4) == "-$4.00" and my_bets.money(1234.5) == "$1,234.50"
    assert my_bets.signed_money(2.5) == "+$2.50" and my_bets.signed_money(0) == "$0.00"
    assert my_bets.money(None) == "—"


# ── The dashboard on a disposable database ─────────────────────────

G_BASE = 9_990_000_200
PLATFORM = "zz-dashboard-test"


def _cleanup(conn):
    conn.execute(text("DELETE FROM betting.slips WHERE platform = :p"), {"p": PLATFORM})
    conn.execute(text("DELETE FROM betting.bankroll_txns WHERE platform = :p"),
                 {"p": PLATFORM})
    conn.execute(text("DELETE FROM raw.games WHERE game_id BETWEEN :a AND :b"),
                 {"a": G_BASE, "b": G_BASE + 9})


@requires_db
def test_dashboard_records_a_parlay_and_groups_it(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from betting import ledger
    from config.settings import local_today
    monkeypatch.setenv("BETTORS", "bettor 1,bettor 2")
    ledger.ensure_schema()
    today = local_today()
    with engine.begin() as conn:
        _cleanup(conn)
        for gid, away, home in ((G_BASE + 1, "BOS", "TOR"), (G_BASE + 2, "MTL", "OTT")):
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                       away_team, game_state)
                VALUES (:g, 20262027, 2, :d, :h, :a, 'FUT')"""),
                         {"g": gid, "d": today, "h": home, "a": away})
    try:
        ledger.record_txn("bettor 1", PLATFORM, "DEPOSIT", 50)
        at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
        at.run()
        assert not at.exception, at.exception
        labels = [t.label for t in at.tabs]
        assert labels == ["📅 Today", "🔍 Check a bet", "📒 My bets", "🧠 Model",
                          "🧪 Backtest", "💰 Bankroll"]

        g1 = game_label(today, "BOS", "TOR")
        g2 = game_label(today, "MTL", "OTT")
        at.session_state["ledger_rows"] = _rows(
            (g1, "Home win", None, -150, None, None, None),
            (g2, "Over", 6.0, -110, None, None, None))
        at.session_state["ledger_ver"] = 100
        at.run()
        at.selectbox(key="slip_platform_choice").set_value(my_bets.NEW_PLATFORM).run()
        at.text_input(key="slip_platform_new").set_value(PLATFORM)
        stake = [n for n in at.number_input if n.label == "Stake in $"][0]
        stake.set_value(10.0)
        [b for b in at.button if b.label == "Save bet"][0].click().run()
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]

        slips = ledger.load_slips()
        mine = slips[slips["platform"] == PLATFORM]
        assert len(mine) == 1
        s = mine.iloc[0]
        # -150 x -110: 1.6667 x 1.9091 = 3.1818 -> +218
        assert (s["bettor"], s["stake"], s["price_american"], bool(s["is_parlay"])) == \
            ("bettor 1", 10.0, 218, True)
        legs = ledger.load_legs([int(s["slip_id"])])
        assert list(legs["bet"]) == ["BOS @ TOR: TOR win", "MTL @ OTT: Over 6 goals"]

        # the parlay is its own group, its legs in one table
        groups = [e for e in at.expander if e.label.startswith(f"#{int(s['slip_id'])} ")]
        assert len(groups) == 1 and "2-leg parlay" in groups[0].label
        assert any(PLATFORM == r for r in ledger.balances()["platform"])
    finally:
        with engine.begin() as conn:
            _cleanup(conn)
