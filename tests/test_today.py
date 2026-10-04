"""
Tests for dashboard/today.py: labels, each bettor's books from the
environment, best prices anywhere and per bettor, the per-game book
table, the picks table (all pure), then the tab rendered by Streamlit's
AppTest from canned query results, so no database is needed.
"""
import pandas as pd
import pytest

from dashboard.today import (BEST_MARK, EVERY_BOOK, american, bet_label,
                             best_price_table, bettor_books, edge_points,
                             fair_chances, game_book_table, limits_in_use,
                             picks_table, side_prices)

T0 = pd.Timestamp("2026-10-10 18:00")          # naive UTC, as captured_at

GAMES = pd.DataFrame({
    "game_id": [1, 2],
    "date": [pd.Timestamp("2026-10-10")] * 2,
    "start_time_utc": [pd.Timestamp("2026-10-10 23:00", tz="UTC")] * 2,
    "away_team": ["TOR", "MTL"],
    "home_team": ["BOS", "OTT"],
})

SNAPS = pd.DataFrame({
    "game_id": [1, 1, 1, 1],
    "book_name": ["DraftKings", "fanduel", "kalshi", "polymarket"],
    "captured_at": [T0] * 4,
    "home_price": [-150, -140, -145, -160],
    "away_price": [130, 118, 125, 135],
})

WHO = {"bettor 1": (frozenset({"kalshi", "polymarket"}), "BETTOR_1_BOOKS"),
       "bettor 2": (EVERY_BOOK, "every book")}


class TestLabels:
    def test_bet_label_spells_the_team(self):
        assert bet_label("HOME", "TOR", "BOS") == "BOS win"
        assert bet_label("AWAY", "TOR", "BOS") == "TOR win"
        assert bet_label("OVER", "TOR", "BOS") == "Over"

    def test_numbers(self):
        assert american(120) == "+120" and american(-150) == "-150"
        assert american(None) == "—"
        assert edge_points(0.069) == "+6.9 pts"
        assert edge_points(0.025) == "+2.5 pts"


class TestBettorBooks:
    def test_own_books_then_shared_then_every_book(self):
        env = {"BETTOR_1_BOOKS": " Kalshi, polymarket ,,",
               "BETTABLE_BOOKS": "draftkings"}
        got = bettor_books(env, ["bettor 1", "bettor 2"])
        assert got == {
            "bettor 1": (frozenset({"kalshi", "polymarket"}), "BETTOR_1_BOOKS"),
            "bettor 2": (frozenset({"draftkings"}), "BETTABLE_BOOKS")}

    def test_nothing_set_means_every_book(self):
        got = bettor_books({}, ["bettor 1"])
        assert got == {"bettor 1": (EVERY_BOOK, "every book")}

    def test_a_setting_past_the_list_adds_a_bettor(self):
        got = bettor_books({"BETTOR_3_BOOKS": "fanduel"}, ["bettor 1"])
        assert list(got) == ["bettor 1", "bettor 3"]
        assert got["bettor 3"] == (frozenset({"fanduel"}), "BETTOR_3_BOOKS")

    def test_labels_default_to_bettors_setting(self, monkeypatch):
        monkeypatch.setenv("BETTORS", "bettor A,bettor B")
        assert list(bettor_books({})) == ["bettor A", "bettor B"]


class TestBestPrices:
    def test_side_prices_long_form(self):
        p = side_prices(SNAPS)
        assert len(p) == 8 and set(p["side"]) == {"HOME", "AWAY"}
        assert set(p["book"]) == {"draftkings", "fanduel", "kalshi", "polymarket"}

    def test_fair_chance_is_the_median_no_vig(self):
        f = fair_chances(SNAPS)
        assert set(f) == {1}
        assert 0.55 < f[1] < 0.62

    def test_best_anywhere_and_per_bettor(self):
        t = best_price_table(GAMES, SNAPS, WHO)
        # game 2 has no prices: left out
        assert list(t["Game"]) == ["TOR @ BOS", "TOR @ BOS"]
        away, home = t.iloc[0], t.iloc[1]
        assert away["Bet"] == "TOR win" and home["Bet"] == "BOS win"
        # +135 (polymarket) is the best TOR price anywhere and at bettor 1's books
        assert away["Best price anywhere"] == "+135 at polymarket"
        assert away["Best for bettor 1"] == "+135 at polymarket"
        # -140 (fanduel) is best for BOS; bettor 1 only has kalshi/polymarket
        assert home["Best price anywhere"] == "-140 at fanduel"
        assert home["Best for bettor 1"] == "-145 at kalshi"
        assert home["Best for bettor 2"] == "-140 at fanduel"
        assert away["Books quoting"] == 4

    def test_ties_list_every_book(self):
        snaps = SNAPS.assign(away_price=[135, 135, 125, 120])
        t = best_price_table(GAMES, snaps, WHO)
        assert t.iloc[0]["Best price anywhere"] == "+135 at draftkings, fanduel"

    def test_bettor_with_no_quoting_book(self):
        who = {"bettor 1": (frozenset({"betmgm"}), "BETTOR_1_BOOKS")}
        t = best_price_table(GAMES, SNAPS, who)
        assert set(t["Best for bettor 1"]) == {"not offered"}

    def test_game_book_table_marks_and_order(self):
        g = next(GAMES.itertuples())
        t = game_book_table(g, SNAPS, WHO)
        assert len(t) == 8
        away = t[t["Bet"] == "TOR win"]
        assert list(away["Price"]) == ["+135", "+130", "+125", "+118"]
        top = away.iloc[0]
        assert top["Book"].startswith("polymarket (exchange")
        assert BEST_MARK in top["Best"] and "best for bettor 1" in top["Best"]
        assert "best for bettor 2" in top["Best"]
        home = t[t["Bet"] == "BOS win"]
        kalshi = home[home["Book"].str.startswith("kalshi")].iloc[0]
        assert kalshi["Best"] == "best for bettor 1"
        assert home[home["Book"] == "draftkings"].iloc[0]["Best"] == ""

    def test_empty_snapshots(self):
        empty = SNAPS.iloc[0:0]
        assert best_price_table(GAMES, empty, WHO).empty
        assert game_book_table(next(GAMES.itertuples()), empty, WHO).empty


class TestPicksTable:
    def test_rows(self):
        recs = pd.DataFrame({
            "start_time_utc": [pd.Timestamp("2026-10-10 23:00", tz="UTC")],
            "away_team": ["NYR"], "home_team": ["BOS"], "side": ["AWAY"],
            "best_book": ["fanduel"], "best_price": [120], "model_prob": [0.563],
            "implied_prob_novig": [0.495], "edge_pct": [0.069],
            "recommended_stake": [20.0]})
        t = picks_table(recs)
        r = t.iloc[0]
        assert (r["Game"], r["Bet"], r["Book"], r["Price"], r["Edge"], r["Stake"]) == \
            ("NYR @ BOS", "NYR win", "fanduel", "+120", "+6.9 pts", "$20.00")
        assert r["Model's chance"] == "56.3%"
        assert r["Puck drop (your time)"] != "TBD"


class TestLimits:
    def test_defaults_shown_in_dollars(self):
        got = {label: value for label, value, _ in limits_in_use(1000)}
        assert got == {"Per bet": "2% ($20)", "Per day": "10% ($100)",
                       "Per game": "4% ($40)", "Bets per game": "3"}


def _tab_script():
    """Rendered by AppTest: the Today tab on canned query results."""
    import pandas as pd
    import streamlit as st

    from dashboard import today

    t0 = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=1)

    def read(sql, params=None):
        if "betting.recommendations" in sql:
            return pd.DataFrame({
                "start_time_utc": [pd.Timestamp("2026-10-10 23:00", tz="UTC")],
                "away_team": ["NYR"], "home_team": ["BOS"], "side": ["AWAY"],
                "best_book": ["fanduel"], "best_price": [120],
                "model_prob": [0.563], "implied_prob_novig": [0.495],
                "edge_pct": [0.069], "recommended_stake": [20.0]})
        if "raw.odds_snapshots" in sql:
            return pd.DataFrame({
                "game_id": [1, 1], "book_name": ["fanduel", "kalshi"],
                "captured_at": [t0, t0], "home_price": [-140, -150],
                "away_price": [120, 130]})
        return pd.DataFrame({
            "game_id": [1, 2], "date": [pd.Timestamp("2026-10-10")] * 2,
            "start_time_utc": [pd.Timestamp("2026-10-10 23:00", tz="UTC")] * 2,
            "away_team": ["NYR", "MTL"], "home_team": ["BOS", "OTT"]})

    today.render(st, read)


def test_tab_renders_without_a_database(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BETTORS", "bettor 1,bettor 2")
    monkeypatch.setenv("BETTOR_1_BOOKS", "kalshi")
    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [s.value for s in at.subheader] == ["Pending picks", "Prices by book"]
    picks = at.dataframe[0].value
    assert list(picks["Bet"]) == ["NYR win"] and list(picks["Edge"]) == ["+6.9 pts"]
    best = at.dataframe[1].value
    assert list(best["Best for bettor 1"]) == ["+130 at kalshi", "-150 at kalshi"]
    assert list(best["Best price anywhere"]) == ["+130 at kalshi", "-140 at fanduel"]
    assert [m.label for m in at.metric] == ["Per bet", "Per day", "Per game",
                                            "Bets per game"]
    assert len(at.expander) == 1           # one priced game
    assert any("MTL @ OTT" in str(d.value.get("Game", "")) for d in at.dataframe)
