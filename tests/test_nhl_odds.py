"""
Tests for ingestion/nhl_odds.py, the free NHL odds feed stored beside The
Odds API. Parsing runs on trimmed copies of real responses captured on
2026-09-28 (tests/fixtures/nhl_partner_us.json, nhl_partner_ca.json,
nhl_schedule.json). No network: requests.get is mocked. The database
tests run only against a disposable copy (see tests/conftest.py); they use
synthetic games 9999030001 and 9999030002 on 2031-02-15 and delete every
row they add.
"""
import datetime as dt
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import nhl_odds
from ingestion.nhl_odds import (cents_diff, decimal_to_american, excluded_books,
                                format_report, freshness, pair_prices,
                                parse_partner, parse_schedule, prob_diff,
                                select_rows, summarize_freshness,
                                summarize_pairs, swapped_books, to_american)

UTC = dt.timezone.utc
FIXTURES = Path(__file__).parent / "fixtures"
CAR, TOR, BOS, EDM, VGK = 2026020001, 2026020002, 2026020003, 2026020004, 2026020005


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def by_market(rows):
    return {(r["game_id"], r["market"]): r for r in rows}


# ── Prices ─────────────────────────────────────────────────────────

class TestPrices:
    def test_decimal_to_american(self):
        assert decimal_to_american(2.12) == 112
        assert decimal_to_american(1.72) == -139      # 100 / 0.72 = 138.9
        assert decimal_to_american(2.0) == 100
        assert decimal_to_american(1.5) == -200
        assert decimal_to_american(3.25) == 225
        assert decimal_to_american(1.29) == -345
        assert decimal_to_american(1.99) == -101
        for bad in (1.0, 0.5, -2, "x", None, float("nan")):
            assert decimal_to_american(bad) is None, bad

    def test_to_american_reads_both_formats(self):
        assert to_american("-125") == -125
        assert to_american("+104") == 104
        assert to_american(-125.0) == -125          # partner-game numbers
        assert to_american(105.0) == 105
        assert to_american("100") == 100            # even money
        assert to_american("2.12") == 112           # decimal (schedule)
        assert to_american("1.72") == -139
        for bad in (None, "", "abc", True, "nan", "+1.5", "-50", "1.0", "0"):
            assert to_american(bad) is None, bad

    def test_book_key(self):
        assert nhl_odds.book_key("DraftKings") == "draftkings"
        assert nhl_odds.book_key("Fan Duel") == "fanduel"
        assert nhl_odds.book_key(None) == ""


# ── partner-game ───────────────────────────────────────────────────

class TestParsePartner:
    def test_draftkings_four_markets(self):
        rows = parse_partner(load("nhl_partner_us.json"), "partner-US")
        got = by_market(rows)
        assert set(got) == {(g, m) for g in (CAR, TOR)
                            for m in ("ml", "pl", "total", "ml3")}
        assert {r["book"] for r in rows} == {"draftkings"}
        assert {r["source"] for r in rows} == {"partner-US"}
        ml = got[(CAR, "ml")]
        assert (ml["home_price"], ml["away_price"]) == (-125, 105)
        assert ml["feed_updated_utc"] == dt.datetime(2026, 8, 28, 18, 0, 38)   # naive UTC
        assert ml["start"] == dt.datetime(2026, 9, 29, 21, 0, tzinfo=UTC)
        pl = got[(CAR, "pl")]
        assert (pl["home_price"], pl["away_price"], pl["line"]) == (190, -230, -1.5)
        total = got[(CAR, "total")]
        assert (total["over_price"], total["under_price"], total["line"]) == (105, -125, 6.5)
        assert total["home_price"] is None
        ml3 = got[(CAR, "ml3")]
        assert (ml3["home_price"], ml3["away_price"], ml3["draw_price"]) == (135, 145, 310)

    def test_tie_no_bet_line_is_not_the_moneyline(self):
        # MTL@TOR also lists MONEY_LINE_2_WAY_TNB at -115 / -115
        ml = by_market(parse_partner(load("nhl_partner_us.json"), "partner-US"))[(TOR, "ml")]
        assert (ml["home_price"], ml["away_price"]) == (-112, -108)

    def test_fanduel_canada(self):
        rows = parse_partner(load("nhl_partner_ca.json"), "partner-CA")
        got = by_market(rows)
        assert {r["book"] for r in rows} == {"fanduel"}
        assert (got[(CAR, "ml")]["home_price"], got[(CAR, "ml")]["away_price"]) == (-125, 104)
        assert (got[(CAR, "total")]["over_price"], got[(CAR, "total")]["under_price"]) == (114, -140)
        assert got[(CAR, "ml")]["feed_updated_utc"] == dt.datetime(2026, 9, 19, 12, 30)

    @staticmethod
    def _game(home_odds, away_odds):
        return {"lastUpdatedUTC": "2026-09-29T12:00:00Z",
                "bettingPartner": {"partnerId": 9, "name": "DraftKings"},
                "games": [{"gameId": CAR, "startTimeUTC": "2026-09-29T21:00:00Z",
                           "homeTeam": {"odds": home_odds},
                           "awayTeam": {"odds": away_odds}}]}

    def test_over_and_under_found_on_either_team(self):
        payload = self._game(
            [{"description": "OVER_UNDER", "value": -110.0, "qualifier": "U5.5"}],
            [{"description": "OVER_UNDER", "value": -110.0, "qualifier": "O5.5"}])
        (row,) = parse_partner(payload, "partner-US")
        assert (row["market"], row["line"]) == ("total", 5.5)

    def test_a_market_missing_a_side_or_mismatched_is_left_out(self):
        payload = self._game(
            [{"description": "PUCK_LINE", "value": 190.0, "qualifier": "-1.5"},
             {"description": "OVER_UNDER", "value": -110.0, "qualifier": "O6.5"},
             {"description": "MONEY_LINE_2_WAY", "value": -125.0, "qualifier": ""}],
            [{"description": "OVER_UNDER", "value": -110.0, "qualifier": "U5.5"},
             {"description": "PUCK_LINE", "value": -230.0, "qualifier": "-1.5"}])
        # no away moneyline; totals at different numbers; both sides at -1.5
        assert parse_partner(payload, "partner-US") == []

    def test_a_three_way_price_passed_off_as_a_moneyline_is_dropped(self, caplog):
        payload = self._game(
            [{"description": "MONEY_LINE_2_WAY", "value": 135.0, "qualifier": ""}],
            [{"description": "MONEY_LINE_2_WAY", "value": 145.0, "qualifier": ""}])
        with caplog.at_level(logging.WARNING, logger="nhl.ingestion.nhl_odds"):
            assert parse_partner(payload, "partner-US") == []
        assert "don't add up like a real price" in caplog.text
        assert "add up to 0.83" in caplog.text

    def test_malformed_payloads_give_no_rows(self):
        assert parse_partner(None, "partner-US") == []
        assert parse_partner({"games": [{"gameId": None}, "junk"]}, "partner-US") == []
        assert parse_partner({"games": [{"gameId": CAR, "homeTeam": None}]},
                             "partner-US") == []


# ── schedule ───────────────────────────────────────────────────────

class TestParseSchedule:
    def test_books_named_from_odds_partners_and_decimal_converted(self):
        rows = parse_schedule(load("nhl_schedule.json"))
        car = {r["book"]: r for r in rows if r["game_id"] == CAR}
        assert set(car) == {"fanduel", "veikkaus", "tipsport", "sportradar", "draftkings"}
        assert (car["draftkings"]["home_price"], car["draftkings"]["away_price"]) == (-125, 105)
        assert (car["fanduel"]["home_price"], car["fanduel"]["away_price"]) == (-125, 104)
        assert (car["tipsport"]["home_price"], car["tipsport"]["away_price"]) == (-137, -101)
        assert (car["veikkaus"]["home_price"], car["veikkaus"]["away_price"]) == (112, -139)
        assert {r["market"] for r in rows} == {"ml"}
        assert {r["source"] for r in rows} == {"schedule"}
        assert all(r["feed_updated_utc"] is None and r["state"] == "FUT" for r in rows)

    def test_every_game_of_the_odds_date_and_none_after(self):
        rows = parse_schedule(load("nhl_schedule.json"))
        assert {r["game_id"] for r in rows} == {CAR, TOR, BOS, EDM, VGK}
        assert len(rows) == 5 + 5 + 4 + 4 + 4        # books quoting each game

    def test_unlisted_provider_is_named_by_its_id(self):
        payload = {"oddsPartners": [], "gameWeek": [{"games": [{
            "id": CAR, "startTimeUTC": "2026-09-29T21:00:00Z", "gameState": "FUT",
            "homeTeam": {"odds": [{"providerId": 42, "value": "-120"}]},
            "awayTeam": {"odds": [{"providerId": 42, "value": "+100"}]}}]}]}
        (row,) = parse_schedule(payload)
        assert row["book"] == "provider42"


# ── Checks before storing ──────────────────────────────────────────

class TestSwappedBooks:
    def test_veikkaus_flagged_on_the_captured_slate(self):
        rows = (parse_schedule(load("nhl_schedule.json"))
                + parse_partner(load("nhl_partner_us.json"), "partner-US")
                + parse_partner(load("nhl_partner_ca.json"), "partner-CA"))
        flagged = swapped_books(rows)
        assert set(flagged) == {"veikkaus"}
        disagreed, judged = flagged["veikkaus"]
        assert disagreed >= 2 and 2 * disagreed >= judged

    def test_books_that_agree_are_not_flagged(self):
        rows = [{"game_id": g, "market": "ml", "book": b, "home_price": h,
                 "away_price": a}
                for g in (1, 2, 3)
                for b, h, a in (("a", -150, 130), ("b", -145, 125), ("c", -155, 135))]
        assert swapped_books(rows) == {}
        rows += [{"game_id": g, "market": "ml", "book": "d", "home_price": 130,
                  "away_price": -150} for g in (1, 2, 3)]
        assert swapped_books(rows) == {"d": (3, 3)}


class TestSelectRows:
    NOW = dt.datetime(2026, 9, 29, 20, 0, tzinfo=UTC)
    STARTS = {CAR: dt.datetime(2026, 9, 29, 21, 0, tzinfo=UTC),
              TOR: dt.datetime(2026, 9, 29, 23, 0, tzinfo=UTC)}

    def row(self, game_id=CAR, book="draftkings", start="feed", state=None):
        start = self.STARTS.get(game_id) if start == "feed" else start
        return {"game_id": game_id, "book": book, "start": start, "state": state}

    def test_excluded_books_setting(self):
        assert excluded_books({}) == frozenset({"veikkaus"})
        assert excluded_books({"NHL_FEED_EXCLUDE_BOOKS": ""}) == frozenset()
        assert excluded_books({"NHL_FEED_EXCLUDE_BOOKS": " Veikkaus , Tipsport,"}) == {
            "veikkaus", "tipsport"}

    def test_what_is_kept_and_why_the_rest_is_not(self):
        rows = [self.row(),                                    # kept
                self.row(book="veikkaus"),                     # excluded
                self.row(game_id=9),                           # unknown to raw.games
                self.row(state="LIVE"),                        # under way
                self.row(start=self.NOW),                      # puck drop now
                self.row(game_id=TOR, start=None)]             # DB start used: kept
        kept, skipped = select_rows(rows, self.STARTS, self.NOW, {"veikkaus"})
        assert [(r["game_id"], r["book"]) for r in kept] == [
            (CAR, "draftkings"), (TOR, "draftkings")]
        assert skipped == {"excluded": 1, "unknown_game": 1, "in_play": 2, "no_start": 0}

    def test_no_start_time_anywhere_is_not_stored(self):
        kept, skipped = select_rows([self.row(start=None)], {CAR: None}, self.NOW, set())
        assert kept == [] and skipped["no_start"] == 1


# ── store / snapshot without a database ────────────────────────────

class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, starts):
        self.starts, self.statements = starts, []

    def execute(self, stmt, params=None):
        self.statements.append((str(stmt), params))
        if "FROM raw.games" in str(stmt):
            return _Result([(g, s) for g, s in self.starts.items()
                            if g in params["ids"]])
        return _Result([])


class _FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def begin(self):
        yield self.conn


def _all_fixture_rows():
    return (parse_partner(load("nhl_partner_us.json"), "partner-US")
            + parse_partner(load("nhl_partner_ca.json"), "partner-CA")
            + parse_schedule(load("nhl_schedule.json")))


class TestStore:
    NOW = dt.datetime(2026, 9, 29, 18, 0, 5, 123, tzinfo=UTC)
    STARTS = {g: None for g in (CAR, TOR, BOS, EDM, VGK)}

    @pytest.fixture()
    def conn(self, monkeypatch):
        conn = _FakeConn(self.STARTS)
        monkeypatch.setattr(nhl_odds, "engine", _FakeEngine(conn))
        return conn

    def inserted(self, conn):
        (sql, params), = [(s, p) for s, p in conn.statements if "INSERT" in s]
        return sql, params

    def test_writes_only_its_own_table(self, conn, monkeypatch):
        monkeypatch.delenv("NHL_FEED_EXCLUDE_BOOKS", raising=False)
        n = nhl_odds.store(_all_fixture_rows(), self.NOW)
        sql, params = self.inserted(conn)
        assert "INSERT INTO raw.nhl_feed_snapshots" in sql
        assert not any("odds_snapshots" in s.replace("nhl_feed_snapshots", "")
                       for s, _ in conn.statements)
        assert n == len(params) == 8 + 8 + 22 - 5        # veikkaus left out
        assert "veikkaus" not in {p["book"] for p in params}
        assert {p["captured_at"] for p in params} == {dt.datetime(2026, 9, 29, 18, 0, 5, 123)}
        assert set(params[0]) == {"captured_at", "game_id", "source", "book", "market",
                                  "home_price", "away_price", "over_price",
                                  "under_price", "draw_price", "line", "feed_updated_utc"}

    def test_swapped_book_is_reported_and_kept_when_not_excluded(
            self, conn, monkeypatch, caplog):
        monkeypatch.setenv("NHL_FEED_EXCLUDE_BOOKS", "")
        with caplog.at_level(logging.INFO, logger="nhl.ingestion.nhl_odds"):
            n = nhl_odds.store(_all_fixture_rows(), self.NOW)
        assert n == 8 + 8 + 22
        warned = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("veikkaus looks side-swapped" in r.getMessage() for r in warned)

    def test_excluded_swapped_book_only_noted(self, conn, monkeypatch, caplog):
        monkeypatch.delenv("NHL_FEED_EXCLUDE_BOOKS", raising=False)
        with caplog.at_level(logging.INFO, logger="nhl.ingestion.nhl_odds"):
            nhl_odds.store(_all_fixture_rows(), self.NOW)
        notes = [r for r in caplog.records if "side-swapped" in r.getMessage()]
        assert notes and all(r.levelno == logging.INFO for r in notes)
        assert "veikkaus (excluded)" in notes[0].getMessage()

    def test_in_play_and_unknown_games_are_not_stored(self, conn, caplog):
        after_first_drop = dt.datetime(2026, 9, 29, 21, 30, tzinfo=UTC)
        del conn.starts[TOR]
        with caplog.at_level(logging.INFO, logger="nhl.ingestion.nhl_odds"):
            nhl_odds.store(_all_fixture_rows(), after_first_drop)
        _, params = self.inserted(conn)
        assert {p["game_id"] for p in params} == {BOS, EDM, VGK}
        assert "for games raw.games doesn't have" in caplog.text

    def test_nothing_to_store_makes_no_insert(self, conn):
        assert nhl_odds.store([], self.NOW) == 0
        assert not any("INSERT" in s for s, _ in conn.statements)


class TestSnapshot:
    @pytest.fixture()
    def wired(self, monkeypatch):
        """snapshot() with the network faked from the fixtures and store()
        captured; returns (urls requested, rows handed to store)."""
        urls, stored = [], []
        pages = {"/partner-game/US/now": load("nhl_partner_us.json"),
                 "/partner-game/CA/now": load("nhl_partner_ca.json"),
                 "/schedule/2026-09-29": load("nhl_schedule.json")}

        def fake_get(url, timeout=None, headers=None):
            urls.append(url)
            r = requests.Response()
            path = url.replace(nhl_odds.API_WEB, "")
            r.status_code, r.reason = (200, "OK") if path in pages else (404, "Not Found")
            r._content = json.dumps(pages.get(path, {})).encode()
            return r

        monkeypatch.setattr(nhl_odds.requests, "get", fake_get)
        monkeypatch.setattr(nhl_odds.time, "sleep", lambda s: None)
        monkeypatch.setattr(nhl_odds, "ensure_schema", lambda: None)
        monkeypatch.setattr(nhl_odds, "ensure_table", lambda: None)
        monkeypatch.setattr(nhl_odds, "local_today", lambda: dt.date(2026, 9, 29))
        monkeypatch.setattr(nhl_odds, "store",
                            lambda rows, now: stored.append((rows, now)) or len(rows))
        return urls, stored, pages

    def test_three_sources_requested_and_parsed(self, wired):
        urls, stored, _ = wired
        assert nhl_odds.snapshot() == 8 + 8 + 22
        assert urls == ["https://api-web.nhle.com/v1/partner-game/US/now",
                        "https://api-web.nhle.com/v1/partner-game/CA/now",
                        "https://api-web.nhle.com/v1/schedule/2026-09-29"]
        (rows, now), = stored
        assert {r["source"] for r in rows} == {"partner-US", "partner-CA", "schedule"}
        assert now.tzinfo is not None

    def test_a_failing_source_does_not_stop_the_others(self, wired, caplog):
        urls, _stored, pages = wired
        del pages["/partner-game/CA/now"]
        with caplog.at_level(logging.ERROR, logger="nhl.ingestion.nhl_odds"):
            assert nhl_odds.snapshot() == 8 + 22
        assert "NHL odds feed partner-CA failed: HTTP 404 Not Found" in caplog.text
        assert len(urls) == 3

    def test_a_layout_change_is_logged_not_raised(self, wired, caplog, monkeypatch):
        def broken(payload, source):
            raise KeyError("games")
        monkeypatch.setattr(nhl_odds, "parse_partner", broken)
        with caplog.at_level(logging.ERROR, logger="nhl.ingestion.nhl_odds"):
            assert nhl_odds.snapshot() == 22
        assert "could not read the response" in caplog.text

    def test_idle_days_make_no_request_when_asked(self, wired, monkeypatch):
        urls, stored, _ = wired
        monkeypatch.setattr(nhl_odds, "upcoming_start_times", lambda now, horizon: [])
        assert nhl_odds.snapshot(skip_when_idle=True) == 0
        assert urls == [] and stored == []

    def test_pauses_between_requests(self, wired, monkeypatch):
        pauses = []
        monkeypatch.setattr(nhl_odds.time, "sleep", pauses.append)
        nhl_odds.snapshot()
        assert pauses == [nhl_odds.REQUEST_PAUSE_S] * 2


# ── Comparing with The Odds API (pure) ─────────────────────────────

T0 = dt.datetime(2026, 9, 29, 15, 0)       # naive UTC, as stored


def feed_row(at, home=-125, away=105, market="ml", book="draftkings",
             source="partner-US", line=None, game_id=CAR, **extra):
    row = {"game_id": game_id, "captured_at": at, "source": source, "book": book,
           "market": market, "home_price": home, "away_price": away,
           "over_price": None, "under_price": None, "draw_price": None,
           "line": line, "feed_updated_utc": None}
    row.update(extra)
    return row


def odds_row(at, home=-120, away=100, market="ml", book="draftkings", line=None,
             game_id=CAR, **extra):
    row = {"game_id": game_id, "captured_at": at, "source": "odds-api", "book": book,
           "market": market, "home_price": home, "away_price": away,
           "over_price": None, "under_price": None, "line": line}
    row.update(extra)
    return row


class TestPairing:
    def test_gap_measures(self):
        assert cents_diff(-120, -125) == 5
        assert cents_diff(105, -105) == 10         # across even money
        assert cents_diff(-110, 110) == -20
        assert cents_diff(None, -110) is None
        assert prob_diff(-125, -120) == pytest.approx((125 / 225 - 120 / 220) * 100)
        assert prob_diff(-125, -120) > 0           # the feed pays less

    def test_nearest_snapshot_within_ten_minutes(self):
        odds = [odds_row(T0 + dt.timedelta(minutes=9)),
                odds_row(T0 - dt.timedelta(minutes=4), home=-125, away=105),
                odds_row(T0 + dt.timedelta(minutes=2), book="fanduel")]
        pairs = pair_prices([feed_row(T0)], odds)
        assert [(p["side"], p["status"], p["odds_price"], p["cents"]) for p in pairs] == [
            ("home", "ok", -125, 0), ("away", "ok", 105, 0)]
        assert pairs[0]["minutes_apart"] == -4.0

    def test_eleven_minutes_is_unpaired(self):
        pairs = pair_prices([feed_row(T0)], [odds_row(T0 + dt.timedelta(minutes=11))])
        assert {p["status"] for p in pairs} == {"unpaired"}
        assert all(p["odds_price"] is None and p["prob_pp"] is None for p in pairs)
        # exactly ten minutes still pairs
        pairs = pair_prices([feed_row(T0)], [odds_row(T0 + dt.timedelta(minutes=10))])
        assert {p["status"] for p in pairs} == {"ok"}

    def test_different_line_is_not_compared(self):
        feed = [feed_row(T0, market="total", home=None, away=None,
                         over_price=105, under_price=-125, line=6.5)]
        from decimal import Decimal
        same = [odds_row(T0, market="total", home=None, away=None,
                         over_price=100, under_price=-120, line=Decimal("6.5"))]
        other = [odds_row(T0, market="total", home=None, away=None,
                          over_price=-140, under_price=120, line=Decimal("5.5"))]
        ok = pair_prices(feed, same)
        assert [(p["side"], p["cents"]) for p in ok] == [("over", 5), ("under", -5)]
        differs = pair_prices(feed, other)
        assert {p["status"] for p in differs} == {"line_differs"}
        assert all(p["cents"] is None for p in differs)

    def test_only_books_and_markets_both_sources_have(self):
        feed = [feed_row(T0, book="tipsport", source="schedule"),
                feed_row(T0, market="ml3", draw_price=310)]
        assert pair_prices(feed, [odds_row(T0)]) == []

    def test_summary(self):
        feed = [feed_row(T0), feed_row(T0 + dt.timedelta(minutes=30))]
        odds = [odds_row(T0 + dt.timedelta(minutes=1), home=-125, away=100)]
        (s,) = summarize_pairs(pair_prices(feed, odds))
        assert (s["paired"], s["identical"], s["unpaired"], s["line_differs"]) == (2, 1, 2, 0)
        assert s["mean_abs_cents"] == 2.5
        assert s["max_abs_pp"] == pytest.approx(abs(prob_diff(105, 100)))


class TestFreshness:
    def test_changes_counted_per_series(self):
        stamp = dt.datetime(2026, 8, 28, 18, 0, 38)
        rows = [feed_row(T0 + dt.timedelta(minutes=15 * i), home=h, feed_updated_utc=stamp)
                for i, h in enumerate((-125, -125, -130, -130))]
        rows += [odds_row(T0 + dt.timedelta(minutes=15 * i), home=h)
                 for i, h in enumerate((-120, -122, -125))]
        series = {(s["source"], s["book"]): s for s in freshness(rows)}
        feed = series[("partner-US", "draftkings")]
        assert (feed["snapshots"], feed["changes"], feed["feed_stamps"]) == (4, 1, 1)
        assert feed["last_change_at"] == T0 + dt.timedelta(minutes=30)
        odds = series[("odds-api", "draftkings")]
        assert (odds["snapshots"], odds["changes"], odds["feed_stamps"]) == (3, 2, 0)
        summary = {s["source"]: s for s in summarize_freshness(freshness(rows))}
        assert summary["partner-US"]["moved"] == 1 and summary["odds-api"]["changes"] == 2

    def test_a_moved_line_counts_as_a_change(self):
        rows = [feed_row(T0, market="pl", line=-1.5),
                feed_row(T0 + dt.timedelta(minutes=15), market="pl", line=1.5)]
        (s,) = freshness(rows)
        assert s["changes"] == 1


class TestReport:
    def test_report_with_and_without_pairs(self):
        games = [{"game_id": CAR}]
        feed = [feed_row(T0)]
        empty = {"date": dt.date(2026, 9, 29), "games": games, "feed_rows": 1,
                 "odds_rows": 0, "pairs": pair_prices(feed, []),
                 "pair_summary": summarize_pairs(pair_prices(feed, [])),
                 "freshness": freshness(feed),
                 "freshness_summary": summarize_freshness(freshness(feed))}
        out = format_report(empty, detail=True)
        assert "No Odds API snapshots of draftkings or fanduel" in out
        assert re.search(r"partner-US\s+draftkings\s+ml", out) and "unpaired" in out
        odds = [odds_row(T0)]
        full = dict(empty, odds_rows=1, pairs=pair_prices(feed, odds),
                    pair_summary=summarize_pairs(pair_prices(feed, odds)))
        assert "No Odds API snapshots" not in format_report(full)
        none = dict(empty, games=[])
        assert "No games on this date" in format_report(none)

    def test_report_prints_on_a_windows_console(self):
        """cp1252 (the Windows console) can't print characters such as the
        not-equal sign: the report must be plain ASCII."""
        feed = [feed_row(T0), feed_row(T0, market="total", home=None, away=None,
                                       over_price=105, under_price=-125, line=6.5)]
        odds = [odds_row(T0)]
        result = {"date": dt.date(2026, 9, 29), "games": [{"game_id": CAR}],
                  "feed_rows": 2, "odds_rows": 1, "pairs": pair_prices(feed, odds),
                  "pair_summary": summarize_pairs(pair_prices(feed, odds)),
                  "freshness": freshness(feed + odds),
                  "freshness_summary": summarize_freshness(freshness(feed + odds))}
        format_report(result, detail=True).encode("ascii")
        format_report(dict(result, odds_rows=0, games=[])).encode("ascii")


# ── Command line ───────────────────────────────────────────────────

class TestCommandLine:
    @pytest.fixture()
    def no_work(self, monkeypatch):
        monkeypatch.setattr(nhl_odds, "snapshot", lambda **kw: pytest.fail("snapshot ran"))
        monkeypatch.setattr(nhl_odds, "compare_feeds",
                            lambda *a, **kw: pytest.fail("compare ran"))

    @pytest.mark.parametrize("argv", [["--help"], ["snapshot", "--help"],
                                      ["compare", "--help"]])
    def test_help_runs_nothing(self, no_work, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            nhl_odds.main(argv)
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_bad_date_and_missing_command_are_usage_errors(self, no_work):
        for argv in (["compare", "--date", "2026-13-01"], []):
            with pytest.raises(SystemExit) as exc:
                nhl_odds.main(argv)
            assert exc.value.code == 2

    def test_commands_reach_their_functions(self, monkeypatch, capsys):
        calls = []
        monkeypatch.setattr(nhl_odds, "snapshot", lambda: calls.append("snapshot") or 7)
        monkeypatch.setattr(nhl_odds, "compare_feeds",
                            lambda d: calls.append(d) or {"date": d, "games": [],
                                                          "feed_rows": 0, "odds_rows": 0})
        assert nhl_odds.main(["snapshot"]) == 0
        assert "Stored 7 NHL feed price rows" in capsys.readouterr().out
        assert nhl_odds.main(["compare", "--date", "2026-09-29"]) == 0
        assert nhl_odds.main(["compare"]) == 0
        assert calls == ["snapshot", dt.date(2026, 9, 29), None]

    def test_module_help_runs_nothing(self):
        """python -m ingestion.nhl_odds --help: usage, exit 0, before any
        database or network call (the database is unreachable here)."""
        import os
        import subprocess
        import sys
        root = Path(__file__).parent.parent
        out = subprocess.run(
            [sys.executable, "-m", "ingestion.nhl_odds", "--help"],
            cwd=root, capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(root), "ODDS_API_KEY": "your_key_here",
                 "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "1"})
        assert out.returncode == 0, out.stderr
        assert "usage:" in out.stdout and "snapshot" in out.stdout


# ── Database (a disposable copy only) ──────────────────────────────

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


@requires_db
class TestDatabase:
    IDS = (9_999_030_001, 9_999_030_002)            # synthetic
    DAY = dt.date(2031, 2, 15)                       # no real games then
    START = dt.datetime(2031, 2, 16, 0, 0, tzinfo=UTC)
    NOW = dt.datetime(2031, 2, 15, 18, 0, tzinfo=UTC)

    @pytest.fixture()
    def games(self):
        from config.migrate import ensure_schema
        ensure_schema()
        nhl_odds.ensure_table()
        with engine.begin() as conn:
            for i, game_id in enumerate(self.IDS):
                conn.execute(text("""
                    INSERT INTO raw.games (game_id, season, game_type, date,
                        start_time_utc, home_team, away_team, game_state)
                    VALUES (:g, 20302031, 2, :d, :s, :h, :a, 'FUT')
                """), {"g": game_id, "d": self.DAY,
                       "s": self.START + dt.timedelta(hours=i),
                       "h": ("CAR", "TOR")[i], "a": ("FLA", "MTL")[i]})
        yield
        with engine.begin() as conn:
            for table in ("raw.nhl_feed_snapshots", "raw.odds_snapshots", "raw.games"):
                conn.execute(text(f"DELETE FROM {table} WHERE game_id = ANY(:g)"),
                             {"g": list(self.IDS)})

    def _fixture_rows_on_synthetic_games(self):
        remap = {CAR: self.IDS[0], TOR: self.IDS[1]}
        rows = []
        for r in parse_partner(load("nhl_partner_us.json"), "partner-US"):
            rows.append({**r, "game_id": remap[r["game_id"]], "start": None})
        return rows

    def test_table_and_columns(self, games):
        nhl_odds._table_ready = False
        nhl_odds.ensure_table()                      # safe to repeat
        with engine.connect() as conn:
            cols = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'raw' AND table_name = 'nhl_feed_snapshots'
            """)).scalars().all()
        assert set(cols) == {"id", "captured_at", "game_id", "source", "book", "market",
                             "home_price", "away_price", "over_price", "under_price",
                             "draw_price", "line", "feed_updated_utc"}

    def test_store_writes_the_feed_table_and_never_odds_snapshots(self, games):
        def odds_count():
            with engine.connect() as conn:
                return conn.execute(text("SELECT COUNT(*) FROM raw.odds_snapshots")).scalar()
        before = odds_count()
        n = nhl_odds.store(self._fixture_rows_on_synthetic_games(), self.NOW)
        assert n == 8
        assert odds_count() == before
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT captured_at, source, book, over_price, under_price, line,
                       feed_updated_utc
                FROM raw.nhl_feed_snapshots
                WHERE game_id = :g AND market = 'total'
            """), {"g": self.IDS[0]}).one()
        assert row.captured_at == dt.datetime(2031, 2, 15, 18, 0)     # naive UTC
        assert (row.source, row.book) == ("partner-US", "draftkings")
        assert (row.over_price, row.under_price, float(row.line)) == (105, -125, 6.5)
        assert row.feed_updated_utc == dt.datetime(2026, 8, 28, 18, 0, 38)

    def test_compare_feeds_pairs_and_counts_changes(self, games):
        rows = self._fixture_rows_on_synthetic_games()
        nhl_odds.store(rows, self.NOW)
        moved = [dict(r, away_price=110) if r["market"] == "ml" else r for r in rows]
        nhl_odds.store(moved, self.NOW + dt.timedelta(minutes=30))
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.odds_snapshots (game_id, captured_at, book_name,
                    market_type, home_price, away_price, over_price, under_price, line)
                VALUES (:g, :t, 'draftkings', 'ml', -120, 100, NULL, NULL, NULL),
                       (:g, :t, 'draftkings', 'total', NULL, NULL, 100, -120, 5.5)
            """), {"g": self.IDS[0], "t": dt.datetime(2031, 2, 15, 18, 3)})

        result = nhl_odds.compare_feeds(self.DAY)
        assert [g["game_id"] for g in result["games"]] == list(self.IDS)
        assert (result["feed_rows"], result["odds_rows"]) == (16, 2)
        ml = [p for p in result["pairs"]
              if p["game_id"] == self.IDS[0] and p["market"] == "ml"]
        first = [p for p in ml if p["status"] == "ok"]
        assert [(p["side"], p["feed_price"], p["odds_price"], p["cents"]) for p in first] == [
            ("home", -125, -120, -5), ("away", 105, 100, 5)]
        assert first[0]["minutes_apart"] == 3.0
        assert {p["status"] for p in ml if p["feed_at"] == dt.datetime(2031, 2, 15, 18, 30)} == {
            "unpaired"}                                  # 27 minutes from the odds row
        total = [p for p in result["pairs"] if p["game_id"] == self.IDS[0]
                 and p["market"] == "total" and p["odds_price"] is not None]
        assert {p["status"] for p in total} == {"line_differs"}
        series = {(s["game_id"], s["source"], s["market"]): s for s in result["freshness"]}
        assert series[(self.IDS[0], "partner-US", "ml")]["changes"] == 1
        assert series[(self.IDS[0], "partner-US", "total")]["changes"] == 0
        assert series[(self.IDS[0], "odds-api", "ml")]["snapshots"] == 1
        assert re.search(r"partner-US\s+draftkings\s+ml\s+home",
                         format_report(result, detail=True))
