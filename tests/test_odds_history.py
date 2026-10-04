"""
Tests for ingestion/odds_history.py: the purchase plan (start-time
clusters, morning snapshots, spread order), the never-buy-twice check,
parsing a recorded historical response (tests/fixtures/
odds_api_historical_nhl.json, a trimmed real 2024-10-04 snapshot), the
in-play drop, the credit caps, paid calls that fail after payment
(paid_unparsed) and their re-parse, and key redaction. No network (the getter
is a stub). The pure tests need no database (the saver is a list); the
database tests at the end run only against a disposable copy (see
tests/conftest.py) and write only synthetic rows (event ids starting
"dbtest-", purpose "dbtest", season 20302031), which they delete.
"""
import datetime as dt
import json
import logging
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import odds_history as oh

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="needs a disposable database (see tests/conftest.py)")

UTC = dt.timezone.utc
FIXTURE = Path(__file__).parent / "fixtures" / "odds_api_historical_nhl.json"
BOOKS = ",".join(oh.DEFAULT_BOOKMAKERS)


def t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def g(game_id, start, season=20242025, date=None, home="BOS", away="OTT"):
    start = t(start) if isinstance(start, str) else start
    return {"game_id": game_id, "season": season, "home_team": home, "away_team": away,
            "date": date or (start - dt.timedelta(hours=5)).date() if start else date,
            "start_time_utc": start}


@pytest.fixture()
def body():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ── Plan ──────────────────────────────────────────────────────────

class TestClusters:
    def test_starts_within_75_minutes_of_the_first_share_a_cluster(self):
        starts = [t("2025-01-11T00:00"), t("2025-01-11T00:30"), t("2025-01-11T01:15"),
                  t("2025-01-11T01:16"), t("2025-01-11T03:00"), t("2025-01-11T00:00")]
        assert oh.cluster_starts(starts) == [
            [t("2025-01-11T00:00"), t("2025-01-11T00:30"), t("2025-01-11T01:15")],
            [t("2025-01-11T01:16")],
            [t("2025-01-11T03:00")]]

    def test_a_cluster_is_anchored_on_its_first_start_not_chained(self):
        # 7:00, 8:00, 9:00 ET: 9:00 is 120 min after the anchor, a new cluster
        starts = [t("2025-01-11T00:00"), t("2025-01-11T01:00"), t("2025-01-11T02:00")]
        assert len(oh.cluster_starts(starts)) == 2

    def test_close_plan_is_first_start_minus_10_per_cluster(self):
        games = [g(1, "2025-01-11T00:00"), g(2, "2025-01-11T00:00"),
                 g(3, "2025-01-11T00:30"), g(4, "2025-01-11T03:00"),
                 g(5, None, date=dt.date(2025, 1, 10))]          # no start: left out
        plan = oh.plan_requests(games, "close")
        assert [(p.requested_ts, p.n_games) for p in plan] == [
            (t("2025-01-10T23:50"), 3), (t("2025-01-11T02:50"), 1)]
        assert {p.purpose for p in plan} == {"close"}
        assert plan[0].game_date == dt.date(2025, 1, 10)

    def test_morning_is_10am_chicago_in_winter_and_summer(self):
        assert oh.morning_time(dt.date(2025, 1, 10)) == t("2025-01-10T16:00")
        assert oh.morning_time(dt.date(2025, 4, 10)) == t("2025-04-10T15:00")
        games = [g(1, "2025-01-11T00:00"), g(2, "2025-01-11T03:00")]
        (p,) = oh.plan_requests(games, "morning")
        assert (p.requested_ts, p.n_games) == (t("2025-01-10T16:00"), 2)

    def test_morning_skipped_when_every_game_starts_before_it(self):
        early = g(1, "2025-01-10T15:00", date=dt.date(2025, 1, 10))
        assert oh.plan_requests([early], "morning") == []

    def test_unknown_purpose(self):
        with pytest.raises(ValueError):
            oh.plan_requests([g(1, "2025-01-11T00:00")], "opening")

    def test_spread_order_prefixes_cover_the_range(self):
        order = oh.spread_order(8)
        assert sorted(order) == list(range(8))
        assert order[:4] == [0, 4, 2, 6]
        assert sorted(oh.spread_order(223)) == list(range(223))
        first_half = oh.spread_order(223)[:112]
        assert min(first_half) == 0 and max(first_half) > 200
        assert oh.spread_order(0) == [] and oh.spread_order(1) == [0]

    def test_build_plan_keeps_step_priority(self):
        games = [g(1, "2025-01-11T00:00"), g(2, "2024-01-11T00:00", season=20232024)]
        plan = oh.build_plan([("close", 20242025), ("morning", 20242025),
                              ("close", 20232024)], games)
        assert [(p.purpose, p.season) for p in plan] == [
            ("close", 20242025), ("morning", 20242025), ("close", 20232024)]

    def test_parse_steps(self):
        assert oh.parse_steps("close:20242025, morning:20242025") == [
            ("close", 20242025), ("morning", 20242025)]
        with pytest.raises(Exception):
            oh.parse_steps("opening:20242025")
        with pytest.raises(Exception):
            oh.parse_steps("close:2024")


class TestCost:
    def test_ten_credits_per_market_with_up_to_ten_books(self):
        assert oh.call_cost("h2h,totals", BOOKS) == 20
        assert oh.call_cost("h2h", BOOKS) == 10
        eleven = BOOKS + ",bovada2"
        assert oh.call_cost("h2h,totals", eleven) == 40

    def test_default_books_include_the_reference_books(self):
        assert len(oh.DEFAULT_BOOKMAKERS) == 10
        assert {"pinnacle", "draftkings", "fanduel"} <= set(oh.DEFAULT_BOOKMAKERS)


class TestCovered:
    def test_same_requested_time(self):
        done = [{"requested_ts": dt.datetime(2025, 1, 10, 23, 50),
                 "snapshot_ts": None, "next_ts": None}]
        assert oh.is_covered(t("2025-01-10T23:50"), done)

    def test_inside_a_bought_snapshot_window(self):
        done = [{"requested_ts": dt.datetime(2025, 1, 10, 23, 50),
                 "snapshot_ts": dt.datetime(2025, 1, 10, 23, 45, 38),
                 "next_ts": dt.datetime(2025, 1, 10, 23, 50, 38)}]
        assert oh.is_covered(t("2025-01-10T23:46"), done)
        assert oh.is_covered(t("2025-01-10T23:45:38"), done)
        assert not oh.is_covered(t("2025-01-10T23:50:38"), done)
        assert not oh.is_covered(t("2025-01-10T23:45"), done)


class TestCoveringFetches:
    ROWS = [
        {"requested_ts": 1, "markets": "h2h,totals", "bookmakers": BOOKS, "status": "ok"},
        {"requested_ts": 2, "markets": "h2h", "bookmakers": "pinnacle", "status": "ok"},
        {"requested_ts": 3, "markets": "h2h,totals", "bookmakers": BOOKS, "status": "error"},
        {"requested_ts": 4, "markets": "h2h", "bookmakers": "regions:us", "status": "probe"},
        {"requested_ts": 5, "markets": "h2h,totals", "bookmakers": "pinnacle", "status": "empty"},
    ]

    def ids(self, *a, **k):
        return [r["requested_ts"] for r in oh.covering_fetches(self.ROWS, *a, **k)]

    def test_any_books_by_default(self):
        assert self.ids("h2h,totals", BOOKS) == [1, 5]
        assert self.ids("h2h", "draftkings") == [1, 2, 5]

    def test_a_call_missing_a_market_does_not_cover(self):
        assert 2 not in self.ids("totals", BOOKS)

    def test_same_books_only(self):
        assert self.ids("h2h,totals", BOOKS, same_books=True) == [1]


# ── Parsing a recorded response ───────────────────────────────────

class TestParseSnapshot:
    GAMES = [g(2024020001, "2024-10-04T17:00", home="BUF", away="NJD"),
             g(2024020002, "2024-10-05T15:00", home="NJD", away="BUF"),
             g(2024020010, "2024-10-09T23:00", home="MTL", away="TOR")]

    def test_rows_per_book_market_and_side(self, body):
        rows, stats = oh.parse_snapshot(body, t("2024-10-04T16:50"), self.GAMES)
        assert stats["events"] == 3 and stats["matched"] == 3 and stats["in_play"] == 0
        # 3 events x 3 books x (2 h2h + 2 totals)
        assert len(rows) == stats["rows"] == 36
        first = [r for r in rows if r["game_id"] == 2024020001 and r["book"] == "draftkings"]
        by = {(r["market"], r["side"]): r for r in first}
        assert by[("h2h", "home")]["price"] == 124          # Buffalo, the home team
        assert by[("h2h", "away")]["price"] == -148
        assert by[("totals", "over")]["price"] == 102
        assert by[("totals", "over")]["point"] == 6.5
        assert by[("h2h", "home")]["point"] is None
        r = by[("h2h", "home")]
        assert r["snapshot_ts"] == dt.datetime(2024, 10, 4, 16, 45, 39)     # naive UTC
        assert r["requested_ts"] == dt.datetime(2024, 10, 4, 16, 50)
        assert r["commence_time"] == dt.datetime(2024, 10, 4, 17, 10)
        assert r["book_updated_at"] == dt.datetime(2024, 10, 4, 16, 45, 27)
        assert r["event_id"] == "e1dd2bc0fa38ee53116f047cf3d0327e"

    def test_accented_team_name_matches(self, body):
        rows, _ = oh.parse_snapshot(body, t("2024-10-04T16:50"), self.GAMES)
        assert any(r["game_id"] == 2024020010 for r in rows)

    def test_unmatched_event_is_kept_without_a_game(self, body):
        rows, stats = oh.parse_snapshot(body, t("2024-10-04T16:50"), self.GAMES[:1])
        assert stats["unmatched"] == 2
        loose = [r for r in rows if r["game_id"] is None]
        assert loose and all(r["home_name"] and r["away_name"] for r in loose)

    def test_started_events_are_dropped(self, body):
        # snapshot after the first game's commence_time: it is in play
        body["timestamp"] = "2024-10-04T17:15:00Z"
        rows, stats = oh.parse_snapshot(body, t("2024-10-04T17:20"), self.GAMES)
        assert stats["in_play"] == 1
        assert not any(r["game_id"] == 2024020001 for r in rows)

    def test_nhl_start_before_api_commence_also_counts_as_started(self, body):
        # 17:05: past the NHL's 17:00 puck drop, before the API's 17:10
        body["timestamp"] = "2024-10-04T17:05:00Z"
        rows, stats = oh.parse_snapshot(body, t("2024-10-04T17:05"), self.GAMES)
        assert stats["in_play"] == 1
        assert not any(r["game_id"] == 2024020001 for r in rows)

    def test_no_timestamp_or_odd_outcomes(self, body):
        assert oh.parse_snapshot({"data": body["data"]}, t("2024-10-04T16:50"), []) == (
            [], {"events": 0, "in_play": 0, "matched": 0, "unmatched": 0, "rows": 0})
        mk = body["data"][0]["bookmakers"][0]["markets"]
        mk[0]["outcomes"].append({"name": "Draw", "price": 400})
        mk[0]["outcomes"][0]["price"] = None
        mk.append({"key": "spreads", "outcomes": [{"name": "Over", "price": 100}]})
        rows, _ = oh.parse_snapshot(body, t("2024-10-04T16:50"), self.GAMES)
        assert len(rows) == 35          # the None price and Draw dropped, spreads ignored


# ── The fetch loop ────────────────────────────────────────────────

class _Getter:
    def __init__(self, body, remaining=10000, last=20, status=200):
        self.body, self.remaining, self.last, self.status = body, remaining, last, status
        self.calls = []

    def __call__(self, path, params):
        self.calls.append((path, params))
        self.remaining -= self.last
        headers = {"x-requests-last": str(self.last),
                   "x-requests-remaining": str(self.remaining)}
        if self.status != 200:
            return None, headers, self.status
        return json.loads(json.dumps(self.body)), headers, 200


def _plan(n, start="2024-10-04T16:50"):
    base = t(start)
    return [oh.PlannedFetch(base + dt.timedelta(days=i), "close", 20242025,
                            (base + dt.timedelta(days=i)).date(), 1) for i in range(n)]


class TestRunFetch:
    def run(self, plan, getter, budget, done=None, **kw):
        saved = []
        report = oh.run_fetch(plan, "h2h,totals", BOOKS, budget, TestParseSnapshot.GAMES,
                              done if done is not None else [], getter=getter,
                              saver=lambda rows, fetch: saved.append((rows, fetch)),
                              pause=0, **kw)
        return report, saved

    def test_buys_logs_and_never_buys_twice(self, body):
        getter = _Getter(body)
        done = []
        plan = _plan(1)
        report, saved = self.run(plan, getter, oh.Budget(max_credits=100), done)
        assert report.calls == 1 and report.credits == 20 and report.rows == 36
        (path, params), = getter.calls
        assert path == "/historical/sports/icehockey_nhl/odds"
        assert params == {"date": "2024-10-04T16:50:00Z", "markets": "h2h,totals",
                          "bookmakers": BOOKS, "oddsFormat": "american"}
        rows, fetch = saved[0]
        assert fetch["status"] == "ok" and fetch["credits"] == 20 and fetch["n_rows"] == 36
        assert fetch["snapshot_ts"] == dt.datetime(2024, 10, 4, 16, 45, 39)
        assert fetch["next_ts"] == dt.datetime(2024, 10, 4, 16, 50, 39)
        # second run: the logged fetch covers the request, nothing is bought
        report2, _ = self.run(plan, getter, oh.Budget(max_credits=100), done)
        assert report2.calls == 0 and report2.skipped == 1 and len(getter.calls) == 1

    def test_stops_at_this_runs_cap(self, body):
        getter = _Getter(body)
        report, _ = self.run(_plan(5), getter, oh.Budget(max_credits=50))
        assert report.calls == 2 and report.credits == 40
        assert "cap of 50" in report.stopped

    def test_stops_at_the_all_time_cap(self, body):
        getter = _Getter(body)
        report, _ = self.run(_plan(5), getter,
                             oh.Budget(max_credits=1000, cap_total=100, logged_before=60))
        assert report.calls == 2
        assert "all-time cap" in report.stopped

    def test_stops_at_the_reserve(self, body):
        getter = _Getter(body, remaining=6070)
        report, _ = self.run(_plan(5), getter,
                             oh.Budget(max_credits=1000, reserve=6000, remaining=6070))
        # 6070 -> 6050 -> 6030 -> 6010; the next would leave 5990
        assert report.calls == 3
        assert "reserve" in report.stopped

    def test_quota_error_stops_and_is_logged_as_error(self, body):
        getter = _Getter(body, last=0, status=401)
        report, saved = self.run(_plan(3), getter, oh.Budget(max_credits=1000))
        assert report.calls == 1 and report.errors == 1
        assert saved[0][1]["status"] == "error" and saved[0][0] == []
        assert "401" in report.stopped

    def test_empty_snapshot_costs_nothing_and_is_marked_empty(self, body):
        empty = {k: body[k] for k in ("timestamp", "previous_timestamp", "next_timestamp")}
        empty["data"] = []
        report, saved = self.run(_plan(1), _Getter(empty, last=0), oh.Budget(max_credits=20))
        assert report.credits == 0 and saved[0][1]["status"] == "empty"

    def test_a_200_without_timestamp_is_an_error_and_retried(self, body):
        broken = {"data": body["data"]}                  # no timestamp at all
        done = []
        report, saved = self.run(_plan(1), _Getter(broken), oh.Budget(max_credits=100), done)
        ((rows, fetch),) = saved
        assert fetch["status"] == "error" and rows == [] and fetch["credits"] == 20
        assert report.errors == 1 and done == []
        assert oh.covering_fetches([fetch], "h2h,totals", BOOKS) == []
        # the next run asks again
        getter = _Getter(body)
        report2, saved2 = self.run(_plan(1), getter, oh.Budget(max_credits=100), done)
        assert report2.calls == 1 and saved2[0][1]["status"] == "ok"

    def test_stops_after_five_failed_calls_in_a_row(self, body):
        calls = []

        def timeout(path, params):              # no response, no headers
            calls.append(params["date"])
            return None, {}, None

        budget = oh.Budget(max_credits=1000, remaining=10000)
        report, saved = self.run(_plan(8), timeout, budget)
        assert report.calls == len(calls) == oh.MAX_CONSECUTIVE_ERRORS == 5
        assert "5 failed calls in a row" in report.stopped
        # no credit headers: the expected cost is counted as spent
        assert report.credits == budget.spent == 100 and budget.remaining == 9900
        assert [f["credits"] for _, f in saved] == [20] * 5
        assert {f["status"] for _, f in saved} == {"error"}

    def test_errors_that_are_not_in_a_row_do_not_stop(self, body):
        n = {"i": 0}
        ok = _Getter(body)

        def flaky(path, params):
            n["i"] += 1
            if n["i"] % 2:
                return None, {"x-requests-last": "0"}, 500
            return ok(path, params)

        report, _ = self.run(_plan(12), flaky, oh.Budget(max_credits=10000))
        assert report.calls == 12 and report.errors == 6 and report.stopped is None

    def test_missing_headers_count_against_the_cap(self, body):
        def no_headers(path, params):
            return json.loads(json.dumps(body)), {}, 200

        report, saved = self.run(_plan(5), no_headers, oh.Budget(max_credits=50))
        assert report.calls == 2 and report.credits == 40 and "cap of 50" in report.stopped

    def test_limit_and_raw_copy(self, body, tmp_path):
        report, _ = self.run(_plan(3), _Getter(body), oh.Budget(max_credits=1000),
                             limit=1, raw_dir=tmp_path)
        assert report.calls == 1 and "--limit" in report.stopped
        (f,) = list(tmp_path.iterdir())
        stem = oh.raw_stem("close", t("2024-10-04T16:50"), "h2h,totals", BOOKS)
        assert f.name == stem + ".json.gz"
        assert stem.startswith("close_2024-10-04T165000Z_") and len(stem.split("_")[-1]) == 8

    def test_raw_copy_never_overwrites(self, body, tmp_path):
        plan = _plan(1)
        self.run(plan, _Getter(body), oh.Budget(max_credits=1000), raw_dir=tmp_path)
        first = oh.find_raw(tmp_path, oh.raw_stem("close", plan[0].requested_ts,
                                                  "h2h,totals", BOOKS))
        before = first.read_bytes()
        # the same request bought again (a fresh fetch log): a second file
        self.run(plan, _Getter(body), oh.Budget(max_credits=1000), raw_dir=tmp_path)
        names = sorted(f.name for f in tmp_path.iterdir())
        assert len(names) == 2 and names[1].endswith("_2.json.gz")
        assert first.read_bytes() == before
        assert oh.find_raw(tmp_path, first.name[:-len(".json.gz")]).name == names[1]


    def test_describe_plan_counts_what_is_left(self):
        plan = _plan(3)
        done = [{"requested_ts": oh._naive(plan[0].requested_ts),
                 "snapshot_ts": None, "next_ts": None}]
        lines = oh.describe_plan(plan, done, 20)
        assert "1 already bought; to buy 2 x 20 = 40 credits" in lines[0]
        assert lines[-1] == "  total still to buy: 40 credits"


class TestRawNames:
    def test_hash_follows_the_book_list_not_its_order(self):
        when = t("2024-10-04T16:50")
        a = oh.raw_stem("close", when, "h2h,totals", "pinnacle,draftkings")
        assert a == oh.raw_stem("close", when, "totals,h2h", "draftkings, pinnacle")
        assert a != oh.raw_stem("close", when, "h2h,totals", "pinnacle,fanduel")
        assert a != oh.raw_stem("close", when, "h2h", "pinnacle,draftkings")

    def test_save_and_find(self, tmp_path):
        stem = oh.raw_stem("morning", t("2024-10-04T15:00"), "h2h", "pinnacle")
        assert oh.find_raw(tmp_path, stem) is None
        p1 = oh.save_raw(tmp_path, stem, {"n": 1})
        p2 = oh.save_raw(tmp_path, stem, {"n": 2})
        p3 = oh.save_raw(tmp_path, stem, {"n": 3})
        assert (p1.name, p2.name, p3.name) == (f"{stem}.json.gz", f"{stem}_2.json.gz",
                                               f"{stem}_3.json.gz")
        assert oh.find_raw(tmp_path, stem) == p3
        (tmp_path / f"{stem}_x.json.gz").write_bytes(b"")          # not a copy name
        other = oh.raw_stem("morning", t("2024-10-04T15:00"), "h2h", "fanduel")
        oh.save_raw(tmp_path, other, {"n": 9})
        assert oh.find_raw(tmp_path, stem) == p3


class TestPaidUnparsed:
    """A paid call that fails while parsing or storing is still logged
    (paid_unparsed, its credits), counts as bought, and keeps its raw copy
    for `reparse`."""

    def run(self, plan, getter, saver, done, tmp_path):
        return oh.run_fetch(plan, "h2h,totals", BOOKS, oh.Budget(max_credits=1000),
                            TestParseSnapshot.GAMES, done, getter=getter, saver=saver,
                            raw_dir=tmp_path, pause=0)

    def test_store_failure_is_logged_as_paid_unparsed(self, body, tmp_path):
        saved = []

        def saver(rows, fetch):
            if rows:
                raise RuntimeError("insert failed")
            saved.append(dict(fetch))

        done, plan = [], _plan(2)
        report = self.run(plan, _Getter(body), saver, done, tmp_path)
        assert report.calls == 2 and report.credits == 40 and report.errors == 2
        assert [f["status"] for f in saved] == ["paid_unparsed", "paid_unparsed"]
        assert all(f["credits"] == 20 and f["n_rows"] == 0 for f in saved)
        # parsed before the store failed: the snapshot window is logged too
        assert saved[0]["snapshot_ts"] == dt.datetime(2024, 10, 4, 16, 45, 39)
        assert len(list(tmp_path.iterdir())) == 2          # raw copies kept
        # treated as bought: a re-run buys nothing
        getter = _Getter(body)
        again = self.run(plan, getter, saver, done, tmp_path)
        assert again.calls == 0 and again.skipped == 2 and getter.calls == []
        assert oh.covering_fetches(saved, "h2h,totals", BOOKS) == saved

    def test_parse_failure_is_logged_as_paid_unparsed(self, body, tmp_path, monkeypatch):
        def broken(*a, **k):
            raise KeyError("bookmakers")

        monkeypatch.setattr(oh, "parse_snapshot", broken)
        saved = []
        report = self.run(_plan(1), _Getter(body), lambda r, f: saved.append((r, dict(f))),
                          [], tmp_path)
        assert report.errors == 1 and report.credits == 20
        ((rows, fetch),) = saved
        assert rows == [] and fetch["status"] == "paid_unparsed" and fetch["credits"] == 20

    def test_a_log_that_cannot_be_written_stops_the_run(self, body, tmp_path):
        def saver(rows, fetch):
            raise RuntimeError("database down")

        getter = _Getter(body)
        with pytest.raises(RuntimeError):
            self.run(_plan(3), getter, saver, [], tmp_path)
        assert len(getter.calls) == 1
        assert len(list(tmp_path.iterdir())) == 1          # the paid copy survives

    def test_raw_copy_reparses_to_the_same_rows(self, body, tmp_path):
        plan = _plan(1)
        saved = []
        self.run(plan, _Getter(body), lambda r, f: saved.append((r, dict(f))), [], tmp_path)
        path = oh.find_raw(tmp_path, oh.raw_stem("close", plan[0].requested_ts,
                                                 "h2h,totals", BOOKS))
        rows, update = oh.reparse_file(path, oh._naive(plan[0].requested_ts),
                                       TestParseSnapshot.GAMES)
        assert rows == saved[0][0]
        assert update["status"] == "ok" and update["n_rows"] == 36
        assert update["snapshot_ts"] == saved[0][1]["snapshot_ts"]


class TestRematch:
    GAMES = TestParseSnapshot.GAMES

    def ev(self, event_id, home, away, commence):
        return {"event_id": event_id, "home_name": home, "away_name": away,
                "commence_time": commence}

    def test_matches_by_home_team_and_start(self):
        events = [
            # naive UTC, as stored; the API's 17:10 vs the NHL's 17:00
            self.ev("a", "Buffalo Sabres", "New Jersey Devils", dt.datetime(2024, 10, 4, 17, 10)),
            self.ev("b", "New Jersey Devils", "Buffalo Sabres", t("2024-10-05T15:05")),
            self.ev("c", "Montréal Canadiens", "Toronto Maple Leafs", t("2024-10-09T23:00")),
        ]
        assert oh.rematch_pairs(events, self.GAMES) == {
            "a": 2024020001, "b": 2024020002, "c": 2024020010}

    def test_leaves_out_what_cannot_match(self):
        events = [
            self.ev("no_time", "Buffalo Sabres", "New Jersey Devils", None),
            self.ev("unknown", "Quebec Nordiques", "Buffalo Sabres", t("2024-10-04T17:00")),
            self.ev("too_far", "Buffalo Sabres", "New Jersey Devils", t("2024-10-05T01:00")),
            self.ev("wrong_home", "New Jersey Devils", "Buffalo Sabres", t("2024-10-04T17:00")),
        ]
        assert oh.rematch_pairs(events, self.GAMES) == {}

    def test_a_game_without_a_start_time_is_not_a_candidate(self):
        games = [g(7, None, date=dt.date(2024, 10, 4), home="BUF", away="NJD")]
        events = [self.ev("a", "Buffalo Sabres", "New Jersey Devils", t("2024-10-04T17:00"))]
        assert oh.rematch_pairs(events, games) == {}

    def test_season_of(self):
        assert oh.season_of(dt.datetime(2024, 10, 4)) == 20242025
        assert oh.season_of(dt.datetime(2025, 6, 20)) == 20242025
        assert oh.season_of(dt.datetime(2025, 8, 1)) == 20252026


class TestCollectStartTimes:
    def test_one_call_per_week_from_first_to_last_date(self):
        asked = []

        def week(d):
            asked.append(d)
            if d == "2030-10-17":
                raise ConnectionError("schedule down")      # skipped, not fatal
            return {"gameWeek": [{"games": [
                {"id": 2030020001, "startTimeUTC": "2030-10-10T23:00:00Z"},
                {"id": 2030020002, "startTimeUTC": None},          # no time: left out
                {"startTimeUTC": "2030-10-11T23:00:00Z"},          # no id: left out
            ]}]}

        found = oh.collect_start_times(dt.date(2030, 10, 10), dt.date(2030, 10, 24), week,
                                       pause=0)
        assert asked == ["2030-10-10", "2030-10-17", "2030-10-24"]
        assert found == {2030020001: t("2030-10-10T23:00")}


# ── Start-time fill and HTTP ──────────────────────────────────────

def test_int_header_is_case_blind():
    assert oh._int_header({"X-Requests-Remaining": "19960"}, "x-requests-remaining") == 19960
    assert oh._int_header({}, "x-requests-last") is None
    assert oh._int_header({"x-requests-last": "n/a"}, "x-requests-last") is None


def test_http_get_never_logs_the_key(monkeypatch, caplog):
    key = "fakekey0123456789abcdef0123456789"
    monkeypatch.setattr(oh, "_api_key", lambda: key)
    monkeypatch.setattr("ingestion.odds_api.ODDS_API_KEY", key)

    class Resp:
        status_code, reason, headers = 401, "Unauthorized", {}
        text = f"bad key {key}"

        def json(self):
            return {"message": f"apiKey={key} is not valid"}

    monkeypatch.setattr(oh.requests, "get", lambda *a, **k: Resp())
    with caplog.at_level(logging.DEBUG):
        body, _h, status = oh.http_get(oh.HIST_PATH, {"date": "2024-10-04T16:50:00Z"})
    assert body is None and status == 401
    assert key not in caplog.text and "401" in caplog.text


def test_no_key_makes_no_request(monkeypatch):
    monkeypatch.setattr(oh, "_api_key", lambda: "")
    monkeypatch.setattr(oh.requests, "get", lambda *a, **k: pytest.fail("request made"))
    assert oh.http_get(oh.HIST_PATH, {}) == (None, {}, None)


def test_help_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(oh, "ensure_tables", lambda: pytest.fail("--help touched the DB"))
    with pytest.raises(SystemExit) as exc:
        oh.main(["--help"])
    assert exc.value.code == 0
    assert "historical" in capsys.readouterr().out


class TestFetchStart:
    """`fetch` will not start when the account's remaining credits are
    unknown, unless told to."""

    def setup(self, monkeypatch, remaining):
        calls = []
        monkeypatch.setattr(oh, "ensure_tables", lambda: None)
        monkeypatch.setattr(oh, "load_games", lambda seasons: [g(1, "2025-01-11T00:00")])
        monkeypatch.setattr(oh, "load_done", lambda *a, **k: [])
        monkeypatch.setattr(oh, "credits_logged", lambda: 0)
        monkeypatch.setattr(oh, "remaining_credits", lambda: remaining)
        monkeypatch.setattr(oh, "run_fetch",
                            lambda *a, **k: calls.append(a[3]) or oh.RunReport())
        return calls

    def test_refuses_when_remaining_is_unknown(self, monkeypatch, capsys):
        calls = self.setup(monkeypatch, None)
        assert oh.main(["fetch", "--max-credits", "100"]) == 2
        assert calls == [] and "could not be read" in capsys.readouterr().out

    def test_allow_unknown_remaining(self, monkeypatch):
        calls = self.setup(monkeypatch, None)
        assert oh.main(["fetch", "--max-credits", "100", "--allow-unknown-remaining"]) == 0
        assert len(calls) == 1 and calls[0].remaining is None

    def test_known_remaining_starts(self, monkeypatch):
        calls = self.setup(monkeypatch, 9000)
        assert oh.main(["fetch", "--max-credits", "100"]) == 0
        assert calls[0].remaining == 9000


@pytest.mark.parametrize("cmd", [["plan"], ["fetch", "--max-credits", "100"]])
def test_same_books_only_needs_explicit_steps(monkeypatch, capsys, cmd):
    monkeypatch.setattr(oh, "ensure_tables", lambda: pytest.fail("ran without --steps"))
    with pytest.raises(SystemExit) as exc:
        oh.main(cmd + ["--same-books-only"])
    assert exc.value.code == 2 and "--steps" in capsys.readouterr().err


def test_same_books_only_with_steps_runs(monkeypatch):
    seen = {}
    monkeypatch.setattr(oh, "ensure_tables", lambda: None)
    monkeypatch.setattr(oh, "load_games", lambda seasons: seen.setdefault("seasons", seasons)
                        and [])
    monkeypatch.setattr(oh, "load_done",
                        lambda m, b, same_books=False: seen.setdefault("same", same_books)
                        and [])
    assert oh.main(["plan", "--same-books-only", "--steps", "close:20232024"]) == 0
    assert seen == {"seasons": [20232024], "same": True}


def test_default_steps_without_same_books_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(oh, "ensure_tables", lambda: None)
    monkeypatch.setattr(oh, "load_games", lambda seasons: seen.setdefault("seasons", seasons)
                        and [])
    monkeypatch.setattr(oh, "load_done", lambda *a, **k: [])
    assert oh.main(["plan"]) == 0
    assert seen["seasons"] == sorted({s for _, s in oh.parse_steps(oh.DEFAULT_STEPS)})


def test_fetch_requires_a_credit_cap(monkeypatch, capsys):
    monkeypatch.setattr(oh, "ensure_tables", lambda: pytest.fail("ran without a cap"))
    with pytest.raises(SystemExit):
        oh.main(["fetch"])


# ── Database (disposable copy only) ───────────────────────────────

DB_SEASON = 20302031


def _db_body(body):
    """The fixture moved to 2031 with dbtest- event ids, so nothing it
    writes can collide with real purchased rows on a cloned database."""
    raw = json.dumps(body).replace("2024-10-", "2031-10-")
    moved = json.loads(raw)
    for ev in moved["data"]:
        ev["id"] = "dbtest-" + ev["id"]
    return moved


@pytest.fixture()
def db_clean():
    oh.ensure_tables()

    def clean():
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.odds_history WHERE event_id LIKE 'dbtest-%'"))
            conn.execute(text("DELETE FROM raw.odds_history_fetches WHERE purpose = 'dbtest'"))
            conn.execute(text("DELETE FROM raw.games WHERE season IN (:a, :b)"),
                         {"a": DB_SEASON, "b": DB_SEASON + 10001})

    clean()
    yield
    clean()


@requires_db
class TestOnDatabase:
    def test_status_column_is_widened_for_paid_unparsed(self, db_clean):
        with engine.begin() as conn:
            if conn.execute(text("SELECT COALESCE(MAX(LENGTH(status)), 0) "
                                 "FROM raw.odds_history_fetches")).scalar() <= 10:
                conn.execute(text("ALTER TABLE raw.odds_history_fetches "
                                  "ALTER COLUMN status TYPE VARCHAR(10)"))
        oh.ensure_tables()
        oh.ensure_tables()                  # a second run changes nothing
        with engine.connect() as conn:
            width = conn.execute(text("""
                SELECT character_maximum_length FROM information_schema.columns
                WHERE table_schema = 'raw' AND table_name = 'odds_history_fetches'
                  AND column_name = 'status'""")).scalar()
        assert width == 16

    def test_reparse_loads_a_paid_unparsed_fetch_from_its_raw_copy(self, body, db_clean,
                                                                   tmp_path):
        moved = _db_body(body)
        requested = dt.datetime(2031, 10, 4, 16, 50)
        oh.save_raw(tmp_path, oh.raw_stem("dbtest", requested, "h2h,totals", BOOKS), moved)
        oh.store([], {"requested_ts": requested, "purpose": "dbtest", "season": DB_SEASON,
                      "markets": "h2h,totals", "bookmakers": BOOKS, "snapshot_ts": None,
                      "next_ts": None, "credits": 20, "n_events": 0, "n_rows": 0,
                      "status": "paid_unparsed"})
        out = oh.reparse_unparsed(tmp_path)
        assert out["loaded"] >= 1
        with engine.connect() as conn:
            log = conn.execute(text("SELECT * FROM raw.odds_history_fetches "
                                    "WHERE purpose = 'dbtest'")).mappings().one()
            n = conn.execute(text("SELECT COUNT(*) FROM raw.odds_history "
                                  "WHERE event_id LIKE 'dbtest-%'")).scalar()
        assert (log["status"], log["n_rows"], log["credits"]) == ("ok", 36, 20)
        assert log["snapshot_ts"] == dt.datetime(2031, 10, 4, 16, 45, 39)
        assert n == 36
        # nothing left to load: a second pass changes nothing
        oh.reparse_unparsed(tmp_path)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM raw.odds_history "
                                     "WHERE event_id LIKE 'dbtest-%'")).scalar() == 36

    def _row(self, side, price, event="dbtest-e1"):
        return {"snapshot_ts": dt.datetime(2031, 10, 4, 16, 45, 39),
                "requested_ts": dt.datetime(2031, 10, 4, 16, 50), "event_id": event,
                "game_id": None, "commence_time": dt.datetime(2031, 10, 4, 17, 10),
                "home_name": "Buffalo Sabres", "away_name": "New Jersey Devils",
                "book": "pinnacle", "market": "h2h", "side": side, "price": price,
                "point": None, "book_updated_at": None}

    def _fetch(self):
        return {"requested_ts": dt.datetime(2031, 10, 4, 16, 50), "purpose": "dbtest",
                "season": DB_SEASON, "markets": "h2h", "bookmakers": "pinnacle",
                "snapshot_ts": dt.datetime(2031, 10, 4, 16, 45, 39), "next_ts": None,
                "credits": 10, "n_events": 1, "n_rows": 2, "status": "ok"}

    def test_store_is_idempotent_on_rows(self, db_clean):
        rows = [self._row("home", 120), self._row("away", -140)]
        oh.store(rows, self._fetch())
        # the same snapshot again, one price changed: the first stored row wins
        oh.store([self._row("home", 999), self._row("away", -140)], self._fetch())
        with engine.connect() as conn:
            got = conn.execute(text("""
                SELECT side, price FROM raw.odds_history
                WHERE event_id = 'dbtest-e1' ORDER BY side""")).all()
            logged = conn.execute(text("SELECT COUNT(*) FROM raw.odds_history_fetches "
                                       "WHERE purpose = 'dbtest'")).scalar()
        assert [tuple(r) for r in got] == [("away", -140), ("home", 120)]
        assert logged == 2                  # every call is logged, even a repeat

    def test_store_rolls_back_rows_when_the_log_fails(self, db_clean):
        bad = dict(self._fetch(), status=None)          # NOT NULL: the log insert fails
        with pytest.raises(Exception):
            oh.store([self._row("home", 120)], bad)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM raw.odds_history "
                                     "WHERE event_id = 'dbtest-e1'")).scalar() == 0

    def test_fill_start_times_writes_only_null_starts_in_the_season(self, db_clean):
        other = DB_SEASON + 10001
        set_before = dt.datetime(2030, 10, 12, 0, 0, tzinfo=UTC)
        with engine.begin() as conn:
            for gid, season, day, start in (
                    (9999030001, DB_SEASON, "2030-10-10", None),
                    (9999030002, DB_SEASON, "2030-10-20", None),
                    (9999030003, DB_SEASON, "2030-10-11", set_before),
                    (9999030004, other, "2031-10-10", None)):
                conn.execute(text("""
                    INSERT INTO raw.games (game_id, season, game_type, date, start_time_utc,
                                           home_team, away_team)
                    VALUES (:g, :s, 2, :d, :t, 'BUF', 'MTL')
                """), {"g": gid, "s": season, "d": day, "t": start})
        asked = []

        def week(d):
            asked.append(d)
            return {"gameWeek": [{"games": [
                {"id": gid, "startTimeUTC": "2030-10-15T23:30:00Z"}
                for gid in (9999030001, 9999030002, 9999030003, 9999030004, 9999030099)]}]}

        assert oh.fill_start_times(DB_SEASON, fetch_week=week, pause=0) == 2
        assert asked == ["2030-10-10", "2030-10-17"]
        with engine.connect() as conn:
            got = dict(conn.execute(text("""
                SELECT game_id, start_time_utc FROM raw.games
                WHERE game_id BETWEEN 9999030001 AND 9999030004""")).all())
        filled = dt.datetime(2030, 10, 15, 23, 30, tzinfo=UTC)
        assert got[9999030001] == filled and got[9999030002] == filled
        assert got[9999030003] == set_before            # already set: untouched
        assert got[9999030004] is None                  # another season: untouched
        # nothing left to fill: no schedule call at all
        assert oh.fill_start_times(DB_SEASON, fetch_week=week, pause=0) == 0
        assert len(asked) == 2

    def test_rematch_links_rows_once_the_game_exists(self, db_clean):
        oh.store([self._row("home", 120), self._row("away", -140)], self._fetch())
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, start_time_utc,
                                       home_team, away_team)
                VALUES (9999030005, :s, 2, '2031-10-04', '2031-10-04 17:00+00', 'BUF', 'NJD')
            """), {"s": DB_SEASON + 10001})
        assert oh.rematch_unmatched() >= 2
        with engine.connect() as conn:
            ids = {r[0] for r in conn.execute(text(
                "SELECT game_id FROM raw.odds_history WHERE event_id = 'dbtest-e1'"))}
        assert ids == {9999030005}
