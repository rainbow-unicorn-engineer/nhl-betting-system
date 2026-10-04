"""
Tests for ingestion/odds_history.py: the purchase plan (start-time
clusters, morning snapshots, spread order), the never-buy-twice check,
parsing a recorded historical response (tests/fixtures/
odds_api_historical_nhl.json, a trimmed real 2024-10-04 snapshot), the
in-play drop, the credit caps, and key redaction. No network (the getter
is a stub) and no database (the saver is a list).
"""
import datetime as dt
import json
import logging
from pathlib import Path

import pytest

from ingestion import odds_history as oh

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

    def test_limit_and_raw_copy(self, body, tmp_path):
        report, _ = self.run(_plan(3), _Getter(body), oh.Budget(max_credits=1000),
                             limit=1, raw_dir=tmp_path)
        assert report.calls == 1 and "--limit" in report.stopped
        (f,) = list(tmp_path.iterdir())
        assert f.name == "close_2024-10-04T165000Z.json.gz"

    def test_describe_plan_counts_what_is_left(self):
        plan = _plan(3)
        done = [{"requested_ts": oh._naive(plan[0].requested_ts),
                 "snapshot_ts": None, "next_ts": None}]
        lines = oh.describe_plan(plan, done, 20)
        assert "1 already bought; to buy 2 x 20 = 40 credits" in lines[0]
        assert lines[-1] == "  total still to buy: 40 credits"


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


def test_fetch_requires_a_credit_cap(monkeypatch, capsys):
    monkeypatch.setattr(oh, "ensure_tables", lambda: pytest.fail("ran without a cap"))
    with pytest.raises(SystemExit):
        oh.main(["fetch"])
