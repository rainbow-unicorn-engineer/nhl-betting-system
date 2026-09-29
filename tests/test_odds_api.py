"""
Tests for ingestion/odds_api.py — event matching (pure), the in-play skip,
and the guarantee that the API key never reaches the logs. No network:
requests.get is mocked; no database: snapshot_odds runs on a fake engine.
"""
import datetime as dt
import json
import logging
from contextlib import contextmanager

import pytest
import requests

from ingestion import odds_api
from ingestion.odds_api import _redact, match_event, parse_commence

UTC = dt.timezone.utc
FAKE_KEY = "fakekey0123456789abcdef0123456789"


def event(home="Boston Bruins", away="Ottawa Senators",
          commence="2026-01-16T00:00:00Z", bookmakers=None):
    return {"home_team": home, "away_team": away, "commence_time": commence,
            "bookmakers": bookmakers or []}


def game(game_id, date, start=None, home="BOS"):
    return {"game_id": game_id, "home_team": home, "date": date,
            "start_time_utc": start}


class TestMatchEvent:
    NOON_ET = dt.datetime(2026, 1, 15, 17, 0, tzinfo=UTC)

    def test_evening_game_matches_across_utc_midnight(self):
        # 7pm ET on Jan 15 is 00:00 UTC on Jan 16; raw.games.date is Jan 15.
        # The old UTC-date match looked for a Jan 16 game and missed it.
        games = [game(1, dt.date(2026, 1, 15),
                      dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC))]
        assert match_event(event(), games, self.NOON_ET) == (1, "ok")

    def test_start_time_compared_as_instants_across_offsets(self):
        # start_time_utc read back from Postgres may carry a non-UTC offset
        est = dt.timezone(dt.timedelta(hours=-5))
        games = [game(1, dt.date(2026, 1, 15),
                      dt.datetime(2026, 1, 15, 19, 0, tzinfo=est))]
        assert match_event(event(), games, self.NOON_ET) == (1, "ok")

    def test_date_fallback_uses_eastern_date_when_start_unknown(self):
        jan15 = [game(1, dt.date(2026, 1, 15))]
        assert match_event(event(), jan15, self.NOON_ET) == (1, "ok")
        jan16 = [game(2, dt.date(2026, 1, 16))]   # the UTC date: wrong day
        assert match_event(event(), jan16, self.NOON_ET) == (None, "no_game")

    def test_nearest_start_within_six_hours_wins(self):
        games = [game(1, dt.date(2026, 1, 15),
                      dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC)),
                 game(2, dt.date(2026, 1, 16),
                      dt.datetime(2026, 1, 17, 0, 0, tzinfo=UTC))]
        assert match_event(event(commence="2026-01-16T01:00:00Z"), games,
                           self.NOON_ET) == (1, "ok")
        # 7.5h from the nearest known start: no match, and rows WITH a
        # start time never fall back to the date rule
        assert match_event(event(commence="2026-01-16T07:30:00Z"), games,
                           self.NOON_ET) == (None, "no_game")

    def test_in_play_event_is_skipped(self):
        games = [game(1, dt.date(2026, 1, 15),
                      dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC))]
        after_drop = dt.datetime(2026, 1, 16, 0, 30, tzinfo=UTC)
        assert match_event(event(), games, after_drop) == (None, "started")
        at_drop = dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
        assert match_event(event(), games, at_drop) == (None, "started")

    def test_unknown_team_and_bad_time(self):
        assert match_event(event(home="Quebec Nordiques"), [],
                           self.NOON_ET) == (None, "unknown_team")
        assert match_event(event(commence="not a time"), [],
                           self.NOON_ET) == (None, "no_time")
        assert match_event(event(commence=""), [],
                           self.NOON_ET) == (None, "no_time")

    def test_parse_commence_is_aware_utc(self):
        t = parse_commence("2026-01-16T00:00:00Z")
        assert t == dt.datetime(2026, 1, 16, tzinfo=UTC)
        assert t.utcoffset() == dt.timedelta(0)


# ── snapshot_odds on a fake engine ─────────────────────────────────

class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _FakeConn:
    def __init__(self, games):
        self.games, self.inserts = games, []

    def execute(self, stmt, params=None):
        if "FROM raw.games" in str(stmt):
            return _Rows(self.games)
        self.inserts.append(params)


class _FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def begin(self):
        yield self.conn


def _h2h(book, home, away, home_price, away_price):
    return {"key": book, "markets": [{"key": "h2h", "outcomes": [
        {"name": home, "price": home_price},
        {"name": away, "price": away_price}]}]}


class TestSnapshotOdds:
    def test_in_play_prices_never_stored(self, monkeypatch):
        now = dt.datetime.now(UTC).replace(microsecond=0)
        upcoming = now + dt.timedelta(hours=2)
        started = now - dt.timedelta(minutes=30)
        games = [game(1, upcoming.date(), upcoming, home="BOS"),
                 game(2, started.date(), started, home="TOR")]
        conn = _FakeConn(games)
        monkeypatch.setattr(odds_api, "engine", _FakeEngine(conn))
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)

        events = [
            event("Boston Bruins", "Ottawa Senators",
                  upcoming.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  [_h2h("draftkings", "Boston Bruins", "Ottawa Senators", -120, 100)]),
            event("Toronto Maple Leafs", "Montreal Canadiens",
                  started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  [_h2h("draftkings", "Toronto Maple Leafs",
                        "Montreal Canadiens", -900, 550)]),
        ]
        assert odds_api.snapshot_odds(game_odds=events) == 1
        (row,) = conn.inserts
        assert row["game_id"] == 1
        assert (row["home_price"], row["away_price"]) == (-120, 100)
        assert row["captured_at"].tzinfo is None          # naive UTC, as stored

    def test_unmatched_events_warn_with_count(self, monkeypatch, caplog):
        conn = _FakeConn([])
        monkeypatch.setattr(odds_api, "engine", _FakeEngine(conn))
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)
        later = (dt.datetime.now(UTC) + dt.timedelta(days=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        with caplog.at_level(logging.WARNING, logger="nhl.ingestion.odds_api"):
            assert odds_api.snapshot_odds(game_odds=[event(commence=later)]) == 0
        assert "1 Odds API event(s) matched no raw.games row" in caplog.text

    def test_markets_passed_through_to_fetch(self, monkeypatch):
        seen = []
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)
        monkeypatch.setattr(odds_api, "upcoming_start_times",
                            lambda now, horizon: [now + dt.timedelta(hours=3)])
        monkeypatch.setattr(odds_api, "fetch_current_odds",
                            lambda markets: seen.append(markets) or [])
        assert odds_api.snapshot_odds(markets="h2h") == 0
        assert seen == ["h2h"]

    def test_no_game_within_24h_means_no_request(self, monkeypatch, caplog):
        """In season the API also lists later days, so a request on a day
        with no game still costs credits: skip it."""
        horizons = []
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)
        monkeypatch.setattr(odds_api, "upcoming_start_times",
                            lambda now, horizon: horizons.append(horizon) or [])
        monkeypatch.setattr(odds_api, "fetch_current_odds",
                            lambda markets: pytest.fail("request made"))
        with caplog.at_level(logging.INFO, logger="nhl.ingestion.odds_api"):
            assert odds_api.snapshot_odds() == 0
        assert horizons == [dt.timedelta(hours=24)]
        assert "skipping the Odds API request" in caplog.text

    def test_given_events_need_no_schedule_check(self, monkeypatch):
        conn = _FakeConn([])
        monkeypatch.setattr(odds_api, "engine", _FakeEngine(conn))
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)
        monkeypatch.setattr(odds_api, "upcoming_start_times",
                            lambda now, horizon: pytest.fail("not needed"))
        assert odds_api.snapshot_odds(game_odds=[]) == 0


# ── close --due timing ─────────────────────────────────────────────

class TestCloseDue:
    """close_due's rule with the lead and gap passed in (40 and 25 here),
    so these tests hold whatever the defaults or .env say."""
    NOW = dt.datetime(2026, 1, 15, 23, 40, tzinfo=UTC)   # 6:40 pm ET
    LEAD, GAP = dt.timedelta(minutes=40), dt.timedelta(minutes=25)

    def due(self, starts, last, now=None):
        return odds_api.close_due(starts, last, now or self.NOW,
                                  self.LEAD, self.GAP)

    def test_game_within_the_lead_is_due(self):
        due, why = self.due([self.NOW + dt.timedelta(minutes=20)], None)
        assert due and "20 minute" in why

    def test_start_just_after_utc_midnight(self):
        # 7:05 pm ET on Jan 15 is 00:05 UTC on Jan 16: a different UTC date
        start = dt.datetime(2026, 1, 16, 0, 5, tzinfo=UTC)
        assert self.due([start], None)[0]
        # and from the other side of midnight
        now = dt.datetime(2026, 1, 16, 0, 1, tzinfo=UTC)
        assert self.due([dt.datetime(2026, 1, 16, 0, 30, tzinfo=UTC)],
                        None, now)[0]

    def test_nothing_starting_soon_is_not_due(self):
        later = self.NOW + dt.timedelta(minutes=41)
        started = self.NOW - dt.timedelta(minutes=5)
        assert self.due([], None) == (
            False, "no game starts in the next 40 minutes")
        assert not self.due([later, started, self.NOW], None)[0]

    def test_recent_ml_snapshot_is_not_due(self):
        soon = [self.NOW + dt.timedelta(minutes=30)]
        recent = self.NOW - dt.timedelta(minutes=15)
        due, why = self.due(soon, recent)
        assert not due and "15 minute(s) ago" in why
        older = self.NOW - dt.timedelta(minutes=25)
        assert self.due(soon, older)[0]

    def test_close_is_due_reads_the_database_as_instants(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(odds_api, "ensure_schema", lambda: None)

        def starts(now, horizon):
            seen["window"] = (now, horizon)
            return [now + dt.timedelta(minutes=10)]
        monkeypatch.setattr(odds_api, "upcoming_start_times", starts)
        monkeypatch.setattr(odds_api, "last_ml_capture", lambda: None)
        assert odds_api.close_is_due(self.NOW)[0]
        assert seen["window"] == (self.NOW, odds_api.CLOSE_LEAD)


def _cycle(starts, offset_min, delays_s=(0,), lead=dt.timedelta(minutes=16),
           gap=dt.timedelta(minutes=16)):
    """Replays `close --due` every 15 minutes over one evening, the cycle
    starting offset_min minutes past the hour; each run starts late by
    the next of delays_s seconds in turn (a scheduler that fires a little
    late). Returns the instants a close was taken."""
    t = dt.datetime(2026, 1, 15, 22, offset_min, tzinfo=UTC)
    last, closes, i = None, [], 0
    while t < dt.datetime(2026, 1, 16, 1, 30, tzinfo=UTC):
        now = t + dt.timedelta(seconds=delays_s[i % len(delays_s)])
        if odds_api.close_due(starts, last, now, lead, gap)[0]:
            closes.append(now)
            last = now
        t += dt.timedelta(minutes=15)
        i += 1
    return closes


class TestCloseDefaults:
    """16/16 on the 15-minute cycle: one close per start time, taken in
    the last 16 minutes before puck drop."""
    START = dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC)   # 7:00 pm ET

    @pytest.mark.parametrize("delays_s", [(0,), (0, 59), (59, 0), (30,)])
    def test_a_game_gets_exactly_one_close_in_its_last_16_minutes(self, delays_s):
        for offset in range(15):          # whatever minute the cycle starts on
            closes = _cycle([self.START], offset, delays_s)
            assert len(closes) == 1, (offset, closes)
            before = self.START - closes[0]
            assert dt.timedelta(0) < before <= dt.timedelta(minutes=16), (
                offset, before)

    def test_start_times_30_minutes_apart_get_one_close_each(self):
        later = self.START + dt.timedelta(minutes=30)
        for offset in range(15):
            closes = _cycle([self.START, later], offset, (0, 59))
            assert len(closes) == 2, (offset, closes)
            for start in (self.START, later):
                assert sum(dt.timedelta(0) < start - c <= dt.timedelta(minutes=16)
                           for c in closes) == 1, (offset, start, closes)

    def test_the_old_40_25_took_more_than_one_close_per_start(self):
        counts = {len(_cycle([self.START], offset,
                             lead=dt.timedelta(minutes=40),
                             gap=dt.timedelta(minutes=25)))
                  for offset in range(15)}
        assert max(counts) >= 2

    def test_close_due_defaults_are_the_module_settings(self):
        import inspect
        params = inspect.signature(odds_api.close_due).parameters
        assert params["lead"].default == odds_api.CLOSE_LEAD
        assert params["min_gap"].default == odds_api.CLOSE_MIN_GAP


def _close_settings(env: dict) -> dict:
    """CLOSE_LEAD and CLOSE_MIN_GAP (in minutes) and stderr from a fresh
    import of ingestion.odds_api with the given variables set (the module
    reads them at import). Both start out empty, which counts as unset,
    so .env (which never overrides a variable already set) can't leak in."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    code = ("import json; from ingestion import odds_api as o; "
            "print(json.dumps([o.CLOSE_LEAD.total_seconds() / 60, "
            "o.CLOSE_MIN_GAP.total_seconds() / 60]))")
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True,
        text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(root),
             "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "1",
             "CLOSE_LEAD_MINUTES": "", "CLOSE_MIN_GAP_MINUTES": "", **env})
    assert out.returncode == 0, out.stderr[-2000:]
    lead, gap = json.loads(out.stdout.strip().splitlines()[-1])
    return {"lead": lead, "gap": gap, "stderr": out.stderr}


def test_close_defaults_are_16_and_16():
    got = _close_settings({})
    assert (got["lead"], got["gap"]) == (16, 16)
    assert "CLOSE_" not in got["stderr"]


def test_close_settings_come_from_the_environment():
    got = _close_settings({"CLOSE_LEAD_MINUTES": " 30 ",
                           "CLOSE_MIN_GAP_MINUTES": "0"})
    assert (got["lead"], got["gap"]) == (30, 0)


def test_malformed_close_settings_log_an_error_and_fall_back():
    """A typo must not crash the import: that would abort every command
    that imports the module, the whole daily chain included."""
    got = _close_settings({"CLOSE_LEAD_MINUTES": "4O",
                           "CLOSE_MIN_GAP_MINUTES": "-5"})
    assert (got["lead"], got["gap"]) == (16, 16)
    assert "CLOSE_LEAD_MINUTES='4O' is not a number of minutes" in got["stderr"]
    assert "CLOSE_MIN_GAP_MINUTES='-5' is not a number of minutes" in got["stderr"]
    for lead in ("0", "nan", "inf", "1e400", "16 min"):
        assert _close_settings({"CLOSE_LEAD_MINUTES": lead})["lead"] == 16, lead


# ── The command line never spends credits on --help ────────────────

def test_help_exits_without_a_snapshot(monkeypatch, capsys):
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("--help took a snapshot"))
    with pytest.raises(SystemExit) as exc:
        odds_api.main(["--help"])
    assert exc.value.code == 0
    assert "--markets" in capsys.readouterr().out


def test_markets_option_reaches_the_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    odds_api.main(["--markets", "h2h"])
    odds_api.main([])
    assert calls == [{"markets": "h2h"}, {"markets": "h2h,spreads,totals"}]


def test_module_help_runs_nothing():
    """python -m ingestion.odds_api --help: usage and exit 0, before any
    database or network call (both are unreachable here)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    out = subprocess.run(
        [sys.executable, "-m", "ingestion.odds_api", "--help"],
        cwd=root, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(root), "ODDS_API_KEY": "your_key_here",
             "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "1"})
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout and "--markets" in out.stdout


# ── The API key never reaches the logs ─────────────────────────────

URL_WITH_KEY = (f"{odds_api.BASE_URL}/sports/{odds_api.SPORT}/odds"
                f"?apiKey={FAKE_KEY}&markets=h2h&regions=us%2Cus2")


def _response(status, reason, body, headers=None):
    r = requests.Response()
    r.status_code, r.reason, r.url = status, reason, URL_WITH_KEY
    r._content = body.encode() if isinstance(body, str) else body
    r.headers.update(headers or {})
    return r


def _logged(caplog) -> str:
    return caplog.text + "\n".join(r.getMessage() for r in caplog.records)


@pytest.fixture()
def fake_key(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", FAKE_KEY)


class TestRedaction:
    def test_redact_replaces_key_and_query_value(self):
        s = f"401 Client Error: Unauthorized for url: {URL_WITH_KEY}"
        out = _redact(s, key=FAKE_KEY)
        assert FAKE_KEY not in out
        assert "apiKey=***&markets=h2h" in out
        # any apiKey= value is scrubbed, even one that isn't the configured key
        assert _redact("?apiKey=someotherkey&x=1", key="") == "?apiKey=***&x=1"
        assert _redact(f"the key is {FAKE_KEY}.", key=FAKE_KEY) == "the key is ***."

    def test_401_logs_status_and_api_message_never_the_key(
            self, monkeypatch, caplog, fake_key):
        calls = []
        body = json.dumps({"message": f"API key {FAKE_KEY} is not valid",
                           "error_code": "INVALID_KEY"})

        def fake_get(url, params=None, timeout=None):
            calls.append(params)
            return _response(401, "Unauthorized", body)

        monkeypatch.setattr(odds_api.requests, "get", fake_get)
        with caplog.at_level(logging.DEBUG):
            assert odds_api.fetch_current_odds() == []
        assert calls and calls[0]["apiKey"] == FAKE_KEY    # it was really sent
        text = _logged(caplog)
        assert FAKE_KEY not in text
        assert "Odds API request failed: HTTP 401 Unauthorized: API key *** is not valid" in text

    def test_non_json_body_redacted_before_truncation(
            self, monkeypatch, caplog, fake_key):
        # the key straddles the 200-char cut: redacting after truncating
        # would leak its first half
        body = "<html>" + "x" * 180 + FAKE_KEY + "</html>"
        monkeypatch.setattr(odds_api.requests, "get",
                            lambda *a, **k: _response(502, "Bad Gateway", body))
        with caplog.at_level(logging.DEBUG):
            assert odds_api.fetch_current_odds() == []
        text = _logged(caplog)
        assert FAKE_KEY[:10] not in text
        assert "HTTP 502 Bad Gateway: <html>xxx" in text

    def test_connection_error_names_host_and_class_only(
            self, monkeypatch, caplog, fake_key):
        def boom(*a, **k):
            raise requests.ConnectionError(
                "HTTPSConnectionPool(host='api.the-odds-api.com', port=443): "
                f"Max retries exceeded with url: /v4/sports/icehockey_nhl/odds"
                f"?apiKey={FAKE_KEY}&markets=h2h")

        monkeypatch.setattr(odds_api.requests, "get", boom)
        with caplog.at_level(logging.DEBUG):
            assert odds_api.fetch_current_odds() == []
        text = _logged(caplog)
        assert FAKE_KEY not in text
        assert ("Odds API request failed: could not reach "
                "api.the-odds-api.com (ConnectionError)") in text

    def test_timeouts(self, monkeypatch, caplog, fake_key):
        for exc in (requests.ReadTimeout, requests.ConnectTimeout):
            def boom(*a, _exc=exc, **k):
                raise _exc(f"timed out: {URL_WITH_KEY}")
            monkeypatch.setattr(odds_api.requests, "get", boom)
            with caplog.at_level(logging.DEBUG):
                assert odds_api.fetch_current_odds() == []
        text = _logged(caplog)
        assert FAKE_KEY not in text
        assert text.count("Odds API request failed: timed out after 30s") >= 2

    def test_unexpected_error_is_redacted(self, monkeypatch, caplog, fake_key):
        def boom(*a, **k):
            raise ValueError(f"weird failure for {URL_WITH_KEY}")

        monkeypatch.setattr(odds_api.requests, "get", boom)
        with caplog.at_level(logging.DEBUG):
            assert odds_api.fetch_current_odds() == []
        text = _logged(caplog)
        assert FAKE_KEY not in text
        assert "ValueError: weird failure" in text and "apiKey=***" in text

    def test_placeholder_key_is_unset_and_no_request_made(self, monkeypatch, caplog):
        monkeypatch.setattr(odds_api, "ODDS_API_KEY", "your_key_here")
        monkeypatch.setattr(odds_api.requests, "get",
                            lambda *a, **k: pytest.fail("request made"))
        with caplog.at_level(logging.ERROR):
            assert odds_api.fetch_current_odds() == []
        assert "ODDS_API_KEY is not set" in caplog.text

    def test_success_logs_credit_usage(self, monkeypatch, caplog, fake_key):
        headers = {"x-requests-remaining": "494", "x-requests-used": "6",
                   "x-requests-last": "2"}
        monkeypatch.setattr(odds_api.requests, "get",
                            lambda *a, **k: _response(200, "OK", "[]", headers))
        with caplog.at_level(logging.INFO):
            assert odds_api.fetch_current_odds(markets="h2h") == []
        assert "Credits remaining 494, used 6, this call cost 2" in caplog.text
        assert FAKE_KEY not in _logged(caplog)

    def test_urllib3_debug_url_is_scrubbed(self, caplog, fake_key):
        log = logging.getLogger("urllib3.connectionpool")
        with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
            log.debug('%s://%s:%s "%s %s %s" %s %s', "https",
                      "api.the-odds-api.com", 443, "GET",
                      URL_WITH_KEY.split(".com")[1], "HTTP/1.1", 200, None)
        assert caplog.records
        assert FAKE_KEY not in _logged(caplog)
        assert "apiKey=***" in caplog.text
