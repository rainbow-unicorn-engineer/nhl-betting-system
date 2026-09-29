"""
Tests for ingestion/nhl_api.py schedule upserts: start_time_utc parsing
(pure), the season filter and schedule state (fake NHL client, no
network), and the upsert's conflict rules (DB; skipped without one).
"""
import datetime as dt
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion.nhl_api import _parse_start_time

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")

UTC = dt.timezone.utc


def test_parse_start_time():
    assert _parse_start_time("2026-01-16T00:00:00Z") == dt.datetime(
        2026, 1, 16, tzinfo=UTC)
    assert _parse_start_time("2026-01-16T00:00:00Z").tzinfo is not None
    assert _parse_start_time(None) is None
    assert _parse_start_time("") is None
    assert _parse_start_time("TBD") is None


def _season_window(monkeypatch, season):
    import ingestion.nhl_api as nhl
    calls = []
    monkeypatch.setattr(nhl, "ingest_schedule",
                        lambda start, end, season=None:
                        calls.append((start, end, season)))
    monkeypatch.setattr(nhl, "backfill_boxscores", lambda **kw: None)
    monkeypatch.setattr(nhl, "backfill_team_stats", lambda **kw: None)
    nhl.ingest_season(season)
    return calls[0]


def test_ingest_season_window_covers_september_openers(monkeypatch):
    """2026-27 opened Sept 29; an Oct 1 window silently dropped the opener."""
    start, end, season = _season_window(monkeypatch, 20262027)
    assert start == "2026-09-01"
    assert end == "2027-09-30"          # every window ends Sept 30
    assert season == 20262027           # other seasons' games are dropped


def test_season_windows_reach_the_late_covid_playoffs(monkeypatch):
    """The 2019-20 playoffs ran to Sept 28, 2020 (bubble) and the 2020-21
    Final to July 7, 2021: both inside their own season's window."""
    start, end, season = _season_window(monkeypatch, 20192020)
    assert (start, end, season) == ("2019-09-01", "2020-09-30", 20192020)
    assert end >= "2020-09-28"
    assert _season_window(monkeypatch, 20202021)[1] >= "2021-07-07"


def _sched_game(game_id, game_type, state="OFF", schedule_state="OK"):
    return {"id": game_id, "gameType": game_type, "gameState": state,
            "gameScheduleState": schedule_state,
            "startTimeUTC": "2020-09-06T00:00:00Z",
            "homeTeam": {"abbrev": "TBL", "score": 2},
            "awayTeam": {"abbrev": "NYI", "score": 1},
            "venue": {"default": "Rogers Place"},
            "gameOutcome": {"lastPeriodType": "OT"}}


@pytest.fixture()
def fake_schedule(monkeypatch):
    """ingestion.nhl_api with a fake nhlpy client: the weekly call whose
    7 days hold Sept 5, 2020 returns a 2019-20 bubble playoff game (id
    2019030215), a 2020-21 game and a postponed 2020-21 game; every other
    week is empty. Returns the list of records _upsert_game received."""
    import ingestion.nhl_api as nhl
    game_day = dt.date(2020, 9, 5)
    week = {"gameWeek": [{"date": str(game_day), "games": [
        _sched_game(2019030215, 3),
        _sched_game(2020020001, 2),
        _sched_game(2020020002, 2, state="FUT", schedule_state="PPD"),
    ]}]}

    def weekly_schedule(date):
        first = dt.date.fromisoformat(date)
        if first <= game_day < first + dt.timedelta(days=7):
            return week
        return {"gameWeek": []}

    upserted = []
    monkeypatch.setattr(nhl, "client", SimpleNamespace(
        schedule=SimpleNamespace(weekly_schedule=weekly_schedule)))
    monkeypatch.setattr(nhl, "ensure_schema", lambda: None)
    monkeypatch.setattr(nhl, "_upsert_game", upserted.append)
    monkeypatch.setattr(nhl.time, "sleep", lambda s: None)
    monkeypatch.setattr(nhl, "backfill_boxscores", lambda **kw: None)
    monkeypatch.setattr(nhl, "backfill_team_stats", lambda **kw: None)
    return nhl, upserted


def test_season_load_drops_prior_season_bubble_playoffs(fake_schedule):
    """ingest_season(20202021) starts Sept 1, 2020, when the 2019-20
    playoffs were still being played: those games must not be stored as
    2020-21 games."""
    nhl, upserted = fake_schedule
    nhl.ingest_season(20202021)
    ids = [r["game_id"] for r in upserted]
    assert 2019030215 not in ids
    assert ids == [2020020001, 2020020002]
    assert {r["season"] for r in upserted} == {20202021}


def test_2019_20_load_keeps_its_bubble_playoff_games(fake_schedule):
    """ingest_season(20192020) runs to Sept 30, 2020, so its September
    bubble playoff games are stored, as 2019-20 games; the 2020-21 games
    in the same week are left to the 2020-21 load."""
    nhl, upserted = fake_schedule
    nhl.ingest_season(20192020)
    assert [(r["game_id"], r["season"]) for r in upserted] == [
        (2019030215, 20192020)]


def test_daily_window_keeps_every_season(fake_schedule):
    """No season given (the daily refresh): every game in the window is
    kept, each with the season its id says."""
    nhl, upserted = fake_schedule
    nhl.ingest_schedule("2020-09-01", "2020-09-01")
    seasons = {r["game_id"]: r["season"] for r in upserted}
    assert seasons == {2019030215: 20192020, 2020020001: 20202021,
                       2020020002: 20202021}


def test_schedule_state_is_stored(fake_schedule):
    nhl, upserted = fake_schedule
    nhl.ingest_schedule("2020-09-01", "2020-09-01", season=20202021)
    states = {r["game_id"]: r["schedule_state"] for r in upserted}
    assert states == {2020020001: "OK", 2020020002: "PPD"}


@requires_db
class TestUpcomingStartTimes:
    """ingestion.odds_api.upcoming_start_times, which decides whether a
    snapshot or a close is due: postponed and cancelled games inside the
    window are left out."""
    IDS = (9_999_020_002, 9_999_020_003, 9_999_020_004)
    NOW = dt.datetime(2031, 1, 15, 17, 0, tzinfo=UTC)    # no real games then

    @pytest.fixture()
    def games(self):
        from config.migrate import ensure_schema
        from ingestion.nhl_api import _upsert_game
        ensure_schema()
        for game_id, hours, state in ((self.IDS[0], 2, "OK"),
                                      (self.IDS[1], 3, "PPD"),
                                      (self.IDS[2], 4, "CNCL")):
            _upsert_game({"game_id": game_id, "season": 20302031,
                          "game_type": 2, "date": "2031-01-15",
                          "start_time_utc": self.NOW + dt.timedelta(hours=hours),
                          "home_team": "BOS", "away_team": "OTT",
                          "home_score": None, "away_score": None,
                          "game_state": "FUT", "schedule_state": state,
                          "venue": "TD Garden", "is_ot": None, "is_so": None})
        yield
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.games WHERE game_id = ANY(:g)"),
                         {"g": list(self.IDS)})

    def test_postponed_game_in_the_window_is_left_out(self, games):
        from ingestion.odds_api import upcoming_start_times
        got = upcoming_start_times(self.NOW, dt.timedelta(hours=24))
        assert got == [self.NOW + dt.timedelta(hours=2)]
        # the postponed game is inside a window that ends on its start
        assert upcoming_start_times(
            self.NOW + dt.timedelta(hours=2, minutes=30),
            dt.timedelta(minutes=30)) == []


@requires_db
class TestUpsertGame:
    GAME_ID = 9_999_020_001          # synthetic, far outside real id ranges

    @pytest.fixture()
    def cleanup(self):
        from config.migrate import ensure_schema
        ensure_schema()
        yield
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"),
                         {"g": self.GAME_ID})

    def _record(self, date, start, schedule_state=None):
        return {"game_id": self.GAME_ID, "season": 20262027, "game_type": 2,
                "date": date, "start_time_utc": start, "home_team": "BOS",
                "away_team": "OTT", "home_score": None, "away_score": None,
                "game_state": "FUT", "schedule_state": schedule_state,
                "venue": "TD Garden", "is_ot": None, "is_so": None}

    def _row(self):
        with engine.connect() as conn:
            return conn.execute(text("""
                SELECT date, start_time_utc, schedule_state
                FROM raw.games WHERE game_id = :g
            """), {"g": self.GAME_ID}).one()

    def test_schedule_state_updates_and_is_never_erased(self, cleanup):
        from ingestion.nhl_api import _upsert_game
        start = dt.datetime(2026, 11, 3, 0, 0, tzinfo=UTC)
        _upsert_game(self._record("2026-11-02", start, "OK"))
        assert self._row().schedule_state == "OK"
        _upsert_game(self._record("2026-11-02", start, "PPD"))
        assert self._row().schedule_state == "PPD"     # postponed
        _upsert_game(self._record("2026-11-02", start, None))
        assert self._row().schedule_state == "PPD"     # payload without it
        _upsert_game(self._record("2026-12-08", start, "OK"))
        assert self._row().schedule_state == "OK"      # rescheduled

    def test_postponed_game_moves_date_and_keeps_known_start(self, cleanup):
        from ingestion.nhl_api import _upsert_game
        first = dt.datetime(2026, 11, 3, 0, 0, tzinfo=UTC)
        _upsert_game(self._record("2026-11-02", first))
        row = self._row()
        assert row.date == dt.date(2026, 11, 2)
        assert row.start_time_utc == first

        moved = dt.datetime(2026, 12, 9, 0, 30, tzinfo=UTC)
        _upsert_game(self._record("2026-12-08", moved))
        row = self._row()
        assert row.date == dt.date(2026, 12, 8)        # date follows the schedule
        assert row.start_time_utc == moved

        _upsert_game(self._record("2026-12-08", None))  # payload without a time
        assert self._row().start_time_utc == moved      # never erased
