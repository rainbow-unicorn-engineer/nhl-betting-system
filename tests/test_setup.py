"""
tests/test_setup.py
Smoke tests to verify the system can import and connect.
Run: pytest tests/ -v
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_config_imports():
    from config.settings import DATABASE_URL, DATA_DIR, BACKFILL_SEASONS
    assert "postgresql" in DATABASE_URL
    assert len(BACKFILL_SEASONS) > 0


def test_database_url_names_psycopg2_driver():
    """SQLAlchemy 2.1 maps a bare postgresql:// URL to psycopg (v3), which
    the project does not install — the driver must be explicit."""
    from config.settings import DATABASE_URL, engine
    assert DATABASE_URL.startswith("postgresql+psycopg2://")
    assert engine.dialect.driver == "psycopg2"


def test_season_for_july_boundary():
    from config.settings import season_for
    assert season_for(dt.date(2026, 6, 30)) == 20252026
    assert season_for(dt.date(2026, 7, 1)) == 20262027
    assert season_for(dt.date(2026, 10, 7)) == 20262027
    assert season_for(dt.date(2026, 12, 31)) == 20262027
    assert season_for(dt.date(2027, 1, 1)) == 20262027
    assert season_for(dt.date(2027, 4, 15)) == 20262027


def test_current_season_follows_local_date():
    from config.settings import (BACKFILL_SEASONS, CURRENT_SEASON,
                                 local_today, season_for)
    if not os.getenv("NHL_SEASON"):
        assert CURRENT_SEASON == season_for(local_today())
    assert BACKFILL_SEASONS[-1] == CURRENT_SEASON
    starts = [s // 10000 for s in BACKFILL_SEASONS]
    assert starts == list(range(starts[0], starts[-1] + 1))
    assert all(s % 10000 == s // 10000 + 1 for s in BACKFILL_SEASONS)
    if not os.getenv("BACKFILL_FIRST_SEASON"):
        assert BACKFILL_SEASONS[0] == 20202021


def _settings_in_subprocess(env_overrides: dict, cwd=PROJECT_ROOT) -> dict:
    """Import config.settings fresh with the given env (module constants
    are computed at import time)."""
    code = ("import json; from config import settings as s; "
            "print(json.dumps({'season': s.CURRENT_SEASON, "
            "'backfill': s.BACKFILL_SEASONS, 'tz': s.local_tz_name(), "
            "'offset_h': s.local_now().utcoffset().total_seconds() / 3600, "
            "'today': str(s.local_today()), 'data_dir': str(s.DATA_DIR), "
            "'derived': s.season_for(s.local_today())}))")
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), **env_overrides}
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd,
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    got["stderr"] = out.stderr
    return got


def test_env_overrides_season_backfill_and_timezone():
    got = _settings_in_subprocess({"NHL_SEASON": "20242025",
                                   "BACKFILL_FIRST_SEASON": "20222023",
                                   "LOCAL_TIMEZONE": "America/Chicago"})
    assert got["season"] == 20242025
    assert got["backfill"] == [20222023, 20232024, 20242025]
    assert got["tz"] == "America/Chicago"
    assert got["offset_h"] in (-5.0, -6.0)      # CDT / CST
    from zoneinfo import ZoneInfo
    chicago_today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    assert got["today"] in (str(chicago_today),
                            str(chicago_today + dt.timedelta(days=1)))


def test_unknown_timezone_falls_back_to_machine_zone():
    got = _settings_in_subprocess({"LOCAL_TIMEZONE": "Mars/Olympus_Mons"})
    assert "this machine" in got["tz"]


def test_tzdata_folder_name_falls_back_instead_of_crashing():
    """"America" is a folder in the tz database: ZoneInfo raises
    IsADirectoryError (PermissionError on Windows), not a lookup error."""
    got = _settings_in_subprocess({"LOCAL_TIMEZONE": "America"})
    assert "this machine" in got["tz"]


def test_malformed_seasons_log_an_error_and_fall_back():
    got = _settings_in_subprocess({"NHL_SEASON": "2025",
                                   "BACKFILL_FIRST_SEASON": "20202022"})
    assert got["season"] == got["derived"]
    assert got["backfill"][0] == 20202021
    assert "NHL_SEASON='2025' is not a season" in got["stderr"]
    assert "BACKFILL_FIRST_SEASON='20202022' is not a season" in got["stderr"]
    got = _settings_in_subprocess({"NHL_SEASON": "2025-2026"})
    assert got["season"] == got["derived"]


def test_first_backfill_season_after_the_current_one_is_clamped():
    """An empty BACKFILL_SEASONS made backfill report success with nothing
    loaded: the list always holds at least the season in use."""
    got = _settings_in_subprocess({"NHL_SEASON": "20222023",
                                   "BACKFILL_FIRST_SEASON": "20242025"})
    assert got["season"] == 20222023
    assert got["backfill"] == [20222023]
    assert ("BACKFILL_FIRST_SEASON (20242025) is later than the season in "
            "use (20222023") in got["stderr"]
    # NHL_SEASON pinned before the default first season (2020-21)
    got = _settings_in_subprocess({"NHL_SEASON": "20192020",
                                   "BACKFILL_FIRST_SEASON": ""})
    assert got["backfill"] == [20192020]
    assert "BACKFILL_FIRST_SEASON (20202021) is later" in got["stderr"]
    # a first season in the future, with the season from today's date
    got = _settings_in_subprocess({"NHL_SEASON": "",
                                   "BACKFILL_FIRST_SEASON": "20992100"})
    assert got["backfill"] == [got["derived"]]


def test_first_backfill_season_equal_to_the_current_one_is_fine():
    got = _settings_in_subprocess({"NHL_SEASON": "20222023",
                                   "BACKFILL_FIRST_SEASON": "20222023"})
    assert got["backfill"] == [20222023]
    assert "BACKFILL_FIRST_SEASON" not in got["stderr"]


def test_relative_data_dir_resolves_against_the_repo(tmp_path):
    """.env.example sets DATA_DIR=./data: that is the repo data folder
    from any starting folder, not a new folder wherever a job starts."""
    got = _settings_in_subprocess({"DATA_DIR": "./data"}, cwd=tmp_path)
    assert Path(got["data_dir"]) == PROJECT_ROOT / "data"
    assert not (tmp_path / "data").exists()


def test_ingestion_imports():
    from ingestion import nhl_api
    from ingestion import moneypuck
    from ingestion import odds_api
    assert hasattr(nhl_api, "daily_refresh")
    assert hasattr(moneypuck, "load_shots_to_db")
    assert hasattr(odds_api, "snapshot_odds")


def test_nhl_client_creates():
    """CRITICAL: nhl-api-py imports as `nhlpy`, not `nhl_api_py`."""
    from nhlpy import NHLClient
    client = NHLClient()
    assert client is not None


def test_toi_conversion():
    from ingestion.nhl_api import _toi_to_seconds
    assert _toi_to_seconds("20:00") == 1200
    assert _toi_to_seconds("0:45") == 45
    assert _toi_to_seconds("65:30") == 3930
    assert _toi_to_seconds("--:--") == 0
    assert _toi_to_seconds("") == 0
    assert _toi_to_seconds(None) == 0


def test_team_name_mapping():
    from ingestion.odds_api import _TEAM_NAME_TO_ABBREV
    unique_abbrevs = set(_TEAM_NAME_TO_ABBREV.values())
    assert len(unique_abbrevs) >= 30, f"Only {len(unique_abbrevs)} unique team abbreviations mapped"
