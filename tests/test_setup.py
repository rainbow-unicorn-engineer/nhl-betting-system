"""
tests/test_setup.py
Smoke tests to verify the system can import and connect.
Run: pytest tests/ -v
"""


def test_config_imports():
    from config.settings import DATABASE_URL, DATA_DIR, BACKFILL_SEASONS
    assert "postgresql" in DATABASE_URL
    assert len(BACKFILL_SEASONS) > 0


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
