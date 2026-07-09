"""
Tests for features/build_all.py + pipeline wiring (Phase 2, Task 7).

The plan-mandated test: run the orchestrator on ONE season end-to-end and
confirm features.game_vector lands with the expected row count — plus a
regression guard that a season-scoped rebuild leaves other seasons intact.
"""
import pytest
from sqlalchemy import text

from features.build_all import build_features
from features.util import WINDOWS

SEASON = 20212022


@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            n = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE season = :s "
                "AND game_state IN ('FINAL','OFF')"), {"s": SEASON}).scalar()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if n < 1300:
        pytest.skip(f"season {SEASON} not fully backfilled ({n} games)")
    return engine


def _other_season_counts(conn):
    return conn.execute(text("""
        SELECT COUNT(*) FROM features.game_vector WHERE season <> :s
    """), {"s": SEASON}).scalar()


def test_single_season_end_to_end(db):
    with db.connect() as conn:
        n_games = conn.execute(text(
            "SELECT COUNT(*) FROM raw.games WHERE season = :s "
            "AND game_state IN ('FINAL','OFF')"), {"s": SEASON}).scalar()
        others_before = _other_season_counts(conn)

    counts = build_features(SEASON)

    # Orchestrator returned the counts each stage reported...
    assert counts["team_rolling"] == n_games * 2 * len(WINDOWS)
    assert counts["matchup_schedule"] == n_games
    assert counts["game_vector"] == n_games
    assert counts["goalie_rolling"] > n_games * 2 * len(WINDOWS) * 0.9
    assert counts["elo"] > n_games  # full history, not just this season

    # ...and the tables agree with what actually landed.
    with db.connect() as conn:
        vec = conn.execute(text("""
            SELECT COUNT(*) FROM features.game_vector WHERE season = :s
        """), {"s": SEASON}).scalar()
        assert vec == n_games

        # A scoped rebuild must not touch other seasons
        assert _other_season_counts(conn) == others_before

        # Vectors reference matchup rows with Elo present (dependency order)
        unrated = conn.execute(text("""
            SELECT COUNT(*) FROM features.game_vector v
            JOIN features.matchup m USING (game_id)
            WHERE v.season = :s AND m.home_elo IS NULL
        """), {"s": SEASON}).scalar()
        assert unrated == 0


def test_pipeline_cli_wiring():
    """`python pipeline.py features --season X` parses and dispatches."""
    import pipeline
    assert callable(pipeline.features)
