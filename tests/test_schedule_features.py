"""
Tests for features/schedule_features.py (Phase 2, Task 4).

Unit tests drive compute_schedule_features() on a synthetic two-team schedule
(NY-geo vs LA-geo) covering the plan-mandated b2b case, rest days, travel
distance, DST-aware timezone shift, and season-opener semantics. Integration
tests verify real 2020-21 rows written to features.matchup.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from features.schedule_features import (
    build_schedule_features, compute_schedule_features, haversine_km,
)

NY = dict(lat=40.7505, lon=-73.9934, tz="America/New_York")
LA = dict(lat=34.0430, lon=-118.2673, tz="America/Los_Angeles")
NY_TO_LA_KM = 3936.0  # MSG -> Crypto.com Arena great-circle, ~±10 km


def make_schedule(rows):
    """Each row: (game_id, date, home_team, away_team, venue) with venue a
    dict like NY/LA above. Season fixed to 20202021."""
    return pd.DataFrame([
        {"game_id": gid, "season": 20202021, "date": date,
         "home_team": home, "away_team": away,
         "venue_lat": v["lat"], "venue_lon": v["lon"], "venue_timezone": v["tz"]}
        for gid, date, home, away, v in rows
    ])


def two_team_series():
    # NYA hosts twice (Jan 1-2), then travels to LAB (Jan 5-6)
    return make_schedule([
        (1, "2021-01-01", "NYA", "LAB", NY),
        (2, "2021-01-02", "NYA", "LAB", NY),
        (3, "2021-01-05", "LAB", "NYA", LA),
        (4, "2021-01-06", "LAB", "NYA", LA),
    ])


def game(feats, game_id):
    rows = feats[feats["game_id"] == game_id]
    assert len(rows) == 1
    return rows.iloc[0]


class TestHaversine:
    def test_known_distance(self):
        d = haversine_km(NY["lat"], NY["lon"], LA["lat"], LA["lon"])
        assert d == pytest.approx(NY_TO_LA_KM, abs=15)

    def test_zero_distance(self):
        assert haversine_km(NY["lat"], NY["lon"], NY["lat"], NY["lon"]) == 0.0


class TestComputeScheduleFeatures:
    def test_back_to_back(self):
        """The plan-mandated case: consecutive calendar days -> b2b, rest 0."""
        feats = compute_schedule_features(two_team_series())
        g2 = game(feats, 2)
        assert g2["home_rest_days"] == 0 and g2["home_b2b"] == True
        assert g2["away_rest_days"] == 0 and g2["away_b2b"] == True

    def test_rest_days_count_idle_days(self):
        feats = compute_schedule_features(two_team_series())
        g3 = game(feats, 3)  # Jan 2 -> Jan 5: two idle days
        assert g3["home_rest_days"] == 2
        assert g3["away_rest_days"] == 2

    def test_season_opener_semantics(self):
        feats = compute_schedule_features(two_team_series())
        g1 = game(feats, 1)
        for side in ("home", "away"):
            assert pd.isna(g1[f"{side}_rest_days"])
            assert g1[f"{side}_b2b"] == False
            assert pd.isna(g1[f"{side}_travel_km"])
            assert g1[f"{side}_tz_shift"] == 0
            assert g1[f"{side}_game_num"] == 1

    def test_travel_km(self):
        feats = compute_schedule_features(two_team_series())
        g2 = game(feats, 2)   # home stand: nobody moved
        assert g2["home_travel_km"] == 0.0
        assert g2["away_travel_km"] == 0.0
        g3 = game(feats, 3)   # both sides flew NY -> LA
        assert g3["home_travel_km"] == pytest.approx(NY_TO_LA_KM, abs=15)
        assert g3["away_travel_km"] == pytest.approx(NY_TO_LA_KM, abs=15)

    def test_tz_shift_westward(self):
        feats = compute_schedule_features(two_team_series())
        g3 = game(feats, 3)   # NY (UTC-5) -> LA (UTC-8) in January
        assert g3["home_tz_shift"] == -3
        assert g3["away_tz_shift"] == -3
        g4 = game(feats, 4)   # stayed in LA
        assert g4["home_tz_shift"] == 0

    def test_game_num_and_stage(self):
        feats = compute_schedule_features(two_team_series())
        assert game(feats, 4)["home_game_num"] == 4
        assert game(feats, 4)["away_game_num"] == 4
        assert (feats["season_stage"] == "EARLY").all()

    def test_tz_shift_includes_dst_change(self):
        """Crossing the spring-forward weekend while flying east: 3 zones of
        travel + 1 hour of DST = 4 hours of local-clock displacement. The
        feature measures circadian disruption, so this is intended."""
        sched = make_schedule([
            (1, "2021-03-13", "LAB", "NYA", LA),  # PST, UTC-8
            (2, "2021-03-15", "NYA", "LAB", NY),  # EDT, UTC-4 (DST began Mar 14)
        ])
        feats = compute_schedule_features(sched)
        assert game(feats, 2)["home_tz_shift"] == 4
        assert game(feats, 2)["away_tz_shift"] == 4

    def test_rest_resets_across_seasons(self):
        sched = make_schedule([
            (1, "2021-05-01", "NYA", "LAB", NY),
        ])
        s2 = make_schedule([(2, "2021-10-15", "NYA", "LAB", NY)])
        s2["season"] = 20212022
        feats = compute_schedule_features(pd.concat([sched, s2], ignore_index=True))
        g2 = game(feats, 2)
        assert pd.isna(g2["home_rest_days"])  # opener, not 166 days of rest
        assert g2["home_game_num"] == 1


# ─────────────────────────────────────────────
# Integration (requires populated DB + venue seed)
# ─────────────────────────────────────────────
SEASON = 20202021


@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            n_games = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE season = :s"), {"s": SEASON}).scalar()
            n_seeded = conn.execute(text(
                "SELECT COUNT(*) FROM raw.teams WHERE latitude IS NOT NULL")).scalar()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if n_games < 800:
        pytest.skip(f"season {SEASON} not backfilled ({n_games} games)")
    if n_seeded < 32:
        pytest.skip(f"venue seed not applied ({n_seeded} teams with coords)")
    return engine


@pytest.fixture(scope="module")
def built(db):
    build_schedule_features(SEASON)
    return db


class TestIntegration:
    def test_every_game_has_a_row(self, built):
        with built.connect() as conn:
            missing = conn.execute(text("""
                SELECT COUNT(*) FROM raw.games g
                LEFT JOIN features.matchup m USING (game_id)
                WHERE g.season = :s AND m.game_id IS NULL
            """), {"s": SEASON}).scalar()
        assert missing == 0

    def test_real_back_to_backs_flagged(self, built):
        """Independently find home-team b2bs in raw.games; matchup must agree."""
        with built.connect() as conn:
            disagree = conn.execute(text("""
                WITH team_games AS (
                    SELECT game_id, date, home_team AS team, TRUE AS is_home
                    FROM raw.games WHERE season = :s
                    UNION ALL
                    SELECT game_id, date, away_team, FALSE
                    FROM raw.games WHERE season = :s
                ), with_prev AS (
                    SELECT game_id, team, is_home,
                           date - LAG(date) OVER (PARTITION BY team ORDER BY date, game_id) AS gap
                    FROM team_games
                )
                SELECT COUNT(*) FROM with_prev wp
                JOIN features.matchup m USING (game_id)
                WHERE wp.gap IS NOT NULL AND wp.is_home
                  AND (m.home_b2b <> (wp.gap = 1)
                       OR m.home_rest_days <> wp.gap - 1)
            """), {"s": SEASON}).scalar()
        assert disagree == 0

    def test_openers_and_ranges(self, built):
        with built.connect() as conn:
            bad = conn.execute(text("""
                SELECT COUNT(*) FROM features.matchup m
                JOIN raw.games g USING (game_id)
                WHERE g.season = :s AND (
                    (m.home_game_num = 1 AND (m.home_rest_days IS NOT NULL
                                              OR m.home_travel_km IS NOT NULL
                                              OR m.home_b2b))
                    OR m.home_travel_km < 0 OR m.away_travel_km < 0
                    OR m.home_travel_km > 5000 OR m.away_travel_km > 5000
                    -- ±4: 3 zones of travel plus one hour of DST changeover
                    OR ABS(m.home_tz_shift) > 4 OR ABS(m.away_tz_shift) > 4
                    OR m.season_stage NOT IN ('EARLY','MID','LATE')
                )
            """), {"s": SEASON}).scalar()
        assert bad == 0
