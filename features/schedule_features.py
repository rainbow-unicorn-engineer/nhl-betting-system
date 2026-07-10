"""
features/schedule_features.py
Schedule + travel context features (Phase 2, Task 4).
Populates the schedule-owned columns of features.matchup.

Everything here is knowable strictly before puck drop — the league schedule
is public — so these features are point-in-time safe by nature. Each value
derives only from the team's PREVIOUS game and the current game's venue.

Per team and game:
- rest_days:  full days off since the team's previous game (0 = back-to-back,
              1 = one idle day, ...). NULL for a team's first game of season.
- b2b:        rest_days == 0. False for season openers.
- travel_km:  great-circle distance from the previous game's venue to this
              game's venue (venue = home team's arena; raw.teams seed data).
              NULL for season openers.
- tz_shift:   integer hours of timezone change from the previous game's venue
              (positive = moved east). 0 for season openers. DST-aware via
              IANA zone offsets at the respective game dates.
- game_num:   1-based game number within (team, season); playoff games
              continue the count past 82.
- season_stage: EARLY/MID/LATE classified from the HOME team's game number.

The matchup table is shared with elo.py (home/away_elo) and Task 6
(starter ids), so this module UPSERTS only the columns it owns and never
deletes rows.
"""
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from features.util import season_stage

logger = logging.getLogger("nhl.features.schedule")

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km (inputs in degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _utc_offset_hours(tz_name: str, on_date) -> float:
    """UTC offset in hours of an IANA zone at noon local on a given date."""
    dt = datetime(on_date.year, on_date.month, on_date.day, 12,
                  tzinfo=ZoneInfo(tz_name))
    return dt.utcoffset().total_seconds() / 3600.0


def load_schedule(season: Optional[int] = None) -> pd.DataFrame:
    """One row per game with home/away teams and the venue geo of the game."""
    season_filter = "WHERE g.season = :season" if season else ""
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT g.game_id, g.season, g.date, g.home_team, g.away_team,
                   t.latitude::float AS venue_lat, t.longitude::float AS venue_lon,
                   t.venue_timezone
            FROM raw.games g
            JOIN raw.teams t ON t.team_abbrev = g.home_team
            {season_filter}
            ORDER BY g.date, g.game_id
        """), conn, params={"season": season} if season else {})
    if df["venue_lat"].isna().any():
        missing = df.loc[df["venue_lat"].isna(), "home_team"].unique()
        raise ValueError(
            f"Venue seed data missing for teams {sorted(missing)} — "
            f"run db/seed_venues.sql first."
        )
    return df


def compute_schedule_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function: per-game schedule features from a load_schedule() frame.
    Returns one row per game_id with home_/away_ column pairs.
    """
    # Long format: one row per (game, team), venue geo = the game's arena
    long = pd.concat([
        schedule.assign(team=schedule["home_team"], is_home=True),
        schedule.assign(team=schedule["away_team"], is_home=False),
    ], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["team", "season", "date", "game_id"]).reset_index(drop=True)

    grouped = long.groupby(["team", "season"], sort=False)
    long["game_num"] = grouped.cumcount() + 1
    prev = grouped[["date", "venue_lat", "venue_lon", "venue_timezone"]].shift(1)

    long["rest_days"] = (long["date"] - prev["date"]).dt.days - 1
    long["b2b"] = long["rest_days"] == 0  # NaN rest (opener) -> False
    long["travel_km"] = haversine_km(
        prev["venue_lat"], prev["venue_lon"], long["venue_lat"], long["venue_lon"])

    # DST-aware tz offsets, memoized per (zone, date) — cheap at NHL scale
    offsets = {}

    def offset(tz, d):
        if pd.isna(tz) or pd.isna(d):
            return np.nan
        key = (tz, d.date())
        if key not in offsets:
            offsets[key] = _utc_offset_hours(tz, d)
        return offsets[key]

    cur_off = np.array([offset(tz, d) for tz, d
                        in zip(long["venue_timezone"], long["date"])])
    prev_off = np.array([offset(tz, d) for tz, d
                         in zip(prev["venue_timezone"], prev["date"])])
    with np.errstate(invalid="ignore"):
        long["tz_shift"] = np.round(cur_off - prev_off)
    long["tz_shift"] = long["tz_shift"].fillna(0).astype(int)  # openers -> 0

    # Pivot back to one row per game
    home = long[long["is_home"]].set_index("game_id")
    away = long[~long["is_home"]].set_index("game_id")
    out = pd.DataFrame({
        "home_team": home["home_team"],
        "away_team": home["away_team"],
        "home_rest_days": home["rest_days"],
        "away_rest_days": away["rest_days"],
        "home_b2b": home["b2b"],
        "away_b2b": away["b2b"],
        "home_travel_km": home["travel_km"].round(1),
        "away_travel_km": away["travel_km"].round(1),
        "home_tz_shift": home["tz_shift"],
        "away_tz_shift": away["tz_shift"],
        "home_game_num": home["game_num"],
        "away_game_num": away["game_num"],
        "season_stage": home["game_num"].map(season_stage),
    })
    return out.reset_index()


_UPSERT_SQL = text("""
    INSERT INTO features.matchup
        (game_id, home_team, away_team,
         home_rest_days, away_rest_days, home_b2b, away_b2b,
         home_travel_km, away_travel_km, home_tz_shift, away_tz_shift,
         home_game_num, away_game_num, season_stage)
    VALUES
        (:game_id, :home_team, :away_team,
         :home_rest_days, :away_rest_days, :home_b2b, :away_b2b,
         :home_travel_km, :away_travel_km, :home_tz_shift, :away_tz_shift,
         :home_game_num, :away_game_num, :season_stage)
    ON CONFLICT (game_id) DO UPDATE SET
        home_team = EXCLUDED.home_team,
        away_team = EXCLUDED.away_team,
        home_rest_days = EXCLUDED.home_rest_days,
        away_rest_days = EXCLUDED.away_rest_days,
        home_b2b = EXCLUDED.home_b2b,
        away_b2b = EXCLUDED.away_b2b,
        home_travel_km = EXCLUDED.home_travel_km,
        away_travel_km = EXCLUDED.away_travel_km,
        home_tz_shift = EXCLUDED.home_tz_shift,
        away_tz_shift = EXCLUDED.away_tz_shift,
        home_game_num = EXCLUDED.home_game_num,
        away_game_num = EXCLUDED.away_game_num,
        season_stage = EXCLUDED.season_stage,
        computed_at = NOW()
""")


def write_schedule_features(df: pd.DataFrame) -> int:
    """Upsert schedule-owned matchup columns (Elo/starter columns untouched)."""
    records = df.replace({np.nan: None}).to_dict("records")
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, records)
    logger.info(f"Upserted schedule features for {len(records)} games")
    return len(records)


def build_schedule_features(season: Optional[int] = None) -> int:
    """Build schedule/travel features for one season or all seasons."""
    scope = season or "ALL"
    logger.info(f"Building schedule features (season={scope})...")
    schedule = load_schedule(season)
    if schedule.empty:
        logger.warning(f"No games found (season={scope}); nothing to build.")
        return 0
    return write_schedule_features(compute_schedule_features(schedule))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build schedule features in features.matchup")
    parser.add_argument("--season", type=int, default=None,
                        help="Season as YYYYYYYY (e.g. 20212022); omit for all")
    args = parser.parse_args()
    build_schedule_features(args.season)
