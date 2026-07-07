"""
ingestion/moneypuck.py
MoneyPuck CSV data loader — free shot-level NHL data with pre-computed xG back to 2007-08.
Downloads: https://moneypuck.com/data.htm
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from config.settings import engine, DATA_DIR

logger = logging.getLogger("nhl.ingestion.moneypuck")


def download_shots_csv(season_start_year: int, force: bool = False) -> Path:
    """Download MoneyPuck shots CSV for a given season (season_start_year e.g. 2024)."""
    csv_path = DATA_DIR / f"moneypuck_shots_{season_start_year}.csv"
    if csv_path.exists() and not force:
        logger.info(f"MoneyPuck shots file already exists: {csv_path}")
        return csv_path

    url = f"https://peter-tanner.com/moneypuck/downloads/shots_{season_start_year}.zip"
    logger.info(f"Downloading MoneyPuck shots for {season_start_year}...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        if url.endswith(".zip"):
            import zipfile, io
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                with zf.open(csv_name) as f:
                    csv_path.write_bytes(f.read())
        else:
            csv_path.write_bytes(resp.content)

        logger.info(f"Downloaded to {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
        return csv_path

    except Exception as e:
        logger.error(f"Download failed: {e}. Place the CSV manually at {csv_path}")
        raise


def load_shots_to_db(season_start_year: int, csv_path: Path = None):
    """
    Parse MoneyPuck shots CSV and load into raw.shots (vectorized).

    MoneyPuck format gotchas (verified against live 2024 file):
    - `game_id` is short-form (e.g. 20001); the real NHL game_id is
      season_start_year * 1_000_000 + short_id (e.g. 2024020001)
    - `season` is the start year (2024); our convention is 20242025
    - `time` is seconds elapsed in the GAME, not the period
    - `event` values are SHOT / MISS / GOAL (blocked shots not included)
    """
    if csv_path is None:
        csv_path = DATA_DIR / f"moneypuck_shots_{season_start_year}.csv"

    if not csv_path.exists():
        logger.error(f"Shots CSV not found: {csv_path}")
        return 0

    logger.info(f"Loading MoneyPuck shots from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    season = season_start_year * 10000 + (season_start_year + 1)

    out = pd.DataFrame()
    out["game_id"] = season_start_year * 1_000_000 + df["game_id"].astype(int)
    out["season"] = season
    out["period"] = df["period"].fillna(0).astype(int)
    out["time_elapsed"] = df["time"].fillna(0).astype(int)  # seconds into game
    out["team"] = df["teamCode"].astype(str)
    out["shooter_id"] = df["shooterPlayerId"].astype("Int64")
    out["goalie_id"] = df["goalieIdForShot"].astype("Int64")
    out["x"] = df["arenaAdjustedXCord"]
    out["y"] = df["arenaAdjustedYCord"]
    out["shot_type"] = df["shotType"].where(df["shotType"].notna(), None)

    event = df["event"].astype(str).str.upper()
    out["event_type"] = event.where(event.isin(["SHOT", "GOAL", "MISS", "BLOCK"]), "SHOT")
    out["is_goal"] = df["goal"].fillna(0).astype(bool)
    out["xg_moneypuck"] = df["xGoal"]

    home_sk = df["homeSkatersOnIce"].fillna(5).astype(int).astype(str)
    away_sk = df["awaySkatersOnIce"].fillna(5).astype(int).astype(str)
    is_home = df["isHomeTeam"].fillna(0).astype(int) == 1
    out["strength"] = np.where(is_home, home_sk + "v" + away_sk, away_sk + "v" + home_sk)

    out["score_state"] = (
        df["homeTeamGoals"].fillna(0).astype(int) - df["awayTeamGoals"].fillna(0).astype(int)
    )
    out["is_rebound"] = df["shotRebound"].fillna(0).astype(bool)
    out["is_rush"] = df["shotRush"].fillna(0).astype(bool)
    out["shot_distance"] = df["shotDistance"]
    out["shot_angle"] = df["shotAngle"]

    # FK safety: raw.shots.game_id references raw.games. Drop (and report)
    # shots for games we don't have — e.g. if the schedule backfill is
    # incomplete for this season.
    with engine.connect() as conn:
        known = {
            r[0] for r in conn.execute(
                text("SELECT game_id FROM raw.games WHERE season = :s"), {"s": season}
            )
        }
    missing_mask = ~out["game_id"].isin(known)
    if missing_mask.any():
        n_missing_games = out.loc[missing_mask, "game_id"].nunique()
        logger.warning(
            f"Dropping {int(missing_mask.sum())} shots from {n_missing_games} games "
            f"not present in raw.games for season {season} — run the NHL API "
            f"backfill for this season first if this number is large."
        )
        out = out[~missing_mask]

    logger.info(f"Inserting {len(out)} shots into raw.shots...")
    chunk_size = 10000
    inserted = 0

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.shots WHERE season = :s"), {"s": season})

        for start in range(0, len(out), chunk_size):
            chunk = out.iloc[start:start + chunk_size]
            chunk.to_sql("shots", conn, schema="raw", if_exists="append", index=False, method="multi")
            inserted += len(chunk)
            if inserted % 50000 == 0:
                logger.info(f"  ... {inserted}/{len(out)} shots inserted")

    logger.info(f"Loaded {inserted} shots for season starting {season_start_year}")
    return inserted


def ingest_season_shots(season: int, force_download: bool = False) -> int:
    """
    Download + load MoneyPuck shots for one season, given our season
    convention (e.g. 20242025 -> start year 2024).
    """
    start_year = season // 10000
    csv_path = download_shots_csv(start_year, force=force_download)
    return load_shots_to_db(start_year, csv_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.moneypuck <start_year> [--download]")
        sys.exit(1)

    year = int(sys.argv[1])
    do_download = "--download" in sys.argv

    csv = download_shots_csv(year, force=True) if do_download else DATA_DIR / f"moneypuck_shots_{year}.csv"
    load_shots_to_db(year, csv)
