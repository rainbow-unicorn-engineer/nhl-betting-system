"""
ingestion/moneypuck.py
MoneyPuck CSV data loader — free shot-level NHL data with pre-computed xG back to 2007-08.
Downloads: https://moneypuck.com/data.htm
"""
import logging
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text
from tqdm import tqdm

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
    """Parse MoneyPuck shots CSV and load into raw.shots."""
    if csv_path is None:
        csv_path = DATA_DIR / f"moneypuck_shots_{season_start_year}.csv"

    if not csv_path.exists():
        logger.error(f"Shots CSV not found: {csv_path}")
        return 0

    logger.info(f"Loading MoneyPuck shots from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing shots"):
        home_sk = row.get("homeSkatersOnIce", 5)
        away_sk = row.get("awaySkatersOnIce", 5)
        is_home = row.get("isHomeTeam", 0)
        strength = f"{int(home_sk)}v{int(away_sk)}" if is_home else f"{int(away_sk)}v{int(home_sk)}"

        home_goals = row.get("homeTeamGoals", 0)
        away_goals = row.get("awayTeamGoals", 0)
        score_state = int(home_goals) - int(away_goals)

        event_type = str(row.get("event", "SHOT")).upper()
        if event_type not in ("SHOT", "GOAL", "MISS", "BLOCK"):
            event_type = "SHOT"

        record = {
            "game_id": int(row.get("game_id", 0)),
            "season": int(row.get("season", season_start_year * 10000 + season_start_year + 1)),
            "period": int(row.get("period", 0)),
            "time_elapsed": int(row.get("time", 0)),
            "team": str(row.get("teamCode", "")),
            "shooter_id": _safe_int(row.get("shooterPlayerId")),
            "goalie_id": _safe_int(row.get("goalieIdForShot")),
            "x": _safe_float(row.get("arenaAdjustedXCord")),
            "y": _safe_float(row.get("arenaAdjustedYCord")),
            "shot_type": str(row.get("shotType", "")) if pd.notna(row.get("shotType")) else None,
            "event_type": event_type,
            "is_goal": bool(row.get("goal", 0)),
            "xg_moneypuck": _safe_float(row.get("xGoal")),
            "strength": strength,
            "score_state": score_state,
            "is_rebound": bool(row.get("shotRebound", 0)),
            "is_rush": bool(row.get("shotRush", 0)),
            "shot_distance": _safe_float(row.get("shotDistance")),
            "shot_angle": _safe_float(row.get("shotAngle")),
        }
        records.append(record)

    logger.info(f"Inserting {len(records)} shots into raw.shots...")
    insert_df = pd.DataFrame(records)
    chunk_size = 10000
    inserted = 0

    with engine.begin() as conn:
        season_val = insert_df["season"].iloc[0] if len(insert_df) > 0 else 0
        conn.execute(text("DELETE FROM raw.shots WHERE season = :s"), {"s": season_val})

        for start in range(0, len(insert_df), chunk_size):
            chunk = insert_df.iloc[start:start + chunk_size]
            chunk.to_sql("shots", conn, schema="raw", if_exists="append", index=False, method="multi")
            inserted += len(chunk)
            if inserted % 50000 == 0:
                logger.info(f"  ... {inserted}/{len(records)} shots inserted")

    logger.info(f"Loaded {inserted} shots for season starting {season_start_year}")
    return inserted


def _safe_int(val):
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.moneypuck <start_year> [--download]")
        sys.exit(1)

    year = int(sys.argv[1])
    do_download = "--download" in sys.argv

    csv = download_shots_csv(year, force=True) if do_download else DATA_DIR / f"moneypuck_shots_{year}.csv"
    load_shots_to_db(year, csv)
