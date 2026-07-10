"""
features/goalie_features.py
Goalie rolling features with Buhlmann credibility shrinkage (Phase 2, Task 3).
Populates features.goalie_rolling — one row per (game_id, goalie_id, window).

This fills the biggest gap found in the reviewed repos: goalie-conditioned
prediction. Methodology per saiemgilani/Goalie_Model_NHL (clean-room):
Buhlmann credibility Z = n / (n + k) blends a goalie's own windowed rate with
the league mean, so rookies / backups / early-season goalies shrink hard
toward league average and workhorses keep their raw number.

Design decisions (documented for Task 6 consumers):
- An observation is an APPEARANCE (toi_seconds > 0). Dressed backups with
  0 TOI are excluded. `starts_in_window` counts prior appearances in the
  window (98%+ of appearances are starts; relief stints carry real signal
  and the per-60 / ratio-of-sums rates weight them by exposure anyway).
- Windows roll within (goalie, season), ordered by (date, game_id), and use
  only STRICTLY PRIOR appearances (shift-then-roll — same structural
  no-leakage guarantee as team_features).
- raw.goalie_games.xga/gsax are never populated by the boxscore ingestion;
  all xG-based stats are computed from raw.shots (MoneyPuck) per goalie-game.
  GSAx = xGA − actual goals on unblocked attempts (empty-net and shootout
  goals are naturally excluded — they carry no goalie xG).
- High-danger = on-target shots (SHOT/GOAL events) with xG ≥ 0.20
  (league avg xG per goal-event is ~0.20; per on-target shot ~0.06).
- League priors are taken from the PREVIOUS season (strictly point-in-time);
  for the earliest season in the DB we fall back to long-run constants.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from features.util import WINDOWS

logger = logging.getLogger("nhl.features.goalie")

HD_XG_THRESHOLD = 0.20

# Fallbacks when no prior season exists in the DB (earliest season only).
LEAGUE_SV_FALLBACK = 0.905     # long-run NHL league save percentage
LEAGUE_GSAX60_FALLBACK = 0.0   # GSAx is zero-sum against the xG model

# Buhlmann credibility constant, in units of appearances.
# Estimated by estimate_k() via one-way random-effects ANOVA on
# per-appearance SV% (>=5 appearances, >=10 shots faced per appearance).
# Per-season estimates over the backfilled history: 57.6 (2020-21),
# 75.1 (2021-22), 38.1 (2022-23), 65.6 (2023-24), 93.8 (2024-25) —
# noisy, as variance-ratio estimates are, so we use the pooled mean.
# Interpretation: a goalie needs ~66 starts before their own record
# outweighs the league prior (Z=0.5) — single-season SV% is mostly noise,
# consistent with the goalie-analytics literature.
DEFAULT_K = 66.0

BASE_SQL = """
WITH shot_agg AS (
    SELECT game_id, goalie_id,
           COUNT(*)                                        AS fen_att,
           SUM(is_goal::int)                               AS fen_goals,
           SUM(xg_moneypuck)                               AS xga_shots,
           COUNT(*) FILTER (WHERE event_type IN ('SHOT','GOAL')
                              AND xg_moneypuck >= :hd)     AS hd_att,
           SUM(is_goal::int) FILTER (WHERE event_type IN ('SHOT','GOAL')
                              AND xg_moneypuck >= :hd)     AS hd_goals
    FROM raw.shots
    GROUP BY game_id, goalie_id
)
SELECT gg.game_id, gg.player_id AS goalie_id, g.season, g.date,
       gg.is_starter, gg.saves, gg.shots_against, gg.goals_against,
       gg.toi_seconds,
       sa.fen_att, sa.fen_goals, sa.xga_shots, sa.hd_att, sa.hd_goals
FROM raw.goalie_games gg
JOIN raw.games g ON g.game_id = gg.game_id
LEFT JOIN shot_agg sa ON sa.game_id = gg.game_id AND sa.goalie_id = gg.player_id
WHERE gg.toi_seconds > 0
  AND g.game_state IN ('FINAL', 'OFF')
  {season_filter}
ORDER BY gg.player_id, g.season, g.date, gg.game_id
"""


def load_goalie_base(season: Optional[int] = None) -> pd.DataFrame:
    """Load per-appearance goalie stats joined with per-goalie shot xG."""
    season_filter = "AND g.season = :season" if season else ""
    sql = BASE_SQL.format(season_filter=season_filter)
    params = {"hd": HD_XG_THRESHOLD}
    if season:
        params["season"] = season
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    # GSAx per appearance: expected minus actual goals on unblocked attempts
    df["gsax"] = df["xga_shots"] - df["fen_goals"]
    return df


def league_priors(season: int) -> Tuple[float, float]:
    """
    Point-in-time league priors for shrinkage targets: the PREVIOUS season's
    league SV% and league GSAx/60. Falls back to long-run constants when no
    prior season exists in the DB.
    """
    prev = season - 10001  # 20212022 -> 20202021
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT SUM(gg.saves)::float / NULLIF(SUM(gg.shots_against), 0) AS sv
            FROM raw.goalie_games gg JOIN raw.games g USING (game_id)
            WHERE g.season = :prev AND gg.toi_seconds > 0
        """), {"prev": prev}).one()
        gsax_row = conn.execute(text("""
            SELECT 3600.0 * SUM(s.xg_moneypuck - s.is_goal::int)
                   / NULLIF(SUM(gg.toi_seconds), 0) AS gsax60
            FROM raw.goalie_games gg
            JOIN raw.games g USING (game_id)
            LEFT JOIN raw.shots s ON s.game_id = gg.game_id
                                 AND s.goalie_id = gg.player_id
            WHERE g.season = :prev AND gg.toi_seconds > 0
        """), {"prev": prev}).one()

    if row.sv is None:
        logger.info(f"No prior season for {season}; using fallback league priors.")
        return LEAGUE_SV_FALLBACK, LEAGUE_GSAX60_FALLBACK
    gsax60 = gsax_row.gsax60 if gsax_row.gsax60 is not None else LEAGUE_GSAX60_FALLBACK
    return float(row.sv), float(gsax60)


def estimate_k(season: int, min_appearances: int = 5, min_shots: int = 10) -> float:
    """
    Estimate the Buhlmann credibility constant k (in appearances) from one
    season via one-way random-effects ANOVA on per-appearance SV%:

        k = sigma^2_within / sigma^2_between

    where sigma^2_within is the pooled variance of a goalie's single-game SV%
    around their own mean, and sigma^2_between is the variance of true
    goalie means (method-of-moments: (MSB - MSW) / m0).
    """
    base = load_goalie_base(season)
    base = base[base["shots_against"] >= min_shots].copy()
    base["game_sv"] = base["saves"] / base["shots_against"]

    counts = base.groupby("goalie_id")["game_sv"].transform("count")
    base = base[counts >= min_appearances]

    groups = base.groupby("goalie_id")["game_sv"]
    m = groups.count()
    means = groups.mean()
    grand = base["game_sv"].mean()
    N, I = len(base), len(m)

    ssw = ((base["game_sv"] - base["goalie_id"].map(means)) ** 2).sum()
    msw = ssw / (N - I)
    msb = (m * (means - grand) ** 2).sum() / (I - 1)
    m0 = (N - (m ** 2).sum() / N) / (I - 1)
    var_between = (msb - msw) / m0

    if var_between <= 0:
        logger.warning(f"Non-positive between-goalie variance for {season}; "
                       f"falling back to DEFAULT_K={DEFAULT_K}")
        return DEFAULT_K

    k = float(msw / var_between)
    logger.info(f"estimate_k({season}): msw={msw:.6f} var_between={var_between:.6f} "
                f"k={k:.1f} (goalies={I}, appearances={N})")
    return k


_SUM_COLS = ["saves", "shots_against", "goals_against", "toi_seconds",
             "xga_shots", "gsax", "fen_att", "fen_goals", "hd_att", "hd_goals"]


def compute_goalie_rolling(base: pd.DataFrame, windows=WINDOWS,
                           k: float = DEFAULT_K,
                           league_sv: float = LEAGUE_SV_FALLBACK,
                           league_gsax60: float = LEAGUE_GSAX60_FALLBACK) -> pd.DataFrame:
    """
    Pure function: rolling goalie features for every (appearance, window).
    Windows use only strictly-prior appearances within (goalie, season).
    """
    base = base.sort_values(["goalie_id", "season", "date", "game_id"]).reset_index(drop=True)
    grouped = base.groupby(["goalie_id", "season"], sort=False)

    prior = grouped[_SUM_COLS].shift(1)
    n_prior = grouped.cumcount()

    out_frames = []
    for w in windows:
        roll = (
            prior.groupby([base["goalie_id"], base["season"]], sort=False)[_SUM_COLS]
            .rolling(w, min_periods=1).sum()
            .reset_index(drop=True)
        )
        n = n_prior.clip(upper=w)
        z = n / (n + k)

        f = pd.DataFrame({
            "game_id": base["game_id"],
            "goalie_id": base["goalie_id"],
            "window_size": w,
            "season": base["season"],
            "starts_in_window": n.astype(int),
            "credibility_z": z,
        })
        f["sv_pct"] = roll["saves"] / roll["shots_against"]
        f["gsax_per60"] = (3600.0 * roll["gsax"] / roll["toi_seconds"]).clip(-99.0, 99.0)
        f["hd_sv_pct"] = 1.0 - roll["hd_goals"] / roll["hd_att"]
        f["fenwick_sv_pct"] = 1.0 - roll["fen_goals"] / roll["fen_att"]
        f["xga_per60"] = (3600.0 * roll["xga_shots"] / roll["toi_seconds"]).clip(upper=99.0)

        # Buhlmann blend. With n=0 the raw rate is NaN but Z=0 — the shrunk
        # value is exactly the league prior, never NaN.
        f["shrunk_sv_pct"] = np.where(
            n > 0, z * f["sv_pct"] + (1 - z) * league_sv, league_sv)
        f["shrunk_gsax"] = np.where(
            n > 0, z * f["gsax_per60"] + (1 - z) * league_gsax60, league_gsax60)

        out_frames.append(f)

    out = pd.concat(out_frames, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


_FEATURE_COLS = [
    "game_id", "goalie_id", "window_size",
    "sv_pct", "gsax_per60", "hd_sv_pct", "fenwick_sv_pct", "xga_per60",
    "starts_in_window", "credibility_z", "shrunk_sv_pct", "shrunk_gsax",
]


def write_goalie_rolling(df: pd.DataFrame) -> int:
    """Replace features.goalie_rolling rows for the seasons present in df."""
    seasons = sorted(df["season"].unique().tolist())
    payload = df[_FEATURE_COLS]

    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM features.goalie_rolling
            WHERE game_id IN (SELECT game_id FROM raw.games WHERE season = ANY(:seasons))
        """), {"seasons": seasons})
        payload.to_sql("goalie_rolling", conn, schema="features",
                       if_exists="append", index=False,
                       chunksize=10000, method="multi")

    logger.info(f"Wrote {len(payload)} goalie_rolling rows for seasons {seasons}")
    return len(payload)


def build_goalie_rolling(season: Optional[int] = None, k: float = DEFAULT_K) -> int:
    """
    Build features.goalie_rolling for one season or all seasons. Seasons are
    processed independently because each has its own point-in-time league
    prior (the previous season's league rates).
    """
    if season:
        seasons = [season]
    else:
        with engine.connect() as conn:
            seasons = [r[0] for r in conn.execute(text(
                "SELECT DISTINCT season FROM raw.games ORDER BY season"))]

    total = 0
    for s in seasons:
        base = load_goalie_base(s)
        if base.empty:
            logger.warning(f"No goalie appearances for season {s}; skipping.")
            continue
        league_sv, league_gsax60 = league_priors(s)
        logger.info(f"Season {s}: league priors sv={league_sv:.4f} "
                    f"gsax60={league_gsax60:+.3f}, k={k}")
        feats = compute_goalie_rolling(base, k=k, league_sv=league_sv,
                                       league_gsax60=league_gsax60)
        total += write_goalie_rolling(feats)
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build features.goalie_rolling")
    parser.add_argument("--season", type=int, default=None,
                        help="Season as YYYYYYYY (e.g. 20212022); omit for all")
    parser.add_argument("--estimate-k", action="store_true",
                        help="Print the empirical k for --season and exit")
    args = parser.parse_args()

    if args.estimate_k:
        if not args.season:
            parser.error("--estimate-k requires --season")
        print(f"k = {estimate_k(args.season):.1f}")
    else:
        build_goalie_rolling(args.season)
