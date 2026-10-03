"""
features/player_shots.py
Point-in-time player features for the shots-on-goal props model
(models/props_sog.py).

One row per skater game in which he played (toi_seconds > 0), regular
season and playoffs. Every feature on a row is computed from games
STRICTLY BEFORE that row's date ("shift-then-roll"): the player's own
history is shifted one appearance before rolling (a skater plays at most
one game a day), team histories are shifted one team game, and every
league-wide rate is a trailing window over dates < the row's date. So
rewriting, or deleting, every row on or after a date leaves the features
of that date's rows unchanged (tested in tests/test_props_sog.py).

Sources, all read with SELECT only (load_inputs):
- raw.skater_games: shots on goal (the target, `shots`), toi_seconds,
  position, team. NOT USED: pp_toi_seconds, sh_toi_seconds, pp_goals,
  pp_assists, fow, fol. On the live database those columns are all zero
  (their defaults) because ingestion/nhl_stats.py has never run there, so
  a power-play-time feature would be a column of zeros, not information.
- raw.shots (MoneyPuck, shot level): shot ATTEMPTS by shooter, every
  type stored (the query takes SHOT, GOAL, MISS and BLOCK) — the shot-
  attempt rate, a less noisy cousin of the shots-on-goal rate. On the
  live database raw.shots holds NO blocked attempts (event types SHOT,
  GOAL, MISS only; checked 2026-10-02), so the rate is in practice
  UNBLOCKED attempts (Fenwick), not all attempts (Corsi). It also agrees
  with the box score: shots on goal from raw.shots match
  raw.skater_games.shots on 99.3% of player-games. A game with no
  raw.shots rows at all has
  UNKNOWN attempts (NaN), not zero; attempt rates are ratios of sums over
  the games whose attempts are known.
- raw.team_games: team shots on goal per game -> own team's shots for and
  the opponent's shots against.
- raw.games: dates, home/away, the schedule (back-to-backs, games missed)
  and the scores for pre-game Elo (features.elo.compute_elo, the same pure
  function that fills features.matchup: it records each game's rating
  BEFORE that game's result is applied).

Drift: league shots on goal per 60 minutes fell from 6.39 (2021-22) to
5.61 (2025-26). Rate features are therefore also given RELATIVE to the
trailing league rate (`*_rel`: divided by the trailing-365-day rate for
the player's position, or the league team shots per game for team
features), so a tree reads "vs the league right now" rather than a level
tied to one season. The level itself belongs to the model's offset
(models/props_sog.py, exposure baseline and drift ratio).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nhl.features.player_shots")

ROLL_GAMES = (5, 10, 20, 40)         # player rate windows (appearances)
TOI_GAMES = (5, 10)                  # TOI mean / std windows
TEAM_GAMES = (10, 20)                # team shots windows (team games)
EXP_TOI_GAMES = 10                   # expected TOI: mean of last 10
EXP_TOI_RANGE_GAMES = 20             # ... capped to the last-20 range
B0_GAMES = 10                        # rolling-average baseline window
B0_MIN_GAMES = 5                     # fewer prior games -> league mean
LEAGUE_WINDOW_DAYS = 365
SHRINK_TOI_SECONDS = 300 * 60        # k = 300 minutes of TOI

# Pseudo-observations that keep league rates defined on the first dates
# of the data (empty trailing window). Their weight is tiny next to a
# season of real data (~800,000 skater minutes), so they only matter in
# the first days of 2020-21, which are training-only rows.
LEAGUE_PRIOR_SECONDS = 6000 * 60
LEAGUE_PRIOR_SOG60 = {"F": 7.0, "D": 4.5, "ALL": 6.2}
LEAGUE_PRIOR_GAMES = 2000            # pseudo player-games for SOG per game
LEAGUE_PRIOR_SOG_PG = {"F": 1.9, "D": 1.3}
TEAM_PRIOR_GAMES = 100               # pseudo team-games
TEAM_PRIOR_SOG = 30.0

FORWARD_POSITIONS = {"C", "L", "R", "F", "LW", "RW"}

FEATURES = (
    [f"sog60_l{n}_rel" for n in ROLL_GAMES] + ["sog60_season_rel"]
    + [f"att60_l{n}_rel" for n in ROLL_GAMES] + ["att60_season_rel"]
    + [f"toi_mean_l{n}" for n in TOI_GAMES]
    + [f"toi_std_l{n}" for n in TOI_GAMES]
    + ["days_since_last", "team_games_missed", "is_d", "is_home"]
    + [f"opp_sa_l{n}_rel" for n in TEAM_GAMES]
    + [f"team_sf_l{n}_rel" for n in TEAM_GAMES]
    + ["b2b", "elo_diff"]
)


def position_group(pos) -> str:
    """'D' for defensemen, 'F' for every forward position."""
    return "D" if str(pos).strip().upper() == "D" else "F"


# ── Point-in-time primitives (pure) ────────────────────────────────

def trailing_sums(dates, groups, values: dict,
                  window_days: int = LEAGUE_WINDOW_DAYS) -> dict:
    """Per row, the sum of each array in `values` over rows of the same
    group whose date d' satisfies date - window_days <= d' < date. Rows
    on the row's own date (and later) never count."""
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize().to_numpy()
    g = np.asarray(groups)
    out = {k: np.zeros(len(d)) for k in values}
    for grp in pd.unique(g):
        rows = np.flatnonzero(g == grp)
        dd = d[rows]
        udates, inv = np.unique(dd, return_inverse=True)
        lo = np.searchsorted(udates, udates - np.timedelta64(window_days, "D"),
                             side="left")
        for k, v in values.items():
            day = np.bincount(inv, weights=np.asarray(v, float)[rows],
                              minlength=len(udates))
            cs = np.concatenate([[0.0], np.cumsum(day)])
            # dates in [udate - window, udate): unique days lo .. i-1
            win = cs[np.arange(len(udates))] - cs[lo]
            out[k][rows] = win[inv]
    return out


def _prev_roll(df: pd.DataFrame, key, col: str, n: int, how: str = "sum",
               min_periods: int = 1) -> pd.Series:
    """Shift-then-roll: `how` of `col` over the previous n rows of the same
    key (rows must be sorted in time within each key)."""
    prev = df.groupby(key, sort=False)[col].shift(1)
    roll = prev.groupby([df[k] for k in np.atleast_1d(key)], sort=False) \
        .rolling(n, min_periods=min_periods)
    res = getattr(roll, how)()
    return res.reset_index(level=list(range(len(np.atleast_1d(key)))),
                           drop=True).reindex(df.index)


def _prev_cumsum(df: pd.DataFrame, key, col: str) -> pd.Series:
    """Sum of `col` over all earlier rows of the same key (0 for the first)."""
    return df.groupby(key, sort=False)[col].cumsum() - df[col]


def _ratio60(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num * 3600.0 / den, np.nan)


# ── Builder (pure) ─────────────────────────────────────────────────

def _base_frame(skaters, games, attempts) -> pd.DataFrame:
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    final = g[g["game_state"].isin(["FINAL", "OFF"])
              & g["game_type"].isin([2, 3])]
    df = skaters.merge(
        final[["game_id", "season", "date", "game_type",
               "home_team", "away_team"]], on="game_id", how="inner")
    df = df[(df["toi_seconds"].fillna(0) > 0) & df["shots"].notna()].copy()
    df["toi_seconds"] = df["toi_seconds"].astype(float)
    df["sog"] = df["shots"].astype(float)
    df["pos_group"] = df["position"].map(position_group)
    df["is_home"] = (df["team"] == df["home_team"]).astype(float)
    df["opp"] = np.where(df["is_home"] == 1.0, df["away_team"], df["home_team"])

    covered = set(attempts["game_id"].unique())
    att = attempts.rename(columns={"shooter_id": "player_id"})[
        ["game_id", "player_id", "attempts"]]
    df = df.merge(att, on=["game_id", "player_id"], how="left")
    known = df["game_id"].isin(covered)
    df["att"] = np.where(known, df["attempts"].fillna(0.0), np.nan)
    df["att_known"] = known.astype(float)
    df = df.drop(columns=["attempts"])
    return df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)


def _team_frame(team_games, games) -> pd.DataFrame:
    """Per team game: shots for and against, shift-then-roll means."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    final = g[g["game_state"].isin(["FINAL", "OFF"])
              & g["game_type"].isin([2, 3])][["game_id", "date"]]
    tg = team_games[["game_id", "team", "sog"]].merge(final, on="game_id")
    opp = tg[["game_id", "team", "sog"]].rename(
        columns={"team": "opp_team", "sog": "sa"})
    tg = tg.merge(opp, on="game_id")
    tg = tg[tg["team"] != tg["opp_team"]].rename(columns={"sog": "sf"})
    tg["sf"] = tg["sf"].astype(float)
    tg["sa"] = tg["sa"].astype(float)
    tg = tg.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    lg = trailing_sums(tg["date"], np.zeros(len(tg)),
                       {"s": tg["sf"].fillna(0.0),
                        "n": tg["sf"].notna().astype(float)})
    league_pg = ((lg["s"] + TEAM_PRIOR_GAMES * TEAM_PRIOR_SOG)
                 / (lg["n"] + TEAM_PRIOR_GAMES))
    out = tg[["game_id", "team"]].copy()
    for n in TEAM_GAMES:
        out[f"team_sf_l{n}_rel"] = _prev_roll(tg, "team", "sf", n, "mean") / league_pg
        out[f"team_sa_l{n}_rel"] = _prev_roll(tg, "team", "sa", n, "mean") / league_pg
    return out


def _schedule(games) -> pd.DataFrame:
    """One row per (team, game) of regular season + playoff games."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g = g[g["game_type"].isin([2, 3])]
    return pd.concat([
        g[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"}),
        g[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)


def build_player_features(skaters: pd.DataFrame, games: pd.DataFrame,
                          attempts: pd.DataFrame,
                          team_games: pd.DataFrame) -> pd.DataFrame:
    """The modelling frame. Inputs (plain frames, no DB access):
      skaters    player_id, game_id, team, position, toi_seconds, shots
      games      game_id, season, date, game_type, home_team, away_team,
                 home_score, away_score, game_state
      attempts   game_id, shooter_id, attempts (all attempt types; one row
                 per shooter with >= 1 attempt in a game with shot data)
      team_games game_id, team, sog
    Returns one row per played skater game with the target `sog`, the
    FEATURES columns, and the baseline ingredients: n_prior,
    league_pos_sog60, league_sog60, league_pos_sog_pg, shrunk_sog60,
    exp_toi, b0_mean."""
    df = _base_frame(skaters, games, attempts)
    key = "player_id"
    by_player = df.groupby(key, sort=False)
    df["n_prior"] = by_player.cumcount().astype(float)

    # league rates: trailing 365 days, strictly before the date
    lg_pos = trailing_sums(df["date"], df["pos_group"],
                           {"s": df["sog"], "t": df["toi_seconds"],
                            "n": np.ones(len(df))})
    prior60 = df["pos_group"].map(LEAGUE_PRIOR_SOG60).to_numpy()
    df["league_pos_sog60"] = ((lg_pos["s"] + LEAGUE_PRIOR_SECONDS * prior60 / 3600.0)
                              / (lg_pos["t"] + LEAGUE_PRIOR_SECONDS) * 3600.0)
    prior_pg = df["pos_group"].map(LEAGUE_PRIOR_SOG_PG).to_numpy()
    df["league_pos_sog_pg"] = ((lg_pos["s"] + LEAGUE_PRIOR_GAMES * prior_pg)
                               / (lg_pos["n"] + LEAGUE_PRIOR_GAMES))
    lg_all = trailing_sums(df["date"], np.zeros(len(df)),
                           {"s": df["sog"], "t": df["toi_seconds"]})
    df["league_sog60"] = ((lg_all["s"] + LEAGUE_PRIOR_SECONDS
                           * LEAGUE_PRIOR_SOG60["ALL"] / 3600.0)
                          / (lg_all["t"] + LEAGUE_PRIOR_SECONDS) * 3600.0)

    # player's career-to-date (in this database) shrunk SOG per 60
    cum_s = _prev_cumsum(df, key, "sog")
    cum_t = _prev_cumsum(df, key, "toi_seconds")
    df["shrunk_sog60"] = ((cum_s + SHRINK_TOI_SECONDS * df["league_pos_sog60"] / 3600.0)
                          / (cum_t + SHRINK_TOI_SECONDS) * 3600.0)

    # expected TOI: last-10 mean capped to the last-20 range (the cap is a
    # no-op whenever both windows are full — a mean of a subset lies in
    # the subset's range — and is kept as the specified guard)
    mean10 = _prev_roll(df, key, "toi_seconds", EXP_TOI_GAMES, "mean")
    lo20 = _prev_roll(df, key, "toi_seconds", EXP_TOI_RANGE_GAMES, "min")
    hi20 = _prev_roll(df, key, "toi_seconds", EXP_TOI_RANGE_GAMES, "max")
    df["exp_toi"] = mean10.clip(lower=lo20, upper=hi20)

    # B0: SOG per game over the last 10 appearances; league mean if < 5
    b0 = _prev_roll(df, key, "sog", B0_GAMES, "mean")
    df["b0_mean"] = np.where(df["n_prior"] >= B0_MIN_GAMES, b0,
                             df["league_pos_sog_pg"])

    # rolling player rates, relative to his position's league rate
    df["att_toi"] = df["toi_seconds"] * df["att_known"]
    df["att0"] = df["att"].fillna(0.0)
    rel = df["league_pos_sog60"].to_numpy()
    for n in ROLL_GAMES:
        s = _prev_roll(df, key, "sog", n)
        t = _prev_roll(df, key, "toi_seconds", n)
        a = _prev_roll(df, key, "att0", n)
        at = _prev_roll(df, key, "att_toi", n)
        df[f"sog60_l{n}_rel"] = _ratio60(s, t) / rel
        df[f"att60_l{n}_rel"] = _ratio60(a, at) / rel
    pkey = ["player_id", "season"]
    df["sog60_season_rel"] = _ratio60(_prev_cumsum(df, pkey, "sog"),
                                      _prev_cumsum(df, pkey, "toi_seconds")) / rel
    df["att60_season_rel"] = _ratio60(_prev_cumsum(df, pkey, "att0"),
                                      _prev_cumsum(df, pkey, "att_toi")) / rel

    for n in TOI_GAMES:
        df[f"toi_mean_l{n}"] = _prev_roll(df, key, "toi_seconds", n, "mean") / 60.0
        df[f"toi_std_l{n}"] = _prev_roll(df, key, "toi_seconds", n, "std",
                                         min_periods=2) / 60.0

    last_date = by_player["date"].shift(1)
    df["days_since_last"] = (df["date"] - last_date).dt.days.astype(float)

    # team games his CURRENT team played strictly between his last
    # appearance and this game (the schedule is known in advance)
    sched = _schedule(games)
    missed = np.full(len(df), np.nan)
    has_last = last_date.notna().to_numpy()
    for team, rows in df[has_last].groupby("team").groups.items():
        tdates = np.sort(sched.loc[sched["team"] == team, "date"].to_numpy())
        idx = np.asarray(rows)
        hi = np.searchsorted(tdates, df.loc[idx, "date"].to_numpy(), side="left")
        lo = np.searchsorted(tdates, last_date.loc[idx].to_numpy(), side="right")
        missed[idx] = np.maximum(hi - lo, 0)
    df["team_games_missed"] = missed

    df["is_d"] = (df["pos_group"] == "D").astype(float)

    # team shots: own shots for, opponent's shots against
    tf = _team_frame(team_games, games)
    own = tf[["game_id", "team"] + [f"team_sf_l{n}_rel" for n in TEAM_GAMES]]
    opp = tf[["game_id", "team"] + [f"team_sa_l{n}_rel" for n in TEAM_GAMES]] \
        .rename(columns={"team": "opp",
                         **{f"team_sa_l{n}_rel": f"opp_sa_l{n}_rel"
                            for n in TEAM_GAMES}})
    df = df.merge(own, on=["game_id", "team"], how="left") \
           .merge(opp, on=["game_id", "opp"], how="left")

    # back-to-back: his team also played the previous calendar day
    prev_day = sched.assign(date=sched["date"] + pd.Timedelta(days=1))[
        ["team", "date"]].drop_duplicates().assign(b2b=1.0)
    df = df.merge(prev_day, on=["team", "date"], how="left")
    df["b2b"] = df["b2b"].fillna(0.0)

    # pre-game Elo (each game's rating before its own result)
    from features.elo import compute_elo
    elo_in = games[["game_id", "season", "date", "home_team", "away_team",
                    "home_score", "away_score", "game_state"]].copy()
    elo_in["date"] = pd.to_datetime(elo_in["date"]).dt.normalize()
    elos, _ = compute_elo(elo_in)
    df = df.merge(elos, on="game_id", how="left")
    df["elo_diff"] = np.where(df["is_home"] == 1.0,
                              df["home_elo"] - df["away_elo"],
                              df["away_elo"] - df["home_elo"]).astype(float)

    keep = (["player_id", "game_id", "season", "date", "game_type", "team",
             "opp", "pos_group", "toi_seconds", "sog", "att", "n_prior",
             "league_pos_sog60", "league_sog60", "league_pos_sog_pg",
             "shrunk_sog60", "exp_toi", "b0_mean"] + FEATURES)
    return (df[list(dict.fromkeys(keep))]
            .sort_values(["date", "game_id", "player_id"])
            .reset_index(drop=True))


# ── Loading (SELECT only) ──────────────────────────────────────────

def load_inputs(conn=None) -> dict:
    """The four input frames for build_player_features, read with SELECT
    statements only."""
    from sqlalchemy import text

    from config.settings import engine

    if conn is None:
        with engine.connect() as c:
            return load_inputs(c)
    q = lambda sql: pd.read_sql(text(sql), conn)
    games = q("""
        SELECT game_id, season, date, game_type, home_team, away_team,
               home_score, away_score, game_state
        FROM raw.games ORDER BY date, game_id""")
    skaters = q("""
        SELECT player_id, game_id, team, position, toi_seconds, shots
        FROM raw.skater_games""")
    attempts = q("""
        SELECT game_id, shooter_id, COUNT(*) AS attempts
        FROM raw.shots
        WHERE shooter_id IS NOT NULL
          AND event_type IN ('SHOT', 'GOAL', 'MISS', 'BLOCK')
        GROUP BY game_id, shooter_id""")
    team_games = q("SELECT game_id, team, sog FROM raw.team_games")
    return {"skaters": skaters, "games": games, "attempts": attempts,
            "team_games": team_games}


def load_player_features(conn=None) -> pd.DataFrame:
    inp = load_inputs(conn)
    df = build_player_features(inp["skaters"], inp["games"], inp["attempts"],
                               inp["team_games"])
    logger.info(f"Player shot features: {len(df)} skater games, "
                f"{df['player_id'].nunique()} players")
    return df
