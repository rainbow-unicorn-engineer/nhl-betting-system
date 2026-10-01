"""
features/goalie_role.py
Starter-ROLE features for the goalie defending an attack row (experiment,
2026-10): who is this goalie to his team right now, how rested is he, and
how good is his career-carried save percentage compared with the team's
other goalie. Computed in memory with pandas from raw.goalie_games +
raw.games; nothing is written to the database.

Why: features.goalie_rolling resets every season and shrinks with
Buhlmann k=66, so it barely varies (std ~0.004) and measured as net noise
for totals. But in past seasons a team whose BACKUP started (a goalie with
fewer season-to-date starts than the team leader) allowed ~0.11-0.15 more
goals. These features describe the role directly.

Point-in-time rule: every feature for a game on date D uses only rows of
games played on dates STRICTLY BEFORE D (never the game itself, never a
same-day game, never a later game).

Definitions (pre-registered; "appearance" = a goalie_games row with
toi_seconds > 0; a team-game's "starter" = the flagged starter, else the
goalie with the most TOI — the same rule as features/build_vectors.py):
  role_start_share_10    his share of his team's starts over the team's
                         previous 10 games, any season (fewer if fewer were
                         played); NaN if the team has < 3 prior games.
  role_start_share_season his share of the team's starts so far this
                         season; NaN when the team has played 0 games of
                         the season.
  role_is_primary        1 if his season-to-date starts for the team >= the
                         team leader's (ties are all primary). When the team
                         has played 0 games of the season: 1 if his starts
                         for the team in the PREVIOUS season (season code
                         - 10001) >= that season's leader's; NaN if the team
                         has no games in the previous season either.
  role_rest_days         days since his previous appearance (any team,
                         any season), capped at 10; NaN if he has no prior
                         appearance in the data.
  role_started_yesterday 1 if he appeared on the previous calendar day,
                         else 0 (0 with no prior appearance).
  role_carry_sv          shots-weighted save% over his previous 60
                         appearances (across seasons and teams), shrunk
                         toward the league:
                             carry_sv = (S + K * s_bar * p) / (A + K * s_bar)
                         S, A = his saves and shots against summed over
                         those <= 60 appearances; p = league save% (sum of
                         saves / sum of shots against, all appearances) over
                         the 365 days before D (dates in [D-365d, D));
                         s_bar = league mean shots against per appearance
                         over the same window; K = 30, so the prior counts
                         as 30 average appearances' worth of shots. With no
                         league rows in the window (the first date in the
                         data): p = 0.905, s_bar = 28.0.
  role_starter_minus_alt carry_sv minus the carry_sv (same formula, same
                         date D) of the team's OTHER most-used goalie over
                         the team's previous 20 games (most starts; ties go
                         to the most recent starter); 0 if no other goalie
                         started any of those 20 games.
"""
import numpy as np
import pandas as pd
from sqlalchemy import text

ROLE_FEATURES = ["role_start_share_10", "role_start_share_season",
                 "role_is_primary", "role_rest_days",
                 "role_started_yesterday", "role_carry_sv",
                 "role_starter_minus_alt"]

SHARE_GAMES = 10
SHARE_MIN_GAMES = 3
REST_CAP = 10
CARRY_APPEARANCES = 60
CARRY_K = 30.0
LEAGUE_DAYS = 365
LEAGUE_SV_FALLBACK = 0.905
LEAGUE_SHOTS_FALLBACK = 28.0
ALT_GAMES = 20


# ── Loading (read-only) ────────────────────────────────────────────

def load_appearances(conn) -> pd.DataFrame:
    """Every goalie appearance of a completed game: goalie_id, game_id,
    team, date, season, is_starter, toi_seconds, saves, shots_against."""
    df = pd.read_sql(text("""
        SELECT gg.player_id AS goalie_id, gg.game_id, gg.team, g.date,
               g.season, gg.is_starter, gg.toi_seconds,
               gg.saves, gg.shots_against
        FROM raw.goalie_games gg
        JOIN raw.games g USING (game_id)
        WHERE gg.toi_seconds > 0 AND g.game_state IN ('FINAL', 'OFF')
    """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Pure computation ───────────────────────────────────────────────

def team_starters(app: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game_id): date, season and the starter (flagged,
    else most TOI), sorted by team, date, game_id."""
    a = app.sort_values(["game_id", "team", "is_starter", "toi_seconds"],
                        ascending=[True, True, False, False])
    st = a.drop_duplicates(["game_id", "team"])
    st = st.rename(columns={"goalie_id": "starter_id"})
    return (st[["team", "game_id", "date", "season", "starter_id"]]
            .sort_values(["team", "date", "game_id"]).reset_index(drop=True))


class _CarrySV:
    """carry_sv(goalie, date) lookups, point-in-time (strictly before date)."""

    def __init__(self, app: pd.DataFrame):
        a = app.sort_values(["goalie_id", "date", "game_id"])
        self.by_goalie = {}
        for gid, g in a.groupby("goalie_id", sort=False):
            self.by_goalie[gid] = (
                g["date"].to_numpy(dtype="datetime64[ns]"),
                np.concatenate([[0.0], np.cumsum(g["saves"].to_numpy(float))]),
                np.concatenate([[0.0], np.cumsum(g["shots_against"].to_numpy(float))]))
        day = (app.groupby("date")
               .agg(saves=("saves", "sum"), shots=("shots_against", "sum"),
                    n=("saves", "size")).sort_index())
        self.days = day.index.to_numpy(dtype="datetime64[ns]")
        self.c_saves = np.concatenate([[0.0], np.cumsum(day["saves"].to_numpy(float))])
        self.c_shots = np.concatenate([[0.0], np.cumsum(day["shots"].to_numpy(float))])
        self.c_n = np.concatenate([[0.0], np.cumsum(day["n"].to_numpy(float))])

    def league(self, date) -> tuple:
        """(league save%, mean shots per appearance) over [date-365d, date)."""
        d = np.datetime64(pd.Timestamp(date), "ns")
        hi = np.searchsorted(self.days, d, side="left")
        lo = np.searchsorted(self.days, d - np.timedelta64(LEAGUE_DAYS, "D"),
                             side="left")
        shots = self.c_shots[hi] - self.c_shots[lo]
        n = self.c_n[hi] - self.c_n[lo]
        if n <= 0 or shots <= 0:
            return LEAGUE_SV_FALLBACK, LEAGUE_SHOTS_FALLBACK
        return (self.c_saves[hi] - self.c_saves[lo]) / shots, shots / n

    def __call__(self, goalie_id, date) -> float:
        p, s_bar = self.league(date)
        prior = CARRY_K * s_bar
        rec = self.by_goalie.get(goalie_id)
        if rec is None:
            return float(p)
        dates, cs, ca = rec
        i = np.searchsorted(dates, np.datetime64(pd.Timestamp(date), "ns"),
                            side="left")
        j = max(0, i - CARRY_APPEARANCES)
        S, A = cs[i] - cs[j], ca[i] - ca[j]
        return float((S + prior * p) / (A + prior))


def compute_role_features(app: pd.DataFrame,
                          queries: pd.DataFrame) -> pd.DataFrame:
    """ROLE_FEATURES for each query row (game_id, team, goalie_id, date,
    season): the goalie defending for `team` in that game. Uses only
    appearances dated strictly before the query's date. Returns a frame
    aligned to `queries`' index."""
    app = app.copy()
    app["date"] = pd.to_datetime(app["date"])
    tgs = team_starters(app)
    teams = {}
    for team, g in tgs.groupby("team", sort=False):
        teams[team] = (g["date"].to_numpy(dtype="datetime64[ns]"),
                       g["season"].to_numpy(),
                       g["starter_id"].to_numpy())
    goalie_dates = {gid: np.sort(g["date"].to_numpy(dtype="datetime64[ns]"))
                    for gid, g in app.groupby("goalie_id", sort=False)}
    carry = _CarrySV(app)

    out = np.full((len(queries), len(ROLE_FEATURES)), np.nan)
    q = queries.reset_index(drop=True)
    for r, row in enumerate(q.itertuples(index=False)):
        gid = row.goalie_id
        if pd.isna(gid):
            continue
        d = np.datetime64(pd.Timestamp(row.date), "ns")
        season = int(row.season)
        f = {}

        tdates, tseas, tstart = teams.get(
            row.team, (np.array([], "datetime64[ns]"), np.array([]), np.array([])))
        i = np.searchsorted(tdates, d, side="left")      # prior team games
        prior_st = tstart[:i]
        prior_se = tseas[:i]

        last10 = prior_st[-SHARE_GAMES:] if i else prior_st
        f["role_start_share_10"] = (np.mean(last10 == gid)
                                    if i >= SHARE_MIN_GAMES else np.nan)

        this = prior_st[prior_se == season]
        f["role_start_share_season"] = np.mean(this == gid) if len(this) else np.nan

        ref = this if len(this) else prior_st[prior_se == season - 10001]
        if len(ref):
            ids, counts = np.unique(ref, return_counts=True)
            mine = counts[ids == gid].sum()
            f["role_is_primary"] = float(mine >= counts.max())

        gdates = goalie_dates.get(gid, np.array([], "datetime64[ns]"))
        k = np.searchsorted(gdates, d, side="left")
        if k:
            gap = (d - gdates[k - 1]) / np.timedelta64(1, "D")
            f["role_rest_days"] = float(min(gap, REST_CAP))
            f["role_started_yesterday"] = float(gap == 1)
        else:
            f["role_started_yesterday"] = 0.0

        mine_sv = carry(gid, d)
        f["role_carry_sv"] = mine_sv
        last20 = prior_st[-ALT_GAMES:] if i else prior_st
        others = [s for s in last20 if s != gid and not pd.isna(s)]
        if others:
            ids, counts = np.unique(others, return_counts=True)
            best = ids[counts == counts.max()]
            if len(best) > 1:      # tie: the most recent starter among them
                for s in last20[::-1]:
                    if s in set(best):
                        best = [s]
                        break
            f["role_starter_minus_alt"] = mine_sv - carry(best[0], d)
        else:
            f["role_starter_minus_alt"] = 0.0

        out[r] = [f.get(c, np.nan) for c in ROLE_FEATURES]
    return pd.DataFrame(out, columns=ROLE_FEATURES, index=queries.index)


def defending_role_frame(games: pd.DataFrame, app: pd.DataFrame) -> tuple:
    """(home_def, away_def): ROLE_FEATURES of the goalie defending each
    attack row, aligned to `games` rows. home_def describes the AWAY
    starter (who faces the home attack); away_def the HOME starter.
    games needs game_id, date, season, home_team, away_team,
    home_starter_id, away_starter_id."""
    qa = pd.DataFrame({"game_id": games["game_id"].to_numpy(),
                       "team": games["away_team"].to_numpy(),
                       "goalie_id": games["away_starter_id"].to_numpy(),
                       "date": pd.to_datetime(games["date"]).to_numpy(),
                       "season": games["season"].to_numpy()})
    qh = qa.assign(team=games["home_team"].to_numpy(),
                   goalie_id=games["home_starter_id"].to_numpy())
    both = compute_role_features(app, pd.concat([qa, qh], ignore_index=True))
    n = len(games)
    return (both.iloc[:n].reset_index(drop=True),
            both.iloc[n:].reset_index(drop=True))
