"""
features/xg.py
Shot features for our own expected-goals model (models/xg.py), and the
pure helpers that turn shot-level xG into the team, goalie and player
inputs the feature store already uses.

xG (expected goals) → the chance that one shot becomes a goal, judged
only from what is known when the shot is taken: where it came from, what
kind of shot it was, the game situation and what happened just before.
Adding a team's xG over a game measures the quality of its chances, not
its luck in finishing them.

One row per UNBLOCKED shot attempt in raw.shots (event types SHOT, GOAL,
MISS; MoneyPuck's file has no blocked attempts). Every input is a fact
about the shot itself or about EARLIER shots in the same game:

- Location: MoneyPuck's arena-adjusted x and y (→ corrected for rinks
  whose scorers record shots too near or far), its shot distance (feet
  from the net) and shot angle (degrees off the centre line, sign
  dropped).
- Shot type: WRIST, SNAP, SLAP, BACK(hand), TIP, DEFL(ection), WRAP
  (around). About 0.6% of shots have no type, and those are goals twice
  as often as typed shots (a recording quirk, not hockey), so a missing
  type is mapped to WRIST, the most common type, rather than given its
  own category the model could learn the quirk from.
- MoneyPuck's rebound and rush flags (→ a shot within seconds of an
  earlier shot, and a shot soon after the puck left the other end).
- Strength (→ skaters on the ice for each side, shooter first, e.g. 5v4
  is a power play): shooter and defender skater counts; empty net (the
  defending side has 6+ skaters, so its goalie is pulled); own net empty.
- Game state: period (playoff overtimes count as period 4), seconds into
  the period, overtime, playoff game, score difference from the
  shooter's side (raw.shots stores it as home minus away BEFORE the
  shot: checked 2026-10-04, the shot after a home goal reads +1 more),
  shooter's team at home.
- Prior event (derived from raw.shots itself, in time order within the
  game): seconds since the previous unblocked attempt, whether that was
  the same team, how far the puck moved between the two (same period
  only), how fast the shot angle changed (degrees per second; a fast
  change is a goalie forced to move across), whether the previous
  attempt was a goal, its distance, and the shooting team's attempts in
  the previous 10 seconds.

Never an input: the event type (a MISS can't be a goal), is_goal of the
shot itself, MoneyPuck's xG, or anything later in the game. Tests rewrite
and delete later shots and confirm earlier rows' features don't change.
"""
import numpy as np
import pandas as pd

SHOT_TYPES = ["WRIST", "SNAP", "SLAP", "BACK", "TIP", "DEFL", "WRAP"]
SHOT_TYPE_CODE = {t: i for i, t in enumerate(SHOT_TYPES)}
PP_STRENGTHS = ("5v4", "5v3", "4v3")      # as features/team_features.py
HD_XG_THRESHOLD = 0.20                     # as features/goalie_features.py

CORE_FEATURES = [
    "distance", "angle", "x_abs", "y_abs", "shot_type_code",
    "is_rebound", "is_rush", "shooter_skaters", "defender_skaters",
    "empty_net", "own_net_empty", "period", "period_seconds", "is_ot",
    "is_playoff", "score_diff", "shooter_home",
]
PRIOR_FEATURES = [
    "secs_since_prev", "prev_same_team", "prev_dist_moved",
    "prev_angle_rate", "prev_was_goal", "prev_distance",
    "team_att_last10s",
]
ALL_FEATURES = CORE_FEATURES + PRIOR_FEATURES
CATEGORICAL = ["shot_type_code"]

SECS_CAP = 600.0


def _parse_strength(strength: pd.Series) -> tuple:
    parts = strength.fillna("5v5").astype(str).str.extract(r"^(\d+)v(\d+)$")
    a = pd.to_numeric(parts[0], errors="coerce").fillna(5.0)
    b = pd.to_numeric(parts[1], errors="coerce").fillna(5.0)
    return a.clip(3, 6).to_numpy(float), b.clip(3, 6).to_numpy(float)


def shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Pure: the model inputs for every shot (same row order as `shots`).

    `shots` needs: shot_id, game_id, period, time_elapsed, team, home_team,
    game_type, x, y, shot_type, strength, score_state, is_rebound, is_rush,
    shot_distance, shot_angle, is_goal (read ONLY from earlier shots, for
    prev_was_goal)."""
    s = shots.reset_index(drop=True)
    out = pd.DataFrame(index=s.index)
    out["distance"] = s["shot_distance"].astype(float)
    out["angle"] = s["shot_angle"].astype(float).abs()
    out["x_abs"] = s["x"].astype(float).abs()
    out["y_abs"] = s["y"].astype(float).abs()
    st = s["shot_type"].where(s["shot_type"].isin(SHOT_TYPES), "WRIST")
    out["shot_type_code"] = st.map(SHOT_TYPE_CODE).astype(int)
    out["is_rebound"] = s["is_rebound"].fillna(False).astype(float)
    out["is_rush"] = s["is_rush"].fillna(False).astype(float)
    a, b = _parse_strength(s["strength"])
    out["shooter_skaters"] = a
    out["defender_skaters"] = b
    out["empty_net"] = (b >= 6).astype(float)
    out["own_net_empty"] = (a >= 6).astype(float)
    period = s["period"].astype(int)
    out["period"] = period.clip(upper=4).astype(float)
    out["period_seconds"] = (s["time_elapsed"].astype(float)
                             - 1200.0 * (period - 1)).clip(0, 1200)
    out["is_ot"] = (period >= 4).astype(float)
    out["is_playoff"] = (s["game_type"].astype(int) == 3).astype(float)
    home = (s["team"] == s["home_team"]).to_numpy()
    ss = s["score_state"].fillna(0).astype(float).to_numpy()
    out["score_diff"] = np.clip(np.where(home, ss, -ss), -4, 4)
    out["shooter_home"] = home.astype(float)

    # Prior event: earlier shots of the same game only. raw.shots' shot_id
    # follows MoneyPuck's file order, which is time order (checked: time
    # never decreases in shot_id order within a game).
    order = np.lexsort((s["shot_id"].to_numpy(), s["time_elapsed"].to_numpy(),
                        s["game_id"].to_numpy()))
    o = s.iloc[order]
    g = o.groupby("game_id", sort=False)
    t = o["time_elapsed"].astype(float)
    prev_t = g["time_elapsed"].shift(1).astype(float)
    prev_team = g["team"].shift(1)
    prev_period = g["period"].shift(1)
    same_period = (prev_period == o["period"]).to_numpy()
    secs = (t - prev_t).clip(lower=0, upper=SECS_CAP)
    pr = pd.DataFrame(index=o.index)
    pr["secs_since_prev"] = secs
    pr["prev_same_team"] = np.where(prev_team.isna(), np.nan,
                                    (prev_team == o["team"]).astype(float))
    dx = o["x"].astype(float) - g["x"].shift(1).astype(float)
    dy = o["y"].astype(float) - g["y"].shift(1).astype(float)
    pr["prev_dist_moved"] = np.where(same_period, np.hypot(dx, dy), np.nan)
    ang = o["shot_angle"].astype(float)
    prev_ang = g["shot_angle"].shift(1).astype(float)
    pr["prev_angle_rate"] = np.where(
        same_period, (ang - prev_ang).abs() / np.maximum(secs, 1.0), np.nan)
    prev_goal = g["is_goal"].shift(1)
    pr["prev_was_goal"] = np.where(prev_goal.isna(), np.nan,
                                   prev_goal.fillna(False).astype(float))
    pr["prev_distance"] = g["shot_distance"].shift(1).astype(float)
    pr["team_att_last10s"] = _team_recent_attempts(o, 10.0)
    for c in PRIOR_FEATURES:
        out[c] = pr[c].reindex(s.index).to_numpy(dtype=float)
    return out[ALL_FEATURES]


def _team_recent_attempts(o: pd.DataFrame, window: float) -> np.ndarray:
    """For each shot (rows sorted by game, time), the shooting team's
    earlier attempts in the same game within `window` seconds before it
    (same-second attempts listed earlier count; later ones never do)."""
    res = np.zeros(len(o))
    pos = np.arange(len(o))
    key = o["game_id"].astype(str) + "|" + o["team"].astype(str)
    t = o["time_elapsed"].to_numpy(float)
    for _, idx in pd.Series(pos).groupby(key.to_numpy()).groups.items():
        idx = np.asarray(idx)
        tt = t[idx]
        lo = np.searchsorted(tt, tt - window, side="left")
        res[idx] = np.arange(len(idx)) - lo
    return res


# ── Shot-level xG -> the feature store's inputs (pure) ─────────────

def team_xg_sums(shots: pd.DataFrame, xg_col: str) -> pd.DataFrame:
    """Per (game_id, team): xg = sum of xG over the team's unblocked
    attempts, pp_xg = the same on its power plays (NaN when it had no
    power-play attempt, as SQL's SUM ... FILTER returns NULL)."""
    s = shots[["game_id", "team", "strength", xg_col]].copy()
    s["pp"] = np.where(s["strength"].isin(PP_STRENGTHS), s[xg_col], np.nan)
    agg = s.groupby(["game_id", "team"]).agg(
        xg=(xg_col, "sum"), pp_xg=("pp", lambda v: v.sum(min_count=1)))
    return agg.reset_index()


def goalie_xg_sums(shots: pd.DataFrame, xg_col: str,
                   hd: float = HD_XG_THRESHOLD) -> pd.DataFrame:
    """Per (game_id, goalie_id): xga_shots, hd_att, hd_goals exactly as
    features/goalie_features.py's SQL computes them, from any xG column."""
    s = shots[["game_id", "goalie_id", "event_type", "is_goal", xg_col]].copy()
    s = s[s["goalie_id"].notna()]
    on_target = s["event_type"].isin(["SHOT", "GOAL"])
    hd_mask = on_target & (s[xg_col] >= hd)
    s["hd"] = hd_mask.astype(int)
    s["hd_goal"] = (hd_mask & s["is_goal"].astype(bool)).astype(int)
    agg = s.groupby(["game_id", "goalie_id"]).agg(
        xga_shots=(xg_col, "sum"), hd_att=("hd", "sum"),
        hd_goals=("hd_goal", "sum"))
    return agg.reset_index()


def gsax60_prior(appearances: pd.DataFrame, shots: pd.DataFrame,
                 xg_col: str) -> float | None:
    """The league GSAx/60 prior exactly as goalie_features.league_priors
    computes it in SQL: appearances (game_id, player_id, toi_seconds;
    toi > 0) LEFT JOIN their shots faced, then
    3600 * SUM(xg - goal) / SUM(toi) over the JOINED rows. (The SQL counts
    each appearance's TOI once per shot it faced; replicated as is so the
    MoneyPuck arm reproduces the stored features.) None when no shots."""
    a = appearances[["game_id", "player_id", "toi_seconds"]]
    s = shots[["game_id", "goalie_id", "is_goal", xg_col]].rename(
        columns={"goalie_id": "player_id"})
    j = a.merge(s, on=["game_id", "player_id"], how="left")
    num = (j[xg_col] - j["is_goal"].astype(float)).sum(min_count=1)
    den = j["toi_seconds"].sum()
    if pd.isna(num) or not den:
        return None
    return float(3600.0 * num / den)


# ── Player xG features for the props test (pure) ───────────────────

PROPS_XG_FEATURES = ["ixg60_l10_rel", "ixg60_l20_rel", "ixg60_season_rel",
                     "xg_per_att_l20_rel", "opp_xga_l20_rel"]


def player_xg_features(frame: pd.DataFrame, shots: pd.DataFrame,
                       xg_col: str) -> pd.DataFrame:
    """Adds PROPS_XG_FEATURES to a features.player_shots frame, every value
    from games strictly BEFORE the row's date (shift-then-roll):
    - ixg60_l{10,20}_rel, ixg60_season_rel: his own xG per 60 minutes over
      his last 10 / 20 appearances and season to date, divided by his
      position's trailing-365-day league xG per 60 (dates strictly before);
    - xg_per_att_l20_rel: his xG per unblocked attempt over his last 20
      appearances (shot quality), divided by the trailing league xG per
      attempt;
    - opp_xga_l20_rel: the opponent's xG allowed per game over its last 20
      games, divided by the trailing league xG per team game.
    A game with no shot data at all is UNKNOWN (left out of the sums), not
    zero, as the props frame treats attempts."""
    from features.player_shots import _prev_cumsum, _prev_roll, _ratio60, trailing_sums

    df = frame.copy()
    df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
    covered = set(shots["game_id"].unique())
    ix = (shots.dropna(subset=["shooter_id"])
          .groupby(["game_id", "shooter_id"])[xg_col].sum()
          .rename("_ixg").reset_index()
          .rename(columns={"shooter_id": "player_id"}))
    ix["player_id"] = ix["player_id"].astype(df["player_id"].dtype)
    df = df.merge(ix, on=["game_id", "player_id"], how="left")
    known = df["game_id"].isin(covered).to_numpy()
    df["_known"] = known.astype(float)
    df["_ixg0"] = np.where(known, df["_ixg"].fillna(0.0), 0.0)
    df["_ktoi"] = df["toi_seconds"].astype(float) * df["_known"]
    df["_katt"] = np.where(known, df["att"].fillna(0.0), 0.0)

    df = df.sort_values(["player_id", "_date", "game_id"])
    lg_pos = trailing_sums(df["_date"], df["pos_group"],
                           {"x": df["_ixg0"], "t": df["_ktoi"]})
    league_pos = _ratio60(lg_pos["x"], lg_pos["t"])
    lg = trailing_sums(df["_date"], np.zeros(len(df)),
                       {"x": df["_ixg0"], "a": df["_katt"]})
    with np.errstate(divide="ignore", invalid="ignore"):
        league_qual = np.where(lg["a"] > 0, lg["x"] / lg["a"], np.nan)
    key = "player_id"
    for n in (10, 20):
        x = _prev_roll(df, key, "_ixg0", n)
        t = _prev_roll(df, key, "_ktoi", n)
        df[f"ixg60_l{n}_rel"] = _ratio60(x, t) / league_pos
    pkey = ["player_id", "season"]
    df["ixg60_season_rel"] = _ratio60(_prev_cumsum(df, pkey, "_ixg0"),
                                      _prev_cumsum(df, pkey, "_ktoi")) / league_pos
    x20 = _prev_roll(df, key, "_ixg0", 20).to_numpy()
    a20 = _prev_roll(df, key, "_katt", 20).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["xg_per_att_l20_rel"] = np.where(a20 > 0, x20 / a20, np.nan) / league_qual

    # opponent's xG against per game, team games strictly before
    tsum = team_xg_sums(shots, xg_col)[["game_id", "team", "xg"]]
    games = (df[["game_id", "_date", "team", "opp"]]
             .drop_duplicates(["game_id", "team"]))
    opp_side = games.rename(columns={"team": "_t", "opp": "team"})[
        ["game_id", "_date", "team", "_t"]]
    tg = opp_side.merge(tsum.rename(columns={"team": "_t", "xg": "xga"}),
                        on=["game_id", "_t"], how="left")
    tg["_known"] = tg["game_id"].isin(covered).astype(float)
    tg["xga0"] = tg["xga"].fillna(0.0) * tg["_known"]
    tg = tg.drop_duplicates(["game_id", "team"]).sort_values(
        ["team", "_date", "game_id"]).reset_index(drop=True)
    tl = trailing_sums(tg["_date"], np.zeros(len(tg)),
                       {"x": tg["xga0"], "n": tg["_known"]})
    with np.errstate(divide="ignore", invalid="ignore"):
        league_pg = np.where(tl["n"] > 0, tl["x"] / tl["n"], np.nan)
    xs = _prev_roll(tg, "team", "xga0", 20).to_numpy()
    ns = _prev_roll(tg, "team", "_known", 20).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tg["opp_xga_l20_rel"] = np.where(ns > 0, xs / ns, np.nan) / league_pg
    df = df.merge(tg[["game_id", "team", "opp_xga_l20_rel"]].rename(
        columns={"team": "opp"}), on=["game_id", "opp"], how="left")

    df = df.drop(columns=["_date", "_ixg", "_known", "_ixg0", "_ktoi", "_katt"])
    df[PROPS_XG_FEATURES] = df[PROPS_XG_FEATURES].replace([np.inf, -np.inf], np.nan)
    return (df.sort_values(["date", "game_id", "player_id"])
            .reset_index(drop=True))


# ── Loading (SELECT only) ──────────────────────────────────────────

SHOTS_SQL = """
SELECT s.shot_id, s.game_id, s.season, g.date, g.game_type, g.home_team,
       s.period, s.time_elapsed, s.team, s.shooter_id, s.goalie_id,
       s.x::float AS x, s.y::float AS y, s.shot_type, s.event_type,
       s.is_goal, s.xg_moneypuck::float AS xg_moneypuck, s.strength,
       s.score_state, s.is_rebound, s.is_rush,
       s.shot_distance::float AS shot_distance,
       s.shot_angle::float AS shot_angle
FROM raw.shots s
JOIN raw.games g USING (game_id)
WHERE s.event_type IN ('SHOT', 'GOAL', 'MISS')
ORDER BY s.game_id, s.time_elapsed, s.shot_id
"""


def load_shots(conn=None) -> pd.DataFrame:
    """Every unblocked attempt with its game context, read with SELECT."""
    from sqlalchemy import text

    from config.settings import engine

    if conn is None:
        with engine.connect() as c:
            return load_shots(c)
    df = pd.read_sql(text(SHOTS_SQL), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df
