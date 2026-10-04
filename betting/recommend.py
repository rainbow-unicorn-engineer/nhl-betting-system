"""
betting/recommend.py
The in-season daily recommendation job (Phase 3, final piece).

Scores an upcoming slate with the production lgbm_market model and runs
every game through betting.engine, writing models.predictions (audit trail
for every scored game) and betting.recommendations (only games clearing the
edge threshold). Invoked by `python pipeline.py recommend` and at the end
of the daily/odds chains.

Pre-game vectors for scheduled games reuse the exact historical builders:
compute_rolling / compute_goalie_rolling are shift-then-roll, so appending
the slate's games as stats-less rows yields each team's (and projected
starter's) rolling window over all completed games strictly before the
slate date — bit-identical to what the historical build would later store
for those games (verified in tests/test_recommend.py).

Starters: confirmed starters arrive with the Daily Faceoff scraper in
Phase 4. Until then the projected starter is the goalie with the most
starts over the team's last 10 completed games (ties -> most recent), and
starter_fallback_{home,away} is set to 1.0 so the model knows the starter
is unconfirmed.

Odds: the fair (no-vig) probability is the MEDIAN across the freshest
snapshot of each book (raw.odds_snapshots, at most MAX_ODDS_AGE_HOURS old);
each side is then priced at the best available price across the books you
can bet (BETTABLE_BOOKS; unset = every book) — line shopping. When no
snapshot exists (e.g. simulation against history) the single reference
line in raw.historical_odds is used for both. Games with no line anywhere
are scored by the market-blind fallback model and never bet (an edge
claimed against no market is untestable).

Frozen picks: a pick is issued ONCE per game, at the price of the
snapshot it came from (priced_at). Later runs never re-price or delete it;
they only add picks for games that have none, within what is left of the
day's exposure budget. The pre-game `pipeline.py close` snapshot then
grades it (betting/settle.py: CLV against the last pre-puck-drop quote).
A pick settled VOID (postponed game) no longer counts as the game's pick,
so the game can get a new one when it is played on its new date. Each
pick stores the game's start time as written (scheduled_start), which
settlement compares with the actual start to spot a moved game.

Exposure caps: cap_daily_exposure() is the one (pure) allocation rule.
Strongest edges first, it keeps a pick only while three limits hold: the
day's budget (MAX_DAILY_PCT of bankroll), and per game at most
MAX_BETS_PER_GAME bets and MAX_GAME_STAKE_PCT of bankroll, counting every
market (bets on one game are correlated → they tend to win or lose
together). It runs once when deciding, then again inside
write_recommendations under the writers' advisory lock, against the
stakes re-read there, so overlapping daily/odds runs can't together pass
a limit. SKIPPED and voided picks (postponed or cancelled games) count
toward none of them. Moneyline issues at most one pick per game, so the
per-game limits guard the totals and props picks still to come.

Totals: models.totals.GATE_PASSED is read once per run. While it is False
(the default) the totals model is predictions-only: its PMFs are stored,
no totals pick is made. If it is True the job logs an ERROR instead of
betting, because no totals betting path exists yet (settlement grades
moneyline picks only).

Simulation: --date <past date> --simulate treats that day's completed
games as an upcoming slate. fit_production gets the same date as cutoff,
so training, features, and odds are all strictly pre-slate — the honest
dress rehearsal for the 2026-27 paper-trading season.
"""
import argparse
import logging
import os
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# The exposure limits (.env-overridable, validated) are read in engine.py
from betting.engine import (MAX_BETS_PER_GAME, MAX_DAILY_PCT,
                            MAX_GAME_STAKE_PCT, decimal_odds, evaluate_market,
                            EDGE_MIN_ML, game_cap_reason)
from config.migrate import ensure_schema
from config.settings import engine as db, local_today

logger = logging.getLogger("nhl.betting.recommend")

BANKROLL = float(os.getenv("BANKROLL", "1000"))
MAX_ODDS_AGE_HOURS = float(os.getenv("MAX_ODDS_AGE_HOURS", "18"))
RECENT_TEAM_GAMES = 10          # starter projection window
# Odds API bookmaker keys you can actually bet, e.g. "draftkings,fanduel".
# The fair price still uses every book; only the best price is restricted.
BETTABLE_BOOKS = frozenset(b.strip().lower() for b in
                           os.getenv("BETTABLE_BOOKS", "").split(",")
                           if b.strip())


# ── Slate ──────────────────────────────────────────────────────────

def load_slate(target_date, simulate: bool = False) -> pd.DataFrame:
    """Games on target_date joined with their matchup rows (schedule/Elo
    features are built for scheduled games by the daily feature build).
    Live mode takes only games that have not started and are not
    postponed, suspended or cancelled; simulate takes the whole day."""
    state_filter = "" if simulate else """
        AND g.game_state NOT IN ('FINAL', 'OFF', 'LIVE', 'CRIT')
        AND COALESCE(g.schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
        AND (g.start_time_utc IS NULL OR g.start_time_utc > NOW())"""
    with db.connect() as conn:
        return pd.read_sql(text(f"""
            SELECT g.game_id, g.season, g.date, g.home_team, g.away_team,
                   g.game_type, g.home_score, g.away_score,
                   m.home_rest_days, m.away_rest_days, m.home_b2b, m.away_b2b,
                   m.home_travel_km, m.away_travel_km,
                   m.home_tz_shift, m.away_tz_shift,
                   m.home_game_num, m.away_game_num, m.season_stage,
                   m.home_elo, m.away_elo
            FROM raw.games g
            JOIN features.matchup m USING (game_id)
            WHERE g.date = :d AND g.game_type IN (2, 3) {state_filter}
            ORDER BY g.game_id
        """), conn, params={"d": target_date})


# ── Pre-game features (as-of the slate date) ───────────────────────

def _team_wide_asof(slate: pd.DataFrame, season: int, target_date) -> pd.DataFrame:
    """team_rolling values as of target_date for the slate's teams, via the
    historical builder with the slate appended as stats-less rows."""
    from features.team_features import compute_rolling, load_base
    from features.build_vectors import TEAM_STATS

    base = load_base(season)
    base = base[base["date"] < target_date]
    synth = pd.concat([
        slate.assign(team=slate["home_team"]),
        slate.assign(team=slate["away_team"]),
    ])[["game_id", "season", "team"]]
    synth = synth.assign(date=target_date).reindex(columns=base.columns)

    feats = compute_rolling(pd.concat([base, synth], ignore_index=True))
    feats = feats[feats["game_id"].isin(slate["game_id"])]
    feats = feats.astype({c: float for c in TEAM_STATS + ["games_played"]})
    wide = feats.pivot(index=["game_id", "team"], columns="window_size",
                       values=TEAM_STATS + ["games_played"])
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


def project_starters(slate: pd.DataFrame, season: int, target_date) -> pd.DataFrame:
    """Starter per (game_id, team). Daily Faceoff rows for the slate date
    win when they resolve to a player id (starter_fallback=0 only when
    DF says 'Confirmed'); teams without one fall back to the heuristic —
    most starts over the team's last RECENT_TEAM_GAMES completed games,
    ties broken by most recent start, always starter_fallback=1."""
    from ingestion.dailyfaceoff import ensure_table
    ensure_table()
    with db.connect() as conn:
        df_rows = pd.read_sql(text("""
            SELECT team, goalie_id, confirmation FROM raw.starting_goalies
            WHERE game_date = :d AND goalie_id IS NOT NULL
        """), conn, params={"d": target_date})
    confirmed = {r.team: (int(r.goalie_id),
                          0 if r.confirmation == "Confirmed" else 1)
                 for r in df_rows.itertuples()}

    teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    with db.connect() as conn:
        starts = pd.read_sql(text("""
            WITH team_games AS (
                SELECT gg.team, gg.player_id, g.date, g.game_id,
                       DENSE_RANK() OVER (PARTITION BY gg.team
                                          ORDER BY g.date DESC, g.game_id DESC
                       ) AS game_rank
                FROM raw.goalie_games gg
                JOIN raw.games g USING (game_id)
                WHERE gg.is_starter AND g.season = :season
                  AND g.date < :d AND gg.team = ANY(:teams)
            )
            SELECT team, player_id,
                   COUNT(*) FILTER (WHERE game_rank <= :recent) AS recent_starts,
                   MAX(date) AS last_start
            FROM team_games
            GROUP BY team, player_id
        """), conn, params={"season": season, "d": target_date,
                            "teams": teams, "recent": RECENT_TEAM_GAMES})

    if starts.empty:
        picks = pd.DataFrame(columns=["team", "goalie_id"])
    else:
        picks = (starts.sort_values(["recent_starts", "last_start"],
                                    ascending=False)
                 .drop_duplicates("team")
                 .rename(columns={"player_id": "goalie_id"})
                 [["team", "goalie_id"]])

    long = pd.concat([
        slate[["game_id"]].assign(team=slate["home_team"].values),
        slate[["game_id"]].assign(team=slate["away_team"].values),
    ], ignore_index=True)
    out = long.merge(picks, on="team", how="left")
    out["starter_fallback"] = 1
    for i, r in out.iterrows():
        if r["team"] in confirmed:
            gid, fb = confirmed[r["team"]]
            out.loc[i, ["goalie_id", "starter_fallback"]] = [gid, fb]
    return out


def _goalie_wide_asof(starters: pd.DataFrame, season: int, target_date) -> pd.DataFrame:
    """goalie_rolling values as of target_date for the projected starters,
    via the historical builder with slate appearances appended stats-less."""
    from features.goalie_features import (DEFAULT_K, compute_goalie_rolling,
                                          league_priors, load_goalie_base)
    from features.build_vectors import GOALIE_STATS
    from features.util import WINDOWS

    empty = pd.DataFrame(columns=["game_id", "goalie_id"] +
                         [f"{s}_w{w}" for w in WINDOWS for s in GOALIE_STATS])
    known = starters.dropna(subset=["goalie_id"])
    if known.empty:
        return empty

    base = load_goalie_base(season)
    base = base[base["date"] < target_date]
    synth = (known.rename(columns={})[["game_id", "goalie_id"]]
             .assign(season=season, date=target_date)
             .reindex(columns=base.columns))

    league_sv, league_gsax60 = league_priors(season)
    feats = compute_goalie_rolling(
        pd.concat([base, synth], ignore_index=True),
        k=DEFAULT_K, league_sv=league_sv, league_gsax60=league_gsax60)
    feats = feats[feats["game_id"].isin(known["game_id"])]
    if feats.empty:
        return empty
    feats = feats.astype({c: float for c in GOALIE_STATS})
    wide = feats.pivot(index=["game_id", "goalie_id"], columns="window_size",
                       values=GOALIE_STATS)
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


# ── Odds ───────────────────────────────────────────────────────────

def load_market(game_ids: list, asof: Optional[datetime] = None,
                max_age_hours: float = MAX_ODDS_AGE_HOURS,
                books: Optional[frozenset] = None) -> pd.DataFrame:
    """One row per game with a line: consensus fair prob + best price per
    side, plus the captured_at of the snapshot each best price came from
    (home_priced_at / away_priced_at; None for the historical line).
    Snapshots first (multi-book, line-shopped), historical_odds as the
    single-book fallback. Games with no line are absent from the result.

    books (default BETTABLE_BOOKS; empty = all): the fair probability is
    the median over EVERY book, but best prices come only from these. A
    game no allowed book prices gets None prices, so it can't be bet. The
    historical reference line (simulation) is not filtered."""
    books = BETTABLE_BOOKS if books is None else books
    asof = asof or datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC
    cutoff = asof - timedelta(hours=max_age_hours)
    with db.connect() as conn:
        snaps = pd.read_sql(text("""
            SELECT DISTINCT ON (game_id, book_name)
                   game_id, book_name, captured_at, home_price, away_price
            FROM raw.odds_snapshots
            WHERE market_type = 'ml' AND game_id = ANY(:ids)
              AND home_price IS NOT NULL AND away_price IS NOT NULL
              AND captured_at BETWEEN :cutoff AND :asof
            ORDER BY game_id, book_name, captured_at DESC
        """), conn, params={"ids": list(map(int, game_ids)),
                            "cutoff": cutoff, "asof": asof})
        hist = pd.read_sql(text("""
            SELECT game_id, provider AS book_name, home_ml AS home_price,
                   away_ml AS away_price
            FROM raw.historical_odds
            WHERE game_id = ANY(:ids)
              AND home_ml IS NOT NULL AND away_ml IS NOT NULL
        """), conn, params={"ids": list(map(int, game_ids))})

    hist = hist[~hist["game_id"].isin(snaps["game_id"])]
    return summarize_market(snaps, hist, books)


def summarize_market(snaps: pd.DataFrame, hist: pd.DataFrame,
                     books: frozenset = frozenset()) -> pd.DataFrame:
    """load_market's pure half (see there). snaps: game_id, book_name,
    captured_at, home_price, away_price; hist: the same minus captured_at."""
    from features.util import american_implied_prob

    snaps = snaps.assign(bettable=(snaps["book_name"].str.lower()
                                   .isin(sorted(books)) if books else True))
    hist = hist.assign(bettable=True)
    parts = [f for f in (snaps, hist) if not f.empty]
    if not parts:
        return pd.DataFrame(columns=[
            "game_id", "fair_home_prob", "n_books",
            "home_price", "home_book", "home_priced_at",
            "away_price", "away_book", "away_priced_at"])
    lines = pd.concat(parts, ignore_index=True)
    if "captured_at" not in lines:
        lines["captured_at"] = None       # historical line only: no snapshot

    ph = lines["home_price"].map(american_implied_prob)
    pa = lines["away_price"].map(american_implied_prob)
    lines["novig_home"] = ph / (ph + pa)

    def _when(best):
        t = best["captured_at"]
        return None if t is None or pd.isna(t) else pd.Timestamp(t).to_pydatetime()

    rows = []
    for gid, g in lines.groupby("game_id"):
        row = {"game_id": gid,
               "fair_home_prob": float(g["novig_home"].median()),
               "n_books": len(g),
               "home_price": None, "home_book": None, "home_priced_at": None,
               "away_price": None, "away_book": None, "away_priced_at": None}
        shop = g[g["bettable"].astype(bool)]
        if not shop.empty:
            best_h = shop.loc[shop["home_price"].map(decimal_odds).idxmax()]
            best_a = shop.loc[shop["away_price"].map(decimal_odds).idxmax()]
            row.update({
                "home_price": int(best_h["home_price"]),
                "home_book": best_h["book_name"],
                "home_priced_at": _when(best_h),
                "away_price": int(best_a["away_price"]),
                "away_book": best_a["book_name"],
                "away_priced_at": _when(best_a),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ── Vector assembly + scoring ──────────────────────────────────────

def build_slate_vectors(slate: pd.DataFrame, target_date,
                        asof: Optional[datetime] = None) -> pd.DataFrame:
    """FEATURE_NAMES-ordered pre-game vectors for the slate (in memory,
    never written to features.game_vector — that table is history only)."""
    from features.build_vectors import FEATURE_NAMES, assemble

    season = int(slate["season"].iloc[0])
    starters = project_starters(slate, season, target_date)
    market = load_market(slate["game_id"].tolist(), asof=asof)
    market = market.rename(columns={"fair_home_prob": "market_home_prob"})

    df = assemble(
        slate,
        _team_wide_asof(slate, season, target_date),
        starters,
        _goalie_wide_asof(starters, season, target_date),
        market=market[["game_id", "market_home_prob"]],
    )
    matrix = df[FEATURE_NAMES].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise ValueError(f"{bad} non-finite elements in slate vectors — "
                         f"refusing to score")
    return df


def score_slate(slate_vectors: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """P(home win) per slate game from a production model trained on all
    labeled games strictly before cutoff_date (None = everything)."""
    from features.build_vectors import FEATURE_NAMES
    from models.lgbm import fit_production, score_production

    prod = fit_production(cutoff_date)
    X = slate_vectors[FEATURE_NAMES].to_numpy(dtype=float)
    out = slate_vectors[["game_id"]].copy()
    out["prob_home"] = score_production(prod, X, FEATURE_NAMES)
    out["market_available"] = slate_vectors["market_available"].values
    return out


# ── Totals PMFs (predictions only — no recommendations) ───────────
#
# The totals model has NOT passed its walk-forward gate (models/totals.py
# STATUS). PMFs (probability mass functions → the chance of each possible
# goal count) are still scored and stored per slate for the bet checker,
# the arbitrage/middle alerts and future props work. The job reads
# models.totals.GATE_PASSED once per run (log_totals_gate): False keeps
# totals predictions-only; True logs an ERROR, because there is no totals
# betting path yet — settlement grades moneyline picks only — so flipping
# the switch alone must not look like totals betting started.


def log_totals_gate() -> None:
    """Read models.totals.GATE_PASSED and log what it means for this run
    (the job calls it once per run). Either way no totals pick is
    written: the job has no totals pick writer and settlement has no
    totals grading. True only turns the line into an ERROR, so a flipped
    switch that changes nothing can't go unnoticed."""
    from models import totals as T
    if T.GATE_PASSED:
        logger.error("models.totals.GATE_PASSED is True, but there is no "
                     "totals betting path yet: this job writes no totals "
                     "picks and settlement grades moneyline picks only. "
                     "Totals stay predictions-only until both are built")
    else:
        logger.info("Totals are predictions-only: the totals model has not "
                    "passed its gate (models.totals.GATE_PASSED is False), "
                    "so no totals bet is recommended")


def score_totals(slate: pd.DataFrame, target_date,
                 cutoff_date=None) -> pd.DataFrame:
    """Total-goals PMFs for the slate via the attack-row totals model
    (drift-corrected for the slate's season, margin-reweighted joint)."""
    from models import totals as T

    season = int(slate["season"].iloc[0])
    starters = project_starters(slate, season, target_date)
    st = starters.set_index(["game_id", "team"])["goalie_id"]
    games = slate.copy()
    games["home_starter_id"] = [
        st.get((g, t)) for g, t in zip(games["game_id"], games["home_team"])]
    games["away_starter_id"] = [
        st.get((g, t)) for g, t in zip(games["game_id"], games["away_team"])]

    team_wide = _team_wide_asof(slate, season, target_date)
    goalie_wide = _goalie_wide_asof(starters, season, target_date)
    Xh, Xa = T.build_attack_matrix(games, team_wide, goalie_wide)

    prod = T.fit_production(cutoff_date)
    # season: the drift correction uses this season's games played so far
    out = T.score_production(prod, Xh, Xa, T.ATTACK_FEATURES, season=season)

    res = slate[["game_id"]].copy()
    res["expected_total"] = out["expected_total"]
    res["pmf_home"] = list(out["pmf_home"])
    res["pmf_away"] = list(out["pmf_away"])
    res["pmf_total"] = list(out["pmf_total"])
    return res


def write_total_predictions(scored: pd.DataFrame, lines: pd.DataFrame) -> int:
    """Upsert one 'total' prediction row per game: PMFs + P(over) at the
    consensus line when one exists."""
    from models.totals import MODEL_NAME as T_NAME, MODEL_VERSION as T_VER
    from models.totals import prob_over

    line_map = dict(zip(lines["game_id"], lines["line"])) if not lines.empty \
        else {}
    with db.begin() as conn:
        model_id = _model_id(conn, T_NAME, T_VER,
                             hint="python -m models.totals")
        for r in scored.itertuples():
            line = line_map.get(r.game_id)
            p_over = None
            if line is not None:
                tp = np.asarray(r.pmf_total)[None, :]
                p_over = round(float(prob_over(tp, [line])[0][0]), 4)
            conn.execute(text("""
                INSERT INTO models.predictions
                    (game_id, model_id, market_type, total_over_prob,
                     total_line, home_goals_pmf, away_goals_pmf)
                VALUES (:g, :m, 'total', :po, :line, :ph, :pa)
                ON CONFLICT (game_id, model_id, market_type) DO UPDATE SET
                    total_over_prob = EXCLUDED.total_over_prob,
                    total_line = EXCLUDED.total_line,
                    home_goals_pmf = EXCLUDED.home_goals_pmf,
                    away_goals_pmf = EXCLUDED.away_goals_pmf,
                    created_at = NOW()
            """), {"g": int(r.game_id), "m": model_id,
                   "po": p_over,
                   "line": float(line) if line is not None else None,
                   "ph": [float(x) for x in r.pmf_home],
                   "pa": [float(x) for x in r.pmf_away]})
    return len(scored)


def load_total_lines(game_ids: list, asof: Optional[datetime] = None,
                     max_age_hours: float = MAX_ODDS_AGE_HOURS) -> pd.DataFrame:
    """Consensus (median) total line per game from fresh snapshots."""
    asof = asof or datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC
    cutoff = asof - timedelta(hours=max_age_hours)
    with db.connect() as conn:
        snaps = pd.read_sql(text("""
            SELECT DISTINCT ON (game_id, book_name)
                   game_id, book_name, line
            FROM raw.odds_snapshots
            WHERE market_type = 'total' AND game_id = ANY(:ids)
              AND line IS NOT NULL
              AND captured_at BETWEEN :cutoff AND :asof
            ORDER BY game_id, book_name, captured_at DESC
        """), conn, params={"ids": list(map(int, game_ids)),
                            "cutoff": cutoff, "asof": asof})
    if snaps.empty:
        return pd.DataFrame(columns=["game_id", "line"])
    return (snaps.groupby("game_id")["line"].median()
            .reset_index())


# ── Persistence ────────────────────────────────────────────────────

def _model_id(conn, name: str = None, version: str = None,
              hint: str = "python -m models.lgbm") -> int:
    if name is None:
        from models.lgbm import MODEL_NAME, MODEL_VERSION
        name, version = MODEL_NAME, MODEL_VERSION
    row = conn.execute(text("""
        SELECT model_id FROM models.model_registry
        WHERE model_name = :n AND version = :v
    """), {"n": name, "v": version}).fetchone()
    if row is None:
        raise RuntimeError(f"{name} {version} not in registry — "
                           f"run `{hint}` first")
    return row[0]


def write_predictions(scored: pd.DataFrame) -> dict:
    """Upsert one ml prediction row per scored game; returns
    {game_id: prediction_id} for linking recommendations."""
    with db.begin() as conn:
        model_id = _model_id(conn)
        ids = {}
        for r in scored.itertuples():
            ids[r.game_id] = conn.execute(text("""
                INSERT INTO models.predictions
                    (game_id, model_id, market_type, home_win_prob, away_win_prob)
                VALUES (:g, :m, 'ml', :ph, :pa)
                ON CONFLICT (game_id, model_id, market_type) DO UPDATE SET
                    home_win_prob = EXCLUDED.home_win_prob,
                    away_win_prob = EXCLUDED.away_win_prob,
                    created_at = NOW()
                RETURNING prediction_id
            """), {"g": int(r.game_id), "m": model_id,
                   "ph": round(float(r.prob_home), 4),
                   "pa": round(1.0 - float(r.prob_home), 4)}).scalar()
    return ids


_ISSUED_SQL = """
    SELECT r.game_id, r.market_type, r.status, r.recommended_stake,
           EXISTS (SELECT 1 FROM betting.placed_bets p
                   WHERE p.rec_id = r.rec_id AND p.result = 'VOID') AS voided
    FROM betting.recommendations r
    JOIN raw.games g USING (game_id)
    WHERE g.date = :d
"""


def load_issued_picks(target_date, conn=None) -> pd.DataFrame:
    """Picks already issued for games on target_date, every market and
    any status, including games that have since started — they still
    count against the day's and their game's exposure limits. voided =
    settled as VOID (postponed or cancelled game). conn: read inside that
    connection's transaction (write_recommendations, under its lock)."""
    if conn is not None:
        return pd.read_sql(text(_ISSUED_SQL), conn, params={"d": target_date})
    with db.connect() as c:
        return pd.read_sql(text(_ISSUED_SQL), c, params={"d": target_date})


def frozen_games(issued: pd.DataFrame) -> set:
    """Games that already have their MONEYLINE pick (load_issued_picks
    rows) and so are never re-decided: every issued ml pick, SKIPPED
    included, except picks settled VOID. A voided pick's game (postponed,
    then played on a new date) is open for a new pick. Picks in other
    markets don't freeze the moneyline; a frame without a market_type
    column is taken as all moneyline."""
    if issued.empty:
        return set()
    if "market_type" in issued:
        issued = issued[issued["market_type"].eq("ml")]
    live = issued[~issued["voided"].eq(True)]          # None/NaN = not voided
    return set(live["game_id"].astype(int))


def _live_picks(issued: pd.DataFrame) -> pd.DataFrame:
    """Issued picks that still carry a stake: all but SKIPPED ones
    (released by hand) and voided ones (the book returned the stake)."""
    return issued[(issued["status"] != "SKIPPED")
                  & ~issued["voided"].fillna(False).astype(bool)]


def committed_stake(issued: pd.DataFrame) -> float:
    """Stake already committed for the day (load_issued_picks rows), every
    market: every issued pick except SKIPPED and voided ones."""
    if issued.empty:
        return 0.0
    return float(_live_picks(issued)["recommended_stake"].astype(float).sum())


def game_exposure(issued: pd.DataFrame) -> tuple:
    """({game_id: bets}, {game_id: stake}) already on each game
    (load_issued_picks rows), every market, counting the same picks as
    committed_stake: all but SKIPPED and voided ones."""
    if issued.empty:
        return {}, {}
    live = _live_picks(issued)
    bets, stakes = {}, {}
    for gid, stake in zip(live["game_id"].astype(int),
                          live["recommended_stake"].astype(float).fillna(0.0)):
        bets[int(gid)] = bets.get(int(gid), 0) + 1
        stakes[int(gid)] = stakes.get(int(gid), 0.0) + float(stake)
    return bets, stakes


def allocate_exposure(recs: list, committed: float, bankroll: float,
                      picked=frozenset(), game_bets: dict = None,
                      game_stakes: dict = None,
                      max_daily_pct: float = MAX_DAILY_PCT,
                      max_bets_per_game: int = None,
                      max_game_stake_pct: float = None) -> list:
    """The exposure caps, pure and market-agnostic. recs: dicts with
    game_id, edge_pct and recommended_stake, from any market. Returns
    [(rec, reason)] strongest edge first, for every rec whose game is not
    in `picked` (games that already have their pick in this market are
    never re-decided): reason is None for a rec to issue, else why it was
    skipped. A rec is kept while all three limits still hold with it:
    - the day: committed + kept stakes <= bankroll * max_daily_pct;
    - its game's bets: game_bets[g] (bets already issued on game g, any
      market, SKIPPED and voided excluded) + kept ones <= max_bets_per_game;
    - its game's stake: game_stakes[g] + kept stakes on g
      <= bankroll * max_game_stake_pct.
    A rec that doesn't fit is skipped, and a smaller one, or one on
    another game, after it can still fit. Per-game defaults: the
    MAX_BETS_PER_GAME / MAX_GAME_STAKE_PCT settings."""
    if max_bets_per_game is None:
        max_bets_per_game = MAX_BETS_PER_GAME
    if max_game_stake_pct is None:
        max_game_stake_pct = MAX_GAME_STAKE_PCT
    bets = {int(g): int(n) for g, n in (game_bets or {}).items()}
    staked = {int(g): float(x) for g, x in (game_stakes or {}).items()}
    budget = bankroll * max_daily_pct
    spent = float(committed)
    out = []
    for r in sorted(recs, key=lambda r: -float(r["edge_pct"])):
        gid = int(r["game_id"])
        if gid in picked:
            continue
        stake = float(r["recommended_stake"])
        if spent + stake > budget + 1e-9:        # 1e-9: float noise only
            out.append((r, f"daily cap: {spent:.2f} of {budget:.2f} "
                           f"already staked"))
            continue
        why = game_cap_reason(stake, bets.get(gid, 0), staked.get(gid, 0.0),
                              bankroll, max_bets_per_game, max_game_stake_pct)
        if why is not None:
            out.append((r, why))
            continue
        spent += stake
        bets[gid] = bets.get(gid, 0) + 1
        staked[gid] = staked.get(gid, 0.0) + stake
        out.append((r, None))
    return out


def cap_daily_exposure(recs: list, committed: float, bankroll: float,
                       picked=frozenset(),
                       max_daily_pct: float = MAX_DAILY_PCT,
                       game_bets: dict = None, game_stakes: dict = None,
                       max_bets_per_game: int = None,
                       max_game_stake_pct: float = None) -> list:
    """The recs to issue under the daily and per-game caps, strongest edge
    first (allocate_exposure without the skip reasons; see there)."""
    return [r for r, why in allocate_exposure(
                recs, committed, bankroll, picked, game_bets, game_stakes,
                max_daily_pct, max_bets_per_game, max_game_stake_pct)
            if why is None]


def write_recommendations(recs: list, slate_game_ids: list, slate_date=None,
                          bankroll: float = BANKROLL) -> int:
    """Insert ml recommendations for slate games that have none yet,
    within the day's exposure budget.
    Picks are frozen: a game with ANY ml recommendation (any status) keeps
    it unchanged at its issued price; nothing is ever deleted or re-priced
    (re-pricing every run is what made same-book CLV read 0). The one
    exception is a pick settled VOID: it no longer blocks a new pick.
    Each row stores the game's start_time_utc as scheduled_start.
    The caps are applied again here, under the writers' lock, against the
    stakes re-read for slate_date (default: the slate games' date), so a
    run that decided before another run's picks landed is trimmed instead
    of pushing the day past MAX_DAILY_PCT, or a game past its per-game
    limits."""
    ids = list(map(int, slate_game_ids))
    with db.begin() as conn:
        # Serialize writers (launchd can fire daily + odds together on wake)
        # so two runs can't both issue a pick for the same game, or both
        # spend the same remaining budget
        conn.execute(text("SELECT pg_advisory_xact_lock(20260928)"))
        picked = {r[0] for r in conn.execute(text("""
            SELECT DISTINCT r.game_id FROM betting.recommendations r
            WHERE r.game_id = ANY(:ids) AND r.market_type = 'ml'
              AND NOT EXISTS (SELECT 1 FROM betting.placed_bets p
                              WHERE p.rec_id = r.rec_id AND p.result = 'VOID')
        """), {"ids": ids})}
        if slate_date is None:
            slate_date = conn.execute(text("""
                SELECT MIN(date) FROM raw.games WHERE game_id = ANY(:ids)
            """), {"ids": ids}).scalar()
        issued = load_issued_picks(slate_date, conn=conn)
        picked |= frozen_games(issued)
        game_bets, game_stakes = game_exposure(issued)
        allowed = cap_daily_exposure(recs, committed_stake(issued), bankroll,
                                     picked, game_bets=game_bets,
                                     game_stakes=game_stakes)
        trimmed = [r for r in recs if r["game_id"] not in picked
                   and not any(r is a for a in allowed)]
        if trimmed:
            logger.info(f"Caps re-checked under the lock: {len(trimmed)} "
                        f"pick(s) dropped, because less of {slate_date}'s "
                        f"budget, or of their games' per-game limits, is "
                        f"left than when they were decided")
        to_insert = [dict(r, priced_at=r.get("priced_at")) for r in allowed]
        if to_insert:
            conn.execute(text("""
                INSERT INTO betting.recommendations
                    (game_id, prediction_id, market_type, side, model_prob,
                     best_book, best_price, implied_prob_novig, edge_pct,
                     kelly_fraction, recommended_stake, status, priced_at,
                     scheduled_start)
                VALUES (:game_id, :prediction_id, 'ml', :side, :model_prob,
                        :best_book, :best_price, :implied_prob_novig,
                        :edge_pct, :kelly_fraction, :recommended_stake,
                        'PENDING', :priced_at,
                        (SELECT start_time_utc FROM raw.games
                         WHERE game_id = :game_id))
            """), to_insert)
    return len(to_insert)


# ── The job ────────────────────────────────────────────────────────

def generate_recommendations(target_date=None, bankroll: float = BANKROLL,
                             edge_min: float = None, dry_run: bool = False,
                             simulate: bool = False,
                             only_games=None) -> pd.DataFrame:
    """Score the slate, decide bets through the engine, persist. Returns
    the frame of NEW recommendations (possibly empty); games that already
    have a pick keep it and are not re-decided.
    only_games: when given (a set of game ids, possibly empty), every
    slate game is still scored and its prediction written, but only these
    games may get a new pick. The news monitor (betting/news.py) passes
    the games whose stored price it could confirm is still the market's."""
    ensure_schema()
    target_date = target_date or local_today()
    if edge_min is None:
        edge_min = float(os.getenv("EDGE_MIN_ML", EDGE_MIN_ML))

    slate = load_slate(target_date, simulate=simulate)
    if slate.empty:
        logger.info(f"No games on {target_date} — nothing to recommend")
        return pd.DataFrame()
    logger.info(f"Slate {target_date}: {len(slate)} games "
                f"(simulate={simulate}, edge_min={edge_min:.1%})")

    # In simulation the "now" of odds freshness is midnight before the
    # slate — snapshots and training data are both strictly pre-slate.
    asof = (datetime.combine(target_date, datetime.min.time())
            if simulate else None)
    vectors = build_slate_vectors(slate, target_date, asof=asof)
    scored = score_slate(vectors, cutoff_date=target_date if simulate else None)

    # Bettable prices: fresh snapshots (or the reference line in simulation)
    market = load_market(slate["game_id"].tolist(), asof=asof)
    logger.info("Best prices from BETTABLE_BOOKS: " + ", ".join(sorted(BETTABLE_BOOKS))
                + " (fair odds still use every book)" if BETTABLE_BOOKS
                else "Best prices from every book (BETTABLE_BOOKS unset)")

    merged = scored.merge(market, on="game_id", how="left").merge(
        slate[["game_id", "home_team", "away_team"]], on="game_id")

    # Frozen picks: games that already have one are not re-decided (a
    # voided pick doesn't count), and their stakes (except SKIPPED and
    # voided) come out of the day's budget
    issued = load_issued_picks(target_date)
    picked = frozen_games(issued)
    committed = committed_stake(issued)
    game_bets, game_stakes = game_exposure(issued)
    if picked:
        logger.info(f"{len(picked)} game(s) on {target_date} already have a "
                    f"pick (kept at its issued price); {committed:.2f} "
                    f"already committed")

    def _price(p):
        return None if p is None or pd.isna(p) else p

    if only_games is not None:
        only_games = {int(g) for g in only_games}
        logger.info(f"Only {len(only_games)} game(s) may get a new pick this "
                    f"run: " + (", ".join(map(str, sorted(only_games))) or "none"))

    candidates = []
    for g in merged.itertuples():
        if pd.isna(g.fair_home_prob):
            continue                      # no line -> never bet
        if only_games is not None and int(g.game_id) not in only_games:
            continue
        d = evaluate_market(g.prob_home, g.fair_home_prob,
                            _price(g.home_price), _price(g.away_price),
                            edge_min)
        if d is None:
            continue
        if d.side == "HOME":
            book, priced_at = g.home_book, _price(g.home_priced_at)
        else:
            book, priced_at = g.away_book, _price(g.away_priced_at)
        # float() casts: psycopg2 cannot adapt numpy scalars
        candidates.append({
            "game_id": int(g.game_id), "prediction_id": None,
            "side": d.side, "model_prob": round(float(d.model_prob), 4),
            "best_book": book, "best_price": int(d.price),
            # the SIDE's no-vig fair probability (d.market_prob); consensus
            # CLV in betting/settle.py compares the close against it
            "implied_prob_novig": round(float(d.market_prob), 4),
            "edge_pct": round(float(d.edge), 4),
            "kelly_fraction": round(float(d.kelly), 4),
            "recommended_stake": float(round(bankroll * d.stake_pct, 2)),
            "priced_at": (pd.Timestamp(priced_at).to_pydatetime()
                          if priced_at is not None else None),
            "matchup": f"{g.away_team} @ {g.home_team}",
        })

    # Daily and per-game caps, strongest edges first; games issued earlier
    # are frozen and never re-decided (checked again under the write lock)
    day_budget = bankroll * MAX_DAILY_PCT
    recs = []
    for r, why in allocate_exposure(candidates, committed, bankroll, picked,
                                    game_bets, game_stakes):
        if why is None:
            recs.append(r)
        else:
            logger.info(f"  skipping {r['matchup']} ({r['side']} "
                        f"{r['best_price']:+d}, edge {r['edge_pct']:.1%}): "
                        f"{why}")
    spent = committed + sum(r["recommended_stake"] for r in recs)

    for r in recs:
        logger.info(f"  BET {r['matchup']}: {r['side']} {r['best_price']:+d} "
                    f"({r['best_book']}) edge {r['edge_pct']:.1%} "
                    f"stake {r['recommended_stake']:.2f}")
    logger.info(f"{len(recs)} new recommendation(s) from {len(slate)} games, "
                f"{spent:.2f} staked of {day_budget:.2f} daily budget")

    if not dry_run:
        pred_ids = write_predictions(scored)
        for r in recs:
            r["prediction_id"] = pred_ids.get(r["game_id"])
        payload = [{k: v for k, v in r.items() if k != "matchup"}
                   for r in recs]
        n = write_recommendations(payload, slate["game_id"].tolist(),
                                  slate_date=target_date, bankroll=bankroll)
        logger.info(f"Wrote {len(pred_ids)} predictions, {n} recommendations")

    # Totals PMFs: predictions only, never recommendations (see
    # log_totals_gate; models/totals.py STATUS has the gate verdict)
    log_totals_gate()
    try:
        t_scored = score_totals(slate, target_date,
                                cutoff_date=target_date if simulate else None)
        logger.info("Expected totals: "
                    + ", ".join(f"{g}:{t:.2f}" for g, t in
                                zip(t_scored['game_id'],
                                    t_scored['expected_total'])))
        if not dry_run:
            t_lines = load_total_lines(slate["game_id"].tolist(), asof=asof)
            write_total_predictions(t_scored, t_lines)
    except Exception as e:
        logger.error(f"Totals scoring failed (non-fatal): {e}")

    return pd.DataFrame(recs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Daily betting recommendations")
    parser.add_argument("--date", type=date_cls.fromisoformat, default=None,
                        help="Slate date YYYY-MM-DD (default: today, local time)")
    parser.add_argument("--simulate", action="store_true",
                        help="Treat a past date's games as an upcoming slate "
                             "(training cutoff = that date)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and decide but write nothing")
    parser.add_argument("--bankroll", type=float, default=BANKROLL)
    parser.add_argument("--edge-min", type=float, default=None)
    args = parser.parse_args()
    generate_recommendations(args.date, bankroll=args.bankroll,
                             edge_min=args.edge_min, dry_run=args.dry_run,
                             simulate=args.simulate)
