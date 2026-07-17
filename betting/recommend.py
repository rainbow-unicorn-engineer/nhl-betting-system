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
each side is then priced at the best available price across those books
(line shopping). When no snapshot exists (e.g. simulation against history)
the single reference line in raw.historical_odds is used for both. Games
with no line anywhere are scored by the market-blind fallback model and
never bet (an edge claimed against no market is untestable).

Simulation: --date <past date> --simulate treats that day's completed
games as an upcoming slate. fit_production gets the same date as cutoff,
so training, features, and odds are all strictly pre-slate — the honest
dress rehearsal for the 2026-27 paper-trading season.
"""
import argparse
import logging
import os
from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from betting.engine import (MAX_DAILY_PCT, decimal_odds, evaluate_market,
                            EDGE_MIN_ML)
from config.settings import engine as db

logger = logging.getLogger("nhl.betting.recommend")

BANKROLL = float(os.getenv("BANKROLL", "1000"))
MAX_ODDS_AGE_HOURS = float(os.getenv("MAX_ODDS_AGE_HOURS", "18"))
RECENT_TEAM_GAMES = 10          # starter projection window


# ── Slate ──────────────────────────────────────────────────────────

def load_slate(target_date, simulate: bool = False) -> pd.DataFrame:
    """Games on target_date joined with their matchup rows (schedule/Elo
    features are built for scheduled games by the daily feature build).
    Live mode takes only not-yet-final games; simulate takes the whole day."""
    state_filter = "" if simulate else \
        "AND g.game_state NOT IN ('FINAL', 'OFF')"
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
    """Projected starter per (game_id, team): most starts over the team's
    last RECENT_TEAM_GAMES completed games, ties broken by most recent
    start. starter_fallback=1 marks the projection as unconfirmed."""
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
    out["starter_fallback"] = 1  # unconfirmed until Daily Faceoff lands
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
                max_age_hours: float = MAX_ODDS_AGE_HOURS) -> pd.DataFrame:
    """One row per game with a line: consensus fair prob + best price per
    side. Snapshots first (multi-book, line-shopped), historical_odds as
    the single-book fallback. Games with no line are absent from the result."""
    from features.util import american_implied_prob

    asof = asof or datetime.utcnow()
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
    lines = pd.concat([snaps, hist], ignore_index=True)
    if lines.empty:
        return pd.DataFrame(columns=[
            "game_id", "fair_home_prob", "n_books",
            "home_price", "home_book", "away_price", "away_book"])

    ph = lines["home_price"].map(american_implied_prob)
    pa = lines["away_price"].map(american_implied_prob)
    lines["novig_home"] = ph / (ph + pa)

    rows = []
    for gid, g in lines.groupby("game_id"):
        best_h = g.loc[g["home_price"].map(decimal_odds).idxmax()]
        best_a = g.loc[g["away_price"].map(decimal_odds).idxmax()]
        rows.append({
            "game_id": gid,
            "fair_home_prob": float(g["novig_home"].median()),
            "n_books": len(g),
            "home_price": int(best_h["home_price"]),
            "home_book": best_h["book_name"],
            "away_price": int(best_a["away_price"]),
            "away_book": best_a["book_name"],
        })
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
# The totals model has NOT passed its walk-forward gate (it neither beats
# its environment baseline nor the market line; models/totals.py STATUS).
# PMFs are still scored and persisted per slate for dashboard visibility
# and future props work, but no totals bet is ever recommended until the
# gate passes with live O/U prices.

def score_totals(slate: pd.DataFrame, target_date,
                 cutoff_date=None) -> pd.DataFrame:
    """Total-goals PMFs for the slate via the attack-row totals model."""
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
    out = T.score_production(prod, Xh, Xa, T.ATTACK_FEATURES)

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
    asof = asof or datetime.utcnow()
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


def write_recommendations(recs: list, slate_game_ids: list) -> int:
    """Replace PENDING ml recommendations for the slate. Rows already
    APPROVED/PLACED/SKIPPED are the user's decisions — left untouched, and
    their games are not re-recommended."""
    with db.begin() as conn:
        decided = {r[0] for r in conn.execute(text("""
            SELECT DISTINCT game_id FROM betting.recommendations
            WHERE game_id = ANY(:ids) AND market_type = 'ml'
              AND status <> 'PENDING'
        """), {"ids": list(map(int, slate_game_ids))})}
        conn.execute(text("""
            DELETE FROM betting.recommendations
            WHERE game_id = ANY(:ids) AND market_type = 'ml'
              AND status = 'PENDING'
        """), {"ids": list(map(int, slate_game_ids))})
        to_insert = [r for r in recs if r["game_id"] not in decided]
        if to_insert:
            conn.execute(text("""
                INSERT INTO betting.recommendations
                    (game_id, prediction_id, market_type, side, model_prob,
                     best_book, best_price, implied_prob_novig, edge_pct,
                     kelly_fraction, recommended_stake, status)
                VALUES (:game_id, :prediction_id, 'ml', :side, :model_prob,
                        :best_book, :best_price, :implied_prob_novig,
                        :edge_pct, :kelly_fraction, :recommended_stake,
                        'PENDING')
            """), to_insert)
    return len(to_insert)


# ── The job ────────────────────────────────────────────────────────

def generate_recommendations(target_date=None, bankroll: float = BANKROLL,
                             edge_min: float = None, dry_run: bool = False,
                             simulate: bool = False) -> pd.DataFrame:
    """Score the slate, decide bets through the engine, persist. Returns
    the recommendation frame (possibly empty)."""
    target_date = target_date or date_cls.today()
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

    merged = scored.merge(market, on="game_id", how="left").merge(
        slate[["game_id", "home_team", "away_team"]], on="game_id")

    candidates = []
    for g in merged.itertuples():
        if pd.isna(g.fair_home_prob):
            continue                      # no line -> never bet
        d = evaluate_market(g.prob_home, g.fair_home_prob,
                            g.home_price, g.away_price, edge_min)
        if d is None:
            continue
        book = g.home_book if d.side == "HOME" else g.away_book
        candidates.append((g, d, book))

    # Daily exposure cap: strongest edges first
    candidates.sort(key=lambda c: -c[1].edge)
    day_budget = bankroll * MAX_DAILY_PCT
    recs, spent = [], 0.0
    for g, d, book in candidates:
        stake = round(bankroll * d.stake_pct, 2)
        if spent + stake > day_budget:
            logger.info(f"  daily cap: skipping {g.away_team}@{g.home_team} "
                        f"({d.side} {d.price:+d}, edge {d.edge:.1%})")
            continue
        spent += stake
        # float() casts: psycopg2 cannot adapt numpy scalars
        recs.append({
            "game_id": int(g.game_id), "prediction_id": None,
            "side": d.side, "model_prob": round(float(d.model_prob), 4),
            "best_book": book, "best_price": int(d.price),
            "implied_prob_novig": round(float(d.market_prob), 4),
            "edge_pct": round(float(d.edge), 4),
            "kelly_fraction": round(float(d.kelly), 4),
            "recommended_stake": float(stake),
            "matchup": f"{g.away_team} @ {g.home_team}",
        })

    for r in recs:
        logger.info(f"  BET {r['matchup']}: {r['side']} {r['best_price']:+d} "
                    f"({r['best_book']}) edge {r['edge_pct']:.1%} "
                    f"stake {r['recommended_stake']:.2f}")
    logger.info(f"{len(recs)} recommendation(s) from {len(slate)} games, "
                f"{spent:.2f} staked of {day_budget:.2f} daily budget")

    if not dry_run:
        pred_ids = write_predictions(scored)
        for r in recs:
            r["prediction_id"] = pred_ids.get(r["game_id"])
        payload = [{k: v for k, v in r.items() if k != "matchup"}
                   for r in recs]
        n = write_recommendations(payload, slate["game_id"].tolist())
        logger.info(f"Wrote {len(pred_ids)} predictions, {n} recommendations")

    # Totals PMFs: predictions only, never recommendations — the totals
    # model has not passed its gate (see models/totals.py STATUS)
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
                        help="Slate date YYYY-MM-DD (default today)")
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
