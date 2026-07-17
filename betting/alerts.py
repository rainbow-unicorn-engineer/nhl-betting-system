"""
betting/alerts.py
Arbitrage + middling alerts over raw.odds_snapshots (Phase 3).

Runs after every odds snapshot (pipeline.py odds chain). Uses only the
FRESHEST quote per (game, book, market, line) within MAX_AGE_MINUTES —
a stale side is how books bait arb traps, so age is enforced, and the
snapshot cadence itself (launchd, near game time) bounds staleness.

Arbitrage (2-way): for each (game, market, line), take the best price per
side across books. If 1/dec_a + 1/dec_b < 1 the pair locks a profit of
1/(1/dec_a + 1/dec_b) - 1 with stakes proportional to implied
probabilities. Lines must MATCH for pl/total pairs (an over 5.5 against
an under 6.5 is not an arb — it's a middle candidate).

Middles (totals): pair the best OVER at a low line with the best UNDER at
a higher line across books. Both bets win when the total lands strictly
between the lines (and pushes refund at integer boundaries). When the
daily job has stored total PMFs for the slate, the alert carries the
EXACT model EV of the middle (sum over the total distribution of the
per-outcome net); without a PMF it reports the conservative breakeven
middle probability p* = max_loss / (middle_net + max_loss), which assumes
every non-middle outcome is the worst one.

Alerts are logged (WARNING for arbs — they are perishable) and returned
as DataFrames for the dashboard. Set ALERTS_NOTIFY=1 for a macOS
notification on arb hits.
"""
import logging
import os
import subprocess
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from betting.engine import decimal_odds
from config.settings import engine as db

logger = logging.getLogger("nhl.betting.alerts")

MAX_AGE_MINUTES = float(os.getenv("ALERTS_MAX_AGE_MINUTES", "30"))
MIN_ARB_PROFIT = float(os.getenv("MIN_ARB_PROFIT", "0.001"))   # 0.1%
MIN_MIDDLE_EV = 0.0            # report any +EV middle when a PMF exists


# ── Loading ────────────────────────────────────────────────────────

def load_latest_lines(asof: Optional[datetime] = None,
                      max_age_minutes: float = MAX_AGE_MINUTES) -> pd.DataFrame:
    """Freshest quote per (game, book, market, line) for upcoming games."""
    asof = asof or datetime.utcnow()
    cutoff = asof - timedelta(minutes=max_age_minutes)
    with db.connect() as conn:
        return pd.read_sql(text("""
            SELECT DISTINCT ON (o.game_id, o.book_name, o.market_type, o.line)
                   o.game_id, o.book_name, o.market_type, o.line,
                   o.home_price, o.away_price, o.over_price, o.under_price,
                   o.captured_at, g.home_team, g.away_team, g.date
            FROM raw.odds_snapshots o
            JOIN raw.games g USING (game_id)
            WHERE o.captured_at BETWEEN :cutoff AND :asof
              AND g.game_state NOT IN ('FINAL', 'OFF')
            ORDER BY o.game_id, o.book_name, o.market_type, o.line,
                     o.captured_at DESC
        """), conn, params={"cutoff": cutoff, "asof": asof})


def load_total_pmfs(game_ids: list) -> dict:
    """{game_id: total_pmf row} from the daily job's stored predictions."""
    from models.totals import MODEL_NAME, total_pmf
    if not game_ids:
        return {}
    with db.connect() as conn:
        rows = pd.read_sql(text("""
            SELECT DISTINCT ON (p.game_id)
                   p.game_id, p.home_goals_pmf, p.away_goals_pmf
            FROM models.predictions p
            JOIN models.model_registry r USING (model_id)
            WHERE p.game_id = ANY(:ids) AND p.market_type = 'total'
              AND r.model_name = :name
            ORDER BY p.game_id, p.created_at DESC
        """), conn, params={"ids": list(map(int, game_ids)),
                            "name": MODEL_NAME})
    return {int(r.game_id): total_pmf(
                np.asarray([r.home_goals_pmf], dtype=float),
                np.asarray([r.away_goals_pmf], dtype=float))[0]
            for r in rows.itertuples() if r.home_goals_pmf is not None}


# ── Arbitrage (pure) ───────────────────────────────────────────────

def arb_stakes(dec_a: float, dec_b: float, total_stake: float = 1.0) -> tuple:
    """(stake_a, stake_b, profit_pct). Stakes equalize the return on both
    outcomes; profit_pct is the locked return on total_stake."""
    ia, ib = 1.0 / dec_a, 1.0 / dec_b
    s = ia + ib
    return (total_stake * ia / s, total_stake * ib / s, 1.0 / s - 1.0)


def find_arbs(lines: pd.DataFrame,
              min_profit: float = MIN_ARB_PROFIT) -> pd.DataFrame:
    """Two-way arbs per (game, market, line): best price per side across
    books. Sides with missing prices are skipped; pl/total pairs only
    match at identical lines."""
    sides = {"ml": ("home_price", "away_price"),
             "pl": ("home_price", "away_price"),
             "total": ("over_price", "under_price")}
    rows = []
    key = lines["line"].fillna(-999.0)
    for (gid, market, _line), g in lines.groupby(
            ["game_id", "market_type", key]):
        if market not in sides:
            continue
        col_a, col_b = sides[market]
        a = g.dropna(subset=[col_a])
        b = g.dropna(subset=[col_b])
        if a.empty or b.empty:
            continue
        best_a = a.loc[a[col_a].map(decimal_odds).idxmax()]
        best_b = b.loc[b[col_b].map(decimal_odds).idxmax()]
        dec_a = decimal_odds(best_a[col_a])
        dec_b = decimal_odds(best_b[col_b])
        stake_a, stake_b, profit = arb_stakes(dec_a, dec_b)
        if profit < min_profit:
            continue
        rows.append({
            "game_id": gid, "market": market,
            "matchup": f"{best_a['away_team']} @ {best_a['home_team']}",
            "line": best_a["line"],
            "side_a": col_a.split("_")[0].upper(),
            "book_a": best_a["book_name"], "price_a": int(best_a[col_a]),
            "side_b": col_b.split("_")[0].upper(),
            "book_b": best_b["book_name"], "price_b": int(best_b[col_b]),
            "stake_a": round(stake_a, 4), "stake_b": round(stake_b, 4),
            "profit_pct": round(profit, 5),
        })
    return pd.DataFrame(rows).sort_values("profit_pct", ascending=False) \
        if rows else pd.DataFrame()


# ── Middles (pure) ─────────────────────────────────────────────────

def middle_ev(tpmf: np.ndarray, low: float, high: float,
              dec_over: float, dec_under: float) -> float:
    """Exact EV per 2 units (1 on each side) of over@low + under@high
    under a total-goals PMF. Pushes refund the pushed side's stake."""
    totals = np.arange(len(tpmf))
    over_ret = np.where(totals > low, dec_over,
                        np.where(totals == low, 1.0, 0.0))
    under_ret = np.where(totals < high, dec_under,
                         np.where(totals == high, 1.0, 0.0))
    return float((tpmf * (over_ret + under_ret)).sum() - 2.0)


def middle_breakeven(dec_over: float, dec_under: float) -> float:
    """Conservative breakeven P(middle): assume every non-middle outcome
    pays the WORSE single-winner return."""
    middle_net = dec_over + dec_under - 2.0
    worst_net = min(dec_over, dec_under) - 2.0
    return -worst_net / (middle_net - worst_net)


def find_middles(lines: pd.DataFrame, pmfs: dict = None) -> pd.DataFrame:
    """Totals middles per game: best over at each line x best under at
    every HIGHER line. EV from the stored PMF when available, else the
    conservative breakeven middle probability."""
    pmfs = pmfs or {}
    rows = []
    totals = lines[lines["market_type"] == "total"].dropna(subset=["line"])
    for gid, g in totals.groupby("game_id"):
        overs = g.dropna(subset=["over_price"]).copy()
        unders = g.dropna(subset=["under_price"])
        if overs.empty or unders.empty:
            continue
        overs["dec"] = overs["over_price"].map(decimal_odds)
        best_over = overs.loc[overs.groupby("line")["dec"].idxmax()]
        for o in best_over.itertuples():
            cands = unders[unders["line"] > o.line]
            if cands.empty:
                continue
            for lu, gu in cands.groupby("line"):
                u = gu.loc[gu["under_price"].map(decimal_odds).idxmax()]
                d_o, d_u = decimal_odds(o.over_price), decimal_odds(u["under_price"])
                row = {
                    "game_id": gid,
                    "matchup": f"{o.away_team} @ {o.home_team}",
                    "over_line": float(o.line), "over_book": o.book_name,
                    "over_price": int(o.over_price),
                    "under_line": float(lu), "under_book": u["book_name"],
                    "under_price": int(u["under_price"]),
                    "max_loss": round(2.0 - min(d_o, d_u), 4),
                    "middle_net": round(d_o + d_u - 2.0, 4),
                    "breakeven_p_middle": round(middle_breakeven(d_o, d_u), 4),
                }
                tpmf = pmfs.get(int(gid))
                if tpmf is not None:
                    row["model_ev"] = round(
                        middle_ev(tpmf, o.line, lu, d_o, d_u), 4)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model_ev" in df.columns:
        keep = df["model_ev"].fillna(-1) > MIN_MIDDLE_EV
        # without a PMF, surface only wide, cheap middles (>= 1 goal
        # window and breakeven under the ~25% a one-goal NHL middle hits)
        keep |= (df["model_ev"].isna()
                 & (df["under_line"] - df["over_line"] >= 1.0)
                 & (df["breakeven_p_middle"] < 0.25))
        df = df[keep]
        return df.sort_values("model_ev", ascending=False, na_position="last")
    return df[(df["under_line"] - df["over_line"] >= 1.0)
              & (df["breakeven_p_middle"] < 0.25)]


# ── The job ────────────────────────────────────────────────────────

def _notify(title: str, body: str) -> None:
    if os.getenv("ALERTS_NOTIFY") != "1":
        return
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "{title}"'],
            timeout=5, capture_output=True)
    except Exception:                      # notification is best-effort
        pass


def run_alerts(asof: Optional[datetime] = None) -> dict:
    """Scan the freshest snapshot lines; log + return arb/middle frames."""
    lines = load_latest_lines(asof=asof)
    if lines.empty:
        logger.info("No fresh odds snapshots to scan")
        return {"arbs": pd.DataFrame(), "middles": pd.DataFrame()}

    arbs = find_arbs(lines)
    pmfs = load_total_pmfs(lines["game_id"].unique().tolist())
    middles = find_middles(lines, pmfs)

    for a in (arbs.itertuples() if not arbs.empty else []):
        msg = (f"ARB {a.matchup} {a.market}"
               + (f" line {a.line}" if pd.notna(a.line) else "")
               + f": {a.side_a} {a.price_a:+d} ({a.book_a}) / "
                 f"{a.side_b} {a.price_b:+d} ({a.book_b}) "
                 f"-> {a.profit_pct:.2%} locked "
                 f"(stakes {a.stake_a:.3f}/{a.stake_b:.3f})")
        logger.warning(msg)
        _notify("NHL arb", msg)
    for m in (middles.itertuples() if not middles.empty else []):
        ev = getattr(m, "model_ev", None)
        logger.warning(
            f"MIDDLE {m.matchup}: O{m.over_line} {m.over_price:+d} "
            f"({m.over_book}) / U{m.under_line} {m.under_price:+d} "
            f"({m.under_book}) max-loss {m.max_loss:.3f}u "
            + (f"model EV {ev:+.3f}u" if ev is not None and pd.notna(ev)
               else f"breakeven P(mid) {m.breakeven_p_middle:.1%}"))

    logger.info(f"Alert scan: {len(lines)} quotes, {len(arbs)} arbs, "
                f"{len(middles)} middles")
    return {"arbs": arbs, "middles": middles}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_alerts()
