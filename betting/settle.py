"""
betting/settle.py
Paper-trading settlement + CLV grading + bankroll ledger (Phase 4 #1).

Paper trading is the project's gate to real money (PROJECT_CONTEXT §2.6:
500+ paper bets, CLV as the north-star). This module closes the loop the
recommendation job opened:

- Every moneyline recommendation is treated as an auto-placed PAPER bet
  at its recommended book/price/stake the moment it is written. When the
  game goes FINAL, a betting.placed_bets row (is_paper = TRUE) is created
  with the result and P&L settled by the SAME engine function the
  backtest uses, and the recommendation is marked SETTLED so slate
  re-scores can never touch it.
- Voids: a pick on a game that is postponed (schedule_state PPD) or
  cancelled (CNCL), or whose game finally started more than VOID_SHIFT
  (3 hours) away from the start it had when the pick was written
  (recommendations.scheduled_start: a postponed game played on its new
  date), is settled VOID: pnl 0, no CLV, recommendation SETTLED. A pick
  with no scheduled_start (written before the column existed) falls back
  to VOID_AFTER: voided when the game started more than 36 hours after
  the pick was priced. A book voids such a bet; settling it at the stale
  price would book a result the bettor could never have had. Voids are
  not bets: they count as no bet, stake, win or loss in the ledger and
  the CLV report, commit no stake to the day's budget, and leave the
  game open for a new pick (betting/recommend.py).
- CLV (the KPI): clv = implied_prob(closing) - implied_prob(placed),
  per §7. The closing price is the same book's LAST snapshot taken
  strictly AFTER the pick was priced (recommendations.priced_at) and
  strictly BEFORE puck drop (raw.games.start_time_utc). Both bounds
  matter: picks are frozen at the snapshot they came from, so that
  snapshot must not grade itself (CLV would always read 0); and the odds
  feed keeps listing games while they are in play, so a later snapshot
  can hold live prices. The pre-game `pipeline.py close` run takes the
  snapshot that becomes the close. When that book has no qualifying
  snapshot, the fallback is the consensus (median no-vig) of every
  book's last quote in the same window, stored with closing_line NULL so
  same-book and consensus CLV are distinguishable. That consensus close
  has no margin in it, so it is compared like for like with the pick's
  own no-vig fair probability (recommendations.implied_prob_novig, the
  side's consensus fair probability when the pick was issued), not with
  the vig-inclusive placed price, which would read about half the margin
  low. When nothing qualifies, CLV is NULL (unknown), never 0.
- Timestamps (settled_at, decided_at) are naive UTC, like captured_at and
  priced_at.
- betting.bankroll_log is REBUILT from placed_bets on every run —
  idempotent by construction, compounding PAPER_START_BANKROLL through
  settled dates with per-day bet counts, ROI, and average CLV.

Nothing here places, tracks, or settles real money; is_paper stays TRUE
until a human explicitly records a real bet (a later Phase 4 surface).
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import text

from betting.engine import BetDecision, settle as engine_settle
from betting.recommend import BANKROLL as PAPER_START_BANKROLL
from config.migrate import ensure_schema
from config.settings import engine as db
from features.util import american_implied_prob

logger = logging.getLogger("nhl.betting.settle")

VOID_STATES = {"PPD": "postponed", "CNCL": "cancelled"}
# A start this far from the one stored with the pick = a moved game. Wide
# enough for a time change on the same day, far short of a new date.
VOID_SHIFT = timedelta(hours=3)
# Fallback when the pick has no scheduled_start: a start this long after
# pricing = a moved game. It can't tell a pick made a day ahead
# (recommend --date <tomorrow>) from a moved game, hence scheduled_start.
VOID_AFTER = timedelta(hours=36)


def _utc_now() -> datetime:
    """Now as naive UTC, the convention of every stored timestamp here."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(t) -> Optional[datetime]:
    """A timestamp as naive UTC: aware values (start_time_utc) converted,
    naive ones (priced_at, created_at) taken as UTC already. None/NaT ->
    None."""
    if t is None or pd.isna(t):
        return None
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.to_pydatetime()


def void_reason(schedule_state, start_time_utc, priced_at, created_at,
                scheduled_start=None, is_final: bool = True,
                max_shift: timedelta = VOID_SHIFT,
                max_move: timedelta = VOID_AFTER) -> Optional[str]:
    """Why a pick must be VOID instead of settled, or None. Pure.

    PPD/CNCL schedule state voids. So does a FINAL game that was moved,
    where a book would have voided the bet: its puck drop
    (start_time_utc) is more than max_shift away, either way, from the
    start stored with the pick when it was written (scheduled_start).
    Only when scheduled_start is missing (a pick written before the
    column existed), a puck drop more than max_move after the pick was
    priced (priced_at, else created_at) counts as moved instead. No start
    time = no move check."""
    if isinstance(schedule_state, str) and schedule_state in VOID_STATES:
        return VOID_STATES[schedule_state]
    start = _naive_utc(start_time_utc)
    if not is_final or start is None:
        return None
    scheduled = _naive_utc(scheduled_start)
    if scheduled is not None:
        if abs(start - scheduled) > max_shift:
            hours = (start - scheduled).total_seconds() / 3600
            return (f"started {hours:+.1f}h from its start when the pick was "
                    f"made (rescheduled)")
        return None
    issued = _naive_utc(priced_at)
    if issued is None:
        issued = _naive_utc(created_at)
    if issued is not None and start - issued > max_move:
        hours = (start - issued).total_seconds() / 3600
        return f"started {hours:.0f}h after the pick was priced (rescheduled)"
    return None


def compute_clv(closing_price: Optional[int], closing_implied: Optional[float],
                placed_price: int, pick_novig: Optional[float]) -> Optional[float]:
    """CLV, like for like. Pure. Same-book close (closing_price set):
    implied(close) - implied(placed), both with that book's margin in.
    Consensus close (closing_price None): a no-vig probability, so minus
    the pick's own no-vig fair probability (pick_novig). None = unknown."""
    if closing_implied is None:
        return None
    if closing_price is not None:
        return round(closing_implied - float(american_implied_prob(placed_price)), 4)
    if pick_novig is None or pd.isna(pick_novig):
        return None
    return round(closing_implied - float(pick_novig), 4)


def unsettled_recommendations() -> pd.DataFrame:
    """Moneyline recommendations with no placed_bets row yet whose game is
    FINAL, or postponed/cancelled (to be voided)."""
    with db.connect() as conn:
        return pd.read_sql(text("""
            SELECT r.rec_id, r.game_id, r.side, r.best_book, r.best_price,
                   r.recommended_stake, r.edge_pct, r.implied_prob_novig,
                   r.created_at, r.priced_at, r.scheduled_start,
                   g.date AS game_date, g.start_time_utc, g.schedule_state,
                   (g.game_state IN ('FINAL', 'OFF')
                    AND g.home_score IS NOT NULL) AS is_final,
                   (g.home_score > g.away_score) AS home_won
            FROM betting.recommendations r
            JOIN raw.games g USING (game_id)
            WHERE r.market_type = 'ml'
              AND r.status IN ('PENDING', 'APPROVED')
              AND ((g.game_state IN ('FINAL', 'OFF') AND g.home_score IS NOT NULL)
                   OR g.schedule_state IN ('PPD', 'CNCL'))
              AND NOT EXISTS (SELECT 1 FROM betting.placed_bets p
                              WHERE p.rec_id = r.rec_id)
            ORDER BY g.date, r.rec_id
        """), conn)


def closing_quote(game_id: int, book: str, side: str,
                  priced_at: Optional[datetime] = None) -> tuple:
    """(closing_price | None, closing_implied_prob | None).

    Window: captured_at > priced_at (no lower bound when None) and
    captured_at < puck drop (no upper bound, with a warning, when the
    game has no start_time_utc). Same-book last snapshot in the window
    preferred (price + its implied prob); consensus fallback = median
    no-vig prob of every book's last quote in the window (probability
    only — there is no single price to store). (None, None) = unknown."""
    price_col = "home_price" if side == "HOME" else "away_price"
    params = {"g": int(game_id)}
    after = ""
    if priced_at is not None:
        after = "AND o.captured_at > :priced_at"
        params["priced_at"] = priced_at
    with db.connect() as conn:
        start = conn.execute(text("""
            SELECT start_time_utc FROM raw.games WHERE game_id = :g
        """), {"g": int(game_id)}).scalar()
        last = pd.read_sql(text(f"""
            SELECT DISTINCT ON (o.book_name)
                   o.book_name, o.home_price, o.away_price
            FROM raw.odds_snapshots o
            JOIN raw.games g USING (game_id)
            WHERE o.game_id = :g AND o.market_type = 'ml'
              AND o.home_price IS NOT NULL AND o.away_price IS NOT NULL
              {after}
              AND (g.start_time_utc IS NULL
                   OR o.captured_at < (g.start_time_utc AT TIME ZONE 'UTC'))
            ORDER BY o.book_name, o.captured_at DESC
        """), conn, params=params)
    if start is None:
        logger.warning(f"  game {game_id} has no start_time_utc: closing "
                       f"window has no puck-drop bound (re-run the schedule "
                       f"ingest to fill it)")
    if last.empty:
        logger.info(f"  game {game_id}: no closing snapshot after the pick "
                    f"-> CLV unknown")
        return None, None
    same = last[last["book_name"] == book]
    if not same.empty:
        price = int(same.iloc[0][price_col])
        return price, float(american_implied_prob(price))
    ph = last["home_price"].map(american_implied_prob)
    pa = last["away_price"].map(american_implied_prob)
    novig_home = (ph / (ph + pa)).median()
    return None, float(novig_home if side == "HOME" else 1.0 - novig_home)


_INSERT_BET = text("""
    INSERT INTO betting.placed_bets
        (rec_id, book_name, placed_price, stake_amount,
         placed_at, result, pnl, closing_line, clv,
         settled_at, is_paper)
    VALUES (:rec, :book, :price, :stake, :placed_at,
            :result, :pnl, :closing, :clv, :now, TRUE)
""")
_MARK_SETTLED = text("""
    UPDATE betting.recommendations
    SET status = 'SETTLED', decided_at = :now
    WHERE rec_id = :rec
""")


def settle_paper() -> int:
    """Settle (or void) every due paper bet; returns the number settled,
    voids included."""
    ensure_schema()
    due = unsettled_recommendations()
    if due.empty:
        logger.info("No paper bets due for settlement")
        rebuild_bankroll_log()
        return 0

    n_void = 0
    with db.begin() as conn:
        for r in due.itertuples():
            stake = float(r.recommended_stake)
            priced_at = _naive_utc(r.priced_at)
            placed_at = priced_at or _naive_utc(r.created_at)
            reason = void_reason(r.schedule_state, r.start_time_utc,
                                 priced_at, r.created_at,
                                 scheduled_start=r.scheduled_start,
                                 is_final=bool(r.is_final))
            if reason:
                now = _utc_now()
                conn.execute(_INSERT_BET, {
                    "rec": int(r.rec_id), "book": r.best_book,
                    "price": int(r.best_price), "stake": stake,
                    "placed_at": placed_at, "result": "VOID", "pnl": 0.0,
                    "closing": None, "clv": None, "now": now})
                conn.execute(_MARK_SETTLED, {"rec": int(r.rec_id), "now": now})
                n_void += 1
                logger.info(f"  voided paper bet game {r.game_id} {r.side} "
                            f"{int(r.best_price):+d}: game {reason}")
                continue

            decision = BetDecision(side=r.side, price=int(r.best_price),
                                   model_prob=0.0, market_prob=0.0,
                                   edge=0.0, kelly=0.0, stake_pct=0.0)
            pnl = engine_settle(decision, bool(r.home_won), stake)
            closing_price, closing_implied = closing_quote(
                r.game_id, r.best_book, r.side, priced_at)
            clv = compute_clv(closing_price, closing_implied,
                              int(r.best_price), r.implied_prob_novig)
            now = _utc_now()
            conn.execute(_INSERT_BET, {
                "rec": int(r.rec_id), "book": r.best_book,
                "price": int(r.best_price), "stake": stake,
                "placed_at": placed_at,
                "result": "WIN" if pnl > 0 else "LOSS",
                "pnl": round(pnl, 2), "closing": closing_price,
                "clv": clv, "now": now})
            conn.execute(_MARK_SETTLED, {"rec": int(r.rec_id), "now": now})
            logger.info(
                f"  settled paper bet game {r.game_id} {r.side} "
                f"{int(r.best_price):+d}: {'WIN' if pnl > 0 else 'LOSS'} "
                f"{pnl:+.2f}" + (f" clv {clv:+.4f}" if clv is not None else ""))

    n = len(due)
    logger.info(f"Settled {n} paper bet(s)"
                + (f", {n_void} of them VOID" if n_void else ""))
    rebuild_bankroll_log()
    return n


def rebuild_bankroll_log(start_bankroll: float = PAPER_START_BANKROLL) -> int:
    """Recompute betting.bankroll_log from settled paper bets, in date
    order, compounding from start_bankroll. Idempotent."""
    with db.connect() as conn:
        bets = pd.read_sql(text("""
            SELECT g.date, p.pnl, p.stake_amount, p.clv
            FROM betting.placed_bets p
            JOIN betting.recommendations r USING (rec_id)
            JOIN raw.games g USING (game_id)
            WHERE p.is_paper AND p.result IS NOT NULL
              AND p.result <> 'VOID'          -- a void is not a bet
            ORDER BY g.date
        """), conn)

    with db.begin() as conn:
        conn.execute(text("DELETE FROM betting.bankroll_log"))
        if bets.empty:
            return 0
        balance = float(start_bankroll)
        rows = []
        for day, g in bets.groupby("date", sort=True):
            opening = balance
            day_pnl = float(g["pnl"].sum())
            balance += day_pnl
            staked = float(g["stake_amount"].sum())
            clv = g["clv"].dropna()
            rows.append({
                "date": day, "opening_balance": round(opening, 2),
                "gross_pnl": round(day_pnl, 2),
                "closing_balance": round(balance, 2),
                "total_bets": len(g),
                "wins": int((g["pnl"] > 0).sum()),
                "losses": int((g["pnl"] < 0).sum()),
                "pushes": 0,
                "roi_pct": round(100.0 * day_pnl / staked, 3) if staked else 0.0,
                "clv_avg": round(float(clv.mean()), 4) if len(clv) else None,
            })
        conn.execute(text("""
            INSERT INTO betting.bankroll_log
                (date, opening_balance, gross_pnl, closing_balance,
                 total_bets, wins, losses, pushes, roi_pct, clv_avg)
            VALUES (:date, :opening_balance, :gross_pnl, :closing_balance,
                    :total_bets, :wins, :losses, :pushes, :roi_pct, :clv_avg)
        """), rows)
    return len(rows)


def clv_report() -> Optional[pd.DataFrame]:
    """Aggregate paper-trail report: overall + by claimed-edge bucket.
    This is the table that eventually validates (or kills) the edge
    threshold — the same buckets as the historical backtest. Voids are
    left out."""
    ensure_schema()          # reads placed_bets.is_paper
    with db.connect() as conn:
        bets = pd.read_sql(text("""
            SELECT p.pnl, p.stake_amount, p.clv, r.edge_pct
            FROM betting.placed_bets p
            JOIN betting.recommendations r USING (rec_id)
            WHERE p.is_paper AND p.result IS NOT NULL
              AND p.result <> 'VOID'
        """), conn)
    if bets.empty:
        logger.info("No settled paper bets yet")
        return None

    def agg(g):
        staked = g["stake_amount"].sum()
        clv = g["clv"].dropna()
        return pd.Series({
            "bets": len(g),
            "with_clv": len(clv),
            "hit": round(float((g["pnl"] > 0).mean()), 3),
            "staked": round(float(staked), 2),
            "pnl": round(float(g["pnl"].sum()), 2),
            "roi": round(float(g["pnl"].sum() / staked), 4) if staked else 0.0,
            "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
            "clv_pos": round(float((clv > 0).mean()), 3) if len(clv) else None,
        })

    buckets = pd.cut(bets["edge_pct"].astype(float),
                     [0.025, 0.04, 0.06, 0.09, 1.0],
                     labels=["2.5-4%", "4-6%", "6-9%", "9%+"], right=False)
    report = pd.concat([
        bets.groupby(buckets, observed=True).apply(agg, include_groups=False),
        agg(bets).to_frame("ALL").T,
    ])
    print(report.to_string())
    n_clv = int(bets["clv"].notna().sum())
    print(f"\n{n_clv} of {len(bets)} settled bets have a CLV "
          f"(the rest had no closing snapshot after the pick)")
    return report


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Paper-bet settlement + CLV")
    parser.add_argument("--report", action="store_true",
                        help="Print the CLV/ROI report instead of settling")
    args = parser.parse_args()
    clv_report() if args.report else settle_paper()
