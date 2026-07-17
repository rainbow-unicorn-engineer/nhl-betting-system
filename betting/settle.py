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
- CLV (the KPI): clv = implied_prob(closing) - implied_prob(placed),
  per §7. The closing price is the LAST stored snapshot for the same
  book and side (after puck drop the odds feed stops listing the game,
  so the final stored quote is the closing quote); when that book has no
  snapshot, the fallback is the last-quote consensus (median no-vig)
  across books, stored with closing_line NULL so same-book and consensus
  CLV are distinguishable.
- betting.bankroll_log is REBUILT from placed_bets on every run —
  idempotent by construction, compounding PAPER_START_BANKROLL through
  settled dates with per-day bet counts, ROI, and average CLV.

Nothing here places, tracks, or settles real money; is_paper stays TRUE
until a human explicitly records a real bet (a later Phase 4 surface).
"""
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from betting.engine import BetDecision, settle as engine_settle
from betting.recommend import BANKROLL as PAPER_START_BANKROLL
from config.settings import engine as db
from features.util import american_implied_prob

logger = logging.getLogger("nhl.betting.settle")

_DDL = """
ALTER TABLE betting.placed_bets
    ADD COLUMN IF NOT EXISTS is_paper BOOLEAN NOT NULL DEFAULT TRUE
"""


def ensure_schema() -> None:
    with db.begin() as conn:
        conn.execute(text(_DDL))


def unsettled_recommendations() -> pd.DataFrame:
    """Moneyline recommendations whose game is FINAL and which have no
    placed_bets row yet."""
    with db.connect() as conn:
        return pd.read_sql(text("""
            SELECT r.rec_id, r.game_id, r.side, r.best_book, r.best_price,
                   r.recommended_stake, r.edge_pct, r.created_at,
                   g.date AS game_date,
                   (g.home_score > g.away_score) AS home_won
            FROM betting.recommendations r
            JOIN raw.games g USING (game_id)
            WHERE r.market_type = 'ml'
              AND r.status IN ('PENDING', 'APPROVED')
              AND g.game_state IN ('FINAL', 'OFF')
              AND g.home_score IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM betting.placed_bets p
                              WHERE p.rec_id = r.rec_id)
            ORDER BY g.date, r.rec_id
        """), conn)


def closing_quote(game_id: int, book: str, side: str) -> tuple:
    """(closing_price | None, closing_implied_prob | None).

    Same-book last snapshot preferred (price + its implied prob);
    consensus fallback = median no-vig prob of every book's last quote
    (probability only — there is no single price to store)."""
    price_col = "home_price" if side == "HOME" else "away_price"
    with db.connect() as conn:
        last = pd.read_sql(text(f"""
            SELECT DISTINCT ON (book_name)
                   book_name, home_price, away_price
            FROM raw.odds_snapshots
            WHERE game_id = :g AND market_type = 'ml'
              AND home_price IS NOT NULL AND away_price IS NOT NULL
            ORDER BY book_name, captured_at DESC
        """), conn, params={"g": int(game_id)})
    if last.empty:
        return None, None
    same = last[last["book_name"] == book]
    if not same.empty:
        price = int(same.iloc[0][price_col])
        return price, float(american_implied_prob(price))
    ph = last["home_price"].map(american_implied_prob)
    pa = last["away_price"].map(american_implied_prob)
    novig_home = (ph / (ph + pa)).median()
    return None, float(novig_home if side == "HOME" else 1.0 - novig_home)


def settle_paper() -> int:
    """Settle every due paper bet; returns the number settled."""
    ensure_schema()
    due = unsettled_recommendations()
    if due.empty:
        logger.info("No paper bets due for settlement")
        rebuild_bankroll_log()
        return 0

    with db.begin() as conn:
        for r in due.itertuples():
            decision = BetDecision(side=r.side, price=int(r.best_price),
                                   model_prob=0.0, market_prob=0.0,
                                   edge=0.0, kelly=0.0, stake_pct=0.0)
            stake = float(r.recommended_stake)
            pnl = engine_settle(decision, bool(r.home_won), stake)
            closing_price, closing_implied = closing_quote(
                r.game_id, r.best_book, r.side)
            clv = None
            if closing_implied is not None:
                clv = round(closing_implied
                            - float(american_implied_prob(r.best_price)), 4)
            conn.execute(text("""
                INSERT INTO betting.placed_bets
                    (rec_id, book_name, placed_price, stake_amount,
                     placed_at, result, pnl, closing_line, clv,
                     settled_at, is_paper)
                VALUES (:rec, :book, :price, :stake, :placed_at,
                        :result, :pnl, :closing, :clv, :now, TRUE)
            """), {"rec": int(r.rec_id), "book": r.best_book,
                   "price": int(r.best_price), "stake": stake,
                   "placed_at": r.created_at,
                   "result": "WIN" if pnl > 0 else "LOSS",
                   "pnl": round(pnl, 2), "closing": closing_price,
                   "clv": clv, "now": datetime.now()})
            conn.execute(text("""
                UPDATE betting.recommendations
                SET status = 'SETTLED', decided_at = :now
                WHERE rec_id = :rec
            """), {"rec": int(r.rec_id), "now": datetime.now()})
            logger.info(
                f"  settled paper bet game {r.game_id} {r.side} "
                f"{int(r.best_price):+d}: {'WIN' if pnl > 0 else 'LOSS'} "
                f"{pnl:+.2f}" + (f" clv {clv:+.4f}" if clv is not None else ""))

    n = len(due)
    logger.info(f"Settled {n} paper bet(s)")
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
    threshold — the same buckets as the historical backtest."""
    with db.connect() as conn:
        bets = pd.read_sql(text("""
            SELECT p.pnl, p.stake_amount, p.clv, r.edge_pct
            FROM betting.placed_bets p
            JOIN betting.recommendations r USING (rec_id)
            WHERE p.is_paper AND p.result IS NOT NULL
        """), conn)
    if bets.empty:
        logger.info("No settled paper bets yet")
        return None

    def agg(g):
        staked = g["stake_amount"].sum()
        clv = g["clv"].dropna()
        return pd.Series({
            "bets": len(g),
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
