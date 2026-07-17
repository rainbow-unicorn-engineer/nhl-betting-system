"""
betting/backtest.py
Payout backtest of the full strategy chain (Phase 3).

Simulates: walk-forward OOF model probabilities (never trained on the
games they score) -> engine.evaluate_moneyline -> quarter-Kelly staking ->
settlement at the actual offered prices, compounding a bankroll through
the games in date order with the daily exposure cap enforced.

Price universe: ONLY provider='DraftKings' rows of raw.historical_odds —
true bettable two-way prices (the Unibet era is 3-way regulation lines and
would corrupt payouts; see docs/historical_odds.md). That confines the
backtest to the 2025-26 season, ~1,000 games. Small sample: the result is
a sanity check of the machinery and rough edge, not a proof of long-run
ROI — the multi-season proof accumulates live via CLV + the paper trail.

Honesty notes baked in:
- The DraftKings line is near-closing: we simulate betting INTO the
  sharpest price with no line shopping — conservative on both edge
  availability and payout.
- A flat-stake (1 unit) variant is reported alongside Kelly so the
  conclusion doesn't hinge on staking.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text

from betting.engine import MAX_DAILY_PCT, evaluate_moneyline, settle
from config.settings import engine as db

logger = logging.getLogger("nhl.betting.backtest")

START_BANKROLL = 100.0     # units; percentages are what matter


def load_priced_games() -> pd.DataFrame:
    """Games with true two-way prices (DraftKings era) + outcomes."""
    with db.connect() as conn:
        return pd.read_sql(text("""
            SELECT h.game_id, g.date, h.home_ml, h.away_ml,
                   (g.home_score > g.away_score) AS home_won
            FROM raw.historical_odds h
            JOIN raw.games g USING (game_id)
            WHERE h.provider = 'DraftKings'
              AND g.game_state IN ('FINAL','OFF')
            ORDER BY g.date, h.game_id
        """), conn)


@dataclass
class BacktestResult:
    n_games: int
    n_bets: int
    hit_rate: float
    total_staked: float
    pnl: float
    roi: float
    end_bankroll: float
    max_drawdown: float
    flat_pnl_units: float
    flat_roi: float
    avg_edge: float
    bets: pd.DataFrame = field(repr=False, default=None)


def run_backtest(oof: pd.DataFrame = None,
                 start_bankroll: float = START_BANKROLL) -> BacktestResult:
    """oof: DataFrame with game_id + prob_home (walk-forward OOF). If None,
    the lgbm walk-forward is run to produce it."""
    if oof is None:
        from models.lgbm import run_lgbm
        oof = run_lgbm(register=False)["oof"]

    games = load_priced_games().merge(
        oof[["game_id", "prob_home"]], on="game_id", how="inner")
    logger.info(f"Backtest universe: {len(games)} priced games with OOF probs")

    bankroll, peak, max_dd = start_bankroll, start_bankroll, 0.0
    flat_pnl = 0.0
    day_spend, cur_day = 0.0, None
    rows = []

    for g in games.itertuples():
        d = evaluate_moneyline(g.prob_home, g.home_ml, g.away_ml)
        if d is None:
            continue
        if g.date != cur_day:
            cur_day, day_spend = g.date, 0.0
        stake = bankroll * d.stake_pct
        if day_spend + stake > bankroll * MAX_DAILY_PCT:
            continue                      # daily exposure cap
        day_spend += stake

        pnl = settle(d, g.home_won, stake)
        bankroll += pnl
        flat_pnl += settle(d, g.home_won, 1.0)
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)
        rows.append({"game_id": g.game_id, "date": g.date, "side": d.side,
                     "price": d.price, "model_prob": round(d.model_prob, 4),
                     "market_prob": round(d.market_prob, 4),
                     "edge": round(d.edge, 4), "stake": round(stake, 3),
                     "pnl": round(pnl, 3),
                     "bankroll": round(bankroll, 2),
                     "won": pnl > 0})

    bets = pd.DataFrame(rows)
    n = len(bets)
    staked = bets["stake"].sum() if n else 0.0
    result = BacktestResult(
        n_games=len(games), n_bets=n,
        hit_rate=float(bets["won"].mean()) if n else 0.0,
        total_staked=float(staked),
        pnl=float(bankroll - start_bankroll),
        roi=float((bankroll - start_bankroll) / staked) if staked else 0.0,
        end_bankroll=float(bankroll),
        max_drawdown=float(max_dd),
        flat_pnl_units=float(flat_pnl),
        flat_roi=float(flat_pnl / n) if n else 0.0,
        avg_edge=float(bets["edge"].mean()) if n else 0.0,
        bets=bets)

    logger.info(
        f"BACKTEST: {result.n_bets} bets over {result.n_games} priced games "
        f"| hit {result.hit_rate:.3f} | staked {result.total_staked:.1f}u "
        f"| pnl {result.pnl:+.2f}u | ROI {result.roi:+.3%} "
        f"| flat-stake ROI {result.flat_roi:+.3%} "
        f"| max drawdown {result.max_drawdown:.1%} "
        f"| avg edge {result.avg_edge:.3f}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_backtest()
