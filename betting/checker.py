"""
betting/checker.py
Bet checker + parlay evaluator (Phase 3) — "is this slip +EV?"

Evaluates user-supplied bets against the system's stored model
probabilities (models.predictions, written by the daily recommendation
job) using the same engine math as the recommendation path.

Leg string formats (CLI and API):
    "AWY@HOM ml home -125"            moneyline, side home/away
    "AWY@HOM total over 6.5 -110"     totals at any line (priced from the
                                      stored per-side PMFs, so the line
                                      does NOT need to match the book's)

Semantics locked here:
- Model probabilities come from the LATEST stored prediction for the
  game+market (lgbm_market for ml, poisson_totals for totals). The
  checker never trains models — run `python pipeline.py recommend` (or
  the daily chain) first if a slate hasn't been scored.
- Integer total lines carry push mass: EV counts a push as stake
  returned, and the reported win probability is P(win), not P(not-lose).
- Edge is measured against the implied probability OF THE OFFERED PRICE
  (vig included) — the checker sees one side of one book, so no no-vig
  de-vigging is possible. This makes checker edges CONSERVATIVE relative
  to the daily job's consensus no-vig edges.
- Parlays multiply per-leg outcomes under cross-game independence. A
  pushed leg contributes a factor of 1 (industry settlement rule), so the
  expected payout multiplier is prod_i (p_win_i * dec_i + p_push_i). For
  a custom combined price (boosts) the multiplier is scaled by
  dec_offered / prod(dec_i) — exact for standard pricing, proportional
  approximation for boosts (flagged in the report).
- Legs sharing a game are CORRELATED and independence is wrong there;
  the report says so and withholds a verdict (the joint same-game model
  is Phase 5 scope).
- The totals model has not passed its validation gate (models/totals.py
  STATUS): totals-leg EVs are reported with a warning and never earn a
  "BET" verdict on their own.

Verdicts: BET (edge >= per-market minimum), THIN (positive EV below
threshold), PASS (-EV), NO-MODEL (no stored prediction).
"""
import argparse
import logging
from dataclasses import dataclass, field
from datetime import date as date_cls
from math import prod
from typing import List, Optional

import numpy as np
from sqlalchemy import text

from betting.engine import (EDGE_MIN_ML, KELLY_FRACTION, MAX_STAKE_PCT,
                            decimal_odds)
from config.settings import engine as db
from features.util import american_implied_prob

logger = logging.getLogger("nhl.betting.checker")

EDGE_MIN_TOTAL = 0.030      # locked §7: totals >= 3.0%


@dataclass
class Leg:
    away: str
    home: str
    market: str                  # 'ml' | 'total'
    side: str                    # HOME/AWAY | OVER/UNDER
    price: int                   # American odds offered
    line: Optional[float] = None # totals only
    date: Optional[date_cls] = None
    # filled by evaluation:
    game_id: Optional[int] = None
    p_win: Optional[float] = None
    p_push: float = 0.0
    ev: Optional[float] = None
    edge: Optional[float] = None
    verdict: str = "NO-MODEL"
    notes: List[str] = field(default_factory=list)


def parse_leg(spec: str, on_date: Optional[date_cls] = None) -> Leg:
    """'BOS@TOR ml away -125' | 'BOS@TOR total over 6.5 -110'"""
    parts = spec.replace(",", " ").split()
    matchup = parts[0].upper()
    away, home = matchup.split("@")
    market = parts[1].lower()
    if market == "ml":
        if len(parts) != 4:
            raise ValueError(f"ml leg needs 'AWY@HOM ml side price': {spec!r}")
        return Leg(away=away, home=home, market="ml",
                   side=parts[2].upper(), price=int(parts[3]), date=on_date)
    if market == "total":
        if len(parts) != 5:
            raise ValueError(
                f"total leg needs 'AWY@HOM total side line price': {spec!r}")
        return Leg(away=away, home=home, market="total",
                   side=parts[2].upper(), line=float(parts[3]),
                   price=int(parts[4]), date=on_date)
    raise ValueError(f"Unknown market {market!r} (ml|total)")


def _resolve_game(conn, leg: Leg) -> Optional[int]:
    """game_id by teams (+ date when given; else the next scheduled or
    most recent matchup)."""
    if leg.date:
        row = conn.execute(text("""
            SELECT game_id FROM raw.games
            WHERE home_team = :h AND away_team = :a AND date = :d
        """), {"h": leg.home, "a": leg.away, "d": leg.date}).fetchone()
    else:
        row = conn.execute(text("""
            SELECT game_id FROM raw.games
            WHERE home_team = :h AND away_team = :a AND date >= CURRENT_DATE
            ORDER BY date LIMIT 1
        """), {"h": leg.home, "a": leg.away}).fetchone()
    return row[0] if row else None


def _model_probability(conn, leg: Leg) -> Optional[tuple]:
    """(p_win, p_push) from the latest stored prediction, or None."""
    if leg.market == "ml":
        from models.lgbm import MODEL_NAME
        row = conn.execute(text("""
            SELECT p.home_win_prob
            FROM models.predictions p
            JOIN models.model_registry r USING (model_id)
            WHERE p.game_id = :g AND p.market_type = 'ml'
              AND r.model_name = :name
            ORDER BY p.created_at DESC LIMIT 1
        """), {"g": leg.game_id, "name": MODEL_NAME}).fetchone()
        if row is None or row[0] is None:
            return None
        ph = float(row[0])
        return (ph if leg.side == "HOME" else 1.0 - ph), 0.0

    from models.totals import MODEL_NAME as T_NAME
    from models.totals import prob_over, total_pmf
    row = conn.execute(text("""
        SELECT p.home_goals_pmf, p.away_goals_pmf
        FROM models.predictions p
        JOIN models.model_registry r USING (model_id)
        WHERE p.game_id = :g AND p.market_type = 'total'
          AND r.model_name = :name
        ORDER BY p.created_at DESC LIMIT 1
    """), {"g": leg.game_id, "name": T_NAME}).fetchone()
    if row is None or row[0] is None:
        return None
    tp = total_pmf(np.asarray([row[0]], dtype=float),
                   np.asarray([row[1]], dtype=float))
    p_over, p_push = prob_over(tp, [leg.line])
    p_over, p_push = float(p_over[0]), float(p_push[0])
    p_win = p_over if leg.side == "OVER" else 1.0 - p_over - p_push
    return p_win, p_push


def evaluate_leg(leg: Leg) -> Leg:
    """Fill a leg's model probability, EV per unit, edge, and verdict."""
    with db.connect() as conn:
        leg.game_id = _resolve_game(conn, leg)
        if leg.game_id is None:
            leg.notes.append(f"no game found for {leg.away}@{leg.home}"
                             + (f" on {leg.date}" if leg.date else ""))
            return leg
        probs = _model_probability(conn, leg)

    if probs is None:
        leg.notes.append("no stored prediction — run the daily "
                         "recommendation job for this slate first")
        return leg

    leg.p_win, leg.p_push = probs
    dec = decimal_odds(leg.price)
    p_lose = 1.0 - leg.p_win - leg.p_push
    leg.ev = leg.p_win * (dec - 1.0) - p_lose           # push = stake back
    implied = american_implied_prob(leg.price)
    leg.edge = leg.p_win / (leg.p_win + p_lose) - implied \
        if leg.p_push else leg.p_win - implied

    edge_min = EDGE_MIN_ML if leg.market == "ml" else EDGE_MIN_TOTAL
    if leg.ev <= 0:
        leg.verdict = "PASS"
    elif leg.edge >= edge_min:
        leg.verdict = "BET"
    else:
        leg.verdict = "THIN"
    if leg.market == "total":
        leg.notes.append("totals model has NOT passed its gate — treat "
                         "this EV as unvalidated (models/totals.py)")
        if leg.verdict == "BET":
            leg.verdict = "THIN"
    return leg


def evaluate_parlay(legs: List[Leg],
                    combined_price: Optional[int] = None) -> dict:
    """Evaluate a slip: one leg = straight bet; several = parlay under
    cross-game independence with push-aware settlement."""
    legs = [evaluate_leg(l) for l in legs]
    report = {"legs": legs, "correlated": False, "verdict": "NO-MODEL",
              "notes": []}

    if any(l.p_win is None for l in legs):
        report["notes"].append("one or more legs lack a stored model "
                               "prediction — no combined EV computed")
        return report

    game_ids = [l.game_id for l in legs]
    if len(set(game_ids)) < len(game_ids):
        report["correlated"] = True
        report["notes"].append(
            "legs share a game: independence math is WRONG for same-game "
            "parlays; verdict withheld (joint model is Phase 5)")

    decs = [decimal_odds(l.price) for l in legs]
    fair_dec = prod(decs)
    dec_offered = decimal_odds(combined_price) if combined_price else fair_dec
    if combined_price and any(l.p_push > 0 for l in legs):
        report["notes"].append("custom combined price with pushable legs: "
                               "push handling is a proportional approximation")

    p_all_win = prod(l.p_win for l in legs)
    exp_multiplier = (dec_offered / fair_dec) \
        * prod(l.p_win * d + l.p_push for l, d in zip(legs, decs))
    ev = exp_multiplier - 1.0

    b = dec_offered - 1.0
    kelly = max(0.0, (b * p_all_win - (1.0 - p_all_win)) / b)
    stake_pct = min(kelly * KELLY_FRACTION, MAX_STAKE_PCT)

    report.update({
        "n_legs": len(legs),
        "p_win_all": p_all_win,
        "decimal_offered": dec_offered,
        "decimal_fair_product": fair_dec,
        "ev_per_unit": ev,
        "kelly": kelly,
        "stake_pct": stake_pct if ev > 0 else 0.0,
    })
    if report["correlated"]:
        return report

    if ev <= 0:
        report["verdict"] = "PASS"
    elif all(l.verdict == "BET" for l in legs):
        report["verdict"] = "BET"
    else:
        report["verdict"] = "THIN"
    return report


def format_report(report: dict, bankroll: float = None) -> str:
    lines = []
    for l in report["legs"]:
        head = (f"{l.away}@{l.home} {l.market} {l.side}"
                + (f" {l.line}" if l.line is not None else "")
                + f" {l.price:+d}")
        if l.p_win is None:
            lines.append(f"  {head}: NO-MODEL ({'; '.join(l.notes)})")
            continue
        lines.append(
            f"  {head}: p={l.p_win:.3f}"
            + (f" push={l.p_push:.3f}" if l.p_push else "")
            + f" ev={l.ev:+.3f}/u edge={l.edge:+.1%} -> {l.verdict}"
            + (f"  [{'; '.join(l.notes)}]" if l.notes else ""))
    if "ev_per_unit" in report:
        lines.append(
            f"  COMBINED ({report['n_legs']} leg"
            f"{'s' if report['n_legs'] > 1 else ''}): "
            f"P(win)={report['p_win_all']:.3f} at "
            f"{report['decimal_offered']:.2f} (fair product "
            f"{report['decimal_fair_product']:.2f}) "
            f"EV={report['ev_per_unit']:+.3f}/u "
            f"quarter-Kelly={report['stake_pct']:.2%} of bankroll"
            + (f" (= {report['stake_pct'] * bankroll:.2f})" if bankroll else ""))
    lines.append(f"  VERDICT: {report['verdict']}")
    for n in report["notes"]:
        lines.append(f"  ! {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Check a bet or parlay against the model",
        epilog='Example: python -m betting.checker --date 2026-01-15 '
               '--leg "MTL@BUF ml away -125" --leg "SJS@WSH total over 6.5 -110"')
    parser.add_argument("--leg", action="append", required=True,
                        help="'AWY@HOM ml side price' or "
                             "'AWY@HOM total side line price'")
    parser.add_argument("--date", type=date_cls.fromisoformat, default=None,
                        help="Game date (default: next matchup)")
    parser.add_argument("--price", type=int, default=None,
                        help="Combined parlay price if boosted (American)")
    parser.add_argument("--bankroll", type=float, default=None)
    args = parser.parse_args()

    legs = [parse_leg(s, args.date) for s in args.leg]
    print(format_report(evaluate_parlay(legs, args.price), args.bankroll))
