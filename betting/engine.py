"""
betting/engine.py
The strategy layer (Phase 3): edge detection + quarter-Kelly staking.

Pure functions only — every rule here is locked by PROJECT_CONTEXT §7 and
unit-tested; the daily recommendation job and the backtest both call these
so simulated and live behavior cannot drift apart.

Rules (locked):
- Bet only when model edge >= EDGE_MIN for the market (moneyline 2.5%).
- Stake = KELLY_FRACTION (0.25) of the full Kelly fraction, capped at
  MAX_STAKE_PCT (2%) of bankroll per bet and MAX_DAILY_PCT (10%) per day.
- Edge is measured against the NO-VIG implied probability; payouts are
  settled at the actual (vig-inclusive) price. Both matter: edge vs the
  fair line, cash at the offered line.
"""
from dataclasses import dataclass
from typing import Optional

from features.util import american_implied_prob

EDGE_MIN_ML = 0.025
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.02
MAX_DAILY_PCT = 0.10


def no_vig_probs(home_ml: float, away_ml: float) -> tuple:
    """Fair (no-vig) win probabilities from a two-sided moneyline."""
    ph, pa = american_implied_prob(home_ml), american_implied_prob(away_ml)
    return ph / (ph + pa), pa / (ph + pa)


def decimal_odds(american: float) -> float:
    """American -> decimal. decimal_odds(-150)=1.667, decimal_odds(130)=2.3"""
    a = float(american)
    return 1.0 + (100.0 / -a if a < 0 else a / 100.0)


def kelly_fraction(p: float, american: float) -> float:
    """Full-Kelly optimal bankroll fraction for win prob p at a price.
    f* = (b*p - q)/b with b = decimal - 1. Negative edge -> 0."""
    b = decimal_odds(american) - 1.0
    f = (b * p - (1.0 - p)) / b
    return max(0.0, f)


@dataclass
class BetDecision:
    side: str                 # HOME or AWAY
    price: int                # American odds taken
    model_prob: float         # our probability for that side
    market_prob: float        # no-vig probability for that side
    edge: float               # model_prob - market_prob
    kelly: float              # full-Kelly fraction
    stake_pct: float          # of bankroll, after quarter-Kelly + cap


def evaluate_moneyline(model_home_prob: float,
                       home_ml: float, away_ml: float,
                       edge_min: float = EDGE_MIN_ML) -> Optional[BetDecision]:
    """The one decision function: given our probability and a two-sided
    line, return the bet to make (at most one side) or None."""
    mkt_home, mkt_away = no_vig_probs(home_ml, away_ml)
    for side, p_model, p_mkt, price in (
            ("HOME", model_home_prob, mkt_home, home_ml),
            ("AWAY", 1.0 - model_home_prob, mkt_away, away_ml)):
        edge = p_model - p_mkt
        if edge < edge_min:
            continue
        kelly = kelly_fraction(p_model, price)
        if kelly <= 0.0:      # +edge vs no-vig can still be -EV vs the vig
            continue
        stake_pct = min(kelly * KELLY_FRACTION, MAX_STAKE_PCT)
        return BetDecision(side=side, price=int(price),
                           model_prob=p_model, market_prob=p_mkt,
                           edge=edge, kelly=kelly, stake_pct=stake_pct)
    return None


def settle(decision: BetDecision, home_won: bool, stake: float) -> float:
    """PnL of a settled moneyline bet (OT/SO included, no pushes)."""
    won = home_won if decision.side == "HOME" else not home_won
    return stake * (decimal_odds(decision.price) - 1.0) if won else -stake
