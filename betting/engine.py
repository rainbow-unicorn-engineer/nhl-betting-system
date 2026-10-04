"""
betting/engine.py
The strategy layer (Phase 3): edge detection + quarter-Kelly staking.

Pure functions only — every rule here is locked by PROJECT_CONTEXT §7 and
unit-tested; the daily recommendation job and the backtest both call these
so simulated and live behavior cannot drift apart.

Rules (locked):
- Bet only when model edge >= EDGE_MIN for the market (moneyline 2.5%).
- Stake = KELLY_FRACTION (0.25) of the full Kelly fraction (Kelly → the
  bet size that grows a bankroll fastest IF the win chances are right;
  a quarter of it gives up a little growth for far smaller swings),
  capped at MAX_STAKE_PCT (default 2%) of bankroll per bet and
  MAX_DAILY_PCT (default 10%) per day.
- Per game, at most MAX_BETS_PER_GAME (3) bets and MAX_GAME_STAKE_PCT
  (default 4%) of bankroll staked, counting every market.
- The four limits can be changed in .env (same names). A fraction must
  be above 0 and at most 1 (0.02 = 2%); a bad value logs an error and
  the default is used, so a typo can't stop the daily run. The defaults
  below stay the locked rule; `python -m betting.montecarlo` shows what
  larger limits do to a bankroll. Bets on one game are
  correlated → they tend to win or lose together (a high-scoring game
  moves the total, the puck line and the scorers' props at once), so
  three bets on one game are riskier than three bets on three games.
  That is §7's "max 3 correlated bets per game". game_cap_reason() is
  the check; betting/recommend.py applies it with the daily cap,
  strongest edges first.
- Edge is measured against the NO-VIG implied probability; payouts are
  settled at the actual (vig-inclusive) price. Both matter: edge vs the
  fair line, cash at the offered line.
"""
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from features.util import american_implied_prob

# The same .env config/settings.py loads (it never overrides a variable
# already set). Loaded here too because this module is often imported
# before config.settings, and the limits below are read at import.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("nhl.betting.engine")

EDGE_MIN_ML = 0.025
KELLY_FRACTION = 0.25

# The locked defaults (PROJECT_CONTEXT §7). The live values below can be
# overridden in .env; these never change.
DEFAULT_MAX_STAKE_PCT = 0.02        # of bankroll, one bet
DEFAULT_MAX_DAILY_PCT = 0.10        # of bankroll, every bet issued on one day
DEFAULT_MAX_BETS_PER_GAME = 3       # bets on one game, any market (§7)
DEFAULT_MAX_GAME_STAKE_PCT = 0.04   # of bankroll, all bets on one game together


def limit_setting(name: str, default, parse, valid, what: str):
    """An exposure limit from the environment. Unset or blank = default; a
    malformed or out-of-range value logs an error and uses the default, so
    a typo in .env can't abort the import, and with it the daily chain."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
        if not valid(value):
            raise ValueError(raw)
        return value
    except (ValueError, OverflowError):
        logger.error(f"{name}={raw!r} is not {what} — using the default, "
                     f"{default:g}")
        return default


def fraction_setting(name: str, default: float) -> float:
    """A share of bankroll from the environment: above 0, at most 1."""
    return limit_setting(name, default, float,
                         lambda v: math.isfinite(v) and 0 < v <= 1,
                         f"a fraction of bankroll above 0 and at most 1 "
                         f"({default:g} = {default:.0%})")


MAX_STAKE_PCT = fraction_setting("MAX_STAKE_PCT", DEFAULT_MAX_STAKE_PCT)
MAX_DAILY_PCT = fraction_setting("MAX_DAILY_PCT", DEFAULT_MAX_DAILY_PCT)
MAX_BETS_PER_GAME = limit_setting(
    "MAX_BETS_PER_GAME", DEFAULT_MAX_BETS_PER_GAME, int, lambda v: v >= 1,
    "a whole number of bets, 1 or more")
MAX_GAME_STAKE_PCT = fraction_setting("MAX_GAME_STAKE_PCT",
                                      DEFAULT_MAX_GAME_STAKE_PCT)


def cap_warnings(stake_pct: float = None, daily_pct: float = None,
                 game_pct: float = None) -> List[str]:
    """Plain-English warnings for limits that contradict each other (the
    settings in use by default). The caps skip a bet that doesn't fit;
    they never shrink it, so a per-bet cap above the per-day or per-game
    cap means a bet that large is never issued."""
    stake_pct = MAX_STAKE_PCT if stake_pct is None else stake_pct
    daily_pct = MAX_DAILY_PCT if daily_pct is None else daily_pct
    game_pct = MAX_GAME_STAKE_PCT if game_pct is None else game_pct
    out = []
    if stake_pct > daily_pct:
        out.append(f"MAX_STAKE_PCT ({stake_pct:.0%}) is above MAX_DAILY_PCT "
                   f"({daily_pct:.0%}): a bet bigger than the day's limit is "
                   f"skipped, not made smaller.")
    if stake_pct > game_pct:
        out.append(f"MAX_STAKE_PCT ({stake_pct:.0%}) is above "
                   f"MAX_GAME_STAKE_PCT ({game_pct:.0%}): a bet bigger than "
                   f"the per-game limit is skipped, not made smaller.")
    return out


for _w in cap_warnings():
    logger.warning(_w)


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


def evaluate_market(model_home_prob: float, fair_home_prob: float,
                    home_price: Optional[float], away_price: Optional[float],
                    edge_min: float = EDGE_MIN_ML) -> Optional[BetDecision]:
    """The one decision function, line-shopping form: edge is measured
    against a fair (no-vig) probability that may come from a consensus of
    books, while each side is priced at the best available price (possibly
    from different books). A side with no price is not bettable."""
    for side, p_model, p_fair, price in (
            ("HOME", model_home_prob, fair_home_prob, home_price),
            ("AWAY", 1.0 - model_home_prob, 1.0 - fair_home_prob, away_price)):
        if price is None:
            continue
        edge = p_model - p_fair
        if edge < edge_min:
            continue
        kelly = kelly_fraction(p_model, price)
        if kelly <= 0.0:      # +edge vs no-vig can still be -EV vs the vig
            continue
        stake_pct = min(kelly * KELLY_FRACTION, MAX_STAKE_PCT)
        return BetDecision(side=side, price=int(price),
                           model_prob=p_model, market_prob=p_fair,
                           edge=edge, kelly=kelly, stake_pct=stake_pct)
    return None


def evaluate_moneyline(model_home_prob: float,
                       home_ml: float, away_ml: float,
                       edge_min: float = EDGE_MIN_ML) -> Optional[BetDecision]:
    """Single-book form: fair probability and prices from one two-sided
    line. The backtest uses this; the daily job uses evaluate_market."""
    fair_home, _ = no_vig_probs(home_ml, away_ml)
    return evaluate_market(model_home_prob, fair_home, home_ml, away_ml,
                           edge_min)


def game_cap_reason(stake: float, bets_on_game: int, staked_on_game: float,
                    bankroll: float,
                    max_bets: int = MAX_BETS_PER_GAME,
                    max_game_stake_pct: float = MAX_GAME_STAKE_PCT
                    ) -> Optional[str]:
    """Whether one more bet of `stake` fits the per-game limits, for any
    market. bets_on_game / staked_on_game: the bets already on that game
    and their total stake, counting every market and the new bets kept
    earlier in the same run. Returns None when it fits, otherwise a
    plain-English reason."""
    if bets_on_game + 1 > max_bets:
        return (f"per-game limit: {bets_on_game} bet(s) already on this "
                f"game, max {max_bets}")
    limit = bankroll * max_game_stake_pct
    if staked_on_game + stake > limit + 1e-9:     # 1e-9: float noise only
        return (f"per-game stake limit: {staked_on_game:.2f} already on this "
                f"game + {stake:.2f} would pass {limit:.2f} "
                f"({max_game_stake_pct:.0%} of bankroll)")
    return None


def settle(decision: BetDecision, home_won: bool, stake: float) -> float:
    """PnL of a settled moneyline bet (OT/SO included, no pushes)."""
    won = home_won if decision.side == "HOME" else not home_won
    return stake * (decimal_odds(decision.price) - 1.0) if won else -stake
