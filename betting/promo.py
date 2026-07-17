"""
betting/promo.py
Promo-hedging calculator v1 (design: docs/promo_hedging_calculator.md).

Basis: the two-bettor legal structure in docs/texas_execution_options.md.
Every leg belongs to exactly one bettor placing his own funds on a venue
legal for him — Bettor A (partner, Louisiana sportsbooks, in his name)
or Bettor B (Gavin, Kalshi/Polymarket in Texas). The calculator computes
equal-profit hedge stakes and reports PER-BETTOR P&L per outcome; it
never pools stakes and has no concept of one bettor placing for another.

All hedge math is in decimal odds against the hedge side's EFFECTIVE
decimal (venue fees folded in). Fee coefficients are parameters —
Polymarket already moved its sports taker fee 0.03 -> 0.05 in July 2026.
"""
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from betting.engine import decimal_odds

logger = logging.getLogger("nhl.betting.promo")

KALSHI_TAKER = 0.07
POLYMARKET_TAKER = 0.05


# ── Venue cost models ──────────────────────────────────────────────

def effective_decimal(venue: str, price=None, contract_price: float = None,
                      fee_coeff: float = None, maker_fee: float = 0.0) -> float:
    """Effective decimal odds of the hedge side after venue costs.

    venue: 'book' (American price), 'kalshi'/'polymarket' taker
    (contract_price in (0,1)), 'kalshi_maker'/'polymarket_maker'
    (contract_price; fees ~0, queue risk borne by the user).
    """
    if venue == "book":
        return decimal_odds(price)
    q = contract_price
    if q is None or not 0.0 < q < 1.0:
        raise ValueError("prediction-market venues need contract_price in (0,1)")
    if venue == "kalshi":
        c = KALSHI_TAKER if fee_coeff is None else fee_coeff
        return 1.0 / (q + c * q * (1.0 - q))
    if venue == "polymarket":
        c = POLYMARKET_TAKER if fee_coeff is None else fee_coeff
        return 1.0 / (q + c * q * (1.0 - q))
    if venue in ("kalshi_maker", "polymarket_maker"):
        return 1.0 / (q + maker_fee)
    raise ValueError(f"unknown venue {venue!r}")


# ── Hedge math (pure, decimal odds) ────────────────────────────────

@dataclass
class HedgePlan:
    promo_type: str
    hedge_stake: float          # dollars on the hedge side
    guaranteed: float           # profit locked across outcomes (pre-slippage)
    conversion: float           # guaranteed / promo value
    bonus_win_pnl: float        # bonus-bettor outcome legs
    bonus_lose_pnl: float
    hedge_win_pnl: float        # hedge-bettor outcome legs
    hedge_lose_pnl: float
    notes: List[str] = field(default_factory=list)


def hedge_free_bet(F: float, dec_bonus: float, dec_hedge: float) -> HedgePlan:
    """Free bet (stake NOT returned): equal-profit hedge."""
    h = F * (dec_bonus - 1.0) / dec_hedge
    guaranteed = F * (dec_bonus - 1.0) - h
    return HedgePlan(
        promo_type="free_bet", hedge_stake=h,
        guaranteed=guaranteed, conversion=guaranteed / F,
        bonus_win_pnl=F * (dec_bonus - 1.0), bonus_lose_pnl=0.0,
        hedge_win_pnl=h * (dec_hedge - 1.0), hedge_lose_pnl=-h)


def hedge_risk_free(B: float, dec_bonus: float, dec_hedge: float,
                    refund_value: float = 0.70) -> HedgePlan:
    """First-bet-insurance: cash stake B, refunded as a free bet worth
    refund_value per dollar if it loses."""
    h = B * (dec_bonus - refund_value) / dec_hedge
    guaranteed = B * (dec_bonus - 1.0) - h
    plan = HedgePlan(
        promo_type="risk_free", hedge_stake=h,
        guaranteed=guaranteed, conversion=guaranteed / B,
        bonus_win_pnl=B * (dec_bonus - 1.0),
        bonus_lose_pnl=-B + refund_value * B,
        hedge_win_pnl=h * (dec_hedge - 1.0), hedge_lose_pnl=-h)
    plan.notes.append(f"refund free bet valued at {refund_value:.0%} of "
                      f"face — realize it via hedge_free_bet when it arrives")
    return plan


def hedge_profit_boost(B: float, dec_bonus: float, boost: float,
                       dec_hedge: float) -> HedgePlan:
    """Profit boost of `boost` (e.g. 0.5 = +50% winnings) on cash stake B."""
    d_boosted = 1.0 + (dec_bonus - 1.0) * (1.0 + boost)
    h = B * d_boosted / dec_hedge
    guaranteed = B * (d_boosted - 1.0) - h
    plan = HedgePlan(
        promo_type="profit_boost", hedge_stake=h,
        guaranteed=guaranteed, conversion=guaranteed / B,
        bonus_win_pnl=B * (d_boosted - 1.0), bonus_lose_pnl=-B,
        hedge_win_pnl=h * (dec_hedge - 1.0), hedge_lose_pnl=-h)
    if guaranteed <= 0:
        plan.notes.append("boost does not clear the round-trip vig at these "
                          "odds — not worth locking; consider unhedged only "
                          "if the model says the boosted side is +EV")
    return plan


def best_free_bet_conversion(F: float, candidates: list,
                             dec_hedge_fn) -> tuple:
    """Pick the bonus odds that maximize conversion. candidates:
    [(label, dec_bonus, hedge_kwargs)] where dec_hedge_fn(**kwargs) gives
    the matching hedge side's effective decimal."""
    best = None
    for label, dec_bonus, kw in candidates:
        plan = hedge_free_bet(F, dec_bonus, dec_hedge_fn(**kw))
        if best is None or plan.conversion > best[1].conversion:
            best = (label, plan)
    return best


def apply_slippage(plan: HedgePlan, slippage: float = 0.01) -> HedgePlan:
    """Widen the guarantee for price movement between the two legs
    (both bettors can place near-simultaneously, so default is small)."""
    plan.guaranteed *= (1.0 - slippage)
    plan.conversion *= (1.0 - slippage)
    plan.notes.append(f"guarantee shown net of {slippage:.1%} slippage")
    return plan


def format_plan(plan: HedgePlan, bonus_bettor: str = "A (LA book)",
                hedge_bettor: str = "B (TX prediction market)") -> str:
    """Per-bettor outcome table — stakes and P&L are never pooled."""
    lines = [
        f"{plan.promo_type}: hedge stake {plan.hedge_stake:.2f} "
        f"-> locked {plan.guaranteed:+.2f} ({plan.conversion:.1%} conversion)",
        f"  outcome bonus-side WINS : {bonus_bettor} {plan.bonus_win_pnl:+.2f}"
        f" | {hedge_bettor} {plan.hedge_lose_pnl:+.2f}",
        f"  outcome bonus-side LOSES: {bonus_bettor} {plan.bonus_lose_pnl:+.2f}"
        f" | {hedge_bettor} {plan.hedge_win_pnl:+.2f}",
    ]
    lines += [f"  ! {n}" for n in plan.notes]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Promo-hedging calculator (two-bettor legal structure)",
        epilog="Example: python -m betting.promo free_bet --amount 100 "
               "--bonus-odds 400 --hedge-venue kalshi --contract-price 0.22")
    parser.add_argument("promo", choices=["free_bet", "risk_free",
                                          "profit_boost"])
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--bonus-odds", type=int, required=True,
                        help="American odds of the promo leg")
    parser.add_argument("--hedge-venue", default="book",
                        choices=["book", "kalshi", "polymarket",
                                 "kalshi_maker", "polymarket_maker"])
    parser.add_argument("--hedge-odds", type=int, default=None,
                        help="American odds (book hedge)")
    parser.add_argument("--contract-price", type=float, default=None,
                        help="Contract price in (0,1) (prediction markets)")
    parser.add_argument("--refund-value", type=float, default=0.70)
    parser.add_argument("--boost", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.01)
    args = parser.parse_args()

    d_h = effective_decimal(args.hedge_venue, price=args.hedge_odds,
                            contract_price=args.contract_price)
    d_b = decimal_odds(args.bonus_odds)
    if args.promo == "free_bet":
        plan = hedge_free_bet(args.amount, d_b, d_h)
    elif args.promo == "risk_free":
        plan = hedge_risk_free(args.amount, d_b, d_h, args.refund_value)
    else:
        plan = hedge_profit_boost(args.amount, d_b, args.boost, d_h)
    print(format_plan(apply_slippage(plan, args.slippage)))
