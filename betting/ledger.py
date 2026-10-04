"""
betting/ledger.py
The bet ledger: the bets the bettors really placed, how they settled, and
how much money is on each platform.

Words used here (plain English, → marks a translation):
- slip → one bet ticket. A single bet has one leg; a parlay → several
  bets ("legs") joined into one ticket that pays only if every leg wins,
  at a bigger price.
- stake → the money put on the bet. payout → the money paid back,
  the stake included (a $10 bet at +150 that wins pays out $25).
  P/L → profit or loss: payout minus stake.
- American odds → -150 means bet $150 to win $100; +130 means bet $100
  to win $130.
- moneyline (ml) → which team wins, overtime and shootout included.
  puck line (pl) → a moneyline with a goal handicap, e.g. TOR -1.5 wins
  only if Toronto wins by 2 or more. over/under (total) → whether both
  teams together score more or fewer goals than the line. shots-on-goal
  prop (prop_sog) → whether one player takes more or fewer shots on goal
  than the line.
- push → the result lands exactly on the line (6 goals on a 6.0 line):
  the stake comes back, nobody wins. void → the bet is cancelled (game
  postponed, player did not play): the stake comes back.
- cash out → taking the book's offer to settle a bet early for a set
  amount.
- bonus bet → a bet staked with promo credit instead of cash: a win
  pays the profit only (the credit is not returned), and a loss costs no
  cash.

What it does:
- record_slip(): a single bet or a parlay, with its legs, in one
  transaction (betting.slips + betting.slip_legs). Bettor labels come from
  .env BETTORS (comma list, default "bettor 1,bettor 2"), never from code.
- settle_slips(): decides every undecided leg it can from final scores
  and box scores (raw.games, raw.skater_games), then settles each OPEN
  slip whose result is known. `pipeline.py daily` runs it after the paper
  settlement. Legs of market 'other' are settled by hand
  (set_leg_result).
- Settlement rules:
  * moneyline: the final score, overtime and shootout included (the NHL
    score already counts the shootout winner's extra goal).
  * puck line: (side's goals - other side's goals) + handicap: above 0
    wins, below 0 loses, exactly 0 pushes (only possible on a whole-number
    handicap).
  * totals: home + away goals (a shootout counts as one goal for the
    winner, as books count it) against the line; equal is a push.
  * shots on goal: the player's box-score shots against the line; equal
    is a push; a player with no box-score row in a game whose box score is
    loaded did not play, so the leg is void (books void it too).
  * a postponed (PPD) or cancelled (CNCL) game voids its legs.
  * parlay: any losing leg loses the slip. Pushed and void legs count as
    a factor of 1 (they drop out): the payout is the combined price with
    those legs' own prices divided out. When a pushed or void leg has no
    price of its own the slip stays OPEN and needs settling by hand. If
    every leg pushed or voided, the stake comes back (PUSH, or VOID when
    every leg voided).
- Balances (balance_table): per bettor and platform, deposits -
  withdrawals + bonuses + adjustments + settled P/L - stakes of open
  bets: the cash the platform should show. Paper slips (practice bets)
  are kept out. A bonus bet's stake is not cash, so it never leaves the
  balance; only its winnings come in.
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from math import prod
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import text

from betting.engine import decimal_odds
from config.migrate import ensure_schema
from config.settings import engine as db

logger = logging.getLogger("nhl.betting.ledger")

MARKETS = ("ml", "pl", "total", "prop_sog", "other")
SIDES = {"ml": ("HOME", "AWAY"), "pl": ("HOME", "AWAY"),
         "total": ("OVER", "UNDER"), "prop_sog": ("OVER", "UNDER")}
NEEDS_LINE = ("pl", "total", "prop_sog")
SLIP_STATUSES = ("OPEN", "WON", "LOST", "PUSH", "VOID", "CASHED_OUT")
SETTLED_STATUSES = SLIP_STATUSES[1:]
LEG_RESULTS = ("WIN", "LOSS", "PUSH", "VOID")
TXN_KINDS = ("DEPOSIT", "WITHDRAWAL", "BONUS", "ADJUSTMENT")
VOID_STATES = ("PPD", "CNCL")
DEFAULT_BETTORS = ("bettor 1", "bettor 2")


def _split(value: Optional[str]) -> List[str]:
    """A comma list, trimmed, empties and repeats dropped, order kept."""
    out = []
    for part in (value or "").split(","):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


def bettors() -> List[str]:
    """The bettor labels from .env BETTORS (default "bettor 1,bettor 2")."""
    return _split(os.getenv("BETTORS")) or list(DEFAULT_BETTORS)


def configured_platforms() -> List[str]:
    """Platform names from .env PLATFORMS (optional; suggestions only, any
    name can be recorded)."""
    return _split(os.getenv("PLATFORMS"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Odds arithmetic (pure) ─────────────────────────────────────────

def american_from_decimal(dec: float) -> int:
    """Decimal odds -> American, rounded to a whole number. 2.5 -> +150,
    1.6667 -> -150. Even money (2.0) is +100."""
    if dec <= 1.0:
        raise ValueError(f"decimal odds must be above 1.0, got {dec}")
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return -int(round(100.0 / (dec - 1.0)))


def combined_price(prices: Iterable[int]) -> int:
    """A parlay's fair combined American price: the legs' decimal odds
    multiplied. [-110, -110] -> +264."""
    prices = list(prices)
    if not prices or any(p is None for p in prices):
        raise ValueError("every leg needs its odds to work out a combined price")
    return american_from_decimal(prod(decimal_odds(p) for p in prices))


def valid_american(price) -> bool:
    """American odds are -100 or lower, or +100 or higher."""
    try:
        return abs(int(price)) >= 100
    except (TypeError, ValueError):
        return False


# ── Leg settlement (pure) ──────────────────────────────────────────

def leg_result(market: str, side: str, line: Optional[float],
               home_score: Optional[int], away_score: Optional[int],
               is_final: bool, schedule_state: Optional[str] = None,
               player_shots: Optional[int] = None,
               box_loaded: bool = False) -> Optional[str]:
    """WIN / LOSS / PUSH / VOID for one leg, or None while it can't be
    decided (game not final, box score not loaded, market 'other').
    Scores are the final ones, overtime and shootout included."""
    if market == "other":
        return None
    if schedule_state in VOID_STATES:
        return "VOID"
    if not is_final or home_score is None or away_score is None:
        return None
    side = side.upper()

    def compare(value: float, against: float, over_side: str) -> str:
        if value == against:
            return "PUSH"
        return "WIN" if (value > against) == (side == over_side) else "LOSS"

    if market == "ml":
        if home_score == away_score:
            return None               # an NHL final is never tied: data problem
        home_won = home_score > away_score
        return "WIN" if home_won == (side == "HOME") else "LOSS"
    if market == "pl":
        margin = (home_score - away_score) if side == "HOME" else (away_score - home_score)
        return compare(margin + float(line), 0.0, side)
    if market == "total":
        return compare(float(home_score + away_score), float(line), "OVER")
    if market == "prop_sog":
        if not box_loaded:
            return None
        if player_shots is None:
            return "VOID"             # no box-score row: did not play
        return compare(float(player_shots), float(line), "OVER")
    raise ValueError(f"unknown market {market!r}")


# ── Slip settlement (pure) ─────────────────────────────────────────

@dataclass
class Outcome:
    status: str                   # OPEN or one of SETTLED_STATUSES
    payout: Optional[float]       # cash paid back (stake included); None while OPEN
    note: str = ""


def slip_outcome(stake: float, price_american: int, legs: List[tuple],
                 is_bonus_bet: bool = False) -> Outcome:
    """The slip's result from its legs' results.

    legs: (result, leg_price_american) per leg; result None = undecided.
    Any LOSS loses the slip at once. Otherwise it waits for every leg.
    PUSH/VOID legs drop out as a factor of 1: the slip's price is divided
    by their own decimal odds. A bonus bet pays the profit only, and
    nothing on a push, void or loss (the credit, not cash, comes back)."""
    stake = float(stake)
    results = [r for r, _ in legs]
    if "LOSS" in results:
        return Outcome("LOST", 0.0)
    if any(r is None for r in results):
        return Outcome("OPEN", None)
    if "WIN" not in results:
        status = "VOID" if all(r == "VOID" for r in results) else "PUSH"
        return Outcome(status, 0.0 if is_bonus_bet else round(stake, 2))
    dropped = [p for r, p in legs if r in ("PUSH", "VOID")]
    if any(p is None for p in dropped):
        return Outcome("OPEN", None,
                       "a pushed or void leg has no odds of its own, so the "
                       "reduced payout can't be worked out: settle it by hand")
    prices = [p for _, p in legs]
    if (len(legs) > 1 and None not in prices
            and combined_price(prices) == int(price_american)):
        # the standard parlay price: the winning legs' odds multiplied,
        # exactly as the book recomputes it (no rounding through American)
        factor = prod(decimal_odds(p) for r, p in legs if r == "WIN")
    else:
        # a single bet, a boosted price, or legs without their own odds
        factor = decimal_odds(price_american) / prod(decimal_odds(p) for p in dropped)
    note = (f"{len(dropped)} leg(s) pushed or voided and dropped out"
            if dropped else "")
    if is_bonus_bet:
        return Outcome("WON", round(stake * (factor - 1.0), 2), note)
    return Outcome("WON", round(stake * factor, 2), note)


def slip_pnl(status: str, stake: float, payout: Optional[float],
             is_bonus_bet: bool = False) -> Optional[float]:
    """Cash profit or loss of a settled slip; None while OPEN. A bonus
    bet's stake was not cash, so its P/L is its payout."""
    if status == "OPEN" or payout is None or pd.isna(payout):
        return None
    return round(float(payout) - (0.0 if is_bonus_bet else float(stake)), 2)


# ── Balances (pure) ────────────────────────────────────────────────

_TXN_SIGN = {"DEPOSIT": 1.0, "WITHDRAWAL": -1.0, "BONUS": 1.0, "ADJUSTMENT": 1.0}

BALANCE_COLUMNS = ["bettor", "platform", "deposits", "withdrawals", "bonuses",
                   "adjustments", "settled_pl", "open_stakes", "balance",
                   "bets", "won", "lost", "staked", "roi"]


def balance_table(txns: pd.DataFrame, slips: pd.DataFrame) -> pd.DataFrame:
    """Per bettor and platform: money in and out, settled P/L, stakes still
    riding, and the resulting balance. Paper slips are left out.

    txns: bettor, platform, kind, amount. slips: bettor, platform, stake,
    status, payout, is_bonus_bet, is_paper.
    balance = deposits - withdrawals + bonuses + adjustments + settled_pl
              - open_stakes (cash stakes of OPEN slips).
    roi → return on investment: the P/L of settled cash bets (bonus bets
    and stake-back pushes/voids left out) / the cash staked on them."""
    keys = ["bettor", "platform"]
    txns = txns if txns is not None else pd.DataFrame(columns=keys + ["kind", "amount"])
    slips = slips if slips is not None else pd.DataFrame(
        columns=keys + ["stake", "status", "payout", "is_bonus_bet", "is_paper"])
    if not slips.empty:
        slips = slips[~slips["is_paper"].astype(bool)].copy()

    rows = {}

    def row(b, p):
        return rows.setdefault((b, p), {
            "bettor": b, "platform": p, "deposits": 0.0, "withdrawals": 0.0,
            "bonuses": 0.0, "adjustments": 0.0, "settled_pl": 0.0,
            "open_stakes": 0.0, "bets": 0, "won": 0, "lost": 0, "staked": 0.0,
            "staked_pl": 0.0})

    column = {"DEPOSIT": "deposits", "WITHDRAWAL": "withdrawals",
              "BONUS": "bonuses", "ADJUSTMENT": "adjustments"}
    for t in txns.itertuples():
        row(t.bettor, t.platform)[column[t.kind]] += float(t.amount)
    for s in slips.itertuples():
        r = row(s.bettor, s.platform)
        bonus = bool(s.is_bonus_bet)
        r["bets"] += 1
        if s.status == "OPEN":
            if not bonus:
                r["open_stakes"] += float(s.stake)
            continue
        pnl = slip_pnl(s.status, s.stake, s.payout, bonus) or 0.0
        r["settled_pl"] += pnl
        if s.status in ("WON", "LOST"):
            r["won" if s.status == "WON" else "lost"] += 1
        if not bonus and s.status not in ("PUSH", "VOID"):
            r["staked"] += float(s.stake)
            r["staked_pl"] += pnl

    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        return pd.DataFrame(columns=BALANCE_COLUMNS)
    out["balance"] = (out["deposits"] - out["withdrawals"] + out["bonuses"]
                      + out["adjustments"] + out["settled_pl"] - out["open_stakes"])
    out["roi"] = [round(pl / st, 4) if st else None
                  for pl, st in zip(out["staked_pl"], out["staked"])]
    money = ["deposits", "withdrawals", "bonuses", "adjustments", "settled_pl",
             "open_stakes", "balance", "staked"]
    out[money] = out[money].round(2)
    return out[BALANCE_COLUMNS].sort_values(keys).reset_index(drop=True)


def running_pl(slips: pd.DataFrame) -> pd.DataFrame:
    """Settled real slips in settlement order with each bettor's running
    P/L (cum_pl) and everyone's together (cum_pl_all)."""
    cols = ["settled_at", "slip_id", "bettor", "platform", "pnl", "cum_pl", "cum_pl_all"]
    if slips is None or slips.empty:
        return pd.DataFrame(columns=cols)
    s = slips[(~slips["is_paper"].astype(bool)) & (slips["status"] != "OPEN")].copy()
    if s.empty:
        return pd.DataFrame(columns=cols)
    s["pnl"] = [slip_pnl(st, sk, po, bool(b)) for st, sk, po, b in
                zip(s["status"], s["stake"], s["payout"], s["is_bonus_bet"])]
    s = s.sort_values(["settled_at", "slip_id"]).reset_index(drop=True)
    s["cum_pl"] = s.groupby("bettor")["pnl"].cumsum().round(2)
    s["cum_pl_all"] = s["pnl"].cumsum().round(2)
    return s[cols]


# ── Descriptions (pure) ────────────────────────────────────────────

def describe_leg(market: str, side: str, line, away: Optional[str] = None,
                 home: Optional[str] = None, player: Optional[str] = None) -> str:
    """One leg in plain words: 'BOS @ TOR: TOR win', 'Over 6.5 goals'."""
    away, home, player = (None if v is None or (isinstance(v, float) and pd.isna(v))
                          else v for v in (away, home, player))
    game = f"{away} @ {home}: " if away and home else ""
    side_u = (side or "").upper()
    team = home if side_u == "HOME" else away
    ln = None if line is None or pd.isna(line) else float(line)
    if market == "ml":
        return f"{game}{team or side_u.title()} win"
    if market == "pl":
        return f"{game}{team or side_u.title()} {ln:+g} (puck line)"
    if market == "total":
        return f"{game}{side_u.title()} {ln:g} goals"
    if market == "prop_sog":
        return f"{game}{player or 'player'} {side_u.lower()} {ln:g} shots on goal"
    return f"{game}{side}"


# ── Recording ──────────────────────────────────────────────────────

@dataclass
class LegInput:
    market: str
    side: str
    game_id: Optional[int] = None
    line: Optional[float] = None
    price_american: Optional[int] = None
    player_id: Optional[int] = None
    rec_id: Optional[int] = None


def validate_slip(bettor: str, platform: str, stake, legs: List[LegInput],
                  price_american: Optional[int] = None) -> tuple:
    """(slip price, problems in plain English). Pure. The slip price is
    the one given, else the single leg's odds, else the parlay's legs
    multiplied."""
    problems = []
    if bettor not in bettors():
        problems.append(f"Unknown bettor {bettor!r}: the bettors are "
                        f"{', '.join(bettors())} (set BETTORS in .env).")
    if not (platform or "").strip():
        problems.append("Pick or type the platform the bet was placed on.")
    elif len(platform.strip()) > 40:
        problems.append("Platform names are at most 40 characters.")
    try:
        if stake is None or float(stake) <= 0:
            problems.append("The stake must be more than $0.")
    except (TypeError, ValueError):
        problems.append("The stake must be a number.")
    if not legs:
        problems.append("A bet needs at least one leg.")
    for i, leg in enumerate(legs, start=1):
        if leg.market not in MARKETS:
            problems.append(f"Leg {i}: unknown market {leg.market!r}.")
            continue
        if leg.market != "other":
            if leg.game_id is None:
                problems.append(f"Leg {i}: pick the game.")
            if leg.side.upper() not in SIDES[leg.market]:
                problems.append(f"Leg {i}: side must be one of "
                                f"{', '.join(SIDES[leg.market])}.")
        elif not (leg.side or "").strip():
            problems.append(f"Leg {i}: describe the bet.")
        if leg.market in NEEDS_LINE and (leg.line is None or pd.isna(leg.line)):
            problems.append(f"Leg {i}: this bet needs a line, e.g. 6.5 or -1.5.")
        if leg.market == "prop_sog" and leg.player_id is None:
            problems.append(f"Leg {i}: pick the player.")
        if leg.price_american is not None and not valid_american(leg.price_american):
            problems.append(f"Leg {i}: American odds are -100 or lower, or "
                            f"+100 or higher.")
    if price_american is not None and not valid_american(price_american):
        problems.append("The bet's odds are -100 or lower, or +100 or higher.")
    if problems:
        return None, problems

    if price_american is not None:
        price = int(price_american)
        if len(legs) == 1 and legs[0].price_american is not None \
                and int(legs[0].price_american) != price:
            problems.append("The single bet's odds and its leg's odds differ: "
                            "enter one or make them match.")
    elif len(legs) == 1:
        price = legs[0].price_american
        if price is None:
            problems.append("Enter the odds.")
    else:
        try:
            price = combined_price(l.price_american for l in legs)
        except ValueError:
            price = None
            problems.append("Enter the parlay's combined odds, or the odds of "
                            "every leg.")
    return (None if problems else int(price)), problems


_INSERT_SLIP = text("""
    INSERT INTO betting.slips
        (bettor, platform, placed_at, stake, price_american, is_parlay,
         is_bonus_bet, notes, is_paper, created_at)
    VALUES (:bettor, :platform, :placed_at, :stake, :price, :is_parlay,
            :bonus, :notes, :paper, :now)
    RETURNING slip_id
""")
_INSERT_LEG = text("""
    INSERT INTO betting.slip_legs
        (slip_id, leg_no, game_id, market, side, line, price_american,
         player_id, rec_id)
    VALUES (:slip, :leg_no, :game_id, :market, :side, :line, :price,
            :player_id, :rec_id)
""")


def record_slip(bettor: str, platform: str, stake: float, legs: List[LegInput],
                price_american: Optional[int] = None,
                placed_at: Optional[datetime] = None, notes: Optional[str] = None,
                is_paper: bool = False, is_bonus_bet: bool = False,
                conn=None) -> int:
    """Write one slip and its legs in one transaction; returns slip_id.
    Raises ValueError (with every problem, in plain English) when the
    input is incomplete."""
    price, problems = validate_slip(bettor, platform, stake, legs, price_american)
    if problems:
        raise ValueError(" ".join(problems))
    ensure_schema()
    now = _utc_now()
    params = {"bettor": bettor, "platform": platform.strip(),
              "placed_at": placed_at or now, "stake": round(float(stake), 2),
              "price": price, "is_parlay": len(legs) > 1,
              "bonus": bool(is_bonus_bet), "notes": (notes or "").strip() or None,
              "paper": bool(is_paper), "now": now}

    def write(c):
        slip_id = c.execute(_INSERT_SLIP, params).scalar()
        for n, leg in enumerate(legs, start=1):
            leg_price = leg.price_american
            if leg_price is None and len(legs) == 1:
                leg_price = price
            c.execute(_INSERT_LEG, {
                "slip": slip_id, "leg_no": n,
                "game_id": None if leg.game_id is None else int(leg.game_id),
                "market": leg.market,
                "side": leg.side.strip() if leg.market == "other" else leg.side.upper(),
                "line": None if leg.line is None or pd.isna(leg.line) else float(leg.line),
                "price": None if leg_price is None else int(leg_price),
                "player_id": None if leg.player_id is None else int(leg.player_id),
                "rec_id": None if leg.rec_id is None else int(leg.rec_id)})
        return int(slip_id)

    if conn is not None:
        slip_id = write(conn)
    else:
        with db.begin() as c:
            slip_id = write(c)
    logger.info(f"Recorded slip {slip_id}: {bettor} on {platform.strip()}, "
                f"{len(legs)} leg(s), stake {float(stake):.2f} at {price:+d}"
                + (" (paper)" if is_paper else ""))
    return slip_id


def record_txn(bettor: str, platform: str, kind: str, amount: float,
               note: Optional[str] = None, ts: Optional[datetime] = None) -> int:
    """One deposit, withdrawal, bonus or adjustment; returns txn_id.
    Deposits, withdrawals and bonuses are positive amounts (the kind gives
    the direction); an adjustment is signed (+ adds to the balance)."""
    kind = kind.upper()
    problems = []
    if bettor not in bettors():
        problems.append(f"Unknown bettor {bettor!r}.")
    if not (platform or "").strip():
        problems.append("Pick or type the platform.")
    if kind not in TXN_KINDS:
        problems.append(f"Kind must be one of {', '.join(TXN_KINDS)}.")
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        problems.append("The amount must be a number.")
        amount = None
    if amount is not None:
        if kind == "ADJUSTMENT" and amount == 0:
            problems.append("An adjustment can't be $0.")
        elif kind != "ADJUSTMENT" and amount <= 0:
            problems.append("The amount must be more than $0 (the kind says "
                            "which way the money goes).")
    if problems:
        raise ValueError(" ".join(problems))
    ensure_schema()
    with db.begin() as c:
        return int(c.execute(text("""
            INSERT INTO betting.bankroll_txns (bettor, platform, ts, kind, amount, note)
            VALUES (:b, :p, :ts, :k, :a, :n) RETURNING txn_id
        """), {"b": bettor, "p": platform.strip(), "ts": ts or _utc_now(),
               "k": kind, "a": amount, "n": (note or "").strip() or None}).scalar())


# ── Settlement (database) ──────────────────────────────────────────

_UNDECIDED_LEGS = text("""
    SELECT l.slip_id, l.leg_no, l.market, l.side, l.line, l.player_id,
           g.home_score, g.away_score, g.schedule_state,
           (g.game_state IN ('FINAL', 'OFF') AND g.home_score IS NOT NULL) AS is_final,
           sg.shots AS player_shots,
           EXISTS (SELECT 1 FROM raw.skater_games b
                   WHERE b.game_id = l.game_id) AS box_loaded
    FROM betting.slip_legs l
    JOIN raw.games g ON g.game_id = l.game_id
    LEFT JOIN raw.skater_games sg
           ON sg.game_id = l.game_id AND sg.player_id = l.player_id
    WHERE l.result IS NULL AND l.market <> 'other'
      AND ((g.game_state IN ('FINAL', 'OFF') AND g.home_score IS NOT NULL)
           OR g.schedule_state IN ('PPD', 'CNCL'))
    ORDER BY l.slip_id, l.leg_no
""")


def _settle_open_slips(conn, slip_ids: Optional[List[int]] = None) -> dict:
    """Re-evaluate OPEN slips (all, or the given ones) from their legs."""
    where = "s.status = 'OPEN'"
    params = {}
    if slip_ids is not None:
        where += " AND s.slip_id = ANY(:ids)"
        params["ids"] = [int(i) for i in slip_ids]
    rows = conn.execute(text(f"""
        SELECT s.slip_id, s.stake, s.price_american, s.is_bonus_bet,
               l.result, l.price_american AS leg_price
        FROM betting.slips s JOIN betting.slip_legs l USING (slip_id)
        WHERE {where}
        ORDER BY s.slip_id, l.leg_no
    """), params).fetchall()
    by_slip = {}
    for r in rows:
        by_slip.setdefault(r.slip_id, {"row": r, "legs": []})["legs"].append(
            (r.result, r.leg_price))
    counts = {s: 0 for s in SETTLED_STATUSES}
    now = _utc_now()
    for slip_id, item in by_slip.items():
        r = item["row"]
        out = slip_outcome(float(r.stake), int(r.price_american), item["legs"],
                           bool(r.is_bonus_bet))
        if out.status == "OPEN":
            if out.note:
                logger.warning(f"  slip {slip_id}: {out.note}")
            continue
        conn.execute(text("""
            UPDATE betting.slips SET status = :st, payout = :po, settled_at = :now
            WHERE slip_id = :id AND status = 'OPEN'
        """), {"st": out.status, "po": out.payout, "now": now, "id": slip_id})
        counts[out.status] += 1
        logger.info(f"  slip {slip_id}: {out.status}, payout {out.payout:.2f}"
                    + (f" ({out.note})" if out.note else ""))
    return counts


def settle_slips() -> dict:
    """Decide every undecided leg whose game is final (or postponed or
    cancelled), then settle every OPEN slip whose result is now known.
    Returns {'legs': n, 'WON': n, 'LOST': n, ...}. Idempotent."""
    ensure_schema()
    now = _utc_now()
    n_legs = 0
    with db.begin() as conn:
        for leg in conn.execute(_UNDECIDED_LEGS).fetchall():
            result = leg_result(
                leg.market, leg.side,
                None if leg.line is None else float(leg.line),
                leg.home_score, leg.away_score, bool(leg.is_final),
                leg.schedule_state,
                None if leg.player_shots is None else int(leg.player_shots),
                bool(leg.box_loaded))
            if result is None:
                continue
            conn.execute(text("""
                UPDATE betting.slip_legs SET result = :r, settled_at = :now
                WHERE slip_id = :s AND leg_no = :n AND result IS NULL
            """), {"r": result, "now": now, "s": leg.slip_id, "n": leg.leg_no})
            n_legs += 1
        counts = _settle_open_slips(conn)
    counts["legs"] = n_legs
    settled = sum(v for k, v in counts.items() if k != "legs")
    logger.info(f"Bet ledger: {n_legs} leg(s) decided, {settled} slip(s) settled"
                if n_legs or settled else "Bet ledger: nothing to settle")
    return counts


def set_leg_result(slip_id: int, leg_no: int, result: Optional[str]) -> dict:
    """Set (or clear, with None) one leg's result by hand, e.g. a market
    'other' leg or a correction, then work the slip's result out again
    from its legs. A cashed-out slip keeps its cash-out, and a slip whose
    result was set by hand (settle_by_hand) keeps that result and payout:
    the returned counts then hold kept_by_hand=True. Setting the slip back
    to OPEN by hand lets its legs decide again."""
    if result is not None and result not in LEG_RESULTS:
        raise ValueError(f"result must be one of {', '.join(LEG_RESULTS)} or empty")
    ensure_schema()
    with db.begin() as conn:
        n = conn.execute(text("""
            UPDATE betting.slip_legs SET result = :r, settled_at = :at
            WHERE slip_id = :s AND leg_no = :n
        """), {"r": result, "at": None if result is None else _utc_now(),
               "s": int(slip_id), "n": int(leg_no)}).rowcount
        if not n:
            raise ValueError(f"slip {slip_id} has no leg {leg_no}")
        kept = conn.execute(text("""
            SELECT settled_by_hand AND status <> 'OPEN' FROM betting.slips WHERE slip_id = :s
        """), {"s": int(slip_id)}).scalar()
        if kept:
            logger.info(f"slip {slip_id}: leg {leg_no} saved; the slip keeps the "
                        f"result set by hand")
            return {**{s: 0 for s in SETTLED_STATUSES}, "kept_by_hand": True}
        conn.execute(text("""
            UPDATE betting.slips SET status = 'OPEN', payout = NULL, settled_at = NULL
            WHERE slip_id = :s AND status NOT IN ('OPEN', 'CASHED_OUT')
        """), {"s": int(slip_id)})
        return {**_settle_open_slips(conn, [slip_id]), "kept_by_hand": False}


def settle_by_hand(slip_id: int, status: str, payout: Optional[float] = None) -> None:
    """Override a slip's status and payout: a cash-out (status CASHED_OUT,
    payout = the amount taken), a book's own ruling, or back to OPEN
    (payout cleared). A WON/CASHED_OUT slip needs the payout; LOST pays
    0 (any other payout is refused); PUSH/VOID pay the stake back (0 for a
    bonus bet) unless given. The slip is marked settled_by_hand, so a later
    leg correction (set_leg_result) keeps this result; OPEN clears the
    mark, and the legs decide again."""
    status = status.upper()
    if status not in SLIP_STATUSES:
        raise ValueError(f"status must be one of {', '.join(SLIP_STATUSES)}")
    ensure_schema()
    with db.begin() as conn:
        row = conn.execute(text("""
            SELECT stake, is_bonus_bet FROM betting.slips WHERE slip_id = :id
        """), {"id": int(slip_id)}).fetchone()
        if row is None:
            raise ValueError(f"no slip {slip_id}")
        if status == "OPEN":
            payout, settled = None, None
        else:
            settled = _utc_now()
            if payout is None:
                if status in ("WON", "CASHED_OUT"):
                    raise ValueError(f"a {status} slip needs the payout amount")
                payout = 0.0 if status == "LOST" or row.is_bonus_bet else float(row.stake)
            if float(payout) < 0:
                raise ValueError("a payout can't be negative")
            if status == "LOST" and float(payout) != 0:
                raise ValueError("a lost bet pays nothing: leave the payout empty or 0 "
                                 "(for money back, use Cashed out, Push or Void)")
        conn.execute(text("""
            UPDATE betting.slips SET status = :st, payout = :po, settled_at = :at,
                                     settled_by_hand = :hand
            WHERE slip_id = :id
        """), {"st": status, "po": None if payout is None else round(float(payout), 2),
               "at": settled, "hand": status != "OPEN", "id": int(slip_id)})


def delete_slip(slip_id: int) -> bool:
    """Remove a slip recorded by mistake (its legs go with it)."""
    ensure_schema()
    with db.begin() as conn:
        return bool(conn.execute(text("DELETE FROM betting.slips WHERE slip_id = :id"),
                                 {"id": int(slip_id)}).rowcount)


def delete_txn(txn_id: int) -> bool:
    """Remove a deposit/withdrawal/bonus/adjustment recorded by mistake."""
    ensure_schema()
    with db.begin() as conn:
        return bool(conn.execute(text(
            "DELETE FROM betting.bankroll_txns WHERE txn_id = :id"),
            {"id": int(txn_id)}).rowcount)


# ── Reading ────────────────────────────────────────────────────────

def load_slips() -> pd.DataFrame:
    """Every slip, newest first, with its cash P/L (pnl; NaN while OPEN)."""
    ensure_schema()
    with db.connect() as conn:
        s = pd.read_sql(text("""
            SELECT slip_id, bettor, platform, placed_at, stake, price_american,
                   is_parlay, is_bonus_bet, status, payout, settled_at, notes,
                   is_paper, settled_by_hand,
                   (SELECT COUNT(*) FROM betting.slip_legs l
                    WHERE l.slip_id = s.slip_id) AS n_legs
            FROM betting.slips s
            ORDER BY placed_at DESC, slip_id DESC
        """), conn)
    for c in ("stake", "payout"):
        s[c] = s[c].astype(float)
    s["pnl"] = pd.to_numeric(pd.Series(
        [slip_pnl(st, sk, po, bool(b)) for st, sk, po, b in
         zip(s["status"], s["stake"], s["payout"], s["is_bonus_bet"])],
        index=s.index, dtype="object"), errors="coerce").astype(float)
    return s


def load_legs(slip_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Legs with their game, player and a plain-words description."""
    ensure_schema()
    where, params = "", {}
    if slip_ids is not None:
        where, params = "WHERE l.slip_id = ANY(:ids)", {"ids": [int(i) for i in slip_ids]}
    with db.connect() as conn:
        legs = pd.read_sql(text(f"""
            SELECT l.slip_id, l.leg_no, l.game_id, l.market, l.side, l.line,
                   l.price_american, l.player_id, l.rec_id, l.result,
                   g.date AS game_date, g.away_team, g.home_team,
                   g.away_score, g.home_score, p.full_name AS player
            FROM betting.slip_legs l
            LEFT JOIN raw.games g ON g.game_id = l.game_id
            LEFT JOIN raw.players p ON p.player_id = l.player_id
            {where}
            ORDER BY l.slip_id, l.leg_no
        """), conn, params=params)
    legs["bet"] = [describe_leg(m, s, ln, a, h, pl) for m, s, ln, a, h, pl in zip(
        legs["market"], legs["side"], legs["line"], legs["away_team"],
        legs["home_team"], legs["player"])]
    return legs


def load_txns() -> pd.DataFrame:
    ensure_schema()
    with db.connect() as conn:
        t = pd.read_sql(text("""
            SELECT txn_id, bettor, platform, ts, kind, amount, note
            FROM betting.bankroll_txns ORDER BY ts DESC, txn_id DESC
        """), conn)
    t["amount"] = t["amount"].astype(float)
    return t


def balances() -> pd.DataFrame:
    """balance_table() over everything recorded."""
    return balance_table(load_txns(), load_slips())


def known_platforms() -> List[str]:
    """PLATFORMS from .env plus every platform already used, in that order."""
    names = configured_platforms()
    ensure_schema()
    with db.connect() as conn:
        used = [r[0] for r in conn.execute(text("""
            SELECT platform FROM betting.slips UNION
            SELECT platform FROM betting.bankroll_txns ORDER BY 1
        """))]
    return names + [p for p in used if p not in names]


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m betting.ledger",
        description="Settle the bet ledger's open slips from final scores and "
                    "box scores, then print the balances per bettor and platform.")
    parser.add_argument("--no-settle", action="store_true",
                        help="only print the balances")
    args = parser.parse_args(argv)
    if not args.no_settle:
        settle_slips()
    table = balances()
    print("No deposits or bets recorded yet." if table.empty else table.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
