"""
dashboard/today.py — the "📅 Today" tab of dashboard/app.py.

Three things, each explained on screen in plain English:
- Pending picks: for every pick the system wants to bet, the book and
  price to take, the side spelled as the team ("TOR win"), puck drop in
  your time zone, the edge in percentage points and the stake in dollars.
- Prices by book: for every upcoming game and side, the latest pre-game
  price at every book in raw.odds_snapshots (at most MAX_ODDS_AGE_HOURS
  old), with the best price anywhere and the best price at each bettor's
  own books marked. Each bettor's books come from .env BETTOR_<N>_BOOKS
  (N = the bettor's place in BETTORS, from 1), else BETTABLE_BOOKS, else
  every book.
- The stake limits in use (betting/engine.py, .env-overridable).

The pure helpers (labels, best prices, the per-game table) are tested in
tests/test_today.py without Streamlit or a database.
"""
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from betting import engine as limits
from betting.engine import decimal_odds
from betting.ledger import bettors
from config.settings import engine, local_today, to_local
from features.util import american_implied_prob

EVERY_BOOK = frozenset()        # "no restriction": every book counts
BEST_MARK = "★ best anywhere"

# Exchanges quote prices without their trading fee (README "Known issues")
EXCHANGES = frozenset({"kalshi", "polymarket", "novig"})


# ── Pure helpers ───────────────────────────────────────────────────

def bet_label(side: str, away: str, home: str) -> str:
    """A pick's side spelled out: HOME -> "TOR win" (the home team),
    AWAY -> the away team, OVER/UNDER as words."""
    s = str(side).upper()
    if s == "HOME":
        return f"{home} win"
    if s == "AWAY":
        return f"{away} win"
    return s.title()


def american(price) -> str:
    """American odds with their sign: 120 -> "+120", -150 -> "-150"."""
    if price is None or pd.isna(price):
        return "—"
    return f"{int(price):+d}"


def edge_points(edge) -> str:
    """An edge (a probability difference) in percentage points:
    0.069 -> "+6.9 pts"."""
    if edge is None or pd.isna(edge):
        return "—"
    return f"{float(edge) * 100:+.1f} pts"


def dollars(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${float(x):,.2f}"


def chance(p) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{float(p):.1%}"


def puck_drop(ts) -> str:
    """A start time (aware, or naive UTC) in the user's zone."""
    if ts is None or pd.isna(ts):
        return "TBD"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return to_local(t.to_pydatetime()).strftime("%a %b %d %I:%M %p %Z")


def _books(value: Optional[str]) -> frozenset:
    return frozenset(b.strip().lower() for b in (value or "").split(",")
                     if b.strip())


def bettor_books(environ: Mapping[str, str] = None,
                 labels: List[str] = None) -> Dict[str, Tuple[frozenset, str]]:
    """Each bettor's books: {label: (books, where the list came from)}.
    Bettor N (from 1, in BETTORS order) uses BETTOR_<N>_BOOKS when set,
    else BETTABLE_BOOKS, else every book (an empty set). A
    BETTOR_<N>_BOOKS past the end of BETTORS adds "bettor N"."""
    env = os.environ if environ is None else environ
    labels = list(bettors() if labels is None else labels)
    numbered = list(enumerate(labels, start=1))
    extra = sorted({int(m.group(1)) for k in env
                    if (m := re.fullmatch(r"BETTOR_(\d+)_BOOKS", k))
                    and int(m.group(1)) > len(labels)})
    numbered += [(n, f"bettor {n}") for n in extra]
    shared = _books(env.get("BETTABLE_BOOKS"))
    out = {}
    for n, label in numbered:
        own = _books(env.get(f"BETTOR_{n}_BOOKS"))
        if own:
            out[label] = (own, f"BETTOR_{n}_BOOKS")
        elif shared:
            out[label] = (shared, "BETTABLE_BOOKS")
        else:
            out[label] = (EVERY_BOOK, "every book")
    return out


def side_prices(snaps: pd.DataFrame) -> pd.DataFrame:
    """Book snapshots (game_id, book_name, captured_at, home_price,
    away_price; one row per game and book) -> one row per game, side and
    book: game_id, side (HOME/AWAY), book, price, decimal, captured_at."""
    cols = ["game_id", "side", "book", "price", "decimal", "captured_at"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    for side, col in (("AWAY", "away_price"), ("HOME", "home_price")):
        part = snaps[["game_id", "book_name", "captured_at", col]].rename(
            columns={"book_name": "book", col: "price"})
        part = part[part["price"].notna()].assign(side=side)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["book"] = out["book"].str.lower()
    out["price"] = out["price"].astype(int)
    out["decimal"] = out["price"].map(decimal_odds)
    return out[cols]


def _best(rows: pd.DataFrame, books: frozenset) -> Tuple[Optional[int], List[str]]:
    """(best price, the books offering it) among `books` (empty = all)."""
    pool = rows if not books else rows[rows["book"].isin(books)]
    if pool.empty:
        return None, []
    top = pool["decimal"].max()
    hit = pool[pool["decimal"] >= top - 1e-12]
    return int(hit["price"].iloc[0]), sorted(hit["book"])


def fair_chances(snaps: pd.DataFrame) -> Dict[int, float]:
    """{game_id: the home side's fair chance}: each book's two prices with
    the margin removed, then the median across books (as the daily job
    measures the market)."""
    out = {}
    both = snaps.dropna(subset=["home_price", "away_price"])
    for gid, g in both.groupby("game_id"):
        ph = g["home_price"].map(american_implied_prob)
        pa = g["away_price"].map(american_implied_prob)
        out[int(gid)] = float((ph / (ph + pa)).median())
    return out


def best_price_table(games: pd.DataFrame, snaps: pd.DataFrame,
                     who: Dict[str, Tuple[frozenset, str]]) -> pd.DataFrame:
    """One row per upcoming game and side: puck drop, game, bet, the
    market's fair chance, the best price anywhere (and where), the best
    at each bettor's books, and how many books quote it. games: game_id,
    start_time_utc, away_team, home_team."""
    prices = side_prices(snaps)
    fair = fair_chances(snaps)
    rows = []
    for g in games.itertuples():
        mine = prices[prices["game_id"] == g.game_id]
        if mine.empty:
            continue
        for side in ("AWAY", "HOME"):
            s = mine[mine["side"] == side]
            if s.empty:
                continue
            p_home = fair.get(int(g.game_id))
            p = None if p_home is None else (p_home if side == "HOME"
                                             else 1.0 - p_home)
            price, where = _best(s, EVERY_BOOK)
            row = {"Puck drop (your time)": puck_drop(g.start_time_utc),
                   "Game": f"{g.away_team} @ {g.home_team}",
                   "Bet": bet_label(side, g.away_team, g.home_team),
                   "Fair chance": chance(p),
                   "Best price anywhere": f"{american(price)} at {', '.join(where)}"}
            for label, (books, _) in who.items():
                bp, bw = _best(s, books)
                row[f"Best for {label}"] = (f"{american(bp)} at {', '.join(bw)}"
                                            if bp is not None else "not offered")
            row["Books quoting"] = len(s)
            rows.append(row)
    return pd.DataFrame(rows)


def game_book_table(game, snaps: pd.DataFrame,
                    who: Dict[str, Tuple[frozenset, str]]) -> pd.DataFrame:
    """Every book's latest price on one game, both sides: Bet, Book,
    Price, Updated, and Best (★ best anywhere; "best for bettor 1" when it
    is the best of that bettor's books). Sorted by side, best first."""
    prices = side_prices(snaps[snaps["game_id"] == game.game_id])
    rows = []
    for side in ("AWAY", "HOME"):
        s = prices[prices["side"] == side].sort_values(
            ["decimal", "book"], ascending=[False, True])
        if s.empty:
            continue
        top, top_books = _best(s, EVERY_BOOK)
        best_for = {label: _best(s, books)[1] for label, (books, _) in who.items()}
        for r in s.itertuples():
            marks = []
            if r.book in top_books:
                marks.append(BEST_MARK)
            marks += [f"best for {label}" for label, bw in best_for.items()
                      if r.book in bw]
            note = " (exchange: price leaves out its fee)" \
                if r.book in EXCHANGES else ""
            rows.append({"Bet": bet_label(side, game.away_team, game.home_team),
                         "Book": r.book + note,
                         "Price": american(r.price),
                         "Updated (your time)": puck_drop(r.captured_at),
                         "Best": "; ".join(marks)})
    return pd.DataFrame(rows, columns=["Bet", "Book", "Price",
                                       "Updated (your time)", "Best"])


def picks_table(recs: pd.DataFrame) -> pd.DataFrame:
    """Pending recommendations -> the table to show. recs: start_time_utc,
    away_team, home_team, side, best_book, best_price, model_prob,
    implied_prob_novig, edge_pct, recommended_stake."""
    return pd.DataFrame([{
        "Puck drop (your time)": puck_drop(r.start_time_utc),
        "Game": f"{r.away_team} @ {r.home_team}",
        "Bet": bet_label(r.side, r.away_team, r.home_team),
        "Book": r.best_book or "—",
        "Price": american(r.best_price),
        "Model's chance": chance(r.model_prob),
        "Market's fair chance": chance(r.implied_prob_novig),
        "Edge": edge_points(r.edge_pct),
        "Stake": dollars(r.recommended_stake),
    } for r in recs.itertuples()], columns=[
        "Puck drop (your time)", "Game", "Bet", "Book", "Price",
        "Model's chance", "Market's fair chance", "Edge", "Stake"])


def limits_in_use(bankroll: float) -> List[Tuple[str, str, str]]:
    """(label, value, plain-English help) for each stake limit in use."""
    def share(pct):
        return f"{pct:.0%} (${bankroll * pct:,.0f})"
    return [
        ("Per bet", share(limits.MAX_STAKE_PCT),
         "The most on any one bet (MAX_STAKE_PCT). Default 2%."),
        ("Per day", share(limits.MAX_DAILY_PCT),
         "The most on all of one day's picks together (MAX_DAILY_PCT). "
         "Default 10%."),
        ("Per game", share(limits.MAX_GAME_STAKE_PCT),
         "The most on one game, every bet on it together "
         "(MAX_GAME_STAKE_PCT). Default 4%."),
        ("Bets per game", str(limits.MAX_BETS_PER_GAME),
         "The most bets on one game, any market (MAX_BETS_PER_GAME). "
         "Default 3."),
    ]


# ── Database reads ─────────────────────────────────────────────────

def _read(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


PENDING_SQL = """
    SELECT g.start_time_utc, g.away_team, g.home_team, r.side, r.best_book,
           r.best_price, r.model_prob, r.implied_prob_novig, r.edge_pct,
           r.recommended_stake
    FROM betting.recommendations r JOIN raw.games g USING (game_id)
    WHERE r.status = 'PENDING'
    ORDER BY g.start_time_utc NULLS LAST, r.edge_pct DESC LIMIT 50"""

# Upcoming = not started, not postponed/suspended/cancelled (as the daily
# job's slate), from today to two days ahead in the user's calendar
UPCOMING_SQL = """
    SELECT game_id, date, start_time_utc, away_team, home_team FROM raw.games
    WHERE game_state NOT IN ('FINAL', 'OFF', 'LIVE', 'CRIT')
      AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
      AND (start_time_utc IS NULL OR start_time_utc > NOW())
      AND date BETWEEN :today AND :today + 2
    ORDER BY date, start_time_utc, game_id LIMIT 40"""

# Each book's latest moneyline quote taken before puck drop and no older
# than the cutoff (captured_at is naive UTC)
PRICES_SQL = """
    SELECT DISTINCT ON (s.game_id, s.book_name)
           s.game_id, s.book_name, s.captured_at, s.home_price, s.away_price
    FROM raw.odds_snapshots s JOIN raw.games g USING (game_id)
    WHERE s.market_type = 'ml' AND s.game_id = ANY(:ids)
      AND s.home_price IS NOT NULL AND s.away_price IS NOT NULL
      AND s.captured_at BETWEEN :cutoff AND :asof
      AND (g.start_time_utc IS NULL
           OR s.captured_at < (g.start_time_utc AT TIME ZONE 'UTC'))
    ORDER BY s.game_id, s.book_name, s.captured_at DESC"""


def load_book_prices(game_ids: List[int], max_age_hours: float,
                     read: Callable = _read) -> pd.DataFrame:
    asof = datetime.now(timezone.utc).replace(tzinfo=None)
    if not game_ids:
        return pd.DataFrame(columns=["game_id", "book_name", "captured_at",
                                     "home_price", "away_price"])
    return read(PRICES_SQL, {"ids": [int(g) for g in game_ids],
                             "cutoff": asof - timedelta(hours=max_age_hours),
                             "asof": asof})


# ── The tab ────────────────────────────────────────────────────────

def _highlight_best(row) -> list:
    on = BEST_MARK in str(row.get("Best", ""))
    return ["background-color: rgba(46, 160, 67, 0.22)" if on else ""] * len(row)


def render(st, read: Callable = _read) -> None:
    """Draw the tab. st is the streamlit module; read(sql, params) runs a
    query (app.py passes its cached one)."""
    from betting.recommend import BANKROLL, MAX_ODDS_AGE_HOURS

    st.subheader("Pending picks")
    st.caption(
        "The bets the system recommends today. **Bet** is the team to back "
        "to win (overtime and shootout count). **Book** is the sportsbook "
        "or exchange with the best price when the pick was made, and "
        "**Price** is that price in American odds → +120 means a $100 bet "
        "wins $120; -150 means you bet $150 to win $100. **Edge** is the "
        "model's chance minus the market's fair chance (the book's margin "
        "removed), in percentage points → +4.0 pts means the model gives the "
        "team 4 more chances in 100 than the market does. **Stake** is the "
        "suggested bet in dollars: a quarter of the Kelly size → the bet "
        "size that grows a bankroll fastest if the chances are right, within "
        "the limits below. A pick is still a paper bet → tracked, not proven: "
        "see README 'Status'.")
    recs = read(PENDING_SQL)
    if recs.empty:
        st.info("No pending picks — either no games are scheduled, the "
                "daily run hasn't made today's picks yet, or no game "
                "cleared the minimum edge.")
    else:
        st.dataframe(picks_table(recs), width="stretch", hide_index=True)

    st.markdown("**Stake limits in use**")
    cols = st.columns(4)
    for col, (label, value, why) in zip(cols, limits_in_use(BANKROLL)):
        col.metric(label, value, help=why)
    st.caption(
        f"Shares of the bankroll setting (BANKROLL = ${BANKROLL:,.0f}). A "
        "pick that would pass a limit is skipped, weakest edges first. "
        "Change them in .env (MAX_STAKE_PCT, MAX_DAILY_PCT, "
        "MAX_GAME_STAKE_PCT, MAX_BETS_PER_GAME; a share is a number above 0 "
        "and at most 1, so 0.02 = 2%). `python -m betting.montecarlo` "
        "simulates thousands of seasons to show what bigger limits do.")
    for w in limits.cap_warnings():
        st.warning(w)

    st.divider()
    st.subheader("Prices by book")
    games = read(UPCOMING_SQL, {"today": local_today()})
    who = bettor_books()
    st.caption(
        "Every book's latest price before puck drop for the next few days' "
        f"games, from the odds snapshots (none older than "
        f"{MAX_ODDS_AGE_HOURS:g} hours). Books pay different prices for the "
        "same bet, so taking the best one you can use is free money → "
        "**line shopping**. ★ marks the best price anywhere; \"best for "
        "bettor N\" marks the best at the books that bettor uses. **Fair "
        "chance** is the market's view with the margin removed (the median "
        "across books).")
    st.caption("Each bettor's books: " + "; ".join(
        f"{label}: {', '.join(sorted(books)) if books else 'every book'} "
        f"(from {src})" for label, (books, src) in who.items())
        + ". Set BETTOR_1_BOOKS, BETTOR_2_BOOKS, … in .env (Odds API book "
          "keys, comma-separated, in BETTORS order); without one, "
          "BETTABLE_BOOKS is used.")
    if games.empty:
        st.caption("No upcoming games in the next 48 hours.")
        return
    snaps = load_book_prices(games["game_id"].tolist(), MAX_ODDS_AGE_HOURS,
                             read)
    if snaps.empty:
        st.info("No book prices yet for these games: the odds snapshots "
                "(daily run at 9:00, midday odds run, closes before puck "
                "drop) haven't stored any within the age limit.")
        unpriced = games
    else:
        st.dataframe(best_price_table(games, snaps, who), width="stretch",
                     hide_index=True)
        for g in games.itertuples():
            table = game_book_table(g, snaps, who)
            if table.empty:
                continue
            with st.expander(f"{puck_drop(g.start_time_utc)}  "
                             f"{g.away_team} @ {g.home_team}: every book"):
                st.dataframe(table.style.apply(_highlight_best, axis=1),
                             width="stretch", hide_index=True)
        unpriced = games[~games["game_id"].isin(snaps["game_id"])]
    if not unpriced.empty:
        st.markdown("**Games with no prices yet**")
        st.dataframe(pd.DataFrame({
            "Puck drop (your time)": unpriced["start_time_utc"].map(puck_drop),
            "Game": unpriced["away_team"] + " @ " + unpriced["home_team"],
        }), width="stretch", hide_index=True)
