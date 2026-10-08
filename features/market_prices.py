"""
features/market_prices.py
The two-way moneyline market for every season we hold real prices for,
one row per game, from one place.

Terms (each explained once):
- Moneyline → a bet on who wins, overtime and shootout included. A
  two-way price has only those two outcomes (ESPN's older Unibet lines are
  three-way, with a regulation draw as a third outcome, so they are NOT a
  price the bettors could take and are left out here).
- American odds → -120 means risk 120 to win 100; +110 means risk 100 to
  win 110. Decimal odds → what one unit returns in total if it wins
  (-120 → 1.833, +110 → 2.10). "Best price" = the highest decimal odds.
- No-vig probability → a book's implied probabilities with its built-in
  fee (the vig, or margin) taken out. Here: each side's implied
  probability divided by the two sides' sum (the proportional method, the
  same as betting/engine.no_vig_probs and the live pick job).
- Consensus no-vig → the MEDIAN of the books' no-vig home probabilities,
  the same rule the live pick job uses (betting/recommend.summarize_market).
- Pinnacle → the sharpest book (lowest margin, takes big bets), so its
  no-vig price is the hardest yardstick; reported next to the consensus.
- Closing price → the last price before puck drop. Morning price → the
  10:00 Central snapshot of the game day (bought for the bet-timing study).

Sources:
- 2024-25: raw.odds_history (The Odds API's paid history). Closing
  snapshots were bought per cluster of start times, so one game appears
  in several snapshots taken before it; each book's CLOSING quote is its
  last two-sided quote in a snapshot taken before the game's
  raw.games.start_time_utc and at most CLOSE_MAX_LEAD before it (a book
  missing from the final snapshots would otherwise contribute a quote
  from days earlier). The MORNING quote comes from the snapshot bought
  with purpose 'morning' (raw.odds_history_fetches) on the game's own
  date, taken before puck drop.
- 2025-26: raw.historical_odds provider 'DraftKings' (ESPN's closing
  line; no timestamps, one book). Consensus = DraftKings no-vig, no
  Pinnacle, best price = DraftKings. ESPN has no lines for October and
  November 2025, so about 1,014 of 1,394 games are priced.
- Nothing else: Unibet (2020-21..2023-24) is three-way; Kalshi and
  Polymarket have no history anywhere in the database (raw.odds_history
  was bought with 10 sportsbooks, none of them exchanges).

In-play rows (the shared contamination rule, inplay_mask below). ESPN's
stored line for a game is meant to be its closing line, but for 106
Unibet games in 2023-24 it was captured DURING the game (→ in play: the
price already reflects the score), so it leaks the result into any model
that uses it: those 106 games score a log loss of 0.45 instead of about
0.66, and the 34 with a moneyline of 1,000 or more score 0.14. Every
reader of raw.historical_odds as a pre-game market (the production
market feature in features/build_vectors.py, models/lgbm.py, the
simulation fallback in betting/recommend.py and the moneyline v3
diagnostics) treats such a row as no market at all.

Point-in-time: every quote used for a game was captured before that
game's start (tests/test_market_prices.py deletes and rewrites rows at or
after puck drop and checks nothing changes).

Everything except the two loaders is pure (no database), so it is tested
without one. The loaders only read.
"""
from datetime import timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from features.util import american_implied_prob

SHARP_BOOK = "pinnacle"
# Licensed US sportsbooks in the history list (the ones a bettor in a
# legal US state can use). lowvig, betonlineag and bovada are offshore.
US_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us",
            "betrivers", "espnbet")
OFFSHORE_BOOKS = ("lowvig", "betonlineag", "bovada")
EXCHANGE_BOOKS = ("kalshi", "polymarket", "novig")
CLOSE_MAX_LEAD = timedelta(minutes=90)
MORNING_MAX_LEAD = timedelta(hours=18)

SUMMARY_COLUMNS = ["nv_consensus", "nv_pinnacle", "n_books",
                   "best_home_price", "best_home_book",
                   "best_away_price", "best_away_book",
                   "best_us_home_price", "best_us_home_book",
                   "best_us_away_price", "best_us_away_book",
                   "quote_ts"]


def decimal_from_american(price) -> float:
    a = float(price)
    return 1.0 + (100.0 / -a if a < 0 else a / 100.0)


def _naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


# ── Pure: in-play rows in raw.historical_odds ──────────────────────
#
# A row is treated as captured in play when ANY of these holds. Each rule
# comes from what the 2023-24 Unibet rows show (docs/historical_odds.md,
# "In-play rows"):
#  1. a moneyline of 1,000 or more on either side (an implied probability
#     of 91% or more): no NHL game is priced like that before puck drop,
#     and the largest pre-game line in every other season is far smaller
#     (DraftKings 2025-26 tops out at -470);
#  2. both sides' implied probabilities summing to under 0.75: both teams
#     long at once, a tied game late on (pre-game three-way Unibet sums
#     sit near 0.83, two-way DraftKings near 1.04);
#  3. a total (→ the over/under goals line) under 5 or at 8 and above:
#     pre-game NHL totals sit between 5 and 7.5; 2, 3.5, 8.5, 10.5 or 13
#     only happen once goals have been scored or time has run;
#  4. a known stretch: Unibet from 2024-04-08 to the end of 2023-24,
#     where 42 of the 97 rows break rules 1-3 (from 0 to 3 a week before),
#     so the rows that look normal there cannot be trusted either.
# Applied to every provider; rules 1-3 flag no DraftKings row and no
# Unibet row before 2023-24. Together they flag 106 Unibet 2023-24 games.

INPLAY_MAX_ABS_ML = 1000          # rule 1: |moneyline| at or above
INPLAY_MIN_IMPLIED_SUM = 0.75     # rule 2: both sides long
INPLAY_TOTAL_RANGE = (5.0, 8.0)   # rule 3: pre-game totals in [5, 8); NULL kept
INPLAY_STRETCHES = {("Unibet", 20232024): pd.Timestamp("2024-04-08")}  # rule 4


def inplay_mask(df: pd.DataFrame) -> pd.Series:
    """True for a raw.historical_odds row that looks captured in play.
    df: home_ml, away_ml, over_under, season, date and (optional)
    provider; without a provider column rule 4 applies to every row of
    the listed season."""
    h, a = df["home_ml"].astype(float), df["away_ml"].astype(float)
    big = (h.abs() >= INPLAY_MAX_ABS_ML) | (a.abs() >= INPLAY_MAX_ABS_ML)
    s = h.map(american_implied_prob) + a.map(american_implied_prob)
    ou = df["over_under"].astype(float)
    lo, hi = INPLAY_TOTAL_RANGE
    bad_total = ou.notna() & ((ou < lo) | (ou >= hi))
    stretch = pd.Series(False, index=df.index)
    d = pd.to_datetime(df["date"])
    for (provider, season), start in INPLAY_STRETCHES.items():
        hit = (df["season"] == season) & (d >= start)
        if "provider" in df.columns:
            hit &= df["provider"] == provider
        stretch |= hit
    return (big | (s < INPLAY_MIN_IMPLIED_SUM) | bad_total | stretch).astype(bool)


def clear_market(X: np.ndarray, names: list, game_ids, drop_ids) -> np.ndarray:
    """Copy of a feature matrix with the listed games set to 'no market'
    (market_home_prob 0.5, market_available 0), the same values
    features/build_vectors.py writes for a game without a line."""
    X = X.copy()
    rows = pd.Series(np.asarray(game_ids)).isin(set(drop_ids)).to_numpy()
    X[rows, names.index("market_home_prob")] = 0.5
    X[rows, names.index("market_available")] = 0.0
    return X


# ── Pure: quotes ────────────────────────────────────────────────────

def two_sided(rows: pd.DataFrame) -> pd.DataFrame:
    """Long h2h rows (snapshot_ts, game_id, book, side, price[, purpose])
    -> one row per (snapshot_ts, game_id, book) with home_price and
    away_price, both sides required."""
    keys = ["snapshot_ts", "game_id", "book"] + (
        ["purpose"] if "purpose" in rows.columns else [])
    r = rows[rows["side"].isin(["home", "away"])]
    w = (r.pivot_table(index=keys, columns="side", values="price",
                       aggfunc="last")
         .reset_index())
    for c in ("home", "away"):
        if c not in w.columns:
            w[c] = np.nan
    w = w.dropna(subset=["home", "away"]).rename(
        columns={"home": "home_price", "away": "away_price"})
    w.columns.name = None
    return w


def add_no_vig(q: pd.DataFrame) -> pd.DataFrame:
    """novig_home (proportional) and overround (implied sum minus 1)."""
    q = q.copy()
    ph = q["home_price"].map(american_implied_prob).astype(float)
    pa = q["away_price"].map(american_implied_prob).astype(float)
    q["novig_home"] = ph / (ph + pa)
    q["overround"] = ph + pa - 1.0
    return q


def latest_quotes(quotes: pd.DataFrame, starts: pd.Series,
                  max_lead: timedelta) -> pd.DataFrame:
    """Each (game, book)'s last quote taken strictly before the game's
    start and no more than max_lead before it. starts: game_id -> naive
    UTC start time. Games without a start time get no quote."""
    q = quotes.copy()
    q["start"] = q["game_id"].map(starts)
    q = q[q["start"].notna()]
    q = q[(q["snapshot_ts"] < q["start"])
          & (q["start"] - q["snapshot_ts"] <= max_lead)]
    q = q.sort_values(["game_id", "book", "snapshot_ts"])
    return q.groupby(["game_id", "book"], sort=False).tail(1).reset_index(drop=True)


def morning_quotes(quotes: pd.DataFrame, starts: pd.Series,
                   game_dates: pd.Series,
                   max_lead: timedelta = MORNING_MAX_LEAD,
                   tz: str = "America/Chicago") -> pd.DataFrame:
    """Each (game, book)'s quote from a purpose='morning' snapshot taken
    on the game's own schedule date (local to tz), before puck drop and
    within max_lead of it. game_dates: game_id -> date."""
    q = quotes[quotes["purpose"] == "morning"].copy()
    local_day = (q["snapshot_ts"].dt.tz_localize("UTC").dt.tz_convert(tz)
                 .dt.date)
    gd = pd.to_datetime(q["game_id"].map(game_dates)).dt.date
    q = q[local_day.to_numpy() == gd.to_numpy()]
    return latest_quotes(q, starts, max_lead)


def _best(g: pd.DataFrame, side: str, books: Optional[Iterable[str]]):
    col = f"{side}_price"
    pool = g if books is None else g[g["book"].isin(list(books))]
    if pool.empty:
        return None, None
    dec = pool[col].map(decimal_from_american)
    # ties go to the alphabetically first book, so results are reproducible
    top = pool[dec == dec.max()].sort_values("book").iloc[0]
    return int(top[col]), top["book"]


def summarize_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """One row per game: consensus (median) no-vig home probability over
    every quoting book, Pinnacle's no-vig home probability (NaN if it has
    no quote), the number of books, the best price for each side over all
    books and over the licensed US books, and the newest quote time."""
    if quotes.empty:
        return pd.DataFrame(columns=["game_id"] + SUMMARY_COLUMNS)
    q = quotes if "novig_home" in quotes.columns else add_no_vig(quotes)
    rows = []
    for gid, g in q.groupby("game_id", sort=True):
        pin = g.loc[g["book"] == SHARP_BOOK, "novig_home"]
        bh, bhb = _best(g, "home", None)
        ba, bab = _best(g, "away", None)
        uh, uhb = _best(g, "home", US_BOOKS)
        ua, uab = _best(g, "away", US_BOOKS)
        rows.append({
            "game_id": gid,
            "nv_consensus": float(g["novig_home"].median()),
            "nv_pinnacle": float(pin.iloc[0]) if len(pin) else np.nan,
            "n_books": int(g["book"].nunique()),
            "best_home_price": bh, "best_home_book": bhb,
            "best_away_price": ba, "best_away_book": bab,
            "best_us_home_price": uh, "best_us_home_book": uhb,
            "best_us_away_price": ua, "best_us_away_book": uab,
            "quote_ts": (g["snapshot_ts"].max()
                         if "snapshot_ts" in g and g["snapshot_ts"].notna().any()
                         else pd.NaT),
        })
    return pd.DataFrame(rows)


def wide_book_prices(quotes: pd.DataFrame, books: Iterable[str],
                     prefix: str = "") -> pd.DataFrame:
    """Each listed book's home and away price per game, as columns
    {prefix}{book}_home / {prefix}{book}_away (NaN when it has no quote)."""
    books = list(books)
    out = pd.DataFrame({"game_id": sorted(quotes["game_id"].unique())})
    for b in books:
        sub = quotes.loc[quotes["book"] == b, ["game_id", "home_price", "away_price"]]
        sub = sub.rename(columns={"home_price": f"{prefix}{b}_home",
                                  "away_price": f"{prefix}{b}_away"})
        out = out.merge(sub, on="game_id", how="left")
    return out


def assemble_market(close_q: pd.DataFrame, morning_q: Optional[pd.DataFrame],
                    meta: pd.DataFrame, source: str,
                    books: Iterable[str] = ()) -> pd.DataFrame:
    """Close summary (+ per-book close prices) + morning summary with an
    m_ prefix, joined to meta (game_id, season, date, home_win)."""
    close = summarize_quotes(close_q)
    if len(close) and books:
        close = close.merge(wide_book_prices(close_q, books), on="game_id",
                            how="left")
    if morning_q is not None and len(morning_q):
        m = summarize_quotes(morning_q)
        m = m.rename(columns={c: f"m_{c}" for c in m.columns if c != "game_id"})
        close = close.merge(m, on="game_id", how="left")
    out = meta.merge(close, on="game_id", how="inner")
    out["source"] = source
    return out


# ── Loaders (read-only) ─────────────────────────────────────────────

def load_inplay_game_ids(conn) -> list:
    """game_ids whose raw.historical_odds moneyline looks captured in
    play (inplay_mask), sorted."""
    df = pd.read_sql(text("""
        SELECT h.game_id, h.provider, g.season, g.date, h.home_ml,
               h.away_ml, h.over_under
        FROM raw.historical_odds h JOIN raw.games g USING (game_id)
        WHERE h.home_ml IS NOT NULL AND h.away_ml IS NOT NULL
    """), conn)
    if df.empty:
        return []
    return sorted(int(g) for g in df.loc[inplay_mask(df), "game_id"].unique())


def load_history_quotes(conn, seasons: Optional[Iterable[int]] = None) -> tuple:
    """(two-sided quotes with novig, starts, game dates, meta) from
    raw.odds_history h2h rows, labelled with their fetch purpose."""
    season_filter = "AND g.season = ANY(:seasons)" if seasons else ""
    params = {"seasons": list(map(int, seasons))} if seasons else {}
    rows = pd.read_sql(text(f"""
        SELECT o.snapshot_ts, o.game_id, o.book, o.side, o.price,
               COALESCE(f.purpose, 'close') AS purpose
        FROM raw.odds_history o
        JOIN raw.games g USING (game_id)
        LEFT JOIN (SELECT DISTINCT requested_ts, purpose
                   FROM raw.odds_history_fetches
                   WHERE purpose IN ('close', 'morning')) f
               ON f.requested_ts = o.requested_ts
        WHERE o.market = 'h2h' {season_filter}
    """), conn, params=params)
    meta = pd.read_sql(text(f"""
        SELECT g.game_id, g.season, g.date, g.start_time_utc,
               (g.home_score > g.away_score) AS home_win
        FROM raw.games g
        WHERE g.game_id IN (SELECT DISTINCT game_id FROM raw.odds_history
                            WHERE game_id IS NOT NULL) {season_filter}
    """), conn, params=params)
    rows["snapshot_ts"] = _naive_utc(rows["snapshot_ts"])
    meta["start_time_utc"] = _naive_utc(meta["start_time_utc"])
    quotes = add_no_vig(two_sided(rows))
    starts = meta.set_index("game_id")["start_time_utc"]
    dates = meta.set_index("game_id")["date"]
    return quotes, starts, dates, meta.drop(columns="start_time_utc")


def load_espn_dk(conn) -> tuple:
    """(quotes, meta) for DraftKings closing lines from raw.historical_odds."""
    df = pd.read_sql(text("""
        SELECT h.game_id, g.season, g.date,
               (g.home_score > g.away_score) AS home_win,
               h.home_ml AS home_price, h.away_ml AS away_price
        FROM raw.historical_odds h JOIN raw.games g USING (game_id)
        WHERE h.provider = 'DraftKings'
          AND h.home_ml IS NOT NULL AND h.away_ml IS NOT NULL
          AND g.game_state IN ('FINAL', 'OFF')
    """), conn)
    quotes = add_no_vig(df[["game_id", "home_price", "away_price"]]
                        .assign(book="draftkings", snapshot_ts=pd.NaT))
    return quotes, df[["game_id", "season", "date", "home_win"]]


def load_market_prices(conn=None, books: Iterable[str] = US_BOOKS + (SHARP_BOOK,),
                       include_dk: bool = True) -> pd.DataFrame:
    """Every priced game, one row each: season, date, home_win, source
    ('odds_api' or 'espn_draftkings'), the close summary (SUMMARY_COLUMNS),
    each listed book's closing prices (odds_api only), and the morning
    summary as m_* columns where a morning snapshot exists. A game with
    both sources keeps the Odds API row."""
    from config.settings import engine
    own = conn is None
    conn = conn or engine.connect()
    try:
        quotes, starts, dates, meta = load_history_quotes(conn)
        close_q = latest_quotes(quotes, starts, CLOSE_MAX_LEAD)
        morn_q = morning_quotes(quotes, starts, dates)
        parts = [assemble_market(close_q, morn_q, meta, "odds_api", books)]
        if include_dk:
            dk_q, dk_meta = load_espn_dk(conn)
            dk = assemble_market(dk_q, None, dk_meta, "espn_draftkings")
            dk = dk[~dk["game_id"].isin(parts[0]["game_id"])]
            parts.append(dk)
    finally:
        if own:
            conn.close()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_close_quotes(conn) -> pd.DataFrame:
    """Every book's closing quote per game (long form) for 2024-25 plus the
    DraftKings line for 2025-26; for per-book analysis."""
    quotes, starts, _, _ = load_history_quotes(conn)
    dk_q, _ = load_espn_dk(conn)
    close_q = latest_quotes(quotes, starts, CLOSE_MAX_LEAD)
    dk_q = dk_q[~dk_q["game_id"].isin(close_q["game_id"])]
    return pd.concat([close_q, dk_q], ignore_index=True)
