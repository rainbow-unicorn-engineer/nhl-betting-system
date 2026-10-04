"""
betting/news.py
The news monitor: know team news before the market prices it in.

Terms:
  starting goalie   the goalie who starts the game. The biggest single piece
                    of team news in hockey: a backup in net moves a
                    moneyline → the bet on who wins, by several cents
  confirmed         the team or a beat reporter has said who starts; before
                    that Daily Faceoff lists a projection ("Likely",
                    "Unconfirmed")
  line / PP unit    see ingestion/dailyfaceoff_lines.py: which forwards play
                    together, and who plays on the first power-play unit →
                    the five players sent out when the other team takes a
                    penalty
  re-score          run the win/loss model again with the new information
                    and decide whether the game is now worth a bet
  fair chance       a team's chance to win implied by a price with the
                    book's margin → the built-in fee, removed

`python pipeline.py news --due` runs every 15 minutes. On a game day,
from NEWS_START_HOUR (8:00) local time until the day's last puck drop, it:
  1. refreshes Daily Faceoff's starting goalies (ingestion/dailyfaceoff.py,
     one request), the line combinations of every team still to play
     (ingestion/dailyfaceoff_lines.py, at most one request per team, with
     its own politeness gap) and ESPN's injury list (one request);
  2. compares each with what the previous run saw (raw.news_state) and
     writes what changed to raw.news_events, one row per change:
       STARTER_CONFIRMED  a starter is now confirmed
       STARTER_CHANGED    a different goalie is now expected to start (or a
                          confirmed one is no longer confirmed)
       PLAYER_OUT         a player left the projected lineup, joined ESPN's
                          injury list, or his injury status got worse
       PLAYER_IN          a player joined the projected lineup, left ESPN's
                          injury list, or his status improved
       LINE_CHANGE        a forward line or defence pair changed
       PP_UNIT_CHANGE     a power-play unit changed
     The first time a source is seen there is nothing to compare with, so
     it is only remembered (no events);
  3. for every game with starter news (the only news the win/loss model
     uses) that has not started, checks whether the market already moved:
     it takes a free snapshot of the NHL's own odds feed
     (ingestion/nhl_odds.py) and compares each book's fair chance now with
     the same book's fair chance at the last paid odds snapshot
     (raw.odds_snapshots). Only a price from this run's snapshot counts as
     now, and only one taken with the paid snapshot as then; without both
     the move can't be ruled out. A move of NEWS_MOVE_PTS (1.0) percentage
     points or more counts as moved;
  4. re-scores those games' date through the normal recommend path when a
     game has no moneyline pick yet and today's daily run has finished
     (raw.pipeline_runs, config/runs.py). Before that, last night's box
     scores, Elo and rolling stats are not loaded, and an issued pick is
     frozen, so the news is only recorded. Issued picks stay frozen. A new pick
     is allowed only for games whose price did NOT move: the stored price
     is the last paid snapshot's, and after a move it may no longer be on
     offer, so a pick at it would be a phantom edge (and would show a fake
     CLV → closing line value: how much better our price was than the last
     one before puck drop). A moved or unchecked game waits for the next
     paid snapshot (the odds run, or tomorrow's daily run).
It never calls The Odds API (credits). Every step is non-fatal and logged.

Without --due (`python pipeline.py news`) it runs now, whatever the time.

ESPN's list: the news run compares it but does not overwrite raw.injuries,
which keeps the day's first list (taken by the daily run): a list saved in
the evening would hold injuries from that day's afternoon games under the
same date, which a model must never see before those games. The day's
changes are in raw.news_events with their times instead. When today has no
list yet, the news run saves the first one.

Settings (.env): NEWS_START_HOUR (8), NEWS_MIN_GAP_MINUTES (14, so a
catch-up run right after a scheduled one does nothing), NEWS_MOVE_PTS
(1.0). The lineup politeness settings are in ingestion/dailyfaceoff_lines.py.
A NEWS_START_HOUR earlier than the daily run's time (9:00) only records
news: no pick comes from news until today's daily run has finished.
"""
import argparse
import json
import logging
import os
import unicodedata
from datetime import date as date_cls, datetime, time as time_cls, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text

from config.settings import engine, local_now, local_today, to_local

logger = logging.getLogger("nhl.betting.news")

KINDS = ("STARTER_CONFIRMED", "STARTER_CHANGED", "PLAYER_OUT", "PLAYER_IN",
         "LINE_CHANGE", "PP_UNIT_CHANGE")
STARTER_KINDS = ("STARTER_CONFIRMED", "STARTER_CHANGED")

DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.news_events (
        event_id        BIGSERIAL PRIMARY KEY,
        ts              TIMESTAMPTZ NOT NULL,
        game_id         BIGINT REFERENCES raw.games(game_id),
        game_date       DATE,
        team            VARCHAR(3) NOT NULL,
        kind            VARCHAR(20) NOT NULL CHECK (kind IN ('STARTER_CONFIRMED', 'STARTER_CHANGED', 'PLAYER_OUT', 'PLAYER_IN', 'LINE_CHANGE', 'PP_UNIT_CHANGE')),
        source          VARCHAR(20) NOT NULL,
        player_name     VARCHAR(80),
        player_id       INTEGER,
        detail          TEXT,
        previous        TEXT,
        current         TEXT,
        rescored        BOOLEAN,
        new_pick        BOOLEAN,
        market_moved    BOOLEAN,
        market_note     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_news_events_ts ON raw.news_events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_news_events_game ON raw.news_events(game_id)",
    """
    CREATE TABLE IF NOT EXISTS raw.news_state (
        source          VARCHAR(20) NOT NULL,
        team            VARCHAR(3) NOT NULL,
        state           JSONB NOT NULL,
        updated_at      TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (source, team)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw.news_runs (
        run_id          BIGSERIAL PRIMARY KEY,
        started_at      TIMESTAMPTZ NOT NULL,
        finished_at     TIMESTAMPTZ,
        events          INTEGER,
        notes           TEXT
    )
    """,
]

_table_ready = False


def ensure_table() -> None:
    """Create raw.news_events, raw.news_state and raw.news_runs if missing
    (once per process)."""
    global _table_ready
    if _table_ready:
        return
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    _table_ready = True


def _setting(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        if value >= 0:
            return value
    except ValueError:
        pass
    logger.error(f"{name}={raw!r} is not a number of 0 or more; using {default:g}")
    return default


# ── When to run (pure) ─────────────────────────────────────────────

def news_due(now: datetime, starts: Iterable[datetime],
             last_run: Optional[datetime] = None, start_hour: float = None,
             min_gap_minutes: float = None) -> Tuple[bool, str]:
    """Whether a scheduled (--due) run should work now, and why.
    now: aware local time; starts: today's puck drops (aware); last_run:
    when the previous run started. Due on a game day from start_hour local
    until the last puck drop, at most once per min_gap_minutes."""
    start_hour = _setting("NEWS_START_HOUR", 8) if start_hour is None else start_hour
    min_gap = (_setting("NEWS_MIN_GAP_MINUTES", 14) if min_gap_minutes is None
               else min_gap_minutes)
    starts = [s for s in starts if s is not None]
    if not starts:
        return False, "no game today"
    opens = datetime.combine(now.date(), time_cls(0), tzinfo=now.tzinfo) \
        + timedelta(hours=start_hour)
    last_drop = max(starts)
    if now < opens:
        return False, f"before {opens:%H:%M}, when the news checks start"
    if now > last_drop:
        return False, (f"after the day's last puck drop "
                       f"({to_local(last_drop):%H:%M})")
    if last_run is not None and now - last_run < timedelta(minutes=min_gap):
        mins = (now - last_run).total_seconds() / 60
        return False, f"the last news run was {mins:.0f} minute(s) ago"
    return True, (f"{len(starts)} game(s) today, last puck drop "
                  f"{to_local(last_drop):%H:%M}")


# ── Diffs (pure) ───────────────────────────────────────────────────

def _norm(name: Optional[str]) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    return " ".join("".join(c for c in s if not unicodedata.combining(c))
                    .casefold().split())


def _event(team, kind, source, detail, previous=None, current=None,
           player_name=None, player_id=None) -> dict:
    return {"team": team, "kind": kind, "source": source, "detail": detail,
            "previous": previous, "current": current,
            "player_name": player_name, "player_id": player_id}


def _starter_label(s: dict) -> str:
    return f"{s['goalie']} ({s.get('confirmation') or 'projected'})"


def diff_starters(prev: Dict[str, dict], cur: Dict[str, dict]) -> List[dict]:
    """Starter news per team. A state is {game_date, goalie, goalie_id,
    confirmation}. A new game date with a confirmed starter is
    STARTER_CONFIRMED; a different goalie for the same date is
    STARTER_CHANGED; the same goalie becoming 'Confirmed' is
    STARTER_CONFIRMED; a confirmed goalie losing that label is
    STARTER_CHANGED. A projection seen for the first time is no news."""
    events = []
    for team in sorted(cur):
        c, p = cur[team], prev.get(team)
        confirmed = c.get("confirmation") == "Confirmed"
        who = {"player_name": c["goalie"], "player_id": c.get("goalie_id")}
        if p is None or p.get("game_date") != c.get("game_date"):
            if confirmed:
                events.append(_event(team, "STARTER_CONFIRMED", "dailyfaceoff",
                                     f"{c['goalie']} is confirmed to start",
                                     None, _starter_label(c), **who))
            continue
        if _norm(p.get("goalie")) != _norm(c["goalie"]):
            events.append(_event(
                team, "STARTER_CHANGED", "dailyfaceoff",
                f"{c['goalie']} is now expected to start instead of {p['goalie']}"
                + (" (confirmed)" if confirmed else ""),
                _starter_label(p), _starter_label(c), **who))
        elif confirmed and p.get("confirmation") != "Confirmed":
            events.append(_event(team, "STARTER_CONFIRMED", "dailyfaceoff",
                                 f"{c['goalie']} is confirmed to start",
                                 _starter_label(p), _starter_label(c), **who))
        elif not confirmed and p.get("confirmation") == "Confirmed":
            events.append(_event(team, "STARTER_CHANGED", "dailyfaceoff",
                                 f"{c['goalie']} is no longer listed as confirmed",
                                 _starter_label(p), _starter_label(c), **who))
    return events


# ESPN statuses, least to most serious
_SEVERITY = {"day-to-day": 1, "questionable": 1, "doubtful": 2, "out": 2,
             "suspension": 2, "injured reserve": 3}


def _severity(status: Optional[str]) -> int:
    return _SEVERITY.get((status or "").strip().lower(), 1)


def _injury_text(p: dict) -> str:
    return " ".join(x for x in (p.get("status"), f"({p['injury']})" if p.get("injury")
                                else None) if x) or "listed"


def diff_injuries(prev: Dict[str, dict], cur: Dict[str, dict],
                  has_baseline: bool = True) -> List[dict]:
    """ESPN injury-list news per team. A team's state is {espn athlete id:
    {name, status, injury, player_id}}; a team missing from a side has no
    injured player there. New on the list or a more serious status:
    PLAYER_OUT; off the list or a less serious status: PLAYER_IN.
    has_baseline=False (the first run ever): no events."""
    if not has_baseline:
        return []
    events = []
    for team in sorted(set(prev) | set(cur)):
        before, now = prev.get(team) or {}, cur.get(team) or {}
        for key in sorted(set(before) | set(now)):
            b, n = before.get(key), now.get(key)
            if b is None:
                events.append(_event(team, "PLAYER_OUT", "espn",
                                     f"{n['name']} added to ESPN's injury list: "
                                     f"{_injury_text(n)}", "not listed",
                                     _injury_text(n), n["name"], n.get("player_id")))
            elif n is None:
                events.append(_event(team, "PLAYER_IN", "espn",
                                     f"{b['name']} is off ESPN's injury list",
                                     _injury_text(b), "not listed", b["name"],
                                     b.get("player_id")))
            elif (b.get("status") or "") != (n.get("status") or ""):
                worse = _severity(n.get("status")) > _severity(b.get("status"))
                events.append(_event(team, "PLAYER_OUT" if worse else "PLAYER_IN", "espn",
                                     f"{n['name']}: ESPN status "
                                     f"{b.get('status')} → {n.get('status')}",
                                     _injury_text(b), _injury_text(n), n["name"],
                                     n.get("player_id")))
    return events


def lineup_state(rows: Iterable[dict]) -> dict:
    """A team's lines (raw.lineups rows) -> {units: {unit: [names in slot
    order]}, ids: {name: player_id}, ir: [names], gtd: [names]}."""
    units: Dict[str, list] = {}
    ids, ir, gtd = {}, [], []
    for r in sorted(rows, key=lambda r: (r["unit"], r["slot"])):
        name = r["player_name"]
        if r.get("player_id") is not None:
            ids[name] = int(r["player_id"])
        if r["unit"] == "IR":
            ir.append(name)
            continue
        units.setdefault(r["unit"], []).append(name)
        if r.get("game_time_decision") and name not in gtd:
            gtd.append(name)
    return {"units": units, "ids": ids, "ir": ir, "gtd": gtd}


def _is_line(unit: str) -> bool:
    return len(unit) == 2 and unit[0] in "FD" and unit[1].isdigit()


def _dressed(state: dict) -> Dict[str, str]:
    """{player: his line} for the forwards and defencemen in the lines."""
    return {name: unit for unit, names in state.get("units", {}).items()
            if _is_line(unit) for name in names}


def diff_lineups(team: str, prev: Optional[dict], cur: dict) -> List[dict]:
    """Daily Faceoff lines news for one team (lineup_state dicts). prev
    None (never seen): no events."""
    if prev is None:
        return []
    events = []
    ids = {**prev.get("ids", {}), **cur.get("ids", {})}
    was, now = _dressed(prev), _dressed(cur)
    for name in sorted(set(was) - set(now)):
        why = (" (now on injured reserve)" if name in cur.get("ir", [])
               else "")
        events.append(_event(team, "PLAYER_OUT", "dailyfaceoff_lines",
                             f"{name} is no longer in the projected lineup{why}",
                             was[name], "IR" if why else "not in lineup",
                             name, ids.get(name)))
    for name in sorted(set(now) - set(was)):
        gtd = " (game-time decision)" if name in cur.get("gtd", []) else ""
        events.append(_event(team, "PLAYER_IN", "dailyfaceoff_lines",
                             f"{name} joins the projected lineup on {now[name]}{gtd}",
                             "not in lineup", now[name], name, ids.get(name)))
    pu, cu = prev.get("units", {}), cur.get("units", {})
    for unit in sorted(set(pu) | set(cu)):
        if unit == "G":
            continue        # starters come from the starting-goalies page
        before, after = pu.get(unit, []), cu.get(unit, [])
        if set(map(_norm, before)) == set(map(_norm, after)):
            continue
        if _is_line(unit):
            kind = "LINE_CHANGE"
            what = (f"forward line {unit[1]}" if unit[0] == "F"
                    else f"defence pair {unit[1]}")
        elif unit.startswith("PP"):
            kind, what = "PP_UNIT_CHANGE", f"power-play unit {unit[2:]}"
        else:
            continue        # penalty kill and other groups: stored, not news
        events.append(_event(team, kind, "dailyfaceoff_lines", f"{what} changed",
                             ", ".join(before) or "empty", ", ".join(after) or "empty"))
    return events


# ── Market check (pure core) ───────────────────────────────────────

def fair_home(home_price, away_price) -> Optional[float]:
    """The home side's no-vig chance from a pair of American prices."""
    from betting.engine import no_vig_probs
    if home_price is None or away_price is None:
        return None
    try:
        return float(no_vig_probs(float(home_price), float(away_price))[0])
    except Exception:
        return None


def price_move(before: Dict[str, tuple], now: Dict[str, tuple],
               threshold_pts: float = None) -> Tuple[Optional[bool], str]:
    """Did the market move? before/now: {book: (home_price, away_price)}.
    Compares each book's home fair chance in both; moved = the largest
    change is threshold_pts percentage points or more. None when no book
    is in both."""
    threshold = (_setting("NEWS_MOVE_PTS", 1.0) if threshold_pts is None
                 else threshold_pts) / 100
    changes = []
    for book in sorted(set(before) & set(now)):
        b, n = fair_home(*before[book]), fair_home(*now[book])
        if b is not None and n is not None:
            changes.append((abs(n - b), n - b, book, b, n, before[book], now[book]))
    if not changes:
        return None, ("no free price to compare with the last paid odds "
                      "snapshot, so a move can't be ruled out")
    _, delta, book, b, n, pb, pn = max(changes)
    moved = abs(delta) >= threshold - 1e-12
    return moved, (f"{'moved' if moved else 'not moved'}: at {book} the home "
                   f"team's fair chance went {b:.1%} → {n:.1%} ({delta * 100:+.1f} "
                   f"pts; prices {pb[0]:+d}/{pb[1]:+d} → {pn[0]:+d}/{pn[1]:+d}) "
                   f"since the last paid odds snapshot; {len(changes)} book(s) "
                   f"compared")


# ── Database: today, state, events ─────────────────────────────────

def todays_games(on_date: Optional[date_cls] = None) -> List[dict]:
    """Today's games (any state except postponed or cancelled):
    game_id, date, start_time_utc, home_team, away_team, started."""
    on_date = on_date or local_today()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT game_id, date, start_time_utc, home_team, away_team,
                   (game_state IN ('FINAL', 'OFF', 'LIVE', 'CRIT')
                    OR (start_time_utc IS NOT NULL AND start_time_utc <= NOW()))
                       AS started
            FROM raw.games
            WHERE date = :d AND game_type IN (2, 3)
              AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'CNCL')
            ORDER BY start_time_utc, game_id
        """), {"d": on_date}).mappings().fetchall()
    return [dict(r) for r in rows]


def last_run_start() -> Optional[datetime]:
    with engine.connect() as conn:
        return conn.execute(text("SELECT MAX(started_at) FROM raw.news_runs")).scalar()


def load_state(source: str) -> Dict[str, dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT team, state FROM raw.news_state WHERE source = :s
        """), {"s": source}).fetchall()
    return {t: (s if isinstance(s, dict) else json.loads(s)) for t, s in rows}


def save(source: str, states: Dict[str, dict], events: List[dict],
         games_by_team: Dict[str, dict], now: datetime) -> List[dict]:
    """Write the events (with each team's game today) and the new state
    in one transaction. Returns the events with event_id and game_id."""
    with engine.begin() as conn:
        for e in events:
            g = games_by_team.get(e["team"]) or {}
            e["game_id"], e["game_date"] = g.get("game_id"), g.get("date")
            e["event_id"] = conn.execute(text("""
                INSERT INTO raw.news_events
                    (ts, game_id, game_date, team, kind, source, player_name,
                     player_id, detail, previous, current)
                VALUES (:ts, :game_id, :game_date, :team, :kind, :source,
                        :player_name, :player_id, :detail, :previous, :current)
                RETURNING event_id
            """), {**{k: e.get(k) for k in (
                "game_id", "game_date", "team", "kind", "source", "player_name",
                "player_id", "detail", "previous", "current")}, "ts": now}).scalar()
        for team, state in states.items():
            conn.execute(text("""
                INSERT INTO raw.news_state (source, team, state, updated_at)
                VALUES (:s, :t, CAST(:state AS JSONB), :now)
                ON CONFLICT (source, team) DO UPDATE SET
                    state = EXCLUDED.state, updated_at = EXCLUDED.updated_at
            """), {"s": source, "t": team, "state": json.dumps(state, default=str),
                   "now": now})
    for e in events:
        logger.info(f"NEWS {e['team']} {e['kind']}: {e['detail']}")
    return events


# ── The three sources ──────────────────────────────────────────────

def check_starters(games: List[dict], games_by_team: dict, now: datetime) -> List[dict]:
    """Refresh Daily Faceoff's starting goalies for today's game dates and
    diff them with the last run's."""
    from ingestion.dailyfaceoff import ingest_starting_goalies
    dates = sorted({g["date"] for g in games})
    for d in dates:
        ingest_starting_goalies(d)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT team, game_date, goalie_name, goalie_id, confirmation
            FROM raw.starting_goalies WHERE game_date = ANY(:dates)
        """), {"dates": dates}).fetchall()
    cur = {t: {"game_date": str(d), "goalie": n, "goalie_id": gid, "confirmation": c}
           for t, d, n, gid, c in rows if t in games_by_team}
    events = diff_starters(load_state("dailyfaceoff"), cur)
    return save("dailyfaceoff", cur, events, games_by_team, now)


def check_lineups(games: List[dict], games_by_team: dict, now: datetime) -> List[dict]:
    """Refresh the lines of every team still to play today and diff each
    team's latest snapshot with the last run's."""
    from ingestion.dailyfaceoff_lines import ingest_lineups, latest_lineups
    to_play = {}
    for g in games:
        if not g["started"]:
            to_play[g["home_team"]] = g["start_time_utc"]
            to_play[g["away_team"]] = g["start_time_utc"]
    if not to_play:
        return []
    ingest_lineups(to_play, game_date=games[0]["date"])
    prev = load_state("dailyfaceoff_lines")
    states, events = {}, []
    for team, rows in latest_lineups(to_play).items():
        states[team] = lineup_state(rows)
        events += diff_lineups(team, prev.get(team), states[team])
    return save("dailyfaceoff_lines", states, events, games_by_team, now)


def check_injuries(games_by_team: dict, now: datetime) -> List[dict]:
    """Fetch ESPN's injury list and diff it with the last run's. Saves it
    as today's raw.injuries snapshot only when today has none yet."""
    from ingestion import espn_injuries as espn
    rows = espn.parse_injuries(espn.fetch_injuries())
    if not rows:
        logger.warning("ESPN injuries came back empty; no injury news this run")
        return []
    espn.resolve_player_ids(rows)
    today = local_today()
    espn.ensure_table()
    with engine.connect() as conn:
        have_today = conn.execute(text(
            "SELECT 1 FROM raw.injuries WHERE snapshot_date = :d LIMIT 1"),
            {"d": today}).first() is not None
    if not have_today:
        espn.write_injuries(rows, today)
        logger.info(f"ESPN injuries: saved today's first list ({len(rows)} players)")
    cur: Dict[str, dict] = {}
    for r in rows:
        if not r.get("team_abbrev"):
            continue
        cur.setdefault(r["team_abbrev"], {})[str(r["espn_athlete_id"])] = {
            "name": r["player_name"], "status": r.get("status"),
            "injury": r.get("injury_type"), "player_id": r.get("player_id")}
    prev = load_state("espn")
    events = diff_injuries(prev, cur, has_baseline=bool(prev))
    # every team that had injuries keeps a row, emptied when its list clears
    states = {**{t: {} for t in prev}, **cur}
    return save("espn", states, events, games_by_team, now)


# ── Re-scoring after starter news ──────────────────────────────────

# A free-feed price counts as "at the paid snapshot" when it was taken from
# this long before it until PAIRED_AFTER after it (the odds and close
# chains take one within a minute or two of every paid snapshot)
PAIRED_BEFORE = timedelta(minutes=30)
PAIRED_AFTER = timedelta(minutes=10)


def market_check(game_id: int, since: Optional[datetime] = None
                 ) -> Tuple[Optional[bool], str]:
    """Compare the NHL feed's prices now with its prices at the last paid
    odds snapshot of this game (raw.odds_snapshots, moneyline).
    since: when this news run started (naive UTC, like captured_at). Only
    a feed price taken since then counts as "now": an older one, from an
    earlier run, may predate the news, so it can't show the market has not
    reacted. "Before" is a feed price from PAIRED_BEFORE before the paid
    snapshot to PAIRED_AFTER after it. Without since, "now" is any feed
    price after that window."""
    with engine.connect() as conn:
        last_paid = conn.execute(text("""
            SELECT MAX(captured_at) FROM raw.odds_snapshots
            WHERE game_id = :g AND market_type = 'ml'
        """), {"g": game_id}).scalar()
        if last_paid is None:
            return None, ("no paid odds snapshot for this game yet, so there is no "
                          "stored price to bet at")
        lo, edge = last_paid - PAIRED_BEFORE, last_paid + PAIRED_AFTER
        rows = conn.execute(text("""
            SELECT source, book, captured_at, home_price, away_price
            FROM raw.nhl_feed_snapshots
            WHERE game_id = :g AND market = 'ml'
              AND home_price IS NOT NULL AND away_price IS NOT NULL
              AND captured_at >= :lo
            ORDER BY captured_at
        """), {"g": game_id, "lo": lo}).fetchall()
    before, now = {}, {}
    for source, book, captured_at, home, away in rows:
        key, prices = f"{book} ({source})", (int(home), int(away))
        if captured_at <= edge:
            before[key] = prices              # the latest in the window wins
        if captured_at > edge if since is None else captured_at >= since:
            now[key] = prices
    if not now:
        return None, ("no free NHL-feed price from this run, so a move can't be "
                      "ruled out" if since is not None else
                      "no free NHL-feed price since the last paid odds snapshot, "
                      "so a move can't be ruled out")
    if not before:
        return None, ("no free NHL-feed price taken with the last paid odds "
                      "snapshot, so a move can't be ruled out")
    return price_move(before, now)


def rescore_plan(news_games: Iterable[int], open_games: Dict[int, dict],
                 frozen: set, moves: Dict[int, tuple]) -> Dict[date_cls, tuple]:
    """Pure: {date: (games to re-score, games that may get a new pick)}.
    A game with a moneyline pick (frozen) is not re-scored; of the rest,
    only those whose price did not move (moves[g][0] is False) may get a
    pick. Games that have started are left out."""
    plan: Dict[date_cls, tuple] = {}
    for gid in sorted(set(news_games)):
        if gid not in open_games:
            continue
        todo, eligible = plan.setdefault(open_games[gid]["date"], ([], set()))
        if gid in frozen:
            continue
        todo.append(gid)
        if (moves.get(gid) or (None,))[0] is False:
            eligible.add(gid)
    return plan


def rescore(events: List[dict], games: List[dict], now: datetime) -> None:
    """For starter news on games still to play: record whether the market
    moved, then re-score the date through the recommend path (games with
    a moneyline pick keep it; only unmoved games may get a new pick)."""
    from betting.recommend import (frozen_games, generate_recommendations,
                                   load_issued_picks)
    open_games = {g["game_id"]: g for g in games if not g["started"]}
    news_games = sorted({e["game_id"] for e in events
                         if e["kind"] in STARTER_KINDS and e.get("game_id") in open_games})
    if not news_games:
        return
    try:
        from ingestion.nhl_odds import snapshot as nhl_snapshot
        nhl_snapshot(skip_when_idle=True)       # free: api-web.nhle.com
    except Exception as e:
        logger.error(f"NHL feed snapshot for the market check failed (non-fatal): {e}")
    # only feed prices taken by this run count as "now" (naive UTC, like
    # raw.nhl_feed_snapshots.captured_at)
    since = now.astimezone(timezone.utc).replace(tzinfo=None)
    moves = {}
    for gid in news_games:
        try:
            moves[gid] = market_check(gid, since=since)
        except Exception as e:
            moves[gid] = (None, f"market check failed: {e}")
        logger.info(f"Market check, game {gid}: {moves[gid][1]}")

    try:
        from config.runs import finished
        daily_done = finished("daily", local_today())
    except Exception as e:
        logger.error(f"Could not read whether today's daily run finished "
                     f"(non-fatal; no pick from news this run): {e}")
        daily_done = False

    dates = {open_games[g]["date"] for g in news_games}
    frozen = set()
    for d in dates:
        frozen |= frozen_games(load_issued_picks(d))
    rescored, failed, new_picks, waiting = set(), set(), set(), set()
    for d, (todo, eligible) in rescore_plan(news_games, open_games, frozen,
                                            moves).items():
        if not todo:
            logger.info(f"News on {d}: every game with starter news already has "
                        f"its pick (kept as issued)")
            continue
        if not daily_done:
            waiting |= set(todo)
            logger.info(f"News on {d}: waiting for today's daily run (its box "
                        f"scores, Elo and rolling stats are not loaded yet), so "
                        f"the news is recorded and no game is re-scored")
            continue
        try:
            recs = generate_recommendations(d, only_games=eligible)
            rescored |= set(todo)
            if recs is not None and not recs.empty:
                new_picks |= set(int(g) for g in recs["game_id"])
        except Exception as e:
            failed |= set(todo)
            logger.error(f"Re-scoring {d} after starter news failed (non-fatal): {e}")

    with engine.begin() as conn:
        for e in events:
            gid = e.get("game_id")
            if e["kind"] not in STARTER_KINDS or gid not in moves:
                continue
            moved, note = moves[gid]
            if gid in rescored and moved is not False:
                note += ". No new pick from this news: the stored price may be gone"
            if gid in failed:
                note += ". Re-scoring failed (see the log)"
            if gid in waiting:
                note += (". Waiting for today's daily run: no pick from news "
                         "before it has loaded last night's games")
            conn.execute(text("""
                UPDATE raw.news_events SET rescored = :r, new_pick = :p,
                       market_moved = :m, market_note = :n
                WHERE event_id = :id
            """), {"r": None if gid in failed else gid in rescored,
                   "p": gid in new_picks, "m": moved,
                   "n": note, "id": e["event_id"]})


# ── The run ────────────────────────────────────────────────────────

def run_news(due: bool = False) -> int:
    """One news check. Returns the number of events written."""
    from config.migrate import ensure_schema
    ensure_schema()
    ensure_table()
    games = todays_games()
    if due:
        is_due, why = news_due(local_now(), [g["start_time_utc"] for g in games],
                               last_run_start())
        if not is_due:
            logger.info(f"news --due: nothing to do, {why}")
            return 0
        logger.info(f"news --due: checking, {why}")
    if not games:
        logger.info("News: no game today; nothing checked")
        return 0
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        run_id = conn.execute(text("""
            INSERT INTO raw.news_runs (started_at) VALUES (:now) RETURNING run_id
        """), {"now": now}).scalar()
    games_by_team = {}
    for g in games:
        for team in (g["home_team"], g["away_team"]):
            # a team plays once a day; the earliest unstarted game wins
            if team not in games_by_team or games_by_team[team]["started"]:
                games_by_team[team] = g

    events, notes = [], []
    for label, step in (("starters", lambda: check_starters(games, games_by_team, now)),
                        ("lineups", lambda: check_lineups(games, games_by_team, now)),
                        ("injuries", lambda: check_injuries(games_by_team, now))):
        try:
            found = step()
            events += found
            notes.append(f"{label}: {len(found)}")
        except Exception as e:
            logger.error(f"News {label} check failed (non-fatal): {e}")
            notes.append(f"{label}: failed")
    try:
        rescore(events, games, now)
    except Exception as e:
        logger.error(f"Re-scoring after news failed (non-fatal): {e}")
        notes.append("rescore: failed")
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE raw.news_runs SET finished_at = :f, events = :n, notes = :notes
            WHERE run_id = :id
        """), {"f": datetime.now(timezone.utc), "n": len(events),
               "notes": "; ".join(notes), "id": run_id})
    logger.info(f"News: {len(events)} event(s) ({'; '.join(notes)})")
    return len(events)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m betting.news",
        description="Check for team news (starting goalies, lines, injuries) and "
                    "write what changed to raw.news_events. No Odds API request.")
    parser.add_argument("--due", action="store_true",
                        help="only on a game day between NEWS_START_HOUR (8:00) and "
                             "the last puck drop, at most every NEWS_MIN_GAP_MINUTES")
    args = parser.parse_args(argv)
    run_news(due=args.due)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
