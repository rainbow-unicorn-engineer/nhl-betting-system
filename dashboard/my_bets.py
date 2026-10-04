"""
dashboard/my_bets.py — the "📒 My bets" tab of dashboard/app.py.

Where the bettors record the bets they really placed (a single bet or a
parlay), see each parlay as its own group with its legs in one table, and
track deposits, withdrawals, bonuses and the balance on every platform.
The money logic lives in betting/ledger.py; this file is only the screen.

The pure helpers (labels, editor rows -> ledger legs, local time -> UTC)
are tested in tests/test_my_bets.py without Streamlit or a database.
"""
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from betting import ledger
from betting.ledger import LegInput
from config.settings import LOCAL_TZ, engine, local_now, local_today, to_local

# Bet-type choices in the legs table -> (ledger market, side). The first
# four match the Check-a-bet tab's choices, so its rows copy over as-is.
BET_TYPES = {
    "Home win": ("ml", "HOME"),
    "Away win": ("ml", "AWAY"),
    "Over": ("total", "OVER"),
    "Under": ("total", "UNDER"),
    "Home puck line": ("pl", "HOME"),
    "Away puck line": ("pl", "AWAY"),
    "Player shots over": ("prop_sog", "OVER"),
    "Player shots under": ("prop_sog", "UNDER"),
    "Other (describe it)": ("other", None),
}
LEG_COLUMNS = ["Game", "Bet", "Line", "Odds", "Player", "Description", "Pick #"]
NEW_PLATFORM = "➕ New platform…"

STATUS_WORDS = {
    "OPEN": "Open",
    "WON": "Won",
    "LOST": "Lost",
    "PUSH": "Push (stake back)",
    "VOID": "Void (stake back)",
    "CASHED_OUT": "Cashed out",
}
RESULT_WORDS = {"WIN": "Won", "LOSS": "Lost", "PUSH": "Push", "VOID": "Void"}
TXN_WORDS = {
    "DEPOSIT": "Deposit (money in)",
    "WITHDRAWAL": "Withdrawal (money out)",
    "BONUS": "Bonus (cash credited by a promo)",
    "ADJUSTMENT": "Adjustment (+ or −, to match the platform)",
}


# ── Pure helpers ───────────────────────────────────────────────────

def game_label(d, away: str, home: str) -> str:
    """The game's label in the legs table. Same format as the Check-a-bet
    tab, so its rows copy over unchanged."""
    return f"{pd.Timestamp(d):%a %b %d}  {away} @ {home}"


def player_labels(players: pd.DataFrame) -> Dict[str, int]:
    """'Full Name (TEAM)' -> player_id; a repeated label gets the id added."""
    labels: Dict[str, int] = {}
    counts = players["full_name"].astype(str).str.cat(players["team"], sep="|").value_counts()
    for p in players.itertuples():
        label = f"{p.full_name} ({p.team})"
        if counts.get(f"{p.full_name}|{p.team}", 0) > 1 or label in labels:
            label = f"{p.full_name} ({p.team}) #{p.player_id}"
        labels[label] = int(p.player_id)
    return labels


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or \
        (isinstance(v, str) and not v.strip()) or v is pd.NA


def empty_legs(n: int = 1) -> pd.DataFrame:
    """A fresh legs table with n blank rows (bet type Home win)."""
    return pd.DataFrame({
        "Game": pd.Series([None] * n, dtype="object"),
        "Bet": ["Home win"] * n,
        "Line": pd.Series([None] * n, dtype="float"),
        "Odds": pd.Series([None] * n, dtype="float"),
        "Player": pd.Series([None] * n, dtype="object"),
        "Description": pd.Series([None] * n, dtype="object"),
        "Pick #": pd.Series([None] * n, dtype="float"),
    })[LEG_COLUMNS]


def editor_legs(rows: pd.DataFrame, games: Dict[str, int],
                players: Dict[str, int]) -> tuple:
    """Legs-table rows -> (ledger LegInputs, problems in plain English).

    games: game label -> game_id; players: player label -> player_id.
    Fully blank rows are skipped. The ledger's own checks
    (betting.ledger.validate_slip) run afterwards on the result."""
    legs: List[LegInput] = []
    problems: List[str] = []
    for i, r in enumerate(rows.reindex(columns=LEG_COLUMNS).itertuples(index=False), start=1):
        game, bet, line, odds, player, desc, pick = r
        if all(_blank(v) for v in (game, line, odds, player, desc, pick)):
            continue                       # an untouched row
        if _blank(bet) or bet not in BET_TYPES:
            problems.append(f"Row {i}: pick the kind of bet.")
            continue
        market, side = BET_TYPES[bet]
        game_id = games.get(game) if not _blank(game) else None
        if not _blank(game) and game_id is None:
            problems.append(f"Row {i}: that game is no longer in the list; pick it again.")
            continue
        if market == "other":
            if _blank(desc):
                problems.append(f"Row {i}: describe the bet in the Description column.")
                continue
            side = str(desc).strip()
        elif game_id is None:
            problems.append(f"Row {i}: pick the game.")
            continue
        player_id = None
        if market == "prop_sog":
            if _blank(player) or player not in players:
                problems.append(f"Row {i}: pick the player.")
                continue
            player_id = players[player]
        legs.append(LegInput(
            market=market, side=side, game_id=game_id,
            line=None if _blank(line) else float(line),
            price_american=None if _blank(odds) else int(odds),
            player_id=player_id,
            rec_id=None if _blank(pick) else int(pick)))
    if not legs and not problems:
        problems.append("Add at least one leg: pick a game, a bet and the odds.")
    return legs, problems


def checker_rows(slip: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The Check-a-bet tab's rows as legs-table rows."""
    out = empty_legs(0)
    if slip is None or slip.empty:
        return out
    rows = slip.reindex(columns=["Game", "Bet", "Line", "Odds"])
    rows = rows[rows["Bet"].isin(BET_TYPES)]
    return pd.concat([out, rows], ignore_index=True)[LEG_COLUMNS]


def local_to_utc(d: date, t: time) -> datetime:
    """A local date and time (LOCAL_TIMEZONE, else this machine's zone) as
    naive UTC, the ledger's convention."""
    naive = datetime.combine(d, t)
    aware = naive.replace(tzinfo=LOCAL_TZ) if LOCAL_TZ is not None else naive.astimezone()
    return aware.astimezone(timezone.utc).replace(tzinfo=None)


def utc_to_local_text(ts) -> str:
    """Naive UTC -> 'Sat Oct 10 07:05 PM' in the local zone."""
    if ts is None or pd.isna(ts):
        return ""
    aware = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
    return to_local(aware).strftime("%a %b %d %I:%M %p")


def money(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def signed_money(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"+${x:,.2f}" if x > 0 else money(x)


# ── Database reads for the form ────────────────────────────────────

def _read(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def recent_games() -> pd.DataFrame:
    """Games from 3 days ago to a week ahead (a bet may be recorded after
    puck drop)."""
    return _read("""
        SELECT game_id, date, away_team, home_team FROM raw.games
        WHERE date BETWEEN :today - 3 AND :today + 7
        ORDER BY date, start_time_utc, game_id""", {"today": local_today()})


def recent_players(teams: List[str]) -> pd.DataFrame:
    """Skaters who played for these teams in the last ~13 months, each
    with the team of his latest game."""
    if not teams:
        return pd.DataFrame(columns=["player_id", "full_name", "team"])
    return _read("""
        SELECT * FROM (
            SELECT DISTINCT ON (s.player_id) s.player_id, p.full_name, s.team
            FROM raw.skater_games s
            JOIN raw.games g USING (game_id)
            JOIN raw.players p USING (player_id)
            WHERE g.date >= :since
            ORDER BY s.player_id, g.date DESC
        ) latest WHERE team = ANY(:teams)
        ORDER BY team, full_name""",
                 {"since": local_today() - timedelta(days=400), "teams": list(teams)})


def pending_picks() -> pd.DataFrame:
    return _read("""
        SELECT r.rec_id, g.date, g.away_team, g.home_team, r.side,
               r.best_price, r.best_book
        FROM betting.recommendations r JOIN raw.games g USING (game_id)
        WHERE r.status = 'PENDING' AND r.market_type = 'ml'
        ORDER BY r.created_at DESC LIMIT 50""")


# ── The tab ────────────────────────────────────────────────────────

def render(st) -> None:
    """Draw the tab. st is the streamlit module (passed in so the pure
    helpers above import without Streamlit)."""
    ledger.ensure_schema()
    st.subheader("Record a bet")
    st.caption(
        "One row is a single bet. Several rows make a parlay → one ticket "
        "that pays only if every leg wins. Odds are American → -150 means "
        "bet $150 to win $100, +130 means bet $100 to win $130. A puck "
        "line → a win bet with a goal handicap (e.g. -1.5 = must win by 2+); "
        "an over/under → total goals above or below the line.")
    _record_form(st)

    st.divider()
    st.subheader("My bets")
    _bet_list(st)

    st.divider()
    st.subheader("Balances")
    _balances(st)


def _platform_picker(st, key: str) -> str:
    options = ledger.known_platforms() + [NEW_PLATFORM]
    choice = st.selectbox("Platform", options, key=f"{key}_choice",
                          help="The sportsbook or exchange. Set PLATFORMS in "
                               ".env to list yours here.")
    if choice == NEW_PLATFORM:
        return st.text_input("New platform name", key=f"{key}_new", max_chars=40)
    return choice


def _record_form(st) -> None:
    ss = st.session_state
    ss.setdefault("ledger_ver", 0)
    if "ledger_rows" not in ss:
        ss["ledger_rows"] = empty_legs()

    games_df = recent_games()
    games = {game_label(g.date, g.away_team, g.home_team): int(g.game_id)
             for g in games_df.itertuples()}
    teams = sorted(set(games_df["away_team"]) | set(games_df["home_team"]))
    players = player_labels(recent_players(teams))
    if not games:
        st.info("No games in the last 3 days or the next week: only 'Other' "
                "bets can be recorded right now.")

    edited = st.data_editor(
        ss["ledger_rows"],
        column_config={
            "Game": st.column_config.SelectboxColumn(options=list(games), width="large"),
            "Bet": st.column_config.SelectboxColumn(options=list(BET_TYPES), required=True),
            "Line": st.column_config.NumberColumn(
                help="Over/under or shots: the line, e.g. 6.5. Puck line: the "
                     "team's handicap, e.g. -1.5 or +1.5", step=0.5),
            "Odds": st.column_config.NumberColumn(
                help="This leg's American odds, e.g. -130 or +120. For a "
                     "parlay you may leave them out and enter the combined "
                     "odds below", step=1),
            "Player": st.column_config.SelectboxColumn(
                options=list(players), width="medium",
                help="Shots-on-goal bets only"),
            "Description": st.column_config.TextColumn(
                help="'Other' bets only, e.g. 'TOR to win the Cup'"),
            "Pick #": st.column_config.NumberColumn(
                disabled=True, help="The system pick this leg came from, if any"),
        },
        num_rows="dynamic", hide_index=True, width="stretch",
        key=f"ledger_legs_{ss['ledger_ver']}")

    def reset_rows(rows: pd.DataFrame) -> None:
        ss["ledger_rows"] = rows.reset_index(drop=True)
        ss["ledger_ver"] += 1
        st.rerun()

    def kept(rows: pd.DataFrame) -> pd.DataFrame:
        """The rows with anything filled in (an untouched row is dropped)."""
        if rows.empty:
            return rows
        filled = [not all(_blank(r[c]) for c in LEG_COLUMNS if c != "Bet")
                  for _, r in rows.reindex(columns=LEG_COLUMNS).iterrows()]
        return rows[filled]

    c1, c2, c3 = st.columns([1, 2, 1])
    checker = checker_rows(ss.get("checker_slip"))
    if c1.button("Copy legs from Check a bet", disabled=checker.empty,
                 help="Brings over the rows you entered on the Check a bet tab"):
        reset_rows(pd.concat([kept(edited), checker], ignore_index=True))
    picks = pending_picks()
    pick_labels = {
        f"Pick #{p.rec_id}: {p.away_team} @ {p.home_team}, "
        f"{p.home_team if p.side == 'HOME' else p.away_team} win "
        f"{int(p.best_price):+d} ({p.best_book})": p for p in picks.itertuples()}
    chosen = c2.selectbox("Pending system pick", list(pick_labels), index=None,
                          placeholder="Add a leg from a pending pick…",
                          label_visibility="collapsed")
    if c3.button("Add pick", disabled=chosen is None):
        p = pick_labels[chosen]
        label = game_label(p.date, p.away_team, p.home_team)
        if label not in games:
            st.error("That pick's game is not in the list of recent games.")
        else:
            row = empty_legs()
            row.loc[0, ["Game", "Bet", "Odds", "Pick #"]] = [
                label, "Home win" if p.side == "HOME" else "Away win",
                float(p.best_price), float(p.rec_id)]
            reset_rows(pd.concat([kept(edited), row], ignore_index=True))

    c = st.columns(3)
    bettor = c[0].selectbox("Bettor", ledger.bettors(),
                            help="Set the labels with BETTORS in .env")
    with c[1]:
        platform = _platform_picker(st, "slip_platform")
    stake = c[2].number_input("Stake in $", min_value=0.0, step=5.0, value=None,
                              help="The money put on the bet. For a bonus bet, "
                                   "the bonus credit used")
    c = st.columns(3)
    combined = c[0].number_input(
        "Odds for the whole ticket (optional)", value=None, step=1,
        help="A single bet: its odds, if not in the table. A parlay: the "
             "combined odds the platform shows (needed when it boosts them or "
             "a leg has no odds). Left empty, a parlay's legs are multiplied")
    # Defaults set once: a widget whose default changes every minute
    # would lose what was typed into it
    now = local_now()
    ss.setdefault("slip_day", now.date())
    ss.setdefault("slip_time", now.time().replace(second=0, microsecond=0))
    placed_day = c[1].date_input("Placed on", key="slip_day")
    placed_time = c[2].time_input("at (local time)", key="slip_time")
    c = st.columns([1, 1, 2])
    bonus = c[0].checkbox("Bonus bet", help="Staked with promo credit, not cash: "
                          "a win pays the profit only, a loss costs no cash")
    paper = c[1].checkbox("Practice bet", help="Not real money: shown in the list, "
                          "kept out of the balances")
    notes = c[2].text_input("Notes (optional)")

    if st.button("Save bet", type="primary"):
        legs, problems = editor_legs(edited, games, players)
        if not problems:
            _, problems = ledger.validate_slip(
                bettor, platform or "", stake, legs,
                None if combined is None else int(combined))
        for p in problems:
            st.error(p)
        if not problems:
            try:
                slip_id = ledger.record_slip(
                    bettor, platform, stake, legs,
                    None if combined is None else int(combined),
                    placed_at=local_to_utc(placed_day, placed_time),
                    notes=notes, is_paper=paper, is_bonus_bet=bonus)
            except Exception as e:          # the database said no
                st.error(f"Not saved: {e}")
            else:
                ss["ledger_saved"] = (f"Saved bet #{slip_id} "
                                      f"({'parlay, ' + str(len(legs)) + ' legs' if len(legs) > 1 else 'single bet'}).")
                reset_rows(empty_legs())
    if ss.get("ledger_saved"):
        st.success(ss.pop("ledger_saved"))


def _bet_list(st) -> None:
    slips = ledger.load_slips()
    if slips.empty:
        st.info("No bets recorded yet. Saved bets show here; the daily run "
                "settles them from the final scores and box scores.")
        return

    c = st.columns([2, 2, 2, 1])
    who = c[0].multiselect("Bettor", sorted(slips["bettor"].unique()))
    where = c[1].multiselect("Platform", sorted(slips["platform"].unique()))
    status = c[2].multiselect("Status", list(STATUS_WORDS),
                              format_func=STATUS_WORDS.get)
    show_paper = c[3].checkbox("Practice bets", value=True)
    view = slips
    if who:
        view = view[view["bettor"].isin(who)]
    if where:
        view = view[view["platform"].isin(where)]
    if status:
        view = view[view["status"].isin(status)]
    if not show_paper:
        view = view[~view["is_paper"]]

    real = view[~view["is_paper"]]
    settled_cash = real[(real["status"].isin(["WON", "LOST", "CASHED_OUT"]))
                        & ~real["is_bonus_bet"]]
    m = st.columns(4)
    m[0].metric("Open bets", int((view["status"] == "OPEN").sum()))
    m[1].metric("Riding on open bets",
                money(real.loc[(real["status"] == "OPEN") & ~real["is_bonus_bet"], "stake"].sum()),
                help="Cash stakes of real bets not decided yet")
    m[2].metric("Profit / loss", signed_money(real["pnl"].sum(min_count=1)),
                help="P/L → payout minus stake, over settled real bets")
    staked = settled_cash["stake"].sum()
    m[3].metric("ROI", f"{settled_cash['pnl'].sum() / staked:+.1%}" if staked else "—",
                help="Return on investment → profit / cash staked, over settled "
                     "cash bets. Needs hundreds of bets before it means much")
    st.caption("Push → the result landed exactly on the line, so the stake "
               "came back. Void → the bet was cancelled (postponed game, player "
               "did not play), stake back. In a parlay a pushed or void leg "
               "simply drops out and the rest still pay.")

    legs = ledger.load_legs(view["slip_id"].tolist())
    singles = view[~view["is_parlay"]]
    if not singles.empty:
        one = legs[legs["slip_id"].isin(singles["slip_id"])].drop_duplicates("slip_id")
        t = singles.merge(one[["slip_id", "bet"]], on="slip_id", how="left")
        st.markdown("**Single bets**")
        st.dataframe(pd.DataFrame({
            "#": t["slip_id"],
            "Placed": t["placed_at"].map(utc_to_local_text),
            "Bettor": t["bettor"], "Platform": t["platform"], "Bet": t["bet"],
            "Odds": t["price_american"].map(lambda p: f"{int(p):+d}"),
            "Stake": t["stake"].map(money),
            "Status": t["status"].map(STATUS_WORDS.get),
            "Payout": t["payout"].map(money),
            "Profit/loss": t["pnl"].map(signed_money),
            "Notes": [" · ".join(x for x in (
                "bonus bet" if b else "", "practice" if pp else "",
                n if isinstance(n, str) else "") if x)
                for b, pp, n in zip(t["is_bonus_bet"], t["is_paper"], t["notes"])],
        }), hide_index=True, width="stretch")

    parlays = view[view["is_parlay"]]
    if not parlays.empty:
        st.markdown("**Parlays** (each one is its own group: open it to see its legs)")
    for s in parlays.itertuples():
        mine = legs[legs["slip_id"] == s.slip_id]
        decided = mine["result"].notna().sum()
        label = (f"#{s.slip_id} · {s.bettor} · {s.platform} · {len(mine)}-leg parlay · "
                 f"{money(s.stake)} at {int(s.price_american):+d} · "
                 f"{STATUS_WORDS[s.status]}"
                 + (f" · {signed_money(s.pnl)}" if s.status != "OPEN" else
                    f" · {decided}/{len(mine)} legs decided")
                 + (" · practice" if s.is_paper else "")
                 + (" · bonus bet" if s.is_bonus_bet else ""))
        with st.expander(label, expanded=s.status == "OPEN"):
            st.dataframe(pd.DataFrame({
                "Leg": mine["leg_no"],
                "Bet": mine["bet"],
                "Odds": mine["price_american"].map(
                    lambda p: "—" if pd.isna(p) else f"{int(p):+d}"),
                "Result": mine["result"].map(
                    lambda r: "Open" if pd.isna(r) else RESULT_WORDS[r]),
                "Final score": [
                    "" if pd.isna(h) else f"{a_t} {int(a)} – {h_t} {int(h)}"
                    for a_t, a, h_t, h in zip(mine["away_team"], mine["away_score"],
                                              mine["home_team"], mine["home_score"])],
            }), hide_index=True, width="stretch")
            pushed = int(mine["result"].isin(["PUSH", "VOID"]).sum())
            st.caption(
                f"Combined: stake {money(s.stake)} at {int(s.price_american):+d} → "
                + (f"pays {money(s.stake * (ledger.decimal_odds(s.price_american) - (1 if s.is_bonus_bet else 0)))} if every leg wins"
                   if s.status == "OPEN" else
                   f"{STATUS_WORDS[s.status].lower()}, payout {money(s.payout)}, "
                   f"profit/loss {signed_money(s.pnl)}")
                + (f" ({pushed} leg(s) pushed or voided and dropped out)" if pushed else "")
                + (f". Notes: {s.notes}" if isinstance(s.notes, str) and s.notes else ""))

    c = st.columns([1, 3])
    if c[0].button("Settle now", help="Check every open bet against the final "
                   "scores and box scores (the daily run does this too)"):
        counts = ledger.settle_slips()
        settled = sum(v for k, v in counts.items() if k != "legs")
        st.session_state["ledger_settled"] = (f"{counts['legs']} leg(s) decided, "
                                              f"{settled} bet(s) settled.")
        st.rerun()
    if st.session_state.get("ledger_settled"):
        c[1].success(st.session_state.pop("ledger_settled"))
    _fix_by_hand(st, view, legs)


def _fix_by_hand(st, view: pd.DataFrame, legs: pd.DataFrame) -> None:
    with st.expander("Fix or settle a bet by hand"):
        st.caption("For 'Other' bets, a cash-out → taking the platform's offer to "
                   "close a bet early, a platform's own ruling, or a typo.")
        if view.empty:
            st.caption("No bets in the current filter.")
            return
        sid = st.selectbox("Bet #", view["slip_id"].tolist(), key="fix_slip")
        action = st.radio("What to do", ["Set a leg's result", "Cash out",
                                         "Set the result of the whole bet",
                                         "Delete it"], horizontal=True, key="fix_action")
        mine = legs[legs["slip_id"] == sid]
        try:
            if action == "Set a leg's result":
                leg_no = st.selectbox("Leg", mine["leg_no"].tolist(), key="fix_leg",
                                      format_func=lambda n: f"{n}: {mine.set_index('leg_no').loc[n, 'bet']}")
                res = st.selectbox("Result", ["WIN", "LOSS", "PUSH", "VOID", "(not decided)"],
                                   key="fix_res")
                if st.button("Save leg result"):
                    out = ledger.set_leg_result(sid, leg_no,
                                                None if res.startswith("(") else res)
                    if out.get("kept_by_hand"):
                        st.session_state["ledger_settled"] = (
                            f"Leg saved. Bet #{sid} keeps the result you set by hand; to "
                            f"let its legs decide again, set the whole bet back to Open.")
                    st.rerun()
            elif action == "Cash out":
                amount = st.number_input("Cash-out amount in $", min_value=0.0, step=1.0,
                                         value=None, key="fix_cash")
                if st.button("Save cash-out", disabled=amount is None):
                    ledger.settle_by_hand(sid, "CASHED_OUT", amount)
                    st.rerun()
            elif action == "Set the result of the whole bet":
                status = st.selectbox("Result", list(STATUS_WORDS), format_func=STATUS_WORDS.get,
                                      key="fix_status")
                payout = st.number_input("Payout in $ (stake included; needed for Won)",
                                         min_value=0.0, step=1.0, value=None, key="fix_payout")
                if st.button("Save result"):
                    ledger.settle_by_hand(sid, status, payout)
                    st.rerun()
            else:
                if st.button(f"Delete bet #{sid}", type="secondary"):
                    ledger.delete_slip(sid)
                    st.rerun()
        except ValueError as e:
            st.error(str(e))


def _balances(st) -> None:
    st.caption("Balance = deposits − withdrawals + bonuses + adjustments + "
               "profit/loss of settled bets − stakes still riding. It should "
               "match what the platform shows; if not, add an adjustment. "
               "Practice bets are left out.")
    with st.expander("Add a deposit, withdrawal or bonus"):
        c = st.columns(3)
        bettor = c[0].selectbox("Bettor", ledger.bettors(), key="txn_bettor")
        with c[1]:
            platform = _platform_picker(st, "txn_platform")
        kind = c[2].selectbox("Kind", list(TXN_WORDS), format_func=TXN_WORDS.get,
                              key="txn_kind")
        c = st.columns(3)
        amount = c[0].number_input(
            "Amount in $", value=None, step=10.0, key="txn_amount",
            min_value=None if kind == "ADJUSTMENT" else 0.0,
            help="Positive; an adjustment may be negative")
        day = c[1].date_input("Date", value=local_today(), key="txn_day")
        note = c[2].text_input("Note (optional)", key="txn_note")
        if st.button("Save", key="txn_save", disabled=amount is None):
            try:
                ledger.record_txn(bettor, platform or "", kind, amount, note,
                                  ts=local_to_utc(day, local_now().time()))
            except ValueError as e:
                st.error(str(e))
            else:
                st.rerun()

    table = ledger.balances()
    if table.empty:
        st.info("No deposits or bets yet.")
        return
    st.dataframe(pd.DataFrame({
        "Bettor": table["bettor"], "Platform": table["platform"],
        "Balance": table["balance"].map(money),
        "Deposited": table["deposits"].map(money),
        "Withdrawn": table["withdrawals"].map(money),
        "Bonuses": table["bonuses"].map(money),
        "Adjustments": table["adjustments"].map(signed_money),
        "Bet profit/loss": table["settled_pl"].map(signed_money),
        "Riding on open bets": table["open_stakes"].map(money),
        "Bets": table["bets"], "Won": table["won"], "Lost": table["lost"],
        "ROI": table["roi"].map(lambda r: "—" if r is None or pd.isna(r) else f"{r:+.1%}"),
    }), hide_index=True, width="stretch")
    per = table.groupby("bettor")[["balance", "settled_pl"]].sum()
    cols = st.columns(max(len(per), 1))
    for col, (b, r) in zip(cols, per.iterrows()):
        col.metric(f"{b}: all platforms", money(r["balance"]),
                   delta=f"{signed_money(r['settled_pl'])} profit/loss")

    pl = ledger.running_pl(ledger.load_slips())
    if not pl.empty:
        st.markdown("**Running profit/loss** (settled real bets, in the order they settled)")
        chart = pl.pivot_table(index="settled_at", columns="bettor", values="cum_pl",
                               aggfunc="last").ffill()
        st.line_chart(chart)

    txns = ledger.load_txns()
    if not txns.empty:
        with st.expander(f"Money in and out ({len(txns)})"):
            st.dataframe(pd.DataFrame({
                "#": txns["txn_id"], "Date": txns["ts"].map(utc_to_local_text),
                "Bettor": txns["bettor"], "Platform": txns["platform"],
                "Kind": txns["kind"].map(lambda k: TXN_WORDS[k].split(" (")[0]),
                "Amount": txns["amount"].map(money), "Note": txns["note"],
            }), hide_index=True, width="stretch")
            c = st.columns([1, 1, 2])
            tid = c[0].selectbox("Entry #", txns["txn_id"].tolist(), key="txn_del_id")
            if c[1].button("Delete entry"):
                ledger.delete_txn(tid)
                st.rerun()
