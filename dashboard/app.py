"""
dashboard/app.py — Streamlit control room (Phase 3).

Run:  .venv/bin/streamlit run dashboard/app.py

Six tabs:
- Today: pending recommendations + upcoming slate (live during the season)
- Check a bet: a bet or parlay you enter, run through betting/checker.py
- My bets: the bettors' real bets (singles and parlays), their results,
  and the balance on every platform (dashboard/my_bets.py, betting/ledger.py)
- Model: registry, walk-forward metrics, calibration plots
- Backtest: strategy simulation on true-price (DraftKings-era) games
- Bankroll: the system's paper bets, PnL curve, CLV
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from betting.checker import EDGE_MIN_TOTAL, Leg, evaluate_parlay
from betting.engine import EDGE_MIN_ML
from config.migrate import ensure_schema
from config.settings import engine, local_today, to_local
from dashboard import my_bets
from features.util import american_implied_prob

st.set_page_config(page_title="NHL Betting System", page_icon="🏒",
                   layout="wide")
st.title("🏒 NHL Betting System")

ARTIFACTS = Path(__file__).parent.parent / "models" / "artifacts"

# Check-a-bet rows -> checker (market, side)
BET_TYPES = {"Home win": ("ml", "HOME"), "Away win": ("ml", "AWAY"),
             "Over": ("total", "OVER"), "Under": ("total", "UNDER")}

# Checker verdict -> (banner, what it means in plain English)
VERDICTS = {
    "BET": (st.success, "BET: the model's edge clears the minimum "
            f"({EDGE_MIN_ML:.1%} for win bets, {EDGE_MIN_TOTAL:.1%} for "
            "over/unders)."),
    "THIN": (st.warning, "THIN: expected to make a little money, but the "
             "edge is below the minimum, so the system would not bet it."),
    "PASS": (st.error, "PASS: expected to lose money at these odds."),
    "NO-MODEL": (st.info, "NO VERDICT: see the note below."),
    "WITHHELD": (st.info, "NO VERDICT: see the note below."),
}

try:
    ensure_schema()     # the Today tab reads raw.games.start_time_utc
except Exception as e:
    st.warning(f"Schema upgrade check failed: {e}")


@st.cache_data(ttl=300)
def q(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def local_start(ts) -> str:
    """Puck drop in the user's local zone (LOCAL_TIMEZONE)."""
    if ts is None or pd.isna(ts):
        return "TBD"
    return to_local(pd.Timestamp(ts).to_pydatetime()).strftime("%a %I:%M %p %Z")


def pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def slip_legs(slip: pd.DataFrame, games: dict) -> tuple:
    """Check-a-bet editor rows -> (checker Legs, row problems in plain
    English). games maps each dropdown label to its raw.games row."""
    legs, problems = [], []
    for i, row in enumerate(slip.itertuples(), start=1):
        if row.Game not in games or row.Bet not in BET_TYPES \
                or pd.isna(row.Odds):
            problems.append(f"Row {i}: pick a game, a bet, and the odds.")
            continue
        if abs(row.Odds) < 100:
            problems.append(f"Row {i}: American odds are -100 or lower, "
                            "or +100 or higher.")
            continue
        market, side = BET_TYPES[row.Bet]
        if market == "total" and pd.isna(row.Line):
            problems.append(f"Row {i}: an over/under needs a line, e.g. 6.5.")
            continue
        g = games[row.Game]
        legs.append(Leg(away=g.away_team, home=g.home_team, market=market,
                        side=side, price=int(row.Odds),
                        line=float(row.Line) if market == "total" else None,
                        date=pd.Timestamp(g.date).date()))
    return legs, problems


tab_today, tab_check, tab_mine, tab_model, tab_backtest, tab_bankroll = st.tabs(
    ["📅 Today", "🔍 Check a bet", "📒 My bets", "🧠 Model", "🧪 Backtest",
     "💰 Bankroll"])

with tab_today:
    st.subheader("Pending recommendations")
    recs = q("""
        SELECT g.date, g.start_time_utc, g.away_team || ' @ ' || g.home_team AS game,
               r.side, r.best_price AS price, r.model_prob, r.implied_prob_novig,
               r.edge_pct, r.recommended_stake, r.status
        FROM betting.recommendations r JOIN raw.games g USING (game_id)
        WHERE r.status = 'PENDING' ORDER BY r.created_at DESC LIMIT 50""")
    if recs.empty:
        st.info("No pending recommendations — either the slate is empty "
                "(off-season) or no game cleared the edge threshold.")
    else:
        recs.insert(1, "start (local)", recs.pop("start_time_utc").map(local_start))
        st.dataframe(recs, use_container_width=True)

    st.subheader("Upcoming games")
    # "Today" is the user's local date (the DB clock is UTC); the NHL API
    # marks upcoming games FUT/PRE, never SCHEDULED
    slate = q("""
        SELECT date, start_time_utc, away_team, home_team FROM raw.games
        WHERE game_state NOT IN ('FINAL', 'OFF')
          AND date BETWEEN :today AND :today + 2
        ORDER BY date, start_time_utc LIMIT 30""", {"today": local_today()})
    if not slate.empty:
        slate.insert(1, "start (local)", slate.pop("start_time_utc").map(local_start))
        st.dataframe(slate, use_container_width=True)
    else:
        st.caption("No games in the next 48h.")

with tab_check:
    st.subheader("Check a bet or parlay")
    st.caption("One row is a single bet; add rows for a parlay. The model's "
               "chances come from the latest `recommend` run, so a game shows "
               "NO VERDICT until that run has scored it.")
    upcoming = q("""
        SELECT date, away_team, home_team FROM raw.games
        WHERE game_state NOT IN ('FINAL', 'OFF') AND game_type IN (2, 3)
          AND date BETWEEN :today AND :today + 7
        ORDER BY date, start_time_utc""", {"today": local_today()})
    if upcoming.empty:
        st.info("No upcoming games in the next week.")
    else:
        games = {f"{pd.Timestamp(g.date):%a %b %d}  {g.away_team} @ {g.home_team}": g
                 for g in upcoming.itertuples()}
        slip = st.data_editor(
            pd.DataFrame({"Game": [next(iter(games))], "Bet": ["Home win"],
                          "Line": pd.Series([None], dtype="float"),
                          "Odds": [-110]}),
            column_config={
                "Game": st.column_config.SelectboxColumn(
                    options=list(games), required=True, width="large"),
                "Bet": st.column_config.SelectboxColumn(
                    options=list(BET_TYPES), required=True),
                "Line": st.column_config.NumberColumn(
                    help="Over/under only, e.g. 6.5", min_value=0.5, step=0.5),
                "Odds": st.column_config.NumberColumn(
                    help="American odds, e.g. -130 or +120", step=1,
                    required=True),
            },
            num_rows="dynamic", hide_index=True, key="slip")
        st.session_state["checker_slip"] = slip     # My bets can copy these rows
        c1, c2 = st.columns(2)
        boost = c1.number_input("Boosted parlay odds (optional)", value=None,
                                step=1, help="Only if the book offers a "
                                "special combined price, e.g. +450")
        bankroll = c2.number_input("Bankroll in $ (optional)", value=None,
                                   min_value=0.0, step=50.0)

        if st.button("Check", type="primary", key="check"):
            legs, problems = slip_legs(slip, games)
            if boost is not None and abs(boost) < 100:
                problems.append("Boosted odds are -100 or lower, or +100 or "
                                "higher.")
            for p in problems:
                st.error(p)

            if legs and not problems:
                r = evaluate_parlay(legs, int(boost) if boost else None)
                st.dataframe(pd.DataFrame([{
                    "Bet": f"{l.away} @ {l.home}: "
                           + {"HOME": f"{l.home} win", "AWAY": f"{l.away} win",
                              "OVER": f"Over {l.line}",
                              "UNDER": f"Under {l.line}"}[l.side],
                    "Odds": f"{l.price:+d}",
                    "Model chance": pct(l.p_win),
                    "Break-even chance": pct(american_implied_prob(l.price)),
                    "Edge": "—" if l.edge is None else f"{l.edge:+.1%}",
                    "Expected profit per $1": "—" if l.ev is None else f"{l.ev:+.3f}",
                    "Verdict": l.verdict,
                    "Notes": "; ".join(l.notes),
                } for l in r["legs"]]), use_container_width=True, hide_index=True)

                if "ev_per_unit" in r:
                    c = st.columns(4)
                    c[0].metric("Chance every leg wins", pct(r["p_win_all"]))
                    c[1].metric("Payout per $1 (decimal odds)",
                                f"{r['decimal_offered']:.2f}")
                    c[2].metric("Expected profit per $1",
                                f"{r['ev_per_unit']:+.3f}")
                    stake = r["stake_pct"]
                    c[3].metric("Suggested stake",
                                f"${stake * bankroll:,.2f}" if bankroll
                                else f"{stake:.2%} of bankroll",
                                help="A quarter of the Kelly-formula bet "
                                     "size, capped at 2% of bankroll")
                banner, meaning = VERDICTS[r["verdict"]]
                banner(meaning)
                for n in r["notes"]:
                    st.warning(n)

with tab_mine:
    try:
        my_bets.render(st)
    except Exception as e:
        st.error(f"The bet ledger could not load: {e}")

with tab_model:
    st.subheader("Model registry")
    st.dataframe(q("""
        SELECT model_name, version, model_type, trained_through,
               cv_log_loss, cv_brier, cv_auc, cv_accuracy, is_active
        FROM models.model_registry ORDER BY created_at DESC"""),
        use_container_width=True)

    cols = st.columns(2)
    for col, (title, png) in zip(cols, (
            ("lgbm_market (production)", "lgbm_calibration.png"),
            ("baseline_logreg (Phase 2)", "baseline_calibration.png"))):
        p = ARTIFACTS / png
        if p.exists():
            col.caption(title)
            col.image(str(p))

with tab_backtest:
    st.subheader("Strategy backtest — true-price era only")
    st.caption("Walk-forward OOF probabilities → edge threshold → "
               "quarter-Kelly, settled at actual DraftKings prices "
               "(near-closing, no line shopping — conservative). "
               "See docs/phase3_results.md for the full results.")
    if st.button("Run backtest (~1 min)"):
        with st.spinner("Running walk-forward + simulation..."):
            from betting.backtest import run_backtest
            r = run_backtest()
        c = st.columns(5)
        c[0].metric("Bets", r.n_bets)
        c[1].metric("Hit rate", f"{r.hit_rate:.1%}")
        c[2].metric("Kelly ROI", f"{r.roi:+.2%}")
        c[3].metric("Flat ROI", f"{r.flat_roi:+.2%}")
        c[4].metric("Max drawdown", f"{r.max_drawdown:.1%}")
        st.line_chart(r.bets.set_index("date")["bankroll"])
        st.dataframe(r.bets.sort_values("edge", ascending=False).head(25),
                     use_container_width=True)

with tab_bankroll:
    st.subheader("Bankroll log")
    log = q("SELECT * FROM betting.bankroll_log ORDER BY date")
    if log.empty:
        st.info("Empty until live/paper betting starts.")
    else:
        st.line_chart(log.set_index("date")["closing_balance"])
        st.dataframe(log.tail(30), use_container_width=True)

    st.subheader("Placed bets")
    bets = q("""
        SELECT g.date, g.away_team || ' @ ' || g.home_team AS game,
               r.side, b.placed_price, b.stake_amount, b.result, b.pnl, b.clv
        FROM betting.placed_bets b
        JOIN betting.recommendations r USING (rec_id)
        JOIN raw.games g ON g.game_id = r.game_id
        ORDER BY g.date DESC, b.placed_at DESC LIMIT 100""")
    if not bets.empty:
        st.dataframe(bets, use_container_width=True)
    else:
        st.caption("No placed bets yet.")
