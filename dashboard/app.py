"""
dashboard/app.py — Streamlit control room (Phase 3).

Run:  .venv/bin/streamlit run dashboard/app.py

Four tabs:
- Today: pending recommendations + upcoming slate (live during the season)
- Model: registry, walk-forward metrics, calibration plots
- Backtest: strategy simulation on true-price (DraftKings-era) games
- Bankroll: placed bets, PnL curve, CLV once live betting starts
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from config.settings import engine

st.set_page_config(page_title="NHL Betting System", page_icon="🏒",
                   layout="wide")
st.title("🏒 NHL Betting System")

ARTIFACTS = Path(__file__).parent.parent / "models" / "artifacts"


@st.cache_data(ttl=300)
def q(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


tab_today, tab_model, tab_backtest, tab_bankroll = st.tabs(
    ["📅 Today", "🧠 Model", "🧪 Backtest", "💰 Bankroll"])

with tab_today:
    st.subheader("Pending recommendations")
    recs = q("""
        SELECT r.created_at::date AS date, g.away_team || ' @ ' || g.home_team AS game,
               r.side, r.best_price AS price, r.model_prob, r.implied_prob_novig,
               r.edge_pct, r.recommended_stake, r.status
        FROM betting.recommendations r JOIN raw.games g USING (game_id)
        WHERE r.status = 'PENDING' ORDER BY r.created_at DESC LIMIT 50""")
    if recs.empty:
        st.info("No pending recommendations — either the slate is empty "
                "(off-season) or no game cleared the edge threshold.")
    else:
        st.dataframe(recs, use_container_width=True)

    st.subheader("Upcoming games")
    slate = q("""
        SELECT date, away_team, home_team FROM raw.games
        WHERE game_state = 'SCHEDULED' AND date <= CURRENT_DATE + 2
        ORDER BY date LIMIT 30""")
    st.dataframe(slate, use_container_width=True) if not slate.empty else \
        st.caption("No games in the next 48h.")

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
               "See docs/phase3_results.md for the honest read.")
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
        SELECT b.placed_at::date AS date, g.away_team || ' @ ' || g.home_team AS game,
               r.side, b.placed_price, b.stake_amount, b.result, b.pnl, b.clv
        FROM betting.placed_bets b
        JOIN betting.recommendations r USING (rec_id)
        JOIN raw.games g ON g.game_id = r.game_id
        ORDER BY b.placed_at DESC LIMIT 100""")
    st.dataframe(bets, use_container_width=True) if not bets.empty else \
        st.caption("No placed bets yet.")
