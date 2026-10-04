"""
Tests for features/market_prices.py (no database).

The point-in-time tests delete and rewrite every quote taken at or after
a game's puck drop and require the game's market row to stay the same.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from features import market_prices as M

START = pd.Timestamp("2025-01-10 00:00:00")      # naive UTC puck drop (game 1)


def _rows(spec):
    """spec: (snapshot offset in minutes from START, game_id, book, home,
    away[, purpose]) -> long h2h rows."""
    out = []
    for s in spec:
        off, gid, book, h, a = s[:5]
        purpose = s[5] if len(s) > 5 else "close"
        ts = START + timedelta(minutes=off)
        out.append({"snapshot_ts": ts, "game_id": gid, "book": book,
                    "side": "home", "price": h, "purpose": purpose})
        out.append({"snapshot_ts": ts, "game_id": gid, "book": book,
                    "side": "away", "price": a, "purpose": purpose})
    return pd.DataFrame(out)


STARTS = pd.Series({1: START, 2: START + timedelta(hours=1)})
DATES = pd.Series({1: "2025-01-09", 2: "2025-01-09"})


def _close(rows, starts=STARTS):
    q = M.add_no_vig(M.two_sided(rows))
    return M.latest_quotes(q, starts, M.CLOSE_MAX_LEAD)


def test_no_vig_is_proportional():
    q = M.add_no_vig(pd.DataFrame({"home_price": [-110], "away_price": [-110]}))
    assert q["novig_home"].iloc[0] == pytest.approx(0.5)
    assert q["overround"].iloc[0] == pytest.approx(2 * 110 / 210 - 1)
    q = M.add_no_vig(pd.DataFrame({"home_price": [-150], "away_price": [130]}))
    ph, pa = 0.6, 100 / 230
    assert q["novig_home"].iloc[0] == pytest.approx(ph / (ph + pa))


def test_close_takes_each_books_last_quote_before_start():
    rows = _rows([(-40, 1, "pinnacle", -120, 110),
                  (-14, 1, "pinnacle", -130, 118),     # the close
                  (-14, 1, "draftkings", -135, 115),
                  (5, 1, "draftkings", -300, 250)])    # in play: never used
    c = _close(rows)
    pin = c[c["book"] == "pinnacle"].iloc[0]
    assert (pin["home_price"], pin["away_price"]) == (-130, 118)
    dk = c[c["book"] == "draftkings"].iloc[0]
    assert (dk["home_price"], dk["away_price"]) == (-135, 115)


def test_close_drops_quotes_older_than_the_max_lead():
    rows = _rows([(-14, 1, "pinnacle", -130, 118),
                  (-60 * 30, 1, "bovada", -200, 170)])   # a day and more old
    c = _close(rows)
    assert set(c["book"]) == {"pinnacle"}


def test_one_sided_quote_is_dropped():
    rows = _rows([(-14, 1, "pinnacle", -130, 118)])
    rows = pd.concat([rows, pd.DataFrame([{
        "snapshot_ts": START - timedelta(minutes=14), "game_id": 1,
        "book": "fanduel", "side": "home", "price": -140, "purpose": "close"}])])
    assert set(_close(rows)["book"]) == {"pinnacle"}


def test_summary_consensus_pinnacle_and_best_prices():
    rows = _rows([(-14, 1, "pinnacle", -130, 120),
                  (-14, 1, "draftkings", -140, 118),
                  (-14, 1, "fanduel", -125, 105),
                  (-14, 1, "bovada", -135, 125)])
    s = M.summarize_quotes(_close(rows)).iloc[0]
    q = M.add_no_vig(M.two_sided(rows))
    assert s["nv_consensus"] == pytest.approx(q["novig_home"].median())
    assert s["nv_pinnacle"] == pytest.approx(
        q.loc[q["book"] == "pinnacle", "novig_home"].iloc[0])
    assert s["n_books"] == 4
    assert (s["best_home_price"], s["best_home_book"]) == (-125, "fanduel")
    assert (s["best_away_price"], s["best_away_book"]) == (125, "bovada")
    # the US-only best skips bovada (offshore) and pinnacle
    assert (s["best_us_away_price"], s["best_us_away_book"]) == (118, "draftkings")
    assert (s["best_us_home_price"], s["best_us_home_book"]) == (-125, "fanduel")


def test_summary_without_pinnacle_is_nan():
    rows = _rows([(-14, 1, "draftkings", -140, 118)])
    s = M.summarize_quotes(_close(rows)).iloc[0]
    assert np.isnan(s["nv_pinnacle"])


def test_best_price_ties_go_to_first_book_alphabetically():
    rows = _rows([(-14, 1, "fanduel", -120, 100), (-14, 1, "betmgm", -120, 100)])
    s = M.summarize_quotes(_close(rows)).iloc[0]
    assert s["best_home_book"] == "betmgm"


def test_morning_quotes_need_the_morning_purpose_and_the_game_date():
    # 10:00 Central on 2025-01-09 = 16:00 UTC, 8 h before game 1
    rows = _rows([(-8 * 60, 1, "pinnacle", -120, 110, "morning"),
                  (-8 * 60, 2, "pinnacle", -150, 130, "morning"),
                  (-14, 1, "pinnacle", -130, 118, "close"),
                  # a morning snapshot from the day before is not this game's
                  (-32 * 60, 1, "draftkings", -110, -110, "morning")])
    q = M.add_no_vig(M.two_sided(rows))
    m = M.morning_quotes(q, STARTS, DATES)
    assert set(zip(m["game_id"], m["book"])) == {(1, "pinnacle"), (2, "pinnacle")}
    assert m.loc[m["game_id"] == 1, "home_price"].iloc[0] == -120


def test_point_in_time_rows_at_or_after_start_change_nothing():
    base = [(-40, 1, "pinnacle", -120, 110), (-14, 1, "pinnacle", -130, 118),
            (-14, 1, "draftkings", -135, 115), (-75, 2, "fanduel", 140, -160)]
    before = M.summarize_quotes(_close(_rows(base)))
    noisy = base + [(0, 1, "pinnacle", -500, 400),      # exactly at puck drop
                    (30, 1, "draftkings", 300, -400),   # in play
                    (61, 2, "fanduel", -900, 600),      # after game 2's start
                    (60, 2, "bovada", -900, 600)]       # at game 2's start
    after = M.summarize_quotes(_close(_rows(noisy)))
    pd.testing.assert_frame_equal(before, after)


def test_point_in_time_deleting_later_rows_changes_nothing():
    spec = [(-14, 1, "pinnacle", -130, 118), (10, 1, "pinnacle", -200, 170)]
    full = M.summarize_quotes(_close(_rows(spec)))
    cut = M.summarize_quotes(_close(_rows(spec[:1])))
    pd.testing.assert_frame_equal(full, cut)


def test_wide_book_prices_and_assemble():
    rows = _rows([(-14, 1, "pinnacle", -130, 118), (-14, 1, "draftkings", -135, 115),
                  (-30, 2, "draftkings", 120, -140),
                  (-8 * 60, 1, "pinnacle", -120, 110, "morning")])
    q = M.add_no_vig(M.two_sided(rows))
    close = M.latest_quotes(q[q["purpose"] == "close"], STARTS, M.CLOSE_MAX_LEAD)
    morn = M.morning_quotes(q, STARTS, DATES)
    meta = pd.DataFrame({"game_id": [1, 2, 3], "season": 20242025,
                         "date": "2025-01-09", "home_win": [True, False, True]})
    out = M.assemble_market(close, morn, meta, "odds_api",
                            books=("draftkings", "pinnacle"))
    assert list(out["game_id"]) == [1, 2]          # game 3 has no price
    g1 = out.set_index("game_id").loc[1]
    assert g1["draftkings_home"] == -135 and g1["pinnacle_away"] == 118
    assert g1["m_best_home_price"] == -120
    assert np.isnan(out.set_index("game_id").loc[2, "pinnacle_home"])
    assert np.isnan(out.set_index("game_id").loc[2, "m_nv_consensus"])
    assert (out["source"] == "odds_api").all()


def test_decimal_from_american():
    assert M.decimal_from_american(-150) == pytest.approx(1 + 100 / 150)
    assert M.decimal_from_american(130) == pytest.approx(2.3)
