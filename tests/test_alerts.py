"""
Tests for betting/alerts.py — hand-computed arbs and middles on
synthetic snapshot frames (the scanners are pure).
"""
import numpy as np
import pandas as pd
import pytest

from betting.alerts import (arb_stakes, find_arbs, find_middles,
                            middle_breakeven, middle_ev)
from config.settings import check_db_connection

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def _row(game_id=1, book="a", market="ml", line=None, home=None, away=None,
         over=None, under=None):
    return {"game_id": game_id, "book_name": book, "market_type": market,
            "line": line, "home_price": home, "away_price": away,
            "over_price": over, "under_price": under,
            "captured_at": pd.Timestamp("2026-01-15 22:00"),
            "home_team": "BUF", "away_team": "MTL",
            "date": pd.Timestamp("2026-01-15")}


class TestArbs:
    def test_arb_stakes_hand_computed(self):
        # +105 both sides: dec 2.05, implied sum 2/2.05 = 0.97561
        sa, sb, profit = arb_stakes(2.05, 2.05)
        assert sa == pytest.approx(0.5) and sb == pytest.approx(0.5)
        assert profit == pytest.approx(2.05 / 2 - 1)   # 2.5%

    def test_cross_book_ml_arb_found(self):
        lines = pd.DataFrame([
            _row(book="dk", home=105, away=-120),
            _row(book="fd", home=-125, away=105),
        ])
        arbs = find_arbs(lines)
        assert len(arbs) == 1
        a = arbs.iloc[0]
        assert a["book_a"] == "dk" and a["price_a"] == 105
        assert a["book_b"] == "fd" and a["price_b"] == 105
        assert a["profit_pct"] == pytest.approx(2.05 / 2 - 1, abs=1e-5)

    def test_no_arb_within_normal_vig(self):
        lines = pd.DataFrame([
            _row(book="dk", home=-110, away=-110),
            _row(book="fd", home=-108, away=-112),
        ])
        assert find_arbs(lines).empty

    def test_total_lines_must_match(self):
        # over 5.5 +105 / under 6.5 +105 is NOT an arb (different lines)
        lines = pd.DataFrame([
            _row(book="dk", market="total", line=5.5, over=105, under=-130),
            _row(book="fd", market="total", line=6.5, over=-130, under=105),
        ])
        assert find_arbs(lines).empty

    def test_same_line_total_arb(self):
        lines = pd.DataFrame([
            _row(book="dk", market="total", line=6.0, over=104, under=-125),
            _row(book="fd", market="total", line=6.0, over=-125, under=104),
        ])
        arbs = find_arbs(lines)
        assert len(arbs) == 1
        assert arbs.iloc[0]["market"] == "total"


class TestMiddles:
    def test_middle_breakeven_hand_computed(self):
        # both -110: dec 1.909; middle_net = 1.818; worst = -0.0909
        be = middle_breakeven(1.909091, 1.909091)
        assert be == pytest.approx(0.0909091 / (1.8181818 + 0.0909091), abs=1e-5)

    def test_middle_ev_exact_under_known_pmf(self):
        # total is 5 w.p. 0.3, 6 w.p. 0.4, 7 w.p. 0.3; O5.5/U6.5 at +100
        tpmf = np.zeros(10)
        tpmf[5], tpmf[6], tpmf[7] = 0.3, 0.4, 0.3
        # middle hits on 6: net +2; else one side wins: net 0
        ev = middle_ev(tpmf, 5.5, 6.5, 2.0, 2.0)
        assert ev == pytest.approx(0.4 * 2.0 + 0.6 * 0.0)

    def test_middle_ev_push_boundary(self):
        # integer low line 6: total==6 pushes the over (stake back)
        tpmf = np.zeros(10)
        tpmf[6] = 1.0
        # over@6 pushes (1.0), under@7 wins (2.0) -> return 3.0, net +1
        assert middle_ev(tpmf, 6.0, 7.0, 2.0, 2.0) == pytest.approx(1.0)

    def test_find_middles_pairs_and_model_ev(self):
        lines = pd.DataFrame([
            _row(book="dk", market="total", line=5.5, over=100, under=-120),
            _row(book="fd", market="total", line=6.5, over=-120, under=100),
        ])
        tpmf = np.zeros(10)
        tpmf[5], tpmf[6], tpmf[7] = 0.3, 0.4, 0.3
        mids = find_middles(lines, {1: tpmf})
        assert len(mids) == 1
        m = mids.iloc[0]
        assert (m["over_line"], m["under_line"]) == (5.5, 6.5)
        assert m["model_ev"] == pytest.approx(0.8)
        assert m["max_loss"] == pytest.approx(0.0)   # both at +100

    def test_find_middles_without_pmf_filters_expensive(self):
        # half-goal window at heavy juice: filtered without a model EV
        lines = pd.DataFrame([
            _row(book="dk", market="total", line=6.0, over=-115, under=-105),
            _row(book="fd", market="total", line=6.5, over=-110, under=-110),
        ])
        assert find_middles(lines, {}).empty


@requires_db
class TestLoad:
    def test_no_snapshots_is_graceful(self):
        from betting.alerts import run_alerts
        out = run_alerts()
        assert set(out) == {"arbs", "middles"}
