"""
Tests for betting/montecarlo.py on synthetic bets (no database): the
staking caps, the edge estimate, the season simulation's direction and
reproducibility, the real-history replay, and the report.
"""
import numpy as np
import pandas as pd
import pytest

from betting.montecarlo import (STAKINGS, Scenario, Staking, day_stakes,
                                estimate_shrink, pack_days, replay, report,
                                run, season_days, simulate, summarize)

DEFAULTS, REQUESTED, FULL = STAKINGS


def _bets(n_days=60, per_day=3, edge=0.05, seed=0, true_shrink=1.0):
    """Synthetic backtest bets at even money (+100, decimal 2.0): the model
    says 0.5 + edge, the market 0.5. Outcomes drawn at the true chance."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for j in range(per_day):
            pm, pf = 0.5 + edge, 0.5
            rows.append({"date": pd.Timestamp("2025-11-01") + pd.Timedelta(days=d),
                         "game_id": d * 10 + j, "side": "HOME", "price": 100,
                         "decimal": 2.0, "pm": pm, "pf": pf, "edge": pm - pf,
                         # full Kelly at even money: 2p - 1
                         "kelly": 2 * pm - 1,
                         "won": rng.random() < pf + true_shrink * (pm - pf)})
    b = pd.DataFrame(rows)
    b.attrs["n_games"] = n_days * per_day * 2
    return b


class TestDayStakes:
    def test_per_bet_cap_trims(self):
        kelly = np.array([[0.20, 0.04]])
        mask = np.array([[True, True]])
        st = day_stakes(kelly, mask, DEFAULTS)       # quarter: 0.05, 0.01
        assert st[0] == pytest.approx([0.02, 0.01])

    def test_daily_cap_skips_and_a_smaller_bet_still_fits(self):
        rule = Staking("t", "t", 1.0, None, 0.10, None)
        kelly = np.array([[0.06, 0.06, 0.03]])        # strongest edge first
        st = day_stakes(kelly, np.ones((1, 3), bool), rule)
        assert st[0] == pytest.approx([0.06, 0.0, 0.03])

    def test_game_cap_skips(self):
        rule = Staking("t", "t", 1.0, None, None, 0.04)
        st = day_stakes(np.array([[0.05, 0.03]]), np.ones((1, 2), bool), rule)
        assert st[0] == pytest.approx([0.0, 0.03])

    def test_never_more_than_the_bankroll(self):
        st = day_stakes(np.array([[0.6, 0.6]]), np.ones((1, 2), bool), FULL)
        assert st[0].sum() == pytest.approx(1.0)
        assert st[0] == pytest.approx([0.5, 0.5])

    def test_padding_stakes_nothing(self):
        st = day_stakes(np.array([[0.1, 0.1]]), np.array([[True, False]]), FULL)
        assert st[0] == pytest.approx([0.1, 0.0])

    def test_requested_caps_do_not_bind_on_quarter_kelly(self):
        kelly = np.array([[0.19, 0.15, 0.10, 0.08]])
        mask = np.ones((1, 4), bool)
        st = day_stakes(kelly, mask, REQUESTED)
        assert st[0] == pytest.approx(kelly[0] * 0.25)


class TestShrink:
    def test_recovers_a_real_edge_and_no_edge(self):
        real = _bets(n_days=400, per_day=5, edge=0.08, seed=1, true_shrink=1.0)
        none = _bets(n_days=400, per_day=5, edge=0.08, seed=1, true_shrink=0.0)
        s1, se1 = estimate_shrink(real)
        s0, se0 = estimate_shrink(none)
        assert abs(s1 - 1.0) < 3 * se1 and abs(s0) < 3 * se0
        assert s1 > s0 and 0 < se1 < 0.5


class TestSimulate:
    def setup_method(self):
        self.packed = pack_days(_bets())

    def test_pack_sorts_and_pads(self):
        b = _bets(n_days=2, per_day=2)
        b.loc[1, "edge"] = 0.09
        p = pack_days(b.iloc[:3])
        assert p["mask"].tolist() == [[True, True], [True, False]]
        assert p["edge"][0] == pytest.approx([0.09, 0.05])
        assert p["decimal"][1, 1] == 1.0

    def test_reproducible_and_common_random_numbers(self):
        sc = Scenario("claimed", "x", 1.0)
        a = simulate(self.packed, DEFAULTS, sc, 500, 50, seed=3)
        b = simulate(self.packed, DEFAULTS, sc, 500, 50, seed=3)
        assert np.array_equal(a["end"], b["end"])
        # same seasons, bigger stakes: a winning season wins more
        c = simulate(self.packed, REQUESTED, sc, 500, 50, seed=3)
        up = a["end"] > 1.2
        assert (c["end"][up] >= a["end"][up]).mean() > 0.9

    def test_edge_direction(self):
        real = summarize(simulate(self.packed, DEFAULTS,
                                  Scenario("c", "c", 1.0), 2000, 150, 5))
        none = summarize(simulate(self.packed, DEFAULTS,
                                  Scenario("z", "z", 0.0), 2000, 150, 5))
        assert real["median_end"] > 1.05
        # even money with no edge: fair bets, the median drifts slightly down
        assert 0.85 < none["median_end"] <= 1.01
        assert real["p_profit"] > none["p_profit"]

    def test_full_kelly_swings_more(self):
        sc = Scenario("h", "h", 0.5)
        q = summarize(simulate(self.packed, DEFAULTS, sc, 2000, 150, 7))
        f = summarize(simulate(self.packed, FULL, sc, 2000, 150, 7))
        assert f["median_dd"] > q["median_dd"]
        assert f["p_lose_half"] >= q["p_lose_half"]

    def test_metrics_are_bounded(self):
        s = summarize(simulate(self.packed, FULL, Scenario("z", "z", 0.0),
                               500, 100, 9))
        for k in ("p_profit", "p_lose_half", "p_ruin", "median_dd", "p95_dd"):
            assert 0.0 <= s[k] <= 1.0
        assert s["p5_end"] <= s["median_end"] <= s["p95_end"]

    def test_shrink_is_redrawn_per_season(self):
        sim = simulate(self.packed, DEFAULTS, Scenario("b", "b", 0.4, 0.5),
                       4000, 5, seed=11)
        assert sim["shrink"].mean() == pytest.approx(0.4, abs=0.03)
        assert sim["shrink"].std() == pytest.approx(0.5, abs=0.03)


class TestReplay:
    def test_hand_worked_history(self):
        # day 1: one win, day 2: one loss, even money, quarter-Kelly of 0.1
        b = pd.DataFrame({
            "date": pd.to_datetime(["2025-11-01", "2025-11-02"]),
            "game_id": [1, 2], "side": ["HOME", "HOME"], "price": [100, 100],
            "decimal": [2.0, 2.0], "pm": [0.55, 0.55], "pf": [0.5, 0.5],
            "edge": [0.05, 0.05], "kelly": [0.1, 0.1], "won": [True, False]})
        r = replay(pack_days(b), DEFAULTS)
        # stake 0.025 trimmed to 0.02: 1.02, then 1.02 * 0.98
        assert r["end"] == pytest.approx(1.02 * 0.98)
        assert r["max_dd"] == pytest.approx(0.02)
        assert r["n_bets"] == 2


class TestRunAndReport:
    def test_season_days_scale(self):
        packed = pack_days(_bets(n_days=100))
        assert season_days(packed, n_games=1000, season_games=1400) == 140

    def test_report(self):
        result = run(_bets(), n_sims=300, seed=1, season_games=1400)
        t = result["table"]
        assert len(t) == 4 * len(STAKINGS)
        assert set(t["scenario"]) == {"claimed", "half", "zero", "backtest"}
        text = report(result)
        for needle in ("## Results", "Median end", "Ruin", "→",
                       "The real 2025-26 bets, replayed", "Decision"):
            assert needle in text
        assert " nan" not in text.lower() and "$nan" not in text.lower()


def test_cli_help_runs_nothing():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    out = subprocess.run([sys.executable, "-m", "betting.montecarlo", "--help"],
                         cwd=root, capture_output=True, text=True)
    assert out.returncode == 0 and "--sims" in out.stdout
