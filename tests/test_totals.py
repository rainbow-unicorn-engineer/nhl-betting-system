"""
Tests for models/totals.py — every PMF number is hand-computed.
"""
import numpy as np
import pandas as pd
import pytest

from config.settings import check_db_connection
from models.totals import (env_rates, expected_total, nll_of_totals,
                           poisson_pmf, prob_over, total_pmf)

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


class TestPMFMachinery:
    def test_poisson_pmf_normalized_and_shaped(self):
        pmf = poisson_pmf(np.array([2.5, 3.5]))
        assert pmf.shape == (2, 13)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0)
        # mode of Poisson(2.5) is 2
        assert pmf[0].argmax() == 2

    def test_total_pmf_hand_computed_with_ot_shift(self):
        # H ~ [0.5, 0.5] over {0,1}; A ~ [0.4, 0.6] over {0,1}
        # (0,0) p=.20 tie -> T=1;  (0,1) p=.30 -> T=1
        # (1,0) p=.20 -> T=1;      (1,1) p=.30 tie -> T=3
        ph = np.array([[0.5, 0.5]])
        pa = np.array([[0.4, 0.6]])
        tp = total_pmf(ph, pa)
        assert tp.shape == (1, 4)
        np.testing.assert_allclose(tp[0], [0.0, 0.7, 0.0, 0.3], atol=1e-12)
        np.testing.assert_allclose(tp.sum(axis=1), 1.0)

    def test_no_mass_on_even_regulation_ties(self):
        # With the OT shift, T=0 is impossible (0-0 becomes 1)
        ph = poisson_pmf(np.array([3.0]))
        pa = poisson_pmf(np.array([2.7]))
        tp = total_pmf(ph, pa)
        assert tp[0, 0] == 0.0
        np.testing.assert_allclose(tp.sum(axis=1), 1.0)

    def test_prob_over_and_push(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        p_over, p_push = prob_over(tp, [1.5])
        assert p_over[0] == pytest.approx(0.3) and p_push[0] == 0.0
        p_over, p_push = prob_over(tp, [1.0])   # integer line: push at 1
        assert p_over[0] == pytest.approx(0.3)
        assert p_push[0] == pytest.approx(0.7)

    def test_expected_total(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        assert expected_total(tp)[0] == pytest.approx(0.7 + 0.9)

    def test_nll_of_totals(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        assert nll_of_totals(tp, np.array([3]))[0] == pytest.approx(-np.log(0.3))


class TestEnvRates:
    def test_trailing_mean_and_prior_blend(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="6h"))
        y = np.full(1000, 4.0)
        env = env_rates(dates, y, prior=2.0)
        # First rows: no history -> pure prior
        assert env[0] == pytest.approx(2.0)
        # Late rows: prior weight is fixed, data dominates but blend remains
        assert 3.0 < env[-1] < 4.0
        assert env[-1] > env[100]   # monotone approach toward the data

    def test_point_in_time_no_same_day_leak(self):
        dates = pd.Series(pd.to_datetime(
            ["2024-01-01"] * 5 + ["2024-01-02"] * 5 + ["2024-01-03"] * 5))
        y1 = np.array([3.0] * 5 + [3.0] * 5 + [3.0] * 5)
        y2 = y1.copy()
        y2[10:] = 99.0        # change only Jan-3 outcomes
        env1 = env_rates(dates, y1, prior=3.0)
        env2 = env_rates(dates, y2, prior=3.0)
        # Jan-3 rows' own outcomes must not affect their own env values
        np.testing.assert_allclose(env1[10:], env2[10:])


@requires_db
class TestDataset:
    def test_regulation_goals_and_shapes(self):
        from models.totals import ATTACK_FEATURES, load_totals_dataset
        Xh, Xa, y_h, y_a, meta, names = load_totals_dataset()
        assert Xh.shape == Xa.shape == (len(meta), len(ATTACK_FEATURES))
        assert names == ATTACK_FEATURES
        # regulation goals: non-negative ints, and reg total <= settlement
        assert (y_h >= 0).all() and (y_a >= 0).all()
        assert ((y_h + y_a) <= meta["total"].to_numpy()).all()
        # in OT/SO games the settlement total is exactly reg total + 1
        # (spot property: no game has settlement > reg + 1)
        assert ((meta["total"].to_numpy() - (y_h + y_a)) <= 1).all()

    def test_is_home_column_constant_per_matrix(self):
        from models.totals import ATTACK_FEATURES, load_totals_dataset
        Xh, Xa, *_ = load_totals_dataset()
        j = ATTACK_FEATURES.index("is_home")
        assert (Xh[:, j] == 1.0).all() and (Xa[:, j] == 0.0).all()
