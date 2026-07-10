"""Unit tests for features/util.py helpers."""
import pytest

from features.util import WINDOWS, per60, safe_div, season_stage


class TestPer60:
    def test_basic_rate(self):
        # 4 events in a full 60 minutes -> 4.0 per 60
        assert per60(4, 3600) == 4.0

    def test_partial_toi(self):
        # 2 events in 30 minutes -> 4.0 per 60
        assert per60(2, 1800) == 4.0

    def test_zero_toi_returns_default(self):
        assert per60(5, 0) is None
        assert per60(5, 0, default=0.0) == 0.0

    def test_negative_toi_returns_default(self):
        assert per60(5, -100) is None

    def test_none_inputs(self):
        assert per60(None, 3600) is None
        assert per60(3, None) is None

    def test_zero_stat_is_valid(self):
        assert per60(0, 3600) == 0.0


class TestSafeDiv:
    def test_basic(self):
        assert safe_div(10, 4) == 2.5

    def test_zero_denominator(self):
        assert safe_div(10, 0) is None
        assert safe_div(10, 0, default=0.5) == 0.5

    def test_none_inputs(self):
        assert safe_div(None, 5) is None
        assert safe_div(5, None) is None

    def test_zero_numerator(self):
        assert safe_div(0, 5) == 0.0


class TestSeasonStage:
    def test_game_one_is_early(self):
        # game 1 of the season — the classic edge case
        assert season_stage(1) == "EARLY"

    def test_boundaries(self):
        assert season_stage(20) == "EARLY"
        assert season_stage(21) == "MID"
        assert season_stage(62) == "MID"
        assert season_stage(63) == "LATE"
        assert season_stage(82) == "LATE"

    def test_playoff_games_are_late(self):
        assert season_stage(90) == "LATE"

    def test_invalid_game_number(self):
        with pytest.raises(ValueError):
            season_stage(0)
        with pytest.raises(ValueError):
            season_stage(-3)


class TestWindows:
    def test_expected_windows(self):
        assert WINDOWS == [5, 10, 20, 40, 82]
