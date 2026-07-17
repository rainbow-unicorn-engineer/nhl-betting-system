"""
features/util.py
Shared helpers and constants for the feature store (Layer 2).

Everything here must be point-in-time safe by construction: these are pure
functions with no database access, used by the feature builders.
"""
from typing import Optional

# Rolling windows (games) used by team and goalie feature builders.
# 5/10 capture form, 20/40 capture skill, 82 is a full-season baseline.
WINDOWS = [5, 10, 20, 40, 82]

# Season-stage boundaries (game number within a team's season)
EARLY_MAX = 20   # games 1-20
MID_MAX = 62     # games 21-62; 63+ is LATE


def per60(stat: float, toi_seconds: float, default: Optional[float] = None) -> Optional[float]:
    """
    Convert a raw counting stat to a per-60-minutes rate.

    per60(4, 3600) -> 4.0 (4 events in 60 min)
    Zero/negative TOI returns `default` (None) — never divides by zero.
    """
    if toi_seconds is None or toi_seconds <= 0 or stat is None:
        return default
    return stat * 3600.0 / toi_seconds


def safe_div(num: Optional[float], den: Optional[float], default: Optional[float] = None) -> Optional[float]:
    """
    Division that returns `default` instead of raising on zero/None denominator.
    """
    if num is None or den is None or den == 0:
        return default
    return num / den


def season_stage(game_num: int) -> str:
    """
    Classify a team's game number within a season into EARLY/MID/LATE.

    EARLY: games 1-20 (rolling stats unreliable, heavy shrinkage needed)
    MID:   games 21-62
    LATE:  games 63+ (includes any playoff games)

    game_num < 1 raises — a team cannot be playing game 0.
    """
    if game_num < 1:
        raise ValueError(f"game_num must be >= 1, got {game_num}")
    if game_num <= EARLY_MAX:
        return "EARLY"
    if game_num <= MID_MAX:
        return "MID"
    return "LATE"


def american_implied_prob(ml: Optional[float]) -> Optional[float]:
    """
    Convert American odds to the (vig-inclusive) implied probability.

    american_implied_prob(-150) -> 0.6, american_implied_prob(+130) -> ~0.435
    None/zero returns None. Normalize across outcomes to strip the vig.
    """
    if ml is None or ml == 0:
        return None
    ml = float(ml)
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)
