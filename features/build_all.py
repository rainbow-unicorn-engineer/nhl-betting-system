"""
features/build_all.py
Feature-build orchestration (Phase 2, Task 7).

Runs the Task 2-6 builders in dependency order:

    team_features ──┐
    goalie_features ─┼─> build_vectors -> features.game_vector
    schedule_features┤
    elo ─────────────┘

Scope semantics: pass a season (YYYYYYYY) to rebuild just that season's
rows, or None for all seasons. Elo is the one exception — ratings chain
through every prior game, so it ALWAYS recomputes full history regardless
of scope (it is cheap: a single pass over raw.games).

Invoked via `python pipeline.py features [--season YYYYYYYY]` and as the
feature step of the daily refresh chain.
"""
import logging
from typing import Optional

from features.build_vectors import build_game_vectors
from features.elo import build_elo
from features.goalie_features import build_goalie_rolling
from features.schedule_features import build_schedule_features
from features.team_features import build_team_rolling

logger = logging.getLogger("nhl.features.build_all")


def build_features(season: Optional[int] = None) -> dict:
    """
    Run all feature builders in dependency order. Returns row counts per
    stage so callers (pipeline, tests) can assert on completeness.
    """
    scope = season or "ALL"
    logger.info("=" * 60)
    logger.info(f"FEATURE BUILD — scope: {scope}")
    logger.info("=" * 60)

    counts = {
        "team_rolling": build_team_rolling(season),
        "goalie_rolling": build_goalie_rolling(season),
        "matchup_schedule": build_schedule_features(season),
        "elo": build_elo(),  # always full history — ratings chain
        "game_vector": build_game_vectors(season),
    }

    logger.info("FEATURE BUILD COMPLETE: "
                + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build all feature tables")
    parser.add_argument("--season", type=int, default=None,
                        help="Season as YYYYYYYY (e.g. 20212022); omit for all")
    args = parser.parse_args()
    build_features(args.season)
