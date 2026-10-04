"""
Tests for pipeline.py wiring that needs no database or network.
"""
import pytest

import pipeline


def test_close_is_a_moneyline_only_snapshot(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(pipeline, "nhl_feed", lambda **kw: calls.append("nhl_feed"))
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(pipeline, "recommend",
                        lambda: pytest.fail("close must not recommend"))
    monkeypatch.setattr(pipeline, "starters",
                        lambda: pytest.fail("close must not fetch starters"))
    pipeline.close()
    assert calls == [{"markets": "h2h"}, "nhl_feed"]


def test_close_waits_for_network(monkeypatch):
    from ingestion import odds_api
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: False)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("no network, no snapshot"))
    monkeypatch.setattr(pipeline, "nhl_feed",
                        lambda **kw: pytest.fail("no network, no NHL feed"))
    pipeline.close()


def test_close_due_skips_when_not_due(monkeypatch):
    from ingestion import odds_api
    monkeypatch.setattr(odds_api, "close_is_due",
                        lambda: (False, "no game starts in the next 16 minutes"))
    monkeypatch.setattr(pipeline, "_wait_for_network",
                        lambda: pytest.fail("not due: no network wait"))
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("not due: no snapshot"))
    monkeypatch.setattr(pipeline, "nhl_feed",
                        lambda **kw: pytest.fail("not due: no NHL feed either"))
    pipeline.close(due=True)


def test_close_due_snapshots_moneyline_when_due(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(odds_api, "close_is_due", lambda: (True, "1 game soon"))
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(pipeline, "nhl_feed", lambda **kw: calls.append("nhl_feed"))
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    pipeline.close(due=True)
    assert calls == [{"markets": "h2h"}, "nhl_feed"]


def test_plain_close_does_not_check_timing(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(odds_api, "close_is_due",
                        lambda: pytest.fail("plain close never checks"))
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(pipeline, "nhl_feed", lambda **kw: None)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    pipeline.close()
    assert calls == [{"markets": "h2h"}]


class _StatusConn:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        self.sql.append(str(stmt))

        class _Result:
            def scalar(self):
                return 0

            def fetchall(self):
                return []
        return _Result()


def test_status_counts_upcoming_games_by_state(monkeypatch, capsys):
    """The NHL API never sends SCHEDULED (upcoming games are FUT or PRE),
    so counting it always printed 0."""
    conn = _StatusConn()
    monkeypatch.setattr(pipeline, "db_ready", lambda: True)
    monkeypatch.setattr(pipeline, "engine",
                        type("E", (), {"connect": lambda self: conn})())
    pipeline.db_status()
    assert not any("'SCHEDULED'" in s for s in conn.sql)
    assert any("game_state NOT IN ('FINAL', 'OFF')" in s for s in conn.sql)
    assert "Games (upcoming)" in capsys.readouterr().out


def test_setup_seeds_venues_only_when_missing(monkeypatch):
    from config import migrate
    seeded = []
    monkeypatch.setattr(migrate, "seed_venues", lambda: seeded.append(1) or 1)
    monkeypatch.setattr(migrate, "venues_missing", lambda: True)
    assert pipeline.seed_venues_if_missing().startswith("SEEDED")
    monkeypatch.setattr(migrate, "venues_missing", lambda: False)
    assert pipeline.seed_venues_if_missing() == "OK"
    assert seeded == [1]


# ── The chains: which steps run, in what order, and which are non-fatal ──

def _record_chain(monkeypatch, calls):
    """Stub every step the chains call, recording each by name."""
    from ingestion import espn_odds, moneypuck, nhl_api, odds_api
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(nhl_api, "daily_refresh", lambda: calls.append("daily_refresh"))
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(("snapshot_odds", kw)) or 0)
    monkeypatch.setattr(espn_odds, "backfill_historical_odds",
                        lambda season: calls.append("espn_lines"))
    monkeypatch.setattr(moneypuck, "refresh_season", lambda season: calls.append("moneypuck"))
    monkeypatch.setattr(pipeline, "features", lambda season=None: calls.append("features"))
    for step in ("settle", "settle_ledger", "starters", "recommend", "injuries"):
        monkeypatch.setattr(pipeline, step, lambda step=step: calls.append(step))
    monkeypatch.setattr(pipeline, "nhl_feed", lambda **kw: calls.append("nhl_feed"))
    monkeypatch.setattr(pipeline, "nhl_stats",
                        lambda season=None: calls.append("nhl_stats"))
    monkeypatch.setattr(pipeline, "props",
                        lambda **kw: pytest.fail("props never runs in a chain"))
    from config import runs
    monkeypatch.setattr(runs, "mark_finished",
                        lambda job, run_date=None: calls.append(("finished", job)))


def test_daily_adds_the_free_feeds_in_order(monkeypatch):
    calls = []
    _record_chain(monkeypatch, calls)
    pipeline.daily()
    assert calls == ["daily_refresh", ("snapshot_odds", {}), "nhl_feed", "espn_lines",
                     "nhl_stats", "moneypuck", "features", "settle", "settle_ledger",
                     "starters", "injuries", "recommend", ("finished", "daily")]


def test_daily_marks_itself_finished_only_at_the_end(monkeypatch):
    """The news monitor makes no pick before today's daily marker: a daily
    run that stops part-way (here the box-score refresh fails) writes none."""
    from ingestion import nhl_api
    calls = []
    _record_chain(monkeypatch, calls)

    def down():
        raise RuntimeError("NHL API down")
    monkeypatch.setattr(nhl_api, "daily_refresh", down)
    with pytest.raises(RuntimeError):
        pipeline.daily()
    assert ("finished", "daily") not in calls


def test_daily_marker_failure_is_non_fatal(monkeypatch, caplog):
    from config import runs
    calls = []
    _record_chain(monkeypatch, calls)

    def boom(job, run_date=None):
        raise RuntimeError("database gone")
    monkeypatch.setattr(runs, "mark_finished", boom)
    pipeline.daily()
    assert "Could not record the finished daily run" in caplog.text
    assert calls[-1] == "recommend"


def test_odds_pairs_its_snapshot_with_a_free_nhl_feed_snapshot(monkeypatch):
    from betting import alerts
    calls = []
    _record_chain(monkeypatch, calls)
    monkeypatch.setattr(alerts, "run_alerts", lambda: calls.append("alerts"))
    pipeline.odds()
    assert calls == [("snapshot_odds", {}), "nhl_feed", "starters", "recommend", "alerts"]


def test_refresh_is_free_data_only(monkeypatch):
    """The props machine's daily run: no Odds API request, no picks."""
    from ingestion import odds_api
    calls = []
    _record_chain(monkeypatch, calls)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("refresh spends no credits"))
    monkeypatch.setattr(pipeline, "recommend",
                        lambda: pytest.fail("refresh makes no picks"))
    pipeline.refresh()
    assert calls == ["daily_refresh", "nhl_stats", "injuries"]


def test_new_steps_are_non_fatal(monkeypatch):
    from ingestion import espn_injuries, nhl_odds, nhl_stats, props_odds

    def boom(*a, **k):
        raise RuntimeError("database gone")
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(nhl_odds, "snapshot", boom)
    monkeypatch.setattr(espn_injuries, "ingest_injuries", boom)
    monkeypatch.setattr(nhl_stats, "fill_missing", boom)
    monkeypatch.setattr(nhl_stats, "fill_season", boom)
    monkeypatch.setattr(props_odds, "snapshot_props", boom)
    pipeline.nhl_feed()
    pipeline.injuries()
    pipeline.nhl_stats()
    pipeline.nhl_stats(20252026)
    pipeline.props(due=True)


def test_news_is_non_fatal_and_passes_due(monkeypatch, caplog):
    from betting import news

    seen = []
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(news, "run_news", lambda due: seen.append(due))
    pipeline.news(due=True)
    pipeline.news()
    assert seen == [True, False]

    def boom(due):
        raise RuntimeError("database gone")
    monkeypatch.setattr(news, "run_news", boom)
    pipeline.news(due=True)
    assert "News monitor failed (non-fatal)" in caplog.text


def test_news_waits_for_network(monkeypatch):
    from betting import news
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: False)
    monkeypatch.setattr(news, "run_news", lambda due: pytest.fail("no network, no run"))
    pipeline.news(due=True)


def test_ledger_settlement_is_non_fatal(monkeypatch, caplog):
    from betting import ledger

    def boom():
        raise RuntimeError("database gone")
    monkeypatch.setattr(ledger, "settle_slips", boom)
    pipeline.settle_ledger()
    assert "Bet-ledger settlement failed (non-fatal)" in caplog.text


def test_settle_command_settles_paper_picks_then_the_ledger(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "db_ready", lambda: True)
    monkeypatch.setattr(pipeline, "settle", lambda: calls.append("settle"))
    monkeypatch.setattr(pipeline, "settle_ledger", lambda: calls.append("ledger"))
    assert pipeline.main(["settle"]) == 0
    assert calls == ["settle", "ledger"]


def test_nhl_feed_skips_when_idle_by_default(monkeypatch):
    from ingestion import nhl_odds
    seen = []
    monkeypatch.setattr(nhl_odds, "snapshot", lambda skip_when_idle: seen.append(skip_when_idle))
    pipeline.nhl_feed()
    pipeline.nhl_feed(skip_when_idle=False)
    assert seen == [True, False]


def test_nhl_stats_daily_form_and_season_form(monkeypatch, caplog):
    from ingestion import nhl_stats
    seen = []
    monkeypatch.setattr(nhl_stats, "fill_missing",
                        lambda: seen.append("missing") or {"failed_windows": 0})
    monkeypatch.setattr(nhl_stats, "fill_season",
                        lambda s: seen.append(s) or {"failed_windows": 2})
    pipeline.nhl_stats()
    pipeline.nhl_stats(20252026)
    assert seen == ["missing", 20252026]
    assert "2 window(s) failed" in caplog.text


def test_props_passes_due_and_markets_and_waits_for_network(monkeypatch):
    from ingestion import props_odds
    seen = []
    monkeypatch.setattr(props_odds, "snapshot_props", lambda **kw: seen.append(kw) or 0)
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    pipeline.props(due=True, markets="player_points")
    assert seen == [{"markets": "player_points", "due": True}]
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: False)
    pipeline.props()
    assert len(seen) == 1


# ── The command line ───────────────────────────────────────────────

COMMANDS = ("setup", "status", "backfill", "features", "daily", "odds", "close",
            "recommend", "starters", "settle", "refresh", "props", "nhl-odds",
            "compare-feeds", "injuries", "news", "nhl-stats")


@pytest.mark.parametrize("command", COMMANDS)
def test_help_runs_nothing(monkeypatch, capsys, command):
    monkeypatch.setattr(pipeline, "db_ready", lambda: pytest.fail("--help touched the DB"))
    monkeypatch.setattr(pipeline, "setup_check", lambda: pytest.fail("--help ran setup"))
    monkeypatch.setattr(pipeline, "db_status", lambda: pytest.fail("--help ran status"))
    with pytest.raises(SystemExit) as exc:
        pipeline.main([command, "--help"])
    assert exc.value.code == 0
    assert f"python pipeline.py {command}" in capsys.readouterr().out


def test_no_command_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "db_ready", lambda: pytest.fail("no command, no DB"))
    assert pipeline.main([]) == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


def test_unknown_command_and_bad_options_exit_2(monkeypatch):
    monkeypatch.setattr(pipeline, "db_ready", lambda: pytest.fail("bad input, no DB"))
    for argv in (["nope"], ["nhl-stats", "--season", "2025"],
                 ["compare-feeds", "--date", "29/09/2026"], ["props", "--sideways"]):
        with pytest.raises(SystemExit) as exc:
            pipeline.main(argv)
        assert exc.value.code == 2, argv


def test_no_database_runs_nothing(monkeypatch):
    monkeypatch.setattr(pipeline, "db_ready", lambda: False)
    monkeypatch.setattr(pipeline, "props", lambda **kw: pytest.fail("no DB, no props"))
    assert pipeline.main(["props", "--due"]) == 1


@pytest.mark.parametrize("argv, expected", [
    (["props"], ("props", {"due": False, "markets": None})),
    (["props", "--due", "--markets", "player_points"],
     ("props", {"due": True, "markets": "player_points"})),
    (["refresh"], ("refresh", {})),
    (["injuries"], ("injuries", {})),
    (["nhl-stats"], ("nhl_stats", {"season": None})),
    (["nhl-stats", "--season", "20252026"], ("nhl_stats", {"season": 20252026})),
    (["nhl-odds"], ("nhl_feed", {"skip_when_idle": False})),
    (["close", "--due"], ("close", {"due": True})),
    (["news"], ("news", {"due": False})),
    (["news", "--due"], ("news", {"due": True})),
    (["features", "--season", "20242025"], ("features", {"season": 20242025})),
])
def test_commands_dispatch(monkeypatch, argv, expected):
    seen = []
    monkeypatch.setattr(pipeline, "db_ready", lambda: True)
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(pipeline, "props", lambda due, markets: seen.append(
        ("props", {"due": due, "markets": markets})))
    monkeypatch.setattr(pipeline, "refresh", lambda: seen.append(("refresh", {})))
    monkeypatch.setattr(pipeline, "injuries", lambda: seen.append(("injuries", {})))
    monkeypatch.setattr(pipeline, "nhl_stats", lambda season: seen.append(
        ("nhl_stats", {"season": season})))
    monkeypatch.setattr(pipeline, "nhl_feed", lambda skip_when_idle: seen.append(
        ("nhl_feed", {"skip_when_idle": skip_when_idle})))
    monkeypatch.setattr(pipeline, "close", lambda due: seen.append(("close", {"due": due})))
    monkeypatch.setattr(pipeline, "news", lambda due: seen.append(("news", {"due": due})))
    monkeypatch.setattr(pipeline, "features", lambda season: seen.append(
        ("features", {"season": season})))
    assert pipeline.main(argv) == 0
    assert seen == [expected]


def test_compare_feeds_prints_the_report(monkeypatch, capsys):
    from datetime import date
    from ingestion import nhl_odds
    seen = []
    monkeypatch.setattr(pipeline, "db_ready", lambda: True)
    monkeypatch.setattr(nhl_odds, "compare_feeds", lambda d: seen.append(d) or {"r": 1})
    monkeypatch.setattr(nhl_odds, "format_report",
                        lambda result, detail: f"REPORT {result} detail={detail}")
    assert pipeline.main(["compare-feeds", "--date", "2026-09-29", "--detail"]) == 0
    assert seen == [date(2026, 9, 29)]
    assert "REPORT {'r': 1} detail=True" in capsys.readouterr().out
