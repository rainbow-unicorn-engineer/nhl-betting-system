"""
Tests for ingestion/kalshi.py: tickers, team codes, parsing recorded
market and candle responses (tests/fixtures/kalshi_nhl.json: two
historical and two live markets, two historical 1-minute candles and two
live hourly candles, in the two field-name styles), matching events to
games, the candle windows (never past puck drop for the 1-minute close),
the 404 fallback between the live and historical paths, listing pages,
and the closing-price summary. No network. The database tests at the end
run only against a disposable copy (see tests/conftest.py) and write
synthetic rows (tickers starting KXNHLGAME-30, season 20302031), which
they delete.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import kalshi as k
from ingestion.polite import Reply

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="needs a disposable database (see tests/conftest.py)")

UTC = dt.timezone.utc
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "kalshi_nhl.json")
                     .read_text(encoding="utf-8"))


def test_event_ticker_and_codes():
    assert k.parse_event_ticker("KXNHLGAME-25DEC10DETCGY") == (dt.date(2025, 12, 10), "DETCGY")
    assert k.parse_event_ticker("KXNHLGAME-25XYZ10DETCGY") is None
    assert k.parse_event_ticker("KXNHLGAME-25FEB30DETCGY") is None     # no Feb 30
    assert k.parse_event_ticker("") is None
    assert k.market_code("KXNHLGAME-25DEC10DETCGY-DET") == "DET"
    assert k.market_code("nonsense") is None
    assert [k.nhl_code(c) for c in ("LA", "TB", "NJ", "SJ", "BOS", None)] == \
        ["LAK", "TBL", "NJD", "SJS", "BOS", None]


def test_market_rows_from_both_listings():
    for m, source in ((FIXTURE["historical_markets"][0], "historical"),
                      (FIXTURE["live_markets"][0], "live")):
        r = k.market_row(m, source)
        assert r["ticker"] == m["ticker"] and r["source"] == source
        assert r["series_ticker"] == "KXNHLGAME"
        assert r["event_date"] == k.parse_event_ticker(m["event_ticker"])[0]
        assert r["kalshi_team"] == m["ticker"].rsplit("-", 1)[1]
        assert r["result"] in ("yes", "no")
        assert r["settlement_value"] in (0.0, 1.0)
        assert r["open_time"].tzinfo is not None and r["settlement_ts"] > r["open_time"]
        assert json.loads(r["raw"])["ticker"] == m["ticker"]
    settled = [k.market_row(m, "historical") for m in FIXTURE["historical_markets"]]
    assert sorted(r["result"] for r in settled) == ["no", "yes"]   # one winner per event


def test_ts_and_dollars():
    assert k._ts("0001-01-01T00:00:00Z") is None and k._ts(None) is None and k._ts("x") is None
    assert k._ts("2026-06-15T02:58:23Z") == dt.datetime(2026, 6, 15, 2, 58, 23, tzinfo=UTC)
    assert k._ts("2026-06-15T02:58:23") == dt.datetime(2026, 6, 15, 2, 58, 23, tzinfo=UTC)
    assert k._dollars("0.4800") == 0.48 and k._dollars("") is None and k._dollars("x") is None


def test_candles_in_both_field_styles():
    ticker = FIXTURE["historical_markets"][0]["ticker"]
    hist = k.parse_candles(FIXTURE["historical_candles"], ticker, 1)
    live = k.parse_candles(FIXTURE["live_candles"], "T", 60)
    assert len(hist) == 2 and len(live) == 2
    for rows, period in ((hist, 1), (live, 60)):
        for r in rows:
            assert r["period_minutes"] == period
            assert 0 <= r["yes_bid_close"] <= r["yes_ask_close"] <= 1
            assert r["price_close"] is not None and r["volume"] is not None
            assert r["end_period_ts"].tzinfo is not None
    assert hist[0]["end_period_ts"] < hist[1]["end_period_ts"]


def test_candles_drop_bad_rows_and_repeats():
    body = {"candlesticks": [{"end_period_ts": "x"}, {"end_period_ts": 100},
                             {"end_period_ts": 100, "volume": "5"}, {"end_period_ts": 50}]}
    rows = k.parse_candles(body, "T", 60)
    assert [int(r["end_period_ts"].timestamp()) for r in rows] == [50, 100]
    assert rows[1]["volume"] == 5.0 and rows[1]["yes_bid_close"] is None


def game(gid, day, home, away, hour=23):
    return {"game_id": gid, "date": day, "home_team": home, "away_team": away,
            "start_time_utc": dt.datetime(day.year, day.month, day.day, hour, tzinfo=UTC)}


def mrow(event, code, day):
    return {"event_ticker": event, "kalshi_team": code, "event_date": day}


def test_map_games_exact_date_then_one_day_and_kalshi_codes():
    d = dt.date(2025, 12, 10)
    rows = [mrow("E1", "DET", d), mrow("E1", "LA", d),            # LA → LAK
            mrow("E2", "BOS", d), mrow("E2", "TOR", d),           # schedule date a day later
            mrow("E3", "NYR", d), mrow("E3", "MTL", d),           # preseason: no game
            mrow("E4", "CHI", d)]                                 # one market only
    games = [game(1, d, "LAK", "DET"), game(2, d + dt.timedelta(days=1), "TOR", "BOS"),
             game(3, d - dt.timedelta(days=40), "LAK", "DET")]
    out = k.map_games(rows, games)
    assert {e: g["game_id"] for e, g in out.items()} == {"E1": 1, "E2": 2}


def test_map_games_refuses_two_candidates():
    d = dt.date(2025, 12, 10)
    rows = [mrow("E", "BOS", d), mrow("E", "TOR", d)]
    games = [game(1, d - dt.timedelta(days=1), "TOR", "BOS"),
             game(2, d + dt.timedelta(days=1), "BOS", "TOR")]
    assert k.map_games(rows, games) == {}


def test_attach_games_sets_home_side():
    d = dt.date(2025, 12, 10)
    rows = [mrow("E1", "DET", d), mrow("E1", "LA", d), mrow("E9", "XX", d), mrow("E9", "YY", d)]
    n = k.attach_games(rows, [game(1, d, "LAK", "DET")])
    assert n == 1
    assert [(r["game_id"], r["team"], r["is_home"]) for r in rows] == \
        [(1, "DET", False), (1, "LAK", True), (None, "XX", None), (None, "YY", None)]


def test_candle_windows_end_the_minute_close_at_puck_drop():
    open_t = dt.datetime(2026, 6, 10, 4, tzinfo=UTC)
    settle = dt.datetime(2026, 6, 15, 3, tzinfo=UTC)
    start = dt.datetime(2026, 6, 15, 0, tzinfo=UTC)
    w = k.candle_windows({"open_time": open_t, "settlement_ts": settle, "close_time": None},
                         start)
    assert w[0] == (60, int(open_t.timestamp()), int(settle.timestamp()))
    assert w[1] == (1, int(start.timestamp()) - k.CLOSE_WINDOW_MIN * 60, int(start.timestamp()))
    assert k.candle_windows({"open_time": None, "settlement_ts": None, "close_time": None},
                            None) == []


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self.replies.pop(0)


def test_fetch_candles_falls_back_to_the_other_path_on_404():
    row = {"ticker": "KXNHLGAME-26JUN14CARVGK-VGK", "source": "live",
           "open_time": dt.datetime(2026, 6, 10, tzinfo=UTC),
           "settlement_ts": dt.datetime(2026, 6, 15, 3, tzinfo=UTC), "close_time": None}
    c = StubClient([Reply("not_found", http_status=404), Reply("ok", FIXTURE["live_candles"]),
                    Reply("ok", FIXTURE["historical_candles"])])
    status, rows, problem = k.fetch_candles(c, row, dt.datetime(2026, 6, 15, tzinfo=UTC))
    assert status == "ok" and problem is None and len(rows) == 4
    assert "/series/KXNHLGAME/markets/" in c.calls[0][0]
    assert "/historical/markets/" in c.calls[1][0]
    assert c.calls[2][1]["period_interval"] == 1


def test_fetch_candles_error_and_empty():
    row = {"ticker": "T", "source": "historical", "open_time": dt.datetime(2026, 1, 1, tzinfo=UTC),
           "settlement_ts": dt.datetime(2026, 1, 2, tzinfo=UTC), "close_time": None}
    assert k.fetch_candles(StubClient([Reply("error", problem="HTTP 500")]), row, None)[0] == "error"
    assert k.fetch_candles(StubClient([Reply("ok", {"candlesticks": []})]), row, None)[0] == "empty"


def test_list_markets_follows_the_cursor_and_fails_whole():
    c = StubClient([Reply("ok", {"markets": [{"ticker": "a"}], "cursor": "c1"}),
                    Reply("ok", {"markets": [{"ticker": "b"}], "cursor": ""})])
    assert [m["ticker"] for m in k.list_markets(c, "historical")] == ["a", "b"]
    assert c.calls[1][1]["cursor"] == "c1" and "/historical/markets" in c.calls[0][0]
    c = StubClient([Reply("ok", {"markets": [{"ticker": "a"}], "cursor": "c1"}),
                    Reply("error", problem="HTTP 429")])
    assert k.list_markets(c, "live") is None


def test_summarize_closes():
    start = dt.datetime(2026, 1, 1, 0, tzinfo=UTC)
    close = start - dt.timedelta(minutes=2)

    def line(gid, bid, ask, season=20252026):
        return {"game_id": gid, "season": season, "start_time_utc": start, "close_ts": close,
                "bid": bid, "ask": ask}
    lines = [line(1, 0.48, 0.49), line(1, 0.51, 0.52),
             line(2, 0.00, 1.00), line(2, 0.40, 0.42),       # an empty book is no price
             line(3, None, None)]                            # one market only, no close
    s = k.summarize_closes(lines)[20252026]
    assert s["games"] == 3 and s["with_close"] == 1
    assert s["median_spread"] == 0.01 and s["median_overround"] == 0.01
    assert s["median_lead_min"] == 2.0


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        k.main(["--help"])
    assert exc.value.code == 0
    assert "--backfill" in capsys.readouterr().out


# ── Database (disposable copy only) ───────────────────────────────

SEASON = 20302031
GID = 2030020021
EVENT = "KXNHLGAME-30NOV01NYRMTL"


@pytest.fixture
def seeded():
    start = dt.datetime(2030, 11, 1, 23, tzinfo=UTC)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.games (game_id, season, game_type, date, start_time_utc,
                                   home_team, away_team, game_state)
            VALUES (:g, :s, 2, :d, :t, 'MTL', 'NYR', 'OFF') ON CONFLICT (game_id) DO NOTHING
        """), {"g": GID, "s": SEASON, "d": start.date(), "t": start})
    yield start
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.kalshi_candles WHERE ticker LIKE 'KXNHLGAME-30%'"))
        conn.execute(text("DELETE FROM raw.kalshi_markets WHERE ticker LIKE 'KXNHLGAME-30%'"))
        conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GID})


def _market(code, result, start):
    m = {"ticker": f"{EVENT}-{code}", "event_ticker": EVENT, "result": result,
         "status": "finalized", "settlement_value_dollars": "1.0000" if result == "yes" else "0",
         "open_time": (start - dt.timedelta(days=3)).isoformat(),
         "settlement_ts": (start + dt.timedelta(hours=3)).isoformat()}
    return k.market_row(m, "historical")


def _candle(ticker, end, bid, ask, period=1):
    return {"ticker": ticker, "period_minutes": period, "end_period_ts": end,
            **{c: None for c in k._CANDLE_COLS[3:]}, "yes_bid_close": bid, "yes_ask_close": ask}


@requires_db
def test_closing_line_never_reads_an_in_play_candle(seeded):
    """Point-in-time: a candle ending after puck drop (here a 0.95 in-play
    price) must not become the close; rewriting it leaves the close alone."""
    start = seeded
    k.ensure_tables(engine)
    rows = [_market("NYR", "no", start), _market("MTL", "yes", start)]
    k.attach_games(rows, k.load_games(start.date(), start.date(), engine))
    k.store_markets(rows, engine)
    for r in rows:
        k.store_candles(r["ticker"], "ok", [
            _candle(r["ticker"], start - dt.timedelta(minutes=1), 0.45, 0.47),
            _candle(r["ticker"], start + dt.timedelta(minutes=30), 0.94, 0.96)], db=engine)
    before = {r["ticker"]: (float(r["bid"]), float(r["ask"]))
              for r in k.closing_lines(SEASON, engine)}
    with engine.begin() as conn:
        conn.execute(text("UPDATE raw.kalshi_candles SET yes_bid_close = 0.01 "
                          "WHERE ticker LIKE 'KXNHLGAME-30%' AND end_period_ts > :t"),
                     {"t": start})
    after = {r["ticker"]: (float(r["bid"]), float(r["ask"]))
             for r in k.closing_lines(SEASON, engine)}
    assert before == after == {f"{EVENT}-NYR": (0.45, 0.47), f"{EVENT}-MTL": (0.45, 0.47)}
    homes = {r["team"]: r["is_home"] for r in k.closing_lines(SEASON, engine)}
    assert homes == {"MTL": True, "NYR": False}
