"""
Tests for config/migrate.py that need no database: the SQL splitter the
venue seed runs through, applying the seed through a fake engine, and
the command line (--help runs nothing).
"""
from contextlib import contextmanager

import pytest

from config import migrate
from config.migrate import SEED_VENUES_SQL, split_sql


class _FakeConn:
    def __init__(self):
        self.statements = []

    def exec_driver_sql(self, stmt):
        self.statements.append(stmt)


class _FakeEngine:
    def __init__(self):
        self.conn = _FakeConn()

    @contextmanager
    def begin(self):
        yield self.conn


class TestSplitSql:
    def test_semicolons_in_strings_and_comments_do_not_split(self):
        script = ("-- a comment; with a semicolon\n"
                  "INSERT INTO t VALUES ('a;b', 'it''s');\n"
                  "/* block; comment */ SELECT \"odd;name\" FROM t;\n"
                  "SELECT 1 -- trailing; comment\n")
        assert split_sql(script) == [
            "INSERT INTO t VALUES ('a;b', 'it''s')",
            "SELECT \"odd;name\" FROM t",
            "SELECT 1",
        ]

    def test_empty_and_comment_only(self):
        assert split_sql("") == []
        assert split_sql("-- nothing here;\n  ;  \n") == []

    def test_the_real_seed_file_is_one_upsert_of_every_team(self):
        (stmt,) = split_sql(SEED_VENUES_SQL.read_text(encoding="utf-8"))
        assert stmt.startswith("INSERT INTO raw.teams")
        assert "ON CONFLICT (team_abbrev) DO UPDATE SET" in stmt
        assert "Montréal Canadiens" in stmt
        assert stmt.count("'America/") == 33          # 32 franchises + ARI


def test_seed_venues_reads_utf8_and_runs_each_statement(tmp_path):
    sql = tmp_path / "seed.sql"
    sql.write_bytes("-- seed; test\nUPDATE raw.teams SET venue_city = 'Montréal' "
                    "WHERE team_abbrev = 'MTL';\nSELECT 1;\n".encode("utf-8"))
    fake = _FakeEngine()
    assert migrate.seed_venues(sql, db=fake) == 2
    assert fake.conn.statements[0].endswith("'Montréal' WHERE team_abbrev = 'MTL'")
    assert fake.conn.statements[1] == "SELECT 1"


def test_help_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(migrate, "ensure_schema",
                        lambda: pytest.fail("--help ran the migration"))
    monkeypatch.setattr(migrate, "seed_venues",
                        lambda *a, **k: pytest.fail("--help ran the seed"))
    with pytest.raises(SystemExit) as exc:
        migrate.main(["--help"])
    assert exc.value.code == 0
    assert "--seed-venues" in capsys.readouterr().out


def test_seed_venues_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(migrate, "ensure_schema", lambda: calls.append("schema"))
    monkeypatch.setattr(migrate, "seed_venues", lambda: calls.append("seed"))
    migrate.main([])
    migrate.main(["--seed-venues"])
    assert calls == ["schema", "schema", "seed"]


def test_every_upgrade_column_is_in_schema_sql():
    """COLUMNS brings an old database up to db/schema.sql: each column it
    adds must be declared, with the same type, in schema.sql's table."""
    import re
    schema = (migrate.PROJECT_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    for sch, table, column, ddl_type in migrate.COLUMNS:
        body = re.search(rf"CREATE TABLE IF NOT EXISTS {sch}\.{table} \((.*?)\n\);",
                         schema, re.S)
        assert body, f"{sch}.{table} not in schema.sql"
        line = re.search(rf"^\s*{column}\s+(.*)$", body.group(1), re.M)
        assert line, f"{sch}.{table}.{column} not in schema.sql"
        assert line.group(1).startswith(ddl_type.split()[0]), (column, line.group(1))


def test_recommendations_store_the_scheduled_start():
    assert ("betting", "recommendations", "scheduled_start",
            "TIMESTAMPTZ") in migrate.COLUMNS


# ── New tables (TABLES) ────────────────────────────────────────────

import re  # noqa: E402

_SCHEMA_SQL = (migrate.PROJECT_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")


def _table_body(sql: str, schema: str, table: str) -> str:
    """The text between the parentheses of one CREATE TABLE statement."""
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {schema}\.{table} \((.*?)\n\s*\)\s*(;|$)",
                  sql, re.S)
    assert m, f"no CREATE TABLE for {schema}.{table}"
    return m.group(1)


def _definitions(body: str) -> dict:
    """{column: definition} plus {'constraints': [...]}, comments removed and
    spaces collapsed, so two copies compare on what they declare."""
    cols, constraints = {}, []
    for raw in body.splitlines():
        line = re.sub(r"--.*$", "", raw).strip().rstrip(",").strip()
        if not line:
            continue
        line = " ".join(line.split())
        if line.upper().startswith(("PRIMARY KEY (", "UNIQUE (")):
            constraints.append(line)
        else:
            name, _, definition = line.partition(" ")
            cols[name] = definition
    return {"columns": cols, "constraints": constraints}


def _create_table(statements) -> str:
    (stmt,) = [s for s in statements if "CREATE TABLE" in s]
    return stmt


def _indexes(statements) -> list:
    return [" ".join(s.split()) for s in statements if "CREATE INDEX" in s]


def test_tables_hold_only_new_tables_with_their_ddl():
    names = [(s, t) for s, t, _ in migrate.TABLES]
    assert names == [("raw", "nhl_feed_snapshots"), ("raw", "injuries"),
                     ("raw", "prop_odds_hist"), ("raw", "prop_odds_fetches"),
                     ("raw", "prop_snapshots"), ("raw", "lineups"),
                     ("raw", "lineup_fetches"), ("raw", "news_events"),
                     ("raw", "news_state"), ("raw", "news_runs"),
                     ("raw", "pipeline_runs"), ("betting", "slips"),
                     ("betting", "slip_legs"), ("betting", "bankroll_txns")]
    for schema, table, statements in migrate.TABLES:
        create = _create_table(statements)
        assert f"CREATE TABLE IF NOT EXISTS {schema}.{table} (" in create
        for idx in _indexes(statements):
            assert idx.startswith("CREATE INDEX IF NOT EXISTS ")
            assert f" ON {schema}.{table}(" in idx


def test_every_new_table_is_in_schema_sql_with_the_same_columns():
    """A fresh database (schema.sql) and an upgraded one (TABLES) must end
    up with the same tables, column for column."""
    for schema, table, statements in migrate.TABLES:
        mine = _definitions(_table_body(_create_table(statements), schema, table))
        declared = _definitions(_table_body(_SCHEMA_SQL, schema, table))
        assert mine == declared, f"{schema}.{table} differs from db/schema.sql"


def test_every_new_index_is_in_schema_sql():
    flat = " ".join(_SCHEMA_SQL.split())
    for schema, table, statements in migrate.TABLES:
        for idx in _indexes(statements):
            assert idx + ";" in flat, f"{idx} not in db/schema.sql"


def test_module_ddl_matches_migrate():
    """Each module applies its own copy of the DDL on first use; it must
    declare exactly what migrate and schema.sql declare."""
    from betting import news
    from ingestion import (dailyfaceoff_lines, espn_injuries, espn_odds, espn_props,
                           nhl_odds, nhl_stats, props_odds)
    by_table = {t: s for _, t, s in migrate.TABLES}
    for module_ddl in (nhl_odds.DDL, espn_injuries.DDL, espn_props.DDL, props_odds.DDL,
                       dailyfaceoff_lines.DDL, news.DDL):
        for stmt in module_ddl:
            if "CREATE TABLE" in stmt:
                table = re.search(r"CREATE TABLE IF NOT EXISTS raw\.(\w+)", stmt).group(1)
                assert (_definitions(_table_body(stmt, "raw", table))
                        == _definitions(_table_body(_create_table(by_table[table]),
                                                    "raw", table))), table
            else:
                table = re.search(r" ON raw\.(\w+)\(", stmt).group(1)
                assert " ".join(stmt.split()) in _indexes(by_table[table]), stmt
    assert set(espn_odds.HISTORICAL_ODDS_COLUMNS) <= set(migrate.COLUMNS)
    assert ("raw", "skater_games", "stats_filled_at", "TIMESTAMP") in migrate.COLUMNS
    assert nhl_stats.DDL == ["ALTER TABLE raw.skater_games ADD COLUMN IF NOT EXISTS "
                             "stats_filled_at TIMESTAMP"]


# ── ensure_schema on a fake connection ─────────────────────────────

class _SchemaConn:
    def __init__(self, tables, columns):
        self.tables, self.columns, self.ddl = tables, columns, []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "information_schema.tables" in sql:
            return iter(self.tables)
        if "information_schema.columns" in sql:
            return iter(self.columns)
        self.ddl.append(" ".join(sql.split()))
        return None


class _SchemaEngine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def begin(self):
        yield self.conn


def _all_tables():
    return [(s, t) for s, t, _ in migrate.TABLES] + [
        (s, t) for s, t, _, _ in migrate.COLUMNS]


def _all_columns():
    return [(s, t, c) for s, t, c, _ in migrate.COLUMNS]


def _run_ensure_schema(monkeypatch, tables, columns):
    conn = _SchemaConn(tables, columns)
    monkeypatch.setattr(migrate, "engine", _SchemaEngine(conn))
    monkeypatch.setattr(migrate, "_done", False)
    migrate.ensure_schema()
    return conn


def test_up_to_date_database_runs_no_ddl(monkeypatch):
    conn = _run_ensure_schema(monkeypatch, _all_tables(), _all_columns())
    assert conn.ddl == []


def test_missing_table_is_created_with_its_indexes(monkeypatch):
    tables = [t for t in _all_tables() if t != ("raw", "prop_snapshots")]
    conn = _run_ensure_schema(monkeypatch, tables, _all_columns())
    assert len(conn.ddl) == 4
    assert conn.ddl[0].startswith("CREATE TABLE IF NOT EXISTS raw.prop_snapshots (")
    assert all(s.startswith("CREATE INDEX IF NOT EXISTS idx_prop_snapshots_")
               for s in conn.ddl[1:])


def test_missing_columns_are_added(monkeypatch):
    columns = [c for c in _all_columns()
               if c[1] != "historical_odds" and c[2] != "stats_filled_at"]
    conn = _run_ensure_schema(monkeypatch, _all_tables(), columns)
    assert len(conn.ddl) == 15
    assert ("ALTER TABLE raw.historical_odds ADD COLUMN IF NOT EXISTS "
            "over_price INTEGER") in conn.ddl
    assert ("ALTER TABLE raw.skater_games ADD COLUMN IF NOT EXISTS "
            "stats_filled_at TIMESTAMP") in conn.ddl


def test_ensure_schema_runs_once_per_process(monkeypatch):
    conn = _run_ensure_schema(monkeypatch, [], [])
    first = len(conn.ddl)
    assert first > 0
    migrate.ensure_schema()
    assert len(conn.ddl) == first
