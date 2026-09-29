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
