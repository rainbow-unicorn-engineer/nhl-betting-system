"""
Tests for the database-test guard in tests/conftest.py (pure: fake
environments, no database). The guard decides whether the database
tests may run; the flag NHL_ALLOW_DB_TESTS=1 on its own must never send
them to the database .env names, which is the live one.
"""
from tests.conftest import db_test_decision

LIVE = {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "nhl_betting", "POSTGRES_PASSWORD": "secret"}
COPY = {"POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "55432",
        "POSTGRES_DB": "nhl_betting_clone"}
FLAG = {"NHL_ALLOW_DB_TESTS": "1"}


def test_no_opt_in_skips():
    allowed, why = db_test_decision({}, LIVE)
    assert not allowed and "SKIP" in why


def test_flag_with_no_overrides_skips():
    allowed, why = db_test_decision(dict(FLAG), LIVE)
    assert not allowed
    assert "POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB are not set" in why


def test_flag_with_overrides_only_in_env_file_skips():
    # .env naming a copy is not enough: the overrides must be in the
    # environment, where they were set for this run on purpose
    allowed, _ = db_test_decision(dict(FLAG), {**LIVE, **COPY})
    assert not allowed


def test_flag_with_some_overrides_skips():
    env = {**FLAG, "POSTGRES_PORT": "55432", "POSTGRES_DB": "nhl_betting_clone"}
    allowed, why = db_test_decision(env, LIVE)
    assert not allowed and "POSTGRES_HOST is not set" in why


def test_flag_with_overrides_equal_to_env_file_skips():
    env = {**FLAG, "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
           "POSTGRES_DB": "nhl_betting"}
    allowed, why = db_test_decision(env, LIVE)
    assert not allowed and "same database as .env" in why


def test_another_loopback_spelling_is_still_the_same_database():
    env = {**FLAG, "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "05432",
           "POSTGRES_DB": "nhl_betting"}
    assert not db_test_decision(env, LIVE)[0]


def test_overrides_equal_to_the_settings_defaults_skip_when_env_file_is_silent():
    # .env without POSTGRES_* means config/settings.py's defaults
    env = {**FLAG, "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
           "POSTGRES_DB": "nhl_betting"}
    assert not db_test_decision(env, {})[0]


def test_flag_with_different_overrides_allows():
    allowed, why = db_test_decision({**FLAG, **COPY}, LIVE)
    assert allowed
    assert "ENABLED against 127.0.0.1:55432/nhl_betting_clone" in why


def test_another_database_on_the_same_server_allows():
    env = {**FLAG, "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
           "POSTGRES_DB": "nhl_betting_copy"}
    assert db_test_decision(env, LIVE)[0]


def test_test_database_allows():
    # from the environment ...
    allowed, why = db_test_decision({"POSTGRES_DB": "nhl_betting_test"}, LIVE)
    assert allowed and "localhost:5432/nhl_betting_test" in why
    # ... or from .env, with no flag either way
    assert db_test_decision({}, {**LIVE, "POSTGRES_DB": "nhl_betting_test"})[0]


def test_environment_wins_over_env_file_for_the_test_name():
    # .env names a _test database, but this run's environment names the
    # live one: the live one is what the tests would reach, so skip
    env = {"POSTGRES_DB": "nhl_betting"}
    assert not db_test_decision(env, {**LIVE, "POSTGRES_DB": "x_test"})[0]
    # a name that merely contains _test is not a _test database
    assert not db_test_decision({"POSTGRES_DB": "nhl_test_live"}, LIVE)[0]


# ── The guard itself: it must rewrite the environment, not just decide ──

import pytest  # noqa: E402

from tests.conftest import _guard  # noqa: E402


@pytest.fixture()
def live_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("POSTGRES_HOST=localhost\nPOSTGRES_PORT=5432\n"
                 "POSTGRES_DB=nhl_betting\n", encoding="utf-8")
    return f


@pytest.mark.parametrize("environ", [
    {},                                                     # no opt-in
    dict(FLAG),                                             # flag alone
    {**FLAG, "POSTGRES_DB": "nhl_betting_clone"},           # partial overrides
    {**FLAG, "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
     "POSTGRES_DB": "nhl_betting"},                         # same as .env
], ids=["no-flag", "flag-only", "partial", "same-as-env"])
def test_guard_points_blocked_runs_at_a_closed_port(environ, live_env_file):
    _guard(environ, live_env_file)
    assert environ["POSTGRES_HOST"] == "127.0.0.1"
    assert environ["POSTGRES_PORT"] == "1"


def test_guard_leaves_an_allowed_copy_alone(live_env_file):
    environ = {**FLAG, **COPY}
    _guard(environ, live_env_file)
    assert (environ["POSTGRES_HOST"], environ["POSTGRES_PORT"]) == \
        ("127.0.0.1", "55432")


@pytest.mark.parametrize("host,port", [
    ("localhost.", "5432"),          # trailing dot
    ("LOCALHOST", "5432"),
    ("localhost", "+5432"),          # int() accepts these, so does SQLAlchemy
    ("127.0.0.1", "5_432"),
    ("127.0.0.2", "5432"),           # anywhere in 127.0.0.0/8
    ("[::1]", "5432"),
    ("nosuchhost,127.0.0.1", "5432"),  # libpq host list
    ("localhost", "not-a-port"),     # unparseable counts as the same port
])
def test_odd_spellings_of_the_live_database_fail_closed(host, port):
    env = {**FLAG, "POSTGRES_HOST": host, "POSTGRES_PORT": port,
           "POSTGRES_DB": "nhl_betting"}
    allowed, _ = db_test_decision(env, LIVE)
    assert not allowed


def test_importing_conftest_without_overrides_forces_closed_port(tmp_path):
    """End to end: a fresh interpreter importing tests.conftest with no
    POSTGRES_* in its environment ends up on 127.0.0.1:1."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("POSTGRES_") and k != "NHL_ALLOW_DB_TESTS"}
    env["PYTHONPATH"] = str(root)
    for flag in (None, "1"):
        if flag:
            env["NHL_ALLOW_DB_TESTS"] = flag
        out = subprocess.run(
            [sys.executable, "-c",
             "import tests.conftest, os;"
             "print(os.environ['POSTGRES_HOST'], os.environ['POSTGRES_PORT'])"],
            cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert out.stdout.split() == ["127.0.0.1", "1"], out.stderr
