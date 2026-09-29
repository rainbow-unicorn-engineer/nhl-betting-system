"""
tests/conftest.py — keeps a plain `pytest` run off the live database.

pytest loads this file before it imports any test module in tests/, so
the guard below runs before anything imports config.settings (which
builds its database engine from the environment at import time).

Why: the database tests write to whatever database .env names. They
settle every due paper pick and rebuild betting.bankroll_log, rewrite one
game's start_time_utc, delete and re-issue recommendations for their
sample games, overwrite those games' predictions, and re-register models.
On the live database that corrupts real picks and the paper ledger.

So unless you opt in, every database test SKIPS: POSTGRES_HOST/PORT are
pointed at a closed port (127.0.0.1:1) before config.settings is read.
.env is only read here (dotenv_values), never applied.

Opt in only against a disposable database, either way:
  - NHL_ALLOW_DB_TESTS=1 together with POSTGRES_HOST, POSTGRES_PORT and
    POSTGRES_DB set in the environment (not only in .env), naming a
    database other than the one .env names. The flag alone, or overrides
    that name the .env database again, keep the tests skipped: the flag
    must never send the suite to the live database.
  - POSTGRES_DB ends in "_test" (set in the environment or in .env).
"""
import ipaddress
import os
from pathlib import Path
from typing import Mapping, MutableMapping

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent.parent

# config/settings.py's defaults, for a setting neither place sets
_DEFAULTS = {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
             "POSTGRES_DB": "nhl_betting"}
_TARGET_KEYS = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB")


def _hosts(h: str) -> set:
    """Normalised host names in a POSTGRES_HOST value. libpq accepts a
    comma-separated list; brackets, case and a trailing dot are dropped."""
    parts = (p.strip().lower().strip("[]").rstrip(".") for p in h.split(","))
    return {p for p in parts}


def _is_loopback(h: str) -> bool:
    if h in ("", "localhost") or h.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _same_server(a: str, b: str) -> bool:
    """True when the two host values could reach the same server. Fails
    closed: any shared name counts, and any loopback spelling on both
    sides (localhost, localhost., 127.0.0.0/8, ::1) counts as one."""
    ha, hb = _hosts(a), _hosts(b)
    if ha & hb:
        return True
    return any(map(_is_loopback, ha)) and any(map(_is_loopback, hb))


def _same_port(a: str, b: str) -> bool:
    """Ports compared the way SQLAlchemy reads them (int()). Anything that
    doesn't parse counts as the same port, so an odd spelling can never
    slip past the check (fails closed)."""
    try:
        return int(a.strip()) == int(b.strip())
    except ValueError:
        return True


def db_test_decision(environ: Mapping[str, str],
                     file_values: Mapping[str, str]) -> tuple:
    """(allowed, why) for the database tests. Pure.

    environ: the process environment; file_values: what .env sets. As
    with load_dotenv, a variable in the environment wins over .env, and a
    setting in neither place takes config/settings.py's default.

    Allowed when the database name in effect ends in "_test", or when
    NHL_ALLOW_DB_TESTS=1 comes with POSTGRES_HOST, POSTGRES_PORT and
    POSTGRES_DB all set in environ and naming a database other than the
    one .env names (same server, same port and same name = the same
    database). Everything else skips."""
    effective = {**_DEFAULTS, **file_values, **environ}
    db = effective["POSTGRES_DB"]
    target = f"{effective['POSTGRES_HOST']}:{effective['POSTGRES_PORT']}/{db}"
    if db.endswith("_test"):
        return True, (f"database tests ENABLED against {target} (a _test "
                      f"database): they write to it (see tests/conftest.py)")

    if environ.get("NHL_ALLOW_DB_TESTS") != "1":
        return False, ("database tests SKIP. To run them against a disposable "
                       "copy, set NHL_ALLOW_DB_TESTS=1 together with "
                       "POSTGRES_HOST, POSTGRES_PORT and POSTGRES_DB pointing "
                       "at the copy, or use a POSTGRES_DB ending in _test "
                       "(see tests/conftest.py)")

    missing = [k for k in _TARGET_KEYS if not environ.get(k, "").strip()]
    if missing:
        return False, (f"database tests SKIP: NHL_ALLOW_DB_TESTS=1 is ignored, "
                       f"because {', '.join(missing)} "
                       f"{'is' if len(missing) == 1 else 'are'} not set in the "
                       f"environment. Set all three of POSTGRES_HOST, "
                       f"POSTGRES_PORT and POSTGRES_DB for the run, pointing "
                       f"at the copy (see tests/conftest.py)")

    env_file = {**_DEFAULTS, **file_values}
    if (_same_server(environ["POSTGRES_HOST"], env_file["POSTGRES_HOST"])
            and _same_port(environ["POSTGRES_PORT"], env_file["POSTGRES_PORT"])
            and environ["POSTGRES_DB"].strip() == env_file["POSTGRES_DB"].strip()):
        return False, ("database tests SKIP: NHL_ALLOW_DB_TESTS=1 is ignored, "
                       "because POSTGRES_HOST, POSTGRES_PORT and POSTGRES_DB "
                       "name the same database as .env. Point them at a "
                       "disposable copy (see tests/conftest.py)")

    return True, (f"database tests ENABLED against {target}: they write to it "
                  f"(see tests/conftest.py)")


def _guard(environ: MutableMapping[str, str] = os.environ,
           env_file: Path = _ROOT / ".env") -> str:
    """Apply the decision: when the tests may not touch a database, point
    POSTGRES_HOST/PORT in `environ` at a closed port before config.settings
    reads them. Returns the banner line."""
    file_values = {k: v for k, v in dotenv_values(env_file).items()
                   if v is not None} if Path(env_file).exists() else {}
    allowed, why = db_test_decision(environ, file_values)
    if not allowed:
        environ["POSTGRES_HOST"] = "127.0.0.1"
        environ["POSTGRES_PORT"] = "1"
    return why


_MESSAGE = _guard()


def pytest_report_header(config):
    return _MESSAGE


def pytest_sessionstart(session):
    # -q hides the report header; print the line anyway
    if session.config.get_verbosity() < 0:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(_MESSAGE)
