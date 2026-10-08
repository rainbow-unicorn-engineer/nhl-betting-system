"""
Tests for the Windows convenience scripts in ops/windows/: static checks
that run anywhere (line endings, the steps each script runs, the files the
shortcuts point at), and on Windows a run of the PowerShell scripts in
Windows PowerShell 5.1, the version every Windows 11 PC has, that changes
nothing: shortcuts go to a temporary folder, and register-tasks.ps1 is
given a Python that doesn't exist, so it stops before registering a task.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WIN = Path(__file__).resolve().parent.parent / "ops" / "windows"
on_windows = pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell"),
                                reason="needs Windows PowerShell")


def _ps(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(WIN / script), *args],
                          capture_output=True, text=True, timeout=120)


@on_windows
def test_register_tasks_finds_the_repo_in_windows_powershell(tmp_path):
    """Windows PowerShell 5.1 leaves $PSScriptRoot empty in an advanced
    script's parameter defaults; the repo must still be found."""
    missing = tmp_path / "no-python.exe"
    out = _ps("register-tasks.ps1", "-Role", "all", "-IncludeOdds",
              "-PythonPath", str(missing))
    err = out.stdout + out.stderr
    assert out.returncode != 0
    assert "Python not found" in err, err
    assert "empty string" not in err


BATS = ("open-dashboard.bat", "setup-all.bat", "start-db.bat")


@pytest.mark.parametrize("name", BATS)
def test_batch_files_use_windows_line_endings(name):
    """cmd.exe can misread labels (goto) in a batch file with bare LF
    line endings; .gitattributes keeps them CRLF on every checkout."""
    data = (WIN / name).read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")


def test_setup_all_runs_the_steps_in_order():
    text = (WIN / "setup-all.bat").read_text()
    steps = ['call "%~dp0start-db.bat"', '"%PY%" pipeline.py setup',
             '"%PY%" -m config.migrate', '"%PY%" pipeline.py daily',
             'register-tasks.ps1" -Role %NHL_ROLE% -IncludeOdds']
    at = [text.index(s) for s in steps]
    assert at == sorted(at)
    # every step stops the script on failure
    assert text.count("if errorlevel 1 goto failed") == len(steps)
    # the role is settled before anything runs
    assert text.index(":role_ok") < at[0]
    assert "-Role all" not in text


def _role_block() -> str:
    text = (WIN / "setup-all.bat").read_text()
    return text[text.index("rem -- role --"):text.index("rem -- end role --")]


def _run_role_block(tmp_path, role=None, dotenv=None):
    """setup-all.bat's role lines on their own, in cmd.exe, run in tmp_path
    (so the repo's real .env never leaks in; dotenv writes a test .env
    there): the role it would pass to register-tasks.ps1, or the exit code
    it fails with."""
    import os
    bat = tmp_path / "role.bat"
    lines = ["@echo off", "setlocal", *_role_block().splitlines(), "echo ROLE=%NHL_ROLE%",
             "exit /b 0", ":failed", "exit /b %ERRORLEVEL%", ""]
    bat.write_bytes("\r\n".join(lines).encode())
    if dotenv is not None:
        (tmp_path / ".env").write_bytes(dotenv.replace("\n", "\r\n").encode())
    env = {k: v for k, v in os.environ.items() if k.upper() != "NHL_ROLE"}
    if role is not None:
        env["NHL_ROLE"] = role
    return subprocess.run(["cmd", "/d", "/c", str(bat)], capture_output=True, text=True,
                          env=env, timeout=60, cwd=tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="needs cmd.exe")
def test_setup_all_registers_the_free_key_role_unless_told_otherwise(tmp_path):
    """On a free 500-credit key the PC runs the picks jobs only; the props
    jobs (more than 500 credits a month with the picks jobs) need NHL_ROLE=all."""
    out = _run_role_block(tmp_path)
    assert out.returncode == 0 and "ROLE=picks" in out.stdout
    out = _run_role_block(tmp_path, "all")
    assert out.returncode == 0 and "ROLE=all" in out.stdout
    out = _run_role_block(tmp_path, "props")      # -IncludeOdds would refuse it
    assert out.returncode == 2 and "use picks or all" in out.stdout
    assert not any(line.startswith("ROLE=") for line in out.stdout.splitlines())


@pytest.mark.skipif(sys.platform != "win32", reason="needs cmd.exe")
def test_setup_all_reads_the_role_from_dotenv(tmp_path):
    """A paid-key machine sets NHL_ROLE=all in .env; the shortcut honours it,
    and a variable set in the window still wins over .env."""
    out = _run_role_block(tmp_path, dotenv="POSTGRES_DB=x\nNHL_ROLE=all\nOTHER=1\n")
    assert out.returncode == 0 and "ROLE=all" in out.stdout
    out = _run_role_block(tmp_path, "picks", dotenv="NHL_ROLE=all\n")
    assert out.returncode == 0 and "ROLE=picks" in out.stdout
    out = _run_role_block(tmp_path, dotenv="POSTGRES_DB=x\n")
    assert out.returncode == 0 and "ROLE=picks" in out.stdout


def test_open_dashboard_waits_for_the_database_and_stays_local():
    text = (WIN / "open-dashboard.bat").read_text()
    assert 'call "%~dp0start-db.bat"' in text
    assert "-m streamlit run dashboard" + chr(92) + "app.py" in text
    assert "--server.address localhost" in text      # not the whole network
    assert text.index("start-db.bat") < text.index("streamlit run")


def test_shortcut_targets_exist():
    text = (WIN / "create-shortcuts.ps1").read_text()
    for name in ("open-dashboard.bat", "setup-all.bat", "nhl-dashboard.ico",
                 "nhl-setup.ico"):
        assert f"'{name}'" in text
        assert (WIN / name).exists()


@on_windows
def test_create_shortcuts(tmp_path):
    out = _ps("create-shortcuts.ps1", "-Desktop", str(tmp_path))
    assert out.returncode == 0, out.stdout + out.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["NHL Dashboard.lnk",
                                                           "NHL Setup.lnk"]
    # read one back through the Windows shell
    check = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(New-Object -ComObject WScript.Shell).CreateShortcut('{tmp_path / 'NHL Dashboard.lnk'}')"
         ".TargetPath"], capture_output=True, text=True, timeout=60)
    assert check.stdout.strip().lower() == str(WIN / "open-dashboard.bat").lower()


def test_news_task_is_scheduled_every_15_minutes_for_picks_and_all():
    """register-tasks.ps1: the news monitor runs as `news --due` on the
    quarter-hour trigger, in the picks and all roles, never in props."""
    import re
    text = (WIN / "register-tasks.ps1").read_text(encoding="utf-8")
    roles = dict(re.findall(r'"(\w+)"\s*=\s*@\(([^)]*)\)', text))
    assert '"news"' in roles["picks"] and '"news"' in roles["all"]
    assert '"news"' not in roles["props"]
    m = re.search(r'Register-PipelineTask "news" \(New-QuarterHourTrigger\) `\s*'
                  r'\(New-PipelineAction "news --due" "news"\)', text)
    assert m, "news task not registered on the 15-minute trigger"
    # registered inside the picks block, before the props-only else branch
    assert text.index('Register-PipelineTask "news"') < text.index('Register-PipelineTask "refresh"')
