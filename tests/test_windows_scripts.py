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
