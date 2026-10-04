@echo off
rem ops\windows\setup-all.bat: sets this PC up in one go, with a message
rem before each step. The "NHL Setup" desktop shortcut runs it.
rem   1. Docker Desktop and the database (start-db.bat)
rem   2. python pipeline.py setup         checks the database, nhlpy, the Odds API key
rem   3. python -m config.migrate         adds any missing tables and columns
rem   4. python pipeline.py daily         catches up every game since the last
rem                                       run, settles finished bets, makes
rem                                       today's picks (about 3 Odds API credits)
rem   5. register-tasks.ps1 -Role %NHL_ROLE% -IncludeOdds
rem                                       the scheduled jobs, so from then on
rem                                       everything runs by itself
rem Safe to run again: every step skips what is already done.
rem
rem NHL_ROLE chooses the scheduled jobs (register-tasks.ps1 -Role):
rem   picks  (the default) daily, the midday odds run, close --due and
rem          news --due: about 321 Odds API credits a month, inside a free
rem          500-credit key
rem   all    picks plus the props jobs (props at 10:00 and props --due):
rem          more than 500 credits a month, so only with a paid key
rem Registering a role removes the jobs it leaves out. For every job, type
rem   set NHL_ROLE=all
rem in a Command Prompt window, then run this file from that same window.
title NHL Setup
setlocal
pushd "%~dp0..\.."
rem NHL_PYTHON overrides the project's Python (default: the repo's .venv)
set "PY=%CD%\.venv\Scripts\python.exe"
if defined NHL_PYTHON set "PY=%NHL_PYTHON%"
rem -- role --
if not defined NHL_ROLE set "NHL_ROLE=picks"
if /i "%NHL_ROLE%"=="picks" goto role_ok
if /i "%NHL_ROLE%"=="all" goto role_ok
echo NHL_ROLE=%NHL_ROLE% is not a role NHL Setup knows: use picks or all.
cmd /c exit 2
goto failed
:role_ok
rem -- end role --

echo ================================================================
echo  NHL betting system: full setup for this PC
echo  Repo: %CD%
echo  Scheduled jobs: %NHL_ROLE% (set NHL_ROLE=all for the props jobs too)
echo ================================================================
echo.

echo ---- Step 1 of 5: Docker Desktop and the database ----
call "%~dp0start-db.bat"
if errorlevel 1 goto failed
echo.

echo ---- Step 2 of 5: setup check (python pipeline.py setup) ----
"%PY%" pipeline.py setup
if errorlevel 1 goto failed
echo.

echo ---- Step 3 of 5: database upgrade (python -m config.migrate) ----
"%PY%" -m config.migrate
if errorlevel 1 goto failed
echo.

echo ---- Step 4 of 5: catch up and make today's picks (python pipeline.py daily) ----
echo This loads every game since the last run, settles finished bets and
echo makes today's picks. It can take several minutes and spends about 3
echo Odds API credits. Progress messages follow.
"%PY%" pipeline.py daily
if errorlevel 1 goto failed
echo.

echo ---- Step 5 of 5: scheduled jobs (register-tasks.ps1 -Role %NHL_ROLE% -IncludeOdds) ----
echo Registers the daily, midday odds, closing-line and news jobs in Task
echo Scheduler, folder \NHLBetting\, plus the props jobs when NHL_ROLE=all.
echo They run while you are logged on.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0register-tasks.ps1" -Role %NHL_ROLE% -IncludeOdds
if errorlevel 1 goto failed
echo.

echo ================================================================
echo  Setup finished. The jobs now run by themselves every day; their
echo  logs are in the logs folder. Open the dashboard with the
echo  "NHL Dashboard" desktop shortcut.
echo ================================================================
pause
popd
exit /b 0

:failed
set "RC=%ERRORLEVEL%"
echo.
echo ================================================================
echo  That step failed, exit code %RC%. Nothing after it ran.
echo  Read the messages above, fix the problem, and run NHL Setup again:
echo  the steps that already worked are safe to repeat.
echo ================================================================
pause
popd
exit /b %RC%
