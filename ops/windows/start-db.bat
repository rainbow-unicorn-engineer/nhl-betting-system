@echo off
rem ops\windows\start-db.bat: makes sure Docker Desktop is running and the
rem database answers. open-dashboard.bat and setup-all.bat call it; it can
rem also be run on its own. Exit code 0 = the database is ready.
rem Docker Desktop -> the app that runs the database's container (a small
rem self-contained Linux box) on Windows.
setlocal
pushd "%~dp0..\.."
rem NHL_PYTHON overrides the project's Python (default: the repo's .venv)
set "PY=%CD%\.venv\Scripts\python.exe"
if defined NHL_PYTHON set "PY=%NHL_PYTHON%"
if not exist "%PY%" (
    echo [start-db] No Python environment at "%PY%".
    echo [start-db] Install the project first: see "Quick start" in README.md.
    popd & exit /b 1
)

docker info >nul 2>&1
if not errorlevel 1 goto docker_ready

echo [start-db] Docker Desktop is not running. Starting it...
set "DD="
rem Docker Desktop installs for all users or for one user
set "DD1=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
set "DD2=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
for %%P in ("%DD1%" "%DD2%") do (
    if not defined DD if exist "%%~P" set "DD=%%~P"
)
if defined DD (
    start "" "%DD%"
) else (
    rem Newer Docker Desktop versions can be started from the command line
    docker desktop start >nul 2>&1
)

rem Wait up to 3 minutes for the Docker engine
set /a TRIES=0
:wait_docker
set /a TRIES+=1
if %TRIES% gtr 60 (
    echo [start-db] Docker did not start within 3 minutes. Open Docker Desktop
    echo [start-db] from the Start menu, wait until it says "Engine running",
    echo [start-db] then try again.
    popd & exit /b 1
)
timeout /t 3 /nobreak >nul
docker info >nul 2>&1
if errorlevel 1 goto wait_docker

:docker_ready
echo [start-db] Docker is running.

rem The container normally starts with Docker (restart: unless-stopped).
rem Start it if it is stopped, or create it from docker-compose.yml.
docker start nhl_betting_db >nul 2>&1
if errorlevel 1 (
    echo [start-db] Creating the database container with docker compose...
    docker compose up -d
    if errorlevel 1 (
        echo [start-db] docker compose could not start the database.
        popd & exit /b 1
    )
)

rem Wait up to 2 minutes for PostgreSQL to accept a connection with the
rem settings in .env
set /a TRIES=0
:wait_db
set /a TRIES+=1
"%PY%" -c "from config.settings import engine; engine.connect().close()" >nul 2>&1
if not errorlevel 1 goto db_ready
if %TRIES% gtr 60 (
    echo [start-db] The database did not answer within 2 minutes. Check the
    echo [start-db] POSTGRES_* settings in .env and "docker ps".
    popd & exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_db

:db_ready
echo [start-db] The database is ready.
popd
exit /b 0
