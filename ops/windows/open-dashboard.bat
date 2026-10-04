@echo off
rem ops\windows\open-dashboard.bat: opens the NHL dashboard in your browser.
rem The "NHL Dashboard" desktop shortcut (create-shortcuts.ps1) runs this.
rem   1. If the dashboard is already running, it just opens the browser.
rem   2. Otherwise it starts Docker Desktop if needed and waits for the
rem      database (start-db.bat), then starts the dashboard with Streamlit
rem      and opens http://localhost:8501 once it answers.
rem Keep the window open while you use the dashboard; closing it stops it.
title NHL Dashboard
setlocal
pushd "%~dp0..\.."
set "PORT=8501"
set "URL=http://localhost:%PORT%"
rem NHL_PYTHON overrides the project's Python (default: the repo's .venv)
set "PY=%CD%\.venv\Scripts\python.exe"
if defined NHL_PYTHON set "PY=%NHL_PYTHON%"

netstat -ano -p tcp | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo The dashboard is already running. Opening %URL% ...
    start "" "%URL%"
    timeout /t 3 /nobreak >nul
    popd & exit /b 0
)

call "%~dp0start-db.bat"
if errorlevel 1 (
    echo.
    echo The dashboard could not start because the database is not ready.
    echo Read the messages above, then try again.
    pause
    popd & exit /b 1
)

echo.
echo Starting the dashboard at %URL% ...
echo Keep this window open while you use it. Close it to stop the dashboard.
echo.
rem Open the browser in the background as soon as the server answers
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 90; $i++) { try { (New-Object Net.Sockets.TcpClient('127.0.0.1', %PORT%)).Close(); Start-Process '%URL%'; break } catch { Start-Sleep -Seconds 1 } }"
rem localhost only: other computers on the network can't open it
"%PY%" -m streamlit run dashboard\app.py --server.port %PORT% --server.address localhost --server.headless true --browser.gatherUsageStats false
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The dashboard stopped with an error, code %RC%. Read the messages above.
    pause
)
popd
exit /b %RC%
