<#
.SYNOPSIS
    Registers Windows Task Scheduler tasks that mirror ops/launchd/:
    daily at 9:00, the optional midday odds run at 13:00, and
    `close --due` every 15 minutes.

.DESCRIPTION
    Each task runs "<PythonPath> pipeline.py <command>" with the repo as its
    working folder (so .env and DATA_DIR resolve as they do by hand) and
    appends its output to logs\<task>.log in the repo (git ignores logs\).
    Tasks go in the Task Scheduler folder \NHLBetting\ and run as you, only
    while you are logged on (Docker Desktop needs that too), so no password
    is stored. A run missed while the PC was off or asleep starts as soon as
    it can, and a run never overlaps the previous one.

    Each run is started through "conhost.exe --headless", so it opens no
    console window and never takes the focus from what you are doing. That
    needs Windows 10 version 21H2 or later, or Windows 11. On older Windows,
    pass -VisibleConsole.

    Nothing runs this script automatically. Run it again to update the
    tasks; -Unregister removes them.

.PARAMETER RepoPath
    The repo folder. Default: two levels up from this script.

.PARAMETER PythonPath
    The Python the project is installed in. Default: <RepoPath>\.venv\Scripts\python.exe

.PARAMETER IncludeOdds
    Also register the optional midday `odds` run (6 credits a game day).

.PARAMETER DailyTime
    Local time of the daily run. Default 09:00.

.PARAMETER OddsTime
    Local time of the midday odds run. Default 13:00.

.PARAMETER VisibleConsole
    Start cmd.exe directly instead of through "conhost.exe --headless", as
    older versions of this script did. For Windows older than 10 21H2: a
    console window then opens for a moment at every run, every 15 minutes
    for the close task.

.PARAMETER Unregister
    Remove the tasks this script registers, then stop.

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -IncludeOdds

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -RepoPath C:\Working\nhl-betting-system -PythonPath C:\Working\nhl-betting-system\.venv\Scripts\python.exe

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string]$RepoPath = (Join-Path $PSScriptRoot "..\.."),
    [string]$PythonPath = "",
    [switch]$IncludeOdds,
    [string]$DailyTime = "09:00",
    [string]$OddsTime = "13:00",
    [switch]$VisibleConsole,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$TaskFolder = "\NHLBetting\"
$TaskNames = @("daily", "odds", "close")

if ($Unregister) {
    foreach ($name in $TaskNames) {
        $task = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $name -ErrorAction SilentlyContinue
        if ($task) {
            Unregister-ScheduledTask -TaskPath $TaskFolder -TaskName $name -Confirm:$false
            Write-Host "Removed $TaskFolder$name"
        }
    }
    return
}

$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
if (-not $PythonPath) {
    $PythonPath = Join-Path $RepoPath ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath (Join-Path $RepoPath "pipeline.py"))) {
    throw "No pipeline.py in '$RepoPath'. Pass -RepoPath <the repo folder>."
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python not found at '$PythonPath'. Pass -PythonPath <the venv python.exe>."
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
# cmd.exe reads these characters as commands or variables even inside a
# path, so a task would silently run something else. In the headless form
# conhost passes a path without spaces to cmd.exe unquoted, and cmd.exe
# also splits an unquoted path at ( ) , ; and =, so those are refused too
# unless -VisibleConsole (which quotes every path) is used.
$BadChars = if ($VisibleConsole) { '[&|<>^%"]' } else { '[&|<>^%"(),;=]' }
foreach ($path in @($RepoPath, $PythonPath)) {
    if ($path -match $BadChars) {
        throw "'$path' contains a character cmd.exe would misread (& | < > ^ % `" and, for hidden tasks, ( ) , ; =). Move the repo or the venv to a plain path, or use -VisibleConsole."
    }
}
$LogDir = Join-Path $RepoPath "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$CmdExe = Join-Path $env:SystemRoot "System32\cmd.exe"
$ConhostExe = Join-Path $env:SystemRoot "System32\conhost.exe"
if (-not $VisibleConsole -and -not (Test-Path -LiteralPath $ConhostExe)) {
    throw "No conhost.exe at '$ConhostExe'. Run again with -VisibleConsole."
}

function New-PipelineAction([string]$Command, [string]$LogName) {
    # cmd.exe does the appending (>>), because Task Scheduler cannot
    # redirect output itself. PYTHONUTF8=1 keeps log lines that hold
    # non-ASCII characters (an em dash, an accented team name) from failing
    # to encode; with no space before &&, its value is exactly 1.
    #
    # conhost.exe --headless runs cmd.exe with no console window. conhost
    # splits the rest of its command line into arguments and joins them
    # again, quoting any argument that holds a space and escaping quotes
    # inside one with a backslash, which cmd.exe doesn't understand. So the
    # command holds no quotes of its own beyond the two paths, which
    # conhost puts back in quotes when they hold a space.
    $log = Join-Path $LogDir "$LogName.log"
    $inner = "set PYTHONUTF8=1&& `"$PythonPath`" pipeline.py $Command >> `"$log`" 2>&1"
    if ($VisibleConsole) {
        # /s: cmd.exe strips only the outer pair of quotes
        New-ScheduledTaskAction -Execute $CmdExe `
            -Argument "/d /s /c `"$inner`"" -WorkingDirectory $RepoPath
    } else {
        New-ScheduledTaskAction -Execute $ConhostExe `
            -Argument "--headless `"$CmdExe`" /d /s /c $inner" -WorkingDirectory $RepoPath
    }
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

function Register-PipelineTask([string]$Name, $Trigger, $Action, [string]$What) {
    Register-ScheduledTask -TaskPath $TaskFolder -TaskName $Name `
        -Action $Action -Trigger $Trigger -Settings $settings `
        -Description "nhl-betting-system: $What" -Force | Out-Null
    Write-Host "Registered $TaskFolder$Name ($What)"
}

# daily: refresh, full snapshot (6 credits), picks
Register-PipelineTask "daily" (New-ScheduledTaskTrigger -Daily -At $DailyTime) `
    (New-PipelineAction "daily" "daily") "python pipeline.py daily at $DailyTime"

# odds (optional): confirmed starters, picks for games without one, alerts
if ($IncludeOdds) {
    Register-PipelineTask "odds" (New-ScheduledTaskTrigger -Daily -At $OddsTime) `
        (New-PipelineAction "odds" "odds") "python pipeline.py odds at $OddsTime"
} elseif (Get-ScheduledTask -TaskPath $TaskFolder -TaskName "odds" -ErrorAction SilentlyContinue) {
    Write-Host "Left the existing $($TaskFolder)odds task alone (-IncludeOdds not given; -Unregister removes all three)"
}

# close --due every 15 minutes, all day: it snapshots (2 credits) only when
# a game starts within 16 minutes and no moneyline snapshot is under 16
# minutes old, so each start time gets one close, in the last run before
# puck drop, and most runs just log a line and exit.
$closeTrigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$closeTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
Register-PipelineTask "close" $closeTrigger (New-PipelineAction "close --due" "close") `
    "python pipeline.py close --due every 15 minutes"

Write-Host ""
if ($VisibleConsole) {
    Write-Host "The tasks start cmd.exe directly: a console window opens at each run."
} else {
    Write-Host "The tasks run hidden (conhost.exe --headless). If their logs stay empty,"
    Write-Host "this Windows may be too old for that: run the script again with -VisibleConsole."
}
Write-Host "Logs: $LogDir\<task>.log. Check the tasks with:"
Write-Host "  Get-ScheduledTask -TaskPath '$TaskFolder' | Get-ScheduledTaskInfo"
