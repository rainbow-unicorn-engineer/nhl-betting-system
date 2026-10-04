<#
.SYNOPSIS
    Registers the Windows Task Scheduler tasks for one machine role.
    -Role is required: all, picks or props.

.DESCRIPTION
    Each machine has its own .env, its own Odds API key and its own
    database. The owner runs every job on both machines (-Role all);
    picks and props split the jobs between two machines instead:

      -Role all     Everything (the owner's Mac and Windows PC). Registers
                    the picks tasks below plus props and props-due, and
                    not refresh, which daily already covers.

      -Role picks   Moneyline picks and closing lines only (mirrors the
                    Mac's daily, odds, close and news agents). Registers:
                      daily      python pipeline.py daily        at -DailyTime (9:00)
                      odds       python pipeline.py odds         at -OddsTime (13:00), only with -IncludeOdds
                      close      python pipeline.py close --due  every 15 minutes
                      news       python pipeline.py news --due   every 15 minutes: team news
                                                                 (starting goalies, lines,
                                                                 injuries) from 8:00 until
                                                                 the last puck drop on game
                                                                 days; free, no Odds API request

      -Role props   Player-props lines only, no picks. Registers:
                      refresh    python pipeline.py refresh      at -DailyTime (9:00): schedule,
                                                                 box scores, power-play stats and
                                                                 injuries; no odds request, no picks
                      props      python pipeline.py props        at -PropsTime (10:00)
                      props-due  python pipeline.py props --due  every 15 minutes
                    and none of the picks tasks.

    Registering a role removes any task in \NHLBetting\ that the role does
    not include (refresh when switching to all, the picks tasks when
    switching to props), so the folder always matches the role given.

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

.PARAMETER Role
    Required (except with -Unregister). "all": every job, the picks tasks
    plus props and props-due. "picks": daily, close --due and news --due
    every 15 minutes and, with -IncludeOdds, the midday odds run; moneyline
    picks, closing lines and team news. "props": refresh, a morning props snapshot and props --due every
    15 minutes; player-props lines, no picks.

.PARAMETER RepoPath
    The repo folder. Default: two levels up from this script.

.PARAMETER PythonPath
    The Python the project is installed in. Default: <RepoPath>\.venv\Scripts\python.exe

.PARAMETER IncludeOdds
    -Role all or picks: also register the optional midday `odds` run (3
    credits a game day with the default ODDS_BOOKMAKERS).

.PARAMETER DailyTime
    Local time of the daily run (daily for all and picks, refresh for
    props). Default 09:00.

.PARAMETER OddsTime
    Local time of the midday odds run (-IncludeOdds). Default 13:00.

.PARAMETER PropsTime
    Local time of the morning props snapshot (-Role all or props). Default
    10:00, after the 9:00 run has loaded the day's schedule.

.PARAMETER VisibleConsole
    Start cmd.exe directly instead of through "conhost.exe --headless", as
    older versions of this script did. For Windows older than 10 21H2: a
    console window then opens for a moment at every run, every 15 minutes
    for the close and props-due tasks.

.PARAMETER Unregister
    Remove every task this script registers (either role), then stop.

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Role all

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Role props

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Role picks -IncludeOdds

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Role props -RepoPath C:\Working\nhl-betting-system -PythonPath C:\Working\nhl-betting-system\.venv\Scripts\python.exe

.EXAMPLE
    .\ops\windows\register-tasks.ps1 -Unregister
#>
[CmdletBinding(DefaultParameterSetName = "Register")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "Register",
        HelpMessage = "all = every job; picks = moneyline picks, closes and news (daily, close --due, news --due, optional odds); props = player-props lines (refresh, props, props --due)")]
    [ValidateSet("all", "picks", "props")]
    [string]$Role,
    # Resolved below: Windows PowerShell 5.1 leaves $PSScriptRoot empty in
    # an advanced script's parameter defaults
    [string]$RepoPath = "",
    [string]$PythonPath = "",
    [Parameter(ParameterSetName = "Register")]
    [switch]$IncludeOdds,
    [string]$DailyTime = "09:00",
    [string]$OddsTime = "13:00",
    [string]$PropsTime = "10:00",
    [switch]$VisibleConsole,
    [Parameter(Mandatory = $true, ParameterSetName = "Unregister")]
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$TaskFolder = "\NHLBetting\"
$RoleTasks = @{
    "picks" = @("daily", "odds", "close", "news")
    "props" = @("refresh", "props", "props-due")
    # refresh is a subset of daily, so all never schedules both
    "all"   = @("daily", "odds", "close", "news", "props", "props-due")
}
$AllTasks = $RoleTasks["picks"] + $RoleTasks["props"]
$DoesPicks = $Role -in @("all", "picks")
$DoesProps = $Role -in @("all", "props")

function Remove-PipelineTask([string]$Name) {
    $task = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $Name -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskPath $TaskFolder -TaskName $Name -Confirm:$false
        Write-Host "Removed $TaskFolder$Name"
    }
}

if ($Unregister) {
    foreach ($name in $AllTasks) {
        Remove-PipelineTask $name
    }
    return
}

if ($IncludeOdds -and -not $DoesPicks) {
    throw "-IncludeOdds adds the midday moneyline odds run, which belongs to -Role all or picks. The props role makes no picks."
}

if (-not $RepoPath) {
    $RepoPath = Join-Path $PSScriptRoot "..\.."
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

function New-QuarterHourTrigger {
    # Every 15 minutes, all day, every day
    $trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
    $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" `
        -RepetitionInterval (New-TimeSpan -Minutes 15) `
        -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
    return $trigger
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

function Register-PipelineTask([string]$Name, $Trigger, $Action, [string]$What) {
    Register-ScheduledTask -TaskPath $TaskFolder -TaskName $Name `
        -Action $Action -Trigger $Trigger -Settings $settings `
        -Description "nhl-betting-system ($Role role): $What" -Force | Out-Null
    Write-Host "Registered $TaskFolder$Name ($What)"
}

# The folder matches the role given: remove every task the role leaves out
# (the optional odds task is handled with the picks tasks below)
foreach ($name in $AllTasks) {
    if ($RoleTasks[$Role] -notcontains $name) {
        Remove-PipelineTask $name
    }
}

if ($DoesPicks) {
    # daily: refresh, full snapshot (3 credits), free NHL feed, picks
    Register-PipelineTask "daily" (New-ScheduledTaskTrigger -Daily -At $DailyTime) `
        (New-PipelineAction "daily" "daily") "python pipeline.py daily at $DailyTime"

    # odds (optional): confirmed starters, picks for games without one, alerts
    if ($IncludeOdds) {
        Register-PipelineTask "odds" (New-ScheduledTaskTrigger -Daily -At $OddsTime) `
            (New-PipelineAction "odds" "odds") "python pipeline.py odds at $OddsTime"
    } elseif (Get-ScheduledTask -TaskPath $TaskFolder -TaskName "odds" -ErrorAction SilentlyContinue) {
        Write-Host "Left the existing $($TaskFolder)odds task alone (-IncludeOdds not given; -Unregister removes every task)"
    }

    # close --due every 15 minutes, all day: it snapshots (1 credit) only
    # when a game starts within 16 minutes and no moneyline snapshot is
    # under 16 minutes old, so each start time gets one close, in the last
    # run before puck drop, and most runs just log a line and exit.
    Register-PipelineTask "close" (New-QuarterHourTrigger) `
        (New-PipelineAction "close --due" "close") `
        "python pipeline.py close --due every 15 minutes"

    # news --due every 15 minutes, all day: on a game day from 8:00 local
    # until the last puck drop it refreshes Daily Faceoff's starters and
    # lines and ESPN's injury list, records what changed, and re-scores a
    # game whose starter changed (no Odds API request, so no credits);
    # outside that window a run logs one line and exits
    Register-PipelineTask "news" (New-QuarterHourTrigger) `
        (New-PipelineAction "news --due" "news") `
        "python pipeline.py news --due every 15 minutes"
} else {
    # refresh: schedule and box scores (props are matched to raw.games),
    # power-play stats and the ESPN injury list. Free: no Odds API request
    Register-PipelineTask "refresh" (New-ScheduledTaskTrigger -Daily -At $DailyTime) `
        (New-PipelineAction "refresh" "refresh") "python pipeline.py refresh at $DailyTime"
}

if ($DoesProps) {
    # props: one snapshot of every game starting in the next 24 hours, 1
    # credit a game per market returned (nothing for a game without props yet)
    Register-PipelineTask "props" (New-ScheduledTaskTrigger -Daily -At $PropsTime) `
        (New-PipelineAction "props" "props") "python pipeline.py props at $PropsTime"

    # props --due every 15 minutes: one pre-game snapshot per game, 1 to 16
    # minutes before its puck drop; most runs just log a line and exit
    Register-PipelineTask "props-due" (New-QuarterHourTrigger) `
        (New-PipelineAction "props --due" "props-due") `
        "python pipeline.py props --due every 15 minutes"
}

Write-Host ""
if ($VisibleConsole) {
    Write-Host "The tasks start cmd.exe directly: a console window opens at each run."
} else {
    Write-Host "The tasks run hidden (conhost.exe --headless). If their logs stay empty,"
    Write-Host "this Windows may be too old for that: run the script again with -VisibleConsole."
}
Write-Host "Logs: $LogDir\<task>.log. Check the tasks with:"
Write-Host "  Get-ScheduledTask -TaskPath '$TaskFolder' | Get-ScheduledTaskInfo"
