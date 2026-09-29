# Windows Task Scheduler jobs

The Windows counterpart of [ops/launchd/](../launchd/). Task Scheduler is the job scheduler built into Windows. `register-tasks.ps1` registers these tasks in a Task Scheduler folder named `\NHLBetting\`:

| Task | Runs | When | Credits a run |
|---|---|---|---|
| `daily` | `python pipeline.py daily` | 9:00 | 6 |
| `odds` (optional, `-IncludeOdds`) | `python pipeline.py odds` | 13:00 | 6 |
| `close` | `python pipeline.py close --due` | every 15 minutes | 2 when a game is about to start, otherwise 0 |

Times are this PC's local time. Every snapshot run makes no Odds API request, and costs nothing, when no game starts in the next 24 hours. `close --due` takes its 2-credit snapshot only when a game starts within 16 minutes and no moneyline snapshot is less than 16 minutes old, so each start time gets one close, in the last run before puck drop; the other runs log one line and exit. Those two limits are defaults, set by `CLOSE_LEAD_MINUTES` and `CLOSE_MIN_GAP_MINUTES` in `.env`.

**Credits:** at the defaults, the closes average about 8 credits a game day, so the daily run plus the close job comes to at most about 456 credits in any month of the 2026-27 schedule, under the free plan's 500. Adding the midday `odds` run goes over. [Snapshot schedule](../../README.md#snapshot-schedule) in the main README has the numbers.

Nothing runs the script automatically, and it changes nothing on the Mac.

## Register the tasks

From the repo folder in PowerShell:

```powershell
.\ops\windows\register-tasks.ps1 -IncludeOdds      # leave out -IncludeOdds to skip the midday run
```

If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (the same fix as for `Activate.ps1`), or run it as `powershell -ExecutionPolicy Bypass -File .\ops\windows\register-tasks.ps1`.

| Parameter | Default | Meaning |
|---|---|---|
| `-RepoPath` | the repo the script is in | The repo folder |
| `-PythonPath` | `<RepoPath>\.venv\Scripts\python.exe` | The Python the project is installed in. Calling it directly means the task doesn't need to activate the virtual environment |
| `-IncludeOdds` | off | Also register the midday `odds` task |
| `-DailyTime`, `-OddsTime` | `09:00`, `13:00` | Local run times |
| `-VisibleConsole` | off | Start `cmd.exe` directly instead of hidden, for Windows older than 10 21H2 (see [No console windows](#no-console-windows)) |
| `-Unregister` | | Remove all three tasks |

Running the script again replaces the tasks, so that is also how to change a time.

## What the tasks do

- Each task starts in the repo folder, so `.env` is found and a relative `DATA_DIR` resolves to `<repo>\data`.
- Output is appended to `logs\daily.log`, `logs\odds.log` and `logs\close.log` in the repo. The script creates `logs\`, and git ignores it.
- The tasks run as you, only while you are logged on, so no password is stored and no administrator rights are needed. Docker Desktop must be running too, or every run fails at the database check.
- The tasks run hidden: no console window opens, and nothing takes the focus from what you are doing. See the next section.
- A run missed while the PC was off or asleep starts as soon as the PC is back. The pipeline then waits up to 3 minutes for the network, and a late close skips games already under way.
- A task never starts a second copy while one is still running.

## No console windows

A task that runs as the logged-on user and starts `cmd.exe` opens a console window, which would pop up and take the focus every 15 minutes. So each task starts `conhost.exe --headless`, the Windows console host with no window, which then runs `cmd.exe /d /s /c ...` as before. The tasks keep the interactive logon: no administrator rights, no stored password.

- **It needs Windows 10 version 21H2 or later, or Windows 11.** Check with `winver`.
- **Older Windows:** run the script again with `-VisibleConsole`. The tasks then start `cmd.exe` directly, as earlier versions of the script did, and a console window flashes up at each run. If the tasks run but their logs stay empty, this Windows is probably too old for `--headless`, so do the same. To have no windows on older Windows, open each task in Task Scheduler and choose **Run whether user is logged on or not**. Windows then stores your password with the task, and that option can need administrator rights. Runs while you are logged off still fail, because Docker Desktop runs only while you are logged on.
- **`LastTaskResult` reads 0 even when a run fails**, because it reports how `conhost.exe` exited, not the pipeline. Check the log instead. With `-VisibleConsole` it shows the pipeline's own exit code again.
- The repo and Python paths must not contain `&`, `|`, `<`, `>`, `^`, `%`, or quotes, because `cmd.exe` would read them as part of a command. For the hidden (default) tasks they also must not contain `(`, `)`, `,`, `;` or `=`, because a path without spaces reaches `cmd.exe` unquoted and it splits there. The script refuses such a path. Spaces are fine.

## Check, run by hand, remove

```powershell
Get-ScheduledTask -TaskPath '\NHLBetting\' | Get-ScheduledTaskInfo   # hidden tasks show 0 even on failure: read the log
Start-ScheduledTask -TaskPath '\NHLBetting\' -TaskName close        # run one now
Get-Content logs\close.log -Tail 20
Select-String -Path logs\*.log -Pattern "ERROR" | Select-Object -Last 5
Select-String -Path logs\*.log -Pattern "Credits remaining" | Select-Object -Last 3
.\ops\windows\register-tasks.ps1 -Unregister                        # remove them
```

To pause for the off-season, `Get-ScheduledTask -TaskPath '\NHLBetting\' | Disable-ScheduledTask`, and `Enable-ScheduledTask` to resume. Leaving them on costs no credits, but the logs keep growing.

## Two machines, two Odds API keys

Each machine reads its own `.env`, so each can use its own Odds API key, and each key has its own 500 free credits a month. If the Mac and the PC share one key, their snapshots draw on the same 500 credits, and the schedule above uses most of that on one machine alone. Each machine also has its own database, so picks, paper bets and the CLV ledger are kept separately on each.
