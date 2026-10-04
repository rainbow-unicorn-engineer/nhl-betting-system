# Windows: desktop shortcuts, one-step setup, scheduled jobs

## Desktop shortcuts

Run this once from the repo folder in PowerShell:

```powershell
.\ops\windows\create-shortcuts.ps1
```

It puts two shortcuts on the desktop (running it again replaces them, for example after moving the repo):

| Shortcut | Runs | What it does |
|---|---|---|
| **NHL Dashboard** | `open-dashboard.bat` | Starts Docker Desktop if it isn't running, waits for the database, starts the dashboard and opens http://localhost:8501 in your browser. If the dashboard is already running, it just opens the browser. Keep its window open while you use the dashboard; closing the window stops it. The dashboard is reachable from this PC only, not from other computers on the network |
| **NHL Setup** | `setup-all.bat` | The whole setup in one go, below |

Docker Desktop → the app that runs the database's container (a small self-contained Linux box) on Windows. `start-db.bat` is the shared first step of both: it starts Docker Desktop (installed for all users or just for you), starts the `nhl_betting_db` container if it is stopped (or creates it with `docker compose up -d`), and waits until the database accepts a connection with the settings in `.env`. Each waits at most a few minutes and says what to check if it gives up. To use a Python other than the repo's `.venv`, set `NHL_PYTHON` to its `python.exe` first.

## One-step setup (NHL Setup)

`setup-all.bat` prints a heading before each step and stops at the first one that fails, saying which:

1. Docker Desktop and the database (`start-db.bat`).
2. `python pipeline.py setup`: checks the database, nhlpy and the Odds API key, and seeds the arena locations on a new database.
3. `python -m config.migrate`: adds any tables and columns the database is missing.
4. `python pipeline.py daily`: catches up every game since the last run, settles finished bets, and makes today's picks. It can take several minutes and spends about 3 Odds API credits.
5. `register-tasks.ps1 -Role all -IncludeOdds`: registers the scheduled jobs below, so from then on everything runs by itself while you are logged on.

Every step is safe to repeat, so after fixing a problem just run NHL Setup again. Nothing runs it automatically.

# Windows Task Scheduler jobs

Task Scheduler is the job scheduler built into Windows. `register-tasks.ps1` registers the tasks for one **machine role** in a Task Scheduler folder named `\NHLBetting\`. `-Role` is required.

Each machine has its own `.env`, its own Odds API key and its own database. **The owner's setup (2026-10-01): both the Mac and this Windows PC run every job, `-Role all`** (on the Mac, every template in [ops/launchd/](../launchd/)). `-Role picks` and `-Role props` split the jobs between two machines instead.

**`-Role all`** (the Windows PC): the `-Role picks` tasks below plus `props` and `props-due` from `-Role props`. It does not register `refresh`, because `daily` already does everything `refresh` does.

| Task | Runs | When |
|---|---|---|
| `daily` | `python pipeline.py daily` | 9:00 |
| `odds` (optional, `-IncludeOdds`) | `python pipeline.py odds` | 13:00 |
| `close` | `python pipeline.py close --due` | every 15 minutes |
| `props` | `python pipeline.py props` | 10:00 |
| `props-due` | `python pipeline.py props --due` | every 15 minutes |

**Credits for `-Role all`:** the picks tasks' credits plus the props tasks' credits, both given below. That is more than the free plan's 500 a month, so a machine running every job needs a paid key (the 20K plan covers it).

**`-Role props`** (props only, no picks):

| Task | Runs | When | Credits a run |
|---|---|---|---|
| `refresh` | `python pipeline.py refresh` | 9:00 | 0: schedule and box scores, power-play stats, the ESPN injury list. No odds request, no picks |
| `props` | `python pipeline.py props` | 10:00 | 1 per game starting in the next 24 hours, per market returned (0 for a game with no props posted yet) |
| `props-due` | `python pipeline.py props --due` | every 15 minutes | 1 per game per market when a game is about to start, otherwise 0 |

`props --due` requests only the games that start within 16 minutes and have no prop snapshot less than 16 minutes old (`PROPS_CLOSE_LEAD_MINUTES` and `PROPS_CLOSE_MIN_GAP_MINUTES` in `.env`), so each game gets one pre-game snapshot, 1 to 16 minutes before its puck drop. The props tasks need the `refresh` task, because every props line is tied to a game in `raw.games`.

**Credits for props:** a morning and a pre-game snapshot is 2 credits a game for each market. With the default single market (`PROPS_MARKETS=player_shots_on_goal`) that is at most 308 (February) to 464 (January) credits a month on the 2026-27 schedule, under the free plan's 500, narrowly in January. Four markets need the paid 20K plan. A game where no book has posted props yet costs nothing.

**`-Role picks`** (moneyline picks and closing lines only, the same jobs as the Mac's daily, odds and close agents):

| Task | Runs | When | Credits a run |
|---|---|---|---|
| `daily` | `python pipeline.py daily` | 9:00 | 3 |
| `odds` (optional, `-IncludeOdds`) | `python pipeline.py odds` | 13:00 | 3 |
| `close` | `python pipeline.py close --due` | every 15 minutes | 1 when a game is about to start, otherwise 0 |

The credits are for the default `ODDS_BOOKMAKERS` (10 named books bill as one region). `close --due` takes its 1-credit snapshot only when a game starts within 16 minutes and no moneyline snapshot is less than 16 minutes old (`CLOSE_LEAD_MINUTES` and `CLOSE_MIN_GAP_MINUTES`), so each start time gets one close, in the last run before puck drop; the other runs log one line and exit. At the defaults the daily run plus the closes come to at most about 228 credits in any month of the 2026-27 schedule, and the midday `odds` run (about 93 more) fits too. [Snapshot schedule](../../README.md#snapshot-schedule) in the main README has the numbers. The picks chains also take a free NHL-feed snapshot after each Odds API snapshot.

Times are this PC's local time. Every snapshot run makes no Odds API request, and costs nothing, when no game starts in the next 24 hours. Registering a role removes any `\NHLBetting\` task that role leaves out (for example `refresh` when switching to `all`, or the picks tasks when switching to `props`), so the folder always matches the role given. Nothing runs the script automatically, and it changes nothing on the Mac.

## Register the tasks

From the repo folder in PowerShell:

```powershell
.\ops\windows\register-tasks.ps1 -Role all                    # every job (the owner's setup); add -IncludeOdds for the midday run
.\ops\windows\register-tasks.ps1 -Role props                  # props only
.\ops\windows\register-tasks.ps1 -Role picks -IncludeOdds     # picks only; leave out -IncludeOdds to skip the midday run
```

Without `-Role`, PowerShell asks for it (type `!?` at the prompt for help), and a non-interactive run stops with "missing mandatory parameters: Role". If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (the same fix as for `Activate.ps1`), or run it as `powershell -ExecutionPolicy Bypass -File .\ops\windows\register-tasks.ps1 -Role all`.

| Parameter | Default | Meaning |
|---|---|---|
| `-Role` | required | `all`: `daily`, `close`, `props`, `props-due`, and `odds` with `-IncludeOdds`. `props`: `refresh`, `props`, `props-due`. `picks`: `daily`, `close`, and `odds` with `-IncludeOdds` |
| `-RepoPath` | the repo the script is in | The repo folder |
| `-PythonPath` | `<RepoPath>\.venv\Scripts\python.exe` | The Python the project is installed in. Calling it directly means the task doesn't need to activate the virtual environment |
| `-IncludeOdds` | off | `-Role all` or `picks`: also register the midday `odds` task |
| `-DailyTime` | `09:00` | Local time of `daily` (all, picks) or `refresh` (props) |
| `-OddsTime` | `13:00` | Local time of the midday `odds` run |
| `-PropsTime` | `10:00` | Local time of the morning `props` snapshot, after the 9:00 run has loaded the day's schedule |
| `-VisibleConsole` | off | Start `cmd.exe` directly instead of hidden, for Windows older than 10 21H2 (see [No console windows](#no-console-windows)) |
| `-Unregister` | | Remove every task of either role (no `-Role` needed) |

Running the script again replaces the tasks, so that is also how to change a time.

## What the tasks do

- Each task starts in the repo folder, so `.env` is found and a relative `DATA_DIR` resolves to `<repo>\data`.
- Output is appended to `logs\<task>.log` in the repo, named after the task: `daily.log`, `odds.log`, `close.log`, `props.log`, `props-due.log` and `refresh.log`. The script creates `logs\`, and git ignores it.
- The tasks run as you, only while you are logged on, so no password is stored and no administrator rights are needed. Docker Desktop must be running too, or every run fails at the database check.
- The tasks run hidden: no console window opens, and nothing takes the focus from what you are doing. See the next section.
- A run missed while the PC was off or asleep starts as soon as the PC is back. The pipeline then waits up to 3 minutes for the network, and a late close or props snapshot skips games already under way.
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
Start-ScheduledTask -TaskPath '\NHLBetting\' -TaskName props-due    # run one now
Get-Content logs\props-due.log -Tail 20
Select-String -Path logs\*.log -Pattern "ERROR" | Select-Object -Last 5
Select-String -Path logs\*.log -Pattern "Credits remaining" | Select-Object -Last 3
.\ops\windows\register-tasks.ps1 -Unregister                        # remove them
```

To pause for the off-season, `Get-ScheduledTask -TaskPath '\NHLBetting\' | Disable-ScheduledTask`, and `Enable-ScheduledTask` to resume. Leaving them on costs no credits, but the logs keep growing.

## Two machines, two Odds API keys

Each machine reads its own `.env`, so each uses its own Odds API key, and each key has its own 500 free credits a month. The Mac's key pays for moneyline snapshots and closes, and this PC's key pays for props. If the two machines shared one key, their snapshots would draw on the same 500 credits, and the props schedule alone uses most of that.

Each machine also has its own database. The props machine's database holds props lines, the schedule, box scores, power-play stats and injuries; picks, paper bets and the CLV ledger live on the picks machine. A pick can be graded only against closing snapshots taken on the machine that made it, which is why the picks machine takes its own closes.

Put the PC's key in the PC's `.env` as `ODDS_API_KEY`. `PROPS_MARKETS`, `PROPS_BOOKMAKERS`, `PROPS_CLOSE_LEAD_MINUTES` and `PROPS_CLOSE_MIN_GAP_MINUTES` tune the props jobs; [.env.example](../../.env.example) lists them with their defaults.
