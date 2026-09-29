# launchd job templates

Templates for the Mac's scheduled runs. launchd is the job scheduler built into macOS; each job is a small XML file (a plist) in `~/Library/LaunchAgents`. These files are templates. Nothing installs them, and the live Mac's existing `com.nhlbetting.daily` and `com.nhlbetting.odds` plists are not replaced automatically. On Windows, use [ops/windows/](../windows/) instead.

| File | Runs | When | Credits a run |
|---|---|---|---|
| `com.nhlbetting.daily.plist` | `python pipeline.py daily` | 9:00 | 6 |
| `com.nhlbetting.odds.plist` (optional) | `python pipeline.py odds` | 13:00 | 6 |
| `com.nhlbetting.close.plist` | `python pipeline.py close --due` | every 15 minutes | 2 when a game is about to start, otherwise 0 |

Times are the Mac's local time, and the defaults suit a Mac on Central time. Every snapshot run makes no Odds API request, and costs nothing, when no game starts in the next 24 hours. `close --due` takes its 2-credit moneyline snapshot only when some game starts within 16 minutes and no moneyline snapshot is less than 16 minutes old; the other runs log one line and exit. So every start time gets one close, in the last 15-minute run before puck drop, afternoon games included, and the close job has no times to adjust. Those two limits are defaults, set by `CLOSE_LEAD_MINUTES` and `CLOSE_MIN_GAP_MINUTES` in `.env`.

**Credits:** at the defaults, the closes average about 8 credits a game day, so the daily run plus the close job comes to at most about 456 credits in any month of the 2026-27 schedule, under the free plan's 500. Adding the midday `odds` run goes over. [Snapshot schedule](../../README.md#snapshot-schedule) in the main README has the numbers.

Each file has two placeholders:

- `__REPO__`: the repo's absolute path, with no trailing slash, such as `/Users/you/nhl-betting-system`.
- `__PYTHON__`: the Python the project is installed in, usually `__REPO__/.venv/bin/python`. Calling it directly means the job doesn't need to activate the virtual environment.

Each job starts in the repo folder, and a relative `DATA_DIR` such as `./data` from `.env.example` always resolves against the repo, so it is `<repo>/data`. Its output and log lines go to `logs/<job>.log` in the repo, which git ignores. launchd doesn't create that folder, so make it before the first run.

## Install a job

```bash
REPO="$HOME/nhl-betting-system"            # the repo's absolute path
PY="$REPO/.venv/bin/python"
JOB=close                                  # daily, odds, or close
mkdir -p "$REPO/logs"
sed -e "s|__REPO__|$REPO|g" -e "s|__PYTHON__|$PY|g" \
    "$REPO/ops/launchd/com.nhlbetting.$JOB.plist" \
    > "$HOME/Library/LaunchAgents/com.nhlbetting.$JOB.plist"
plutil -lint "$HOME/Library/LaunchAgents/com.nhlbetting.$JOB.plist"
launchctl load "$HOME/Library/LaunchAgents/com.nhlbetting.$JOB.plist"
launchctl list | grep com.nhlbetting
```

`plutil -lint` checks the file. In the `launchctl list` output, the middle column is the job's last exit code.

## On the live Mac

The Mac already runs `com.nhlbetting.daily` and `com.nhlbetting.odds` from its own plists, with their own times and log paths.

1. Add the close job with the steps above, using `JOB=close`. If an earlier fixed-time close (17:30 and 20:30) is already installed, unload it first, as in step 2.
2. Keep the existing daily and odds files, or compare them with the templates first. To replace one, unload it before writing the new file, because `launchctl load` refuses a job that is already loaded:

```bash
launchctl unload "$HOME/Library/LaunchAgents/com.nhlbetting.daily.plist"
# then run the install steps with JOB=daily
```

## Change a time

Edit `Hour` and `Minute` in the installed daily or odds file (or `StartInterval`, in seconds, in the close file), then reload it:

```bash
launchctl unload "$HOME/Library/LaunchAgents/com.nhlbetting.daily.plist"
launchctl load "$HOME/Library/LaunchAgents/com.nhlbetting.daily.plist"
```

- The close job needs no adjusting for the slate: it reads each game's start time from the database. `python pipeline.py close` (without `--due`) still takes a snapshot on demand, for 2 credits.
- A game with no snapshot between its pick and its puck drop settles with a blank CLV.

## Sleep, Docker, and the off-season

- If the Mac is asleep at a daily or odds time, launchd runs the job once when it wakes; several missed times collapse into one run. The 15-minute close runs are different: launchd skips the ones that fall while the Mac sleeps, so a sleeping Mac takes no closes. A Mac that is shut down skips everything.
- After a wake, the pipeline waits up to 3 minutes for the network. A close that runs late skips the games already under way. If the daily and midday jobs fire together, picks are written under a database lock, so they can't both issue a pick for the same game or both spend the same part of the day's budget.
- Docker Desktop must be running, or every job fails at the database check.
- To pause for the off-season, `launchctl unload` each installed file. Leaving them loaded costs no credits, because no request is made while no game starts in the next 24 hours, but the logs keep growing (the close log by a line every 15 minutes).

## Logs

`logs/daily.log`, `logs/odds.log`, and `logs/close.log` grow without limit, so trim or delete them now and then. `grep "Credits remaining" logs/*.log | tail -3` shows the latest credit counts. Since 2026-09-28 the Odds API key never appears in these logs. Logs written by the older jobs may contain it.

## Two machines

Each machine reads its own `.env`, so a second machine (such as the Windows PC) can use its own Odds API key with its own 500 free credits a month. Sharing one key splits those 500 credits between the machines.
