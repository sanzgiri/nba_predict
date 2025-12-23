# Running the Project (uv)

## Prerequisites
- `uv` installed (`uv --version`)
- Python 3.10+ available (3.11 recommended: `python3.11 --version`)

## 1) Create the uv environment
```bash
uv venv --python 3.11 .venv
```

Activate it when you want a shell session inside the venv:
```bash
source .venv/bin/activate
```

You can also skip activation and use `uv run` for all commands.

## 2) Install dependencies
```bash
uv pip install -r requirements.txt
```

## 3) Data setup
Minimum for tests:
- `data/raptors_player_stats.csv` (already present in this repo).

Optional downloads (from `setup.sh`):
```bash
curl -o data/raptors_player_stats.csv \
  https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv
curl -o data/538_nba_elo.csv \
  https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv
```

To fetch recent season game data via NBA API:
```bash
uv run python collect_data.py
```

To fetch rosters, player game logs, and refresh minutes/depth charts:
```bash
uv run python collect_data.py --player-data --player-season-start 2024 --recent-minutes-days 14
```

This also writes:
- `data/player_impact_<year>.csv`
- `data/team_player_adjustments_latest.csv` (used by daily predictions when enabled)
- `data/injury_report_latest.csv` (when an injury source is available)

Player adjustments can be toggled and tuned in `config.py`:
- `player_adj_enabled`
- `player_adj_elo_scale`
- `player_injury_enabled`
- `injury_out_statuses`

Optional injury inputs:
```bash
uv run python collect_data.py --player-data --injury-source auto
uv run python collect_data.py --player-data --injury-source nba --injury-url "<report-url>"
uv run python collect_data.py --player-data --injury-file data/injury_report.csv
```

To build a combined seasons file:
```bash
uv run python collect_data.py --combine-seasons --combine-output data/all_seasons_latest.csv --start-year 2019 --end-year 2024
```

Daily update (refresh current season, rebuild combined file, run predictions):
```bash
tasks/daily_run.sh
```
To pass injury inputs into the daily job:
```bash
INJURY_SOURCE=auto INJURY_URL="<report-url>" tasks/daily_run.sh
```

Recommended backtest window: train on 2021-22 and 2022-23, validate on 2023-24.

Backtest calibration for `player_adj_elo_scale`:
```bash
uv run python backtest_player_adj_scale.py --scales 0,20,40,60,80,100
```

## 8) Daily job (launchd, macOS)
Default schedule is 10:30 local time. Adjust the hour/minute in the plist to match 10:30 ET.

Install:
```bash
cp tasks/launchd/com.sanzgiri.nba_predictions.daily.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.sanzgiri.nba_predictions.daily.plist
```
Edit `IMESSAGE_TO` in the plist to enable text delivery.

Manual run:
```bash
launchctl start com.sanzgiri.nba_predictions.daily
```

Manual run without launchd:
```bash
tasks/daily_run.sh
```

To send iMessage/SMS predictions after each run, set `IMESSAGE_TO`:
```bash
IMESSAGE_TO="+15035551212,+14155550123" tasks/daily_run.sh
```
If there are no games, the message will say "No games today."

Logs:
- `logs/launchd_daily.log`
- `logs/launchd_daily.err`

If you move the repo, update `WorkingDirectory` and log paths in the plist. To supply an NBA injury report URL, set `INJURY_URL` in the plist.
For iMessage/SMS, set `IMESSAGE_TO` in the plist (e.g., `+15035551212`).

## 4) Validate the setup
```bash
uv run python test_installation.py
```

## 5) Run predictions
ELO-based daily predictions (writes to `predictions/`):
```bash
uv run python predict_today.py
```
If `data/all_seasons_2020_2025.csv` or `data/2024_season_data.csv` is missing, the script will fetch data via NBA API.

RAPTOR pipeline (modernized utilities):
```bash
uv run python -m code.raptor_script_utils_v3
```

## 6) Notebooks and app
Jupyter:
```bash
uv run jupyter notebook
```

Streamlit (legacy app uses older RAPTOR utilities):
```bash
uv run streamlit run standalone/run.py
```

## 7) Docker (optional)
```bash
tasks/build_docker.sh
tasks/run_docker.sh
```
Update the repo path inside `tasks/run_docker.sh` before running.
