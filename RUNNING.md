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

To build a combined seasons file:
```bash
uv run python collect_data.py --combine-seasons --start-year 2019 --end-year 2024
```

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
