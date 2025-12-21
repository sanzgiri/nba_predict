#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEASON_END=$(uv run python - <<'PY'
from config import CURRENT_SEASON
print(CURRENT_SEASON["year"])
PY
)
SEASON_START=$((SEASON_END - 1))
HIST_START_YEAR=${HIST_START_YEAR:-2019}
PLAYER_MINUTES_DAYS=${PLAYER_MINUTES_DAYS:-14}
INJURY_SOURCE=${INJURY_SOURCE:-auto}
INJURY_URL=${INJURY_URL:-}
INJURY_FILE=${INJURY_FILE:-}
injury_args=(--injury-source "$INJURY_SOURCE")
if [[ -n "$INJURY_URL" ]]; then
  injury_args+=(--injury-url "$INJURY_URL")
fi
if [[ -n "$INJURY_FILE" ]]; then
  injury_args+=(--injury-file "$INJURY_FILE")
fi

uv run python collect_data.py --start-year "$SEASON_START" --end-year "$SEASON_START" --force --yes
uv run python collect_data.py --skip-games --player-data --player-season-start "$SEASON_START" \
  --recent-minutes-days "$PLAYER_MINUTES_DAYS" --yes "${injury_args[@]}"
uv run python collect_data.py --skip-games --combine-seasons --combine-output data/all_seasons_latest.csv \
  --start-year "$HIST_START_YEAR" --end-year "$SEASON_START" --yes
uv run python predict_today.py
