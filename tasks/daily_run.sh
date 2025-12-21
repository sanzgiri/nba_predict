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

uv run python collect_data.py --start-year "$SEASON_START" --end-year "$SEASON_START" --force --yes
uv run python collect_data.py --skip-games --combine-seasons --combine-output data/all_seasons_latest.csv \
  --start-year "$HIST_START_YEAR" --end-year "$SEASON_START" --yes
uv run python predict_today.py
