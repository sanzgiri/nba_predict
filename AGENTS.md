# Repository Guidelines

## Project Structure & Module Organization
- `code/` holds core model and feature engineering modules (ELO, RAPTOR, backtesting).
- Root scripts (`predict_today.py`, `collect_data.py`, `create_kelly.py`) are primary entry points.
- `standalone/` contains standalone runners and utilities.
- `data/` stores source/reference CSVs; `predictions/` and `perf/` store generated outputs; `logs/` stores runtime logs.
- `notebooks/` is for exploration and analysis; `tasks/` and `Dockerfile` support Docker/Jupyter workflows.

## Build, Test, and Development Commands
- `./setup.sh` verifies Python 3.10+, installs dependencies, and downloads baseline data.
- `python3 -m pip install -r requirements.txt` installs dependencies if you skip the setup script.
- `python3 predict_today.py` runs the daily ELO-based prediction flow.
- `python3 -m code.raptor_script_utils_v3` runs the RAPTOR pipeline.
- `python3 -m pytest` or `python3 test_installation.py` validates the environment.
- `tasks/build_docker.sh` builds the image; `tasks/run_docker.sh` launches Jupyter (update the path in the script to your local repo).

## Coding Style & Naming Conventions
- Python uses 4-space indentation and PEP 8 conventions.
- Use `snake_case` for modules, functions, and scripts (e.g., `predict_today.py`).
- Use `UPPER_SNAKE_CASE` for constants (see `config.py`).
- No formatter or linter is configured; keep changes tidy and consistent with nearby code.

## Testing Guidelines
- Tests are currently lightweight and live as `test_*.py` at the repo root.
- Prefer fast, offline tests; NBA API calls can be rate-limited or unavailable.
- When adding tests, keep names descriptive (e.g., `test_elo_predictor.py`) and cover data I/O assumptions.

## Commit & Pull Request Guidelines
- Recent history uses bot commits like `[bot] update files`; there is no formal convention.
- For human commits, use concise, imperative summaries (e.g., "Add daily ELO prediction output").
- PRs should include a brief description, key commands run with results, and any data file updates with sources and sizes.

## Configuration & Data Notes
- Adjust season settings and model parameters in `config.py` when rolling to a new season.
- If you add or refresh data in `data/`, document provenance in the PR and avoid breaking existing scripts.
