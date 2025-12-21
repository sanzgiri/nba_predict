# Implementation Status

## Implemented
- Season data collection via NBA API with cached refresh, plus combined seasons file for training.
- Daily prediction flow (`predict_today.py`) that trains on the latest combined data before inference and writes outputs to `predictions/`.
- Player signal pipeline: recent minutes, per-minute Game Score impact, team-level ELO adjustments, and optional injury-based minute zeroing.
- Backtest calibration script added; `player_adj_elo_scale` set to 100 based on validation log loss.
- Automation: `tasks/daily_run.sh` refreshes season data, player logs, injuries, recombines seasons, then runs predictions.
- Scheduling: launchd plist for a daily 10:30 ET-equivalent run (adjust for local time).

## Next Steps
- Wire a stable NBA injury report URL or a local report file into the daily job to avoid scraping risk.
- Add alerts/monitoring around daily runs (log review, failure notifications).
