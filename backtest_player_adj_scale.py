#!/usr/bin/env python3
"""
Backtest player adjustment scales on recent seasons to pick an ELO shift.
Default window: train 2021-22/2022-23, validate 2023-24.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

project_root = Path(__file__).parent
code_dir = project_root / "code"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_dir))

from nba_data_fetcher import NBADataFetcher
from elo_predictor import ELOPredictor
from config import DATA_PATHS, MODEL_PARAMS, NBA_TEAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest player adjustment ELO scale.")
    parser.add_argument("--train-start-year", type=int, default=2021, help="Training season start year.")
    parser.add_argument("--train-end-year", type=int, default=2022, help="Training season end year.")
    parser.add_argument("--val-start-year", type=int, default=2023, help="Validation season start year.")
    parser.add_argument("--val-end-year", type=int, default=2023, help="Validation season end year.")
    parser.add_argument(
        "--scales",
        default="0,20,40,60,80,100",
        help="Comma-separated list of player_adj_elo_scale values to test.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=MODEL_PARAMS.get("player_impact_lookback_days", 30),
        help="Lookback window (days) for player impact/minutes.",
    )
    parser.add_argument(
        "--shrinkage-games",
        type=int,
        default=MODEL_PARAMS.get("player_impact_shrinkage_games", 10),
        help="Shrinkage games for player impact.",
    )
    parser.add_argument("--no-calibrate", action="store_true", help="Disable ELO calibrator.")
    parser.add_argument("--output", default="", help="Optional CSV output path for results.")
    return parser.parse_args()


def _season_data_path(season_start_year: int) -> Path:
    return Path(DATA_PATHS["season_data"].format(year=season_start_year + 1))


def _player_logs_path(season_start_year: int) -> Path:
    return Path(DATA_PATHS["player_game_logs"].format(year=season_start_year + 1))


def load_season_games(season_start_year: int) -> pd.DataFrame:
    path = _season_data_path(season_start_year)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)]
    return df.sort_values("date").reset_index(drop=True)


def load_player_logs(season_start_year: int) -> pd.DataFrame:
    path = _player_logs_path(season_start_year)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.dropna(subset=["GAME_DATE"])


def build_adjustments_for_date(
    fetcher: NBADataFetcher,
    logs_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int,
    shrinkage_games: int,
    scale: float,
) -> dict:
    impact = fetcher.compute_player_impact(
        logs_df,
        lookback_days=lookback_days,
        shrinkage_games=shrinkage_games,
        as_of_date=as_of_date,
    )
    minutes = fetcher.compute_recent_player_minutes(
        logs_df,
        lookback_days=lookback_days,
        as_of_date=as_of_date,
    )
    team_adjustments = fetcher.build_team_player_adjustments(
        impact,
        minutes,
        scale=scale,
    )
    if team_adjustments.empty:
        return {}
    return dict(zip(team_adjustments["team"], team_adjustments["player_adj_elo"]))


def evaluate_season(
    predictor: ELOPredictor,
    fetcher: NBADataFetcher,
    games_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    scale: float,
    lookback_days: int,
    shrinkage_games: int,
) -> dict:
    outcomes = []
    probs = []
    adjustment_cache: dict[pd.Timestamp, dict] = {}

    for game in games_df.itertuples():
        game_date = pd.to_datetime(game.date).normalize()
        if game_date not in adjustment_cache:
            adjustment_cache[game_date] = build_adjustments_for_date(
                fetcher,
                logs_df,
                game_date,
                lookback_days,
                shrinkage_games,
                scale,
            )
        adjustments = adjustment_cache[game_date]
        home_adj = float(adjustments.get(game.home_team, 0.0))
        away_adj = float(adjustments.get(game.away_team, 0.0))

        home_prob, _, _ = predictor.predict_game(
            game.home_team,
            game.away_team,
            game_date=game_date,
            home_elo_adjustment=home_adj,
            away_elo_adjustment=away_adj,
        )
        outcome = 1 if game.home_score > game.away_score else 0
        probs.append(home_prob)
        outcomes.append(outcome)

        predictor.update_ratings(game.home_team, game.away_team, int(game.home_score), int(game.away_score))
        predictor._update_team_stats(game.home_team, float(game.home_score), float(game.away_score))
        predictor._update_team_stats(game.away_team, float(game.away_score), float(game.home_score))

        home_location = predictor._resolve_location_code(game.home_team)
        predictor.team_last_played[game.home_team] = game_date
        predictor.team_last_played[game.away_team] = game_date
        predictor.team_last_location[game.home_team] = home_location
        predictor.team_last_location[game.away_team] = home_location

    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)
    accuracy = float(((probs_arr >= 0.5) == outcomes_arr).mean())
    return {
        "games": int(len(outcomes_arr)),
        "log_loss": float(log_loss(outcomes_arr, probs_arr)),
        "brier": float(brier_score_loss(outcomes_arr, probs_arr)),
        "accuracy": accuracy,
    }


def main() -> None:
    args = parse_args()
    scales = [float(value.strip()) for value in args.scales.split(",") if value.strip()]
    if not scales:
        raise ValueError("No scales provided.")

    missing = []
    train_frames = []
    for season_start in range(args.train_start_year, args.train_end_year + 1):
        try:
            train_frames.append(load_season_games(season_start))
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        print("Missing season data files:")
        for path in missing:
            print(f"  - {path}")
        print("Run: uv run python collect_data.py --start-year 2021 --end-year 2023")
        sys.exit(1)

    train_df = pd.concat(train_frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    val_seasons = range(args.val_start_year, args.val_end_year + 1)
    season_games = {}
    season_logs = {}
    for season_start in val_seasons:
        try:
            season_games[season_start] = load_season_games(season_start)
        except FileNotFoundError as exc:
            missing.append(str(exc))
        try:
            season_logs[season_start] = load_player_logs(season_start)
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        print("Missing validation files:")
        for path in sorted(set(missing)):
            print(f"  - {path}")
        print("Run player data fetches per season, for example:")
        print("  uv run python collect_data.py --player-data --player-season-start 2023 --yes")
        sys.exit(1)

    fetcher = NBADataFetcher()
    results = []

    for scale in scales:
        predictor = ELOPredictor()
        predictor.train_on_games(train_df, calibrate=not args.no_calibrate)

        aggregate = {"games": 0, "log_loss": 0.0, "brier": 0.0, "accuracy": 0.0}
        for season_start in val_seasons:
            metrics = evaluate_season(
                predictor,
                fetcher,
                season_games[season_start],
                season_logs[season_start],
                scale,
                args.lookback_days,
                args.shrinkage_games,
            )
            aggregate["games"] += metrics["games"]
            aggregate["log_loss"] += metrics["log_loss"] * metrics["games"]
            aggregate["brier"] += metrics["brier"] * metrics["games"]
            aggregate["accuracy"] += metrics["accuracy"] * metrics["games"]

        total_games = max(1, aggregate["games"])
        results.append({
            "scale": scale,
            "games": aggregate["games"],
            "log_loss": aggregate["log_loss"] / total_games,
            "brier": aggregate["brier"] / total_games,
            "accuracy": aggregate["accuracy"] / total_games,
        })

    results_df = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    print(results_df.to_string(index=False))
    best = results_df.iloc[0]
    print(f"\nBest scale by log_loss: {best['scale']:.1f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
