#!/usr/bin/env python3
"""
Make predictions for today's NBA games using ELO ratings
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from nba_data_fetcher import NBADataFetcher
from elo_predictor import ELOPredictor
from config import CURRENT_SEASON, DATA_PATHS, MODEL_PARAMS


def _get_game_date(game: pd.Series) -> pd.Timestamp:
    game_time = game.get('game_time', '')
    if isinstance(game_time, str) and game_time:
        try:
            parsed = pd.to_datetime(game_time, utc=True)
            if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
                parsed = parsed.tz_convert(None)
            return parsed.normalize()
        except Exception:
            pass
    return pd.Timestamp.now().normalize()

def _pick_training_file() -> Path | None:
    data_dir = Path("data")
    candidates = [data_dir / "all_seasons_latest.csv"]

    combined_files = sorted(
        data_dir.glob("all_seasons_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(combined_files)

    season_end = int(CURRENT_SEASON.get("year", 2025))
    candidates.append(data_dir / f"{season_end}_season_data.csv")
    candidates.append(data_dir / "2024_season_data.csv")

    for path in candidates:
        if path.is_file():
            return path
    return None

def _load_team_adjustments() -> dict:
    path = Path(DATA_PATHS.get('team_player_adjustments', 'data/team_player_adjustments_latest.csv'))
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    if 'team' not in df.columns or 'player_adj_elo' not in df.columns:
        return {}
    return dict(zip(df['team'], df['player_adj_elo']))


def main():
    print("=" * 70)
    print("NBA Game Predictions for Today")
    print("=" * 70)
    print()
    
    # Initialize
    fetcher = NBADataFetcher()
    predictor = ELOPredictor()
    player_adjustments = {}
    if MODEL_PARAMS.get('player_adj_enabled', True):
        player_adjustments = _load_team_adjustments()
    
    # Load and train on recent season data
    print("Loading historical data...")
    
    # Try combined dataset first (latest if available)
    data_file = _pick_training_file()
    if data_file and data_file.is_file():
        print(f"Using training data from {data_file}...")
        df = pd.read_csv(data_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    else:
        # Fall back to current season
        season_start = int(CURRENT_SEASON.get("year", 2025)) - 1
        season_file = Path(f"data/{season_start + 1}_season_data.csv")
        if not season_file.exists():
            print(f"Fetching {season_start}-{str(season_start + 1)[-2:]} season data...")
            df = fetcher.fetch_season_data(season_start)
        else:
            print(f"Using {season_start}-{str(season_start + 1)[-2:]} season data...")
            df = pd.read_csv(season_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
    
    print(f"Training on {len(df)} games...")
    predictor.train_on_games(df, track_accuracy=False)
    
    # Get today's games
    print("\nFetching today's games...")
    today_games = fetcher.get_todays_games()
    
    if len(today_games) == 0:
        print("No games scheduled for today.")
        return
    
    print(f"Found {len(today_games)} games today!\n")
    
    # Make predictions
    print("=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    print()
    
    predictions = []
    for idx, game in today_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = _get_game_date(game)
        
        home_adj = float(player_adjustments.get(home_team, 0.0))
        away_adj = float(player_adjustments.get(away_team, 0.0))

        # Make prediction
        home_prob, home_score, away_score = predictor.predict_game(
            home_team,
            away_team,
            game_date=game_date,
            home_elo_adjustment=home_adj,
            away_elo_adjustment=away_adj,
        )
        
        # Get ELO ratings
        home_elo = predictor.get_rating(home_team)
        away_elo = predictor.get_rating(away_team)
        
        # Format output
        print(f"Game {idx + 1}: {away_team} @ {home_team}")
        print(f"  ELO Ratings: {away_team} {away_elo:.0f} | {home_team} {home_elo:.0f}")
        if home_adj or away_adj:
            print(f"  Player Adj (ELO): {away_team} {away_adj:+.1f} | {home_team} {home_adj:+.1f}")
        print(f"  Predicted Score: {away_score:.1f} - {home_score:.1f}")
        print(f"  Win Probability: {away_team} {(1-home_prob)*100:.1f}% | {home_team} {home_prob*100:.1f}%")
        
        # Pick winner
        if home_prob > 0.5:
            winner = home_team
            confidence = home_prob * 100
        else:
            winner = away_team
            confidence = (1 - home_prob) * 100
        
        print(f"  Predicted Winner: {winner} ({confidence:.1f}% confidence)")
        print()

        predictions.append({
            'date': game_date.date(),
            'away_team': away_team,
            'home_team': home_team,
            'away_elo': predictor.get_rating(away_team),
            'home_elo': predictor.get_rating(home_team),
            'away_player_adj_elo': away_adj,
            'home_player_adj_elo': home_adj,
            'predicted_away_score': away_score,
            'predicted_home_score': home_score,
            'home_win_probability': home_prob,
            'predicted_winner': winner,
        })
    
    # Show current top 10 rankings
    print("=" * 70)
    print("CURRENT ELO RANKINGS (Top 10)")
    print("=" * 70)
    print()
    print(predictor.get_rankings(top_n=10).to_string(index=False))
    
    # Save predictions
    predictions_file = Path("predictions") / f"predictions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    predictions_file.parent.mkdir(exist_ok=True)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(predictions_file, index=False)
    
    print()
    print(f"\n✓ Predictions saved to: {predictions_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
