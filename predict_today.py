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


def main():
    print("=" * 70)
    print("NBA Game Predictions for Today")
    print("=" * 70)
    print()
    
    # Initialize
    fetcher = NBADataFetcher()
    predictor = ELOPredictor()
    
    # Load and train on recent season data
    print("Loading historical data...")
    
    # Try to load combined 5-year dataset first
    data_file = Path("data/all_seasons_2020_2025.csv")
    
    if data_file.exists():
        print("Using combined 5-year dataset (2020-2025)...")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Fall back to single season
        print("Using 2023-24 season data...")
        data_file = Path("data/2024_season_data.csv")
        if not data_file.exists():
            print("Fetching 2023-24 season data...")
            df = fetcher.fetch_season_data(2023)
        else:
            df = pd.read_csv(data_file)
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
        
        # Make prediction
        home_prob, home_score, away_score = predictor.predict_game(
            home_team,
            away_team,
            game_date=game_date
        )
        
        # Get ELO ratings
        home_elo = predictor.get_rating(home_team)
        away_elo = predictor.get_rating(away_team)
        
        # Format output
        print(f"Game {idx + 1}: {away_team} @ {home_team}")
        print(f"  ELO Ratings: {away_team} {away_elo:.0f} | {home_team} {home_elo:.0f}")
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
