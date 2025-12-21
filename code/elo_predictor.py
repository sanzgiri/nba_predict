"""
Simple ELO-based NBA Prediction System
Self-contained predictor that only requires game results (no external ratings)

ELO is a simple rating system where:
- Teams start at 1500
- Winners gain rating points, losers lose points
- Margin of victory matters
- Home court advantage is ~100 points
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import sys
from collections import deque
from geopy import distance as geo_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils import logger
    from config import MODEL_PARAMS, NBA_TEAMS, DATA_PATHS, TEAM_ABBREVIATION_MAPPINGS
except ImportError:
    # Fallback if running standalone
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    MODEL_PARAMS = {
        'home_court_advantage': 100,
        'elo_k_factor': 20,
        'elo_season_carryover': 0.75,
        'elo_mean_reversion': 1505,
    }
    DATA_PATHS = {
        'nba_locations': 'data/nba_locations_upd.csv',
    }
    TEAM_ABBREVIATION_MAPPINGS = {
        'CHA': 'CHO', 'CHR': 'CHO', 'NY': 'NYK', 'NO': 'NOP',
        'NOR': 'NOP', 'BRO': 'BRK', 'BKN': 'BRK', 'GS': 'GSW',
        'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS',
    }
    NBA_TEAMS = {
        'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    }

# ELO Parameters
HOME_ADVANTAGE = MODEL_PARAMS.get('home_court_advantage', 100)
K_FACTOR = MODEL_PARAMS.get('elo_k_factor', 20)
SEASON_CARRYOVER = MODEL_PARAMS.get('elo_season_carryover', 0.75)
MEAN_REVERSION = MODEL_PARAMS.get('elo_mean_reversion', 1505)
REST_DAYS_DEFAULT = MODEL_PARAMS.get('rest_days_default', 6)
REST_DAYS_CAP = MODEL_PARAMS.get('rest_days_cap', 7)
TRAVEL_DISTANCE_CAP = MODEL_PARAMS.get('travel_distance_cap', 3000.0)


class ELOPredictor:
    """Simple ELO-based prediction system for NBA games"""
    
    def __init__(self, initial_elo: float = 1500.0):
        self.initial_elo = initial_elo
        self.ratings: Dict[str, float] = {}
        self.history: list = []
        self.location_index = self._load_location_index()
        self.team_last_played: Dict[str, datetime] = {}
        self.team_last_location: Dict[str, str] = {}
        self.calibrator = None
        self.calibration_metrics: Dict[str, float] = {}
        self.feature_columns: list = []
        self.score_lookback = MODEL_PARAMS.get('score_lookback_games', 10)
        self.team_stats: Dict[str, Dict[str, deque]] = {}
        self.league_avg_points = 110.0
        self.league_avg_total_points = 220.0

    def _load_location_index(self) -> pd.DataFrame:
        path = Path(DATA_PATHS.get('nba_locations', 'data/nba_locations_upd.csv'))
        if not path.exists():
            logger.warning("Location data not found at %s", path)
            return pd.DataFrame()

        df = pd.read_csv(path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.drop(columns=[df.columns[0]])
        df['code'] = df['code'].astype(str).str.strip()
        return df.set_index('code')

    def _resolve_location_code(self, team: str) -> str:
        if not self.location_index.empty and team in self.location_index.index:
            return team
        mapped = TEAM_ABBREVIATION_MAPPINGS.get(team)
        if mapped and not self.location_index.empty and mapped in self.location_index.index:
            return mapped
        return team

    def _distance_miles(self, code_a: Optional[str], code_b: Optional[str]) -> float:
        if not code_a or not code_b or self.location_index.empty:
            return 0.0
        if code_a == code_b:
            return 0.0
        if code_a not in self.location_index.index or code_b not in self.location_index.index:
            return 0.0
        row_a = self.location_index.loc[code_a]
        row_b = self.location_index.loc[code_b]
        miles = geo_distance.distance(
            (row_a['latitude'], row_a['longitude']),
            (row_b['latitude'], row_b['longitude'])
        ).miles
        return float(min(miles, TRAVEL_DISTANCE_CAP))

    def _tz_diff(self, home_code: str, away_code: str) -> float:
        if self.location_index.empty:
            return 0.0
        if home_code not in self.location_index.index or away_code not in self.location_index.index:
            return 0.0
        home_tz = self.location_index.loc[home_code]['tzrel']
        away_tz = self.location_index.loc[away_code]['tzrel']
        return float(home_tz - away_tz)

    def _home_elevation(self, home_code: str) -> float:
        if self.location_index.empty or home_code not in self.location_index.index:
            return 0.0
        return float(self.location_index.loc[home_code]['elevation'])

    def _rest_days(self, last_date: Optional[datetime], game_date: datetime) -> int:
        if isinstance(game_date, pd.Timestamp) and game_date.tzinfo is not None:
            game_date = game_date.tz_convert(None)
        if isinstance(last_date, pd.Timestamp) and last_date.tzinfo is not None:
            last_date = last_date.tz_convert(None)
        if last_date is None:
            return REST_DAYS_DEFAULT
        delta = (game_date - last_date).days
        if delta < 0:
            return REST_DAYS_DEFAULT
        return int(min(delta, REST_DAYS_CAP))

    def _schedule_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        team_last_played: Optional[Dict[str, datetime]] = None,
        team_last_location: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        if team_last_played is None:
            team_last_played = self.team_last_played
        if team_last_location is None:
            team_last_location = self.team_last_location

        home_code = self._resolve_location_code(home_team)
        away_code = self._resolve_location_code(away_team)

        rest_home = self._rest_days(team_last_played.get(home_team), game_date)
        rest_away = self._rest_days(team_last_played.get(away_team), game_date)

        travel_home = self._distance_miles(team_last_location.get(home_team), home_code)
        travel_away = self._distance_miles(team_last_location.get(away_team), home_code)

        tz_diff = self._tz_diff(home_code, away_code)
        home_elevation = self._home_elevation(home_code)
        day_of_week = int(game_date.dayofweek)

        return {
            'rest_home': float(rest_home),
            'rest_away': float(rest_away),
            'travel_home': float(travel_home),
            'travel_away': float(travel_away),
            'tz_diff': float(tz_diff),
            'home_elevation': float(home_elevation),
            'day_of_week': float(day_of_week),
        }

    def _elo_win_prob(self, elo_diff: float) -> float:
        return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

    def _predict_home_win_prob(self, features: Dict[str, float]) -> float:
        if self.calibrator is None:
            return self._elo_win_prob(features['elo_diff'])
        feature_order = self.feature_columns or list(features.keys())
        X = pd.DataFrame([features])[feature_order]
        return float(self.calibrator.predict_proba(X)[0, 1])

    def _fit_calibrator(self):
        if not hasattr(self, "_training_features") or self._training_features.empty:
            logger.warning("No training features available for calibration")
            return

        df = self._training_features.copy()
        df['outcome'] = self._training_outcomes
        df['date'] = self._training_feature_dates
        df = df.dropna().sort_values('date').reset_index(drop=True)

        if df['outcome'].nunique() < 2:
            logger.warning("Calibration skipped: only one class present in outcomes")
            return

        holdout_frac = MODEL_PARAMS.get('calibration_holdout_fraction', 0.2)
        min_samples = MODEL_PARAMS.get('calibration_min_samples', 500)
        if len(df) < min_samples:
            logger.warning("Calibration skipped: only %d samples (<%d)", len(df), min_samples)
            return

        split_idx = int(len(df) * (1 - holdout_frac))
        if split_idx <= 0 or split_idx >= len(df):
            logger.warning("Calibration skipped: invalid holdout split")
            return

        feature_cols = [col for col in df.columns if col not in ['outcome', 'date']]
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df['outcome']
        X_val = val_df[feature_cols]
        y_val = val_df['outcome']

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        self.calibrator = model
        self.feature_columns = feature_cols

        val_probs = model.predict_proba(X_val)[:, 1]
        baseline_probs = 1.0 / (1.0 + np.power(10, -val_df['elo_diff'].to_numpy() / 400.0))

        metrics = {
            'holdout_games': float(len(val_df)),
            'log_loss': float(log_loss(y_val, val_probs)),
            'brier': float(brier_score_loss(y_val, val_probs)),
            'baseline_log_loss': float(log_loss(y_val, baseline_probs)),
            'baseline_brier': float(brier_score_loss(y_val, baseline_probs)),
        }
        self.calibration_metrics = metrics
        logger.info(
            "Calibration holdout log_loss=%.4f (baseline %.4f), brier=%.4f (baseline %.4f) on %d games",
            metrics['log_loss'],
            metrics['baseline_log_loss'],
            metrics['brier'],
            metrics['baseline_brier'],
            int(metrics['holdout_games'])
        )

    def _init_team_stats(self) -> Dict[str, Dict[str, deque]]:
        stats = {}
        for team in NBA_TEAMS:
            stats[team] = {
                'points_for': deque(maxlen=self.score_lookback),
                'points_against': deque(maxlen=self.score_lookback),
                'total_points': deque(maxlen=self.score_lookback),
            }
        return stats

    def _update_team_stats(self, team: str, points_for: float, points_against: float):
        if team not in self.team_stats:
            self.team_stats[team] = {
                'points_for': deque(maxlen=self.score_lookback),
                'points_against': deque(maxlen=self.score_lookback),
                'total_points': deque(maxlen=self.score_lookback),
            }
        self.team_stats[team]['points_for'].append(points_for)
        self.team_stats[team]['points_against'].append(points_against)
        self.team_stats[team]['total_points'].append(points_for + points_against)

    def _team_stat_avg(self, team: str, key: str, default: float) -> float:
        values = self.team_stats.get(team, {}).get(key, [])
        if not values:
            return default
        return float(sum(values) / len(values))

    def _expected_scores(self, home_team: str, away_team: str, elo_diff: float) -> Tuple[float, float]:
        league_points = self.league_avg_points or 110.0
        league_total = self.league_avg_total_points or 220.0

        home_off = self._team_stat_avg(home_team, 'points_for', league_points)
        home_def = self._team_stat_avg(home_team, 'points_against', league_points)
        away_off = self._team_stat_avg(away_team, 'points_for', league_points)
        away_def = self._team_stat_avg(away_team, 'points_against', league_points)

        base_home = (home_off + away_def) / 2.0
        base_away = (away_off + home_def) / 2.0

        home_pace = self._team_stat_avg(home_team, 'total_points', league_total)
        away_pace = self._team_stat_avg(away_team, 'total_points', league_total)
        pace_factor = ((home_pace + away_pace) / 2.0) / league_total if league_total else 1.0

        expected_home = base_home * pace_factor
        expected_away = base_away * pace_factor

        elo_weight = MODEL_PARAMS.get('score_elo_weight', 1.0)
        elo_margin = (elo_diff / 25.0) * elo_weight
        expected_home += elo_margin / 2.0
        expected_away -= elo_margin / 2.0

        min_points = MODEL_PARAMS.get('score_min_points', 80)
        expected_home = max(expected_home, min_points)
        expected_away = max(expected_away, min_points)

        return expected_home, expected_away

    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team"""
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
        return self.ratings[team]
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        home_elo_adjustment: float = 0.0,
        away_elo_adjustment: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Predict the outcome of a game
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date (defaults to today)
            
        Returns:
            Tuple of (home_win_prob, expected_home_score, expected_away_score)
        """
        if game_date is None:
            game_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            game_date = pd.to_datetime(game_date)
            if isinstance(game_date, pd.Timestamp) and game_date.tzinfo is not None:
                game_date = game_date.tz_convert(None)
            game_date = game_date.normalize()

        home_elo = self.get_rating(home_team) + home_elo_adjustment
        away_elo = self.get_rating(away_team) + away_elo_adjustment
        
        # Adjust for home court advantage
        home_elo_adj = home_elo + HOME_ADVANTAGE
        
        # Calculate win probability using ELO formula
        schedule = self._schedule_features(home_team, away_team, game_date)
        features = {'elo_diff': home_elo_adj - away_elo, **schedule}
        expected_home = self._predict_home_win_prob(features)
        
        expected_home_score, expected_away_score = self._expected_scores(
            home_team,
            away_team,
            home_elo_adj - away_elo
        )
        
        return expected_home, expected_home_score, expected_away_score
    
    def update_ratings(self, home_team: str, away_team: str, 
                      home_score: int, away_score: int) -> Tuple[float, float]:
        """
        Update ELO ratings based on game result
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation  
            home_score: Actual home team score
            away_score: Actual away team score
            
        Returns:
            Tuple of (home_rating_change, away_rating_change)
        """
        # Get current ratings
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Adjust for home court
        home_elo_adj = home_elo + HOME_ADVANTAGE
        
        # Expected outcome
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo_adj) / 400.0))
        expected_away = 1.0 - expected_home
        
        # Actual outcome
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif away_score > home_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # Margin of victory multiplier (more decisive wins count more)
        mov = abs(home_score - away_score)
        elo_diff = home_elo_adj - away_elo
        
        # MOV multiplier: ranges from 1 to ~3
        mov_multiplier = np.log(max(1, mov)) * (2.2 / ((elo_diff * 0.001) + 2.2))
        
        # Calculate K-factor with MOV adjustment
        k_adjusted = K_FACTOR * mov_multiplier
        
        # Update ratings
        home_change = k_adjusted * (actual_home - expected_home)
        away_change = k_adjusted * (actual_away - expected_away)
        
        self.ratings[home_team] = home_elo + home_change
        self.ratings[away_team] = away_elo + away_change
        
        return home_change, away_change
    
    def apply_season_carryover(self):
        """Apply mean reversion between seasons"""
        logger.info("Applying season carryover (mean reversion)")
        for team in self.ratings:
            old_rating = self.ratings[team]
            new_rating = old_rating * SEASON_CARRYOVER + MEAN_REVERSION * (1 - SEASON_CARRYOVER)
            self.ratings[team] = new_rating
            logger.debug(f"{team}: {old_rating:.1f} -> {new_rating:.1f}")
    
    def train_on_games(
        self,
        games_df: pd.DataFrame,
        track_accuracy: bool = True,
        calibrate: bool = True,
    ):
        """
        Train the ELO system on historical games
        
        Args:
            games_df: DataFrame with columns: date, home_team, away_team, home_score, away_score
            track_accuracy: Whether to track prediction accuracy during training
        """
        # Filter out G League games - only train on NBA teams
        games_df = games_df[
            games_df['home_team'].isin(NBA_TEAMS) & 
            games_df['away_team'].isin(NBA_TEAMS)
        ].copy()
        
        self.league_avg_points = float(
            (games_df['home_score'].mean() + games_df['away_score'].mean()) / 2.0
        )
        self.league_avg_total_points = float(
            games_df['home_score'].mean() + games_df['away_score'].mean()
        )
        self.team_stats = self._init_team_stats()

        games_df = games_df.sort_values('date').reset_index(drop=True)
        logger.info(f"Training on {len(games_df)} NBA games (G League games filtered out)")
        
        correct_predictions = 0
        total_predictions = 0
        team_last_played: Dict[str, datetime] = {}
        team_last_location: Dict[str, str] = {}
        feature_rows = []
        outcomes = []
        feature_dates = []
        
        for idx, game in games_df.iterrows():
            game_date = pd.to_datetime(game['date']).normalize()
            home_team = game['home_team']
            away_team = game['away_team']

            home_elo = self.get_rating(home_team)
            away_elo = self.get_rating(away_team)
            elo_diff = (home_elo + HOME_ADVANTAGE) - away_elo

            schedule = self._schedule_features(
                home_team,
                away_team,
                game_date,
                team_last_played=team_last_played,
                team_last_location=team_last_location,
            )

            feature_rows.append({
                'elo_diff': float(elo_diff),
                **schedule,
            })
            outcomes.append(1 if game['home_score'] > game['away_score'] else 0)
            feature_dates.append(game_date)

            # Make prediction before updating
            if track_accuracy:
                home_win_prob = self._elo_win_prob(elo_diff)
                predicted_winner = home_team if home_win_prob > 0.5 else away_team
                actual_winner = home_team if game['home_score'] > game['away_score'] else away_team
                
                if predicted_winner == actual_winner:
                    correct_predictions += 1
                total_predictions += 1
            
            # Update ratings based on result
            self.update_ratings(
                home_team,
                away_team,
                int(game['home_score']),
                int(game['away_score'])
            )

            self._update_team_stats(home_team, float(game['home_score']), float(game['away_score']))
            self._update_team_stats(away_team, float(game['away_score']), float(game['home_score']))

            home_location = self._resolve_location_code(home_team)
            team_last_played[home_team] = game_date
            team_last_played[away_team] = game_date
            team_last_location[home_team] = home_location
            team_last_location[away_team] = home_location
            
            # Store history
            self.history.append({
                'date': game['date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_elo': self.ratings[home_team],
                'away_elo': self.ratings[away_team],
            })
        
        if track_accuracy and total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info(f"Training accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")

        self.team_last_played = team_last_played
        self.team_last_location = team_last_location
        self._training_features = pd.DataFrame(feature_rows)
        self._training_outcomes = outcomes
        self._training_feature_dates = feature_dates

        if calibrate:
            self._fit_calibrator()
    
    def get_rankings(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get current ELO rankings"""
        rankings = [
            {'team': team, 'elo_rating': rating}
            for team, rating in self.ratings.items()
        ]
        df = pd.DataFrame(rankings).sort_values('elo_rating', ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'team', 'elo_rating']]
    
    def save_ratings(self, filepath: str):
        """Save current ratings to CSV"""
        df = self.get_rankings()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved ratings to {filepath}")
    
    def load_ratings(self, filepath: str):
        """Load ratings from CSV"""
        df = pd.read_csv(filepath)
        self.ratings = dict(zip(df['team'], df['elo_rating']))
        logger.info(f"Loaded ratings for {len(self.ratings)} teams from {filepath}")


def demo_prediction():
    """Demonstrate the ELO system with a simple example"""
    print("=" * 60)
    print("ELO Prediction System - Demo")
    print("=" * 60)
    
    # Try to load recent season data
    data_file = Path("data/2024_season_data.csv")
    
    if not data_file.exists():
        print("\nNo data file found. Fetching 2023-24 season...")
        try:
            from code.nba_data_fetcher import NBADataFetcher
            fetcher = NBADataFetcher()
            df = fetcher.fetch_season_data(2023)
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("\nUsing synthetic example instead...")
            # Create synthetic example
            df = pd.DataFrame({
                'date': pd.date_range('2024-10-01', periods=10),
                'home_team': ['LAL', 'GSW', 'BOS', 'MIA', 'DEN'] * 2,
                'away_team': ['LAC', 'SAC', 'PHI', 'ATL', 'PHX'] * 2,
                'home_score': [110, 115, 105, 98, 112, 108, 120, 102, 95, 118],
                'away_score': [105, 108, 110, 102, 108, 105, 115, 100, 100, 115],
            })
    else:
        print(f"\nLoading data from {data_file}...")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Training on {len(df)} games")
    
    # Create and train predictor
    predictor = ELOPredictor()
    predictor.train_on_games(df)
    
    # Show top 10 teams
    print("\nTop 10 Teams by ELO Rating:")
    print(predictor.get_rankings(top_n=10).to_string(index=False))
    
    # Make sample predictions for today's games
    print("\nSample Predictions:")
    print("-" * 60)
    
    # Get top 2 teams for demo
    top_teams = predictor.get_rankings(top_n=2)
    team1 = top_teams.iloc[0]['team']
    team2 = top_teams.iloc[1]['team']
    
    prob, home_score, away_score = predictor.predict_game(team1, team2)
    print(f"\n{team1} (Home) vs {team2} (Away)")
    print(f"  Win Probability: {prob:.1%} - {(1-prob):.1%}")
    print(f"  Expected Score: {home_score:.1f} - {away_score:.1f}")
    print(f"  ELO Ratings: {predictor.get_rating(team1):.0f} - {predictor.get_rating(team2):.0f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_prediction()
