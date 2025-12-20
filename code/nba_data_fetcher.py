"""
NBA Data Fetcher - Modern data collection using nba_api
Collects game results, team stats, and player data for specified seasons
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2, playergamelogs, commonteamroster
    from nba_api.stats.static import teams
    from nba_api.live.nba.endpoints import scoreboard
except ImportError:
    raise ImportError("nba_api not installed. Run: pip install nba-api")

try:
    from utils import retry_on_failure, rate_limit, logger, cache_dataframe
    from config import TEAM_ABBREVIATION_MAPPINGS, DATA_PATHS
except ImportError:
    # Fallback if running standalone
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    TEAM_ABBREVIATION_MAPPINGS = {}
    DATA_PATHS = {}
    
    # Simple fallbacks for decorators
    def retry_on_failure(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def rate_limit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def cache_dataframe(filepath, fetch_func, **kwargs):
        import os
        if os.path.exists(filepath) and not kwargs.get('force_refresh', False):
            return pd.read_csv(filepath)
        df = fetch_func()
        df.to_csv(filepath, index=False)
        return df

# Rate limit to avoid NBA API throttling (max 60 requests per minute)
DELAY_BETWEEN_CALLS = 0.6  # seconds


class NBADataFetcher:
    """Fetch NBA game and player data using nba-api"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.teams = teams.get_teams()
        self.team_dict = {t['abbreviation']: t for t in self.teams}

    @staticmethod
    def _season_string(season_start_year: int) -> str:
        return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
        
    @retry_on_failure(max_retries=3, delay=2.0)
    @rate_limit(calls_per_minute=60)
    def get_season_games(self, season: str) -> pd.DataFrame:
        """
        Get all games for a season
        
        Args:
            season: Season string like '2023-24'
        
        Returns:
            DataFrame with game results
        """
        logger.info(f"Fetching games for season {season}")
        
        try:
            # Use leaguegamefinder to get all games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            
            games = gamefinder.get_data_frames()[0]
            
            logger.info(f"Found {len(games)} game records for {season}")
            return games
            
        except Exception as e:
            logger.error(f"Error fetching season games: {e}")
            raise
    
    def process_game_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw game data into usable format
        
        Args:
            games_df: Raw game data from nba_api
            
        Returns:
            Processed DataFrame with one row per game
        """
        # The API returns 2 rows per game (one for each team)
        # We need to combine them into single game rows
        
        # Sort by game date and ID
        games_df = games_df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
        
        processed_games = []
        
        # Process pairs of rows (home/away)
        for game_id in games_df['GAME_ID'].unique():
            game_rows = games_df[games_df['GAME_ID'] == game_id]
            
            if len(game_rows) != 2:
                logger.warning(f"Game {game_id} has {len(game_rows)} rows, skipping")
                continue
            
            row1, row2 = game_rows.iloc[0], game_rows.iloc[1]
            
            # Determine home/away (use MATCHUP field)
            # MATCHUP format: "TEAM @ OPPONENT" or "TEAM vs. OPPONENT"
            matchup1 = str(row1.get('MATCHUP', ''))
            
            if '@' in matchup1:
                # Row1 is away team
                away_row, home_row = row1, row2
            else:
                # Row1 is home team
                home_row, away_row = row1, row2
            
            game_data = {
                'game_id': game_id,
                'date': pd.to_datetime(home_row['GAME_DATE']),
                'season': home_row['SEASON_ID'],
                'home_team': home_row['TEAM_ABBREVIATION'],
                'away_team': away_row['TEAM_ABBREVIATION'],
                'home_score': int(home_row['PTS']),
                'away_score': int(away_row['PTS']),
                'home_win': 1 if home_row['WL'] == 'W' else 0,
            }
            
            processed_games.append(game_data)
        
        df = pd.DataFrame(processed_games)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Processed {len(df)} games")
        return df
    
    def fetch_season_data(self, season_start_year: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and process data for a complete season
        
        Args:
            season_start_year: Starting year of season (e.g., 2023 for 2023-24 season)
            force_refresh: Force re-download even if cached
            
        Returns:
            DataFrame with processed game data
        """
        season_str = self._season_string(season_start_year)
        cache_file = self.data_dir / f"{season_start_year + 1}_season_data.csv"
        
        def fetch_func():
            raw_games = self.get_season_games(season_str)
            time.sleep(DELAY_BETWEEN_CALLS)
            return self.process_game_data(raw_games)
        
        df = cache_dataframe(
            str(cache_file),
            fetch_func,
            max_age_hours=24,  # Cache for 24 hours
            force_refresh=force_refresh
        )
        
        return df

    def fetch_team_rosters(self, season_start_year: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch team rosters for a season.

        Args:
            season_start_year: Starting year of season (e.g., 2024 for 2024-25).
            force_refresh: Force re-download even if cached.

        Returns:
            DataFrame with roster data for all teams.
        """
        season_str = self._season_string(season_start_year)
        cache_file = self.data_dir / f"team_rosters_{season_start_year + 1}.csv"

        def fetch_func():
            rows = []
            for team in self.teams:
                team_id = team['id']
                team_abbrev = team['abbreviation']
                try:
                    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str)
                    df = roster.get_data_frames()[0]
                    df['TEAM_ABBREVIATION'] = team_abbrev
                    rows.append(df)
                    time.sleep(DELAY_BETWEEN_CALLS)
                except Exception as e:
                    logger.warning("Roster fetch failed for %s: %s", team_abbrev, e)
            if not rows:
                return pd.DataFrame()
            return pd.concat(rows, ignore_index=True)

        df = cache_dataframe(
            str(cache_file),
            fetch_func,
            max_age_hours=24,
            force_refresh=force_refresh
        )
        return df

    def fetch_player_game_logs(self, season_start_year: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch player game logs for a season.

        Args:
            season_start_year: Starting year of season (e.g., 2024 for 2024-25).
            force_refresh: Force re-download even if cached.

        Returns:
            DataFrame with player game logs.
        """
        season_str = self._season_string(season_start_year)
        cache_file = self.data_dir / f"{season_start_year + 1}_player_gamelogs.csv"

        def fetch_func():
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season_str,
                season_type_nullable='Regular Season'
            )
            return logs.get_data_frames()[0]

        df = cache_dataframe(
            str(cache_file),
            fetch_func,
            max_age_hours=24,
            force_refresh=force_refresh
        )
        return df

    @staticmethod
    def _parse_minutes(value: object) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        if ':' in text:
            mins, secs = text.split(':', 1)
            try:
                return float(mins) + float(secs) / 60.0
            except ValueError:
                return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def compute_recent_player_minutes(self, logs_df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
        """
        Compute recent average minutes per player.

        Args:
            logs_df: DataFrame from fetch_player_game_logs.
            lookback_days: How many days back to include.

        Returns:
            DataFrame with player minutes averages.
        """
        if logs_df.empty:
            return pd.DataFrame()

        df = logs_df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        cutoff = df['GAME_DATE'].max() - pd.Timedelta(days=lookback_days)
        df = df[df['GAME_DATE'] >= cutoff]
        df['MINUTES'] = df['MIN'].map(self._parse_minutes)

        grouped = (
            df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION'], as_index=False)
            .agg(avg_minutes=('MINUTES', 'mean'), games=('MINUTES', 'count'))
        )
        grouped = grouped.rename(columns={
            'PLAYER_ID': 'player_id',
            'PLAYER_NAME': 'player_name',
            'TEAM_ABBREVIATION': 'team',
        })
        return grouped

    @staticmethod
    def _position_to_numeric(position: str) -> float:
        mapping = {
            'PG': 0.0,
            'SG': 1.0,
            'SF': 2.0,
            'PF': 3.0,
            'C': 4.0,
            'G': 1.0,
            'F': 2.5,
        }
        if not position:
            return 2.0
        tokens = [token.strip().upper() for token in str(position).split('-')]
        values = [mapping.get(token, 2.0) for token in tokens if token]
        if not values:
            return 2.0
        return float(sum(values) / len(values))

    def build_depth_charts(self, rosters_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a depth chart from rosters and recent minutes.

        Returns:
            DataFrame with columns: abb, position, rank, name.
        """
        if rosters_df.empty or minutes_df.empty:
            return pd.DataFrame(columns=['abb', 'position', 'rank', 'name'])

        rosters = rosters_df.copy()
        rosters = rosters.rename(columns={
            'PLAYER_ID': 'player_id',
            'PLAYER': 'player_name',
            'POSITION': 'position_raw',
            'TEAM_ABBREVIATION': 'team',
        })
        merged = rosters.merge(minutes_df, on=['player_id', 'player_name', 'team'], how='left')
        merged['avg_minutes'] = merged['avg_minutes'].fillna(0.0)
        merged['position'] = merged['position_raw'].map(self._position_to_numeric)

        merged = merged.sort_values(['team', 'position', 'avg_minutes'], ascending=[True, True, False])
        merged['rank'] = merged.groupby(['team', 'position']).cumcount()

        depth = merged[['team', 'position', 'rank', 'player_name']].rename(columns={
            'team': 'abb',
            'player_name': 'name',
        })
        return depth.reset_index(drop=True)
    
    def fetch_multiple_seasons(self, start_year: int, end_year: int) -> Dict[int, pd.DataFrame]:
        """
        Fetch data for multiple seasons
        
        Args:
            start_year: First season start year
            end_year: Last season start year
            
        Returns:
            Dictionary mapping season end year to DataFrame
        """
        results = {}
        
        for year in range(start_year, end_year + 1):
            try:
                logger.info(f"Fetching season {year}-{year+1}")
                df = self.fetch_season_data(year)
                results[year + 1] = df
                
                # Be nice to the API
                time.sleep(DELAY_BETWEEN_CALLS)
                
            except Exception as e:
                logger.error(f"Failed to fetch season {year}: {e}")
                continue
        
        return results
    
    @retry_on_failure(max_retries=3, delay=2.0)
    @rate_limit(calls_per_minute=60)
    def get_todays_games(self) -> pd.DataFrame:
        """
        Get today's scheduled games
        
        Returns:
            DataFrame with today's games
        """
        try:
            # Use the live scoreboard endpoint
            board = scoreboard.ScoreBoard()
            games_data = board.games.get_dict()
            
            today_games = []
            
            for game in games_data:
                game_data = {
                    'game_id': game['gameId'],
                    'home_team': game['homeTeam']['teamTricode'],
                    'away_team': game['awayTeam']['teamTricode'],
                    'game_time': game.get('gameTimeUTC', ''),
                    'status': game['gameStatus'],
                }
                today_games.append(game_data)
            
            df = pd.DataFrame(today_games)
            logger.info(f"Found {len(df)} games scheduled for today")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return pd.DataFrame()


def main():
    """Example usage"""
    print("NBA Data Fetcher - Example Usage")
    print("=" * 60)
    
    fetcher = NBADataFetcher()
    
    # Test 1: Fetch a single season
    print("\n1. Fetching 2023-24 season...")
    df_2024 = fetcher.fetch_season_data(2023)
    print(f"   Retrieved {len(df_2024)} games")
    print(f"   Date range: {df_2024['date'].min()} to {df_2024['date'].max()}")
    print(f"   Sample game:")
    print(df_2024.head(1).to_string())
    
    # Test 2: Fetch today's games
    print("\n2. Fetching today's games...")
    today_games = fetcher.get_todays_games()
    if len(today_games) > 0:
        print(f"   Found {len(today_games)} games today:")
        print(today_games.to_string(index=False))
    else:
        print("   No games scheduled today")
    
    print("\n" + "=" * 60)
    print("Data fetcher ready!")


if __name__ == "__main__":
    main()
