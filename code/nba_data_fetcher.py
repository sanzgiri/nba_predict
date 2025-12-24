"""
NBA Data Fetcher - Modern data collection using nba_api
Collects game results, team stats, and player data for specified seasons
"""

import io
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
import requests
from unidecode import unidecode

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
    from config import TEAM_ABBREVIATION_MAPPINGS, DATA_PATHS, MODEL_PARAMS, NBA_TEAMS
except ImportError:
    # Fallback if running standalone
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    TEAM_ABBREVIATION_MAPPINGS = {}
    DATA_PATHS = {}
    MODEL_PARAMS = {}
    NBA_TEAMS = set()
    
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

    def compute_recent_player_minutes(
        self,
        logs_df: pd.DataFrame,
        lookback_days: int = 14,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute recent average minutes per player.

        Args:
            logs_df: DataFrame from fetch_player_game_logs.
            lookback_days: How many days back to include.
            as_of_date: Optional anchor date for the lookback window.

        Returns:
            DataFrame with player minutes averages.
        """
        if logs_df.empty:
            return pd.DataFrame()

        df = logs_df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        anchor = pd.to_datetime(as_of_date).normalize() if as_of_date is not None else df['GAME_DATE'].max()
        if pd.isna(anchor):
            return pd.DataFrame()
        cutoff = anchor - pd.Timedelta(days=lookback_days)
        df = df[(df['GAME_DATE'] >= cutoff) & (df['GAME_DATE'] <= anchor)]
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
    def _normalize_player_name(name: object) -> str:
        text = unidecode(str(name)).lower()
        return re.sub(r"[^a-z0-9]", "", text)

    @staticmethod
    def _normalize_team_abbrev(team: object) -> str:
        if team is None:
            return ""
        text = str(team).strip().upper()
        if text in NBA_TEAMS:
            return text
        return TEAM_ABBREVIATION_MAPPINGS.get(text, text)

    def _normalize_injury_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])
        lower_cols = {col.lower(): col for col in df.columns}

        name_col = None
        team_col = None
        status_col = None
        for candidate in ['player', 'player_name', 'name']:
            if candidate in lower_cols:
                name_col = lower_cols[candidate]
                break
        for candidate in ['team', 'team_abbreviation', 'abbr', 'team_tricode', 'teamabbr']:
            if candidate in lower_cols:
                team_col = lower_cols[candidate]
                break
        for candidate in ['status', 'injury_status', 'game_status', 'availability']:
            if candidate in lower_cols:
                status_col = lower_cols[candidate]
                break

        if not name_col or not team_col or not status_col:
            logger.warning("Injury report missing required columns")
            return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])

        normalized = df[[name_col, team_col, status_col]].copy()
        normalized.columns = ['player_name', 'team', 'status']
        normalized['team'] = normalized['team'].map(self._normalize_team_abbrev)
        return normalized

    def _fetch_rotowire_injuries(self) -> pd.DataFrame:
        url = "https://www.rotowire.com/basketball/tables/injury-report.php"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text), flavor="lxml")
            if not tables:
                return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])
            df = tables[0]
        except Exception as exc:
            logger.warning("Rotowire injury fetch failed: %s", exc)
            return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])

        normalized = self._normalize_injury_df(df)
        if normalized.empty:
            return normalized
        normalized['source'] = 'rotowire'
        return normalized

    def load_injury_report(
        self,
        source: str = "auto",
        url: str = "",
        filepath: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not filepath and not force_refresh:
            cached_path_value = DATA_PATHS.get('injury_report', '')
            cached_path = Path(cached_path_value) if cached_path_value else None
            if cached_path and cached_path.is_file():
                try:
                    return self._normalize_injury_df(pd.read_csv(cached_path))
                except Exception as exc:
                    logger.warning("Failed to read cached injury report %s: %s", cached_path, exc)

        if filepath:
            path = Path(filepath)
            if path.exists():
                try:
                    if path.suffix.lower() == ".json":
                        data = pd.read_json(path)
                        return self._normalize_injury_df(data)
                    return self._normalize_injury_df(pd.read_csv(path))
                except Exception as exc:
                    logger.warning("Failed to read injury file %s: %s", path, exc)

        if source == "none":
            return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])

        if source in ("nba", "auto") and url:
            try:
                if url.endswith(".json"):
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = pd.json_normalize(response.json())
                    normalized = self._normalize_injury_df(data)
                else:
                    if url.endswith(".csv"):
                        data = pd.read_csv(url)
                    else:
                        data = pd.read_html(url)[0]
                    normalized = self._normalize_injury_df(data)
                if not normalized.empty:
                    normalized['source'] = 'nba'
                    return normalized
            except Exception as exc:
                logger.warning("NBA injury report fetch failed: %s", exc)

        if source in ("rotowire", "auto"):
            return self._fetch_rotowire_injuries()

        return pd.DataFrame(columns=['player_name', 'team', 'status', 'source'])

    def apply_injury_adjustments(
        self,
        minutes_df: pd.DataFrame,
        injuries_df: pd.DataFrame,
        out_statuses: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if minutes_df.empty or injuries_df.empty:
            return minutes_df

        out_statuses = out_statuses or MODEL_PARAMS.get('injury_out_statuses', [])
        out_set = {str(status).strip().lower() for status in out_statuses}
        injuries = injuries_df.copy()
        injuries['status_norm'] = injuries['status'].astype(str).str.lower().str.strip()
        injuries = injuries[injuries['status_norm'].isin(out_set)]
        if injuries.empty:
            return minutes_df

        minutes = minutes_df.copy()
        minutes['name_norm'] = minutes['player_name'].map(self._normalize_player_name)
        minutes['team_norm'] = minutes['team'].map(self._normalize_team_abbrev)
        injuries['name_norm'] = injuries['player_name'].map(self._normalize_player_name)
        injuries['team_norm'] = injuries['team'].map(self._normalize_team_abbrev)
        injuries['key'] = injuries['name_norm'] + "::" + injuries['team_norm']
        minutes['key'] = minutes['name_norm'] + "::" + minutes['team_norm']

        out_keys = set(injuries['key'])
        minutes.loc[minutes['key'].isin(out_keys), 'avg_minutes'] = 0.0

        return minutes.drop(columns=['name_norm', 'team_norm', 'key'])

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

    def compute_player_impact(
        self,
        logs_df: pd.DataFrame,
        lookback_days: Optional[int] = None,
        shrinkage_games: Optional[int] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute a simple per-minute player impact metric from game logs.

        Uses Hollinger Game Score per minute with shrinkage toward 0.
        If as_of_date is provided, the lookback window is anchored to that date.
        """
        if logs_df.empty:
            return pd.DataFrame()

        if lookback_days is None:
            lookback_days = MODEL_PARAMS.get('player_impact_lookback_days', 30)
        if shrinkage_games is None:
            shrinkage_games = MODEL_PARAMS.get('player_impact_shrinkage_games', 10)

        df = logs_df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        anchor = pd.to_datetime(as_of_date).normalize() if as_of_date is not None else df['GAME_DATE'].max()
        if pd.isna(anchor):
            return pd.DataFrame()
        cutoff = anchor - pd.Timedelta(days=lookback_days)
        df = df[(df['GAME_DATE'] >= cutoff) & (df['GAME_DATE'] <= anchor)]

        df['MINUTES'] = df['MIN'].map(self._parse_minutes)
        df = df[df['MINUTES'] > 0]

        game_score = (
            df['PTS']
            + 0.4 * df['FGM']
            - 0.7 * df['FGA']
            - 0.4 * (df['FTA'] - df['FTM'])
            + 0.7 * df['OREB']
            + 0.3 * df['DREB']
            + df['AST']
            + 0.7 * df['STL']
            + 0.7 * df['BLK']
            - 0.4 * df['PF']
            - df['TOV']
        )
        df['IMPACT_PER_MIN'] = game_score / df['MINUTES']

        grouped = (
            df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION'], as_index=False)
            .agg(
                impact_per_min=('IMPACT_PER_MIN', 'mean'),
                games=('IMPACT_PER_MIN', 'count'),
                minutes=('MINUTES', 'sum')
            )
        )

        shrink = grouped['games'] / (grouped['games'] + shrinkage_games)
        grouped['impact_per_min'] = grouped['impact_per_min'] * shrink

        grouped = grouped.rename(columns={
            'PLAYER_ID': 'player_id',
            'PLAYER_NAME': 'player_name',
            'TEAM_ABBREVIATION': 'team',
        })
        return grouped

    def build_team_player_adjustments(
        self,
        impact_df: pd.DataFrame,
        minutes_df: pd.DataFrame,
        scale: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Aggregate player impacts to team-level adjustments.
        Returns per-team ELO adjustments.
        """
        if impact_df.empty or minutes_df.empty:
            return pd.DataFrame(columns=['team', 'team_impact_per_min', 'player_adj_elo'])

        if scale is None:
            scale = MODEL_PARAMS.get('player_adj_elo_scale', 60.0)

        merged = impact_df.merge(
            minutes_df,
            on=['player_id', 'player_name', 'team'],
            how='left'
        )
        merged['avg_minutes'] = merged['avg_minutes'].fillna(0.0)
        merged['weighted_impact'] = merged['impact_per_min'] * merged['avg_minutes']

        team_agg = (
            merged.groupby('team', as_index=False)
            .agg(weighted_impact=('weighted_impact', 'sum'), team_minutes=('avg_minutes', 'sum'))
        )
        team_agg['team_minutes'] = team_agg['team_minutes'].replace(0.0, 1.0)
        team_agg['team_impact_per_min'] = team_agg['weighted_impact'] / team_agg['team_minutes']

        league_avg = (
            team_agg['weighted_impact'].sum() / team_agg['team_minutes'].sum()
            if team_agg['team_minutes'].sum() > 0 else 0.0
        )

        team_agg['impact_delta'] = team_agg['team_impact_per_min'] - league_avg
        team_agg['player_adj_elo'] = team_agg['impact_delta'] * scale

        return team_agg[['team', 'team_impact_per_min', 'player_adj_elo']]
    
    @retry_on_failure(max_retries=3, delay=2.0)
    @rate_limit(calls_per_minute=60)
    def _et_today(self) -> datetime.date:
        tz = pytz.timezone("US/Eastern")
        return datetime.now(tz).date()

    def get_todays_games(self, target_date: Optional[datetime.date] = None) -> pd.DataFrame:
        """
        Get today's scheduled games
        
        Returns:
            DataFrame with today's games
        """
        try:
            target_date = target_date or self._et_today()
            # Use the live scoreboard endpoint
            used_explicit_date = False
            try:
                board = scoreboard.ScoreBoard(date=target_date.strftime("%Y-%m-%d"))
                used_explicit_date = True
            except TypeError:
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
            if not df.empty and 'game_time' in df.columns:
                df['game_date_et'] = pd.to_datetime(df['game_time'], utc=True, errors='coerce')
                df['game_date_et'] = df['game_date_et'].dt.tz_convert("US/Eastern").dt.date
                filtered = df[df['game_date_et'] == target_date].copy()
                if not filtered.empty or used_explicit_date:
                    df = filtered
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
