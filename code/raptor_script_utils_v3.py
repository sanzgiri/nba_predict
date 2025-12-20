"""
Updated RAPTOR script utilities with modern dependencies and error handling
This is a modernized version of raptor_script_utils_v2.py
"""

import pandas as pd
from datetime import date, datetime, timezone, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import linear_model
import os
import ast
from statsmodels.tools import add_constant
import statsmodels.api as sm
import unidecode
import re
import logging
from typing import Optional, Dict, Tuple, List

# Import local modules
try:
    from utils import retry_on_failure, rate_limit, logger
    from config import TEAM_ABBREVIATION_MAPPINGS, MODEL_PARAMS
except ImportError:
    # Fallback if running standalone
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    TEAM_ABBREVIATION_MAPPINGS = {
        'CHA': 'CHO', 'CHR': 'CHO', 'NY': 'NYK', 'NO': 'NOP', 
        'NOR': 'NOP', 'BRO': 'BRK', 'BKN': 'BRK', 'GS': 'GSW'
    }
    MODEL_PARAMS = {
        'raptor_slope': 0.84,
        'avg_offensive_rating': 108.9,
        'avg_pace': 99.45,
    }

# Try to import basketball_reference_web_scraper, fall back gracefully
try:
    from basketball_reference_web_scraper import client
    BASKETBALL_REF_AVAILABLE = True
except ImportError:
    logger.warning("basketball_reference_web_scraper not available. Some functions will be disabled.")
    BASKETBALL_REF_AVAILABLE = False

# Try to import nba_api as alternative
try:
    from nba_api.stats.endpoints import scoreboardv2, playergamelogs
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    logger.warning("nba_api not available. Using fallback methods.")
    NBA_API_AVAILABLE = False


def utc_to_local(utc_dt):
    """Utility to get datetime of games in a different time zone"""
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


@retry_on_failure(max_retries=3, delay=2.0)
def get_injured() -> pd.DataFrame:
    """
    Pulls current injury data from rotowire and returns a dataframe of injured players
    
    Note: This web scraping approach may break if the website changes.
    Consider using official NBA injury API or sports data providers.
    
    Returns:
        DataFrame with columns: name, abb, player_id
    """
    try:
        # This is a placeholder - the original scraping method is likely broken
        # TODO: Replace with NBA API injury endpoint or alternative source
        logger.warning("get_injured() using deprecated scraping method")
        
        os.system("wget -q -O injury.txt \"https://www.rotowire.com/basketball/tables/injury-report.php?Team=LAC&Pos=C\"")
        
        if not os.path.exists("injury.txt"):
            logger.error("Failed to download injury data")
            return pd.DataFrame(columns=['name', 'abb', 'player_id'])
        
        with open("injury.txt", "r") as myfile:
            data = myfile.read().replace('\n', '')
        
        data = data[1:-1]
        mydict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(mydict)
        
        dff = df[df.status.isin(['Out', 'Out For Season'])].copy()
        dff.rename(columns={'team': 'abb', 'player': 'name'}, inplace=True)
        
        # Map team abbreviations
        dff.abb = dff.abb.replace(TEAM_ABBREVIATION_MAPPINGS)
        
        # Create player_id (this is a heuristic, not perfect)
        dff['player_id'] = (dff.lastname.map(lambda x: x[:6].lower()) + 
                           dff['firstname'].map(lambda x: x[:2].lower()) + '01')
        
        return dff[['name', 'abb', 'player_id']]
    
    except Exception as e:
        logger.error(f"Error in get_injured(): {e}")
        return pd.DataFrame(columns=['name', 'abb', 'player_id'])


def roster_minutes_injuries(team: str, dfp: pd.DataFrame, dfr: pd.DataFrame, 
                           num_games: int) -> pd.DataFrame:
    """
    Gets baseline minutes projection and joins with raptor ratings
    
    Args:
        team: Team abbreviation
        dfp: DataFrame with player minutes data
        dfr: DataFrame with RAPTOR ratings
        num_games: Lookback period for recent games
    
    Returns:
        DataFrame with projected minutes and ratings
    """
    try:
        games = dfp[dfp.team == team].game_index.unique()
        games = np.sort(games)
        last = games[-num_games:] if len(games) >= num_games else games

        df_temp = dfp[(dfp.team == team) & (dfp.game_index.isin(last))]

        seconds_projection = (df_temp.groupby('slug')[['seconds_played', 'seconds_played_adj']]
                             .mean()).reset_index()
        
        dft = pd.merge(seconds_projection, dfr, left_on='slug', right_on='player_id')
        
        dft['ratings_product_offense'] = dft['raptor_offense'] * dft['seconds_played']
        dft['ratings_product_defense'] = dft['raptor_defense'] * dft['seconds_played']
        dft['ratings_product_pace'] = dft['pace_impact'] * dft['seconds_played']
        
        dft['team'] = team
        
        # Load depth charts if available
        try:
            df_dc = pd.read_csv('data/team_depth_charts.csv').iloc[:, 1:]
            dftm = pd.merge(df_dc, dft, left_on='name', right_on='player_name', how='right')
            dftm.position.fillna(2.0, inplace=True)  # Default to SF position
        except FileNotFoundError:
            logger.warning("Depth charts not found, using simple allocation")
            dftm = dft.copy()
            dftm['position'] = 2.0
        
        # Handle injured players (zero minutes)
        players_to_zap = dftm[dftm.seconds_played_adj == 0].player_name.unique()
        if len(players_to_zap) > 0:
            for player in players_to_zap:
                dftm = zap_player_pos(dftm, player, 0.75)
            
        return dftm
    
    except Exception as e:
        logger.error(f"Error in roster_minutes_injuries() for team {team}: {e}")
        raise


def game_agg(dft_home: pd.DataFrame, dft_away: pd.DataFrame, 
            adj: float, avg_ort: float, avg_pace: float) -> pd.DataFrame:
    """
    Computes features needed by the model to predict scores and win probabilities
    
    Args:
        dft_home: Home team roster with minutes and RAPTOR ratings
        dft_away: Away team roster with minutes and RAPTOR ratings
        adj: RAPTOR adjustment factor
        avg_ort: Average offensive rating
        avg_pace: Average pace
    
    Returns:
        DataFrame with game-level features
    """
    home_team = dft_home['team'].iloc[0]
    dft_home['location'] = 'HOME'
    
    dft_away['location'] = 'AWAY'
    away_team = dft_away['team'].iloc[0]

    df_teams = pd.concat([dft_home, dft_away], ignore_index=True)
    
    cols = ['ratings_product_offense', 'ratings_product_defense', 
            'ratings_product_pace', 'seconds_played']
    
    dft_agg = df_teams.groupby(['location', 'team'])[cols].sum()
    dft_agg['game_index'] = home_team + '_v_' + away_team
    
    dfg = game_raptor_calcs(dft_agg, adj, avg_ort, avg_pace)
    
    return dfg


def game_raptor_calcs(dft: pd.DataFrame, adj: float, avg_ort: float, 
                     avg_pace: float) -> pd.DataFrame:
    """Compute RAPTOR features needed for model"""
    PLAYERS = 5
    ADJ = adj
    AVG_ORT = avg_ort
    AVG_PACE = avg_pace
    
    dft['ORT'] = ADJ * PLAYERS * (dft['ratings_product_offense'] / dft['seconds_played'])
    dft['DRT'] = ADJ * PLAYERS * (dft['ratings_product_defense'] / dft['seconds_played'])
    dft['pace'] = ADJ * PLAYERS * (dft['ratings_product_pace'] / dft['seconds_played'])
    dft['p_scored'] = (AVG_ORT + dft['ORT']) * (dft['pace'] + AVG_PACE) / 100.
    dft['p_allowed'] = (AVG_ORT - dft['DRT']) * (dft['pace'] + AVG_PACE) / 100.
    
    # Reshape to one row per game
    game_data = dft.reset_index().set_index(['game_index', 'location'])[
        ['p_allowed', 'p_scored', 'team']].unstack()
    
    # Format the reshaped data
    vals = game_data.columns.levels[0]
    home_away = game_data.columns.levels[1]
    col_names = [loc + '_' + val for val in vals for loc in home_away]

    dfg = pd.DataFrame(data=game_data.values, columns=col_names, index=game_data.index)
    
    return dfg


def zap_player_pos(dft: pd.DataFrame, player_name: str, pos_w: float) -> pd.DataFrame:
    """
    Helper function to reallocate a player's minutes through the rest of the team
    Takes into account player position when allocating minutes
    
    Args:
        dft: DataFrame with team roster
        player_name: Name of player to remove
        pos_w: Weight for position similarity (0-1)
    
    Returns:
        DataFrame with reallocated minutes
    """
    idx = dft[dft.player_name == player_name].index[0]
    seconds_to_allocate = dft.loc[idx, 'seconds_played']
    position = dft.loc[idx, 'position']

    dff = dft[dft.player_name != player_name].copy()
    T = 60.

    # Higher weight for more similar positions
    pos_score = (1600 - np.abs(400 * (dff.position - position))) / 60.
    score = (1300 - np.abs(dff.seconds_played - 1200)) / T

    w_minutes = np.exp(score) / np.exp(score).sum()
    w_depth = np.exp(pos_score) / np.exp(pos_score).sum()

    # Multiply the position and minutes ranking and then re-normalize
    weight = (pos_w * w_depth) * ((1 - pos_w) * w_minutes)
    weight = weight / weight.sum()

    dff['seconds_played'] = dff['seconds_played'] + seconds_to_allocate * weight
    dff['w_minutes'] = w_minutes
    dff['w_depth'] = w_depth

    dff['ratings_product_offense'] = dff['raptor_offense'] * dff['seconds_played']
    dff['ratings_product_defense'] = dff['raptor_defense'] * dff['seconds_played']
    dff['ratings_product_pace'] = dff['pace_impact'] * dff['seconds_played']

    return dff


# Additional helper functions for the updated pipeline
def get_abbreviation_mapping() -> Dict[str, str]:
    """Helper function to map between different team names"""
    try:
        df_season = pd.read_csv('data/2020_season_data.csv')
        d = dict()
        for i in df_season.groupby(['winning_abbr', 'winning_name'])['away_assists'].count().keys():
            value = i[0]
            key = i[1].upper()
            d[key] = value
        return d
    except FileNotFoundError:
        logger.warning("Season data not found, using default mappings")
        return TEAM_ABBREVIATION_MAPPINGS


if __name__ == "__main__":
    # Test basic functionality
    logger.info("RAPTOR utils module loaded successfully")
    logger.info(f"Basketball Reference available: {BASKETBALL_REF_AVAILABLE}")
    logger.info(f"NBA API available: {NBA_API_AVAILABLE}")
