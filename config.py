"""
Configuration file for NBA Predictions
Centralized settings for data sources, API endpoints, and model parameters
"""

# Data Sources
DATA_SOURCES = {
    # FiveThirtyEight historical data (frozen as of 2022)
    'fivethirtyeight_raptor': 'https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv',
    'fivethirtyeight_elo': 'https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv',
    
    # Basketball Reference (via basketball_reference_web_scraper)
    # Uses the library's client, no direct URLs needed
    
    # NBA Stats API (via nba-api)
    # Uses the library's endpoints, no direct URLs needed
}

# NBA Teams (30 teams)
NBA_TEAMS = {
    'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
}

# Team Name Mappings
# Different sources use different abbreviations
TEAM_ABBREVIATION_MAPPINGS = {
    'CHA': 'CHO',
    'CHR': 'CHO',
    'NY': 'NYK',
    'NO': 'NOP',
    'NOR': 'NOP',
    'BRO': 'BRK',
    'BKN': 'BRK',
    'GS': 'GSW',
    'SA': 'SAS',
    'UTAH': 'UTA',
    'WSH': 'WAS'
}

# Model Parameters
MODEL_PARAMS = {
    # RAPTOR-based model calibration
    'raptor_slope': 0.84,          # Scale for RAPTOR ratings impact
    'avg_offensive_rating': 108.9,  # League average offensive rating
    'avg_pace': 99.45,              # League average pace
    
    # ELO parameters
    'home_court_advantage': 100,    # ELO points for home court
    'elo_k_factor': 20,             # Base K factor for ELO updates
    'elo_season_carryover': 0.75,   # Portion of ELO carried to next season
    'elo_mean_reversion': 1505,     # ELO mean reversion value

    # Schedule context defaults
    'rest_days_default': 6,         # Default rest days when no prior game
    'rest_days_cap': 7,             # Cap on rest days for feature stability
    'travel_distance_cap': 3000.0,  # Cap travel distance in miles

    # Calibration
    'calibration_holdout_fraction': 0.2,
    'calibration_min_samples': 500,

    # Score modeling
    'score_lookback_games': 10,
    'score_elo_weight': 1.0,
    'score_min_points': 80,
    
    # Training parameters
    'lookback_games': 4,            # Number of games for recent form
    'prior_weight': 40.0,           # Weight for Bayesian prior in RAPTOR
    
    # Ridge regression alphas for cross-validation
    'ridge_alphas': [10., 30., 50., 60., 100., 500., 1000., 3000.],
    'cv_folds': 4
}

# Season Parameters (updated for current season)
CURRENT_SEASON = {
    'year': 2026,                   # Season ending year
    'start_date': '2025-10-15',     # Approximate season start
    'end_date': '2026-06-30',       # Approximate season end
}

# Data Paths (relative to project root)
DATA_PATHS = {
    'player_boxscores': 'data/{year}_player_boxscores.csv',
    'season_data': 'data/{year}_season_data.csv',
    'raptor_ratings': 'data/raptors_player_stats.csv',
    'team_depth_charts': 'data/team_depth_charts.csv',
    'nba_locations': 'data/nba_locations_upd.csv',
    'recent_minutes': 'data/recent_player_minutes.csv',
    'team_rosters': 'data/team_rosters_{year}.csv',
    'player_game_logs': 'data/{year}_player_gamelogs.csv',
}

# Feature columns for modeling
FEATURES = {
    'raptor_basic': ['AWAY_p_allowed', 'HOME_p_allowed', 'AWAY_p_scored', 'HOME_p_scored'],
    'elo_features': ['elo_diff', 'home_court_advantage'],
}

# API Rate Limiting
RATE_LIMITS = {
    'basketball_reference': {
        'requests_per_minute': 20,
        'delay_seconds': 3,
    },
    'nba_stats': {
        'requests_per_minute': 60,
        'delay_seconds': 0.6,
    }
}

# Logging Configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/nba_predictions.log'
}

# Kelly Criterion & Betting Parameters
BETTING_PARAMS = {
    'default_bankroll': 10,         # Default betting unit
    'kelly_fraction': 1.0,          # Full Kelly (use 0.5 for half-Kelly)
    'min_edge_threshold': 0.0,      # Minimum edge to place bet
    'temperature_softmax': 0.5,     # Temperature for softmax strategy
}
