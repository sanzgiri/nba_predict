"""
Utility functions for NBA predictions
Includes error handling, logging, and common operations
"""

import logging
import time
import os
from functools import wraps
from typing import Optional, Callable, Any
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
def setup_logging(log_file: Optional[str] = None, level: str = 'INFO'):
    """Configure logging for the application"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if log_file is None:
        log_file = os.path.join(log_dir, f'nba_predictions_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry a function on failure with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} retries failed for {func.__name__}: {str(e)}")
            
            raise last_exception
        
        return wrapper
    return decorator


def rate_limit(calls_per_minute: int = 60):
    """
    Decorator to rate limit function calls
    
    Args:
        calls_per_minute: Maximum number of calls allowed per minute
    """
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def normalize_team_name(team: str, mapping: dict) -> str:
    """
    Normalize team abbreviations across different data sources
    
    Args:
        team: Team abbreviation
        mapping: Dictionary of mappings
    
    Returns:
        Normalized team abbreviation
    """
    return mapping.get(team, team)


def cache_dataframe(filepath: str, fetch_func: Callable, 
                   max_age_hours: float = 24, force_refresh: bool = False) -> pd.DataFrame:
    """
    Cache dataframe to disk and reload if fresh enough
    
    Args:
        filepath: Path to cache file
        fetch_func: Function to fetch fresh data
        max_age_hours: Maximum age of cache in hours
        force_refresh: Force fetch even if cache is fresh
    
    Returns:
        Cached or freshly fetched dataframe
    """
    # Check if cache exists and is fresh
    if not force_refresh and os.path.exists(filepath):
        file_age = time.time() - os.path.getmtime(filepath)
        if file_age < max_age_hours * 3600:
            logger.info(f"Loading cached data from {filepath} (age: {file_age/3600:.1f}h)")
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                logger.warning(f"Failed to load cache from {filepath}: {e}")
    
    # Fetch fresh data
    logger.info(f"Fetching fresh data for {filepath}")
    df = fetch_func()
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Cached data saved to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cache data to {filepath}: {e}")
    
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: list, 
                       min_rows: int = 1, name: str = "DataFrame") -> bool:
    """
    Validate that a dataframe has required structure
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name of dataframe for logging
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if df is None or df.empty:
        raise ValueError(f"{name} is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(f"{name} has {len(df)} rows, minimum {min_rows} required")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} missing required columns: {missing_cols}")
    
    logger.info(f"{name} validated: {len(df)} rows, {len(df.columns)} columns")
    return True


def get_date_range(start_date: str, end_date: str) -> list:
    """
    Generate list of dates between start and end
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        List of datetime objects
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    return dates


def format_prediction_output(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction output for display
    
    Args:
        predictions: Raw predictions dataframe
    
    Returns:
        Formatted dataframe
    """
    df = predictions.copy()
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        if 'prob' in col.lower() or 'win' in col.lower():
            df[col] = df[col].round(3)
        elif 'score' in col.lower() or 'points' in col.lower():
            df[col] = df[col].round(1)
        else:
            df[col] = df[col].round(2)
    
    return df


def print_model_metrics(mae_away: float, mae_home: float, accuracy: float):
    """Print formatted model performance metrics"""
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("=" * 60)
    logger.info(f"MAE (Away Team):  {mae_away:.2f} points")
    logger.info(f"MAE (Home Team):  {mae_home:.2f} points")
    logger.info(f"Win Prediction Accuracy: {accuracy:.1%}")
    logger.info("=" * 60)
