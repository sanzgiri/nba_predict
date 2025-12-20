#!/usr/bin/env python3
"""
Collect NBA data for seasons 2020-2025
This script fetches historical game data using the NBA API
"""

import sys
from pathlib import Path

# Add project root and code directory to path
project_root = Path(__file__).parent
code_dir = project_root / "code"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_dir))

from nba_data_fetcher import NBADataFetcher
from utils import logger


def main():
    print("=" * 70)
    print("NBA Data Collection - Seasons 2020-2025")
    print("=" * 70)
    print()
    
    # Initialize fetcher
    fetcher = NBADataFetcher()
    
    # Define seasons to fetch
    # Note: 2019-20 season was shortened due to COVID
    # 2020-21 season had the bubble playoffs
    seasons_to_fetch = [
        (2019, "2019-20 (COVID-shortened)"),
        (2020, "2020-21 (Bubble season)"),
        (2021, "2021-22"),
        (2022, "2022-23"),
        (2023, "2023-24"),
        (2024, "2024-25 (Current)"),
    ]
    
    print("This will fetch game data for the following seasons:")
    for year, description in seasons_to_fetch:
        print(f"  - {description}")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    print()
    
    # Fetch each season
    results = {}
    for season_start, description in seasons_to_fetch:
        print(f"\nFetching {description}...")
        print("-" * 70)
        
        try:
            df = fetcher.fetch_season_data(season_start, force_refresh=False)
            
            if len(df) > 0:
                print(f"✓ Successfully fetched {len(df)} games")
                print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"  Sample: {df.iloc[0]['away_team']} @ {df.iloc[0]['home_team']}")
                print(f"           {df.iloc[0]['away_score']}-{df.iloc[0]['home_score']}")
                results[season_start + 1] = df
            else:
                print(f"⚠ No data found for season {description}")
                
        except Exception as e:
            print(f"✗ Error fetching {description}: {e}")
            logger.error(f"Failed to fetch season {season_start}: {e}", exc_info=True)
    
    # Summary
    print()
    print("=" * 70)
    print("Data Collection Summary")
    print("=" * 70)
    
    total_games = sum(len(df) for df in results.values())
    print(f"\nSuccessfully collected {len(results)} seasons with {total_games} total games")
    print(f"\nData saved to:")
    
    for season_end, df in results.items():
        filepath = f"data/{season_end}_season_data.csv"
        print(f"  - {filepath} ({len(df)} games)")
    
    if len(results) > 0:
        print("\n✓ Data collection complete!")
        print("\nNext steps:")
        print("  1. Review the data files in the data/ directory")
        print("  2. Run: python3 -c 'from code.elo_predictor import demo_prediction; demo_prediction()'")
        print("  3. Build predictions!")
    else:
        print("\n⚠ No data was collected. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
