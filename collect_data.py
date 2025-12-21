#!/usr/bin/env python3
"""
Collect NBA data for seasons and player-level inputs.
This script fetches historical game data and optional roster/player logs using the NBA API.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root and code directory to path
project_root = Path(__file__).parent
code_dir = project_root / "code"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_dir))

from nba_data_fetcher import NBADataFetcher
from utils import logger
from config import CURRENT_SEASON, DATA_PATHS, MODEL_PARAMS


def parse_args():
    parser = argparse.ArgumentParser(description="Collect NBA data via NBA API")
    parser.add_argument("--start-year", type=int, default=2019, help="Season start year (e.g., 2019)")
    parser.add_argument("--end-year", type=int, default=2024, help="Season start year end (e.g., 2024)")
    parser.add_argument("--skip-games", action="store_true", help="Skip season game data fetch")
    parser.add_argument("--force", action="store_true", help="Force refresh cached data")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--combine-seasons", action="store_true", help="Write combined seasons CSV")
    parser.add_argument(
        "--combine-output",
        default="",
        help="Output path for combined seasons CSV (e.g., data/all_seasons_latest.csv)",
    )
    parser.add_argument("--player-data", action="store_true", help="Fetch rosters and player logs")
    parser.add_argument(
        "--player-season-start",
        type=int,
        default=CURRENT_SEASON['year'] - 1,
        help="Season start year for player data (e.g., 2024)",
    )
    parser.add_argument(
        "--recent-minutes-days",
        type=int,
        default=14,
        help="Lookback window for recent minutes (days)",
    )
    parser.add_argument(
        "--injury-source",
        default=MODEL_PARAMS.get('injury_source', 'auto'),
        choices=["auto", "nba", "rotowire", "none"],
        help="Injury report source for minutes adjustments",
    )
    parser.add_argument(
        "--injury-url",
        default="",
        help="Optional URL for NBA injury report (CSV/JSON/HTML table)",
    )
    parser.add_argument(
        "--injury-file",
        default="",
        help="Optional local CSV/JSON file for injury report",
    )
    return parser.parse_args()


def combine_seasons(data_dir: Path, start_year: int, end_year: int, output_path: str = "") -> Path:
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = data_dir / f"{year + 1}_season_data.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            logger.warning("Missing season data file: %s", file_path)

    if not dfs:
        logger.warning("No season data files found to combine.")
        return data_dir / "all_seasons.csv"

    combined = pd.concat(dfs, ignore_index=True)
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values('date').reset_index(drop=True)

    if output_path:
        output = Path(output_path)
    else:
        output = data_dir / f"all_seasons_{start_year + 1}_{end_year + 1}.csv"
    combined.to_csv(output, index=False)
    logger.info("Combined seasons saved to %s (%d games)", output, len(combined))
    return output


def main():
    print("=" * 70)
    print("NBA Data Collection")
    print("=" * 70)
    print()

    # Initialize fetcher
    fetcher = NBADataFetcher()
    args = parse_args()

    seasons_to_fetch = list(range(args.start_year, args.end_year + 1))

    if not args.skip_games:
        print("This will fetch game data for the following seasons:")
        for year in seasons_to_fetch:
            print(f"  - {year}-{str(year + 1)[-2:]}")
        print()
    if args.player_data:
        print(f"Player data season: {args.player_season_start}-{str(args.player_season_start + 1)[-2:]}")
        print(f"Recent minutes lookback: {args.recent_minutes_days} days")
        print()

    if not args.yes:
        input("Press Enter to continue or Ctrl+C to cancel...")
        print()

    # Fetch each season
    results = {}
    if not args.skip_games:
        for season_start in seasons_to_fetch:
            season_label = f"{season_start}-{str(season_start + 1)[-2:]}"
            print(f"\nFetching {season_label}...")
            print("-" * 70)

            try:
                df = fetcher.fetch_season_data(season_start, force_refresh=args.force)

                if len(df) > 0:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.dropna(subset=['date'])
                    print(f"✓ Successfully fetched {len(df)} games")
                    if 'date' in df.columns and not df['date'].empty:
                        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                    print(f"  Sample: {df.iloc[0]['away_team']} @ {df.iloc[0]['home_team']}")
                    print(f"           {df.iloc[0]['away_score']}-{df.iloc[0]['home_score']}")
                    results[season_start + 1] = df
                else:
                    print(f"⚠ No data found for season {season_label}")

            except Exception as e:
                print(f"✗ Error fetching {season_label}: {e}")
                logger.error(f"Failed to fetch season {season_start}: {e}", exc_info=True)

    if args.player_data:
        print("\nFetching rosters and player logs...")
        print("-" * 70)
        try:
            rosters = fetcher.fetch_team_rosters(args.player_season_start, force_refresh=args.force)
            logs = fetcher.fetch_player_game_logs(args.player_season_start, force_refresh=args.force)
            minutes = fetcher.compute_recent_player_minutes(logs, lookback_days=args.recent_minutes_days)
            injuries = pd.DataFrame()
            if MODEL_PARAMS.get('player_injury_enabled', False) and args.injury_source != "none":
                injuries = fetcher.load_injury_report(
                    source=args.injury_source,
                    url=args.injury_url,
                    filepath=args.injury_file,
                    force_refresh=args.force,
                )
                if not injuries.empty:
                    minutes = fetcher.apply_injury_adjustments(
                        minutes,
                        injuries,
                        out_statuses=MODEL_PARAMS.get('injury_out_statuses', []),
                    )
            depth = fetcher.build_depth_charts(rosters, minutes)
            impact = fetcher.compute_player_impact(logs)
            team_adjustments = fetcher.build_team_player_adjustments(
                impact,
                minutes,
                scale=MODEL_PARAMS.get('player_adj_elo_scale', 60.0),
            )

            minutes_path = Path(DATA_PATHS['recent_minutes'])
            minutes.to_csv(minutes_path, index=False)
            depth_path = Path(DATA_PATHS['team_depth_charts'])
            depth.to_csv(depth_path, index=False)
            if not injuries.empty:
                injuries_path = Path(DATA_PATHS['injury_report'])
                injuries.to_csv(injuries_path, index=False)
            impact_path = Path(DATA_PATHS['player_impact'].format(year=args.player_season_start + 1))
            impact.to_csv(impact_path, index=False)
            adjustments_path = Path(DATA_PATHS['team_player_adjustments'])
            team_adjustments['date'] = pd.Timestamp.now().date()
            team_adjustments.to_csv(adjustments_path, index=False)

            print(f"✓ Saved recent minutes to {minutes_path}")
            print(f"✓ Saved depth charts to {depth_path}")
            if not injuries.empty:
                print(f"✓ Saved injury report to {injuries_path}")
            print(f"✓ Saved player impact to {impact_path}")
            print(f"✓ Saved team adjustments to {adjustments_path}")
        except Exception as e:
            print(f"✗ Error fetching player data: {e}")
            logger.error("Failed to fetch player data: %s", e, exc_info=True)

    combined = False

    # Summary
    print()
    print("=" * 70)
    print("Data Collection Summary")
    print("=" * 70)

    if results:
        total_games = sum(len(df) for df in results.values())
        print(f"\nSuccessfully collected {len(results)} seasons with {total_games} total games")
        print(f"\nData saved to:")

        for season_end, df in results.items():
            filepath = f"data/{season_end}_season_data.csv"
            print(f"  - {filepath} ({len(df)} games)")

    if args.combine_seasons:
        output = combine_seasons(fetcher.data_dir, args.start_year, args.end_year, args.combine_output)
        combined = output.exists()
        print(f"\nCombined seasons file: {output}")

    if args.player_data:
        print("\nPlayer data refreshed.")

    if results or args.player_data or combined:
        print("\n✓ Data collection complete!")
        return 0

    print("\n⚠ No data was collected. Please check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
