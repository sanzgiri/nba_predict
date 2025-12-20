#!/usr/bin/env python3
"""
Test script to verify Priority 1 modernization is working
Run this after setup.sh to validate the installation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    errors = []
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('statsmodels', 'statsmodels'),
        ('nba_api', 'nba-api'),
        ('streamlit', 'streamlit'),
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),
    ]
    
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
            errors.append(package_name)
    
    if errors:
        print(f"\n❌ Failed to import: {', '.join(errors)}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n✅ All imports successful!")
    
    assert not errors, f"Failed to import: {', '.join(errors)}"


def test_local_modules():
    """Test that local modules can be imported"""
    print("\nTesting local modules...")
    errors = []
    
    modules = [
        ('config', 'config.py'),
        ('utils', 'utils.py'),
    ]
    
    for module_name, filename in modules:
        try:
            module = __import__(module_name)
            print(f"  ✓ {filename}")
            
            # Test specific imports
            if module_name == 'config':
                assert hasattr(module, 'MODEL_PARAMS')
                assert hasattr(module, 'TEAM_ABBREVIATION_MAPPINGS')
                print(f"    - MODEL_PARAMS loaded")
            elif module_name == 'utils':
                assert hasattr(module, 'setup_logging')
                assert hasattr(module, 'retry_on_failure')
                print(f"    - Utility functions available")
                
        except (ImportError, AssertionError) as e:
            print(f"  ✗ {filename}: {e}")
            errors.append(filename)
    
    if errors:
        print(f"\n❌ Failed to load: {', '.join(errors)}")
    else:
        print("\n✅ All local modules loaded!")
    
    assert not errors, f"Failed to load: {', '.join(errors)}"


def test_data_files():
    """Test that required data files exist"""
    print("\nTesting data files...")
    
    data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print("  ✗ data/ directory not found")
        print("Run: mkdir -p data")
        assert False, "data/ directory not found"
    
    required_files = [
        'raptors_player_stats.csv',  # Historical RAPTOR data
    ]
    
    optional_files = [
        '538_nba_elo.csv',           # Historical ELO data
        'team_depth_charts.csv',      # Team depth charts
        'nba_locations_upd.csv',      # Location data
    ]
    
    missing_required = []
    missing_optional = []
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (missing)")
            missing_required.append(filename)
    
    for filename in optional_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ○ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ○ {filename} (optional, not found)")
            missing_optional.append(filename)
    
    if missing_required:
        print(f"\n❌ Missing required files: {', '.join(missing_required)}")
        print("Run: ./setup.sh to download")
    else:
        print("\n✅ Required data files present!")
        if missing_optional:
            print(f"ℹ️  Optional files missing: {', '.join(missing_optional)}")
    
    assert not missing_required, f"Missing required files: {', '.join(missing_required)}"


def test_directories():
    """Test that required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = ['data', 'logs', 'perf']
    
    for dirname in required_dirs:
        dirpath = project_root / dirname
        if dirpath.exists():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ✗ {dirname}/ (creating...)")
            dirpath.mkdir(exist_ok=True)
            print(f"  ✓ {dirname}/ (created)")
    
    print("\n✅ All directories ready!")


def test_nba_api():
    """Test NBA API connectivity"""
    print("\nTesting NBA API connectivity...")
    
    try:
        from nba_api.stats.static import teams
        all_teams = teams.get_teams()
        print(f"  ✓ NBA API accessible")
        print(f"  ✓ Found {len(all_teams)} teams")
        
        # Show a sample team
        sample = all_teams[0]
        print(f"  ✓ Sample: {sample['full_name']} ({sample['abbreviation']})")
        
    except Exception as e:
        print(f"  ✗ NBA API error: {e}")
        print("  ℹ️  This is okay - the API may be rate-limited")
        # Don't fail the test for API connectivity issues


def test_config_values():
    """Test that config values are reasonable"""
    print("\nTesting configuration values...")
    
    try:
        from config import MODEL_PARAMS, CURRENT_SEASON
        
        # Check model params
        assert 0 < MODEL_PARAMS['raptor_slope'] < 2
        assert 90 < MODEL_PARAMS['avg_offensive_rating'] < 120
        assert 90 < MODEL_PARAMS['avg_pace'] < 110
        print(f"  ✓ Model parameters in valid range")
        
        # Check season
        assert CURRENT_SEASON['year'] >= 2025
        print(f"  ✓ Current season: {CURRENT_SEASON['year']}")
        
    except Exception as e:
        print(f"  ✗ Config validation error: {e}")
        assert False, f"Config validation failed: {e}"


def main():
    """Run all tests"""
    print("=" * 60)
    print("NBA Predictions - Priority 1 Validation Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    try:
        test_imports()
        results.append(("Package Imports", True))
    except AssertionError:
        results.append(("Package Imports", False))
    
    try:
        test_local_modules()
        results.append(("Local Modules", True))
    except AssertionError:
        results.append(("Local Modules", False))
    
    try:
        test_directories()
        results.append(("Directories", True))
    except AssertionError:
        results.append(("Directories", False))
    
    try:
        test_data_files()
        results.append(("Data Files", True))
    except AssertionError:
        results.append(("Data Files", False))
    
    try:
        test_config_values()
        results.append(("Config Values", True))
    except AssertionError:
        results.append(("Config Values", False))
    
    try:
        test_nba_api()
        results.append(("NBA API", True))
    except AssertionError:
        results.append(("NBA API", False))
    
    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Priority 1 modernization complete.")
        print("\nNext steps:")
        print("  1. Review MODERNIZATION_GUIDE.md for details")
        print("  2. Start working on Priority 2 (Data Collection)")
        print("  3. Check config.py and adjust for current season")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        print("Run './setup.sh' to fix common issues.")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
