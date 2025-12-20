# NBA Predictions - 2025 Modernization Guide

## What's Changed (Priority 1 Updates)

This document describes the infrastructure modernization completed to reactivate the NBA predictions repository after 5 years of inactivity.

### 1. ✅ Updated Dependencies

**New file:** `requirements.txt`

- Updated all packages to modern versions (2024/2025)
- Replaced deprecated packages with current equivalents
- Added new data sources (nba-api as backup to basketball-reference-web-scraper)
- All packages now compatible with Python 3.9+

**Key changes:**
- `scikit-learn>=1.3.0` (was using old `sklearn` import style)
- `streamlit>=1.28.0` (API breaking changes fixed)
- `nba-api>=1.4.0` (new alternative data source)
- `pandas>=2.0.0`, `numpy>=1.24.0` (modern versions)

### 2. ✅ Configuration Management

**New file:** `config.py`

Centralized configuration for:
- Data source URLs (FiveThirtyEight, NBA API)
- Team name mappings across different sources
- Model parameters (ELO, RAPTOR calibration)
- Current season settings
- API rate limiting settings
- Betting parameters (Kelly criterion)

**Benefits:**
- Single source of truth for all settings
- Easy to update for new seasons
- No hardcoded values scattered across files

### 3. ✅ Utility Functions & Error Handling

**New file:** `utils.py`

Added robust utilities:
- **Logging**: Automatic log file creation with timestamps
- **Retry logic**: Automatic retry with exponential backoff for API calls
- **Rate limiting**: Decorator to prevent API throttling
- **Caching**: Disk cache for expensive data fetches
- **Validation**: DataFrame structure validation
- **Error handling**: Graceful degradation instead of crashes

**Example usage:**
```python
from utils import retry_on_failure, cache_dataframe, logger

@retry_on_failure(max_retries=3, delay=2.0)
def fetch_data():
    # This will automatically retry on failure
    return api_call()
```

### 4. ✅ Modernized Core Functions

**New file:** `code/raptor_script_utils_v3.py`

Updated RAPTOR utilities with:
- Modern imports (handles missing packages gracefully)
- Better error messages and logging
- Fallback options when external APIs fail
- Type hints for better code documentation
- Integration with config.py and utils.py

**Key improvements:**
- No silent failures - all errors logged
- Graceful fallbacks when data sources unavailable
- Better code organization and documentation

### 5. ✅ Fixed Breaking Changes

**Fixed in:** `standalone/run.py`

- Updated `st.beta_set_page_config()` → `st.set_page_config()`
- This Streamlit API change would have broken the web app

### 6. ✅ Setup Automation

**New file:** `setup.sh`

One-command setup script that:
1. Checks Python version
2. Warns if not in virtual environment
3. Installs all dependencies
4. Creates required directories
5. Downloads historical data from FiveThirtyEight
6. Tests imports to verify installation

**Usage:**
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Run setup
./setup.sh
```

## Installation

### Quick Start

```bash
# 1. Clone or navigate to repository
cd nba_predictions

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or on Windows: venv\Scripts\activate

# 3. Run setup script
./setup.sh

# 4. Verify installation
python3 -c "import pandas, numpy, sklearn, nba_api; print('All good!')"
```

### Manual Installation

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs perf

# Download historical data
curl -o data/raptors_player_stats.csv \
  https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv
```

## What Still Needs Work

This update completes **Priority 1: Infrastructure Modernization**. Still needed:

### Priority 2: Data Collection (Next)
- [ ] Update data collection to use NBA API
- [ ] Collect 2020-2025 season data
- [ ] Update team rosters and depth charts
- [ ] Find alternative to RAPTOR ratings (frozen at 2022)

### Priority 3: Model Retraining
- [ ] Retrain models on recent data
- [ ] Recalibrate for post-COVID NBA (different pace, HCA)
- [ ] Update feature engineering

### Priority 4: Alternative Approaches
- [ ] Implement ELO-only predictions (simpler, self-contained)
- [ ] Consider Box Plus-Minus instead of RAPTOR
- [ ] Integrate with modern betting APIs

## Testing the Updates

### Test 1: Import All Modules
```python
from config import MODEL_PARAMS, TEAM_ABBREVIATION_MAPPINGS
from utils import setup_logging, retry_on_failure
from code.raptor_script_utils_v3 import get_abbreviation_mapping

print("✓ All imports successful")
```

### Test 2: Load Historical Data
```python
import pandas as pd

# Check if historical data downloaded
df_raptor = pd.read_csv('data/raptors_player_stats.csv')
print(f"✓ Loaded {len(df_raptor)} RAPTOR ratings")
print(f"  Seasons: {df_raptor['season'].min()} - {df_raptor['season'].max()}")
```

### Test 3: NBA API Access
```python
from nba_api.stats.static import teams

all_teams = teams.get_teams()
print(f"✓ NBA API working: {len(all_teams)} teams")
```

## File Structure After Updates

```
nba_predictions/
├── config.py                    # NEW: Centralized configuration
├── utils.py                     # NEW: Utility functions & error handling
├── requirements.txt             # NEW: Modern dependencies
├── setup.sh                     # NEW: Automated setup script
├── MODERNIZATION_GUIDE.md       # NEW: This file
│
├── code/
│   ├── raptor_script_utils_v3.py  # NEW: Modernized utilities
│   ├── raptor_script_utils_v2.py  # OLD: Keep for reference
│   ├── raptor_script_utils.py     # OLD: Original version
│   └── ... (other existing files)
│
├── data/
│   ├── raptors_player_stats.csv   # Downloaded by setup.sh
│   ├── 538_nba_elo.csv           # Downloaded by setup.sh
│   └── ... (other data files)
│
├── logs/                         # NEW: Auto-created for logging
├── standalone/
│   └── run.py                    # UPDATED: Fixed Streamlit API
│
└── ... (other existing files)
```

## Breaking Changes & Migration

### For Existing Code

If you have scripts using the old modules:

**Before:**
```python
from raptor_script_utils_v2 import get_injured, roster_minutes_injuries

# No error handling
df = get_injured()
```

**After:**
```python
from code.raptor_script_utils_v3 import get_injured, roster_minutes_injuries
from utils import logger

# With error handling
try:
    df = get_injured()
    logger.info(f"Loaded {len(df)} injured players")
except Exception as e:
    logger.error(f"Failed to get injuries: {e}")
    df = pd.DataFrame()  # Empty fallback
```

### Configuration Values

**Before:** Hardcoded in scripts
```python
raptor_slope = 0.84
avg_ort = 108.9
```

**After:** Import from config
```python
from config import MODEL_PARAMS

raptor_slope = MODEL_PARAMS['raptor_slope']
avg_ort = MODEL_PARAMS['avg_offensive_rating']
```

## Troubleshooting

### "Import Error: No module named 'X'"

Run the setup script:
```bash
./setup.sh
```

Or install manually:
```bash
pip install -r requirements.txt
```

### "basketball_reference_web_scraper not available"

This is expected - the package may have issues. The code will fall back to nba-api:
```python
# In raptor_script_utils_v3.py:
# Automatically detects and logs which API is available
```

### API Rate Limiting

If you hit rate limits:
```python
from config import RATE_LIMITS

# Check current limits
print(RATE_LIMITS)

# Adjust in config.py if needed
```

## Support & Contributing

### Getting Help
1. Check logs in `logs/` directory
2. Enable debug logging in config.py
3. Review error messages (now much more detailed!)

### Next Steps
See the original README.md for information about the prediction models and approach.

The modernization continues with Priority 2 (Data Collection) - stay tuned!

---

**Last Updated:** October 26, 2025
**Modernization Status:** Priority 1 Complete ✅
