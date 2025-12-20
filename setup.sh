#!/bin/bash

# Setup script for NBA Predictions project
# Modernized for 2025

echo "=========================================="
echo "NBA Predictions - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Python version is 3.10 or higher
python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Python 3.10 or higher is required"
    echo "Current version: $(python3 --version)"
    echo ""
    echo "Please install Python 3.10+ using one of these methods:"
    echo "  - Homebrew: brew install python@3.11"
    echo "  - pyenv: pyenv install 3.11.6 && pyenv global 3.11.6"
    echo "  - Download from: https://www.python.org/downloads/"
    exit 1
fi
echo "✓ Python version is compatible"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "WARNING: Not running in a virtual environment!"
    echo "It's recommended to create one:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate  # On macOS/Linux"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p logs
mkdir -p perf
mkdir -p standalone

# Check if data files exist
echo ""
echo "Checking for data files..."
if [ ! -f "data/raptors_player_stats.csv" ]; then
    echo "WARNING: Historical RAPTOR data not found"
    echo "Downloading from FiveThirtyEight GitHub..."
    curl -o data/raptors_player_stats.csv \
        https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv
fi

# Download ELO data if needed
if [ ! -f "data/538_nba_elo.csv" ]; then
    echo "WARNING: Historical ELO data not found"
    echo "Downloading from FiveThirtyEight GitHub..."
    curl -o data/538_nba_elo.csv \
        https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv
fi

# Test imports
echo ""
echo "Testing Python imports..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
from nba_api.stats.static import teams
print('✓ All core dependencies imported successfully')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "WARNING: Some imports failed. Check requirements.txt"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review config.py for current season parameters"
echo "2. Run 'python3 -m pytest' to test the installation"
echo "3. Check data/ directory for required data files"
echo ""
echo "To get started with predictions:"
echo "  python3 -m code.raptor_script_utils_v3"
echo ""
