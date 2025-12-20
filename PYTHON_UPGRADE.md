# Python Upgrade Guide for macOS

This project requires **Python 3.10 or higher** (you currently have Python 3.9.6).

## Quick Install Options

### Option 1: Homebrew (Recommended for macOS)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11 (or latest)
brew install python@3.11

# Verify installation
python3.11 --version

# Make it the default (optional)
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Or use it directly
python3.11 -m venv venv
source venv/bin/activate
```

### Option 2: pyenv (Best for Managing Multiple Versions)

```bash
# Install pyenv via Homebrew
brew install pyenv

# Add to your shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Install Python 3.11
pyenv install 3.11.6

# Set as global default
pyenv global 3.11.6

# Verify
python3 --version  # Should show 3.11.6
```

### Option 3: Official Python.org Installer

1. Download from: https://www.python.org/downloads/
2. Choose Python 3.11.x or 3.12.x
3. Run the installer
4. Verify: `python3.11 --version`

## After Installing Python 3.10+

Once you have Python 3.10 or higher installed:

```bash
# Navigate to project
cd /Users/sanzgiri/nba_predictions

# Create virtual environment with new Python
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python version in venv
python --version  # Should be 3.11.x

# Now run setup
./setup.sh
```

## Using Multiple Python Versions

If you want to keep Python 3.9.6 alongside the new version:

```bash
# Option A: Use specific version directly
python3.11 -m venv venv
source venv/bin/activate

# Option B: Use pyenv to switch (if installed)
pyenv local 3.11.6  # For this project only
pyenv global 3.11.6  # System-wide
```

## Troubleshooting

### "python3.11: command not found"

If Homebrew installation doesn't make python3.11 available:

```bash
# Find where Homebrew installed it
brew --prefix python@3.11

# Add to PATH
export PATH="$(brew --prefix python@3.11)/bin:$PATH"

# Or create an alias
alias python3.11='/opt/homebrew/opt/python@3.11/bin/python3.11'
```

### "zsh: command not found: brew"

Install Homebrew first:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Check what Python versions you have

```bash
# List all Python installations
ls -la /usr/local/bin/python*
ls -la /opt/homebrew/bin/python*

# Check with pyenv (if installed)
pyenv versions
```

## Why Python 3.10+ is Required

- **pandas 2.0+**: Major performance improvements, better type hints
- **Modern libraries**: Latest versions of scikit-learn, streamlit, etc.
- **Security**: Newer versions have important security patches
- **Performance**: 10-15% faster than Python 3.9
- **Better error messages**: More helpful tracebacks

## Recommended Version

**Python 3.11.6** - Best balance of stability and modern features

## Quick Reference

```bash
# After installing Python 3.11:
cd /Users/sanzgiri/nba_predictions
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

That's it! Your new Python environment will be ready to go.
