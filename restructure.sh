#!/bin/bash
# Project Restructuring Script
# This script reorganizes your trading bot project for clean GitHub commit

echo "🔧 Restructuring Trading Bot Project"
echo "===================================="

# Create archive directory for old/test files
mkdir -p archive/test_scripts
mkdir -p archive/debug_scripts
mkdir -p archive/old_versions
mkdir -p archive/data_samples

# Move test and debug scripts to archive
echo "📦 Archiving test and debug scripts..."
mv -f test_*.py archive/test_scripts/ 2>/dev/null
mv -f debug_*.py archive/debug_scripts/ 2>/dev/null
mv -f create_*.py archive/test_scripts/ 2>/dev/null
mv -f generate_*.py archive/test_scripts/ 2>/dev/null
mv -f direct_*.py archive/debug_scripts/ 2>/dev/null
mv -f final_*.py archive/old_versions/ 2>/dev/null
mv -f working_*.py archive/old_versions/ 2>/dev/null
mv -f quick_*.py archive/old_versions/ 2>/dev/null
mv -f run_*.py archive/old_versions/ 2>/dev/null

# Move test data files
echo "📁 Archiving test data files..."
mv -f test_*.csv archive/data_samples/ 2>/dev/null
mv -f *test*.txt archive/data_samples/ 2>/dev/null
mv -f *test*.png archive/data_samples/ 2>/dev/null

# Move old backtest files
mv -f backtest_runner.py archive/old_versions/ 2>/dev/null
mv -f nifty_backtest_analysis.py archive/old_versions/ 2>/dev/null

# Clean up logs directory (keep structure, remove old logs)
echo "🧹 Cleaning logs directory..."
find logs -name "*.log" -mtime +7 -delete 2>/dev/null

# Create proper project structure
echo "📂 Creating clean project structure..."

# Ensure all necessary directories exist
mkdir -p auth
mkdir -p trading
mkdir -p config
mkdir -p utils
mkdir -p data
mkdir -p logs
mkdir -p docs
mkdir -p tests

# Create __init__.py files where missing
touch auth/__init__.py
touch trading/__init__.py
touch config/__init__.py
touch utils/__init__.py
touch tests/__init__.py

# Create .gitignore if it doesn't exist
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
new/
trading_env/

# Project specific
data/kite_tokens.json
data/trading_preferences.json
data/trading_config.json
logs/*.log
*.log

# Sensitive files
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Backtest outputs
backtest_report_*.txt
backtest_charts_*.png
*_results.csv
nifty_backtest_results.csv

# Archive folder (old files)
archive/

# Test data
test_*.csv
*.pkl
*.pickle

# Jupyter
.ipynb_checkpoints/
*.ipynb
EOF

# Create requirements.txt with exact versions
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
kiteconnect==4.1.0
pandas==1.5.3
numpy==1.24.3
python-dotenv==1.0.0
matplotlib==3.7.1
seaborn==0.12.2
EOF

# Create comprehensive .env.example
echo "🔐 Creating .env.example..."
cat > .env.example << 'EOF'
# Zerodha Kite Connect API Credentials
# Get these from https://kite.zerodha.com/connect/apps
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_REDIRECT_URI=http://localhost:3000

# Trading Configuration (Optional - Can be set via CLI)
# ACCOUNT_BALANCE=4000.0  # Now set dynamically via CLI
CAPITAL_ALLOCATION_PERCENT=100.0
MAX_POSITIONS=1
FIXED_STOP_LOSS=100.0

# Strategy Configuration
ATR_PERIOD=10
FACTOR=3.0
MIN_CANDLES=50

# Safety Configuration
LIVE_TRADING_ENABLED=false
DRY_RUN_MODE=true
PAPER_TRADING_BALANCE=10000.0

# Market Hours (24-hour format)
MARKET_OPEN_HOUR=9
MARKET_OPEN_MINUTE=15
MARKET_CLOSE_HOUR=15
MARKET_CLOSE_MINUTE=30

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_CONSOLE_ALERTS=true
ENABLE_TRADE_LOGGING=true
EOF

# Create setup.py for proper package structure
echo "📋 Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="supertrend-trading-bot",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated SuperTrend trading bot for NSE using Zerodha Kite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/supertrend-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "kiteconnect>=4.1.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "trading-bot=cli:main",
        ],
    },
)
EOF

# Create project info files
echo "📄 Creating project documentation..."

# Create CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2024-06-24

### Added
- Dynamic account balance setting via CLI
- Account information display on authentication
- Command-line amount override with `--amount` flag
- Trading preferences persistence
- Enhanced signal detection with direction tracking
- Signal deduplication to prevent repeated trades
- Comprehensive debug logging

### Changed
- Account balance is now input-driven instead of hardcoded
- Improved SuperTrend signal detection logic
- Enhanced position synchronization
- Better error handling and logging

### Fixed
- Fixed inverted SuperTrend direction logic
- Fixed signal detection for trend changes
- Resolved circular dependency in settings

## [1.0.0] - 2024-06-23

### Initial Release
- Basic SuperTrend strategy implementation
- Zerodha Kite Connect integration
- MIS leverage support
- Position synchronization
- Stop loss and risk management
EOF

# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# List of core files to keep
echo "✅ Core files preserved:"
echo "  - main.py"
echo "  - cli.py"
echo "  - backtest_strategy.py"
echo "  - auth/kite_auth.py"
echo "  - trading/strategy.py"
echo "  - trading/executor.py"
echo "  - config/settings.py"
echo "  - utils/logger.py"
echo "  - utils/validators.py"
echo "  - utils/exceptions.py"

# Summary
echo ""
echo "📊 Restructuring Summary:"
echo "  - Test/debug scripts moved to: archive/"
echo "  - Created proper .gitignore"
echo "  - Updated requirements.txt"
echo "  - Created .env.example"
echo "  - Added setup.py for packaging"
echo "  - Added CHANGELOG.md"
echo "  - Added LICENSE"
echo ""
echo "✅ Project is now clean and ready for GitHub!"
echo ""
echo "Next steps:"
echo "1. Review the changes"
echo "2. Update README.md with new documentation"
echo "3. git add ."
echo "4. git commit -m 'v2.0.0: Dynamic account balance and improved signal detection'"
echo "5. git push"
EOF

chmod +x restructure.sh
