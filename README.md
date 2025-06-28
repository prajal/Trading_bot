# SuperTrend Trading Bot

An automated trading system for the Indian stock market (NSE) using the SuperTrend indicator. The bot supports live trading, backtesting, and data analysis, with robust risk management and flexible data handling. It is designed for use with Zerodha Kite Connect API and supports dynamic capital management, real-time position sync, and MIS leverage.

---

## 📖 Project Overview

- **Strategy**: Uses the SuperTrend indicator for buy/sell signals.
- **Dual Data Approach**: (Live/Analysis) Uses NIFTY 50 data for signal generation and NIFTYBEES for trade execution.
- **Multi-Strategy System**: Supports multiple strategies including Enhanced SuperTrend and Bulletproof SuperTrend.
- **Risk Management**: Comprehensive risk controls with dynamic position sizing and circuit breakers.
- **Backtesting**: Full backtesting capabilities with performance analysis and visualization.

---

## 🚀 Key Features

### **Trading Features**
- ✅ **Multi-Strategy Support**: Enhanced SuperTrend, Bulletproof SuperTrend, and extensible framework
- ✅ **Dual Data Strategy**: NIFTY 50 for signals, NIFTYBEES for execution
- ✅ **Live & Paper Trading**: Safe paper trading mode with live trading option
- ✅ **Real-time Monitoring**: Performance tracking, system health, execution quality
- ✅ **Advanced Risk Management**: Dynamic position sizing, stop losses, circuit breakers

### **Analysis & Backtesting**
- ✅ **Comprehensive Backtesting**: Multi-strategy backtesting with detailed reports
- ✅ **Historical Data Download**: Download data for any NSE instrument
- ✅ **Performance Analytics**: Win rate, Sharpe ratio, drawdown analysis
- ✅ **Data Validation**: Quality checks and cleaning for reliable results

### **CLI Interface**
- ✅ **Unified CLI**: Single interface for all operations (`python cli.py`)
- ✅ **Strategy Management**: List, compare, and select strategies
- ✅ **Easy Configuration**: Interactive setup and validation
- ✅ **Comprehensive Help**: Built-in help and usage examples

---

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+ (Python 3.10-3.11 recommended for best compatibility)
- Zerodha trading account with KiteConnect API access
- Stable internet connection

### **1. Clone & Setup Environment**
```bash
# Clone repository
git clone https://github.com/prajal/Trading_bot.git
cd Trading_bot

# Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Configuration Setup**
```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # or use your preferred editor
```

**Required .env Configuration:**
```bash
# API Credentials (REQUIRED)
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here

# Trading Configuration
ACCOUNT_BALANCE=10000.0
RISK_PER_TRADE=0.02
POSITION_SIZING_METHOD=risk_based

# Safety Settings (IMPORTANT)
LIVE_TRADING_ENABLED=false
DRY_RUN_MODE=true
```

### **3. Authentication**
```bash
# Authenticate with Zerodha
python cli.py auth

# This will:
# - Generate login URL
# - Guide you through authentication
# - Save secure tokens
# - Display account information
```

---

## 🎯 CLI Usage Guide

The bot provides a unified CLI interface for all operations:

### **Authentication & Setup**
```bash
# Authenticate with Zerodha
python cli.py auth

# Test connection and configuration
python cli.py test

# List available strategies
python cli.py strategies
```

### **Trading Operations**
```bash
# Start paper trading (default)
python cli.py trade

# Start live trading (with confirmation)
python cli.py trade --live

# Use specific strategy
python cli.py trade --strategy=bullet

# Use specific instrument
python cli.py trade --instrument=NIFTYBEES
```

### **Backtesting**
```bash
# List available strategies for backtesting
python cli.py backtest --list-strategies

# Run backtest with default strategy
python cli.py backtest --csv historical_data/NIFTYBEES_historical_data.csv

# Run backtest with bulletproof strategy
python cli.py backtest --csv historical_data/NIFTYBEES_historical_data.csv --strategy=bullet

# Run backtest with enhanced strategy
python cli.py backtest --csv historical_data/NIFTYBEES_historical_data.csv --strategy=enhanced
```

### **Performance Analysis**
```bash
# Analyze trading performance
python cli.py performance

# Analyze specific time period
python cli.py performance --days=60
```

### **Help & Information**
```bash
# Show all available commands
python cli.py --help

# Show help for specific command
python cli.py trade --help
python cli.py backtest --help
```

---

## 📊 Script Usage Guide

### **1. CLI Interface (`cli.py`)**
**Purpose**: Main interface for all trading operations
```bash
# Basic usage
python cli.py [command] [options]

# Examples
python cli.py auth                    # Authenticate
python cli.py trade                   # Start paper trading
python cli.py trade --live            # Start live trading
python cli.py backtest --csv data.csv # Run backtest
python cli.py strategies              # List strategies
```

### **2. Data Analysis (`analyze_data.py`)**
**Purpose**: Analyze market data and generate insights
```bash
# Analyze current market data
python analyze_data.py

# Features:
# - Dual data analysis (NIFTY 50 + NIFTYBEES)
# - SuperTrend signal analysis
# - Market condition assessment
# - Performance metrics calculation
```

### **3. Historical Data Download (`historical_data/historical.py`)**
**Purpose**: Download historical data for any NSE instrument
```bash
# Download NIFTYBEES data
python historical_data/historical.py --symbol NIFTYBEES --years 2

# Download NIFTY 50 data
python historical_data/historical.py --symbol NIFTY50 --years 1

# Download any stock
python historical_data/historical.py --symbol RELIANCE --years 1 --interval minute
```

### **4. Backtesting (`backtest/backtest_strategy.py`)**
**Purpose**: Run comprehensive strategy backtests
```bash
# Basic backtest
python backtest/backtest_strategy.py --csv historical_data/NIFTYBEES_historical_data.csv

# Multi-strategy backtest
python backtest/backtest_strategy.py --csv data.csv --strategy=bullet
python backtest/backtest_strategy.py --csv data.csv --strategy=enhanced

# List available strategies
python backtest/backtest_strategy.py --list-strategies

# Features:
# - Multi-strategy support
# - Performance metrics calculation
# - Trade analysis and visualization
# - Risk-adjusted returns
```

### **5. Trading Strategy (`trading/strategy.py`)**
**Purpose**: Core SuperTrend strategy implementation
```python
# The strategy provides:
# - SuperTrend calculation (TradingView compatible)
# - Signal generation (BUY/SELL/HOLD)
# - Confidence scoring
# - Market regime detection
# - Adaptive parameters
```

---

## 📁 Project Structure

```
Trading_bot/
├── 📄 cli.py                           # Main CLI interface
├── 📄 analyze_data.py                  # Data analysis script
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .env                             # Configuration (create from .env.example)
│
├── 📁 auth/                            # Authentication
│   ├── __init__.py
│   └── enhanced_kite_auth.py          # Zerodha authentication
│
├── 📁 config/                          # Configuration management
│   ├── __init__.py
│   ├── enhanced_settings.py           # Enhanced settings
│   └── improved_settings.py           # Improved settings
│
├── 📁 trading/                         # Trading components
│   ├── __init__.py
│   ├── enhanced_executor.py           # Order execution
│   ├── enhanced_strategy.py           # Enhanced strategy
│   └── strategies/                    # Multi-strategy system
│       ├── __init__.py
│       ├── base_strategy.py           # Base strategy class
│       ├── strategy_factory.py        # Strategy factory
│       ├── enhanced_supertrend_wrapper.py
│       └── bulletproof_supertrend_strategy.py
│
├── 📁 backtest/                        # Backtesting
│   └── backtest_strategy.py           # Multi-strategy backtester
│
├── 📁 historical_data/                 # Historical data
│   └── historical.py                  # Data downloader
│
├── 📁 utils/                           # Utilities
│   ├── __init__.py
│   ├── enhanced_logger.py             # Logging system
│   ├── enhanced_risk_manager.py       # Risk management
│   ├── performance_monitor.py         # Performance tracking
│   ├── market_data_validator.py       # Data validation
│   ├── exceptions.py                  # Custom exceptions
│   └── validators.py                  # Data validators
│
├── 📁 data/                            # Runtime data (auto-created)
│   ├── current_session.json
│   ├── execution_history.json
│   ├── performance_metrics.json
│   └── trades_history.json
│
├── 📁 logs/                            # Log files (auto-created)
│   ├── trading.log
│   ├── errors.log
│   └── performance.log
│
├── 📁 backups/                         # Configuration backups
│   └── tokens_backup_*.json
│
└── 📁 historical_data/                 # Downloaded data files
    ├── NIFTYBEES_historical_data.csv
    ├── RELIANCE_historical_data.csv
    └── ...
```

---

## 🎯 Strategy Details

### **Available Strategies**

#### **1. Enhanced SuperTrend (Default)**
- **Key**: `enhanced`
- **Description**: Your current enhanced SuperTrend strategy with adaptive parameters
- **Features**: Adaptive ATR, confidence scoring, market regime detection
- **Usage**: `python cli.py trade` (default)

#### **2. Bulletproof SuperTrend**
- **Key**: `bullet`
- **Description**: Rock-solid SuperTrend with advanced quality filtering and risk management
- **Features**: Quality filtering, dynamic position sizing, multi-level targets
- **Usage**: `python cli.py trade --strategy=bullet`

### **Dual Data Strategy**
The system uses a dual data approach for optimal performance:

1. **NIFTY 50 Index Data**: Used for signal generation
   - More liquid and representative of market conditions
   - Better for SuperTrend calculation
   - Provides cleaner signals

2. **NIFTYBEES Data**: Used for trade execution
   - Actual trading instrument
   - Real price data for order placement
   - Accurate P&L calculation

### **Signal Generation Process**
```
1. Fetch NIFTY 50 data for signal calculation
2. Calculate SuperTrend indicators
3. Generate BUY/SELL signals with confidence scores
4. Apply risk management filters
5. Execute trades on NIFTYBEES using signal direction
6. Monitor and manage positions
```

---

## 📈 Backtesting Guide

### **Running Backtests**
```bash
# Basic backtest with default strategy
python cli.py backtest --csv historical_data/NIFTYBEES_historical_data.csv

# Backtest with specific strategy
python cli.py backtest --csv data.csv --strategy=bullet

# List available strategies
python cli.py backtest --list-strategies
```

### **Backtest Output**
Each backtest generates:
- **Performance Report**: `backtest_report.txt`
- **Trade Details**: `backtest_trades.csv`
- **Visualization**: `backtest_charts.png`

### **Performance Metrics**
- **Total Return**: Overall percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Trade Statistics**: Average trade duration, P&L distribution

### **Data Requirements**
CSV files must contain these columns:
- `date`: Date in YYYY-MM-DD format
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

---

## 📊 Historical Data Management

### **Downloading Data**
```bash
# Download NIFTYBEES data (2 years)
python historical_data/historical.py --symbol NIFTYBEES --years 2

# Download NIFTY 50 data
python historical_data/historical.py --symbol NIFTY50 --years 1

# Download any stock
python historical_data/historical.py --symbol RELIANCE --years 1 --interval minute
```

### **Data Cleaning**
```bash
# Clean data to remove zero-volume candles
python clean.py

# This removes:
# - Zero volume candles
# - Duplicate entries
# - Missing data points
# - Outliers
```

### **Data Validation**
The system automatically validates data quality:
- **Volume Check**: Warns about zero-volume candles
- **Price Validation**: Checks for price anomalies
- **Data Continuity**: Ensures no missing periods
- **Quality Scoring**: Rates overall data quality
- **Automatic Cleaning**: Removes duplicates, missing values, and outliers during processing

### **Data Quality Issues**
```bash
# Issue: "Data validation failed"
# Solution:
# Data cleaning is handled automatically by the system
# If issues persist, re-download clean data:
python historical_data/historical.py --symbol NIFTYBEES --years 1
```

---

## ⚙️ Configuration Guide

### **Trading Configuration**
```bash
# Account Settings
ACCOUNT_BALANCE=10000.0              # Trading capital
RISK_PER_TRADE=0.02                  # 2% risk per trade
POSITION_SIZING_METHOD=risk_based    # Position sizing method

# Risk Management
MAX_DAILY_LOSS=0.05                  # 5% max daily loss
MAX_DRAWDOWN=0.15                    # 15% max drawdown
CIRCUIT_BREAKER_LOSS=0.10            # 10% circuit breaker
```

### **Strategy Configuration**
```bash
# SuperTrend Parameters
ATR_PERIOD=10                        # ATR calculation period
FACTOR=3.0                           # SuperTrend factor
ADAPTIVE_MODE=true                   # Enable adaptive parameters
CONFIDENCE_THRESHOLD=0.6             # Minimum signal confidence

# Advanced Features
MULTI_TIMEFRAME=false                # Multi-timeframe analysis
VOLUME_CONFIRMATION=false            # Volume-based confirmation
```

### **Safety Configuration**
```bash
# Trading Modes
LIVE_TRADING_ENABLED=false           # CRITICAL: Set to true only for live trading
DRY_RUN_MODE=true                    # Safe mode - no real orders

# Limits
MAX_ORDERS_PER_DAY=50                # Daily order limit
MAX_POSITIONS=1                      # Maximum simultaneous positions
```

---

## 🛡️ Risk Management

### **Position Sizing Methods**
- **Risk-Based**: Calculates position size based on defined risk per trade
- **Volatility-Adjusted**: Adjusts size based on market volatility
- **Kelly Criterion**: Optimal position sizing based on win rate and risk/reward
- **Fixed**: Traditional fixed position sizing with leverage

### **Risk Controls**
- ✅ **Dynamic Stop Losses**: Fixed, ATR-based, percentage methods
- ✅ **Trailing Stops**: Customizable trailing stop distances
- ✅ **Daily Loss Limits**: Automatic shutdown on excessive losses
- ✅ **Maximum Drawdown Limits**: Circuit breaker on drawdown
- ✅ **Position Correlation Limits**: Prevents over-concentration
- ✅ **Real-time Risk Monitoring**: Continuous risk assessment

### **Emergency Features**
- **Circuit Breaker**: Automatically stops trading on excessive losses
- **Emergency Stop**: Manual override to close all positions
- **Position Sync**: Handles external position changes gracefully
- **Auto Square-off Protection**: Exits positions before market close

---

## 📊 Monitoring & Analytics

### **Real-time Monitoring**
```bash
# Monitor trading logs
tail -f logs/trading.log

# Monitor specific events
tail -f logs/trading.log | grep -E "(TRADE|SIGNAL|RISK|ERROR)"

# Check performance
python cli.py performance
```

### **Performance Metrics**
- **P&L Tracking**: Real-time profit/loss monitoring
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Execution Quality**: Slippage, fill rates, order success

### **Logging System**
```
📁 logs/
├── trading.log          # Main trading activity
├── errors.log           # Error events only
├── performance.log      # Performance metrics
└── structured.jsonl     # Machine-readable logs
```

---

## 🔍 Troubleshooting

### **Common Issues**

#### **Authentication Problems**
```bash
# Issue: "Connection test failed"
# Solution:
python cli.py auth  # Re-authenticate
# Check API credentials in .env
# Verify market hours
```

#### **No Trades Executing**
```bash
# Issue: Bot runs but no trades
# Check:
# 1. Market is open (9:15 AM - 3:30 PM IST)
# 2. Sufficient volatility for signals
# 3. Risk management not blocking trades
# 4. Confidence threshold not too high

# Debug:
tail -f logs/trading.log | grep -E "(SIGNAL|CONFIDENCE|RISK)"
```

#### **Backtest Issues**
```bash
# Issue: "No trades executed"
# Check:
# 1. Data quality (data cleaning is automatic)
# 2. Strategy parameters
# 3. Market conditions in data period

# Debug:
python cli.py backtest --list-strategies  # Check available strategies
```

#### **Data Quality Issues**
```bash
# Issue: "Data validation failed"
# Solution:
# Data cleaning is handled automatically by the system
# If issues persist, re-download clean data:
python historical_data/historical.py --symbol NIFTYBEES --years 1
```

### **Emergency Procedures**
```bash
# Emergency stop (if bot is running)
pkill -f "python cli.py"

# Check positions manually
python cli.py test  # Test connection and view positions
```

---

## 🚨 Safety & Security

### **Security Features**
- ✅ **No Hardcoded Credentials**: All sensitive data in environment variables
- ✅ **Token Management**: Secure token storage with backup and recovery
- ✅ **Configuration Validation**: Comprehensive validation on startup
- ✅ **Access Control**: API key validation and session management

### **Safety Mechanisms**
- ✅ **Dry Run Mode**: Test all functionality without real trades
- ✅ **Paper Trading**: Simulate trading with virtual money
- ✅ **Position Verification**: Cross-check positions with broker
- ✅ **Order Validation**: Multi-layer order validation before execution

### **Best Practices**
```bash
# Always start with dry run
python cli.py trade  # Defaults to dry run

# Test with small amounts first
ACCOUNT_BALANCE=1000.0

# Monitor logs actively
tail -f logs/trading.log

# Use circuit breakers
CIRCUIT_BREAKER_LOSS=0.05  # 5% max loss
```

---

## 🤝 Contributing

### **Development Setup**
```bash
# Fork repository
git clone https://github.com/prajal/Trading_bot.git
cd Trading_bot

# Create development environment
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy
```

### **Code Quality**
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest
```

### **Contribution Guidelines**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** thoroughly with paper trading
4. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
5. **Push** to branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimers

### **Trading Risks**
- **No Guarantee**: Past performance doesn't guarantee future results
- **Risk of Loss**: Trading involves substantial risk of loss
- **Not Financial Advice**: This software is a tool, not investment advice
- **Your Responsibility**: You are fully responsible for your trading decisions

### **Software Disclaimers**
- **Use at Own Risk**: Software provided "as is" without warranties
- **No Liability**: Authors not liable for trading losses or software issues
- **Testing Required**: Thoroughly test before live trading
- **Compliance**: Ensure compliance with local regulations

---

## 🙏 Credits & Acknowledgments

- **Zerodha** for the KiteConnect API
- **SuperTrend Indicator** creators
- **Python Community** for excellent libraries
- **Open Source Contributors** who make projects like this possible

---

## 📞 Support

### **Getting Help**
1. **Documentation**: Check this README and code comments
2. **Logs**: Review logs in `logs/` directory for detailed information
3. **Issues**: Open GitHub issues with detailed information
4. **Community**: Join discussions in GitHub Discussions

### **Issue Reporting**
When reporting issues, please include:
- **Error message** (full traceback)
- **Log snippets** (relevant portions)
- **Configuration** (sanitized, no credentials)
- **Steps to reproduce**
- **System information**

---

## 🚀 What's Next?

### **Planned Features**
- **Portfolio Management**: Multi-instrument trading
- **Machine Learning**: AI-enhanced signal generation
- **Advanced Analytics**: Comprehensive backtesting suite
- **Web Dashboard**: Real-time monitoring interface
- **Mobile Alerts**: SMS/Push notifications
- **Database Integration**: Historical data storage

### **Roadmap**
- **Q1 2024**: Advanced backtesting and optimization
- **Q2 2024**: Web dashboard and mobile alerts
- **Q3 2024**: Multi-instrument support
- **Q4 2024**: Machine learning integration

---

**Remember: Always trade responsibly and never risk more than you can afford to lose! 🛡️**

---

For more details, see each script's help (`python <script> --help`) or the code comments.

**Happy Trading! 🚀**