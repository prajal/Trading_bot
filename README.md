# Enhanced SuperTrend Trading Bot

An **enterprise-grade** automated trading system for the Indian stock market (NSE) using the SuperTrend indicator. This refactored version includes comprehensive risk management, advanced error handling, real-time performance monitoring, and production-ready security features.

---

## 🚀 **What's New in This Version**

### **Major Enhancements**
- ✅ **Advanced Risk Management** - Dynamic position sizing, circuit breakers, drawdown limits
- ✅ **Enterprise Security** - No hardcoded credentials, token management, backup systems
- ✅ **Real-time Monitoring** - Performance tracking, system health, execution quality
- ✅ **Comprehensive Error Handling** - Retry mechanisms, graceful degradation, auto-recovery
- ✅ **Production-Ready Architecture** - Modular design, structured logging, configuration management
- ✅ **Data Quality Assurance** - Market data validation, cleaning, integrity checks

### **Key Features**
- **Intelligent Position Sizing**: Kelly Criterion, volatility-adjusted, risk-based methods
- **Multi-layer Risk Controls**: Stop losses, trailing stops, daily limits, circuit breakers
- **Advanced SuperTrend Strategy**: Adaptive parameters, market regime detection, confidence scoring
- **Real-time Performance Analytics**: Trade analysis, execution quality, system metrics
- **Professional Logging System**: Structured logs, multiple outputs, session tracking
- **Comprehensive Monitoring**: Health checks, alerts, automatic diagnostics

---

## 📖 **Project Overview**

- **Strategy**: Enhanced SuperTrend indicator with adaptive parameters and confidence scoring
- **Execution**: NIFTY 50 for signal generation, NIFTYBEES for trade execution
- **Risk Management**: Multi-layered approach with dynamic position sizing and circuit breakers
- **Architecture**: Modular, production-ready system with comprehensive error handling

---

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Python 3.8 or higher
- Zerodha trading account with KiteConnect API access
- Minimum 4GB RAM (8GB recommended for optimal performance)
- Stable internet connection

### **1. Clone & Setup Environment**
```bash
# Clone repository
git clone https://github.com/yourusername/enhanced-supertrend-trading-bot.git
cd enhanced-supertrend-trading-bot

# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Configuration Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
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
python auth/enhanced_kite_auth.py

# This will:
# - Generate login URL
# - Guide you through authentication
# - Save secure tokens
# - Display account information
```

---

## 🎯 **Usage Guide**

### **Daily Trading Workflow**

#### **Morning Setup (9:00 AM)**
```bash
# 1. Authenticate and set trading amount
python auth/enhanced_kite_auth.py

# 2. Review configuration
python -c "from config.enhanced_settings import Settings; Settings.print_configuration_summary()"
```

#### **Start Trading (9:15 AM)**
```bash
# For Live Trading (PRODUCTION)
LIVE_TRADING_ENABLED=true DRY_RUN_MODE=false python main.py

# For Paper Trading (RECOMMENDED FOR TESTING)
python main.py  # Uses .env settings
```

#### **Monitor Performance**
```bash
# In another terminal - watch logs
tail -f logs/trading.log

# Monitor specific events
tail -f logs/trading.log | grep -E "(TRADE|SIGNAL|RISK|ERROR)"
```

### **CLI Commands**

#### **Authentication & Setup**
```bash
# Authenticate
python auth/enhanced_kite_auth.py

# Test connection
python -c "from auth.enhanced_kite_auth import KiteAuth; auth = KiteAuth(); print('✅ Connected' if auth.test_connection() else '❌ Failed')"
```

#### **Trading Operations**
```bash
# Start trading (dry run)
python main.py

# Start live trading (with confirmation)
LIVE_TRADING_ENABLED=true DRY_RUN_MODE=false python main.py
```

#### **Analysis & Monitoring**
```bash
# Analyze recent market data
python analyze_data.py

# Generate performance report
python -c "
from utils.performance_monitor import PerformanceMonitor
from config.enhanced_settings import Settings
pm = PerformanceMonitor(Settings.DATA_DIR, Settings.get_trading_config())
print(pm.export_performance_report())
"
```

---

## ⚙️ **Configuration Guide**

### **Trading Configuration**
```bash
# Position Sizing Methods
POSITION_SIZING_METHOD=risk_based      # Recommended
POSITION_SIZING_METHOD=volatility_adjusted
POSITION_SIZING_METHOD=kelly_criterion
POSITION_SIZING_METHOD=fixed

# Risk Management
RISK_PER_TRADE=0.02          # 2% risk per trade
MAX_DAILY_LOSS=0.05          # 5% max daily loss
MAX_DRAWDOWN=0.15            # 15% max drawdown
CIRCUIT_BREAKER_LOSS=0.10    # 10% circuit breaker
```

### **Strategy Configuration**
```bash
# SuperTrend Parameters
ATR_PERIOD=10                # ATR calculation period
FACTOR=3.0                   # SuperTrend factor
ADAPTIVE_MODE=true           # Enable adaptive parameters
CONFIDENCE_THRESHOLD=0.6     # Minimum signal confidence

# Advanced Features
MULTI_TIMEFRAME=false        # Multi-timeframe analysis
VOLUME_CONFIRMATION=false    # Volume-based confirmation
MOMENTUM_FILTER=false        # Momentum filtering
```

### **Safety Configuration**
```bash
# Trading Modes
LIVE_TRADING_ENABLED=false   # CRITICAL: Set to true only for live trading
DRY_RUN_MODE=true           # Safe mode - no real orders

# Limits
MAX_ORDERS_PER_DAY=50       # Daily order limit
MAX_POSITIONS=1             # Maximum simultaneous positions

# Stop Loss Methods
STOP_LOSS_METHOD=atr_based  # atr_based, fixed, percentage
TRAILING_STOP=false         # Enable trailing stops
```

---

## 🛡️ **Risk Management Features**

### **Position Sizing**
- **Risk-Based**: Calculates position size based on defined risk per trade
- **Volatility-Adjusted**: Adjusts size based on market volatility
- **Kelly Criterion**: Optimal position sizing based on win rate and risk/reward
- **Fixed**: Traditional fixed position sizing with leverage

### **Risk Controls**
```
✅ Dynamic Stop Losses (Fixed, ATR-based, Percentage)
✅ Trailing Stops with customizable distance
✅ Daily Loss Limits with automatic shutdown
✅ Maximum Drawdown Limits
✅ Circuit Breakers for emergency stops
✅ Position Correlation Limits
✅ Real-time Risk Level Monitoring
```

### **Emergency Features**
- **Circuit Breaker**: Automatically stops trading on excessive losses
- **Emergency Stop**: Manual override to close all positions
- **Position Sync**: Handles external position changes gracefully
- **Auto Square-off Protection**: Exits positions before market close

---

## 📊 **Monitoring & Analytics**

### **Real-time Monitoring**
- **Performance Metrics**: P&L, win rate, Sharpe ratio, drawdown
- **System Health**: CPU, memory, network usage
- **Execution Quality**: Slippage, fill rates, order success rates
- **Risk Metrics**: Current exposure, risk level, circuit breaker status

### **Logging System**
```
📁 logs/
├── trading.log          # Main trading activity
├── errors.log           # Error events only
├── performance.log      # Performance metrics
└── structured.jsonl     # Machine-readable logs
```

### **Performance Reports**
```bash
# Export comprehensive report
python -c "
from utils.performance_monitor import PerformanceMonitor
from config.enhanced_settings import Settings
pm = PerformanceMonitor(Settings.DATA_DIR, Settings.get_trading_config())
report_path = pm.export_performance_report()
print(f'Report saved: {report_path}')
"
```

---

## 🔧 **Advanced Features**

### **Market Regime Detection**
The system automatically detects market conditions and adjusts strategy parameters:
- **Conservative**: Low volatility environments
- **Aggressive**: Strong trending markets
- **Volatile**: High volatility periods
- **Default**: Normal market conditions

### **Data Quality Assurance**
- **OHLC Validation**: Ensures price data integrity
- **Statistical Outlier Detection**: Identifies and handles anomalous data
- **Data Cleaning**: Automatic correction of common data issues
- **Quality Scoring**: Rates data quality and adjusts strategy accordingly

### **Adaptive Strategy**
- **Dynamic Parameters**: ATR period and factor adjust based on market regime
- **Confidence Scoring**: Each signal receives a confidence score
- **Signal Filtering**: Low-confidence signals are filtered out
- **Multi-timeframe Analysis**: (Optional) Analyze multiple timeframes

---

## 📁 **Project Structure**

```
enhanced-supertrend-trading-bot/
├── 📄 main.py                     # Main trading application
├── 📁 auth/
│   └── enhanced_kite_auth.py      # Authentication with error handling
├── 📁 config/
│   └── enhanced_settings.py      # Configuration management
├── 📁 trading/
│   ├── enhanced_strategy.py      # Enhanced SuperTrend strategy
│   └── enhanced_executor.py      # Order execution with retries
├── 📁 utils/
│   ├── enhanced_logger.py        # Professional logging system
│   ├── enhanced_risk_manager.py  # Risk management system
│   ├── performance_monitor.py    # Performance tracking
│   └── market_data_validator.py  # Data quality assurance
├── 📁 data/                      # Runtime data (auto-created)
├── 📁 logs/                      # Log files (auto-created)
├── 📁 backups/                   # Configuration backups
├── 📄 .env                       # Your configuration (not in git)
├── 📄 .env.example              # Configuration template
└── 📄 requirements.txt          # Python dependencies
```

---

## 🚨 **Safety & Security**

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
DRY_RUN_MODE=true python main.py

# Test with small amounts first
ACCOUNT_BALANCE=1000.0

# Monitor logs actively
tail -f logs/trading.log

# Use circuit breakers
CIRCUIT_BREAKER_LOSS=0.05  # 5% max loss
```

---

## 🎯 **Trading Strategy Details**

### **SuperTrend Implementation**
- **TradingView Compatible**: Matches TradingView's SuperTrend calculations
- **Adaptive Parameters**: Automatically adjusts to market conditions
- **Signal Confidence**: Each signal includes confidence scoring
- **Multi-regime Support**: Different parameters for different market conditions

### **Entry/Exit Logic**
```
📈 BUY Signal:
   - SuperTrend changes from RED to GREEN
   - Confidence ≥ 60%
   - No existing position
   - Risk management approval

📉 SELL Signal:
   - SuperTrend changes from GREEN to RED
   - Confidence ≥ 60%
   - Existing position present
   - Risk management approval
```

### **Risk Management Integration**
- **Position Size**: Calculated based on account balance and risk tolerance
- **Stop Loss**: Multiple methods (fixed, ATR-based, percentage)
- **Take Profit**: Optional profit-taking levels
- **Time-based Exits**: Pre-market close exits to avoid auto square-off

---

## 📈 **Performance Optimization**

### **System Requirements**
- **Minimum**: 2GB RAM, 1 CPU core, 1GB disk space
- **Recommended**: 8GB RAM, 4 CPU cores, 10GB disk space
- **Network**: Stable internet with low latency to exchanges

### **Performance Tuning**
```bash
# Optimize for speed
CHECK_INTERVAL=15           # Faster checking (seconds)
MIN_ORDER_INTERVAL=0.5      # Faster order placement

# Optimize for stability
MAX_CONNECTION_RETRIES=5    # More retries
BACKOFF_MULTIPLIER=3.0      # Longer backoff
```

### **Monitoring Performance**
```bash
# Check system resources
python -c "
from utils.performance_monitor import PerformanceMonitor
from config.enhanced_settings import Settings
pm = PerformanceMonitor(Settings.DATA_DIR, Settings.get_trading_config())
print(pm.get_real_time_dashboard_data())
"
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Authentication Problems**
```bash
# Issue: "Connection test failed"
# Solution:
python auth/enhanced_kite_auth.py  # Re-authenticate
# Check API credentials in .env
# Verify market hours
```

#### **No Trades Executing**
```bash
# Issue: Bot runs but no trades
# Check:
# 1. Market is open
# 2. Sufficient volatility for signals
# 3. Risk management not blocking trades
# 4. Confidence threshold not too high

# Debug:
tail -f logs/trading.log | grep -E "(SIGNAL|CONFIDENCE|RISK)"
```

#### **High Resource Usage**
```bash
# Issue: Bot using too much CPU/RAM
# Solutions:
# 1. Increase CHECK_INTERVAL in .env
# 2. Disable non-essential features
# 3. Reduce logging level
SYSTEM_CHECK_INTERVAL=600   # Check every 10 minutes
LOG_LEVEL=WARNING          # Reduce log verbosity
```

#### **Data Quality Issues**
```bash
# Issue: "Data validation failed"
# Solution:
# Check data quality score
python -c "
from utils.market_data_validator import MarketDataValidator
validator = MarketDataValidator()
# Analyze your data...
"
```

### **Log Analysis**
```bash
# Error analysis
grep -E "ERROR|CRITICAL" logs/trading.log

# Performance analysis
grep "PERF:" logs/performance.log

# Trade analysis
grep "TRADE:" logs/trading.log
```

### **Emergency Procedures**
```bash
# Emergency stop (if bot is running)
pkill -f "python main.py"

# Check positions manually
python -c "
from auth.enhanced_kite_auth import KiteAuth
auth = KiteAuth()
kite = auth.get_kite_instance()
if kite:
    positions = kite.positions()
    print(positions)
"
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Fork repository
git clone https://github.com/yourusername/enhanced-supertrend-trading-bot.git
cd enhanced-supertrend-trading-bot

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

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ **Disclaimers**

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

## 🙏 **Credits & Acknowledgments**

- **Zerodha** for the KiteConnect API
- **SuperTrend Indicator** creators
- **Python Community** for excellent libraries
- **Open Source Contributors** who make projects like this possible

---

## 📞 **Support**

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

## 🚀 **What's Next?**

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

*For the latest updates and detailed documentation, visit our [GitHub Repository](https://github.com/yourusername/enhanced-supertrend-trading-bot)*