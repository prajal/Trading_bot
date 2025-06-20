# SuperTrend Trading Bot

A fully automated Python trading bot that implements the SuperTrend strategy for NSE (Indian Stock Market) using Zerodha Kite Connect API. The bot provides position synchronization, MIS leverage support, and comprehensive safety features.

## 🚀 Features

- **SuperTrend Strategy**: Automated long-only trading based on SuperTrend indicator
- **MIS Leverage Support**: Intelligent quantity calculation with intraday leverage (up to 5x for ETFs)
- **Position Synchronization**: Real-time sync with broker positions to handle external trades/auto square-offs
- **Safety Features**: Multiple safety checks, dry-run mode, and emergency position reset
- **Real-time Monitoring**: Live P&L tracking and signal monitoring
- **Market Hours Awareness**: Automatic market open/close detection
- **Pre-close Exit**: Automatic position closure before market auto square-off

## 📈 Strategy Details

### SuperTrend Indicator
- **Period**: 10 (ATR period)
- **Factor**: 3.0 (multiplier)
- **Signals**: 
  - **BUY**: When SuperTrend turns GREEN (uptrend)
  - **SELL**: When SuperTrend turns RED (downtrend)

### Risk Management
- **Stop Loss**: Fixed ₹100 per trade
- **Position Size**: Dynamic calculation based on MIS leverage
- **Auto Exit**: Positions closed before 3:20 PM to avoid auto square-off
- **Single Position**: Maximum 1 position at a time

### Supported Instruments
The bot comes pre-configured with MIS leverage settings for:
- **NIFTYBEES**: 5x leverage (Primary trading instrument)
- **JUNIORBEES**: 5x leverage
- **BANKBEES**: 4x leverage
- **Large Cap Stocks**: 4x leverage (RELIANCE, TCS, HDFCBANK, etc.)
- **Default**: 3x leverage for other instruments

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Zerodha trading account
- Kite Connect API subscription (₹2000/month)

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd trading_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials**
   
   Create a `.env` file in the project root:
   ```env
   KITE_API_KEY=your_api_key_here
   KITE_API_SECRET=your_api_secret_here
   LIVE_TRADING_ENABLED=true
   DRY_RUN_MODE=false
   ```

4. **Adjust trading parameters** (Optional)
   
   Edit `config/settings.py`:
   ```python
   STRATEGY_PARAMS = {
       "account_balance": 4000.0,  # Your trading capital
       "capital_allocation_percent": 100.0,  # % of capital to use
       "fixed_stop_loss": 100.0,  # Stop loss in rupees
       # ... other parameters
   }
   ```

## 🎯 Usage

### Quick Start Commands

```bash
# 1. Test connection (most important daily command)
python cli.py test

# 2. Start trading (if test passes)
python cli.py trade

# 3. Authentication (only if tokens expired)
python cli.py auth

# 4. Emergency reset (if positions out of sync)
python cli.py reset
```

### Daily Trading Routine

#### Pre-Market (9:00 AM - 9:15 AM)
```bash
# Check if authentication is still valid
python cli.py test
```

If test **passes** ✅:
```bash
# Start trading immediately at 9:15 AM
python cli.py trade
```

If test **fails** ❌:
```bash
# Re-authenticate (takes 2-3 minutes)
python cli.py auth
# Follow browser authentication steps
# Then start trading
python cli.py trade
```

#### During Market Hours
```bash
# Monitor live trading in a separate terminal
tail -f logs/trading.log

# Filter for important events only
tail -f logs/trading.log | grep -E "(SIGNAL|TRADE|POSITION|BUY|SELL)"
```

## 🔐 Authentication Process

1. **Run authentication command**
   ```bash
   python cli.py auth
   ```

2. **Browser steps**:
   - Copy the URL shown in terminal
   - Open in browser and login to Zerodha
   - Authorize the application
   - Copy the **complete redirected URL**
   - Extract `request_token` from URL
   - Paste in terminal

3. **Verify authentication**
   ```bash
   python cli.py test
   ```

## 📊 Monitoring & Logs

### Log File Locations
- **Trading Logs**: `logs/trading.log`
- **Configuration**: `data/trading_config.json`
- **API Tokens**: `data/kite_tokens.json`

### Key Log Messages to Watch

**✅ Good Signs:**
```
Connected to Kite as: Your Name
🟢 LONG ENTRY SIGNAL DETECTED
✅ POSITION OPENED: 71 NIFTYBEES at ₹279.11
📉 POSITION CLOSED (SuperTrend Exit): P&L = ₹79.52
```

**⚠️ Warning Signs:**
```
❌ Connection test failed
ERROR: Failed to establish Kite session
Position out of sync
```

### Live Monitoring Commands
```bash
# Watch all logs
tail -f logs/trading.log

# Watch only trading signals
grep -i "signal" logs/trading.log

# Watch P&L updates
grep "P&L:" logs/trading.log

# Check last 20 log entries
tail -20 logs/trading.log
```

## 🛡️ Safety Features

### Built-in Protections
1. **Position Sync**: Continuous sync with broker to detect external trades
2. **Pre-close Exit**: Automatic exit before 3:20 PM auto square-off
3. **Stop Loss**: Fixed ₹100 stop loss per trade
4. **Single Position**: Only one position allowed at a time
5. **Market Hours**: Trading only during market hours (9:15 AM - 3:30 PM)

### Safety Modes
- **Dry Run Mode**: Test without real trades (`DRY_RUN_MODE=true`)
- **Paper Trading**: Virtual trading mode
- **Live Trading**: Real money trading (`LIVE_TRADING_ENABLED=true`)

## 🚨 Emergency Commands

| Problem | Command | Action |
|---------|---------|--------|
| Authentication expired | `python cli.py auth` | Re-authenticate |
| Position out of sync | `python cli.py reset` | Reset position tracking |
| Bot crashed | `python cli.py trade` | Restart bot |
| Orders not placing | `python cli.py test` | Check connection |

## 💰 Capital & Risk Management

### Default Configuration
- **Account Balance**: ₹4,000
- **Capital Allocation**: 100%
- **Max Leverage**: 5x (for NIFTYBEES)
- **Stop Loss**: ₹100 per trade
- **Max Positions**: 1

### Example Trade Calculation
With ₹4,000 capital and 5x leverage on NIFTYBEES at ₹279:
- **Buying Power**: ₹4,000 × 5 = ₹20,000
- **Quantity**: 71 shares (₹19,809 value)
- **Margin Required**: ₹3,962
- **Potential P&L**: ₹71 per ₹1 price move

## 🔧 Configuration

### Key Configuration Files

**`.env`** - API credentials and safety settings
```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_secret
LIVE_TRADING_ENABLED=true
DRY_RUN_MODE=false
```

**`config/settings.py`** - Trading parameters
```python
STRATEGY_PARAMS = {
    "atr_period": 10,           # SuperTrend ATR period
    "factor": 3.0,              # SuperTrend factor
    "account_balance": 4000.0,   # Your capital
    "fixed_stop_loss": 100.0,   # Stop loss amount
    "max_positions": 1,         # Max concurrent positions
    # ... more parameters
}
```

### Customizing for Different Instruments

To trade different instruments, modify `main.py`:
```python
# Default: NIFTY 50 -> NIFTYBEES
bot.run("256265", "2707457", "NIFTYBEES")

# For Bank NIFTY -> BANKBEES
bot.run("260105", "1364739", "BANKBEES")
```

## 📋 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**2. Authentication Failed**
```bash
python cli.py auth
# Follow browser steps carefully
```

**3. Position Out of Sync**
```bash
python cli.py reset
python cli.py trade
```

**4. No Trading Signals**
- Check if market is open
- Verify historical data availability
- Check SuperTrend calculation logs

**5. Orders Not Executing**
- Verify account balance
- Check if trading is enabled in Zerodha
- Ensure MIS product is available for the instrument

### Debug Mode
Add debug logging by modifying `utils/logger.py`:
```python
logger.setLevel(logging.DEBUG)  # Instead of INFO
```

## 📚 Technical Details

### Project Structure
```
trading_bot/
├── auth/              # Authentication modules
├── config/            # Configuration settings
├── data/              # Token and config storage
├── logs/              # Trading logs
├── trading/           # Strategy and execution
├── utils/             # Utilities and logging
├── cli.py             # Command-line interface
├── main.py            # Main trading bot
└── requirements.txt   # Dependencies
```

### Dependencies
- **kiteconnect**: Zerodha API client
- **pandas**: Data manipulation
- **numpy**: Numerical calculations
- **python-dotenv**: Environment variable management

## ⚠️ Important Disclaimers

1. **Risk Warning**: Trading involves substantial risk. Never trade with money you cannot afford to lose.

2. **No Guarantee**: Past performance does not guarantee future results. The bot may incur losses.

3. **Testing Required**: Always test with small amounts before full deployment.

4. **Market Conditions**: The strategy may not work in all market conditions.

5. **Regulatory Compliance**: Ensure compliance with local trading regulations.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all dependencies are correctly installed
4. Verify API credentials and permissions

## 📄 License

This project is for educational purposes. Use at your own risk.

---

**Happy Trading! 🚀**

Remember: The best strategy is the one you understand and can stick to consistently.