###SuperTrend Trading Bot v2.0######
An automated algorithmic trading bot implementing the SuperTrend strategy for NSE (Indian Stock Market) using Zerodha Kite Connect API. Features dynamic capital management, real-time position synchronization, and MIS leverage support.

## 🚀 Key Features

- **SuperTrend Strategy**: Automated trading based on the SuperTrend indicator
- **Dynamic Capital Management**: Set your trading amount daily based on account balance
- **MIS Leverage Support**: Intelligent position sizing with intraday leverage (up to 5x)
- **Real-time Position Sync**: Handles external trades and auto square-offs gracefully
- **Risk Management**: Fixed stop loss, pre-market close exits, and position limits
- **Multiple Modes**: Dry run, paper trading, and live trading modes
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## 📋 Prerequisites

- Python 3.7 or higher
- Zerodha trading account
- Kite Connect API subscription (₹2000/month)
- Basic understanding of trading and risk management

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/supertrend-trading-bot.git
cd supertrend-trading-bot
```

### 2. Create Virtual Environment
```bash
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and add your Zerodha API credentials:
```env
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
```

## 📱 Daily Trading Workflow

### 1. Morning Authentication (9:00 AM)
```bash
python cli.py auth
```

This will:
- Verify your Kite connection
- Display your account balance
- Let you set your trading amount for the day

Example output:
```
💼 Account Information:
👤 Name: Your Name
💰 Available Cash: ₹50,000.00
📊 Net Worth: ₹75,000.00

🎯 Set Trading Amount for Today
Current available cash: ₹50,000.00
Last trading amount: ₹10,000.00
Enter trading amount (press Enter for ₹10,000.00): 15000

✅ Trading amount set to: ₹15,000.00
```

### 2. Start Trading (9:15 AM)

**For Live Trading:**
```bash
python cli.py trade --live
```

**For Testing (Dry Run):**
```bash
python cli.py trade
```

**With Custom Amount:**
```bash
python cli.py trade --live --amount=20000
```

### 3. Monitor Trading
```bash
# In another terminal, watch live logs
tail -f logs/trading.log

# Watch only important events
tail -f logs/trading.log | grep -E "(SIGNAL|POSITION|Order)"
```

## 📊 Strategy Details

### SuperTrend Parameters
- **ATR Period**: 10 (customizable via .env)
- **Multiplier**: 3.0 (customizable via .env)
- **Timeframe**: Daily candles on minute data

### Entry/Exit Rules
- **BUY**: When SuperTrend changes from RED (bearish) to GREEN (bullish)
- **SELL**: When SuperTrend changes from GREEN to RED
- **Stop Loss**: Fixed ₹100 per trade
- **Auto Exit**: Positions closed before 3:20 PM to avoid auto square-off

### Supported Instruments
Pre-configured MIS leverage for:
- **NIFTYBEES**: 5x leverage (default)
- **BANKBEES**: 4x leverage
- **JUNIORBEES**: 5x leverage
- **Large-cap stocks**: 3-4x leverage

## 🎮 Commands Reference

### Authentication & Setup
```bash
# Authenticate and set daily trading amount
python cli.py auth

# Test connection
python cli.py test
```

### Trading
```bash
# Start trading (dry run mode)
python cli.py trade

# Start live trading
python cli.py trade --live

# Trade with specific amount
python cli.py trade --live --amount=25000
```

### Analysis & Utilities
```bash
# Run backtest on recent data
python cli.py backtest

# Emergency position reset
python cli.py reset
```

## ⚙️ Configuration

### Essential Settings (.env)
```env
# API Credentials (Required)
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_secret

# Strategy Parameters (Optional)
ATR_PERIOD=10
FACTOR=3.0
FIXED_STOP_LOSS=100.0

# Safety Settings
LIVE_TRADING_ENABLED=false  # Set via --live flag
DRY_RUN_MODE=true           # Override with --live
```

### Dynamic Settings (Set via CLI)
- **Trading Amount**: Set daily via `cli.py auth` or `--amount` flag
- **Instrument**: Currently hardcoded to NIFTYBEES (can be modified in main.py)

## 📈 Risk Management

### Capital Allocation
- Set trading amount based on your risk tolerance
- Bot calculates position size using MIS leverage
- Never risks more than the specified stop loss per trade

### Safety Features
1. **Position Synchronization**: Detects external trades/square-offs
2. **Pre-close Exit**: Automatic exit before 3:20 PM
3. **Stop Loss**: Fixed ₹100 per trade (customizable)
4. **Single Position**: Only one position at a time
5. **Market Hours Only**: Trades only during 9:15 AM - 3:30 PM

## 🔍 Monitoring & Debugging

### Log Files
- **Location**: `logs/trading.log`
- **Rotation**: Daily
- **Level**: INFO (change in utils/logger.py for DEBUG)

### Key Log Messages
```
✅ Good Signs:
- "🟢 LONG ENTRY SIGNAL DETECTED"
- "Order placed: BUY X NIFTYBEES"
- "✅ POSITION OPENED"

⚠️ Warning Signs:
- "Position out of sync"
- "Failed to place order"
- "Connection test failed"
```

### Debug Mode
For detailed debugging, use the diagnostic script:
```bash
python debug_signals.py  # (if available in archive/)
```

## 📊 Backtesting

Run backtests on historical data:
```bash
python cli.py backtest
```

Or use the standalone backtester:
```bash
python backtest_strategy.py
```

## 🚨 Troubleshooting

### Common Issues

1. **"Connection test failed"**
   - Run `python cli.py auth` to re-authenticate
   - Check if market is open
   - Verify API credentials

2. **"No trades executed"**
   - Market might be in strong trend (no reversals)
   - Check if sufficient volatility exists
   - Review strategy parameters

3. **"Position out of sync"**
   - Run `python cli.py reset` to reset tracking
   - Check Zerodha terminal for actual positions

4. **Dynamic amount not updating**
   - Restart the bot after setting new amount
   - Check `data/trading_preferences.json`

## 📁 Project Structure

```
supertrend-trading-bot/
├── main.py              # Core trading bot logic
├── cli.py               # Command-line interface
├── backtest_strategy.py # Backtesting framework
├── auth/
│   └── kite_auth.py     # Zerodha authentication
├── trading/
│   ├── strategy.py      # SuperTrend implementation
│   └── executor.py      # Order execution logic
├── config/
│   └── settings.py      # Configuration management
├── utils/
│   ├── logger.py        # Logging utilities
│   ├── validators.py    # Data validation
│   └── exceptions.py    # Custom exceptions
├── data/                # Runtime data (gitignored)
├── logs/                # Trading logs (gitignored)
└── archive/             # Old test scripts (gitignored)
```

## 🔐 Security Notes

1. **Never commit** `.env` file with real credentials
2. **Keep API keys secure** - regenerate if exposed
3. **Use dry run mode** for testing
4. **Start with small amounts** when going live
5. **Monitor actively** during initial live trades

## 📈 Performance Disclaimer

- **No Guarantee**: Past performance doesn't guarantee future results
- **Risk of Loss**: Trading involves substantial risk
- **Not Financial Advice**: This is a tool, not investment advice
- **Your Responsibility**: You are responsible for your trading decisions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Zerodha for Kite Connect API
- Python community for excellent libraries
- SuperTrend indicator creators

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for detailed error messages
3. Open an issue on GitHub with:
   - Error message
   - Log snippets
   - Steps to reproduce

---

**Happy Trading! 🚀**

*Remember: Always trade responsibly and never risk more than you can afford to lose.*
