import argparse
import sys
import json
from pathlib import Path
from auth.kite_auth import KiteAuth
from main import TradingBot
from utils.logger import get_logger
from datetime import datetime, timedelta
import pandas as pd

logger = get_logger(__name__)

# File to store trading preferences
TRADING_PREFS_FILE = Path("data/trading_preferences.json")

def save_trading_preferences(preferences):
    """Save trading preferences to file"""
    TRADING_PREFS_FILE.parent.mkdir(exist_ok=True)
    with open(TRADING_PREFS_FILE, 'w') as f:
        json.dump(preferences, f, indent=2)

def load_trading_preferences():
    """Load trading preferences from file"""
    if TRADING_PREFS_FILE.exists():
        with open(TRADING_PREFS_FILE, 'r') as f:
            return json.load(f)
    return {}

def authenticate():
    """Handle authentication with balance display"""
    auth = KiteAuth()
    
    print("🔐 Kite Connect Authentication")
    print("=" * 50)
    
    # Check existing connection
    if auth.test_connection():
        print("✅ Already authenticated!")
        
        # Get and display account info
        kite = auth.get_kite_instance()
        if kite:
            try:
                profile = kite.profile()
                margins = kite.margins()
                equity = margins.get('equity', {})
                
                available_cash = equity.get('available', {}).get('cash', 0) if isinstance(equity, dict) else 0
                net_worth = equity.get('net', 0) if isinstance(equity, dict) else 0
                
                print(f"\n💼 Account Information:")
                print(f"👤 Name: {profile.get('user_name')}")
                print(f"💰 Available Cash: ₹{available_cash:,.2f}")
                print(f"📊 Net Worth: ₹{net_worth:,.2f}")
                
                # Ask if user wants to set trading amount
                print(f"\n🎯 Set Trading Amount for Today")
                print(f"Current available cash: ₹{available_cash:,.2f}")
                
                # Load previous preference
                prefs = load_trading_preferences()
                last_amount = prefs.get('last_trading_amount', 4000.0)
                
                print(f"Last trading amount: ₹{last_amount:,.2f}")
                
                # Ask for new amount
                while True:
                    amount_input = input(f"Enter trading amount (press Enter for ₹{last_amount:,.2f}): ").strip()
                    
                    if not amount_input:
                        trading_amount = last_amount
                        break
                    
                    try:
                        trading_amount = float(amount_input)
                        if trading_amount <= 0:
                            print("❌ Amount must be positive")
                            continue
                        if trading_amount > available_cash:
                            print(f"⚠️  Warning: Amount exceeds available cash (₹{available_cash:,.2f})")
                            confirm = input("Continue anyway? (yes/no): ").lower().strip()
                            if confirm != 'yes':
                                continue
                        break
                    except ValueError:
                        print("❌ Invalid amount. Please enter a number.")
                
                # Save preference
                prefs['last_trading_amount'] = trading_amount
                prefs['last_update'] = datetime.now().isoformat()
                save_trading_preferences(prefs)
                
                print(f"\n✅ Trading amount set to: ₹{trading_amount:,.2f}")
                print(f"💡 This will be used for today's trading session")
                
            except Exception as e:
                print(f"⚠️  Could not fetch account details: {e}")
        
        return True
    
    # If not authenticated, proceed with login
    login_url = auth.generate_login_url()
    print(f"1. Visit: {login_url}")
    print("2. Login and authorize the app")
    print("3. Copy the request_token from the redirected URL")
    
    request_token = input("Enter request_token: ").strip()
    
    if auth.create_session(request_token):
        print("✅ Authentication successful!")
        
        # After successful auth, show balance and ask for trading amount
        authenticate()  # Recursive call to show balance
        return True
    else:
        print("❌ Authentication failed!")
        return False

def test_connection():
    """Test Kite connection with balance info"""
    auth = KiteAuth()
    if auth.test_connection():
        print("✅ Connection test successful!")
        
        # Show account info
        kite = auth.get_kite_instance()
        if kite:
            try:
                profile = kite.profile()
                margins = kite.margins()
                equity = margins.get('equity', {})
                available_cash = equity.get('available', {}).get('cash', 0) if isinstance(equity, dict) else 0
                
                print(f"👤 User: {profile.get('user_name')}")
                print(f"💰 Available Cash: ₹{available_cash:,.2f}")
                
                # Show current trading amount preference
                prefs = load_trading_preferences()
                if 'last_trading_amount' in prefs:
                    print(f"📊 Trading Amount Set: ₹{prefs['last_trading_amount']:,.2f}")
                else:
                    print("📊 Trading Amount: Not set (run 'python cli.py auth' to set)")
                
            except Exception as e:
                print(f"⚠️  Could not fetch account details: {e}")
        
        return True
    else:
        print("❌ Connection test failed!")
        return False

def start_trading(trading_amount=None, live_mode=False):
    """Start trading bot with dynamic amount"""
    
    # Load preferences if amount not specified
    if trading_amount is None:
        prefs = load_trading_preferences()
        trading_amount = prefs.get('last_trading_amount', 4000.0)
    
    # Update the configuration dynamically
    from config.settings import Settings
    Settings.STRATEGY_PARAMS['account_balance'] = trading_amount
    
    # Show trading configuration
    print(f"💰 Trading with amount: ₹{trading_amount:,.2f}")
    
    # Initialize and run bot
    bot = TradingBot()
    if bot.setup():
        print("🚀 Starting trading bot...")
        print("📊 Trading: NIFTY 50 → NIFTYBEES")
        print(f"💵 Account Balance: ₹{trading_amount:,.2f}")
        print(f"📈 Mode: {'LIVE' if live_mode else 'DRY RUN'}")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        # Default: NIFTY 50 -> NIFTYBEES
        bot.run("256265", "2707457", "NIFTYBEES")
    else:
        print("❌ Failed to setup trading bot")

def emergency_reset():
    """Emergency position reset"""
    print("🚨 EMERGENCY POSITION RESET")
    print("This will help if your bot shows positions that don't exist")
    print("Only use this if auto square-off happened but bot still shows position")
    
    confirm = input("Are you sure you want to reset position tracking? (yes/no): ").lower().strip()
    if confirm == "yes":
        print("✅ Emergency reset completed")
        print("💡 Restart your bot - it will check actual positions on startup")
        logger.warning("EMERGENCY POSITION RESET BY USER")
    else:
        print("❌ Reset cancelled")

def run_backtest():
    """Run 30-day backtest with real data"""
    print("📊 SUPERTREND STRATEGY BACKTEST")
    print("=" * 40)
    
    # Check authentication
    auth = KiteAuth()
    if not auth.test_connection():
        print("❌ Please authenticate first: python cli.py auth")
        return False
    
    kite = auth.get_kite_instance()
    if not kite:
        print("❌ Failed to get Kite instance")
        return False
    
    # Calculate date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("📈 Instrument: NIFTYBEES (2707457)")
    
    try:
        # Fetch historical data
        print("📥 Fetching historical data...")
        historical_data = kite.historical_data(
            instrument_token="2707457",  # NIFTYBEES
            from_date=start_date,
            to_date=end_date,
            interval="day"
        )
        
        if not historical_data:
            print("❌ No data received from Kite")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename columns to match expected format
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
        
        print(f"✅ Loaded {len(df)} data points")
        
        # Import backtesting modules
        try:
            from backtest_strategy import SuperTrendBacktester
        except ImportError:
            print("❌ Backtest modules not found. Please ensure backtest_strategy.py is available")
            return False
        
        # Load trading amount preference
        prefs = load_trading_preferences()
        trading_amount = prefs.get('last_trading_amount', 4000.0)
        
        # Configure backtest parameters
        config = {
            'initial_capital': trading_amount,
            'leverage': 5.0,  # NIFTYBEES leverage
            'stop_loss': 100.0,  # Fixed stop loss
            'commission_per_trade': 20.0,  # Brokerage
            'atr_period': 10,
            'factor': 3.0
        }
        
        print("\n⚙️  Backtest Configuration:")
        print(f"   Initial Capital: ₹{config['initial_capital']:,.2f}")
        print(f"   Leverage: {config['leverage']}x")
        print(f"   Stop Loss: ₹{config['stop_loss']}")
        print(f"   ATR Period: {config['atr_period']}")
        print(f"   Factor: {config['factor']}")
        
        # Run backtest
        print("\n🚀 Running backtest...")
        backtester = SuperTrendBacktester(**config)
        result_df = backtester.run_backtest(df)
        
        # Calculate and display metrics
        if len(backtester.trades) > 0:
            metrics = backtester.calculate_metrics(result_df)
            
            print("\n📊 BACKTEST RESULTS")
            print("=" * 40)
            print(f"Total Trades: {metrics.get('Total Trades', 0)}")
            print(f"Winning Trades: {metrics.get('Winning Trades', 0)}")
            print(f"Win Rate: {metrics.get('Win Rate (%)', 0):.1f}%")
            print(f"Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
            print(f"Final Capital: ₹{backtester.equity_curve[-1]:,.2f}")
            print(f"Max Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"backtest_report_{timestamp}.txt"
            chart_file = f"backtest_charts_{timestamp}.png"
            
            backtester.generate_report(result_df, report_file)
            backtester.plot_results(result_df, chart_file)
            
            print(f"\n💾 Results saved:")
            print(f"   Report: {report_file}")
            print(f"   Charts: {chart_file}")
            
        else:
            print("\n❌ No trades executed during the backtest period")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='SuperTrend Trading Bot')
    parser.add_argument('command', choices=['auth', 'test', 'trade', 'reset', 'backtest'], 
                       help='Command to execute')
    parser.add_argument('--live', action='store_true', 
                       help='Run in live trading mode (default is dry run)')
    parser.add_argument('--amount', type=float, 
                       help='Trading amount to use (overrides saved preference)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    if args.command == 'auth':
        authenticate()
    elif args.command == 'test':
        test_connection()
    elif args.command == 'trade':
        # Determine mode
        if args.live:
            logger.warning("Starting in LIVE trading mode!")
            # Override dry run mode
            from config.settings import Settings
            Settings.SAFETY_CONFIG['live_trading_enabled'] = True
            Settings.SAFETY_CONFIG['dry_run_mode'] = False
        else:
            logger.info("Starting in DRY RUN mode.")
        
        # Start trading with specified or saved amount
        start_trading(trading_amount=args.amount, live_mode=args.live)
    elif args.command == 'reset':
        emergency_reset()
    elif args.command == 'backtest':
        run_backtest()

if __name__ == "__main__":
    main()