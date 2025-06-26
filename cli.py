import argparse
import sys
import json
import os
from pathlib import Path
from auth.kite_auth import KiteAuth
from main import TradingBot
from utils.logger import get_logger
from datetime import datetime, timedelta
import pandas as pd
from config.settings import Settings

logger = get_logger(__name__)

# File to store trading preferences
TRADING_PREFS_FILE = Path("data/trading_preferences.json")
OPTIMIZATION_RESULTS_FILE = Path("data/optimization_results.json")

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

def save_optimization_results(results):
    """Save optimization results"""
    OPTIMIZATION_RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(OPTIMIZATION_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def load_optimization_results():
    """Load saved optimization results"""
    if OPTIMIZATION_RESULTS_FILE.exists():
        with open(OPTIMIZATION_RESULTS_FILE, 'r') as f:
            return json.load(f)
    return None

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
                
                # Show optimization status
                opt_results = load_optimization_results()
                if opt_results:
                    print(f"\n🎯 Optimization Status:")
                    print(f"   Last optimized: {opt_results.get('date', 'Unknown')}")
                    print(f"   Best parameters: ATR={opt_results.get('atr_period', 10)}, Factor={opt_results.get('factor', 3.0)}")
                    print(f"   Expected improvement: {opt_results.get('improvement', 0):+.1f}%")
                
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
                    print("📊 Trading Amount: Not set (run 'python enhanced_cli.py auth' to set)")
                
                # Show optimization status
                opt_results = load_optimization_results()
                if opt_results:
                    print(f"🎯 Optimized Parameters: ATR={opt_results.get('atr_period', 10)}, Factor={opt_results.get('factor', 3.0)}")
                else:
                    print("🎯 Optimization: Not run (run 'python enhanced_cli.py optimize' to optimize)")
                
            except Exception as e:
                print(f"⚠️  Could not fetch account details: {e}")
        
        return True
    else:
        print("❌ Connection test failed!")
        return False

def run_optimization():
    """Run parameter optimization"""
    print("🔍 SuperTrend Parameter Optimization")
    print("=" * 50)
    
    try:
        # Import optimization runner
        from optimization_runner import QuickOptimizer
        
        # Initialize optimizer
        prefs = load_trading_preferences()
        trading_amount = prefs.get('last_trading_amount', 10000)
        
        optimizer = QuickOptimizer(
            initial_capital=trading_amount,
            leverage=5.0,
            stop_loss=100,
            commission=20
        )
        
        # Load data and run optimization
        df = optimizer.load_your_data()
        results_df = optimizer.run_optimization(df)
        
        if results_df is not None:
            # Get best parameters
            best_params = optimizer.create_quick_report(results_df, df)
            
            # Save optimization results
            opt_results = {
                'date': datetime.now().isoformat(),
                'atr_period': int(best_params['atr_period']),
                'factor': float(best_params['factor']),
                'expected_return': float(best_params['total_return']),
                'win_rate': float(best_params['win_rate']),
                'improvement': float(best_params['total_return']) - results_df[(results_df['atr_period']==10) & (results_df['factor']==3.0)]['total_return'].iloc[0] if len(results_df[(results_df['atr_period']==10) & (results_df['factor']==3.0)]) > 0 else 0
            }
            
            save_optimization_results(opt_results)
            
            print(f"\n🎯 OPTIMIZATION COMPLETE!")
            print("=" * 50)
            print(f"✅ Best parameters: ATR={opt_results['atr_period']}, Factor={opt_results['factor']}")
            print(f"✅ Expected return: {opt_results['expected_return']:+.2f}%")
            print(f"✅ Expected win rate: {opt_results['win_rate']:.1f}%")
            print(f"✅ Potential improvement: {opt_results['improvement']:+.2f} percentage points")
            
            print(f"\n📝 NEXT STEPS:")
            print(f"1. Update your .env file:")
            print(f"   ATR_PERIOD={opt_results['atr_period']}")
            print(f"   FACTOR={opt_results['factor']}")
            print(f"2. Test with paper trading first")
            print(f"3. Start with reduced capital when going live")
            
            return True
        else:
            print("❌ Optimization failed!")
            return False
            
    except ImportError:
        print("❌ Optimization module not found!")
        print("💡 Make sure optimization_runner.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def start_trading(trading_amount=None, live_mode=False, optimized=False):
    """Start trading bot with optional optimization"""
    
    # Load preferences if amount not specified
    if trading_amount is None:
        prefs = load_trading_preferences()
        trading_amount = prefs.get('last_trading_amount', 4000.0)
    
    # Apply optimized parameters if requested
    if optimized:
        opt_results = load_optimization_results()
        if opt_results:
            # Update environment variables
            os.environ['ATR_PERIOD'] = str(opt_results['atr_period'])
            os.environ['FACTOR'] = str(opt_results['factor'])
            
            print(f"🎯 Using optimized parameters:")
            print(f"   ATR Period: {opt_results['atr_period']}")
            print(f"   Factor: {opt_results['factor']}")
            print(f"   Expected improvement: {opt_results.get('improvement', 0):+.1f}%")
        else:
            print("⚠️  No optimization results found. Run 'python enhanced_cli.py optimize' first")
            print("📊 Using default parameters: ATR=10, Factor=3.0")
    
    # Update the configuration dynamically
    Settings.STRATEGY_PARAMS['account_balance'] = trading_amount
    
    # Show trading configuration
    print(f"💰 Trading with amount: ₹{trading_amount:,.2f}")
    
    # Initialize and run bot
    bot = TradingBot()
    if bot.setup():
        print("🚀 Starting enhanced trading bot...")
        print("📊 Trading: NIFTY 50 → NIFTYBEES")
        print(f"💵 Account Balance: ₹{trading_amount:,.2f}")
        print(f"📈 Mode: {'LIVE' if live_mode else 'DRY RUN'}")
        print(f"🎯 Parameters: {'OPTIMIZED' if optimized else 'DEFAULT'}")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        # Default: NIFTY 50 -> NIFTYBEES
        bot.run("NIFTYBEES")  # Now using NIFTYBEES for both signal and trading
    else:
        print("❌ Failed to setup trading bot")

def show_optimization_status():
    """Show current optimization status"""
    print("📊 Optimization Status")
    print("=" * 30)
    
    opt_results = load_optimization_results()
    if opt_results:
        print(f"✅ Optimization completed: {opt_results['date']}")
        print(f"🎯 Best parameters: ATR={opt_results['atr_period']}, Factor={opt_results['factor']}")
        print(f"📈 Expected return: {opt_results['expected_return']:+.2f}%")
        print(f"🎲 Expected win rate: {opt_results['win_rate']:.1f}%")
        print(f"⚡ Potential improvement: {opt_results.get('improvement', 0):+.2f} percentage points")
        
        # Check if .env is updated
        current_atr = int(os.getenv('ATR_PERIOD', 10))
        current_factor = float(os.getenv('FACTOR', 3.0))
        
        if current_atr == opt_results['atr_period'] and current_factor == opt_results['factor']:
            print(f"✅ .env file is updated with optimized parameters")
        else:
            print(f"⚠️  .env file not updated. Current: ATR={current_atr}, Factor={current_factor}")
            print(f"💡 Update your .env file:")
            print(f"   ATR_PERIOD={opt_results['atr_period']}")
            print(f"   FACTOR={opt_results['factor']}")
    else:
        print("❌ No optimization results found")
        print("💡 Run: python enhanced_cli.py optimize")

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
        print("❌ Please authenticate first: python enhanced_cli.py auth")
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
        
        # Check for optimized parameters
        opt_results = load_optimization_results()
        if opt_results:
            atr_period = opt_results['atr_period']
            factor = opt_results['factor']
            print(f"🎯 Using optimized parameters: ATR={atr_period}, Factor={factor}")
        else:
            atr_period = 10
            factor = 3.0
            print(f"📊 Using default parameters: ATR={atr_period}, Factor={factor}")
        
        # Configure backtest parameters
        config = {
            'initial_capital': trading_amount,
            'leverage': 5.0,  # NIFTYBEES leverage
            'stop_loss': 100.0,  # Fixed stop loss
            'commission_per_trade': 20.0,  # Brokerage
            'atr_period': atr_period,
            'factor': factor
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
    """Enhanced CLI function with optimization features"""
    parser = argparse.ArgumentParser(description='Enhanced SuperTrend Trading Bot')
    parser.add_argument('command', 
                       choices=['auth', 'test', 'trade', 'reset', 'backtest', 'optimize', 'status'], 
                       help='Command to execute')
    parser.add_argument('--live', action='store_true', 
                       help='Run in live trading mode (default is dry run)')
    parser.add_argument('--optimized', action='store_true',
                       help='Use optimized parameters (run optimize first)')
    parser.add_argument('--amount', type=float, 
                       help='Trading amount to use (overrides saved preference)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n🚀 Enhanced Features:")
        print("  optimize  - Find optimal SuperTrend parameters")
        print("  status    - Show optimization status")
        print("  --optimized - Use optimized parameters for trading")
        return
    
    args = parser.parse_args()
    
    if args.command == 'auth':
        authenticate()
    elif args.command == 'test':
        test_connection()
    elif args.command == 'optimize':
        run_optimization()
    elif args.command == 'status':
        show_optimization_status()
    elif args.command == 'trade':
        # Determine mode
        if args.live:
            logger.warning("Starting in LIVE trading mode!")
            # Override dry run mode
            Settings.SAFETY_CONFIG['live_trading_enabled'] = True
            Settings.SAFETY_CONFIG['dry_run_mode'] = False
        else:
            logger.info("Starting in DRY RUN mode.")
        
        # Print the true, final mode after all settings are loaded
        mode = "LIVE" if Settings.SAFETY_CONFIG.get('live_trading_enabled', False) else "DRY RUN"
        print("\n==============================")
        print(f"  TRADING MODE: {mode}")
        print("==============================")
        if mode == "LIVE":
            print("\n🚨🚨🚨 WARNING: LIVE TRADING ENABLED! 🚨🚨🚨")
            print("You are about to place REAL orders on your broker account.")
            print("Type 'CONFIRM' (all caps) to proceed, or anything else to abort.")
            user_input = input("Proceed with LIVE trading? Type CONFIRM to continue: ")
            if user_input.strip() != "CONFIRM":
                print("❌ Aborted. Live trading not started.")
                return
            print("✅ Live trading confirmed. Proceeding...")
        else:
            print("DRY RUN mode: No real orders will be placed.\n")
        # Start trading with specified or saved amount
        start_trading(trading_amount=args.amount, live_mode=(mode=="LIVE"), optimized=args.optimized)
    elif args.command == 'reset':
        emergency_reset()
    elif args.command == 'backtest':
        run_backtest()

if __name__ == "__main__":
    main()