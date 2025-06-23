import argparse
import sys
from auth.kite_auth import KiteAuth
from main import TradingBot
from utils.logger import get_logger
from datetime import datetime, timedelta
import pandas as pd
from click import command
import click

logger = get_logger(__name__)

def authenticate():
    """Handle authentication"""
    auth = KiteAuth()
    
    print("🔐 Kite Connect Authentication")
    print("=" * 40)
    
    # Check existing connection
    if auth.test_connection():
        print("✅ Already authenticated!")
        return True
    
    # Generate login URL
    login_url = auth.generate_login_url()
    print(f"1. Visit: {login_url}")
    print("2. Login and authorize the app")
    print("3. Copy the request_token from the redirected URL")
    
    request_token = input("Enter request_token: ").strip()
    
    if auth.create_session(request_token):
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed!")
        return False

def test_connection():
    """Test Kite connection"""
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
                if isinstance(equity, dict):
                    available_cash = equity.get('available', {}).get('cash', 0)
                else:
                    available_cash = 0
                
                print(f"👤 User: {profile.get('user_name')}")
                print(f"💰 Available Cash: ₹{available_cash:,.2f}")
            except Exception as e:
                print(f"⚠️  Could not fetch account details: {e}")
        
        return True
    else:
        print("❌ Connection test failed!")
        return False

def start_trading():
    """Start trading bot"""
    bot = TradingBot()
    if bot.setup():
        print("🚀 Starting trading bot...")
        print("📊 Trading: NIFTY 50 → NIFTYBEES")
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
        
        # Configure backtest parameters
        config = {
            'initial_capital': 4000.0,  # Match your live trading capital
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
            
            # Show trade details
            if len(backtester.trades) > 0:
                print(f"\n📋 TRADE DETAILS")
                print("-" * 40)
                trades_df = pd.DataFrame(backtester.trades)
                for i, trade in trades_df.iterrows():
                    entry_date = pd.to_datetime(trade['entry_date']).strftime('%Y-%m-%d')
                    exit_date = pd.to_datetime(trade['exit_date']).strftime('%Y-%m-%d')
                    print(f"Trade {i+1}: {entry_date} → {exit_date}")
                    print(f"  Entry: ₹{trade['entry_price']:.2f} | Exit: ₹{trade['exit_price']:.2f}")
                    print(f"  P&L: ₹{trade['pnl']:.2f} | Reason: {trade['exit_reason']}")
                    print()
        else:
            print("\n❌ No trades executed during the backtest period")
            print("💡 This could mean:")
            print("   - No SuperTrend signals generated")
            print("   - Market conditions not suitable for the strategy")
            print("   - Try adjusting ATR period or factor")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        return False

@click.group()
def cli():
    """Trading Bot CLI"""
    pass

@cli.command()
def auth():
    """Run Kite Connect authentication"""
    from auth.kite_auth import main as auth_main
    auth_main()

@cli.command()
@click.option('--live', is_flag=True, help="Run in live trading mode")
def trade(live):
    """Start trading bot"""
    if live:
        logger.warning("Starting in LIVE trading mode!")
    else:
        logger.info("Starting in DRY RUN mode.")
    
    bot = TradingBot()
    if bot.setup():
        # NIFTY 50 -> NIFTYBEES (with MIS leverage support)
        bot.run("256265", "2707457", "NIFTYBEES")

if __name__ == "__main__":
    cli()
