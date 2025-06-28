#!/usr/bin/env python3
"""
FIXED CLI for SuperTrend Trading Bot
Corrected indentation error around line 242
"""

import argparse
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Enhanced imports
from auth.enhanced_kite_auth import KiteAuth, AuthenticationError
from enhanced_main_trading_bot import EnhancedTradingBot
from config.enhanced_settings import Settings, ConfigurationError
from utils.enhanced_logger import get_logger, log_session_start
from utils.enhanced_risk_manager import RiskManager
from utils.performance_monitor import PerformanceMonitor
from utils.market_data_validator import MarketDataValidator

# Multi-strategy system imports (with fallback)
try:
    from trading.strategies.strategy_factory import StrategyFactory, strategy_manager
    from trading.strategies.base_strategy import StrategyError
    MULTI_STRATEGY_AVAILABLE = True
    print("✅ Multi-strategy system loaded")
except ImportError as e:
    MULTI_STRATEGY_AVAILABLE = False
    print("⚠️  Multi-strategy system not available - using enhanced strategy only")

logger = get_logger(__name__)

class EnhancedCLI:
    """Enhanced Command Line Interface with multi-strategy support"""
    
    def __init__(self):
        self.auth = KiteAuth()
    
    def authenticate(self):
        """Enhanced authentication with comprehensive account info"""
        print("🔐 Enhanced Kite Connect Authentication")
        print("=" * 60)
        
        try:
            # Check existing connection
            if self.auth.test_connection():
                print("✅ Already authenticated!")
                
                # Get comprehensive account info
                account_info = self.auth.get_account_info()
                if account_info:
                    print(f"\n💼 Account Information:")
                    print(f"👤 Name: {account_info['user_name']}")
                    print(f"💰 Available Cash: ₹{account_info['available_cash']:,.2f}")
                    print(f"📊 Net Worth: ₹{account_info['net_worth']:,.2f}")
                    print(f"🏦 Broker: {account_info['broker']}")
                    
                    # Get current trading configuration
                    trading_config = Settings.get_trading_config()
                    print(f"\n📈 Current Trading Configuration:")
                    print(f"   Account Balance: ₹{trading_config.account_balance:,.2f}")
                    print(f"   Risk per Trade: {trading_config.risk_per_trade:.1%}")
                    print(f"   Position Sizing: {trading_config.position_sizing_method}")
                    print(f"   Max Daily Loss: {trading_config.max_daily_loss:.1%}")
                    
                    # Show available strategies
                    if MULTI_STRATEGY_AVAILABLE:
                        print(f"\n🎯 Available Strategies:")
                        strategies = StrategyFactory.list_strategies()
                        for key, info in strategies.items():
                            print(f"   • {key}: {info['name']}")
                        print(f"\n💡 Use --strategy=<key> to select a strategy")
                    
                    # Ask if user wants to update trading amount
                    self._update_trading_amount_interactive(account_info['available_cash'])
                
                return True
            
            # If not authenticated, start authentication process
            print("🔄 Starting authentication process...")
            return self.auth.ensure_authentication(interactive=True)
            
        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def _update_trading_amount_interactive(self, available_cash: float):
        """Interactive trading amount update - FIXED INDENTATION"""
        try:
            current_amount = Settings.get_trading_config().account_balance
            
            print(f"\n🎯 Trading Amount Configuration")
            print(f"Current trading amount: ₹{current_amount:,.2f}")
            print(f"Available cash: ₹{available_cash:,.2f}")
            
            while True:
                response = input(f"Update trading amount? (y/n): ").lower().strip()
                
                if response == 'n':
                    print("Trading amount unchanged.")
                    break
                elif response == 'y':
                    while True:
                        try:
                            amount_input = input(f"Enter new trading amount (₹): ").strip()
                            new_amount = float(amount_input)
                            
                            if new_amount <= 0:
                                print("❌ Amount must be positive")
                                continue
                            
                            if new_amount > available_cash:
                                print(f"⚠️  Warning: Amount exceeds available cash")
                                confirm = input("Continue anyway? (y/n): ").lower().strip()
                                if confirm != 'y':  # FIXED: Added proper indentation
                                    continue
                            
                            # Update the trading amount
                            Settings.update_trading_amount(new_amount)
                            print(f"✅ Trading amount updated to: ₹{new_amount:,.2f}")
                            break
                            
                        except ValueError:
                            print("❌ Invalid amount. Please enter a number.")
                        except Exception as e:
                            print(f"❌ Error updating amount: {e}")
                    break
                else:
                    print("Please enter 'y' or 'n'")
                    
        except Exception as e:
            logger.error(f"Error in interactive trading amount update: {e}")
    
    def test_connection(self):
        """Enhanced connection test with comprehensive diagnostics"""
        print("🔍 Enhanced Connection Diagnostics")
        print("=" * 50)
        
        try:
            # Basic connection test
            if self.auth.test_connection():
                print("✅ Basic connection: PASSED")
                
                # Get connection health
                health = self.auth.get_connection_health()
                print(f"\n📊 Connection Health:")
                print(f"   Status: {health.get('connection_status', 'Unknown').upper()}")
                print(f"   Token Valid: {'✅' if health.get('token_valid') else '❌'}")
                print(f"   Last Validation: {health.get('last_validation', 'Never')}")
                
                # Test market data access
                print(f"\n📈 Market Data Test:")
                try:
                    kite = self.auth.get_kite_instance()
                    if kite:
                        # Test price fetch
                        quote = kite.quote("NSE:NIFTYBEES")
                        if quote:
                            price = quote['NSE:NIFTYBEES']['last_price']
                            print(f"   ✅ Price fetch: NIFTYBEES @ ₹{price}")
                        else:
                            print(f"   ❌ Price fetch: FAILED")
                        
                        # Test historical data
                        historical = kite.historical_data(
                            "2707457",  # NIFTYBEES token
                            datetime.now() - timedelta(days=1),
                            datetime.now(),
                            "minute"
                        )
                        
                        if historical and len(historical) > 0:
                            print(f"   ✅ Historical data: {len(historical)} candles")
                        else:
                            print(f"   ❌ Historical data: FAILED")
                            
                except Exception as e:
                    print(f"   ❌ Market data test failed: {e}")
                
                # Test configuration
                print(f"\n⚙️  Configuration Test:")
                try:
                    validation = Settings.validate_all_configuration()
                    if validation['valid']:
                        print(f"   ✅ Configuration: VALID")
                    else:
                        print(f"   ❌ Configuration: INVALID")
                        for error in validation['errors']:
                            print(f"      - {error}")
                except Exception as e:
                    print(f"   ❌ Configuration test failed: {e}")
                
                return True
            else:
                print("❌ Basic connection: FAILED")
                print("💡 Try: python cli.py auth")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def list_strategies(self):
        """List all available strategies"""
        if not MULTI_STRATEGY_AVAILABLE:
            print("❌ Multi-strategy system not available")
            print("💡 Only enhanced strategy is available")
            return False
            
        print("📊 Available Trading Strategies")
        print("=" * 60)
        
        try:
            strategies = StrategyFactory.list_strategies()
            
            if not strategies:
                print("❌ No strategies available")
                return False
            
            for key, info in strategies.items():
                print(f"\n🎯 Strategy: {key}")
                print(f"   Name: {info['name']}")
                print(f"   Description: {info['description']}")
                print(f"   Class: {info['class_name']}")
                
                # Get detailed info
                try:
                    strategy_info = StrategyFactory.get_strategy_info(key)
                    if 'strategy_metadata' in strategy_info:
                        metadata = strategy_info['strategy_metadata']['metadata']
                        print(f"   Type: {metadata['type']}")
                        print(f"   Risk Level: {metadata['risk_level']}")
                        print(f"   Recommended Timeframes: {', '.join(metadata['recommended_timeframes'])}")
                except Exception as e:
                    logger.debug(f"Could not get detailed info for {key}: {e}")
            
            print(f"\n💡 Usage: python cli.py trade --strategy=<strategy_key>")
            print(f"   Example: python cli.py trade --strategy=bullet")
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing strategies: {e}")
            return False
    
    def start_trading(self, live_mode=False, instrument="NIFTYBEES", strategy="enhanced"):
        """Enhanced trading start with strategy selection"""
        print("🚀 Enhanced SuperTrend Trading Bot")
        print("=" * 60)
        
        try:
            # Validate configuration first
            validation = Settings.validate_all_configuration()
            if not validation['valid']:
                print("❌ Configuration validation failed:")
                for error in validation['errors']:
                    print(f"   • {error}")
                return False
            
            # Show warnings if any
            if validation['warnings']:
                print("⚠️  Configuration warnings:")
                for warning in validation['warnings']:
                    print(f"   • {warning}")
            
            # Handle strategy selection
            if MULTI_STRATEGY_AVAILABLE and strategy != "enhanced":
                # Validate and display strategy selection
                try:
                    strategies = StrategyFactory.list_strategies()
                    if strategy not in strategies:
                        available = list(strategies.keys())
                        print(f"❌ Unknown strategy '{strategy}'. Available: {available}")
                        print("🔄 Falling back to enhanced strategy")
                        strategy = "enhanced"
                    else:
                        strategy_info = strategies[strategy]
                        print(f"\n📊 Selected Strategy:")
                        print(f"   🎯 Key: {strategy}")
                        print(f"   📈 Name: {strategy_info['name']}")
                        print(f"   📋 Description: {strategy_info['description']}")
                        
                except Exception as e:
                    print(f"❌ Strategy validation error: {e}")
                    print("🔄 Falling back to enhanced strategy")
                    strategy = "enhanced"
            else:
                print(f"\n📊 Using Enhanced SuperTrend Strategy (default)")
            
            # Display current configuration
            print(f"\n📊 Trading Configuration:")
            trading_config = Settings.get_trading_config()
            safety_config = Settings.get_safety_config()
            
            print(f"   Account Balance: ₹{trading_config.account_balance:,.2f}")
            print(f"   Risk per Trade: {trading_config.risk_per_trade:.1%}")
            print(f"   Position Sizing: {trading_config.position_sizing_method}")
            print(f"   Max Daily Loss: {trading_config.max_daily_loss:.1%}")
            print(f"   Max Drawdown: {trading_config.max_drawdown:.1%}")
            
            # Determine and confirm trading mode
            if live_mode:
                mode = "LIVE TRADING"
                print(f"\n🚨 {mode} MODE SELECTED 🚨")
                print("=" * 30)
                print("⚠️  You are about to place REAL orders on your broker account!")
                print("⚠️  Real money will be at risk!")
                print("⚠️  Ensure you understand all risks involved!")
                if strategy != "enhanced":
                    print(f"⚠️  Strategy: {strategy}")
                
                confirmation = input("\nType 'CONFIRM LIVE TRADING' to proceed: ")
                if confirmation != "CONFIRM LIVE TRADING":
                    print("❌ Live trading cancelled.")
                    return False
                
                # Update safety config for live trading
                safety_config.live_trading_enabled = True
                safety_config.dry_run_mode = False
                
            else:
                mode = "DRY RUN / PAPER TRADING"
                print(f"\n✅ {mode} MODE")
                print("📝 No real orders will be placed")
                print("💰 Using paper trading balance")
            
            print(f"\n🎯 Trading Details:")
            print(f"   Mode: {mode}")
            print(f"   Strategy: {strategy}")
            print(f"   Instrument: {instrument}")
            print(f"   Session ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Initialize and start the enhanced trading bot
            print(f"\n🔧 Initializing Enhanced Trading Bot...")
            
            bot = EnhancedTradingBot()
            
            if not bot.setup():
                print("❌ Bot setup failed!")
                return False
            
            print("✅ Bot setup completed successfully")
            print(f"\n🚀 Starting trading session...")
            print("📊 Monitor logs: tail -f logs/trading.log")
            print("⏹️  Stop trading: Ctrl+C")
            print("=" * 60)
            
            # Start trading
            bot.run(instrument)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Trading stopped by user")
            return True
        except Exception as e:
            print(f"\n❌ Error starting trading: {e}")
            logger.error(f"CLI trading start error: {e}")
            return False
    
    def analyze_performance(self, days=30):
        """Analyze trading performance"""
        print(f"📊 Performance Analysis (Last {days} days)")
        print("=" * 50)
        
        try:
            # Initialize performance monitor
            trading_config = Settings.get_trading_config()
            pm = PerformanceMonitor(Settings.DATA_DIR, trading_config)
            
            # Get performance metrics
            current_metrics = pm.get_current_metrics()
            performance_metrics = pm.calculate_performance_metrics()
            
            # Display current session
            if current_metrics.get('session_id'):
                print(f"\n📅 Current Session:")
                print(f"   Session ID: {current_metrics['session_id']}")
                print(f"   Duration: {current_metrics.get('session_duration_minutes', 0)} minutes")
                print(f"   Current P&L: ₹{current_metrics.get('current_pnl', 0):.2f}")
                print(f"   Trades Completed: {current_metrics.get('trades_completed', 0)}")
            
            # Display overall performance
            print(f"\n📈 Overall Performance:")
            print(f"   Total Trades: {performance_metrics.get('total_trades', 0)}")
            print(f"   Win Rate: {performance_metrics.get('win_rate', 0):.1%}")
            print(f"   Total P&L: ₹{performance_metrics.get('total_pnl', 0):.2f}")
            print(f"   Average P&L: ₹{performance_metrics.get('average_pnl', 0):.2f}")
            print(f"   Max Win: ₹{performance_metrics.get('max_win', 0):.2f}")
            print(f"   Max Loss: ₹{performance_metrics.get('max_loss', 0):.2f}")
            print(f"   Max Drawdown: {performance_metrics.get('max_drawdown', 0):.1%}")
            print(f"   Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
            
            # Export detailed report
            report_path = pm.export_performance_report()
            print(f"\n💾 Detailed report saved: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing performance: {e}")
            return False

    def run_backtest(self, strategy="enhanced", csv_file=None, list_strategies=False):
        """Run backtest with selected strategy"""
        print("📊 SuperTrend Strategy Backtester - MULTI-STRATEGY VERSION")
        print("=" * 60)
        
        try:
            # Import backtest components
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Import the backtest class
            from backtest.backtest_strategy import SuperTrendBacktester
            
            if list_strategies:
                print("\nAvailable strategies:")
                if MULTI_STRATEGY_AVAILABLE:
                    strategies = StrategyFactory.list_strategies()
                    for key, info in strategies.items():
                        print(f"  {key}: {info['name']} - {info['description']}")
                else:
                    print("  enhanced: Enhanced SuperTrend (default)")
                return True
            
            if not csv_file:
                print("❌ No CSV file provided. Use --csv to specify your historical data file.")
                print("\nUSAGE:")
                print("  python cli.py backtest --csv historical_data/NIFTYBEES_historical_data.csv [--strategy=bullet]")
                print("  python cli.py backtest --list-strategies")
                return False
            
            # Configuration
            config = {
                'initial_capital': 10000,
                'leverage': 5.0,  # 5x leverage like NIFTYBEES
                'stop_loss': 100,
                'commission_per_trade': 20,
                'atr_period': 10,
                'factor': 3.0
            }
            
            print("Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
            
            # Load data
            print(f"Loading historical data from CSV: {csv_file}")
            df = pd.read_csv(csv_file, parse_dates=['date'])
            before = len(df)
            df.drop_duplicates(subset='date', inplace=True)
            after_dupes = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            after_na = len(df)
            dropped_dupes = before - after_dupes
            dropped_na = after_dupes - after_na
            if dropped_dupes > 0 or dropped_na > 0:
                print(f"Data cleaning: dropped {dropped_dupes} duplicate rows and {dropped_na} rows with missing values.")
            df.set_index('date', inplace=True)
            idx_list = list(df.index)
            print(f"Data loaded: {len(df)} rows from {idx_list[0].strftime('%Y-%m-%d')} to {idx_list[-1].strftime('%Y-%m-%d')}")
            print()
            
            # Select strategy
            try:
                if MULTI_STRATEGY_AVAILABLE:
                    strategy_instance = StrategyFactory.create_strategy(strategy)
                    print(f"Using strategy: {strategy}")
                else:
                    # Fallback to enhanced strategy if multi-strategy not available
                    print(f"Multi-strategy not available, using enhanced strategy")
                    strategy_instance = None  # Will use default in backtester
                    strategy = "enhanced"
            except Exception as e:
                print(f"❌ Error: {e}")
                if MULTI_STRATEGY_AVAILABLE:
                    print("Use --list-strategies to see available options.")
                return False
            
            # Initialize backtester with selected strategy
            backtester = SuperTrendBacktester(strategy_instance, **config)
            
            # Run backtest
            print("Running backtest...")
            result_df = backtester.run_backtest(df)
            print()
            
            if len(backtester.trades) > 0:
                print("Generating comprehensive report...")
                backtester.generate_report(result_df, 'fixed_backtest_report.txt')
                print("Generating charts...")
                backtester.plot_results(result_df, 'fixed_backtest_charts.png')
                print("\n🎉 Backtest completed successfully!")
                print(f"📊 {len(backtester.trades)} trades executed")
                print(f"💰 Final portfolio value: ₹{backtester.equity_curve[-1]:,.2f}")
                print(f"📈 Total return: {((backtester.equity_curve[-1] - config['initial_capital']) / config['initial_capital'] * 100):.2f}%")
                return True
            else:
                print("❌ No trades executed!")
                print("This should not happen with the fixed version.")
                return False
                
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            logger.error(f"Backtest error: {e}")
            return False


def main():
    """Main function with proper argument handling"""
    try:
        cli = EnhancedCLI()
        
        parser = argparse.ArgumentParser(description="Enhanced SuperTrend Trading Bot CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Authentication command
        auth_parser = subparsers.add_parser('auth', help='Authenticate with Zerodha Kite')
        
        # Connection test command
        test_parser = subparsers.add_parser('test', help='Test connection and configuration')
        
        # List strategies command
        if MULTI_STRATEGY_AVAILABLE:
            strategies_parser = subparsers.add_parser('strategies', help='List available strategies')
        
        # Trading command
        trade_parser = subparsers.add_parser('trade', help='Start trading')
        trade_parser.add_argument('--live', action='store_true', help='Enable live trading (default: dry run)')
        trade_parser.add_argument('--instrument', default='NIFTYBEES', help='Trading instrument')
        if MULTI_STRATEGY_AVAILABLE:
            trade_parser.add_argument('--strategy', default='enhanced', help='Trading strategy')
        
        # Performance analysis command
        perf_parser = subparsers.add_parser('performance', help='Analyze performance')
        perf_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
        
        # Backtest command
        backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtest')
        backtest_parser.add_argument('--csv', type=str, help='Path to historical OHLC CSV file')
        backtest_parser.add_argument('--strategy', type=str, default='enhanced', help='Strategy key to use (default: enhanced)')
        backtest_parser.add_argument('--list-strategies', action='store_true', help='List all available strategies and exit')
        
        args = parser.parse_args()
        
        if args.command == 'auth':
            success = cli.authenticate()
            sys.exit(0 if success else 1)
            
        elif args.command == 'test':
            success = cli.test_connection()
            sys.exit(0 if success else 1)
            
        elif args.command == 'strategies' and MULTI_STRATEGY_AVAILABLE:
            success = cli.list_strategies()
            sys.exit(0 if success else 1)
            
        elif args.command == 'trade':
            strategy = getattr(args, 'strategy', 'enhanced')
            success = cli.start_trading(
                live_mode=args.live,
                instrument=args.instrument,
                strategy=strategy
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'performance':
            success = cli.analyze_performance(args.days)
            sys.exit(0 if success else 1)
            
        elif args.command == 'backtest':
            success = cli.run_backtest(
                strategy=args.strategy,
                csv_file=args.csv,
                list_strategies=args.list_strategies
            )
            sys.exit(0 if success else 1)
            
        else:
            parser.print_help()
            print("\n🚀 Quick Start:")
            print("  python cli.py auth         # Authenticate first")
            print("  python cli.py test         # Test connection") 
            print("  python cli.py trade        # Start paper trading")
            print("  python cli.py trade --live # Start live trading")
            if MULTI_STRATEGY_AVAILABLE:
                print("  python cli.py strategies   # List strategies")
                print("  python cli.py trade --strategy=bullet  # Use bulletproof strategy")
            print("  python cli.py backtest --csv data.csv --strategy=bullet  # Run backtest")
            print("  python cli.py backtest --list-strategies  # List available strategies")
    
    except KeyboardInterrupt:
        print("\n⏹️  CLI stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()