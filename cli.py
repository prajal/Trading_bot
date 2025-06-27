#!/usr/bin/env python3
"""
Enhanced CLI for SuperTrend Trading Bot
Integrates with all new components: risk management, performance monitoring, etc.
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

logger = get_logger(__name__)

class EnhancedCLI:
    """Enhanced Command Line Interface for the trading bot"""
    
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
        """Interactive trading amount update"""
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
                                if confirm != 'y':
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
    
    def start_trading(self, live_mode=False, instrument="NIFTYBEES"):
        """Enhanced trading start with comprehensive setup"""
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
            print(f"   Instrument: {instrument}")
            print(f"   Strategy: Enhanced SuperTrend")
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
            daily_performance = pm.get_daily_performance(days)
            
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
            
            # Display daily performance summary
            if daily_performance.get('period_summary'):
                summary = daily_performance['period_summary']
                print(f"\n📊 {days}-Day Summary:")
                print(f"   Trading Days: {summary.get('trading_days', 0)}")
                print(f"   Profitable Days: {summary.get('profitable_days', 0)}")
                print(f"   Profit Day Ratio: {summary.get('profit_day_ratio', 0):.1%}")
                print(f"   Average Daily P&L: ₹{summary.get('average_daily_pnl', 0):.2f}")
            
            # Export detailed report
            report_path = pm.export_performance_report()
            print(f"\n💾 Detailed report saved: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing performance: {e}")
            return False
    
    def validate_data(self, symbol="NIFTYBEES"):
        """Validate market data quality"""
        print(f"🔍 Market Data Validation: {symbol}")
        print("=" * 40)
        
        try:
            # Get market data
            kite = self.auth.get_kite_instance()
            if not kite:
                print("❌ Authentication required")
                return False
            
            # Fetch recent data
            token = "2707457"  # NIFTYBEES token
            end_date = datetime.now() 
            start_date = end_date - timedelta(days=7)
            
            print("📥 Fetching market data...")
            data = kite.historical_data(token, start_date, end_date, "minute")
            
            if not data:
                print("❌ No data received")
                return False
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"✅ Data loaded: {len(df)} candles")
            
            # Validate data quality
            validator = MarketDataValidator()
            
            # Basic validation
            is_valid = validator.validate_ohlc_data(df, strict_mode=False)
            print(f"📊 Basic Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            
            # Quality score
            quality_score = validator.calculate_data_quality_score(df)
            print(f"🏆 Quality Score: {quality_score['overall_score']:.1%}")
            print(f"📈 Quality Rating: {quality_score['quality_rating'].upper()}")
            
            # Issues found
            if quality_score.get('issues'):
                print(f"\n⚠️  Issues Found:")
                for i, issue in enumerate(quality_score['issues'][:5], 1):  # Show top 5
                    print(f"   {i}. {issue}")
                if len(quality_score['issues']) > 5:
                    print(f"   ... and {len(quality_score['issues']) - 5} more")
            
            # Generate detailed report
            report = validator.generate_data_quality_report(df, symbol)
            report_file = Settings.DATA_DIR / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n💾 Detailed report saved: {report_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating data: {e}")
            return False
    
    def risk_analysis(self):
        """Analyze current risk metrics"""
        print("🛡️  Risk Analysis")
        print("=" * 30)
        
        try:
            # Initialize risk manager
            trading_config = Settings.get_trading_config()
            risk_config = Settings.get_risk_config()
            rm = RiskManager(trading_config, risk_config, Settings.DATA_DIR)
            
            # Get risk summary
            risk_summary = rm.get_risk_summary()
            
            print(f"📊 Current Risk Status:")
            print(f"   Risk Level: {risk_summary.get('risk_level', 'Unknown').upper()}")
            print(f"   Current Drawdown: {risk_summary.get('current_drawdown', 0):.1%}")
            print(f"   Max Drawdown: {risk_summary.get('max_drawdown', 0):.1%}")
            print(f"   Daily P&L: ₹{risk_summary.get('daily_pnl', 0):.2f}")
            print(f"   Trades Today: {risk_summary.get('trades_today', 0)}")
            
            # Circuit breaker status
            if risk_summary.get('circuit_breaker_active'):
                print(f"\n🚨 CIRCUIT BREAKER: ACTIVE")
            else:
                print(f"\n✅ Circuit Breaker: Inactive")
            
            # Position information
            print(f"\n📊 Position Information:")
            print(f"   Total Positions: {risk_summary.get('total_positions', 0)}")
            print(f"   Position Value: ₹{risk_summary.get('total_position_value', 0):,.2f}")
            print(f"   Unrealized P&L: ₹{risk_summary.get('total_unrealized_pnl', 0):.2f}")
            print(f"   Available Capital: ₹{risk_summary.get('available_capital', 0):,.2f}")
            
            # Risk recommendations
            risk_level = risk_summary.get('risk_level', 'low')
            
            print(f"\n💡 Risk Recommendations:")
            if risk_level == 'low':
                print("   ✅ Risk level is acceptable")
            elif risk_level == 'medium':
                print("   ⚠️  Monitor positions closely")
            elif risk_level == 'high':
                print("   🚨 Consider reducing position sizes")
            elif risk_level == 'critical':
                print("   🚨 IMMEDIATE ACTION REQUIRED")
                print("   🚨 Consider emergency position closure")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in risk analysis: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop all trading activities"""
        print("🚨 EMERGENCY STOP PROCEDURE")
        print("=" * 40)
        
        confirmation = input("Type 'EMERGENCY STOP' to confirm: ")
        if confirmation != "EMERGENCY STOP":
            print("❌ Emergency stop cancelled")
            return False
        
        try:
            print("🛑 Initiating emergency stop...")
            
            # Initialize risk manager
            trading_config = Settings.get_trading_config()
            risk_config = Settings.get_risk_config()
            rm = RiskManager(trading_config, risk_config, Settings.DATA_DIR)
            
            # Get list of positions to close
            symbols_to_close = rm.emergency_close_all_positions()
            
            if symbols_to_close:
                print(f"📋 Positions to close: {', '.join(symbols_to_close)}")
                print("⚠️  Please manually close these positions in your trading terminal")
                print("⚠️  The bot will not place any new orders")
            else:
                print("✅ No open positions found")
            
            print("🛑 Emergency stop completed")
            print("💡 Restart the bot when ready to resume trading")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in emergency stop: {e}")
            return False

def main():
    """Enhanced CLI main function"""
    parser = argparse.ArgumentParser(
        description='Enhanced SuperTrend Trading Bot CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py auth                    # Authenticate with Zerodha
  python cli.py test                    # Test connection and configuration  
  python cli.py trade                   # Start paper trading
  python cli.py trade --live            # Start live trading (with confirmation)
  python cli.py analyze                 # Analyze performance
  python cli.py validate               # Validate market data
  python cli.py risk                   # Risk analysis
  python cli.py emergency              # Emergency stop
        """
    )
    
    parser.add_argument(
        'command',
        choices=['auth', 'test', 'trade', 'analyze', 'validate', 'risk', 'emergency'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading mode (requires confirmation)'
    )
    
    parser.add_argument(
        '--instrument',
        type=str,
        default='NIFTYBEES',
        help='Trading instrument (default: NIFTYBEES)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days for analysis (default: 30)'
    )
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n🚀 Enhanced SuperTrend Trading Bot")
        print("=" * 40)
        print("New Features:")
        print("  • Advanced risk management")
        print("  • Real-time performance monitoring") 
        print("  • Comprehensive data validation")
        print("  • Enhanced error handling")
        print("  • Professional logging system")
        return
    
    args = parser.parse_args()
    cli = EnhancedCLI()
    
    try:
        if args.command == 'auth':
            success = cli.authenticate()
        elif args.command == 'test':
            success = cli.test_connection()
        elif args.command == 'trade':
            success = cli.start_trading(live_mode=args.live, instrument=args.instrument)
        elif args.command == 'analyze':
            success = cli.analyze_performance(days=args.days)
        elif args.command == 'validate':
            success = cli.validate_data(symbol=args.instrument)
        elif args.command == 'risk':
            success = cli.risk_analysis()
        elif args.command == 'emergency':
            success = cli.emergency_stop()
        else:
            parser.print_help()
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"CLI error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()