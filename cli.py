#!/usr/bin/env python3
"""
Enhanced CLI for SuperTrend Trading Bot with Multi-Strategy Support
Integrates with all components: risk management, performance monitoring, multi-strategy system
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
    
    def strategy_info(self, strategy_key: str):
        """Show detailed information about a strategy"""
        if not MULTI_STRATEGY_AVAILABLE:
            print("❌ Multi-strategy system not available")
            return False
            
        print(f"📊 Strategy Information: {strategy_key}")
        print("=" * 50)
        
        try:
            detailed_info = StrategyFactory.get_strategy_info(strategy_key)
            
            # Factory information
            factory_info = detailed_info['factory_info']
            print(f"🏭 Factory Information:")
            print(f"   Name: {factory_info['name']}")
            print(f"   Description: {factory_info['description']}")
            
            # Strategy metadata
            if 'strategy_metadata' in detailed_info:
                metadata = detailed_info['strategy_metadata']['metadata']
                print(f"\n📈 Strategy Metadata:")
                print(f"   Type: {metadata['type']}")
                print(f"   Risk Level: {metadata['risk_level']}")
                print(f"   Version: {metadata['version']}")
                print(f"   Recommended Timeframes: {', '.join(metadata['recommended_timeframes'])}")
                print(f"   Recommended Instruments: {', '.join(metadata['recommended_instruments'])}")
                
                # Parameters
                if 'parameters' in detailed_info['strategy_metadata']:
                    params = detailed_info['strategy_metadata']['parameters']
                    print(f"\n⚙️  Current Parameters:")
                    for param_name, param_value in params.items():
                        print(f"   {param_name}: {param_value}")
                
                # Performance data
                if 'backtested_performance' in metadata and metadata['backtested_performance']:
                    perf = metadata['backtested_performance']
                    print(f"\n📊 Backtested Performance:")
                    print(f"   Win Rate: {perf.get('win_rate', 0):.1%}")
                    print(f"   Profit Factor: {perf.get('profit_factor', 0):.1f}")
                    print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
                    print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting strategy info: {e}")
            return False
    
    def compare_strategies(self, strategies_to_compare=None):
        """Compare multiple strategies"""
        if not MULTI_STRATEGY_AVAILABLE:
            print("❌ Multi-strategy system not available")
            return False
            
        print("🔍 Strategy Comparison")
        print("=" * 50)
        
        if not strategies_to_compare:
            strategies_to_compare = ['enhanced', 'bullet']
        
        try:
            # Get some test data
            if not self.auth.test_connection():
                print("❌ Need authentication for strategy comparison")
                return False
            
            kite = self.auth.get_kite_instance()
            
            # Fetch test data (NIFTY 50)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)
            
            test_data = kite.historical_data(
                "256265",  # NIFTY 50 token
                start_date,
                end_date,
                "minute"
            )
            
            if not test_data:
                print("❌ Could not fetch test data")
                return False
            
            df = pd.DataFrame(test_data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            print(f"📊 Test Data: {len(df)} candles from {start_date.date()} to {end_date.date()}")
            
            # Compare strategies
            comparison_results = strategy_manager.compare_strategies(strategies_to_compare, df)
            
            print(f"\n📈 Strategy Comparison Results:")
            print("=" * 60)
            
            for strategy_key, result in comparison_results.items():
                print(f"\n🎯 {strategy_key.upper()}:")
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                    continue
                
                print(f"   Name: {result['strategy_name']}")
                print(f"   Signal: {result['signal']}")
                
                if 'signal_data' in result:
                    signal_data = result['signal_data']
                    print(f"   Confidence: {signal_data.get('confidence', 0):.1%}")
                    
                    if 'quality' in signal_data:
                        print(f"   Quality: {signal_data['quality']}")
                    
                    if 'risk_level' in signal_data:
                        print(f"   Risk Level: {signal_data['risk_level']}")
                    
                    if 'warnings' in signal_data and signal_data['warnings']:
                        print(f"   Warnings: {len(signal_data['warnings'])} warnings")
                
                if 'health' in result:
                    health = result['health']
                    print(f"   Health Status: {health.get('status', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error comparing strategies: {e}")
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
                        
                        # Get detailed strategy info
                        try:
                            detailed_info = StrategyFactory.get_strategy_info(strategy)
                            if 'strategy_metadata' in detailed_info:
                                metadata = detailed_info['strategy_metadata']['metadata']
                                print(f"   🏷️  Type: {metadata['type']}")
                                print(f"   ⚠️  Risk Level: {metadata['risk_level']}")
                                
                                if 'backtested_performance' in metadata and metadata['backtested_performance']:
                                    perf = metadata['backtested_performance']
                                    print(f"   📊 Expected Performance:")
                                    print(f"      Win Rate: {perf.get('win_rate', 0):.1%}")
                                    print(f"      Profit Factor: {perf.get('profit_factor', 0):.1f}")
                        except Exception as e:
                            logger.debug(f"Could not get detailed strategy info: {e}")
                        
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
            
            # Override strategy if using multi-strategy system
            if MULTI_STRATEGY_AVAILABLE and strategy != "enhanced":
                try:
                    print(f"🎯 Loading {strategy} strategy...")
                    strategy_config = self._get_strategy_config(strategy)
                    trading_strategy = StrategyFactory.create_strategy(strategy, strategy_config)
                    bot.strategy = trading_strategy
                    print(f"✅ {strategy} strategy loaded successfully")
                    
                    # Display strategy parameters if available
                    if hasattr(trading_strategy, 'get_parameter_info'):
                        param_info = trading_strategy.get_parameter_info()
                        if param_info:
                            print(f"\n⚙️  Strategy Parameters:")
                            for param_name, param_data in list(param_info.items())[:5]:  # Show first 5
                                current_val = param_data.get('current_value', 'N/A')
                                desc = param_data.get('description', 'No description')[:50]
                                print(f"   {param_name}: {current_val} - {desc}")
                    
                except Exception as e:
                    print(f"⚠️  Could not load {strategy} strategy: {e}")
                    print("🔄 Using default enhanced strategy")
            
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
    
    def _get_strategy_config(self, strategy_key: str) -> dict:
        """Get configuration for selected strategy"""
        try:
            # Get base trading config
            trading_config = Settings.get_trading_config()
            strategy_config = Settings.get_strategy_config()
            
            # Base configuration that all strategies can use
            base_config = {
                'account_balance': trading_config.account_balance,
                'risk_per_trade': trading_config.risk_per_trade,
                'min_candles': strategy_config.min_candles_required
            }
            
            # Strategy-specific configurations
            if strategy_key == 'enhanced':
                base_config.update({
                    'atr_period': strategy_config.atr_period,
                    'factor': strategy_config.factor,
                    'adaptive_mode': strategy_config.adaptive_mode
                })
            
            elif strategy_key == 'bullet':
                base_config.update({
                    'base_atr_period': strategy_config.atr_period,
                    'base_factor': strategy_config.factor,
                    'adaptive_mode': strategy_config.adaptive_mode,
                    'quality_threshold': strategy_config.confidence_threshold
                })
            
            return base_config
            
        except Exception as e:
            logger.error(f"Error getting strategy config: {e}")
            return {}
    
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
            start_date = end_date - timedelta(days=3)
            
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
            print