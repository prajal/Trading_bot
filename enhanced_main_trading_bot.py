#!/usr/bin/env python3
"""
Enhanced SuperTrend Trading Bot
Integrates all improvements: risk management, error handling, performance monitoring
"""

import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path

# Enhanced imports
from auth.enhanced_kite_auth import KiteAuth, AuthenticationError
from trading.enhanced_strategy import EnhancedSuperTrendStrategy
from trading.enhanced_executor import EnhancedOrderExecutor
from config.enhanced_settings import Settings, ConfigurationError
from utils.enhanced_logger import get_logger, log_session_start, log_session_end
from utils.enhanced_risk_manager import RiskManager, PositionSizeMethod
from utils.performance_monitor import PerformanceMonitor
from utils.market_data_validator import MarketDataValidator

logger = get_logger(__name__)

# Trading instruments configuration
TRADING_INSTRUMENTS = {
    'NIFTY 50': {
        'name': 'NIFTY 50 Index',
        'token': '256265',
        'exchange': 'NSE',
        'mis_leverage': 1.0,
        'description': 'NIFTY 50 Index for signal generation'
    },
    'NIFTYBEES': {
        'name': 'Nippon India ETF Nifty 50 BeES',
        'token': '2707457',
        'exchange': 'NSE',
        'mis_leverage': 5.0,
        'description': 'Tracks NIFTY 50 Index - Primary trading instrument'
    },
    'BANKBEES': {
        'name': 'Nippon India ETF Nifty Bank BeES',
        'token': '2954241',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'Tracks NIFTY Bank Index'
    },
    'JUNIORBEES': {
        'name': 'Nippon India ETF Junior BeES',
        'token': '4632577',
        'exchange': 'NSE',
        'mis_leverage': 5.0,
        'description': 'Tracks NIFTY Next 50 Index'
    },
    'GOLDBEES': {
        'name': 'Nippon India ETF Gold BeES',
        'token': '2800641',
        'exchange': 'NSE',
        'mis_leverage': 3.0,
        'description': 'Tracks Gold prices'
    }
}

class EnhancedTradingBot:
    """
    Enhanced Trading Bot with comprehensive risk management,
    error handling, and performance monitoring
    """
    
    def __init__(self):
        # Load configuration
        try:
            self.trading_config = Settings.get_trading_config()
            self.strategy_config = Settings.get_strategy_config()
            self.risk_config = Settings.get_risk_config()
            self.safety_config = Settings.get_safety_config()
            self.market_hours = Settings.get_market_hours()
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        
        # Initialize components
        self.auth = KiteAuth()
        self.strategy = EnhancedSuperTrendStrategy(
            atr_period=self.strategy_config.atr_period,
            factor=self.strategy_config.factor,
            adaptive_mode=self.strategy_config.adaptive_mode
        )
        
        # Will be initialized in setup()
        self.executor: Optional[EnhancedOrderExecutor] = None
        self.risk_manager: Optional[RiskManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.data_validator: Optional[MarketDataValidator] = None
        
        # Trading state
        self.position = {
            "quantity": 0,
            "entry_price": 0,
            "entry_time": None,
            "symbol": None,
            "token": None,
            "tradingsymbol": None,
            "instrument_token": None,
            "pnl": 0,
            "stop_loss_price": None,
            "trailing_stop_price": None
        }
        
        # Session tracking
        self.session_start_time: Optional[datetime] = None
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.loop_count = 0
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=5)
        
        # Signal tracking
        self.last_signal = None
        self.last_signal_time = None
        self.signal_history: List[Dict[str, Any]] = []
        
        # Circuit breaker state
        self.emergency_stop_triggered = False
        self.emergency_stop_reason = ""
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Enhanced Trading Bot initialized - Session: {self.session_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n⏹️  Shutdown signal received (signal: {signum})")
        self._graceful_shutdown()
        sys.exit(0)
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown with cleanup"""
        try:
            logger.info("Starting graceful shutdown...")
            
            # Check and handle open positions
            if self.position["quantity"] > 0:
                logger.warning("⚠️  Open position detected during shutdown")
                
                if self.executor and self.risk_manager:
                    # Sync with broker to get latest status
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    
                    if self.position["quantity"] > 0:
                        logger.warning(f"Position still open: {self.position['quantity']} {self.position['tradingsymbol']}")
                        logger.warning("💡 Please manually close the position if needed")
                    else:
                        logger.info("✅ Position was already closed externally")
            
            # Generate final reports
            self._generate_session_summary()
            
            # Cleanup components
            if self.performance_monitor:
                self.performance_monitor.save_session_data()
            
            if self.risk_manager:
                self.risk_manager._save_performance_data()
            
            if self.executor:
                self.executor._save_execution_history()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    
    def setup(self) -> bool:
        """Setup trading bot with comprehensive validation"""
        try:
            logger.info("Setting up Enhanced Trading Bot...")
            
            # 1. Validate configuration
            validation_results = Settings.validate_all_configuration()
            if not validation_results['valid']:
                for error in validation_results['errors']:
                    logger.error(f"Configuration error: {error}")
                return False
            
            for warning in validation_results['warnings']:
                logger.warning(f"Configuration warning: {warning}")
            
            # 2. Setup authentication
            logger.info("Establishing Kite connection...")
            kite = self.auth.get_kite_instance()
            if not kite:
                logger.error("Failed to establish Kite connection")
                return False
            
            # 3. Initialize components
            logger.info("Initializing trading components...")
            
            # Order executor
            self.executor = EnhancedOrderExecutor(
                kite=kite, 
                safety_config=self.safety_config,
                data_dir=Settings.DATA_DIR
            )
            
            # Risk manager
            self.risk_manager = RiskManager(
                trading_config=self.trading_config,
                risk_config=self.risk_config,
                data_dir=Settings.DATA_DIR
            )
            
            # Performance monitor
            self.performance_monitor = PerformanceMonitor(
                data_dir=Settings.DATA_DIR,
                trading_config=self.trading_config
            )
            
            # Data validator
            self.data_validator = MarketDataValidator()
            
            # 4. Test critical functions
            logger.info("Testing critical functions...")
            
            # Test data fetching
            test_data = self.executor.get_historical_data_with_retry(
                TRADING_INSTRUMENTS['NIFTYBEES']['token'],
                datetime.now() - timedelta(days=1),
                datetime.now(),
                "minute"
            )
            
            if test_data.empty:
                logger.error("Failed to fetch test data - market data connection issue")
                return False
            
            # Test price fetching
            test_price = self.executor.get_latest_price_with_retry(
                TRADING_INSTRUMENTS['NIFTYBEES']['token']
            )
            
            if not test_price:
                logger.error("Failed to fetch current price - real-time data issue")
                return False
            
            # 5. Display configuration summary
            Settings.print_configuration_summary()
            
            logger.info("✅ Enhanced Trading Bot setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is open with enhanced validation"""
        try:
            now = datetime.now()
            
            # Check if it's a weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's a market holiday (basic implementation)
            # In production, this should check against a holiday calendar
            current_time = now.time()
            
            # Market hours
            market_open = datetime.strptime(
                f"{self.market_hours['open_hour']:02d}:{self.market_hours['open_minute']:02d}", 
                "%H:%M"
            ).time()
            
            market_close = datetime.strptime(
                f"{self.market_hours['close_hour']:02d}:{self.market_hours['close_minute']:02d}", 
                "%H:%M"
            ).time()
            
            return market_open <= current_time <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            if datetime.now() - self.last_health_check < self.health_check_interval:
                return
            
            logger.debug("Performing system health check...")
            
            # 1. Check authentication
            if not self.auth._validate_connection():
                logger.warning("Authentication health check failed")
                return
            
            # 2. Check risk limits
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                
                if risk_summary.get('circuit_breaker_active'):
                    logger.warning("🚨 Circuit breaker is active")
                
                if risk_summary.get('risk_level') == 'critical':
                    logger.warning("🚨 Risk level is CRITICAL")
            
            # 3. Check performance
            if self.performance_monitor:
                perf_summary = self.performance_monitor.get_current_metrics()
                
                if perf_summary.get('current_drawdown', 0) < -0.1:  # 10% drawdown
                    logger.warning(f"⚠️  High drawdown: {perf_summary.get('current_drawdown', 0):.1%}")
            
            # 4. Check execution quality
            if self.executor:
                exec_summary = self.executor.get_execution_summary()
                
                if exec_summary.get('success_rate', 0) < 90:  # Less than 90% success rate
                    logger.warning(f"⚠️  Low execution success rate: {exec_summary.get('success_rate', 0):.1f}%")
            
            self.last_health_check = datetime.now()
            logger.debug("Health check completed")
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        try:
            if not self.risk_manager:
                return False
            
            # Check circuit breaker
            risk_summary = self.risk_manager.get_risk_summary()
            
            if risk_summary.get('circuit_breaker_active'):
                if not self.emergency_stop_triggered:
                    self.emergency_stop_triggered = True
                    self.emergency_stop_reason = "Circuit breaker activated"
                    logger.critical(f"🚨 EMERGENCY STOP: {self.emergency_stop_reason}")
                return True
            
            # Check critical risk level
            if risk_summary.get('risk_level') == 'critical':
                if not self.emergency_stop_triggered:
                    self.emergency_stop_triggered = True
                    self.emergency_stop_reason = "Critical risk level reached"
                    logger.critical(f"🚨 EMERGENCY STOP: {self.emergency_stop_reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return False
    
    def _get_nifty50_data_for_signals(self) -> Optional[pd.DataFrame]:
        """Get NIFTY 50 data for signal generation with validation"""
        try:
            if not self.executor:
                logger.error("Order executor not initialized")
                return None
            
            nifty50_token = TRADING_INSTRUMENTS['NIFTY 50']['token']
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.trading_config.account_balance)  # Use config value
            
            logger.debug(f"Fetching NIFTY 50 data: {from_date} to {to_date}")
            
            df = self.executor.get_historical_data_with_retry(
                nifty50_token, from_date, to_date, "minute"
            )
            
            if df.empty:
                logger.warning("No NIFTY 50 data received")
                return None
            
            # Validate data quality
            if self.data_validator:
                if not self.data_validator.validate_ohlc_data(df):
                    logger.warning("NIFTY 50 data failed validation")
                    return None
            
            # Check minimum candles requirement
            if len(df) < self.strategy_config.min_candles_required:
                logger.warning(f"Insufficient NIFTY 50 data: {len(df)} < {self.strategy_config.min_candles_required}")
                return None
            
            logger.debug(f"NIFTY 50 data loaded successfully: {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching NIFTY 50 data: {e}")
            return None
    
    def _execute_trade_with_risk_management(self, signal: str, signal_data: dict, 
                                          trading_symbol: str, current_price: float):
        """Execute trade with comprehensive risk management"""
        try:
            # Check emergency conditions first
            if self._check_emergency_conditions():
                logger.warning("Trade blocked due to emergency conditions")
                return
            
            # ENTRY LOGIC
            if signal == "BUY" and self.position["quantity"] == 0:
                
                # Validate trade with risk manager
                if self.risk_manager:
                    is_valid, reason = self.risk_manager.validate_trade(
                        trading_symbol, "BUY", 1, current_price  # Preliminary validation
                    )
                    
                    if not is_valid:
                        logger.warning(f"Trade blocked by risk manager: {reason}")
                        return
                    
                    # Calculate optimal position size
                    confidence = signal_data.get('confidence', 1.0)
                    volatility = signal_data.get('volatility')
                    
                    quantity, calc_details = self.risk_manager.calculate_position_size(
                        trading_symbol, current_price, confidence, volatility
                    )
                    
                    if quantity <= 0:
                        logger.warning("Position size calculation resulted in zero quantity")
                        return
                    
                    # Final trade validation with actual quantity
                    is_valid, reason = self.risk_manager.validate_trade(
                        trading_symbol, "BUY", quantity, current_price
                    )
                    
                    if not is_valid:
                        logger.warning(f"Final trade validation failed: {reason}")
                        return
                else:
                    # Fallback position sizing
                    quantity = self._calculate_fallback_position_size(trading_symbol, current_price)
                
                # Log trade intention
                logger.info("🟢 ENHANCED BUY SIGNAL DETECTED")
                logger.info(f"📊 Signal Details:")
                for key, value in signal_data.items():
                    logger.info(f"   {key}: {value}")
                
                logger.info(f"📋 Trade Execution Plan:")
                logger.info(f"   Symbol: {trading_symbol}")
                logger.info(f"   Quantity: {quantity}")
                logger.info(f"   Price: ₹{current_price:.2f}")
                logger.info(f"   Position Value: ₹{quantity * current_price:,.2f}")
                
                if self.risk_manager:
                    logger.info(f"   Risk Calculation Details:")
                    for key, value in calc_details.items():
                        logger.info(f"     {key}: {value}")
                
                # Execute the order
                order_id = self.executor.place_order_with_retry(
                    trading_symbol, "BUY", quantity, current_price
                )
                
                if order_id:
                    # Get actual fill price
                    fill_price = self.executor.get_order_filled_price(order_id)
                    entry_price = fill_price if fill_price else current_price
                    
                    # Update position
                    self.position = {
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "entry_time": datetime.now(),
                        "symbol": trading_symbol,
                        "tradingsymbol": trading_symbol,
                        "token": TRADING_INSTRUMENTS[trading_symbol]['token'],
                        "instrument_token": TRADING_INSTRUMENTS[trading_symbol]['token'],
                        "pnl": 0
                    }
                    
                    # Calculate stop loss
                    if self.risk_manager:
                        stop_loss_price = self.risk_manager.calculate_stop_loss(
                            trading_symbol, entry_price, "BUY", 
                            signal_data.get('atr')
                        )
                        self.position["stop_loss_price"] = stop_loss_price
                        
                        # Update risk manager
                        self.risk_manager.update_position(
                            trading_symbol, quantity, entry_price, current_price, datetime.now()
                        )
                    
                    # Log trade execution
                    logger.log_trade(
                        symbol=trading_symbol,
                        action="BUY",
                        quantity=quantity,
                        price=entry_price,
                        order_id=order_id
                    )
                    
                    logger.info(f"✅ POSITION OPENED: {quantity} {trading_symbol} @ ₹{entry_price:.2f}")
                    logger.info(f"📍 Stop Loss: ₹{self.position.get('stop_loss_price', 'N/A')}")
                    
                    # Update performance monitor
                    if self.performance_monitor:
                        self.performance_monitor.record_trade_entry(
                            trading_symbol, quantity, entry_price, datetime.now()
                        )
                else:
                    logger.error("Failed to execute BUY order")
            
            # EXIT LOGIC
            elif signal == "SELL" and self.position["quantity"] > 0:
                
                # Sync position first
                if self.executor:
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    
                    if self.position["quantity"] == 0:
                        logger.info("Position already closed externally")
                        return
                
                logger.info("🔴 ENHANCED SELL SIGNAL DETECTED")
                logger.info(f"📊 Signal Details:")
                for key, value in signal_data.items():
                    logger.info(f"   {key}: {value}")
                
                # Execute sell order
                order_id = self.executor.place_order_with_retry(
                    self.position["tradingsymbol"], "SELL", self.position["quantity"], current_price
                )
                
                if order_id:
                    # Get actual fill price
                    fill_price = self.executor.get_order_filled_price(order_id)
                    exit_price = fill_price if fill_price else current_price
                    
                    # Calculate P&L
                    realized_pnl = (exit_price - self.position["entry_price"]) * self.position["quantity"]
                    
                    # Log trade exit
                    logger.log_trade(
                        symbol=self.position["tradingsymbol"],
                        action="SELL",
                        quantity=self.position["quantity"],
                        price=exit_price,
                        order_id=order_id,
                        pnl=realized_pnl
                    )
                    
                    logger.info(f"📉 POSITION CLOSED: P&L = ₹{realized_pnl:.2f}")
                    
                    # Update risk manager
                    if self.risk_manager:
                        self.risk_manager.close_position(
                            self.position["tradingsymbol"], exit_price, datetime.now()
                        )
                    
                    # Update performance monitor
                    if self.performance_monitor:
                        self.performance_monitor.record_trade_exit(
                            self.position["tradingsymbol"], exit_price, datetime.now(), realized_pnl
                        )
                    
                    # Reset position
                    self._reset_position()
                else:
                    logger.error("Failed to execute SELL order")
            
            # POSITION MONITORING
            elif self.position["quantity"] > 0:
                
                # Sync position
                if self.executor:
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    
                    if self.position["quantity"] == 0:
                        return  # Position closed externally
                
                # Update current position data
                current_pnl = (current_price - self.position["entry_price"]) * self.position["quantity"]
                self.position["pnl"] = current_pnl
                
                # Update risk manager
                if self.risk_manager:
                    self.risk_manager.update_position(
                        self.position["tradingsymbol"], 
                        self.position["quantity"],
                        self.position["entry_price"],
                        current_price,
                        self.position["entry_time"]
                    )
                    
                    # Check stop loss triggers
                    triggered_stops = self.risk_manager.check_stop_loss_triggers()
                    
                    if triggered_stops:
                        for stop in triggered_stops:
                            logger.warning(f"🛑 {stop['reason']} triggered for {stop['symbol']}")
                            
                            # Execute stop loss
                            order_id = self.executor.place_order_with_retry(
                                stop['symbol'], "SELL", stop['position'].quantity, stop['current_price']
                            )
                            
                            if order_id:
                                realized_pnl = self.risk_manager.close_position(
                                    stop['symbol'], stop['current_price'], datetime.now()
                                )
                                
                                logger.info(f"📉 STOP LOSS EXECUTED: P&L = ₹{realized_pnl:.2f}")
                                self._reset_position()
                            break
                
                # Log position status
                pnl_percent = (current_pnl / (self.position["entry_price"] * self.position["quantity"])) * 100
                
                logger.info(f"📊 Position Status: {self.position['quantity']} {self.position['tradingsymbol']}")
                logger.info(f"💰 P&L: ₹{current_pnl:.2f} ({pnl_percent:+.2f}%)")
                logger.info(f"📈 Entry: ₹{self.position['entry_price']:.2f} | Current: ₹{current_price:.2f}")
                logger.info(f"🛡️  Stop Loss: ₹{self.position.get('stop_loss_price', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _calculate_fallback_position_size(self, symbol: str, price: float) -> int:
        """Fallback position size calculation"""
        try:
            available_capital = self.trading_config.account_balance * 0.1  # Conservative 10%
            leverage = TRADING_INSTRUMENTS.get(symbol, {}).get('mis_leverage', 3.0)
            
            effective_capital = available_capital * leverage
            quantity = int(effective_capital / price)
            
            return max(1, quantity)
            
        except Exception as e:
            logger.error(f"Error in fallback position sizing: {e}")
            return 1
    
    def _reset_position(self):
        """Reset position tracking"""
        self.position = {
            "quantity": 0,
            "entry_price": 0,
            "entry_time": None,
            "symbol": None,
            "token": None,
            "tradingsymbol": None,
            "instrument_token": None,
            "pnl": 0,
            "stop_loss_price": None,
            "trailing_stop_price": None
        }
        logger.info("Position tracking reset")
    
    def _generate_session_summary(self):
        """Generate comprehensive session summary"""
        try:
            session_duration = datetime.now() - self.session_start_time if self.session_start_time else timedelta(0)
            
            summary = {
                'session_id': self.session_id,
                'session_duration': str(session_duration).split('.')[0],
                'loop_iterations': self.loop_count,
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'emergency_stop_reason': self.emergency_stop_reason,
                'final_position': self.position,
                'trading_mode': 'LIVE' if self.safety_config.live_trading_enabled else 'DRY_RUN'
            }
            
            # Add risk summary
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                summary['risk_metrics'] = risk_summary
            
            # Add performance summary
            if self.performance_monitor:
                perf_summary = self.performance_monitor.get_session_summary()
                summary['performance_metrics'] = perf_summary
            
            # Add execution summary
            if self.executor:
                exec_summary = self.executor.get_execution_summary()
                summary['execution_metrics'] = exec_summary
            
            # Log session summary
            log_session_end(summary)
            
            # Display summary
            print("\n" + "="*60)
            print("📊 ENHANCED TRADING SESSION SUMMARY")
            print("="*60)
            print(f"Session ID: {self.session_id}")
            print(f"Duration: {session_duration}")
            print(f"Trading Mode: {'LIVE' if self.safety_config.live_trading_enabled else 'DRY RUN'}")
            print(f"Loop Iterations: {self.loop_count}")
            
            if self.emergency_stop_triggered:
                print(f"🚨 Emergency Stop: {self.emergency_stop_reason}")
            
            # Display key metrics
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                print(f"\n💼 Risk Metrics:")
                print(f"   Risk Level: {risk_summary.get('risk_level', 'Unknown')}")
                print(f"   Daily P&L: ₹{risk_summary.get('daily_pnl', 0):.2f}")
                print(f"   Max Drawdown: {risk_summary.get('max_drawdown', 0):.1%}")
            
            if self.performance_monitor:
                perf_summary = self.performance_monitor.get_session_summary()
                print(f"\n📈 Performance Metrics:")
                print(f"   Total Trades: {perf_summary.get('total_trades', 0)}")
                print(f"   Win Rate: {perf_summary.get('win_rate', 0):.1%}")
                print(f"   Total P&L: ₹{perf_summary.get('total_pnl', 0):.2f}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
    
    def run(self, instrument_symbol: str = 'NIFTYBEES'):
        """Run enhanced trading bot"""
        try:
            # Validate instrument
            if instrument_symbol not in TRADING_INSTRUMENTS:
                logger.error(f"Unknown instrument: {instrument_symbol}")
                return
            
            instrument = TRADING_INSTRUMENTS[instrument_symbol]
            self.session_start_time = datetime.now()
            
            # Log session start
            session_config = {
                'instrument': instrument_symbol,
                'trading_config': self.trading_config.__dict__,
                'strategy_config': self.strategy_config.__dict__,
                'risk_config': self.risk_config.__dict__,
                'safety_config': self.safety_config.__dict__
            }
            log_session_start(session_config)
            
            logger.info("🚀 Starting Enhanced SuperTrend Trading Bot")
            logger.info(f"📊 Configuration:")
            logger.info(f"   Signal Source: NIFTY 50 Index")
            logger.info(f"   Trading Instrument: {instrument_symbol} - {instrument['name']}")
            logger.info(f"   Account Balance: ₹{self.trading_config.account_balance:,.2f}")
            logger.info(f"   Position Sizing: {self.trading_config.position_sizing_method}")
            logger.info(f"   Risk per Trade: {self.trading_config.risk_per_trade:.1%}")
            logger.info(f"   Max Daily Loss: {self.trading_config.max_daily_loss:.1%}")
            logger.info(f"   Trading Mode: {'LIVE' if self.safety_config.live_trading_enabled else 'DRY RUN'}")
            
            # Reset daily counters if needed
            if self.risk_manager:
                self.risk_manager.reset_daily_metrics()
            
            if self.executor:
                self.executor.reset_daily_counters()
            
            # Check for existing positions
            self._check_existing_positions_on_startup()
            
            # Main trading loop
            logger.info("🔄 Entering main trading loop...")
            
            try:
                while True:
                    self.loop_count += 1
                    
                    try:
                        # Check if market is open
                        if not self.is_market_open():
                            logger.debug("Market is closed, waiting...")
                            time.sleep(60)
                            continue
                        
                        # Perform periodic health checks
                        self._perform_health_check()
                        
                        # Check emergency conditions
                        if self._check_emergency_conditions():
                            logger.critical("🚨 Emergency conditions detected - stopping trading")
                            break
                        
                        # Sync existing positions
                        if self.position["quantity"] > 0 and self.executor:
                            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                            
                            if sync_needed and sync_status == "CLOSED_EXTERNALLY":
                                logger.info("Position synchronized - was closed externally")
                                continue
                        
                        # Check for pre-market close exit
                        if (self.executor and self.executor.is_market_close_time() and 
                            self.position["quantity"] > 0):
                            
                            logger.warning("🕒 PRE-MARKET CLOSE - Exiting position")
                            current_price = self.executor.get_latest_price_with_retry(
                                self.position["instrument_token"]
                            )
                            
                            if current_price:
                                order_id = self.executor.place_order_with_retry(
                                    self.position["tradingsymbol"], "SELL", 
                                    self.position["quantity"], current_price
                                )
                                
                                if order_id:
                                    realized_pnl = (current_price - self.position["entry_price"]) * self.position["quantity"]
                                    logger.info(f"📉 PRE-CLOSE EXIT: P&L = ₹{realized_pnl:.2f}")
                                    
                                    if self.risk_manager:
                                        self.risk_manager.close_position(
                                            self.position["tradingsymbol"], current_price, datetime.now()
                                        )
                                    
                                    self._reset_position()
                            continue
                        
                        # Get signal data from NIFTY 50
                        signal_df = self._get_nifty50_data_for_signals()
                        
                        if signal_df is None or signal_df.empty:
                            logger.warning("Could not fetch NIFTY 50 signal data")
                            time.sleep(30)
                            continue
                        
                        # Validate strategy on first run
                        if not hasattr(self, '_strategy_validated'):
                            if self.strategy.validate_signal(signal_df):
                                self._strategy_validated = True
                                logger.info("✅ Strategy validation passed")
                            else:
                                logger.error("❌ Strategy validation failed")
                                time.sleep(60)
                                continue
                        
                        # Get trading signal
                        signal, signal_data = self.strategy.get_signal(
                            signal_df, 
                            has_position=(self.position["quantity"] > 0)
                        )
                        
                        # Get current execution price
                        current_price = self.executor.get_latest_price_with_retry(
                            instrument['token']
                        )
                        
                        if not current_price:
                            logger.warning(f"Could not fetch current price for {instrument_symbol}")
                            time.sleep(30)
                            continue
                        
                        # Add current price to signal data
                        signal_data['current_execution_price'] = current_price
                        
                        # Log market status (every 10th iteration or on signal)
                        if self.loop_count % 10 == 0 or signal in ["BUY", "SELL"]:
                            trend_status = signal_data.get('trend', 'Unknown')
                            direction = signal_data.get('direction', 'Unknown')
                            confidence = signal_data.get('confidence', 0)
                            
                            logger.info(f"📊 Market Status (Loop #{self.loop_count}):")
                            logger.info(f"   NIFTY 50 Trend: {trend_status} (Direction: {direction})")
                            logger.info(f"   {instrument_symbol} Price: ₹{current_price:.2f}")
                            logger.info(f"   Signal: {signal} (Confidence: {confidence:.2f})")
                            logger.info(f"   Position: {self.position['quantity']} shares")
                        
                        # Check for duplicate signals
                        current_time = datetime.now()
                        if signal in ["BUY", "SELL"]:
                            if (self.last_signal == signal and self.last_signal_time and
                                (current_time - self.last_signal_time).seconds < 120):
                                logger.debug(f"Ignoring duplicate {signal} signal")
                                time.sleep(30)
                                continue
                        
                        # Record signal in history
                        if signal != "HOLD":
                            signal_record = {
                                'timestamp': current_time.isoformat(),
                                'signal': signal,
                                'symbol': instrument_symbol,
                                'price': current_price,
                                'confidence': signal_data.get('confidence', 0),
                                'loop_count': self.loop_count
                            }
                            self.signal_history.append(signal_record)
                            
                            # Keep only last 100 signals
                            if len(self.signal_history) > 100:
                                self.signal_history = self.signal_history[-100:]
                        
                        # Execute trade logic
                        self._execute_trade_with_risk_management(
                            signal, signal_data, instrument_symbol, current_price
                        )
                        
                        # Update last signal tracking
                        if signal in ["BUY", "SELL"]:
                            self.last_signal = signal
                            self.last_signal_time = current_time
                        
                        # Update performance monitor
                        if self.performance_monitor:
                            self.performance_monitor.update_loop_metrics(
                                self.loop_count, signal, current_price, self.position
                            )
                        
                        # Sleep between iterations
                        time.sleep(30)  # 30 seconds between checks
                        
                    except KeyboardInterrupt:
                        raise  # Re-raise to be caught by outer try-except
                    except Exception as e:
                        logger.error(f"Error in trading loop iteration {self.loop_count}: {e}")
                        time.sleep(60)  # Wait longer on errors
                        continue
            
            except KeyboardInterrupt:
                logger.info("\n⏹️  Trading stopped by user")
            except Exception as e:
                logger.error(f"Critical error in main trading loop: {e}")
                self.emergency_stop_triggered = True
                self.emergency_stop_reason = f"Critical error: {e}"
            
            finally:
                self._graceful_shutdown()
        
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
            self._graceful_shutdown()
    
    def _check_existing_positions_on_startup(self):
        """Check for existing positions when bot starts"""
        try:
            if not self.executor:
                return
            
            positions = self.executor.kite.positions()
            day_positions = positions.get('day', [])
            
            for pos in day_positions:
                if pos['quantity'] != 0 and pos['exchange'] == 'NSE':
                    symbol = pos['tradingsymbol']
                    quantity = pos['quantity']
                    avg_price = pos['average_price']
                    pnl = pos['pnl']
                    
                    logger.warning(f"EXISTING POSITION FOUND: {quantity} {symbol} | Avg: ₹{avg_price} | P&L: ₹{pnl}")
                    
                    # Check if it's one of our known instruments
                    if symbol in TRADING_INSTRUMENTS:
                        self.position = {
                            "instrument_token": TRADING_INSTRUMENTS[symbol]['token'],
                            "tradingsymbol": symbol,
                            "quantity": quantity,
                            "entry_price": avg_price,
                            "pnl": pnl,
                            "entry_time": datetime.now(),  # Approximate
                            "symbol": symbol,
                            "token": TRADING_INSTRUMENTS[symbol]['token'],
                            "stop_loss_price": None,
                            "trailing_stop_price": None
                        }
                        
                        # Update risk manager
                        if self.risk_manager:
                            current_price = self.executor.get_latest_price_with_retry(
                                TRADING_INSTRUMENTS[symbol]['token']
                            )
                            if current_price:
                                self.risk_manager.update_position(
                                    symbol, quantity, avg_price, current_price, datetime.now()
                                )
                        
                        logger.info(f"✅ TOOK CONTROL OF EXISTING POSITION: {symbol}")
                        break
        
        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")

def main():
    """Main function with enhanced error handling"""
    try:
        print("🚀 Enhanced SuperTrend Trading Bot")
        print("=" * 50)
        
        # Initialize bot
        bot = EnhancedTradingBot()
        
        # Setup bot
        if not bot.setup():
            logger.error("❌ Bot setup failed")
            return False
        
        logger.info("✅ Bot setup completed successfully")
        
        # Start trading
        bot.run("NIFTYBEES")
        
        return True
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please check your .env file and configuration")
        return False
    except AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("💡 Please run authentication: python cli.py auth")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Shutdown requested by user")
        return True
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        print(f"❌ Critical error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)