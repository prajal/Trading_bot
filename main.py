#!/usr/bin/env python3
"""
Enhanced main trading application with configurable instruments
Fixes the index/ETF mismatch issue
"""

import time
import signal
import sys
from datetime import datetime, timedelta
from auth.kite_auth import KiteAuth
from trading.strategy import SuperTrendStrategy
from trading.executor import OrderExecutor
from config.settings import Settings
from utils.logger import get_logger
from typing import Optional, List, Dict, Any

logger = get_logger(__name__)

# Trading instruments configuration
TRADING_INSTRUMENTS = {
    'NIFTYBEES': {
        'name': 'Nippon India ETF Nifty 50 BeES',
        'token': '2707457',
        'exchange': 'NSE',
        'mis_leverage': 5.0,
        'description': 'Tracks NIFTY 50 Index'
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
    },
    'RELIANCE': {
        'name': 'Reliance Industries Ltd',
        'token': '738561',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'Large cap stock'
    },
    'TCS': {
        'name': 'Tata Consultancy Services Ltd',
        'token': '2953217',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'IT Large cap stock'
    },
    'HDFCBANK': {
        'name': 'HDFC Bank Ltd',
        'token': '341249',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'Banking Large cap stock'
    },
    'ICICIBANK': {
        'name': 'ICICI Bank Ltd',
        'token': '1270529',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'Banking Large cap stock'
    },
    'INFY': {
        'name': 'Infosys Ltd',
        'token': '408065',
        'exchange': 'NSE',
        'mis_leverage': 4.0,
        'description': 'IT Large cap stock'
    }
}

class TradingBot:
    """Main trading bot class with configurable instruments"""
    
    def __init__(self):
        self.auth = KiteAuth()
        self.strategy = SuperTrendStrategy(
            atr_period=Settings.STRATEGY_PARAMS['atr_period'],
            factor=Settings.STRATEGY_PARAMS['factor']
        )
        self.executor: Optional[OrderExecutor] = None
        self.position = {
            "quantity": 0,
            "entry_price": 0,
            "entry_time": None,
            "symbol": None,
            "token": None,
            "tradingsymbol": None,
            "instrument_token": None,
            "pnl": 0
        }
        
        # Track last signal to avoid duplicates
        self.last_signal = None
        self.last_signal_time = None
        
        # Trading session tracking
        self.session_start_time = None
        self.session_trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        
        # MIS Leverage settings - now loaded from instruments config
        self.mis_leverage_map = {symbol: info['mis_leverage'] 
                                for symbol, info in TRADING_INSTRUMENTS.items()}
        self.mis_leverage_map['DEFAULT'] = 3.0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n⏹️  Shutdown signal received...")
        self._generate_trading_report()
        sys.exit(0)
    
    def _record_trade(self, entry_price: float, exit_price: float, quantity: int, 
                     entry_time: datetime, exit_time: datetime, exit_reason: str):
        """Record a completed trade"""
        pnl = (exit_price - entry_price) * quantity - (2 * Settings.STRATEGY_PARAMS.get('commission', 20))
        
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'exit_reason': exit_reason,
            'duration': str(exit_time - entry_time).split('.')[0]  # Remove microseconds
        }
        
        self.session_trades.append(trade)
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
            self.max_profit = max(self.max_profit, pnl)
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
            self.max_loss = min(self.max_loss, pnl)
    
    def _generate_trading_report(self):
        """Generate comprehensive trading report"""
        if not self.session_start_time:
            return
        
        session_duration = datetime.now() - self.session_start_time
        
        print("\n" + "=" * 60)
        print("📊 TRADING SESSION REPORT")
        print("=" * 60)
        
        # Session Info
        print(f"\n📅 Session Details:")
        print(f"   Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {str(session_duration).split('.')[0]}")
        print(f"   Mode: {'LIVE' if Settings.SAFETY_CONFIG['live_trading_enabled'] else 'DRY RUN'}")
        
        # Account Info
        print(f"\n💰 Account:")
        print(f"   Trading Amount: ₹{Settings.STRATEGY_PARAMS['account_balance']:,.2f}")
        print(f"   Instrument: {self.position.get('symbol', 'N/A')}")
        print(f"   Leverage Used: {self.mis_leverage_map.get(self.position.get('symbol', 'DEFAULT'), 3.0)}x")
        
        # Trading Summary
        print(f"\n📈 Trading Summary:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.losing_trades}")
        
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
            
            # P&L Summary
            print(f"\n💵 Profit & Loss:")
            print(f"   Total P&L: ₹{self.total_pnl:,.2f}")
            print(f"   Gross Profit: ₹{self.gross_profit:,.2f}")
            print(f"   Gross Loss: ₹{self.gross_loss:,.2f}")
            
            if self.gross_loss > 0:
                profit_factor = self.gross_profit / self.gross_loss
                print(f"   Profit Factor: {profit_factor:.2f}")
            
            print(f"   Best Trade: ₹{self.max_profit:,.2f}")
            print(f"   Worst Trade: ₹{self.max_loss:,.2f}")
            print(f"   Average P&L: ₹{self.total_pnl / self.total_trades:,.2f}")
            
            # Return on Capital
            return_on_capital = (self.total_pnl / Settings.STRATEGY_PARAMS['account_balance']) * 100
            print(f"   Return on Capital: {return_on_capital:+.2f}%")
            
            # Trade Details
            if len(self.session_trades) > 0:
                print(f"\n📋 Trade Details:")
                print("-" * 60)
                for i, trade in enumerate(self.session_trades, 1):
                    entry_time = trade['entry_time'].strftime('%H:%M:%S')
                    exit_time = trade['exit_time'].strftime('%H:%M:%S')
                    print(f"   Trade {i}:")
                    print(f"      Time: {entry_time} → {exit_time} ({trade['duration']})")
                    print(f"      Price: ₹{trade['entry_price']:.2f} → ₹{trade['exit_price']:.2f}")
                    print(f"      Quantity: {trade['quantity']} shares")
                    print(f"      P&L: ₹{trade['pnl']:,.2f} ({trade['return_pct']:+.2f}%)")
                    print(f"      Exit: {trade['exit_reason']}")
        else:
            print("\n   No trades executed during this session")
        
        # Open Position Warning
        if self.position["quantity"] > 0:
            print(f"\n⚠️  WARNING: Open Position")
            print(f"   Symbol: {self.position['tradingsymbol']}")
            print(f"   Quantity: {self.position['quantity']}")
            print(f"   Entry Price: ₹{self.position['entry_price']:.2f}")
            print(f"   Entry Time: {self.position['entry_time'].strftime('%H:%M:%S')}")
        
        print("\n" + "=" * 60)
        print("💡 Report generated at session end")
        print("=" * 60 + "\n")
    
    def get_mis_leverage(self, symbol: str) -> float:
        """Get MIS leverage for a given symbol"""
        return self.mis_leverage_map.get(symbol, self.mis_leverage_map['DEFAULT'])
    
    def calculate_mis_quantity(self, symbol: str, price: float) -> int:
        """Calculate quantity considering MIS leverage"""
        capital = Settings.STRATEGY_PARAMS['account_balance'] * \
                 (Settings.STRATEGY_PARAMS['capital_allocation_percent'] / 100)
        
        mis_leverage = self.get_mis_leverage(symbol)
        effective_capital = capital * mis_leverage
        potential_quantity = int(effective_capital / price)
        margin_required = potential_quantity * (price / mis_leverage)
        
        if margin_required > capital:
            safe_quantity = int(capital / (price / mis_leverage))
            actual_margin = safe_quantity * (price / mis_leverage)
            
            logger.info(f"💰 MIS Calculation: {symbol}")
            logger.info(f"   Available Capital: ₹{capital:,.2f}")
            logger.info(f"   MIS Leverage: {mis_leverage}x")
            logger.info(f"   Price per share: ₹{price:.2f}")
            logger.info(f"   Margin per share: ₹{price/mis_leverage:.2f}")
            logger.info(f"   Safe Quantity: {safe_quantity} shares")
            logger.info(f"   Margin Required: ₹{actual_margin:,.2f}")
            logger.info(f"   Trade Value: ₹{safe_quantity * price:,.2f}")
            
            return safe_quantity
        else:
            logger.info(f"💰 MIS Calculation: {symbol}")
            logger.info(f"   Available Capital: ₹{capital:,.2f}")
            logger.info(f"   MIS Leverage: {mis_leverage}x")
            logger.info(f"   Price per share: ₹{price:.2f}")
            logger.info(f"   Margin per share: ₹{price/mis_leverage:.2f}")
            logger.info(f"   Calculated Quantity: {potential_quantity} shares")
            logger.info(f"   Margin Required: ₹{margin_required:,.2f}")
            logger.info(f"   Trade Value: ₹{potential_quantity * price:,.2f}")
            
            return potential_quantity
    
    def setup(self) -> bool:
        """Setup trading bot"""
        kite = self.auth.get_kite_instance()
        if not kite:
            logger.error("Failed to setup Kite connection")
            return False
        
        self.executor = OrderExecutor(kite)
        logger.info("Trading bot setup complete")
        return True
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        current_time = now.time()
        market_open = datetime.strptime(
            f"{Settings.STRATEGY_PARAMS['market_open_hour']}:{Settings.STRATEGY_PARAMS['market_open_minute']}", 
            "%H:%M"
        ).time()
        market_close = datetime.strptime(
            f"{Settings.STRATEGY_PARAMS['market_close_hour']}:{Settings.STRATEGY_PARAMS['market_close_minute']}", 
            "%H:%M"
        ).time()
        
        return market_open <= current_time <= market_close
    
    def check_existing_positions_on_startup(self):
        """Check for existing positions when bot starts"""
        try:
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
                            "entry_time": datetime.now(),
                            "symbol": symbol,
                            "token": TRADING_INSTRUMENTS[symbol]['token']
                        }
                        logger.info(f"✅ TOOK CONTROL OF EXISTING POSITION: {symbol}")
        
        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")
    
    def run(self, instrument_symbol: str = 'NIFTYBEES'):
        """Run trading bot with specified instrument"""
        
        # Get instrument details
        if instrument_symbol not in TRADING_INSTRUMENTS:
            logger.error(f"❌ Unknown instrument: {instrument_symbol}")
            logger.info(f"Available instruments: {', '.join(TRADING_INSTRUMENTS.keys())}")
            return
        
        instrument = TRADING_INSTRUMENTS[instrument_symbol]
        trading_token = instrument['token']
        
        self.session_start_time = datetime.now()
        logger.info("Starting SuperTrend trading bot...")
        
        leverage = self.get_mis_leverage(instrument_symbol)
        logger.info(f"📊 Trading Setup:")
        logger.info(f"   Instrument: {instrument_symbol} - {instrument['name']}")
        logger.info(f"   Description: {instrument['description']}")
        logger.info(f"   Token: {trading_token}")
        logger.info(f"   MIS Leverage: {leverage}x")
        logger.info(f"   Account Balance: ₹{Settings.STRATEGY_PARAMS['account_balance']:,}")
        logger.info(f"   Capital Allocation: {Settings.STRATEGY_PARAMS['capital_allocation_percent']}%")
        logger.info(f"   SuperTrend Parameters: ATR={Settings.STRATEGY_PARAMS['atr_period']}, Factor={Settings.STRATEGY_PARAMS['factor']}")
        
        self.check_existing_positions_on_startup()
        
        # Track loop iterations for debugging
        loop_count = 0
        
        try:
            while True:
                try:
                    loop_count += 1
                    
                    if not self.is_market_open():
                        logger.info("Market closed. Waiting...")
                        time.sleep(300)
                        continue
                    
                    # Position sync
                    if self.position["quantity"] > 0:
                        if self.executor is None:
                            logger.error("OrderExecutor is not initialized.")
                            return
                        sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                        
                        if sync_needed and sync_status == "CLOSED_EXTERNALLY":
                            logger.info("Position synchronized with broker")
                            continue
                    
                    # Auto square-off check
                    if self.executor.is_market_close_time() and self.position["quantity"] > 0:
                        logger.warning("🕒 APPROACHING AUTO SQUARE-OFF TIME")
                        if self.executor is None:
                            logger.error("OrderExecutor is not initialized.")
                            return
                        sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                        if sync_needed:
                            continue
                    
                    # Get historical data - USING SAME TOKEN FOR SIGNAL AND TRADING
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=Settings.STRATEGY_PARAMS['historical_days'])
                    
                    # DEBUG: Log data fetch
                    logger.debug(f"Fetching data for {instrument_symbol} from {from_date} to {to_date}")
                    
                    if self.executor is None:
                        logger.error("OrderExecutor is not initialized.")
                        return
                    df = self.executor.get_historical_data(trading_token, from_date, to_date)
                    
                    if df.empty or len(df) < Settings.STRATEGY_PARAMS['min_candles_required']:
                        logger.warning(f"Insufficient data: {len(df) if not df.empty else 0} candles")
                        time.sleep(60)
                        continue
                    
                    # Validate SuperTrend (first time only)
                    if not hasattr(self, '_validated_supertrend'):
                        if self.strategy.validate_signal(df):
                            self._validated_supertrend = True
                            logger.info("✅ SuperTrend validation passed")
                        else:
                            logger.error("❌ SuperTrend validation failed! Check your data.")
                            time.sleep(60)
                            continue
                    
                    # Get signal with enhanced detection
                    if self.executor is None:
                        logger.error("OrderExecutor is not initialized.")
                        return
                    signal, signal_data = self.strategy.get_signal(df, has_position=(self.position["quantity"] > 0))
                    
                    # DEBUG: Log signal details every 10th iteration
                    if loop_count % 10 == 0:
                        logger.info(f"DEBUG Loop #{loop_count}: Signal={signal}, Position={self.position['quantity']}, Direction={signal_data.get('direction')}")
                    
                    # Get current price
                    if self.executor is None:
                        logger.error("OrderExecutor is not initialized.")
                        return
                    current_price = self.executor.get_latest_price(trading_token)
                    if not current_price:
                        time.sleep(60)
                        continue
                    
                    # Log status with more details
                    trend_info = signal_data.get('trend', 'Unknown')
                    direction = signal_data.get('direction', 'Unknown')
                    price_vs_st = signal_data.get('price_vs_supertrend', 'Unknown')
                    
                    logger.info(f"📊 Market Status: {trend_info} | Direction: {direction} | Price: ₹{current_price:.2f} | Price vs SuperTrend: {price_vs_st}")
                    
                    # Check for duplicate signals
                    current_time = datetime.now()
                    if signal in ["BUY", "SELL"] and self.last_signal == signal:
                        time_since_last = (current_time - self.last_signal_time).seconds if self.last_signal_time else 999
                        if time_since_last < 120:  # Ignore duplicate signals within 2 minutes
                            logger.debug(f"Ignoring duplicate {signal} signal (last was {time_since_last}s ago)")
                            time.sleep(Settings.STRATEGY_PARAMS['check_interval'])
                            continue
                    
                    # Execute trades with enhanced logic
                    self._execute_signal(signal, signal_data, instrument_symbol, current_price)
                    
                    # Update last signal tracking
                    if signal in ["BUY", "SELL"]:
                        self.last_signal = signal
                        self.last_signal_time = current_time
                    
                    time.sleep(Settings.STRATEGY_PARAMS['check_interval'])
                    
                except KeyboardInterrupt:
                    raise  # Re-raise to be caught by outer try-except
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Trading stopped by user")
            if self.position["quantity"] > 0:
                logger.warning(f"⚠️  WARNING: You have an open position!")
                logger.warning(f"Position: {self.position['quantity']} {self.position['tradingsymbol']}")
                
                try:
                    if self.executor is None:
                        logger.error("OrderExecutor is not initialized.")
                        return
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    if self.position["quantity"] == 0:
                        logger.info("✅ Position was already closed externally")
                    else:
                        logger.warning("💡 Please close manually if needed")
                except:
                    logger.warning("💡 Please check and close manually if needed")
        
        finally:
            # Always generate report when exiting
            self._generate_trading_report()
    
    def _execute_signal(self, signal: str, signal_data: dict, 
                       trading_symbol: str, current_price: float):
        """Execute trading signal with ENHANCED logic"""
        
        # ENTRY LOGIC - ENHANCED
        if signal == "BUY" and self.position["quantity"] == 0:
            # Double-check with broker before entry
            if self.executor is None:
                logger.error("OrderExecutor is not initialized.")
                return
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            if self.position["quantity"] > 0:
                logger.info("Position sync detected existing position, skipping entry")
                return
                
            # Calculate quantity using MIS leverage
            quantity = self.calculate_mis_quantity(trading_symbol, current_price)
            
            if quantity > 0:
                logger.info("🟢 LONG ENTRY SIGNAL DETECTED - EXECUTING TRADE")
                logger.info(f"📊 SuperTrend Details:")
                logger.info(f"   Trend: {signal_data.get('trend', 'Unknown')}")
                logger.info(f"   Price: ₹{current_price:.2f}")
                logger.info(f"   SuperTrend: ₹{signal_data.get('supertrend', 0):.2f}")
                logger.info(f"   Direction: {signal_data.get('direction', 'Unknown')} (1=GREEN/Up, -1=RED/Down)")
                logger.info(f"   Previous Direction: {signal_data.get('previous_direction', 'Unknown')}")
                logger.info(f"   Price vs SuperTrend: {signal_data.get('price_vs_supertrend', 'Unknown')}")
                
                # Calculate trade details for logging
                trade_value = quantity * current_price
                leverage = self.get_mis_leverage(trading_symbol)
                margin_required = trade_value / leverage
                
                logger.info(f"📋 Trade Details:")
                logger.info(f"   Quantity: {quantity} shares")
                logger.info(f"   Price: ₹{current_price:.2f}")
                logger.info(f"   Trade Value: ₹{trade_value:,.2f}")
                logger.info(f"   Margin Required: ₹{margin_required:,.2f}")
                logger.info(f"   Leverage Used: {leverage}x")
                
                if self.executor is None:
                    logger.error("OrderExecutor is not initialized.")
                    return
                order_id = self.executor.place_order(trading_symbol, "BUY", quantity)
                if order_id:
                    # Try to fetch the actual fill price
                    fill_price = None
                    if not (str(order_id).startswith("DRY_RUN") or str(order_id).startswith("PAPER")):
                        fill_price = self.executor.get_order_filled_price(order_id)
                        if fill_price:
                            logger.info(f"✅ Actual fill price for order {order_id}: ₹{fill_price:.2f}")
                        else:
                            logger.warning(f"⚠️  Could not fetch fill price for order {order_id}, using requested price")
                    else:
                        logger.info(f"Simulated order, using requested price")
                    entry_price = fill_price if fill_price else current_price
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
                    logger.info(f"✅ POSITION OPENED: {quantity} {trading_symbol} at ₹{entry_price:.2f}")
                    logger.info(f"✅ Order ID: {order_id}")
            else:
                logger.warning("❌ Calculated quantity is 0. Check your capital settings.")
        
        # EXIT LOGIC - ENHANCED
        elif signal == "SELL" and self.position["quantity"] > 0:
            # Sync before exit
            if self.executor is None:
                logger.error("OrderExecutor is not initialized.")
                return
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            
            if self.position["quantity"] > 0:
                logger.info("🔴 LONG EXIT SIGNAL DETECTED - CLOSING POSITION")
                logger.info(f"📊 SuperTrend Details:")
                logger.info(f"   Trend: {signal_data.get('trend', 'Unknown')}")
                logger.info(f"   Price: ₹{current_price:.2f}")
                logger.info(f"   SuperTrend: ₹{signal_data.get('supertrend', 0):.2f}")
                logger.info(f"   Direction: {signal_data.get('direction', 'Unknown')} (1=GREEN/Up, -1=RED/Down)")
                logger.info(f"   Previous Direction: {signal_data.get('previous_direction', 'Unknown')}")
                logger.info(f"   Price vs SuperTrend: {signal_data.get('price_vs_supertrend', 'Unknown')}")
                
                if self.executor is None:
                    logger.error("OrderExecutor is not initialized.")
                    return
                order_id = self.executor.place_order(
                    self.position["tradingsymbol"], "SELL", self.position["quantity"]
                )
                if order_id:
                    # Try to fetch the actual fill price for exit
                    fill_price = None
                    if not (str(order_id).startswith("DRY_RUN") or str(order_id).startswith("PAPER")):
                        fill_price = self.executor.get_order_filled_price(order_id)
                        if fill_price:
                            logger.info(f"✅ Actual exit fill price for order {order_id}: ₹{fill_price:.2f}")
                        else:
                            logger.warning(f"⚠️  Could not fetch exit fill price for order {order_id}, using requested price")
                    else:
                        logger.info(f"Simulated order, using requested price")
                    exit_price = fill_price if fill_price else current_price
                    pnl = (exit_price - self.position["entry_price"]) * self.position["quantity"]
                    logger.info(f"📉 POSITION CLOSED (SuperTrend Exit): P&L = ₹{pnl:.2f}")
                    logger.info(f"✅ Order ID: {order_id}")
                    # Record the trade
                    self._record_trade(
                        entry_price=self.position["entry_price"],
                        exit_price=exit_price,
                        quantity=self.position["quantity"],
                        entry_time=self.position["entry_time"],
                        exit_time=datetime.now(),
                        exit_reason="SuperTrend Exit"
                    )
                    self._reset_position()
        
        # POSITION MONITORING - ENHANCED
        elif self.position["quantity"] > 0:
            # Position monitoring with sync
            if self.executor is None:
                logger.error("OrderExecutor is not initialized.")
                return
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            
            if self.position["quantity"] == 0:
                return  # Position was closed externally
            
            # Calculate P&L
            pnl = (current_price - self.position["entry_price"]) * self.position["quantity"]
            pnl_percent = (pnl / (self.position["entry_price"] * self.position["quantity"])) * 100
            
            logger.info(f"Position: {self.position['quantity']} {self.position['tradingsymbol']}")
            logger.info(f"P&L: ₹{pnl:.2f} ({pnl_percent:.2f}%) | Entry: ₹{self.position['entry_price']:.2f} | Current: ₹{current_price:.2f}")
            logger.info(f"Trend: {signal_data.get('trend', 'Unknown')} | Price vs SuperTrend: {signal_data.get('price_vs_supertrend', 'Unknown')}")
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss check
            if pnl < -Settings.STRATEGY_PARAMS['fixed_stop_loss']:
                should_exit = True
                exit_reason = "Stop Loss"
                logger.info("🛑 STOP LOSS TRIGGERED")
            
            # Auto square-off protection
            elif self.executor.is_market_close_time():
                should_exit = True
                exit_reason = "Pre-Market Close"
                logger.info("🕒 CLOSING POSITION BEFORE AUTO SQUARE-OFF")
            
            # Additional exit on SuperTrend direction change (redundant but safe)
            elif signal_data.get('direction') == -1 and signal != "SELL":
                should_exit = True
                exit_reason = "SuperTrend Downtrend"
                logger.info("📉 ADDITIONAL EXIT: SuperTrend in downtrend")
            
            if should_exit:
                # Final sync check before selling
                if self.executor is None:
                    logger.error("OrderExecutor is not initialized.")
                    return
                sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                
                if self.position["quantity"] > 0:
                    order_id = self.executor.place_order(
                        self.position["tradingsymbol"], "SELL", self.position["quantity"]
                    )
                    if order_id:
                        logger.info(f"📉 POSITION CLOSED ({exit_reason}): P&L = ₹{pnl:.2f}")
                        logger.info(f"✅ Order ID: {order_id}")
                        
                        # Record the trade
                        self._record_trade(
                            entry_price=self.position["entry_price"],
                            exit_price=current_price,
                            quantity=self.position["quantity"],
                            entry_time=self.position["entry_time"],
                            exit_time=datetime.now(),
                            exit_reason=exit_reason
                        )
                        
                        self._reset_position()
                else:
                    logger.info("Position already closed externally")
        
        # NO POSITION, NO SIGNAL - Enhanced logging
        else:
            if signal == "HOLD":
                logger.debug(f"💤 No action: {signal_data.get('trend', 'Unknown')} - Price: ₹{current_price:.2f}")
            elif signal == "ERROR":
                logger.error(f"❌ Signal calculation error: {signal_data.get('error', 'Unknown')}")
            elif signal == "BUY" and self.position["quantity"] > 0:
                logger.debug("📈 Buy signal but already in position")
            elif signal == "SELL" and self.position["quantity"] == 0:
                logger.debug("📉 Sell signal but no position to close")
    
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
            "pnl": 0
        }
        logger.info("Position tracking reset")

if __name__ == "__main__":
    # Allow command line argument for instrument selection
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperTrend Trading Bot')
    parser.add_argument('--instrument', type=str, default='NIFTYBEES',
                       help=f'Trading instrument (options: {", ".join(TRADING_INSTRUMENTS.keys())})')
    
    args = parser.parse_args()
    
    bot = TradingBot()
    if bot.setup():
        # Use the specified instrument
        bot.run(args.instrument.upper())