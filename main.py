#!/usr/bin/env python3
"""
Main trading application with position synchronization and MIS leverage-aware quantity calculation
"""

import time
from datetime import datetime, timedelta
from auth.kite_auth import KiteAuth
from trading.strategy import SuperTrendStrategy
from trading.executor import OrderExecutor
from config.settings import Settings
from utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__)

class TradingBot:
    """Main trading bot class with position synchronization and MIS leverage support"""
    
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
        
        # MIS Leverage settings for different instruments
        self.mis_leverage_map = {
            'NIFTYBEES': 5.0,    # NIFTY ETFs typically 4-5x
            'JUNIORBEES': 5.0,   # Junior NIFTY ETF
            'BANKBEES': 4.0,     # Bank ETF typically 3-4x
            'LIQUIDBEES': 3.0,   # Liquid ETF lower leverage
            'RELIANCE': 4.0,     # Large cap stocks 3-4x
            'TCS': 4.0,          # Large cap stocks 3-4x
            'HDFCBANK': 4.0,     # Bank stocks 3-4x
            'ICICIBANK': 4.0,    # Bank stocks 3-4x
            'INFY': 4.0,         # IT stocks 3-4x
            'DEFAULT': 3.0       # Default leverage for unknown instruments
        }
    
    def get_mis_leverage(self, symbol: str) -> float:
        """Get MIS leverage for a given symbol"""
        return self.mis_leverage_map.get(symbol, self.mis_leverage_map['DEFAULT'])
    
    def calculate_mis_quantity(self, symbol: str, price: float) -> int:
        """Calculate quantity considering MIS leverage"""
        # Base capital allocation
        capital = Settings.STRATEGY_PARAMS['account_balance'] * \
                 (Settings.STRATEGY_PARAMS['capital_allocation_percent'] / 100)
        
        # Get MIS leverage for this instrument
        mis_leverage = self.get_mis_leverage(symbol)
        
        # Calculate effective buying power with leverage
        effective_capital = capital * mis_leverage
        
        # Calculate potential quantity
        potential_quantity = int(effective_capital / price)
        
        # Calculate actual margin required for this quantity
        margin_required = potential_quantity * (price / mis_leverage)
        
        # Safety check - ensure we don't exceed available capital
        if margin_required > capital:
            # Recalculate with available capital
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
                    
                    # Auto-take control of known trading instruments
                    known_instruments = ['NIFTYBEES', 'JUNIORBEES', 'BANKBEES', 'LIQUIDBEES', 
                                       'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY']
                    
                    if symbol in known_instruments:
                        self.position = {
                            "instrument_token": None,  # Will be determined later
                            "tradingsymbol": symbol,
                            "quantity": quantity,
                            "entry_price": avg_price,
                            "pnl": pnl,
                            "entry_time": datetime.now(),
                            "symbol": symbol,
                            "token": None
                        }
                        logger.info(f"✅ TOOK CONTROL OF EXISTING POSITION: {symbol}")
        
        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")
    
    def run(self, signal_token: str, trading_token: str, trading_symbol: str):
        """Run trading bot with position synchronization and MIS leverage"""
        logger.info("Starting SuperTrend trading bot with position sync and MIS leverage...")
        
        # Log MIS leverage info
        leverage = self.get_mis_leverage(trading_symbol)
        logger.info(f"📊 Trading Setup:")
        logger.info(f"   Symbol: {trading_symbol}")
        logger.info(f"   MIS Leverage: {leverage}x")
        logger.info(f"   Account Balance: ₹{Settings.STRATEGY_PARAMS['account_balance']:,}")
        logger.info(f"   Capital Allocation: {Settings.STRATEGY_PARAMS['capital_allocation_percent']}%")
        
        # Check existing positions on startup
        self.check_existing_positions_on_startup()
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Waiting...")
                    time.sleep(300)
                    continue
                
                # ✅ SYNC POSITION EVERY LOOP
                if self.position["quantity"] > 0:
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    
                    if sync_needed and sync_status == "CLOSED_EXTERNALLY":
                        logger.info("Position synchronized with broker")
                        continue
                
                # ✅ CHECK FOR AUTO SQUARE-OFF TIME
                if self.executor.is_market_close_time() and self.position["quantity"] > 0:
                    logger.warning("🕒 APPROACHING AUTO SQUARE-OFF TIME")
                    sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                    if sync_needed:
                        continue
                
                # Get historical data
                to_date = datetime.now()
                from_date = to_date - timedelta(days=Settings.STRATEGY_PARAMS['historical_days'])
                
                df = self.executor.get_historical_data(signal_token, from_date, to_date)
                
                if df.empty or len(df) < Settings.STRATEGY_PARAMS['min_candles_required']:
                    logger.warning(f"Insufficient data: {len(df) if not df.empty else 0} candles")
                    time.sleep(60)
                    continue
                
                # Get signal
                signal, signal_data = self.strategy.get_signal(df, has_position=(self.position["quantity"] > 0))
                
                # Get current price
                current_price = self.executor.get_latest_price(trading_token)
                if not current_price:
                    time.sleep(60)
                    continue
                
                # Log status
                logger.info(f"Signal: {signal_data.get('trend', 'Unknown')} | Price: ₹{current_price:.2f}")
                
                # Execute trades
                self._execute_signal(signal, signal_data, trading_symbol, current_price)
                
                time.sleep(Settings.STRATEGY_PARAMS['check_interval'])
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                if self.position["quantity"] > 0:
                    logger.warning(f"WARNING: You have an open position!")
                    logger.warning(f"Position: {self.position['quantity']} {self.position['tradingsymbol']}")
                    
                    # Final position sync on exit
                    try:
                        sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                        if self.position["quantity"] == 0:
                            logger.info("✅ Position was already closed externally")
                        else:
                            logger.warning("💡 Please close manually if needed")
                    except:
                        logger.warning("💡 Please check and close manually if needed")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def _execute_signal(self, signal: str, signal_data: dict, 
                       trading_symbol: str, current_price: float):
        """Execute trading signal with position sync and MIS leverage calculation"""
        
        if signal == "BUY" and self.position["quantity"] == 0:
            # Double-check with broker before entry
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            if self.position["quantity"] > 0:
                return  # We actually have a position
                
            # ✅ NEW: Calculate quantity using MIS leverage
            quantity = self.calculate_mis_quantity(trading_symbol, current_price)
            
            if quantity > 0:
                logger.info("🟢 LONG ENTRY SIGNAL DETECTED")
                
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
                
                order_id = self.executor.place_order(trading_symbol, "BUY", quantity)
                if order_id:
                    self.position = {
                        "quantity": quantity,
                        "entry_price": current_price,
                        "entry_time": datetime.now(),
                        "symbol": trading_symbol,
                        "tradingsymbol": trading_symbol,
                        "token": None,
                        "instrument_token": None,
                        "pnl": 0
                    }
                    logger.info(f"✅ POSITION OPENED: {quantity} {trading_symbol} at ₹{current_price:.2f}")
            else:
                logger.warning("❌ Calculated quantity is 0. Check your capital settings.")
        
        elif signal == "SELL" and self.position["quantity"] > 0:
            # Sync before exit
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            
            if self.position["quantity"] > 0:
                logger.info("🔴 LONG EXIT SIGNAL DETECTED")
                order_id = self.executor.place_order(
                    self.position["tradingsymbol"], "SELL", self.position["quantity"]
                )
                if order_id:
                    pnl = (current_price - self.position["entry_price"]) * self.position["quantity"]
                    logger.info(f"📉 POSITION CLOSED (SuperTrend Exit): P&L = ₹{pnl:.2f}")
                    self._reset_position()
        
        elif self.position["quantity"] > 0:
            # Position monitoring with sync
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            
            if self.position["quantity"] == 0:
                return  # Position was closed externally
            
            # Calculate P&L
            pnl = (current_price - self.position["entry_price"]) * self.position["quantity"]
            pnl_percent = (pnl / (self.position["entry_price"] * self.position["quantity"])) * 100
            
            logger.info(f"Position: {self.position['quantity']} {self.position['tradingsymbol']}")
            logger.info(f"P&L: ₹{pnl:.2f} ({pnl_percent:.2f}%) | Entry: ₹{self.position['entry_price']:.2f} | Current: ₹{current_price:.2f}")
            
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
            
            if should_exit:
                # Final sync check before selling
                sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
                
                if self.position["quantity"] > 0:
                    order_id = self.executor.place_order(
                        self.position["tradingsymbol"], "SELL", self.position["quantity"]
                    )
                    if order_id:
                        logger.info(f"📉 POSITION CLOSED ({exit_reason}): P&L = ₹{pnl:.2f}")
                        self._reset_position()
                else:
                    logger.info("Position already closed externally")
    
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

if __name__ == "__main__":
    bot = TradingBot()
    if bot.setup():
        # NIFTY 50 -> NIFTYBEES (with MIS leverage support)
        bot.run("256265", "2707457", "NIFTYBEES")