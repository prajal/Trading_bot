import time
from datetime import datetime, timedelta
from auth.kite_auth import KiteAuth
from trading.strategy import SuperTrendStrategy
from trading.executor import OrderExecutor
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

class TradingBot:
    """Main trading bot class with position synchronization"""
    
    def __init__(self):
        self.auth = KiteAuth()
        self.strategy = SuperTrendStrategy(
            atr_period=Settings.STRATEGY_PARAMS['atr_period'],
            factor=Settings.STRATEGY_PARAMS['factor']
        )
        self.executor = None
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
                    
                    # Auto-take control if it's NIFTYBEES (our trading instrument)
                    if symbol == 'NIFTYBEES':
                        self.position = {
                            "instrument_token": "2707457",
                            "tradingsymbol": symbol,
                            "quantity": quantity,
                            "entry_price": avg_price,
                            "pnl": pnl,
                            "entry_time": datetime.now(),
                            "symbol": symbol,
                            "token": "2707457"
                        }
                        logger.info(f"✅ TOOK CONTROL OF EXISTING POSITION: {symbol}")
        
        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")
    
    def run(self, signal_token: str, trading_token: str, trading_symbol: str):
        """Run trading bot with position synchronization"""
        logger.info("Starting SuperTrend trading bot with position sync...")
        
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
                signal, signal_data = self.strategy.get_signal(df)
                
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
        """Execute trading signal with position sync"""
        
        if signal == "BUY" and self.position["quantity"] == 0:
            # Double-check with broker before entry
            sync_needed, sync_status = self.executor.sync_position_with_broker(self.position)
            if self.position["quantity"] > 0:
                return  # We actually have a position
                
            # Calculate quantity
            capital = Settings.STRATEGY_PARAMS['account_balance'] * \
                     (Settings.STRATEGY_PARAMS['capital_allocation_percent'] / 100)
            quantity = int(capital / current_price)
            
            if quantity > 0:
                logger.info("🟢 LONG ENTRY SIGNAL DETECTED")
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
        # NIFTY 50 -> NIFTYBEES
        bot.run("256265", "2707457", "NIFTYBEES")
