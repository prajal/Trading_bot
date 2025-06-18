from datetime import datetime
import time
import pandas as pd
from typing import Optional, Tuple
from kiteconnect import KiteConnect
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

class OrderExecutor:
    """Handles order execution and position management"""
    
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.safety_config = Settings.SAFETY_CONFIG
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int) -> Optional[str]:
        """Place order with safety checks"""
        if self.safety_config['dry_run_mode']:
            logger.info(f"DRY RUN: {transaction_type} {quantity} {symbol}")
            return f"DRY_RUN_{int(time.time())}"
        
        if not self.safety_config['live_trading_enabled']:
            logger.info(f"PAPER TRADE: {transaction_type} {quantity} {symbol}")
            return f"PAPER_{int(time.time())}"
        
        try:
            # CORRECTED: variety as first positional argument
            order_id = self.kite.place_order(
                variety='regular',
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type=transaction_type,
                quantity=int(quantity),
                product="MIS",
                order_type="MARKET",
                validity="DAY"
            )
            
            logger.info(f"Order placed: {transaction_type} {quantity} {symbol} - ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing {transaction_type} order: {e}")
            return None
    
    def get_latest_price(self, instrument_token: str) -> Optional[float]:
        """Get current market price"""
        try:
            quote = self.kite.quote(instrument_token)
            return quote[str(instrument_token)]["last_price"]
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return None
    
    def get_historical_data(self, instrument_token: str, from_date: datetime, 
                          to_date: datetime, interval: str = "minute") -> pd.DataFrame:
        """Fetch historical data"""
        try:
            data = self.kite.historical_data(instrument_token, from_date, to_date, interval)
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def sync_position_with_broker(self, current_position: dict) -> Tuple[bool, str]:
        """
        Sync internal position tracking with actual broker positions
        Returns: (sync_needed, sync_status)
        """
        try:
            # Get actual positions from broker
            positions = self.kite.positions()
            day_positions = positions.get('day', [])
            
            # Find our trading instrument in actual positions
            trading_symbol = current_position.get('tradingsymbol', '')
            actual_quantity = 0
            actual_pnl = 0
            
            for pos in day_positions:
                if pos['tradingsymbol'] == trading_symbol and pos['exchange'] == 'NSE':
                    actual_quantity = pos['quantity']
                    actual_pnl = pos['pnl']
                    break
            
            # Check if positions are out of sync
            internal_quantity = current_position.get('quantity', 0)
            
            if internal_quantity != actual_quantity:
                logger.warning(f"POSITION SYNC: Internal={internal_quantity}, Actual={actual_quantity}")
                
                if actual_quantity == 0 and internal_quantity > 0:
                    # Position was closed externally (auto square-off, manual close, etc.)
                    logger.info(f"📉 EXTERNAL CLOSE DETECTED: P&L = ₹{actual_pnl:.2f}")
                    
                    # Reset internal position
                    current_position.update({
                        "instrument_token": None,
                        "tradingsymbol": None,
                        "quantity": 0,
                        "entry_price": 0,
                        "pnl": actual_pnl,
                        "entry_time": None
                    })
                    
                    return True, "CLOSED_EXTERNALLY"
                
                elif actual_quantity > 0 and internal_quantity == 0:
                    # Position was opened externally
                    logger.warning("⚠️  EXTERNAL OPEN DETECTED")
                    return True, "OPENED_EXTERNALLY"
            
            return False, "IN_SYNC"
            
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
            return False, "ERROR"
    
    def is_market_close_time(self) -> bool:
        """Check if it's near market close time for auto square-off"""
        now = datetime.now()
        current_time = now.time()
        
        # Market closes at 15:30, auto square-off happens around 15:20
        auto_squareoff_time = datetime.strptime("15:20", "%H:%M").time()
        market_close_time = datetime.strptime("15:30", "%H:%M").time()
        
        return auto_squareoff_time <= current_time <= market_close_time
