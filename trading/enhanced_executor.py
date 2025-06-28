import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class ExecutionQuality(Enum):
    """Execution quality levels"""
    EXCELLENT = "excellent"  # < 0.1% slippage
    GOOD = "good"           # 0.1% - 0.3% slippage
    AVERAGE = "average"     # 0.3% - 0.5% slippage
    POOR = "poor"          # > 0.5% slippage

@dataclass
class OrderRecord:
    """Order execution record"""
    order_id: str
    symbol: str
    transaction_type: str
    quantity: int
    requested_price: float
    executed_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    execution_time: Optional[datetime] = None
    slippage: Optional[float] = None
    execution_quality: Optional[ExecutionQuality] = None
    retry_count: int = 0
    error_message: Optional[str] = None

class EnhancedOrderExecutor:
    """
    Enhanced Order Executor with retry logic, slippage tracking,
    and comprehensive error handling
    """
    
    def __init__(self, kite: KiteConnect, safety_config, data_dir: Path):
        self.kite = kite
        self.safety_config = safety_config
        self.data_dir = data_dir
        
        # Order tracking
        self.order_history: List[OrderRecord] = []
        self.daily_orders = 0
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'average_slippage': 0.0,
            'execution_times': []
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay_base = 1.0  # Base delay in seconds
        self.backoff_multiplier = 2.0
        
        # Rate limiting
        self.last_order_time: Optional[datetime] = None
        self.min_order_interval = 1.0  # Minimum seconds between orders
        
        # Load execution history
        self._load_execution_history()
        
        logger.info("Enhanced Order Executor initialized")
    
    def _load_execution_history(self):
        """Load execution history from file"""
        try:
            history_file = self.data_dir / "execution_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                self.execution_stats = data.get('execution_stats', self.execution_stats)
                logger.info(f"Loaded execution history: {self.execution_stats['total_orders']} total orders")
                
        except Exception as e:
            logger.warning(f"Could not load execution history: {e}")
    
    def _save_execution_history(self):
        """Save execution history to file"""
        try:
            history_file = self.data_dir / "execution_history.json"
            
            # Convert recent order records to dict for JSON serialization
            recent_orders = []
            for order in self.order_history[-100:]:  # Keep last 100 orders
                order_dict = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'transaction_type': order.transaction_type,
                    'quantity': order.quantity,
                    'requested_price': order.requested_price,
                    'executed_price': order.executed_price,
                    'status': order.status.value,
                    'timestamp': order.timestamp.isoformat() if order.timestamp else None,
                    'execution_time': order.execution_time.isoformat() if order.execution_time else None,
                    'slippage': order.slippage,
                    'execution_quality': order.execution_quality.value if order.execution_quality else None,
                    'retry_count': order.retry_count,
                    'error_message': order.error_message
                }
                recent_orders.append(order_dict)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'execution_stats': self.execution_stats,
                'recent_orders': recent_orders
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between orders"""
        if self.last_order_time:
            elapsed = (datetime.now() - self.last_order_time).total_seconds()
            if elapsed < self.min_order_interval:
                wait_time = self.min_order_interval - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
    
    def _calculate_slippage(self, requested_price: float, executed_price: float) -> float:
        """Calculate slippage percentage"""
        if requested_price <= 0:
            return 0.0
        return abs(executed_price - requested_price) / requested_price * 100
    
    def _classify_execution_quality(self, slippage: float) -> ExecutionQuality:
        """Classify execution quality based on slippage"""
        if slippage < 0.1:
            return ExecutionQuality.EXCELLENT
        elif slippage < 0.3:
            return ExecutionQuality.GOOD
        elif slippage < 0.5:
            return ExecutionQuality.AVERAGE
        else:
            return ExecutionQuality.POOR
    
    def place_order_with_retry(self, 
                              symbol: str, 
                              transaction_type: str, 
                              quantity: int,
                              expected_price: Optional[float] = None) -> Optional[str]:
        """
        Place order with comprehensive retry logic and error handling
        
        Args:
            symbol: Trading symbol
            transaction_type: BUY or SELL
            quantity: Number of shares
            expected_price: Expected execution price for slippage calculation
            
        Returns:
            Order ID if successful, None if failed
        """
        
        # Check daily order limit
        if self.daily_orders >= self.safety_config.max_orders_per_day:
            logger.error(f"Daily order limit exceeded: {self.daily_orders}")
            return None
        
        # Handle dry run mode
        if self.safety_config.dry_run_mode:
            order_id = f"DRY_RUN_{int(time.time())}"
            logger.info(f"DRY RUN: {transaction_type} {quantity} {symbol}")
            self._record_dry_run_order(symbol, transaction_type, quantity, order_id)
            return order_id
        
        # Handle paper trading mode
        if not self.safety_config.live_trading_enabled:
            order_id = f"PAPER_{int(time.time())}"
            logger.info(f"PAPER TRADE: {transaction_type} {quantity} {symbol}")
            self._record_paper_trade_order(symbol, transaction_type, quantity, order_id)
            return order_id
        
        # Create order record
        order_record = OrderRecord(
            order_id="",
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            requested_price=expected_price or 0.0,
            timestamp=datetime.now()
        )
        
        # Attempt order placement with retries
        for attempt in range(self.max_retries + 1):
            try:
                order_record.retry_count = attempt
                
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Place the order
                logger.info(f"Placing order (attempt {attempt + 1}): {transaction_type} {quantity} {symbol}")
                
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
                
                order_record.order_id = str(order_id)
                order_record.status = OrderStatus.PENDING
                self.last_order_time = datetime.now()
                self.daily_orders += 1
                
                logger.info(f"Order placed successfully: {order_id}")
                
                # Wait for execution and get fill details
                execution_details = self._wait_for_execution(order_id, timeout=30)
                
                if execution_details:
                    order_record.executed_price = execution_details['average_price']
                    order_record.execution_time = datetime.now()
                    order_record.status = OrderStatus.COMPLETE
                    
                    # Calculate slippage if expected price provided
                    if expected_price:
                        order_record.slippage = self._calculate_slippage(
                            expected_price, execution_details['average_price']
                        )
                        order_record.execution_quality = self._classify_execution_quality(
                            order_record.slippage
                        )
                        
                        logger.info(f"Order executed: ₹{execution_details['average_price']:.2f}, "
                                  f"Slippage: {order_record.slippage:.3f}%, "
                                  f"Quality: {order_record.execution_quality.value}")
                else:
                    logger.warning(f"Could not confirm execution for order {order_id}")
                
                # Update statistics
                self._update_execution_stats(order_record)
                self.order_history.append(order_record)
                self._save_execution_history()
                
                return str(order_id)
                
            except TokenException as e:
                logger.error(f"Token error on attempt {attempt + 1}: {e}")
                order_record.error_message = f"Token error: {e}"
                # Token errors are not retryable
                break
                
            except NetworkException as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                order_record.error_message = f"Network error: {e}"
                
                if attempt < self.max_retries:
                    delay = self.retry_delay_base * (self.backoff_multiplier ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Network error persisted after {self.max_retries} retries")
                    break
                    
            except KiteException as e:
                logger.error(f"Kite API error on attempt {attempt + 1}: {e}")
                order_record.error_message = f"Kite API error: {e}"
                
                # Some Kite errors are retryable, others are not
                if "insufficient funds" in str(e).lower() or "order rejected" in str(e).lower():
                    logger.error("Non-retryable Kite error")
                    break
                elif attempt < self.max_retries:
                    delay = self.retry_delay_base * (self.backoff_multiplier ** attempt)
                    logger.info(f"Retrying Kite error in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                order_record.error_message = f"Unexpected error: {e}"
                
                if attempt < self.max_retries:
                    delay = self.retry_delay_base * (self.backoff_multiplier ** attempt)
                    logger.info(f"Retrying unexpected error in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    break
        
        # All attempts failed
        order_record.status = OrderStatus.FAILED
        self._update_execution_stats(order_record)
        self.order_history.append(order_record)
        self._save_execution_history()
        
        logger.error(f"Order failed after {self.max_retries + 1} attempts: {transaction_type} {quantity} {symbol}")
        return None
    
    def _wait_for_execution(self, order_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for order execution and return execution details"""
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < timeout:
                try:
                    order_history = self.kite.order_history(order_id)
                    
                    if not isinstance(order_history, list) or len(order_history) == 0:
                        time.sleep(1)
                        continue
                    
                    # Get the latest order status
                    latest_order = order_history[-1]
                    status = latest_order.get('status', '').upper()
                    
                    if status == 'COMPLETE':
                        avg_price = latest_order.get('average_price', 0.0)
                        if avg_price and avg_price > 0:
                            return {
                                'average_price': float(avg_price),
                                'filled_quantity': latest_order.get('filled_quantity', 0),
                                'order_timestamp': latest_order.get('order_timestamp'),
                                'exchange_timestamp': latest_order.get('exchange_timestamp')
                            }
                    
                    elif status in ['CANCELLED', 'REJECTED']:
                        logger.warning(f"Order {order_id} was {status}")
                        return None
                    
                    # Order still pending, wait a bit more
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error checking order status: {e}")
                    time.sleep(1)
            
            logger.warning(f"Timeout waiting for order execution: {order_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error waiting for execution: {e}")
            return None
    
    def _record_dry_run_order(self, symbol: str, transaction_type: str, quantity: int, order_id: str):
        """Record dry run order"""
        order_record = OrderRecord(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            requested_price=0.0,
            executed_price=0.0,
            status=OrderStatus.COMPLETE,
            timestamp=datetime.now(),
            execution_time=datetime.now()
        )
        
        self.order_history.append(order_record)
        self._update_execution_stats(order_record)
    
    def _record_paper_trade_order(self, symbol: str, transaction_type: str, quantity: int, order_id: str):
        """Record paper trading order"""
        # Get current market price for paper trading
        current_price = self.get_latest_price_with_retry(symbol)
        
        order_record = OrderRecord(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            requested_price=current_price or 0.0,
            executed_price=current_price or 0.0,
            status=OrderStatus.COMPLETE,
            timestamp=datetime.now(),
            execution_time=datetime.now()
        )
        
        self.order_history.append(order_record)
        self._update_execution_stats(order_record)
    
    def _update_execution_stats(self, order_record: OrderRecord):
        """Update execution statistics"""
        try:
            self.execution_stats['total_orders'] += 1
            
            if order_record.status == OrderStatus.COMPLETE:
                self.execution_stats['successful_orders'] += 1
                
                # Update slippage statistics
                if order_record.slippage is not None:
                    current_avg = self.execution_stats['average_slippage']
                    total_successful = self.execution_stats['successful_orders']
                    
                    # Running average calculation
                    new_avg = ((current_avg * (total_successful - 1)) + order_record.slippage) / total_successful
                    self.execution_stats['average_slippage'] = new_avg
                
                # Update execution time statistics
                if order_record.execution_time and order_record.timestamp:
                    exec_time = (order_record.execution_time - order_record.timestamp).total_seconds()
                    self.execution_stats['execution_times'].append(exec_time)
                    
                    # Keep only last 100 execution times
                    if len(self.execution_stats['execution_times']) > 100:
                        self.execution_stats['execution_times'] = self.execution_stats['execution_times'][-100:]
            
            else:
                self.execution_stats['failed_orders'] += 1
                
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    def get_latest_price_with_retry(self, instrument_token: str, max_retries: int = 3) -> Optional[float]:
        """Get current market price with retry logic"""
        for attempt in range(max_retries):
            try:
                quote = self.kite.quote(instrument_token)
                token_str = str(instrument_token)
                
                if isinstance(quote, dict) and token_str in quote:
                    token_data = quote[token_str]
                    if isinstance(token_data, dict) and 'last_price' in token_data:
                        price = token_data['last_price']
                        if price and price > 0:
                            return float(price)
                
                logger.warning(f"Invalid quote data for {instrument_token} on attempt {attempt + 1}")
                
            except NetworkException as e:
                logger.warning(f"Network error getting price (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                logger.error(f"Error fetching latest price (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        logger.error(f"Failed to get price for {instrument_token} after {max_retries} attempts")
        return None
    
    def get_historical_data_with_retry(self, 
                                     instrument_token: str, 
                                     from_date: datetime, 
                                     to_date: datetime, 
                                     interval: str = "minute",
                                     max_retries: int = 3) -> pd.DataFrame:
        """
        FIXED: Fetch historical data with retry logic and proper 60-day limit handling
        """
        
        # FIXED: Validate date range to ensure it's within API limits
        date_diff = (to_date - from_date).days
        
        # Zerodha API limits by interval
        api_limits = {
            "minute": 60,      # 60 days for minute data
            "3minute": 90,     # 90 days for 3-minute data
            "5minute": 90,     # 90 days for 5-minute data
            "10minute": 90,    # 90 days for 10-minute data
            "15minute": 180,   # 180 days for 15-minute data
            "30minute": 180,   # 180 days for 30-minute data
            "60minute": 365,   # 365 days for hourly data
            "day": 2000        # 2000 days for daily data
        }
        
        max_days = api_limits.get(interval, 60)  # Default to 60 days if interval not found
        
        if date_diff > max_days:
            logger.warning(f"Date range ({date_diff} days) exceeds API limit ({max_days} days) for {interval} data")
            logger.info(f"Automatically adjusting to use last {max_days} days")
            
            # Adjust from_date to stay within limits
            from_date = to_date - timedelta(days=max_days - 1)  # -1 to be safe
            logger.info(f"Adjusted date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching historical data (attempt {attempt + 1}): {instrument_token}")
                logger.debug(f"Date range: {from_date} to {to_date}, Interval: {interval}")
                
                data = self.kite.historical_data(instrument_token, from_date, to_date, interval)
                
                if not data:
                    logger.warning(f"No historical data received for {instrument_token} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                
                # Validate data quality
                if len(df) == 0:
                    logger.warning(f"Empty historical data for {instrument_token}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return pd.DataFrame()
                
                # Check for required columns
                required_columns = ['open', 'high', 'low', 'close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing columns in historical data: {missing_columns}")
                    return pd.DataFrame()
                
                # Check for invalid prices (negative or zero)
                for col in required_columns:
                    if (df[col] <= 0).any():
                        logger.warning(f"Invalid prices found in column {col}")
                        # Remove invalid rows
                        df = df[df[col] > 0]
                
                if len(df) == 0:
                    logger.error("All historical data invalid after cleaning")
                    return pd.DataFrame()
                
                logger.debug(f"Historical data loaded successfully: {len(df)} records for {instrument_token}")
                return df
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for specific API limit errors
                if "interval exceeds" in error_msg or "limit" in error_msg:
                    logger.error(f"API limit error: {e}")
                    
                    # Try to extract the actual limit from error message
                    import re
                    limit_match = re.search(r'(\d+)\s*days?', error_msg)
                    if limit_match:
                        actual_limit = int(limit_match.group(1))
                        logger.info(f"Detected API limit: {actual_limit} days")
                        
                        # Adjust date range and retry
                        if actual_limit < date_diff:
                            from_date = to_date - timedelta(days=actual_limit - 1)
                            logger.info(f"Retrying with adjusted range: {from_date} to {to_date}")
                            continue
                    
                    # If we can't parse the limit, use a very conservative range
                    from_date = to_date - timedelta(days=3)
                    logger.info(f"Using conservative 3-day range: {from_date} to {to_date}")
                    continue
                
                elif "network" in error_msg or "connection" in error_msg:
                    logger.warning(f"Network error getting historical data (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                else:
                    logger.error(f"Error fetching historical data (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
        
        logger.error(f"Failed to get historical data for {instrument_token} after {max_retries} attempts")
        return pd.DataFrame()
    
    def sync_position_with_broker(self, current_position: dict) -> Tuple[bool, str]:
        """
        Sync internal position tracking with actual broker positions
        Returns: (sync_needed, sync_status)
        """
        try:
            # Get actual positions from broker with retry
            positions = None
            for attempt in range(3):
                try:
                    positions = self.kite.positions()
                    break
                except NetworkException as e:
                    logger.warning(f"Network error getting positions (attempt {attempt + 1}): {e}")
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    raise
            
            if not positions:
                logger.error("Could not fetch positions from broker")
                return False, "ERROR"
            
            day_positions = positions.get('day', [])
            
            # Find our trading instrument in actual positions
            trading_symbol = current_position.get('tradingsymbol', '')
            actual_quantity = 0
            actual_pnl = 0
            actual_avg_price = 0
            
            for pos in day_positions:
                if pos['tradingsymbol'] == trading_symbol and pos['exchange'] == 'NSE':
                    actual_quantity = pos['quantity']
                    actual_pnl = pos['pnl'] 
                    actual_avg_price = pos['average_price']
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
                    
                    # Update internal position to match broker
                    current_position.update({
                        "quantity": actual_quantity,
                        "entry_price": actual_avg_price,
                        "pnl": actual_pnl,
                        "entry_time": datetime.now()
                    })
                    
                    return True, "OPENED_EXTERNALLY"
                
                elif actual_quantity != internal_quantity:
                    # Quantity mismatch
                    logger.warning(f"QUANTITY MISMATCH: Internal={internal_quantity}, Actual={actual_quantity}")
                    
                    # Update to actual quantity
                    current_position["quantity"] = actual_quantity
                    current_position["pnl"] = actual_pnl
                    
                    return True, "QUANTITY_ADJUSTED"
            
            return False, "IN_SYNC"
            
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
            return False, "ERROR"
    
    def is_market_close_time(self) -> bool:
        """Check if it's near market close time for auto square-off"""
        try:
            now = datetime.now()
            current_time = now.time()
            
            # Market closes at 15:30, auto square-off happens around 15:20
            auto_squareoff_time = datetime.strptime("15:20", "%H:%M").time()
            market_close_time = datetime.strptime("15:30", "%H:%M").time()
            
            return auto_squareoff_time <= current_time <= market_close_time
            
        except Exception as e:
            logger.error(f"Error checking market close time: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        try:
            total_orders = self.execution_stats['total_orders']
            successful_orders = self.execution_stats['successful_orders']
            
            success_rate = (successful_orders / total_orders * 100) if total_orders > 0 else 0
            
            # Calculate average execution time
            exec_times = self.execution_stats['execution_times']
            avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            # Get recent execution quality distribution
            recent_orders = self.order_history[-50:]  # Last 50 orders
            quality_distribution = {
                'excellent': 0,
                'good': 0, 
                'average': 0,
                'poor': 0,
                'unknown': 0
            }
            
            for order in recent_orders:
                if order.execution_quality:
                    quality_distribution[order.execution_quality.value] += 1
                else:
                    quality_distribution['unknown'] += 1
            
            return {
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'failed_orders': self.execution_stats['failed_orders'],
                'success_rate': success_rate,
                'average_slippage': self.execution_stats['average_slippage'],
                'average_execution_time': avg_exec_time,
                'daily_orders': self.daily_orders,
                'max_daily_orders': self.safety_config.max_orders_per_day,
                'execution_quality_distribution': quality_distribution,
                'recent_orders_count': len(recent_orders)
            }
            
        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return {'error': str(e)}
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of each trading day)"""
        try:
            self.daily_orders = 0
            logger.info("Daily order counters reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily counters: {e}")
    
    def get_order_filled_price(self, order_id: str) -> Optional[float]:
        """Fetch the actual average fill price for a given order_id"""
        try:
            # First check if it's a simulation order
            if str(order_id).startswith(("DRY_RUN", "PAPER")):
                # For simulation orders, find in our history
                for order in self.order_history:
                    if order.order_id == order_id:
                        return order.executed_price
                return None
            
            order_history = self.kite.order_history(order_id)
            
            if not isinstance(order_history, list):
                logger.error(f"Order history for {order_id} is not a list")
                return None
                
            for event in reversed(order_history):
                if isinstance(event, dict):
                    status = event.get('status')
                    avg_price = event.get('average_price')
                    
                    if status == 'COMPLETE' and avg_price:
                        avg_price = float(avg_price)
                        logger.info(f"Fetched fill price for order {order_id}: ₹{avg_price}")
                        return avg_price
            
            logger.warning(f"No fill price found for order {order_id} (not filled yet)")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching fill price for order {order_id}: {e}")
            return None
    
    # Legacy method for backward compatibility
    def place_order(self, symbol: str, transaction_type: str, quantity: int) -> Optional[str]:
        """Legacy place_order method for backward compatibility"""
        return self.place_order_with_retry(symbol, transaction_type, quantity)
    
    def get_latest_price(self, instrument_token: str) -> Optional[float]:
        """Legacy get_latest_price method for backward compatibility"""
        return self.get_latest_price_with_retry(instrument_token)
    
    def get_historical_data(self, instrument_token: str, from_date: datetime, 
                          to_date: datetime, interval: str = "minute") -> pd.DataFrame:
        """Legacy get_historical_data method for backward compatibility"""
        return self.get_historical_data_with_retry(instrument_token, from_date, to_date, interval)