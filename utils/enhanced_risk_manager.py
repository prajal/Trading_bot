import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionSizeMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    risk_level: RiskLevel = RiskLevel.LOW

@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None

class RiskManager:
    """
    Enhanced Risk Management System
    Handles position sizing, risk monitoring, and trade validation
    """
    
    def __init__(self, trading_config, risk_config, data_dir: Path):
        self.trading_config = trading_config
        self.risk_config = risk_config
        self.data_dir = data_dir
        
        # Risk tracking
        self.performance_file = data_dir / "performance_metrics.json"
        self.risk_metrics = RiskMetrics()
        self.positions: Dict[str, PositionInfo] = {}
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time: Optional[datetime] = None
        
        # Trade tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        
        # Load historical performance data
        self._load_performance_data()
        
        logger.info("Risk Manager initialized")
    
    def _load_performance_data(self):
        """Load historical performance data"""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    
                self.equity_curve = data.get('equity_curve', [])
                self.daily_returns = data.get('daily_returns', [])
                self.risk_metrics.max_drawdown = data.get('max_drawdown', 0.0)
                
                logger.info(f"Loaded performance data: {len(self.equity_curve)} equity points")
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'equity_curve': self.equity_curve[-1000:],  # Keep last 1000 points
                'daily_returns': self.daily_returns[-252:],  # Keep last year
                'max_drawdown': self.risk_metrics.max_drawdown,
                'current_drawdown': self.risk_metrics.current_drawdown,
                'risk_level': self.risk_metrics.risk_level.value
            }
            
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              current_price: float, 
                              signal_confidence: float = 1.0,
                              volatility: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate optimal position size based on configured method
        
        Returns:
            Tuple[int, Dict]: (quantity, calculation_details)
        """
        try:
            method = PositionSizeMethod(self.trading_config.position_sizing_method)
            account_balance = self.trading_config.account_balance
            risk_per_trade = self.trading_config.risk_per_trade
            
            # Base calculations
            available_capital = account_balance * (self.trading_config.capital_allocation_percent / 100)
            max_risk_amount = account_balance * risk_per_trade
            
            # Get leverage for symbol
            leverage = self._get_leverage_for_symbol(symbol)
            
            calculation_details = {
                'method': method.value,
                'available_capital': available_capital,
                'max_risk_amount': max_risk_amount,
                'leverage': leverage,
                'signal_confidence': signal_confidence,
                'volatility': volatility
            }
            
            if method == PositionSizeMethod.FIXED:
                quantity = self._calculate_fixed_position_size(
                    available_capital, current_price, leverage
                )
            
            elif method == PositionSizeMethod.RISK_BASED:
                quantity = self._calculate_risk_based_position_size(
                    max_risk_amount, current_price, leverage, signal_confidence
                )
            
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                if volatility is None:
                    volatility = 0.02  # Default 2% volatility
                quantity = self._calculate_volatility_adjusted_position_size(
                    max_risk_amount, current_price, leverage, volatility, signal_confidence
                )
            
            elif method == PositionSizeMethod.KELLY_CRITERION:
                quantity = self._calculate_kelly_position_size(
                    available_capital, current_price, leverage, signal_confidence
                )
            
            else:
                quantity = self._calculate_fixed_position_size(
                    available_capital, current_price, leverage
                )
            
            # Apply risk adjustments
            quantity = self._apply_risk_adjustments(quantity, current_price)
            
            calculation_details['final_quantity'] = quantity
            calculation_details['position_value'] = quantity * current_price
            calculation_details['margin_required'] = (quantity * current_price) / leverage
            
            logger.info(f"Position size calculated: {quantity} shares of {symbol}")
            logger.debug(f"Calculation details: {calculation_details}")
            
            return quantity, calculation_details
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Fallback to conservative fixed sizing
            fallback_quantity = int((available_capital * 0.1) / current_price)
            return max(1, fallback_quantity), {'method': 'fallback', 'error': str(e)}
    
    def _calculate_fixed_position_size(self, capital: float, price: float, leverage: float) -> int:
        """Calculate fixed position size"""
        effective_capital = capital * leverage
        return int(effective_capital / price)
    
    def _calculate_risk_based_position_size(self, 
                                          risk_amount: float, 
                                          price: float, 
                                          leverage: float,
                                          confidence: float) -> int:
        """Calculate position size based on risk amount"""
        # Adjust risk based on confidence
        adjusted_risk = risk_amount * confidence
        
        # Assume stop loss at 5% for risk calculation
        stop_loss_distance = price * 0.05
        
        if stop_loss_distance > 0:
            shares_at_risk = adjusted_risk / stop_loss_distance
            return int(shares_at_risk)
        else:
            return self._calculate_fixed_position_size(adjusted_risk, price, leverage)
    
    def _calculate_volatility_adjusted_position_size(self,
                                                   risk_amount: float,
                                                   price: float,
                                                   leverage: float,
                                                   volatility: float,
                                                   confidence: float) -> int:
        """Calculate position size adjusted for volatility"""
        # Higher volatility = smaller position size
        volatility_adjustment = 1 / (1 + volatility * 5)  # Scale volatility impact
        confidence_adjustment = confidence * 0.8 + 0.2  # Scale confidence impact
        
        adjusted_risk = risk_amount * volatility_adjustment * confidence_adjustment
        
        # Use volatility for stop loss distance
        stop_loss_distance = price * volatility * 2  # 2x volatility for stop
        
        if stop_loss_distance > 0:
            shares_at_risk = adjusted_risk / stop_loss_distance
            return int(shares_at_risk)
        else:
            return self._calculate_risk_based_position_size(risk_amount, price, leverage, confidence)
    
    def _calculate_kelly_position_size(self,
                                     capital: float,
                                     price: float,
                                     leverage: float,
                                     confidence: float) -> int:
        """Calculate position size using Kelly Criterion"""
        # Simplified Kelly: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        # Use historical win rate if available, otherwise use confidence
        win_prob = self.risk_metrics.win_rate if self.risk_metrics.win_rate > 0 else confidence
        loss_prob = 1 - win_prob
        
        # Assume 1:1 risk/reward ratio for simplicity
        odds = 1.0
        
        if win_prob > 0 and loss_prob > 0:
            kelly_fraction = (odds * win_prob - loss_prob) / odds
            # Cap Kelly fraction at 25% for safety
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
        else:
            kelly_fraction = 0.1  # Conservative default
        
        position_value = capital * kelly_fraction
        return int(position_value / price)
    
    def _apply_risk_adjustments(self, quantity: int, price: float) -> int:
        """Apply additional risk adjustments to position size"""
        # Reduce size if circuit breaker is active
        if self.circuit_breaker_triggered:
            quantity = int(quantity * 0.5)
            logger.warning("Position size reduced due to circuit breaker")
        
        # Reduce size based on current drawdown
        if self.risk_metrics.current_drawdown > 0.05:  # 5% drawdown
            drawdown_reduction = min(0.5, self.risk_metrics.current_drawdown * 2)
            quantity = int(quantity * (1 - drawdown_reduction))
            logger.warning(f"Position size reduced by {drawdown_reduction:.1%} due to drawdown")
        
        # Reduce size if too many consecutive losses
        if self.risk_metrics.consecutive_losses >= 3:
            loss_reduction = min(0.3, self.risk_metrics.consecutive_losses * 0.1)
            quantity = int(quantity * (1 - loss_reduction))
            logger.warning(f"Position size reduced due to {self.risk_metrics.consecutive_losses} consecutive losses")
        
        # Ensure minimum quantity
        return max(1, quantity)
    
    def _get_leverage_for_symbol(self, symbol: str) -> float:
        """Get leverage for specific symbol"""
        # Define leverage mapping for different instruments
        leverage_map = {
            'NIFTYBEES': 5.0,
            'BANKBEES': 4.0,
            'JUNIORBEES': 5.0,
            'GOLDBEES': 3.0,
            'RELIANCE': 4.0,
            'TCS': 4.0,
            'HDFCBANK': 4.0,
            'ICICIBANK': 4.0,
            'INFY': 4.0
        }
        
        return leverage_map.get(symbol, 3.0)  # Default 3x leverage
    
    def validate_trade(self, 
                      symbol: str, 
                      side: str, 
                      quantity: int, 
                      price: float) -> Tuple[bool, str]:
        """
        Validate if a trade should be allowed
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            # Check circuit breaker
            if self.circuit_breaker_triggered:
                return False, "Circuit breaker is active"
            
            # Check daily trade limit
            if self.trades_today >= self.trading_config.max_positions * 10:  # 10 trades per position max
                return False, "Daily trade limit exceeded"
            
            # Check daily loss limit
            daily_loss_limit = self.trading_config.account_balance * self.trading_config.max_daily_loss
            if self.daily_pnl < -daily_loss_limit:
                return False, f"Daily loss limit exceeded (₹{abs(self.daily_pnl):.2f})"
            
            # Check position limits
            if side == "BUY" and len(self.positions) >= self.trading_config.max_positions:
                return False, "Maximum positions limit reached"
            
            # Check minimum position value
            position_value = quantity * price
            if position_value < 1000:  # Minimum ₹1000 position
                return False, "Position value too small (minimum ₹1000)"
            
            # Check maximum position value
            max_position_value = self.trading_config.account_balance * 0.5  # 50% max per position
            if position_value > max_position_value:
                return False, f"Position value too large (max ₹{max_position_value:,.0f})"
            
            # All checks passed
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_stop_loss(self, 
                          symbol: str, 
                          entry_price: float, 
                          side: str,
                          atr: Optional[float] = None) -> float:
        """Calculate stop loss price"""
        try:
            method = self.risk_config.stop_loss_method
            
            if method == "fixed":
                # Fixed rupee amount
                stop_distance = self.trading_config.fixed_stop_loss / 100  # Convert to percentage
                
            elif method == "percentage":
                # Fixed percentage
                stop_distance = 0.02  # 2% default
                
            elif method == "atr_based" and atr is not None:
                # ATR-based stop loss
                stop_distance = (atr * 2.0) / entry_price  # 2x ATR
                
            else:
                # Fallback to percentage
                stop_distance = 0.02
            
            if side.upper() == "BUY":
                stop_price = entry_price * (1 - stop_distance)
            else:
                stop_price = entry_price * (1 + stop_distance)
            
            return round(stop_price, 2)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Conservative fallback
            return entry_price * 0.98 if side.upper() == "BUY" else entry_price * 1.02
    
    def update_position(self, 
                       symbol: str, 
                       quantity: int, 
                       entry_price: float, 
                       current_price: float,
                       entry_time: datetime):
        """Update position information"""
        try:
            unrealized_pnl = (current_price - entry_price) * quantity
            
            position = PositionInfo(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                entry_time=entry_time
            )
            
            # Calculate stop loss
            stop_loss_price = self.calculate_stop_loss(symbol, entry_price, "BUY")
            position.stop_loss_price = stop_loss_price
            
            # Calculate trailing stop if enabled
            if self.risk_config.trailing_stop:
                position.trailing_stop_price = self._calculate_trailing_stop(
                    current_price, entry_price, "BUY"
                )
            
            self.positions[symbol] = position
            logger.info(f"Position updated: {symbol} - P&L: ₹{unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime) -> float:
        """Close position and return realized P&L"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Attempted to close non-existent position: {symbol}")
                return 0.0
            
            position = self.positions[symbol]
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            
            # Update performance metrics
            self._update_performance_metrics(realized_pnl)
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol} - Realized P&L: ₹{realized_pnl:.2f}")
            return realized_pnl
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return 0.0
    
    def _calculate_trailing_stop(self, current_price: float, entry_price: float, side: str) -> float:
        """Calculate trailing stop price"""
        try:
            distance = self.risk_config.trailing_stop_distance * 0.01  # Convert to percentage
            
            if side.upper() == "BUY":
                # For long positions, trailing stop moves up with price
                trailing_stop = current_price * (1 - distance)
                # But never below initial stop loss
                initial_stop = entry_price * 0.98
                return max(trailing_stop, initial_stop)
            else:
                # For short positions, trailing stop moves down with price
                trailing_stop = current_price * (1 + distance)
                initial_stop = entry_price * 1.02
                return min(trailing_stop, initial_stop)
                
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return entry_price * 0.98 if side.upper() == "BUY" else entry_price * 1.02
    
    def check_stop_loss_triggers(self) -> List[Dict[str, Any]]:
        """Check if any positions have triggered stop losses"""
        triggered_stops = []
        
        try:
            for symbol, position in self.positions.items():
                should_exit = False
                exit_reason = ""
                
                # Check fixed stop loss
                if position.stop_loss_price:
                    if position.current_price <= position.stop_loss_price:
                        should_exit = True
                        exit_reason = "Stop Loss"
                
                # Check trailing stop
                if position.trailing_stop_price and not should_exit:
                    if position.current_price <= position.trailing_stop_price:
                        should_exit = True
                        exit_reason = "Trailing Stop"
                
                # Check maximum loss per position
                max_loss = self.trading_config.account_balance * self.trading_config.risk_per_trade
                if position.unrealized_pnl < -max_loss:
                    should_exit = True
                    exit_reason = "Maximum Loss Limit"
                
                if should_exit:
                    triggered_stops.append({
                        'symbol': symbol,
                        'position': position,
                        'reason': exit_reason,
                        'current_price': position.current_price
                    })
            
        except Exception as e:
            logger.error(f"Error checking stop loss triggers: {e}")
        
        return triggered_stops
    
    def _update_performance_metrics(self, realized_pnl: float):
        """Update performance metrics with new trade"""
        try:
            # Update daily P&L
            self.daily_pnl += realized_pnl
            self.trades_today += 1
            
            # Update equity curve
            current_equity = self.trading_config.account_balance + self.daily_pnl
            self.equity_curve.append(current_equity)
            
            # Calculate current drawdown
            if len(self.equity_curve) > 1:
                peak = max(self.equity_curve)
                current_drawdown = (current_equity - peak) / peak
                self.risk_metrics.current_drawdown = current_drawdown
                
                # Update max drawdown
                if current_drawdown < self.risk_metrics.max_drawdown:
                    self.risk_metrics.max_drawdown = current_drawdown
            
            # Update consecutive losses
            if realized_pnl < 0:
                self.risk_metrics.consecutive_losses += 1
            else:
                self.risk_metrics.consecutive_losses = 0
            
            # Update win rate
            if len(self.equity_curve) > 10:  # Need some history
                recent_trades = self.equity_curve[-10:]
                wins = sum(1 for i in range(1, len(recent_trades)) if recent_trades[i] > recent_trades[i-1])
                self.risk_metrics.win_rate = wins / (len(recent_trades) - 1)
            
            # Check circuit breaker
            self._check_circuit_breaker()
            
            # Determine risk level
            self._update_risk_level()
            
            # Save performance data
            self._save_performance_data()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should be triggered"""
        try:
            circuit_breaker_loss = self.trading_config.account_balance * self.risk_config.emergency_stop_enabled
            
            if self.risk_config.emergency_stop_enabled:
                # Check daily loss circuit breaker
                if self.daily_pnl < -circuit_breaker_loss:
                    if not self.circuit_breaker_triggered:
                        self.circuit_breaker_triggered = True
                        self.circuit_breaker_time = datetime.now()
                        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED - Daily loss: ₹{abs(self.daily_pnl):.2f}")
                
                # Check drawdown circuit breaker
                if self.risk_metrics.current_drawdown < -0.15:  # 15% drawdown
                    if not self.circuit_breaker_triggered:
                        self.circuit_breaker_triggered = True
                        self.circuit_breaker_time = datetime.now()
                        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED - Drawdown: {self.risk_metrics.current_drawdown:.1%}")
            
            # Auto-reset circuit breaker after 1 hour
            if (self.circuit_breaker_triggered and 
                self.circuit_breaker_time and 
                datetime.now() - self.circuit_breaker_time > timedelta(hours=1)):
                
                self.circuit_breaker_triggered = False
                self.circuit_breaker_time = None
                logger.info("Circuit breaker auto-reset after 1 hour")
                
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
    
    def _update_risk_level(self):
        """Update current risk level based on metrics"""
        try:
            risk_score = 0
            
            # Drawdown factor
            if self.risk_metrics.current_drawdown < -0.1:
                risk_score += 3
            elif self.risk_metrics.current_drawdown < -0.05:
                risk_score += 2
            elif self.risk_metrics.current_drawdown < -0.02:
                risk_score += 1
            
            # Consecutive losses factor
            if self.risk_metrics.consecutive_losses >= 5:
                risk_score += 3
            elif self.risk_metrics.consecutive_losses >= 3:
                risk_score += 2
            elif self.risk_metrics.consecutive_losses >= 2:
                risk_score += 1
            
            # Daily loss factor
            daily_loss_pct = self.daily_pnl / self.trading_config.account_balance
            if daily_loss_pct < -0.05:
                risk_score += 3
            elif daily_loss_pct < -0.03:
                risk_score += 2
            elif daily_loss_pct < -0.01:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                self.risk_metrics.risk_level = RiskLevel.CRITICAL
            elif risk_score >= 4:
                self.risk_metrics.risk_level = RiskLevel.HIGH
            elif risk_score >= 2:
                self.risk_metrics.risk_level = RiskLevel.MEDIUM
            else:
                self.risk_metrics.risk_level = RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error updating risk level: {e}")
            self.risk_metrics.risk_level = RiskLevel.MEDIUM
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            total_position_value = sum(
                pos.quantity * pos.current_price 
                for pos in self.positions.values()
            )
            
            total_unrealized_pnl = sum(
                pos.unrealized_pnl 
                for pos in self.positions.values()
            )
            
            return {
                'risk_level': self.risk_metrics.risk_level.value,
                'current_drawdown': self.risk_metrics.current_drawdown,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'daily_pnl': self.daily_pnl,
                'trades_today': self.trades_today,
                'consecutive_losses': self.risk_metrics.consecutive_losses,
                'win_rate': self.risk_metrics.win_rate,
                'circuit_breaker_active': self.circuit_breaker_triggered,
                'total_positions': len(self.positions),
                'total_position_value': total_position_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'account_balance': self.trading_config.account_balance,
                'available_capital': self.trading_config.account_balance - total_position_value
            }
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {'error': str(e)}
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of each trading day)"""
        try:
            self.trades_today = 0
            self.daily_pnl = 0.0
            
            # Reset circuit breaker if it was triggered yesterday
            if self.circuit_breaker_triggered:
                self.circuit_breaker_triggered = False
                self.circuit_breaker_time = None
                logger.info("Circuit breaker reset for new trading day")
            
            logger.info("Daily risk metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")
    
    def emergency_close_all_positions(self) -> List[str]:
        """Emergency close all positions (returns list of symbols to close)"""
        try:
            symbols_to_close = list(self.positions.keys())
            
            if symbols_to_close:
                logger.critical(f"🚨 EMERGENCY CLOSE triggered for {len(symbols_to_close)} positions")
                
                # Clear positions (actual closing should be handled by executor)
                self.positions.clear()
                
                # Trigger circuit breaker
                self.circuit_breaker_triggered = True
                self.circuit_breaker_time = datetime.now()
            
            return symbols_to_close
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
            return []
