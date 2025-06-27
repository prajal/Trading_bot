import json
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Individual trade record"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    duration_minutes: int = 0
    trade_type: str = "LONG"  # LONG/SHORT
    exit_reason: str = ""  # SIGNAL/STOP_LOSS/MANUAL/PRE_CLOSE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'duration_minutes': self.duration_minutes,
            'trade_type': self.trade_type,
            'exit_reason': self.exit_reason
        }

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_mb': self.memory_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb
        }

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    Tracks trading performance, system metrics, and generates analytics
    """
    
    def __init__(self, data_dir: Path, trading_config):
        self.data_dir = data_dir
        self.trading_config = trading_config
        
        # File paths
        self.trades_file = data_dir / "trades_history.json"
        self.metrics_file = data_dir / "performance_metrics.json"
        self.system_metrics_file = data_dir / "system_metrics.json"
        self.session_file = data_dir / "current_session.json"
        
        # Performance tracking
        self.current_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now(),
            'end_time': None,
            'trades': [],
            'total_pnl': 0.0,
            'loop_count': 0,
            'signals_generated': 0,
            'orders_placed': 0
        }
        
        # Historical data
        self.trades_history: List[TradeRecord] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        # Real-time metrics
        self.current_trade: Optional[TradeRecord] = None
        self.equity_curve: List[float] = [trading_config.account_balance]
        self.daily_returns: List[float] = []
        
        # Performance calculations
        self.max_equity = trading_config.account_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # System monitoring
        self.last_system_check = datetime.now()
        self.system_check_interval = timedelta(minutes=5)
        
        # Load historical data
        self._load_historical_data()
        
        logger.info("Performance Monitor initialized")
    
    def _load_historical_data(self):
        """Load historical performance data"""
        try:
            # Load trades history
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    
                for trade_dict in trades_data:
                    trade = TradeRecord(
                        symbol=trade_dict['symbol'],
                        entry_time=datetime.fromisoformat(trade_dict['entry_time']) if trade_dict['entry_time'] else None,
                        exit_time=datetime.fromisoformat(trade_dict['exit_time']) if trade_dict['exit_time'] else None,
                        entry_price=trade_dict['entry_price'],
                        exit_price=trade_dict['exit_price'],
                        quantity=trade_dict['quantity'],
                        pnl=trade_dict['pnl'],
                        duration_minutes=trade_dict['duration_minutes'],
                        trade_type=trade_dict['trade_type'],
                        exit_reason=trade_dict['exit_reason']
                    )
                    self.trades_history.append(trade)
                
                logger.info(f"Loaded {len(self.trades_history)} historical trades")
            
            # Load performance metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    
                self.equity_curve = metrics_data.get('equity_curve', [self.trading_config.account_balance])
                self.daily_returns = metrics_data.get('daily_returns', [])
                self.max_drawdown = metrics_data.get('max_drawdown', 0.0)
                
                logger.info(f"Loaded performance metrics: {len(self.equity_curve)} equity points")
                
        except Exception as e:
            logger.warning(f"Could not load historical performance data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to files"""
        try:
            # Save trades history
            trades_data = [trade.to_dict() for trade in self.trades_history]
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
            
            # Save performance metrics
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'equity_curve': self.equity_curve[-1000:],  # Keep last 1000 points
                'daily_returns': self.daily_returns[-252:],  # Keep last year
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save current session
            session_data = {
                'session_id': self.current_session['session_id'],
                'start_time': self.current_session['start_time'].isoformat(),
                'end_time': self.current_session['end_time'].isoformat() if self.current_session['end_time'] else None,
                'trades': [trade.to_dict() for trade in self.current_session['trades']],
                'total_pnl': self.current_session['total_pnl'],
                'loop_count': self.current_session['loop_count'],
                'signals_generated': self.current_session['signals_generated'],
                'orders_placed': self.current_session['orders_placed']
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def record_trade_entry(self, symbol: str, quantity: int, entry_price: float, entry_time: datetime):
        """Record trade entry"""
        try:
            self.current_trade = TradeRecord(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                quantity=quantity,
                trade_type="LONG"  # Assuming long trades for now
            )
            
            self.current_session['orders_placed'] += 1
            
            logger.info(f"Trade entry recorded: {quantity} {symbol} @ ₹{entry_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade entry: {e}")
    
    def record_trade_exit(self, symbol: str, exit_price: float, exit_time: datetime, 
                         pnl: float, exit_reason: str = "SIGNAL"):
        """Record trade exit and calculate metrics"""
        try:
            if not self.current_trade or self.current_trade.symbol != symbol:
                logger.warning(f"No matching trade entry found for exit: {symbol}")
                return
            
            # Complete the trade record
            self.current_trade.exit_time = exit_time
            self.current_trade.exit_price = exit_price
            self.current_trade.pnl = pnl
            self.current_trade.exit_reason = exit_reason
            
            # Calculate duration
            if self.current_trade.entry_time:
                duration = exit_time - self.current_trade.entry_time
                self.current_trade.duration_minutes = int(duration.total_seconds() / 60)
            
            # Add to history
            self.trades_history.append(self.current_trade)
            self.current_session['trades'].append(self.current_trade)
            self.current_session['total_pnl'] += pnl
            
            # Update equity curve
            current_equity = self.equity_curve[-1] + pnl
            self.equity_curve.append(current_equity)
            
            # Update max equity and drawdown
            if current_equity > self.max_equity:
                self.max_equity = current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (current_equity - self.max_equity) / self.max_equity
                if self.current_drawdown < self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
            
            logger.info(f"Trade exit recorded: {symbol} P&L=₹{pnl:.2f}, Duration={self.current_trade.duration_minutes}min")
            
            # Clear current trade
            self.current_trade = None
            
            # Save data
            self._save_performance_data()
            
        except Exception as e:
            logger.error(f"Error recording trade exit: {e}")
    
    def update_loop_metrics(self, loop_count: int, signal: str, current_price: float, position: Dict[str, Any]):
        """Update loop-level metrics"""
        try:
            self.current_session['loop_count'] = loop_count
            
            if signal != "HOLD":
                self.current_session['signals_generated'] += 1
            
            # Update current P&L if in position
            if position.get('quantity', 0) > 0 and self.current_trade:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                
                # Update equity curve with unrealized P&L
                if len(self.equity_curve) > 0:
                    base_equity = self.equity_curve[-1] if not self.current_trade else self.equity_curve[-2]
                    current_equity = base_equity + unrealized_pnl
                    
                    # Update drawdown
                    if current_equity > self.max_equity:
                        self.max_equity = current_equity
                        self.current_drawdown = 0.0
                    else:
                        self.current_drawdown = (current_equity - self.max_equity) / self.max_equity
            
            # Periodic system metrics collection
            if datetime.now() - self.last_system_check >= self.system_check_interval:
                self._collect_system_metrics()
                self.last_system_check = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating loop metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_mb=network.bytes_sent / (1024 * 1024),
                network_recv_mb=network.bytes_recv / (1024 * 1024)
            )
            
            self.system_metrics_history.append(metrics)
            
            # Keep only last 1000 system metrics
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-1000:]
            
            # Log warning if resources are high
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 80:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'average_pnl': 0.0,
                    'max_win': 0.0,
                    'max_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'average_trade_duration': 0.0
                }
            
            # Basic trade statistics
            total_trades = len(self.trades_history)
            winning_trades = [t for t in self.trades_history if t.pnl > 0]
            losing_trades = [t for t in self.trades_history if t.pnl <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in self.trades_history)
            average_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            max_win = max((t.pnl for t in self.trades_history), default=0)
            max_loss = min((t.pnl for t in self.trades_history), default=0)
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Return calculation
            total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] if len(self.equity_curve) > 1 else 0
            
            # Sharpe ratio (simplified)
            if len(self.daily_returns) > 1:
                avg_return = np.mean(self.daily_returns)
                std_return = np.std(self.daily_returns)
                sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Average trade duration
            avg_duration = np.mean([t.duration_minutes for t in self.trades_history if t.duration_minutes > 0])
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': average_pnl,
                'max_win': max_win,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                '
