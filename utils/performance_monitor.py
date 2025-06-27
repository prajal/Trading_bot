import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  Warning: psutil not installed. System monitoring disabled.")

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
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    
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
    Performance monitoring system with optional system metrics
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
        
        if not PSUTIL_AVAILABLE:
            logger.warning("System monitoring disabled - psutil not available")
        
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
            
            # Periodic system metrics collection (only if psutil available)
            if PSUTIL_AVAILABLE and datetime.now() - self.last_system_check >= self.system_check_interval:
                self._collect_system_metrics()
                self.last_system_check = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating loop metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics (only if psutil available)"""
        if not PSUTIL_AVAILABLE:
            return
            
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
                'average_trade_duration': avg_duration,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'current_equity': self.equity_curve[-1] if self.equity_curve else self.trading_config.account_balance
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current session metrics"""
        try:
            session_duration = datetime.now() - self.current_session['start_time']
            
            metrics = {
                'session_id': self.current_session['session_id'],
                'session_duration_minutes': int(session_duration.total_seconds() / 60),
                'loop_count': self.current_session['loop_count'],
                'signals_generated': self.current_session['signals_generated'],
                'orders_placed': self.current_session['orders_placed'],
                'trades_completed': len(self.current_session['trades']),
                'current_pnl': self.current_session['total_pnl'],
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'current_equity': self.equity_curve[-1] if self.equity_curve else self.trading_config.account_balance
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {'error': str(e)}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            current_metrics = self.get_current_metrics()
            performance_metrics = self.calculate_performance_metrics()
            
            # System performance summary (only if psutil available)
            if PSUTIL_AVAILABLE and self.system_metrics_history:
                recent_system = self.system_metrics_history[-10:]  # Last 10 readings
                avg_cpu = np.mean([m.cpu_percent for m in recent_system])
                avg_memory = np.mean([m.memory_percent for m in recent_system])
                max_cpu = max([m.cpu_percent for m in recent_system])
                max_memory = max([m.memory_percent for m in recent_system])
            else:
                avg_cpu = avg_memory = max_cpu = max_memory = 0
            
            summary = {
                'session_metrics': current_metrics,
                'performance_metrics': performance_metrics,
                'system_performance': {
                    'average_cpu_percent': avg_cpu,
                    'average_memory_percent': avg_memory,
                    'peak_cpu_percent': max_cpu,
                    'peak_memory_percent': max_memory,
                    'system_checks_performed': len(self.system_metrics_history),
                    'system_monitoring_enabled': PSUTIL_AVAILABLE
                },
                'trading_summary': {
                    'trades_per_hour': self._calculate_trades_per_hour(),
                    'signals_per_hour': self._calculate_signals_per_hour(),
                    'execution_efficiency': self._calculate_execution_efficiency()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {'error': str(e)}
    
    def _calculate_trades_per_hour(self) -> float:
        """Calculate trades per hour for current session"""
        try:
            session_duration = datetime.now() - self.current_session['start_time']
            hours = session_duration.total_seconds() / 3600
            
            if hours > 0:
                return len(self.current_session['trades']) / hours
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_signals_per_hour(self) -> float:
        """Calculate signals per hour for current session"""
        try:
            session_duration = datetime.now() - self.current_session['start_time']
            hours = session_duration.total_seconds() / 3600
            
            if hours > 0:
                return self.current_session['signals_generated'] / hours
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_execution_efficiency(self) -> float:
        """Calculate execution efficiency (orders placed vs signals generated)"""
        try:
            if self.current_session['signals_generated'] > 0:
                return self.current_session['orders_placed'] / self.current_session['signals_generated']
            return 0.0
            
        except Exception:
            return 0.0
    
    def export_performance_report(self, filepath: Optional[Path] = None) -> str:
        """Export comprehensive performance report"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = self.data_dir / f"performance_report_{timestamp}.json"
            
            report = {
                'report_generated': datetime.now().isoformat(),
                'session_summary': self.get_session_summary(),
                'performance_metrics': self.calculate_performance_metrics(),
                'configuration': {
                    'account_balance': self.trading_config.account_balance,
                    'risk_per_trade': self.trading_config.risk_per_trade,
                    'max_daily_loss': self.trading_config.max_daily_loss,
                    'position_sizing_method': self.trading_config.position_sizing_method
                },
                'system_info': {
                    'psutil_available': PSUTIL_AVAILABLE,
                    'system_monitoring_enabled': PSUTIL_AVAILABLE
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return f"Error: {e}"
    
    def save_session_data(self):
        """Save current session data"""
        try:
            self.current_session['end_time'] = datetime.now()
            self._save_performance_data()
            logger.info("Session data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        try:
            current_metrics = self.get_current_metrics()
            performance_metrics = self.calculate_performance_metrics()
            
            # Recent system metrics (last 5 readings) - only if available
            recent_system = self.system_metrics_history[-5:] if self.system_metrics_history else []
            
            dashboard_data = {
                'current_time': datetime.now().isoformat(),
                'session_id': self.current_session['session_id'],
                'session_duration': current_metrics.get('session_duration_minutes', 0),
                'current_equity': current_metrics.get('current_equity', 0),
                'current_pnl': current_metrics.get('current_pnl', 0),
                'current_drawdown': current_metrics.get('current_drawdown', 0),
                'trades_today': len(self.current_session['trades']),
                'signals_today': self.current_session['signals_generated'],
                'loop_count': self.current_session['loop_count'],
                'win_rate': performance_metrics.get('win_rate', 0),
                'total_trades': performance_metrics.get('total_trades', 0),
                'system_cpu': recent_system[-1].cpu_percent if recent_system else 0,
                'system_memory': recent_system[-1].memory_percent if recent_system else 0,
                'system_monitoring_available': PSUTIL_AVAILABLE,
                'last_update': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}