import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured logs with additional context
    """
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if self.include_extra_fields and hasattr(record, '__dict__'):
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'message', 'exc_info', 'exc_text', 
                              'stack_info']:
                    try:
                        # Only include serializable values
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

class ColoredConsoleFormatter(logging.Formatter):
    """
    Console formatter with colors for different log levels
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self):
        super().__init__()
        self.use_colors = sys.stdout.isatty()  # Only use colors if output is a terminal
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            # Format: [TIMESTAMP] LEVEL: MESSAGE
            formatted = (f"{color}[{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"{record.levelname}: {record.getMessage()}{reset}")
            
            # Add exception info if present
            if record.exc_info:
                formatted += f"\n{color}{self.formatException(record.exc_info)}{reset}"
            
            return formatted
        else:
            # No colors for non-terminal output
            formatted = (f"[{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"{record.levelname}: {record.getMessage()}")
            
            if record.exc_info:
                formatted += f"\n{self.formatException(record.exc_info)}"
            
            return formatted

class TradingLogFilter(logging.Filter):
    """
    Custom filter to add trading-specific context to log records
    """
    
    def __init__(self):
        super().__init__()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add session ID to all records
        record.session_id = self.session_id
        
        # Add trading context if available
        if hasattr(record, 'symbol'):
            record.trading_symbol = getattr(record, 'symbol', '')
        
        if hasattr(record, 'order_id'):
            record.order_id = getattr(record, 'order_id', '')
        
        return True

class EnhancedLogger:
    """
    Enhanced logging system for trading bot with multiple handlers,
    structured logging, and performance tracking
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logs_dir = base_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create different log files
        self.main_log_file = self.logs_dir / 'trading.log'
        self.error_log_file = self.logs_dir / 'errors.log'
        self.performance_log_file = self.logs_dir / 'performance.log'
        self.structured_log_file = self.logs_dir / 'structured.jsonl'
        
        # Track logger instances
        self._loggers: Dict[str, logging.Logger] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'log_entries': 0,
            'errors': 0,
            'warnings': 0,
            'session_start': datetime.now()
        }
        
        # Custom filter
        self.trading_filter = TradingLogFilter()
        
        # Initialize logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredConsoleFormatter())
        console_handler.addFilter(self.trading_filter)
        root_logger.addHandler(console_handler)
        
        # Main log file (rotating)
        main_file_handler = RotatingFileHandler(
            self.main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        main_file_handler.setLevel(logging.DEBUG)
        main_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s'
        ))
        main_file_handler.addFilter(self.trading_filter)
        root_logger.addHandler(main_file_handler)
        
        # Error log file (only errors and critical)
        error_file_handler = RotatingFileHandler(
            self.error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s\n'
            'Session: %(session_id)s\n'
            '%(pathname)s\n'
        ))
        error_file_handler.addFilter(self.trading_filter)
        root_logger.addHandler(error_file_handler)
        
        # Structured log file (JSON format)
        structured_handler = TimedRotatingFileHandler(
            self.structured_log_file,
            when='midnight',
            interval=1,
            backupCount=30
        )
        structured_handler.setLevel(logging.INFO)
        structured_handler.setFormatter(StructuredFormatter())
        structured_handler.addFilter(self.trading_filter)
        root_logger.addHandler(structured_handler)
        
        # Performance log file
        performance_handler = RotatingFileHandler(
            self.performance_log_file,
            maxBytes=2*1024*1024,  # 2MB
            backupCount=3
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] PERF: %(message)s'
        ))
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(performance_handler)
        perf_logger.propagate = False  # Don't propagate to root logger
        
        # Set levels for specific loggers to reduce noise
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('kiteconnect').setLevel(logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
            
            # Add custom methods for trading-specific logging
            self._add_trading_methods(logger)
        
        return self._loggers[name]
    
    def _add_trading_methods(self, logger: logging.Logger):
        """Add trading-specific logging methods to logger"""
        
        def log_trade(symbol: str, action: str, quantity: int, price: float, 
                     order_id: str = None, pnl: float = None, **kwargs):
            """Log trade execution"""
            extra = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'log_type': 'trade'
            }
            
            if order_id:
                extra['order_id'] = order_id
            if pnl is not None:
                extra['pnl'] = pnl
            
            extra.update(kwargs)
            
            message = f"TRADE: {action} {quantity} {symbol} @ ₹{price:.2f}"
            if pnl is not None:
                message += f" | P&L: ₹{pnl:.2f}"
            
            logger.info(message, extra=extra)
        
        def log_signal(symbol: str, signal: str, confidence: float = None, 
                      direction: str = None, **kwargs):
            """Log trading signal"""
            extra = {
                'symbol': symbol,
                'signal': signal,
                'log_type': 'signal'
            }
            
            if confidence is not None:
                extra['confidence'] = confidence
            if direction:
                extra['direction'] = direction
            
            extra.update(kwargs)
            
            message = f"SIGNAL: {signal} for {symbol}"
            if confidence is not None:
                message += f" (confidence: {confidence:.2f})"
            
            logger.info(message, extra=extra)
        
        def log_performance(metric: str, value: float, context: str = None, **kwargs):
            """Log performance metric"""
            extra = {
                'metric': metric,
                'value': value,
                'log_type': 'performance'
            }
            
            if context:
                extra['context'] = context
            
            extra.update(kwargs)
            
            message = f"PERF: {metric} = {value}"
            if context:
                message += f" ({context})"
            
            # Log to performance logger
            perf_logger = logging.getLogger('performance')
            perf_logger.info(message, extra=extra)
        
        def log_error_with_context(error: Exception, context: str = None, **kwargs):
            """Log error with additional context"""
            extra = {
                'error_type': type(error).__name__,
                'log_type': 'error',
                'error_message': str(error)
            }
            
            if context:
                extra['context'] = context
            
            extra.update(kwargs)
            
            message = f"ERROR: {type(error).__name__}: {error}"
            if context:
                message = f"ERROR in {context}: {type(error).__name__}: {error}"
            
            logger.error(message, extra=extra, exc_info=True)
        
        # Add methods to logger instance
        logger.log_trade = log_trade
        logger.log_signal = log_signal
        logger.log_performance = log_performance
        logger.log_error_with_context = log_error_with_context
    
    def get_performance_logger(self) -> logging.Logger:
        """Get performance-specific logger"""
        return logging.getLogger('performance')
    
    def log_session_start(self, config: Dict[str, Any]):
        """Log session start with configuration"""
        logger = self.get_logger('session')
        logger.info("Trading session started", extra={
            'log_type': 'session_start',
            'config': config,
            'session_id': self.trading_filter.session_id
        })
    
    def log_session_end(self, summary: Dict[str, Any]):
        """Log session end with summary"""
        logger = self.get_logger('session')
        logger.info("Trading session ended", extra={
            'log_type': 'session_end',
            'summary': summary,
            'session_id': self.trading_filter.session_id
        })
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        try:
            stats = {
                'session_duration': str(datetime.now() - self.performance_metrics['session_start']),
                'log_files': {
                    'main_log_size': self._get_file_size(self.main_log_file),
                    'error_log_size': self._get_file_size(self.error_log_file),
                    'performance_log_size': self._get_file_size(self.performance_log_file),
                    'structured_log_size': self._get_file_size(self.structured_log_file)
                },
                'loggers_created': len(self._loggers),
                'session_id': self.trading_filter.session_id
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size"""
        try:
            if file_path.exists():
                size = file_path.stat().st_size
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size < 1024.0:
                        return f"{size:.1f} {unit}"
                    size /= 1024.0
                return f"{size:.1f} TB"
            return "0 B"
        except Exception:
            return "Unknown"
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.logs_dir.glob('*.log*'):
                try:
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        print(f"Cleaned up old log file: {log_file.name}")
                except Exception as e:
                    print(f"Error cleaning up {log_file.name}: {e}")
                    
        except Exception as e:
            print(f"Error during log cleanup: {e}")

# Global logger instance
_global_logger_instance: Optional[EnhancedLogger] = None

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with enhanced features
    This is the main function used throughout the application
    """
    global _global_logger_instance
    
    if _global_logger_instance is None:
        # Initialize with default location
        try:
            from config.enhanced_settings import Settings
            base_dir = Settings.BASE_DIR
        except ImportError:
            base_dir = Path(__file__).parent.parent
        
        _global_logger_instance = EnhancedLogger(base_dir)
    
    return _global_logger_instance.get_logger(name)

def get_performance_logger() -> logging.Logger:
    """Get performance-specific logger"""
    global _global_logger_instance
    
    if _global_logger_instance is None:
        get_logger('init')  # Initialize the global instance
    
    return _global_logger_instance.get_performance_logger()

def log_session_start(config: Dict[str, Any]):
    """Log session start"""
    global _global_logger_instance
    
    if _global_logger_instance is None:
        get_logger('init')
    
    _global_logger_instance.log_session_start(config)

def log_session_end(summary: Dict[str, Any]):
    """Log session end"""
    global _global_logger_instance
    
    if _global_logger_instance is None:
        get_logger('init')
    
    _global_logger_instance.log_session_end(summary)

def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics"""
    global _global_logger_instance
    
    if _global_logger_instance is None:
        return {'error': 'Logger not initialized'}
    
    return _global_logger_instance.get_log_stats()

def cleanup_old_logs(days_to_keep: int = 30):
    """Clean up old log files"""
    global _global_logger_instance
    
    if _global_logger_instance is None:
        get_logger('init')
    
    _global_logger_instance.cleanup_old_logs(days_to_keep)