import os
from dotenv import load_dotenv
from typing import Dict, Any
import json
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """Trading configuration with validation"""
    def __init__(self):
        self.account_balance = float(os.getenv('ACCOUNT_BALANCE', '4000.0'))
        self.capital_allocation_percent = float(os.getenv('CAPITAL_ALLOCATION_PERCENT', '100.0'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '1'))
        self.fixed_stop_loss = float(os.getenv('FIXED_STOP_LOSS', '100.0'))
        
        # Validation
        if self.account_balance <= 0:
            raise ValueError("Account balance must be positive")
        if not 0 < self.capital_allocation_percent <= 100:
            raise ValueError("Capital allocation must be between 0 and 100")
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        if self.fixed_stop_loss <= 0:
            raise ValueError("Stop loss must be positive")

class StrategyConfig:
    """Strategy configuration with validation"""
    def __init__(self):
        self.atr_period = int(os.getenv('ATR_PERIOD', '10'))
        self.factor = float(os.getenv('FACTOR', '3.0'))
        self.min_candles_required = int(os.getenv('MIN_CANDLES', '50'))
        
        # Validation
        if self.atr_period < 1:
            raise ValueError("ATR period must be positive")
        if self.factor <= 0:
            raise ValueError("Factor must be positive")
        if self.min_candles_required < 10:
            raise ValueError("Minimum candles must be at least 10")

class Settings:
    """Centralized settings with validation"""
    
    # API Configuration
    KITE_API_KEY = os.getenv('KITE_API_KEY')
    KITE_API_SECRET = os.getenv('KITE_API_SECRET')
    KITE_REDIRECT_URI = os.getenv('KITE_REDIRECT_URI', 'http://localhost:3000')
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    CONFIG_DIR = BASE_DIR / 'config'
    
    TOKEN_FILE = DATA_DIR / 'kite_tokens.json'
    TRADING_CONFIG_FILE = DATA_DIR / 'trading_config.json'
    LOG_FILE = LOGS_DIR / 'trading.log'
    
    # Initialize configurations with validation
    try:
        TRADING_CONFIG = TradingConfig()
        STRATEGY_CONFIG = StrategyConfig()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please check your .env file or use default values")
        # Use defaults if validation fails
        TRADING_CONFIG = TradingConfig()
        STRATEGY_CONFIG = StrategyConfig()
    
    # Safety Configuration
    SAFETY_CONFIG = {
        "live_trading_enabled": os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true',
        "dry_run_mode": os.getenv('DRY_RUN_MODE', 'true').lower() == 'true',
        "paper_trading_balance": float(os.getenv('PAPER_TRADING_BALANCE', '10000.0')),
        "enable_console_alerts": os.getenv('ENABLE_CONSOLE_ALERTS', 'true').lower() == 'true',
        "enable_trade_logging": os.getenv('ENABLE_TRADE_LOGGING', 'true').lower() == 'true'
    }
    
    # Market hours
    MARKET_OPEN_HOUR = int(os.getenv('MARKET_OPEN_HOUR', '9'))
    MARKET_OPEN_MINUTE = int(os.getenv('MARKET_OPEN_MINUTE', '15'))
    MARKET_CLOSE_HOUR = int(os.getenv('MARKET_CLOSE_HOUR', '15'))
    MARKET_CLOSE_MINUTE = int(os.getenv('MARKET_CLOSE_MINUTE', '30'))
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.CONFIG_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_configuration(cls):
        """Validate all configuration settings"""
        errors = []
        
        # Check API credentials
        if not cls.KITE_API_KEY:
            errors.append("KITE_API_KEY not set")
        if not cls.KITE_API_SECRET:
            errors.append("KITE_API_SECRET not set")
        
        # Check trading configuration
        try:
            cls.TRADING_CONFIG
        except ValueError as e:
            errors.append(f"Trading config error: {e}")
        
        # Check strategy configuration
        try:
            cls.STRATEGY_CONFIG
        except ValueError as e:
            errors.append(f"Strategy config error: {e}")
        
        return errors
    
    @classmethod
    def print_configuration(cls):
        """Print current configuration"""
        print("📊 Current Configuration:")
        print(f"   Account Balance: ₹{cls.TRADING_CONFIG.account_balance:,.2f}")
        print(f"   Capital Allocation: {cls.TRADING_CONFIG.capital_allocation_percent}%")
        print(f"   Max Positions: {cls.TRADING_CONFIG.max_positions}")
        print(f"   Stop Loss: ₹{cls.TRADING_CONFIG.fixed_stop_loss}")
        print(f"   ATR Period: {cls.STRATEGY_CONFIG.atr_period}")
        print(f"   Factor: {cls.STRATEGY_CONFIG.factor}")
        print(f"   Live Trading: {cls.SAFETY_CONFIG['live_trading_enabled']}")
        print(f"   Dry Run Mode: {cls.SAFETY_CONFIG['dry_run_mode']}")
