import os
from dotenv import load_dotenv
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Centralized settings class with dynamic account balance support"""
    
    # API Configuration
    KITE_API_KEY = os.getenv('KITE_API_KEY', 't4otrxd7h438r47b')
    KITE_API_SECRET = os.getenv('KITE_API_SECRET', '7eeyv2x2c3dje7cg3typakyzozidzbq4')
    KITE_REDIRECT_URI = os.getenv('KITE_REDIRECT_URI', 'http://localhost:3000')
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    CONFIG_DIR = BASE_DIR / 'config'
    
    TOKEN_FILE = DATA_DIR / 'kite_tokens.json'
    TRADING_CONFIG_FILE = DATA_DIR / 'trading_config.json'
    TRADING_PREFS_FILE = DATA_DIR / 'trading_preferences.json'
    LOG_FILE = LOGS_DIR / 'trading.log'
    
    # Trading Configuration
    STRATEGY_PARAMS = {
        "atr_period": int(os.getenv('ATR_PERIOD', '10')),
        "factor": float(os.getenv('FACTOR', '3.0')),
        "account_balance": float(os.getenv('ACCOUNT_BALANCE', '4000.0')),  # Default value
        "capital_allocation_percent": float(os.getenv('CAPITAL_ALLOCATION_PERCENT', '100.0')),
        "fixed_stop_loss": float(os.getenv('FIXED_STOP_LOSS', '100.0')),
        "max_positions": int(os.getenv('MAX_POSITIONS', '1')),
        "min_trade_amount": 100,
        "max_trade_amount": 50000,
        "market_open_hour": int(os.getenv('MARKET_OPEN_HOUR', '9')),
        "market_open_minute": int(os.getenv('MARKET_OPEN_MINUTE', '15')),
        "market_close_hour": int(os.getenv('MARKET_CLOSE_HOUR', '15')),
        "market_close_minute": int(os.getenv('MARKET_CLOSE_MINUTE', '30')),
        "check_interval": 30,
        "historical_days": 3,
        "min_candles_required": int(os.getenv('MIN_CANDLES', '50'))
    }
    
    # Safety Configuration
    SAFETY_CONFIG = {
        "live_trading_enabled": os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true',
        "dry_run_mode": os.getenv('DRY_RUN_MODE', 'true').lower() == 'true',
        "paper_trading_balance": float(os.getenv('PAPER_TRADING_BALANCE', '10000.0')),
        "enable_console_alerts": True,
        "enable_trade_logging": True
    }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.CONFIG_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def load_trading_config(cls) -> Dict[str, Any]:
        """Load trading configuration from file"""
        if cls.TRADING_CONFIG_FILE.exists():
            with open(cls.TRADING_CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_trading_config(cls, config: Dict[str, Any]):
        """Save trading configuration to file"""
        cls.ensure_directories()
        with open(cls.TRADING_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def get_trading_amount(cls):
        """Get trading amount from preferences or environment"""
        # Ensure directories exist
        cls.ensure_directories()
        
        # First check if preferences file exists
        if cls.TRADING_PREFS_FILE.exists():
            try:
                with open(cls.TRADING_PREFS_FILE, 'r') as f:
                    prefs = json.load(f)
                    amount = prefs.get('last_trading_amount', cls.STRATEGY_PARAMS['account_balance'])
                    # Update the STRATEGY_PARAMS with loaded amount
                    cls.STRATEGY_PARAMS['account_balance'] = amount
                    return amount
            except Exception as e:
                pass
        
        # Return current value from STRATEGY_PARAMS
        return cls.STRATEGY_PARAMS['account_balance']
    
    @classmethod
    def update_trading_amount(cls, amount: float):
        """Update trading amount dynamically"""
        cls.STRATEGY_PARAMS['account_balance'] = amount
        
        # Also save to preferences
        prefs = {}
        if cls.TRADING_PREFS_FILE.exists():
            try:
                with open(cls.TRADING_PREFS_FILE, 'r') as f:
                    prefs = json.load(f)
            except:
                pass
        
        prefs['last_trading_amount'] = amount
        prefs['last_update'] = datetime.now().isoformat()
        
        cls.ensure_directories()
        with open(cls.TRADING_PREFS_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)

# Load trading amount on startup
Settings.get_trading_amount()