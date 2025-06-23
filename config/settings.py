import os
from dotenv import load_dotenv
from typing import Dict, Any
import json
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Centralized settings class"""
    
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
    LOG_FILE = LOGS_DIR / 'trading.log'
    
    # Trading Configuration
    STRATEGY_PARAMS = {
        "atr_period": 10,
        "factor": 3.0,
        "account_balance": 4000.0,
        "capital_allocation_percent": 100.0,
        "fixed_stop_loss": 100.0,
        "max_positions": 1,
        "min_trade_amount": 100,
        "max_trade_amount": 4000,
        "market_open_hour": 9,
        "market_open_minute": 15,
        "market_close_hour": 15,
        "market_close_minute": 30,
        "check_interval": 30,
        "historical_days": 3,
        "min_candles_required": 50
    }
    
    # Safety Configuration
    SAFETY_CONFIG = {
        "live_trading_enabled": os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true',
        "dry_run_mode": os.getenv('DRY_RUN_MODE', 'true').lower() == 'true',
        "paper_trading_balance": 10000,
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
