#!/usr/bin/env python3
"""
Quick Improvements for Trading Bot
==================================

This script implements immediate improvements for better configuration management,
error handling, and data validation before tomorrow's trading.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_env_template():
    """Create .env.example template"""
    env_content = """# Zerodha Kite Connect API Credentials
# Get these from https://kite.zerodha.com/connect/apps
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_REDIRECT_URI=http://localhost:3000

# Trading Configuration
ACCOUNT_BALANCE=4000.0
CAPITAL_ALLOCATION_PERCENT=100.0
MAX_POSITIONS=1
FIXED_STOP_LOSS=100.0

# Strategy Configuration
ATR_PERIOD=10
FACTOR=3.0
MIN_CANDLES=50

# Safety Configuration
LIVE_TRADING_ENABLED=false
DRY_RUN_MODE=true
PAPER_TRADING_BALANCE=10000.0

# Market Hours (24-hour format)
MARKET_OPEN_HOUR=9
MARKET_OPEN_MINUTE=15
MARKET_CLOSE_HOUR=15
MARKET_CLOSE_MINUTE=30

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_CONSOLE_ALERTS=true
ENABLE_TRADE_LOGGING=true
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env.example template")

def create_improved_settings():
    """Create improved settings with validation"""
    settings_content = '''import os
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
'''
    
    with open('config/improved_settings.py', 'w') as f:
        f.write(settings_content)
    
    print("✅ Created improved settings with validation")

def create_data_validators():
    """Create data validation utilities"""
    validators_content = '''from typing import Any, Dict, Union
import pandas as pd

class ValidationError(Exception):
    """Custom validation error"""
    pass

def validate_price(price: Union[int, float]) -> float:
    """Validate price data"""
    if not isinstance(price, (int, float)) or price <= 0:
        raise ValidationError("Price must be a positive number")
    return float(price)

def validate_quantity(quantity: int) -> int:
    """Validate quantity"""
    if not isinstance(quantity, int) or quantity <= 0:
        raise ValidationError("Quantity must be a positive integer")
    return quantity

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """Validate DataFrame has required columns"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    return df

def validate_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC data"""
    required_columns = ['open', 'high', 'low', 'close']
    df = validate_dataframe(df, required_columns)
    
    # Check for negative prices
    for col in required_columns:
        if (df[col] <= 0).any():
            raise ValidationError(f"Negative or zero prices found in {col}")
    
    # Check OHLC relationship
    if not ((df['high'] >= df['low']).all() and 
            (df['high'] >= df['open']).all() and 
            (df['high'] >= df['close']).all() and
            (df['low'] <= df['open']).all() and 
            (df['low'] <= df['close']).all()):
        raise ValidationError("Invalid OHLC relationship")
    
    return df

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary"""
    required_fields = ['account_balance', 'capital_allocation_percent']
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")
    
    if config['account_balance'] <= 0:
        raise ValidationError("Account balance must be positive")
    
    if not 0 < config['capital_allocation_percent'] <= 100:
        raise ValidationError("Capital allocation must be between 0 and 100")
    
    return config
'''
    
    with open('utils/validators.py', 'w') as f:
        f.write(validators_content)
    
    print("✅ Created data validation utilities")

def create_custom_exceptions():
    """Create custom exception classes"""
    exceptions_content = '''class TradingBotError(Exception):
    """Base exception for trading bot"""
    pass

class AuthenticationError(TradingBotError):
    """Authentication related errors"""
    pass

class OrderExecutionError(TradingBotError):
    """Order execution errors"""
    pass

class DataError(TradingBotError):
    """Data related errors"""
    pass

class ConfigurationError(TradingBotError):
    """Configuration errors"""
    pass

class ValidationError(TradingBotError):
    """Validation errors"""
    pass

class MarketError(TradingBotError):
    """Market related errors"""
    pass
'''
    
    with open('utils/exceptions.py', 'w') as f:
        f.write(exceptions_content)
    
    print("✅ Created custom exception classes")

def create_startup_checker():
    """Create startup configuration checker"""
    checker_content = '''#!/usr/bin/env python3
"""
Startup Configuration Checker
============================

This script validates all configuration before starting the trading bot.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_configuration():
    """Check all configuration settings"""
    print("🔍 Checking Configuration...")
    
    try:
        from config.improved_settings import Settings
        from utils.validators import ValidationError
        from utils.exceptions import ConfigurationError
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Please run: python quick_improvements.py first")
        return False
    
    # Validate configuration
    errors = Settings.validate_configuration()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("💡 Please fix these errors before starting trading")
        return False
    
    # Print configuration
    Settings.print_configuration()
    
    # Check directories
    try:
        Settings.ensure_directories()
        print("✅ Directories created/verified")
    except Exception as e:
        print(f"❌ Directory Error: {e}")
        return False
    
    # Check .env file
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("💡 Copy .env.example to .env and add your credentials")
        return False
    
    print("✅ Configuration check passed!")
    return True

def main():
    """Main function"""
    print("🎯 Trading Bot Configuration Checker")
    print("=" * 40)
    
    if check_configuration():
        print("\\n🚀 Ready to start trading!")
        print("💡 Run: python cli.py test")
    else:
        print("\\n❌ Configuration issues found")
        print("💡 Please fix the issues above before trading")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('check_config.py', 'w') as f:
        f.write(checker_content)
    
    print("✅ Created startup configuration checker")

def main():
    """Main function to implement quick improvements"""
    print("🚀 Implementing Quick Improvements")
    print("=" * 40)
    
    # Create improvements
    create_env_template()
    create_improved_settings()
    create_data_validators()
    create_custom_exceptions()
    create_startup_checker()
    
    print("\n✅ Quick improvements completed!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your Zerodha API credentials to .env")
    print("3. Run: python check_config.py")
    print("4. If all checks pass, you're ready for tomorrow!")
    
    print("\n💡 Files created:")
    print("   - .env.example (environment template)")
    print("   - config/improved_settings.py (better settings)")
    print("   - utils/validators.py (data validation)")
    print("   - utils/exceptions.py (custom exceptions)")
    print("   - check_config.py (startup checker)")

if __name__ == "__main__":
    main() 