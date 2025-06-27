import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json

# Load environment variables from .env file
load_dotenv()

class ConfigurationError(Exception):
    """Configuration related errors"""
    pass

@dataclass
class TradingConfig:
    """Trading configuration with validation"""
    account_balance: float
    capital_allocation_percent: float = 100.0
    max_positions: int = 1
    fixed_stop_loss: float = 100.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.15  # 15% max drawdown
    position_sizing_method: str = "fixed"  # "fixed", "risk_based", "volatility_adjusted"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.account_balance <= 0:
            raise ConfigurationError("Account balance must be positive")
        if not 0 < self.capital_allocation_percent <= 100:
            raise ConfigurationError("Capital allocation must be between 0 and 100")
        if self.max_positions <= 0:
            raise ConfigurationError("Max positions must be positive")
        if self.fixed_stop_loss <= 0:
            raise ConfigurationError("Stop loss must be positive")
        if not 0 < self.risk_per_trade <= 0.1:
            raise ConfigurationError("Risk per trade must be between 0 and 10%")

@dataclass
class StrategyConfig:
    """Strategy configuration with validation"""
    atr_period: int = 10
    factor: float = 3.0
    min_candles_required: int = 50
    adaptive_mode: bool = True
    multi_timeframe_analysis: bool = False
    volume_confirmation: bool = False
    momentum_filter: bool = False
    confidence_threshold: float = 0.6  # Minimum confidence for trades
    
    def __post_init__(self):
        """Validate strategy configuration"""
        if self.atr_period < 1:
            raise ConfigurationError("ATR period must be positive")
        if self.factor <= 0:
            raise ConfigurationError("Factor must be positive")
        if self.min_candles_required < 10:
            raise ConfigurationError("Minimum candles must be at least 10")
        if not 0 < self.confidence_threshold <= 1:
            raise ConfigurationError("Confidence threshold must be between 0 and 1")

@dataclass
class RiskConfig:
    """Risk management configuration"""
    stop_loss_method: str = "fixed"  # "fixed", "atr_based", "percentage"
    trailing_stop: bool = False
    trailing_stop_distance: float = 2.0  # ATR multiplier for trailing stop
    max_correlation_exposure: float = 0.5  # Max exposure to correlated positions
    volatility_adjustment: bool = True
    emergency_stop_enabled: bool = True
    
    def __post_init__(self):
        if self.stop_loss_method not in ["fixed", "atr_based", "percentage"]:
            raise ConfigurationError("Invalid stop loss method")

@dataclass
class SafetyConfig:
    """Safety configuration"""
    live_trading_enabled: bool = False
    dry_run_mode: bool = True
    paper_trading_balance: float = 10000.0
    enable_console_alerts: bool = True
    enable_trade_logging: bool = True
    max_orders_per_day: int = 50
    circuit_breaker_loss: float = 0.1  # 10% circuit breaker
    
    def __post_init__(self):
        if self.paper_trading_balance <= 0:
            raise ConfigurationError("Paper trading balance must be positive")

class Settings:
    """Enhanced centralized settings with validation and security"""
    
    # API Configuration with validation
    _KITE_API_KEY = os.getenv('KITE_API_KEY')
    _KITE_API_SECRET = os.getenv('KITE_API_SECRET')
    KITE_REDIRECT_URI = os.getenv('KITE_REDIRECT_URI', 'http://localhost:3000')
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    CONFIG_DIR = BASE_DIR / 'config'
    BACKUP_DIR = BASE_DIR / 'backups'
    
    TOKEN_FILE = DATA_DIR / 'kite_tokens.json'
    TRADING_CONFIG_FILE = DATA_DIR / 'trading_config.json'
    TRADING_PREFS_FILE = DATA_DIR / 'trading_preferences.json'
    PERFORMANCE_FILE = DATA_DIR / 'performance_metrics.json'
    LOG_FILE = LOGS_DIR / 'trading.log'
    ERROR_LOG_FILE = LOGS_DIR / 'errors.log'
    
    # Configuration instances
    _trading_config: Optional[TradingConfig] = None
    _strategy_config: Optional[StrategyConfig] = None
    _risk_config: Optional[RiskConfig] = None
    _safety_config: Optional[SafetyConfig] = None
    
    @classmethod
    def validate_credentials(cls) -> bool:
        """Validate API credentials are present"""
        if not cls._KITE_API_KEY:
            raise ConfigurationError("KITE_API_KEY environment variable not set")
        if not cls._KITE_API_SECRET:
            raise ConfigurationError("KITE_API_SECRET environment variable not set")
        if len(cls._KITE_API_KEY) < 10:
            raise ConfigurationError("KITE_API_KEY appears to be invalid")
        return True
    
    @classmethod
    def get_kite_api_key(cls) -> str:
        """Get API key with validation"""
        cls.validate_credentials()
        return cls._KITE_API_KEY
    
    @classmethod
    def get_kite_api_secret(cls) -> str:
        """Get API secret with validation"""
        cls.validate_credentials()
        return cls._KITE_API_SECRET
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.CONFIG_DIR, cls.BACKUP_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def get_trading_config(cls) -> TradingConfig:
        """Get trading configuration with validation"""
        if cls._trading_config is None:
            account_balance = float(os.getenv('ACCOUNT_BALANCE', '10000.0'))
            
            # Load from preferences file if exists
            if cls.TRADING_PREFS_FILE.exists():
                try:
                    with open(cls.TRADING_PREFS_FILE, 'r') as f:
                        prefs = json.load(f)
                        account_balance = prefs.get('last_trading_amount', account_balance)
                except Exception:
                    pass  # Use default if file is corrupted
            
            cls._trading_config = TradingConfig(
                account_balance=account_balance,
                capital_allocation_percent=float(os.getenv('CAPITAL_ALLOCATION_PERCENT', '100.0')),
                max_positions=int(os.getenv('MAX_POSITIONS', '1')),
                fixed_stop_loss=float(os.getenv('FIXED_STOP_LOSS', '100.0')),
                risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
                max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
                max_drawdown=float(os.getenv('MAX_DRAWDOWN', '0.15')),
                position_sizing_method=os.getenv('POSITION_SIZING_METHOD', 'fixed')
            )
        
        return cls._trading_config
    
    @classmethod
    def get_strategy_config(cls) -> StrategyConfig:
        """Get strategy configuration with validation"""
        if cls._strategy_config is None:
            cls._strategy_config = StrategyConfig(
                atr_period=int(os.getenv('ATR_PERIOD', '10')),
                factor=float(os.getenv('FACTOR', '3.0')),
                min_candles_required=int(os.getenv('MIN_CANDLES', '50')),
                adaptive_mode=os.getenv('ADAPTIVE_MODE', 'true').lower() == 'true',
                multi_timeframe_analysis=os.getenv('MULTI_TIMEFRAME', 'false').lower() == 'true',
                volume_confirmation=os.getenv('VOLUME_CONFIRMATION', 'false').lower() == 'true',
                momentum_filter=os.getenv('MOMENTUM_FILTER', 'false').lower() == 'true',
                confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
            )
        
        return cls._strategy_config
    
    @classmethod
    def get_risk_config(cls) -> RiskConfig:
        """Get risk management configuration"""
        if cls._risk_config is None:
            cls._risk_config = RiskConfig(
                stop_loss_method=os.getenv('STOP_LOSS_METHOD', 'fixed'),
                trailing_stop=os.getenv('TRAILING_STOP', 'false').lower() == 'true',
                trailing_stop_distance=float(os.getenv('TRAILING_STOP_DISTANCE', '2.0')),
                max_correlation_exposure=float(os.getenv('MAX_CORRELATION_EXPOSURE', '0.5')),
                volatility_adjustment=os.getenv('VOLATILITY_ADJUSTMENT', 'true').lower() == 'true',
                emergency_stop_enabled=os.getenv('EMERGENCY_STOP', 'true').lower() == 'true'
            )
        
        return cls._risk_config
    
    @classmethod
    def get_safety_config(cls) -> SafetyConfig:
        """Get safety configuration"""
        if cls._safety_config is None:
            cls._safety_config = SafetyConfig(
                live_trading_enabled=os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true',
                dry_run_mode=os.getenv('DRY_RUN_MODE', 'true').lower() == 'true',
                paper_trading_balance=float(os.getenv('PAPER_TRADING_BALANCE', '10000.0')),
                enable_console_alerts=os.getenv('ENABLE_CONSOLE_ALERTS', 'true').lower() == 'true',
                enable_trade_logging=os.getenv('ENABLE_TRADE_LOGGING', 'true').lower() == 'true',
                max_orders_per_day=int(os.getenv('MAX_ORDERS_PER_DAY', '50')),
                circuit_breaker_loss=float(os.getenv('CIRCUIT_BREAKER_LOSS', '0.1'))
            )
        
        return cls._safety_config
    
    @classmethod
    def get_market_hours(cls) -> Dict[str, int]:
        """Get market hours configuration"""
        return {
            'open_hour': int(os.getenv('MARKET_OPEN_HOUR', '9')),
            'open_minute': int(os.getenv('MARKET_OPEN_MINUTE', '15')),
            'close_hour': int(os.getenv('MARKET_CLOSE_HOUR', '15')),
            'close_minute': int(os.getenv('MARKET_CLOSE_MINUTE', '30')),
            'pre_close_exit_minutes': int(os.getenv('PRE_CLOSE_EXIT_MINUTES', '10'))
        }
    
    @classmethod
    def update_trading_amount(cls, amount: float):
        """Update trading amount dynamically with validation"""
        if amount <= 0:
            raise ConfigurationError("Trading amount must be positive")
        
        # Update the config
        if cls._trading_config:
            cls._trading_config.account_balance = amount
        
        # Save to preferences
        prefs = {}
        if cls.TRADING_PREFS_FILE.exists():
            try:
                with open(cls.TRADING_PREFS_FILE, 'r') as f:
                    prefs = json.load(f)
            except Exception:
                pass
        
        prefs['last_trading_amount'] = amount
        prefs['last_update'] = datetime.now().isoformat()
        
        cls.ensure_directories()
        with open(cls.TRADING_PREFS_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)
    
    @classmethod
    def validate_all_configuration(cls) -> Dict[str, Any]:
        """Validate all configuration settings"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate credentials
            cls.validate_credentials()
            
            # Validate all configs
            trading_config = cls.get_trading_config()
            strategy_config = cls.get_strategy_config()
            risk_config = cls.get_risk_config()
            safety_config = cls.get_safety_config()
            
            # Check for potential issues
            if trading_config.account_balance < 5000:
                validation_results['warnings'].append("Account balance is quite low for live trading")
            
            if safety_config.live_trading_enabled and safety_config.dry_run_mode:
                validation_results['errors'].append("Cannot have both live trading and dry run enabled")
                validation_results['valid'] = False
            
            if strategy_config.confidence_threshold < 0.5:
                validation_results['warnings'].append("Low confidence threshold may result in poor quality signals")
            
        except ConfigurationError as e:
            validation_results['errors'].append(str(e))
            validation_results['valid'] = False
        except Exception as e:
            validation_results['errors'].append(f"Unexpected validation error: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    @classmethod
    def print_configuration_summary(cls):
        """Print current configuration summary"""
        print("📊 Current Configuration Summary:")
        print("=" * 50)
        
        try:
            trading_config = cls.get_trading_config()
            strategy_config = cls.get_strategy_config()
            risk_config = cls.get_risk_config()
            safety_config = cls.get_safety_config()
            market_hours = cls.get_market_hours()
            
            print(f"\n💰 Trading Configuration:")
            print(f"   Account Balance: ₹{trading_config.account_balance:,.2f}")
            print(f"   Capital Allocation: {trading_config.capital_allocation_percent}%")
            print(f"   Max Positions: {trading_config.max_positions}")
            print(f"   Risk per Trade: {trading_config.risk_per_trade:.1%}")
            print(f"   Max Daily Loss: {trading_config.max_daily_loss:.1%}")
            print(f"   Position Sizing: {trading_config.position_sizing_method}")
            
            print(f"\n📈 Strategy Configuration:")
            print(f"   ATR Period: {strategy_config.atr_period}")
            print(f"   Factor: {strategy_config.factor}")
            print(f"   Adaptive Mode: {strategy_config.adaptive_mode}")
            print(f"   Confidence Threshold: {strategy_config.confidence_threshold:.1%}")
            print(f"   Multi-Timeframe: {strategy_config.multi_timeframe_analysis}")
            
            print(f"\n🛡️  Risk Management:")
            print(f"   Stop Loss Method: {risk_config.stop_loss_method}")
            print(f"   Trailing Stop: {risk_config.trailing_stop}")
            print(f"   Emergency Stop: {risk_config.emergency_stop_enabled}")
            print(f"   Max Drawdown: {trading_config.max_drawdown:.1%}")
            
            print(f"\n🔒 Safety Configuration:")
            print(f"   Live Trading: {safety_config.live_trading_enabled}")
            print(f"   Dry Run Mode: {safety_config.dry_run_mode}")
            print(f"   Circuit Breaker: {safety_config.circuit_breaker_loss:.1%}")
            print(f"   Max Orders/Day: {safety_config.max_orders_per_day}")
            
            print(f"\n🕒 Market Hours:")
            print(f"   Open: {market_hours['open_hour']:02d}:{market_hours['open_minute']:02d}")
            print(f"   Close: {market_hours['close_hour']:02d}:{market_hours['close_minute']:02d}")
            
        except Exception as e:
            print(f"❌ Error displaying configuration: {e}")
    
    @classmethod
    def create_backup_config(cls):
        """Create backup of current configuration"""
        cls.ensure_directories()
        
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'trading_config': cls.get_trading_config().__dict__,
            'strategy_config': cls.get_strategy_config().__dict__,
            'risk_config': cls.get_risk_config().__dict__,
            'safety_config': cls.get_safety_config().__dict__,
            'market_hours': cls.get_market_hours()
        }
        
        backup_file = cls.BACKUP_DIR / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return backup_file

# Initialize and validate configuration on import
try:
    Settings.ensure_directories()
    validation_results = Settings.validate_all_configuration()
    
    if not validation_results['valid']:
        print("⚠️  Configuration Validation Failed:")
        for error in validation_results['errors']:
            print(f"   ❌ {error}")
    
    if validation_results['warnings']:
        print("⚠️  Configuration Warnings:")
        for warning in validation_results['warnings']:
            print(f"   ⚠️  {warning}")

except Exception as e:
    print(f"❌ Failed to initialize settings: {e}")
    print("💡 Please check your .env file and environment variables")