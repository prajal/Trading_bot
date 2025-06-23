class TradingBotError(Exception):
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
