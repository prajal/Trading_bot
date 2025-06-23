from typing import Any, Dict, Union
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
