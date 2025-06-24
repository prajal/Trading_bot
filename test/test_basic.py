#!/usr/bin/env python3
"""
Basic functionality tests for the trading bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from auth.kite_auth import KiteAuth
        print("✅ auth.kite_auth")
    except Exception as e:
        print(f"❌ auth.kite_auth: {e}")
    
    try:
        from trading.strategy import SuperTrendStrategy
        print("✅ trading.strategy")
    except Exception as e:
        print(f"❌ trading.strategy: {e}")
    
    try:
        from trading.executor import OrderExecutor
        print("✅ trading.executor")
    except Exception as e:
        print(f"❌ trading.executor: {e}")
    
    try:
        from config.settings import Settings
        print("✅ config.settings")
    except Exception as e:
        print(f"❌ config.settings: {e}")
    
    try:
        from utils.logger import get_logger
        print("✅ utils.logger")
    except Exception as e:
        print(f"❌ utils.logger: {e}")

def test_settings():
    """Test settings configuration"""
    print("\nTesting settings...")
    
    try:
        from config.settings import Settings
        
        # Check if settings load properly
        print(f"✅ Account Balance: ₹{Settings.STRATEGY_PARAMS['account_balance']:,.2f}")
        print(f"✅ ATR Period: {Settings.STRATEGY_PARAMS['atr_period']}")
        print(f"✅ Factor: {Settings.STRATEGY_PARAMS['factor']}")
        
        # Test dynamic amount update
        Settings.update_trading_amount(5000)
        print(f"✅ Updated Amount: ₹{Settings.STRATEGY_PARAMS['account_balance']:,.2f}")
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")

def test_strategy():
    """Test strategy basics"""
    print("\nTesting strategy...")
    
    try:
        from trading.strategy import SuperTrendStrategy
        import pandas as pd
        import numpy as np
        
        # Create simple test data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(50) * 0.5,
            'high': prices + abs(np.random.randn(50)),
            'low': prices - abs(np.random.randn(50)),
            'close': prices
        }, index=dates)
        
        # Test strategy
        strategy = SuperTrendStrategy()
        signal, signal_data = strategy.get_signal(df)
        
        print(f"✅ Strategy test passed")
        print(f"   Signal: {signal}")
        print(f"   Trend: {signal_data.get('trend', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Running Basic Tests")
    print("=" * 40)
    
    test_imports()
    test_settings()
    test_strategy()
    
    print("\n✅ Basic tests completed!")

if __name__ == "__main__":
    main()
