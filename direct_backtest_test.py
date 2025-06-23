#!/usr/bin/env python3
"""
Direct Backtest Test - Use Fixed SuperTrend Logic Directly
=========================================================

Test backtesting by directly using the corrected strategy from trading/strategy.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data_with_clear_trends():
    """Create data with VERY clear trends that should generate signals"""
    
    # Create 60 days with dramatic price changes
    dates = pd.date_range('2023-01-01', periods=60, freq='D')
    
    data = []
    base_price = 100
    
    for i, date in enumerate(dates):
        if i < 20:
            # Strong uptrend - price goes up 5 every day
            price = base_price + (i * 5)
        elif i < 40:
            # Strong downtrend - price goes down 5 every day
            price = base_price + 100 - ((i - 20) * 5)
        else:
            # Strong uptrend again - price goes up 5 every day
            price = base_price + ((i - 40) * 5)
        
        # Create simple OHLC around the price
        data.append({
            'date': date,
            'open': price - 1,
            'high': price + 2,
            'low': price - 2,
            'close': price
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    print(f"📊 Test data created: {len(df)} days")
    print(f"📈 Price movement: {df['close'].iloc[0]:.0f} → {df['close'].iloc[19]:.0f} → {df['close'].iloc[39]:.0f} → {df['close'].iloc[-1]:.0f}")
    print(f"💫 This should DEFINITELY generate multiple signals!")
    
    return df

def simple_backtest_with_corrected_strategy():
    """Run simple backtest using the corrected strategy directly"""
    
    print("🧪 DIRECT BACKTEST TEST WITH CORRECTED STRATEGY")
    print("=" * 60)
    
    # Import your corrected strategy
    try:
        from trading.strategy import SuperTrendStrategy
        print("✅ Successfully imported corrected SuperTrendStrategy")
    except ImportError as e:
        print(f"❌ Failed to import SuperTrendStrategy: {e}")
        return False
    
    # Create clear trending data
    df = create_test_data_with_clear_trends()
    
    # Initialize strategy with sensitive parameters
    strategy = SuperTrendStrategy(atr_period=5, factor=2.0)
    print(f"⚙️  Using sensitive parameters: ATR=5, Factor=2.0")
    
    # Simple backtest variables
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    trades = []
    capital = 10000
    shares = 0
    
    print(f"\n🚀 Running simple backtest...")
    
    # Iterate through each day
    for i in range(10, len(df)):  # Start from day 10 to have enough ATR data
        
        # Get data up to current day
        current_df = df.iloc[:i+1].copy()
        current_price = current_df['close'].iloc[-1]
        current_date = current_df.index[-1]
        
        # Get signal from strategy
        signal, signal_data = strategy.get_signal(current_df, has_position=(position > 0))
        
        # Execute trades based on signals
        if signal == "BUY" and position == 0:
            # Calculate position size
            shares = int(capital / current_price)
            if shares > 0:
                entry_price = current_price
                position = 1
                capital -= shares * current_price
                
                date_str = current_date.strftime('%Y-%m-%d')
                trend = signal_data.get('trend', 'Unknown')
                print(f"📈 BUY:  {shares} shares at ₹{current_price:.0f} on {date_str} | {trend}")
        
        elif signal == "SELL" and position == 1:
            # Close position
            pnl = (current_price - entry_price) * shares
            capital += shares * current_price
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': current_price,
                'shares': shares,
                'pnl': pnl
            })
            
            date_str = current_date.strftime('%Y-%m-%d')
            trend = signal_data.get('trend', 'Unknown')
            print(f"📉 SELL: {shares} shares at ₹{current_price:.0f} on {date_str} | {trend} | P&L: ₹{pnl:.0f}")
            
            position = 0
            shares = 0
    
    # Results
    print(f"\n📊 BACKTEST RESULTS:")
    print(f"   Total trades: {len(trades)}")
    print(f"   Final capital: ₹{capital:,.0f}")
    
    if len(trades) > 0:
        total_pnl = sum(trade['pnl'] for trade in trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / len(trades) * 100
        
        print(f"   Total P&L: ₹{total_pnl:.0f}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Return: {((capital + (shares * current_price if position > 0 else 0) - 10000) / 10000 * 100):.1f}%")
        
        print(f"\n✅ SUCCESS: Strategy generated {len(trades)} trades!")
        return True
    else:
        print(f"\n❌ FAILED: No trades generated even with clear trending data")
        
        # Debug: Check what signals were generated
        print(f"\n🔍 DEBUG: Checking signals throughout the period...")
        for i in range(10, min(len(df), 30)):  # Check first 20 test points
            test_df = df.iloc[:i+1].copy()
            signal, signal_data = strategy.get_signal(test_df)
            
            if signal in ["BUY", "SELL"]:
                date_str = test_df.index[-1].strftime('%Y-%m-%d')
                price = test_df['close'].iloc[-1]
                trend = signal_data.get('trend', 'Unknown')
                direction = signal_data.get('direction', 'Unknown')
                
                print(f"   {date_str}: {signal} signal | Price: ₹{price:.0f} | {trend} | Dir: {direction}")
        
        return False

def test_strategy_validation():
    """Test the strategy validation function"""
    
    print(f"\n🔍 TESTING STRATEGY VALIDATION")
    print("=" * 50)
    
    from trading.strategy import SuperTrendStrategy
    
    # Create test data
    df = create_test_data_with_clear_trends()
    strategy = SuperTrendStrategy(atr_period=5, factor=2.0)
    
    # Run validation
    is_valid = strategy.validate_signal(df)
    
    if is_valid:
        print("✅ Strategy validation passed!")
    else:
        print("❌ Strategy validation failed!")
    
    return is_valid

def main():
    """Main test function"""
    
    print("🎯 DIRECT BACKTEST TEST")
    print("=" * 50)
    
    # Test 1: Strategy validation
    validation_passed = test_strategy_validation()
    
    # Test 2: Simple backtest
    backtest_passed = simple_backtest_with_corrected_strategy()
    
    print(f"\n🎯 FINAL DIAGNOSIS:")
    print("=" * 50)
    
    if validation_passed and backtest_passed:
        print("✅ Your corrected strategy is working perfectly!")
        print("💡 The issue is that quick_backtest.py is still using the old backtest_strategy.py")
        print("🔧 Solution: Replace backtest_strategy.py with the fixed version")
    elif validation_passed and not backtest_passed:
        print("⚠️  Strategy validates but doesn't generate trades")
        print("💡 May need more sensitive parameters or different test data")
    else:
        print("❌ There's still an issue with the strategy implementation")
        print("💡 Need to debug the SuperTrend calculation further")

if __name__ == "__main__":
    main()
