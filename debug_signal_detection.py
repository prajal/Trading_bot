#!/usr/bin/env python3
"""
Debug Signal Detection - Find Why BUY Signals Aren't Being Processed
==================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_extreme_test_data():
    """Create data that will DEFINITELY generate both BUY and SELL signals"""
    
    dates = pd.date_range('2023-01-01', periods=40, freq='D')
    data = []
    
    # Phase 1: Strong uptrend (20 days) - 100 to 140
    for i in range(20):
        price = 100 + (i * 2)
        data.append({
            'date': dates[i],
            'open': price - 0.5,
            'high': price + 1,
            'low': price - 1,
            'close': price
        })
    
    # Phase 2: Strong downtrend (20 days) - 140 to 100
    for i in range(20):
        price = 140 - (i * 2)
        data.append({
            'date': dates[20 + i],
            'open': price + 0.5,
            'high': price + 1,
            'low': price - 1,
            'close': price
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    print(f"📊 Extreme test data: {len(df)} days")
    print(f"📈 Price movement: {df['close'].iloc[0]:.0f} → {df['close'].iloc[19]:.0f} → {df['close'].iloc[-1]:.0f}")
    
    return df

def debug_all_signals():
    """Debug every single signal generated"""
    
    print("🔍 DEBUGGING ALL SIGNALS")
    print("=" * 50)
    
    from trading.strategy import SuperTrendStrategy
    
    # Create test data
    df = create_extreme_test_data()
    
    # Use very sensitive parameters
    strategy = SuperTrendStrategy(atr_period=3, factor=1.0)
    print(f"⚙️  Using VERY sensitive parameters: ATR=3, Factor=1.0")
    
    # Check signals at every single day
    all_signals = []
    
    for i in range(10, len(df)):  # Start from day 10
        current_df = df.iloc[:i+1].copy()
        current_price = current_df['close'].iloc[-1]
        current_date = current_df.index[-1]
        
        try:
            signal, signal_data = strategy.get_signal(current_df, has_position=False)
            
            # Record all signals
            signal_info = {
                'day': i,
                'date': current_date,
                'price': current_price,
                'signal': signal,
                'trend': signal_data.get('trend', 'Unknown'),
                'direction': signal_data.get('direction', 'Unknown'),
                'prev_direction': signal_data.get('previous_direction', 'Unknown')
            }
            all_signals.append(signal_info)
            
            # Print significant signals
            if signal in ['BUY', 'SELL']:
                date_str = current_date.strftime('%m-%d')
                print(f"🎯 {signal:4} signal on {date_str} | Price: ₹{current_price:3.0f} | "
                      f"{signal_data.get('trend', 'Unknown'):15} | "
                      f"Dir: {signal_data.get('previous_direction', '?')} → {signal_data.get('direction', '?')}")
        
        except Exception as e:
            continue
    
    # Analyze all signals
    buy_signals = [s for s in all_signals if s['signal'] == 'BUY']
    sell_signals = [s for s in all_signals if s['signal'] == 'SELL']
    
    print(f"\n📊 SIGNAL ANALYSIS:")
    print(f"   Total BUY signals:  {len(buy_signals)}")
    print(f"   Total SELL signals: {len(sell_signals)}")
    
    if len(buy_signals) == 0:
        print(f"\n❌ NO BUY SIGNALS DETECTED!")
        print(f"💡 This is the root cause of the backtest issue")
        
        # Check direction changes manually
        print(f"\n🔍 Checking direction changes manually...")
        for i in range(1, len(all_signals)):
            current = all_signals[i]
            previous = all_signals[i-1]
            
            if (previous['direction'] == -1 and current['direction'] == 1):
                date_str = current['date'].strftime('%m-%d')
                print(f"   {date_str}: Direction change -1 → 1 (should be BUY) | Price: ₹{current['price']:.0f} | Signal: {current['signal']}")
            elif (previous['direction'] == 1 and current['direction'] == -1):
                date_str = current['date'].strftime('%m-%d')
                print(f"   {date_str}: Direction change 1 → -1 (should be SELL) | Price: ₹{current['price']:.0f} | Signal: {current['signal']}")
    
    return buy_signals, sell_signals

def test_individual_signal_generation():
    """Test signal generation at specific points where we know signals should occur"""
    
    print(f"\n🎯 TESTING INDIVIDUAL SIGNAL GENERATION")
    print("=" * 50)
    
    from trading.strategy import SuperTrendStrategy
    
    # Create data with guaranteed signal at specific point
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    data = []
    
    # Create data that goes: down, down, down, then UP, UP, UP
    for i in range(30):
        if i < 15:
            price = 200 - (i * 2)  # Go down from 200 to 170
        else:
            price = 170 + ((i - 15) * 3)  # Go up from 170 to 215
        
        data.append({
            'date': dates[i],
            'open': price - 0.5,
            'high': price + 1,
            'low': price - 1,
            'close': price
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    print(f"📊 Controlled test data: {len(df)} days")
    print(f"📉 Days 1-15: {df['close'].iloc[0]:.0f} → {df['close'].iloc[14]:.0f} (downtrend)")
    print(f"📈 Days 16-30: {df['close'].iloc[15]:.0f} → {df['close'].iloc[-1]:.0f} (uptrend)")
    
    strategy = SuperTrendStrategy(atr_period=5, factor=2.0)
    
    # Test at the transition point (around day 16-18)
    for test_day in [16, 17, 18, 19, 20]:
        test_df = df.iloc[:test_day+1].copy()
        
        try:
            signal, signal_data = strategy.get_signal(test_df, has_position=False)
            
            date_str = test_df.index[-1].strftime('%m-%d')
            price = test_df['close'].iloc[-1]
            trend = signal_data.get('trend', 'Unknown')
            direction = signal_data.get('direction', 'Unknown')
            prev_direction = signal_data.get('previous_direction', 'Unknown')
            
            print(f"   Day {test_day:2} ({date_str}): {signal:4} | Price: ₹{price:3.0f} | "
                  f"{trend:15} | Dir: {prev_direction} → {direction}")
            
            if signal == 'BUY':
                print(f"   ✅ BUY SIGNAL FOUND at day {test_day}!")
                return True
        
        except Exception as e:
            print(f"   Day {test_day:2}: Error - {e}")
    
    print(f"   ❌ No BUY signals found even at obvious transition points")
    return False

def main():
    """Main debugging function"""
    
    print("🔧 SIGNAL DETECTION DEBUG SESSION")
    print("=" * 50)
    
    # Test 1: Debug all signals
    buy_signals, sell_signals = debug_all_signals()
    
    # Test 2: Test at specific transition points
    found_buy = test_individual_signal_generation()
    
    print(f"\n🎯 DEBUG CONCLUSION:")
    print("=" * 50)
    
    if len(buy_signals) > 0 or found_buy:
        print("✅ BUY signals ARE being generated!")
        print("💡 The issue must be in the backtest execution loop")
    else:
        print("❌ BUY signals are NOT being generated!")
        print("💡 There's a bug in the signal generation logic")
        print("🔍 Possible issues:")
        print("   1. Direction change detection logic is wrong")
        print("   2. 'has_position' parameter affecting signal generation")
        print("   3. SuperTrend calculation not detecting uptrend properly")

if __name__ == "__main__":
    main()
