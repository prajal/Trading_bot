#!/usr/bin/env python3
"""
Debug SuperTrend Signal Generation
=================================

Let's debug why no signals are being generated.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_debug_data():
    """Create data specifically designed to generate SuperTrend signals"""
    
    # Create 100 days of data with CLEAR trend changes
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    data = []
    price = 300.0  # Start at 300
    
    for i, date in enumerate(dates):
        if i < 30:  # Strong uptrend (30 days)
            daily_change = np.random.normal(2.0, 1.0)  # Strong upward bias
        elif i < 60:  # Strong downtrend (30 days)  
            daily_change = np.random.normal(-2.0, 1.0)  # Strong downward bias
        else:  # Strong uptrend again (remaining days)
            daily_change = np.random.normal(2.5, 1.0)  # Very strong upward bias
        
        price = max(price + daily_change, 200.0)  # Don't go below 200
        
        # Generate realistic OHLC
        open_price = price + np.random.normal(0, 0.5)
        close = price + np.random.normal(0, 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, 1.0))
        low = min(open_price, close) - abs(np.random.normal(0, 1.0))
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2)
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    print(f"📊 Debug data created: {len(df)} days")
    print(f"📈 Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    print(f"🔄 Total price change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
    
    return df

def debug_supertrend_calculation():
    """Debug the SuperTrend calculation step by step"""
    
    print("🔍 DEBUGGING SUPERTREND CALCULATION")
    print("=" * 50)
    
    # Import your strategy
    try:
        from trading.strategy import SuperTrendStrategy
        print("✅ Successfully imported SuperTrendStrategy")
    except ImportError as e:
        print(f"❌ Failed to import SuperTrendStrategy: {e}")
        return False
    
    # Create test data
    df = create_debug_data()
    
    # Initialize strategy
    strategy = SuperTrendStrategy(atr_period=10, factor=3.0)
    print(f"✅ Strategy initialized: ATR={strategy.atr_period}, Factor={strategy.factor}")
    
    # Test SuperTrend calculation
    try:
        print(f"\n🔢 Calculating SuperTrend...")
        df_with_st = strategy.calculate_supertrend(df)
        print(f"✅ SuperTrend calculation completed")
        
        # Check for NaN values
        nan_count = df_with_st['supertrend'].isna().sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values in SuperTrend")
        else:
            print(f"✅ No NaN values found")
        
        # Check direction values
        unique_directions = df_with_st['direction'].unique()
        print(f"📊 Unique direction values: {unique_directions}")
        
        # Check for direction changes (signals)
        direction_changes = (df_with_st['direction'].diff() != 0).sum() - 1  # -1 for first NaN
        print(f"🔄 Direction changes found: {direction_changes}")
        
        # Show sample of the data
        print(f"\n📋 SAMPLE DATA (Last 10 rows):")
        sample_cols = ['close', 'atr', 'supertrend', 'direction']
        if 'trend_desc' in df_with_st.columns:
            sample_cols.append('trend_desc')
        
        sample_data = df_with_st[sample_cols].tail(10)
        for idx, row in sample_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            print(f"   {date_str} | Close: ₹{row['close']:6.2f} | ST: ₹{row['supertrend']:6.2f} | ATR: ₹{row['atr']:4.2f} | Dir: {row['direction']:2}")
        
        # Analyze direction changes in detail
        print(f"\n🔍 ANALYZING DIRECTION CHANGES:")
        prev_direction = None
        change_count = 0
        
        for i, (idx, row) in enumerate(df_with_st.iterrows()):
            if i == 0:
                prev_direction = row['direction']
                continue
                
            if row['direction'] != prev_direction:
                change_count += 1
                change_type = "RED→GREEN (BUY)" if row['direction'] == 1 else "GREEN→RED (SELL)"
                date_str = idx.strftime('%Y-%m-%d')
                print(f"   {change_count}. {date_str} | {change_type} | Price: ₹{row['close']:.2f} | ST: ₹{row['supertrend']:.2f}")
                prev_direction = row['direction']
        
        if change_count == 0:
            print("❌ No direction changes found!")
            print("💡 This means SuperTrend never changes from up to down or vice versa")
            
            # Check if all directions are the same
            all_directions = df_with_st['direction'].dropna()
            if len(all_directions.unique()) == 1:
                only_direction = all_directions.iloc[0]
                trend_name = "GREEN (Uptrend)" if only_direction == 1 else "RED (Downtrend)"
                print(f"🔍 All directions are: {only_direction} ({trend_name})")
                print(f"💡 This suggests the trend never changes in your data")
                
                # Check price vs SuperTrend relationship
                price_above_st = (df_with_st['close'] > df_with_st['supertrend']).sum()
                price_below_st = (df_with_st['close'] < df_with_st['supertrend']).sum()
                print(f"📊 Price above SuperTrend: {price_above_st} days")
                print(f"📊 Price below SuperTrend: {price_below_st} days")
        
        return df_with_st, change_count > 0
        
    except Exception as e:
        print(f"❌ Error in SuperTrend calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_signal_generation():
    """Test the signal generation method"""
    
    print(f"\n🎯 TESTING SIGNAL GENERATION")
    print("=" * 50)
    
    # Import strategy
    from trading.strategy import SuperTrendStrategy
    
    # Create test data
    df = create_debug_data()
    strategy = SuperTrendStrategy(atr_period=10, factor=3.0)
    
    # Test signal at different points
    test_points = [30, 40, 50, 60, 70, 80, 90]
    
    for point in test_points:
        if point < len(df):
            test_df = df.iloc[:point+1].copy()
            
            if len(test_df) >= 20:  # Need minimum data
                try:
                    signal, signal_data = strategy.get_signal(test_df)
                    
                    date_str = test_df.index[-1].strftime('%Y-%m-%d')
                    price = test_df['close'].iloc[-1]
                    trend = signal_data.get('trend', 'Unknown')
                    direction = signal_data.get('direction', 'Unknown')
                    
                    print(f"   Day {point:2} | {date_str} | Price: ₹{price:6.2f} | Signal: {signal:4} | {trend} | Dir: {direction}")
                    
                except Exception as e:
                    print(f"   Day {point:2} | Error: {e}")

def test_with_simple_data():
    """Test with very simple trending data"""
    
    print(f"\n🧪 TESTING WITH SIMPLE TRENDING DATA")
    print("=" * 50)
    
    # Create super simple data: just go up, then down, then up
    simple_data = []
    dates = pd.date_range('2023-01-01', periods=60, freq='D')
    
    base_price = 100
    for i, date in enumerate(dates):
        if i < 20:
            price = base_price + i * 2  # Go up steadily
        elif i < 40:
            price = base_price + 40 - (i - 20) * 2  # Go down steadily
        else:
            price = base_price + (i - 40) * 2  # Go up again
        
        simple_data.append({
            'date': date,
            'open': price,
            'high': price + 1,
            'low': price - 1, 
            'close': price
        })
    
    simple_df = pd.DataFrame(simple_data)
    simple_df.set_index('date', inplace=True)
    
    print(f"📊 Simple data: {len(simple_df)} days, ₹{simple_df['close'].min():.0f} to ₹{simple_df['close'].max():.0f}")
    
    # Test with this simple data
    from trading.strategy import SuperTrendStrategy
    strategy = SuperTrendStrategy(atr_period=5, factor=2.0)  # More sensitive
    
    try:
        df_with_st = strategy.calculate_supertrend(simple_df)
        
        # Count direction changes
        direction_changes = (df_with_st['direction'].diff() != 0).sum() - 1
        print(f"🔄 Direction changes in simple data: {direction_changes}")
        
        if direction_changes > 0:
            print("✅ Simple data generates signals!")
            
            # Show the changes
            prev_direction = None
            for i, (idx, row) in enumerate(df_with_st.iterrows()):
                if i == 0:
                    prev_direction = row['direction']
                    continue
                    
                if row['direction'] != prev_direction:
                    change_type = "RED→GREEN (BUY)" if row['direction'] == 1 else "GREEN→RED (SELL)"
                    date_str = idx.strftime('%Y-%m-%d')
                    print(f"   {date_str} | {change_type} | Price: ₹{row['close']:.0f}")
                    prev_direction = row['direction']
        else:
            print("❌ Even simple data doesn't generate signals!")
            
    except Exception as e:
        print(f"❌ Error with simple data: {e}")

def main():
    """Main debug function"""
    
    print("🔧 SUPERTREND DEBUG SESSION")
    print("=" * 50)
    
    # Test 1: SuperTrend calculation
    df_result, has_signals = debug_supertrend_calculation()
    
    if not has_signals:
        # Test 2: Signal generation method
        test_signal_generation()
        
        # Test 3: Very simple data
        test_with_simple_data()
    
    print(f"\n🎯 DIAGNOSIS:")
    if has_signals:
        print("✅ SuperTrend calculation is working and generating signals!")
        print("💡 The issue might be in the backtesting framework.")
    else:
        print("❌ SuperTrend is not generating direction changes.")
        print("💡 Possible issues:")
        print("   1. Factor too high (try 1.5-2.0)")
        print("   2. ATR period too long (try 5-7)")
        print("   3. Data doesn't have enough volatility")
        print("   4. SuperTrend calculation logic issue")

if __name__ == "__main__":
    main()
