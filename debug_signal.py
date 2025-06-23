#!/usr/bin/env python3
"""
Signal Diagnostic Script
========================

This script helps debug why signals aren't being detected properly.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth.kite_auth import KiteAuth
from trading.strategy import SuperTrendStrategy
from trading.executor import OrderExecutor
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

def diagnose_signals():
    """Diagnose signal detection issues"""
    print("🔍 SIGNAL DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Setup
    auth = KiteAuth()
    kite = auth.get_kite_instance()
    if not kite:
        print("❌ Failed to connect to Kite")
        return
    
    executor = OrderExecutor(kite)
    strategy = SuperTrendStrategy(
        atr_period=Settings.STRATEGY_PARAMS['atr_period'],
        factor=Settings.STRATEGY_PARAMS['factor']
    )
    
    # Get data
    to_date = datetime.now()
    from_date = to_date - timedelta(days=Settings.STRATEGY_PARAMS['historical_days'])
    
    print(f"📅 Fetching data from {from_date} to {to_date}")
    
    # NIFTY 50 index data
    signal_token = "256265"
    df = executor.get_historical_data(signal_token, from_date, to_date)
    
    if df.empty:
        print("❌ No data fetched")
        return
    
    print(f"✅ Fetched {len(df)} candles")
    print(f"📊 Data interval: {df.index[1] - df.index[0]}")
    
    # Calculate SuperTrend
    print("\n🔢 Calculating SuperTrend...")
    df_with_st = strategy.calculate_supertrend(df)
    
    # Find direction changes
    direction_changes = []
    for i in range(1, len(df_with_st)):
        prev_dir = df_with_st['direction'].iloc[i-1]
        curr_dir = df_with_st['direction'].iloc[i]
        
        if prev_dir != curr_dir:
            direction_changes.append({
                'date': df_with_st.index[i],
                'prev_direction': prev_dir,
                'curr_direction': curr_dir,
                'price': df_with_st['close'].iloc[i],
                'supertrend': df_with_st['supertrend'].iloc[i],
                'signal': 'BUY' if prev_dir == -1 and curr_dir == 1 else 'SELL'
            })
    
    print(f"\n📊 DIRECTION CHANGES FOUND: {len(direction_changes)}")
    print("-" * 50)
    
    # Show last 10 direction changes
    for change in direction_changes[-10:]:
        date_str = change['date'].strftime('%Y-%m-%d %H:%M')
        signal = change['signal']
        price = change['price']
        prev = "RED" if change['prev_direction'] == -1 else "GREEN"
        curr = "RED" if change['curr_direction'] == -1 else "GREEN"
        
        emoji = "🟢" if signal == "BUY" else "🔴"
        print(f"{emoji} {date_str}: {signal} signal | {prev} → {curr} | Price: ₹{price:.2f}")
    
    # Test signal generation on recent data
    print(f"\n🧪 TESTING SIGNAL GENERATION (Last 5 candles)")
    print("-" * 50)
    
    for i in range(max(0, len(df) - 5), len(df)):
        test_df = df.iloc[:i+1].copy()
        
        if len(test_df) >= 20:  # Need minimum data
            signal, signal_data = strategy.get_signal(test_df, has_position=False)
            
            date_str = test_df.index[-1].strftime('%Y-%m-%d %H:%M')
            price = test_df['close'].iloc[-1]
            direction = signal_data.get('direction', '?')
            prev_direction = signal_data.get('previous_direction', '?')
            trend = signal_data.get('trend', 'Unknown')
            
            print(f"{date_str} | Signal: {signal:4} | Price: ₹{price:.2f} | {trend}")
            print(f"   Direction: {prev_direction} → {direction}")
    
    # Check current market state
    print(f"\n📊 CURRENT MARKET STATE")
    print("-" * 50)
    
    signal, signal_data = strategy.get_signal(df, has_position=False)
    current_price = df['close'].iloc[-1]
    current_st = signal_data.get('supertrend', 0)
    current_direction = signal_data.get('direction', '?')
    current_trend = signal_data.get('trend', 'Unknown')
    
    print(f"Current Signal: {signal}")
    print(f"Current Trend: {current_trend}")
    print(f"Current Direction: {current_direction}")
    print(f"Current Price: ₹{current_price:.2f}")
    print(f"SuperTrend: ₹{current_st:.2f}")
    print(f"Price vs ST: {signal_data.get('price_vs_supertrend', 'Unknown')}")
    
    # Check with position
    print(f"\n🔄 SIGNAL WITH POSITION")
    signal_with_pos, signal_data_with_pos = strategy.get_signal(df, has_position=True)
    print(f"Signal (with position): {signal_with_pos}")
    
    # Recommendations
    print(f"\n💡 DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    if len(direction_changes) == 0:
        print("❌ No direction changes found!")
        print("   - Data might be insufficient")
        print("   - Market might be in strong trend")
        print("   - Try adjusting ATR period or factor")
    elif signal == "HOLD" and current_direction == 1:
        print("✅ Market is in UPTREND (GREEN)")
        print("   - Waiting for downtrend before next BUY signal")
        print("   - This is normal behavior")
    elif signal == "HOLD" and current_direction == -1:
        print("✅ Market is in DOWNTREND (RED)")
        print("   - BUY signal will trigger when trend turns GREEN")
        print("   - Keep monitoring")
    else:
        print(f"🔍 Current state: {signal}")
        print("   - Check if position tracking is correct")
        print("   - Verify signal execution logic")

def main():
    """Main function"""
    try:
        diagnose_signals()
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
