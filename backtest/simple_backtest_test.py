#!/usr/bin/env python3
"""
Simple standalone backtest script to test RELIANCE data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required modules
try:
    from trading.enhanced_strategy import EnhancedSuperTrendStrategy
    print("✅ Successfully imported EnhancedSuperTrendStrategy")
except ImportError as e:
    print(f"❌ Failed to import strategy: {e}")
    sys.exit(1)

def simple_backtest(csv_file):
    """Run a simple backtest"""
    print(f"\n📊 Simple Backtest Test")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        print(f"✅ Loaded {len(df)} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize strategy
    print("\nInitializing strategy...")
    strategy = EnhancedSuperTrendStrategy(atr_period=10, factor=3.0)
    
    # Simple backtest logic
    print("\nRunning simple backtest...")
    position = 0  # 0 = no position, 1 = long
    trades = []
    capital = 10000
    shares = 0
    entry_price = 0
    
    # Skip first 50 rows for warm-up
    for i in range(50, len(df)):
        current_date = df.index[i]
        current_price = df['close'].iloc[i]
        
        # Get historical data up to this point
        historical_data = df.iloc[:i+1]
        
        # Get signal
        signal, signal_data = strategy.get_signal(historical_data, has_position=(position == 1))
        
        # Process signals
        if signal == "BUY" and position == 0:
            # Enter position
            shares = int((capital * 5) / current_price)  # 5x leverage
            if shares > 0:
                entry_price = current_price
                position = 1
                print(f"📈 BUY: {shares} shares at ₹{current_price:.2f} on {current_date.strftime('%Y-%m-%d')}")
                
        elif signal == "SELL" and position == 1:
            # Exit position
            pnl = (current_price - entry_price) * shares - 40  # 40 rupees commission
            capital += pnl
            trades.append({
                'entry': entry_price,
                'exit': current_price,
                'shares': shares,
                'pnl': pnl
            })
            print(f"📉 SELL: {shares} shares at ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
            position = 0
            shares = 0
    
    # Summary
    print(f"\n📊 Backtest Summary:")
    print(f"Total trades: {len(trades)}")
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Final Capital: ₹{capital:.2f}")
        print(f"Return: {((capital - 10000) / 10000 * 100):.2f}%")
    else:
        print("No trades executed")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python simple_backtest_test.py <csv_file>")
        print("Example: python simple_backtest_test.py historical_data/RELIANCE_historical_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    simple_backtest(csv_file)

if __name__ == "__main__":
    main()
