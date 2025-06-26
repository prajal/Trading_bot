#!/usr/bin/env python3
"""
Script to analyze and print OHLC and SuperTrend data
Useful for debugging and understanding market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from auth.kite_auth import KiteAuth
from trading.strategy import SuperTrendStrategy
from tabulate import tabulate
import argparse
import sys
import os
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Trading instruments configuration
TRADING_INSTRUMENTS = {
    'NIFTYBEES': {'name': 'Nippon India ETF Nifty 50 BeES', 'token': '2707457'},
    'BANKBEES': {'name': 'Nippon India ETF Nifty Bank BeES', 'token': '2954241'},
    'JUNIORBEES': {'name': 'Nippon India ETF Junior BeES', 'token': '4632577'},
    'GOLDBEES': {'name': 'Nippon India ETF Gold BeES', 'token': '2800641'},
    'RELIANCE': {'name': 'Reliance Industries Ltd', 'token': '738561'},
    'TCS': {'name': 'Tata Consultancy Services Ltd', 'token': '2953217'},
    'HDFCBANK': {'name': 'HDFC Bank Ltd', 'token': '341249'},
    'ICICIBANK': {'name': 'ICICI Bank Ltd', 'token': '1270529'},
    'INFY': {'name': 'Infosys Ltd', 'token': '408065'}
}

class SuperTrendAnalyzer:
    """Analyze OHLC and SuperTrend data"""
    
    def __init__(self, atr_period=10, factor=3.0):
        self.auth = KiteAuth()
        self.strategy = SuperTrendStrategy(atr_period=atr_period, factor=factor)
        self.kite = None
    
    def setup(self):
        """Setup Kite connection"""
        self.kite = self.auth.get_kite_instance()
        if not self.kite:
            print("❌ Failed to connect to Kite")
            return False
        print("✅ Connected to Kite")
        return True
    
    def get_historical_data(self, instrument_token, days=1, interval="minute"):
        """Fetch historical data"""
        try:
            if not self.kite:
                print("❌ Kite connection not available")
                return pd.DataFrame()
                
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                instrument_token, 
                from_date, 
                to_date, 
                interval
            )
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def print_latest_candle(self, symbol="NIFTYBEES"):
        """Print the last 3 candles and SuperTrend data in concise format, showing ENTRY↑/EXIT↓/HOLD signals."""
        if symbol not in TRADING_INSTRUMENTS:
            print(f"❌ Unknown instrument: {symbol}")
            return
        instrument = TRADING_INSTRUMENTS[symbol]
        df = self.get_historical_data(instrument['token'], days=1)
        if df.empty:
            print("❌ No data available")
            return
        try:
            df_with_st = self.strategy.calculate_supertrend(df)
        except Exception as e:
            print(f"❌ Error calculating SuperTrend: {e}")
            return
        n = 3  # Number of bars to print
        start_idx = max(1, len(df_with_st) - n)
        for i in range(start_idx, len(df_with_st)):
            latest = df_with_st.iloc[i]
            latest_time = pd.to_datetime(df_with_st.index[i]).strftime('%H:%M:%S')
            o = latest['open']
            h = latest['high']
            l = latest['low']
            c = latest['close']
            st = latest['supertrend']
            st_dir = 'UP' if latest['direction'] == 1 else 'DOWN'
            delta = c - st
            delta_str = f"{delta:+.2f}"
            # Detect signal
            prev_dir = df_with_st['direction'].iloc[i-1] if i > 0 else latest['direction']
            curr_dir = latest['direction']
            if prev_dir == -1 and curr_dir == 1:
                signal = 'ENTRY↑'
            elif prev_dir == 1 and curr_dir == -1:
                signal = 'EXIT↓'
            else:
                signal = 'HOLD'
            print(f"{latest_time} | O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} | ST:{st:.2f} ({st_dir}) | Δ:{delta_str} | {signal}")

def main():
    """Main function with continuous monitoring"""
    parser = argparse.ArgumentParser(description="Live SuperTrend Analysis")
    parser.add_argument("--symbol", default="NIFTYBEES", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds (default: 60)")
    args = parser.parse_args()
    
    analyzer = SuperTrendAnalyzer()
    
    if not analyzer.setup():
        return
    
    print(f"🚀 Starting live SuperTrend monitoring for {args.symbol}")
    print(f"⏱️  Update interval: {args.interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            analyzer.print_latest_candle(args.symbol)
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
