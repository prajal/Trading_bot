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
    'NIFTY 50': {'name': 'NIFTY 50 Index', 'token': '256265'},
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
    """Analyze OHLC and SuperTrend data using dual data approach"""
    
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
    
    def get_nifty50_data_for_signals(self, days=1, interval="minute"):
        """Fetch NIFTY 50 data for SuperTrend signal generation"""
        try:
            if not self.kite:
                print("❌ Kite connection not available")
                return pd.DataFrame()
                
            nifty50_token = TRADING_INSTRUMENTS['NIFTY 50']['token']
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                nifty50_token, 
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
            print(f"❌ Error fetching NIFTY 50 data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, instrument_token, days=1, interval="minute"):
        """Fetch historical data for price display"""
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
        """Print the last 3 candles using dual data approach: NIFTY 50 signals + NIFTYBEES prices."""
        if symbol not in TRADING_INSTRUMENTS:
            print(f"❌ Unknown instrument: {symbol}")
            return
        
        # Get NIFTY 50 data for signals
        nifty50_df = self.get_nifty50_data_for_signals(days=1)
        if nifty50_df.empty:
            print("❌ No NIFTY 50 data available for signals")
            return
        
        # Get NIFTYBEES data for price display
        instrument = TRADING_INSTRUMENTS[symbol]
        bees_df = self.get_historical_data(instrument['token'], days=1)
        if bees_df.empty:
            print("❌ No NIFTYBEES data available")
            return
        
        try:
            # Calculate SuperTrend on NIFTY 50 data
            nifty50_with_st = self.strategy.calculate_supertrend(nifty50_df)
            
            # Get latest NIFTYBEES price
            latest_bees = bees_df.iloc[-1]
            latest_bees_time = pd.to_datetime(bees_df.index[-1]).strftime('%H:%M:%S')
            
            # Get latest NIFTY 50 signal
            latest_nifty50 = nifty50_with_st.iloc[-1]
            nifty50_time = pd.to_datetime(nifty50_with_st.index[-1]).strftime('%H:%M:%S')
            
            # Detect signal from NIFTY 50
            if len(nifty50_with_st) > 1:
                prev_dir = nifty50_with_st['direction'].iloc[-2]
                curr_dir = latest_nifty50['direction']
                if prev_dir == -1 and curr_dir == 1:
                    signal = 'ENTRY↑'
                elif prev_dir == 1 and curr_dir == -1:
                    signal = 'EXIT↓'
                else:
                    signal = 'HOLD'
            else:
                signal = 'HOLD'
            
            # Display dual data approach
            st_dir = 'UP' if latest_nifty50['direction'] == 1 else 'DOWN'
            delta = latest_bees['close'] - latest_nifty50['supertrend']
            delta_str = f"{delta:+.2f}"
            
            print(f"{latest_bees_time} | NIFTY 50 Signal: {st_dir} | NIFTYBEES: ₹{latest_bees['close']:.2f} | ST: {latest_nifty50['supertrend']:.2f} | Δ:{delta_str} | {signal}")
            
        except Exception as e:
            print(f"❌ Error calculating SuperTrend: {e}")
            return

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
