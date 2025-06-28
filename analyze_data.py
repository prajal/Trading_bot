#!/usr/bin/env python3
"""
FIXED: Script to analyze and print OHLC and SuperTrend data
Updated to work with enhanced project structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from auth.enhanced_kite_auth import KiteAuth  # FIXED: Updated import
from trading.enhanced_strategy import EnhancedSuperTrendStrategy  # FIXED: Updated import
# Handle optional dependencies gracefully
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("⚠️  Note: tabulate not installed. Using simple table format.")
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
    """Enhanced SuperTrend Analyzer using the enhanced strategy"""
    
    def __init__(self, atr_period=10, factor=3.0):
        self.auth = KiteAuth()
        self.strategy = EnhancedSuperTrendStrategy(
            atr_period=atr_period, 
            factor=factor,
            adaptive_mode=True  # Enable adaptive mode
        )
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
            
            # FIXED: Use conservative 3-day limit to avoid API issues
            from_date = to_date - timedelta(days=3)
            
            print(f"📥 Fetching NIFTY 50 data from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
            
            data = self.kite.historical_data(
                nifty50_token, 
                from_date, 
                to_date, 
                interval
            )
            
            if not data:
                print("❌ No data received")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            print(f"✅ Loaded {len(df)} NIFTY 50 candles")
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
            # FIXED: Use conservative 3-day limit
            from_date = to_date - timedelta(days=3)
            
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
    
    def analyze_current_market(self, symbol="NIFTYBEES"):
        """Enhanced analysis of current market using dual data approach"""
        if symbol not in TRADING_INSTRUMENTS:
            print(f"❌ Unknown instrument: {symbol}")
            return
        
        print(f"\n🔍 Enhanced SuperTrend Analysis: {symbol}")
        print("=" * 60)
        
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
            # Get enhanced signal from NIFTY 50 data
            signal, signal_data = self.strategy.get_signal(nifty50_df, has_position=False)
            
            # Get latest NIFTYBEES price
            latest_bees = bees_df.iloc[-1]
            latest_bees_time = pd.to_datetime(bees_df.index[-1]).strftime('%H:%M:%S')
            
            print(f"📊 Market Analysis ({latest_bees_time}):")
            print(f"   Signal Source: NIFTY 50 Index")
            print(f"   Trading Price: {symbol} @ ₹{latest_bees['close']:.2f}")
            print(f"   Volume: {latest_bees.get('volume', 0):,}")
            
            print(f"\n🎯 Enhanced SuperTrend Signal:")
            print(f"   Signal: {signal}")
            print(f"   Trend: {signal_data.get('trend', 'Unknown')}")
            print(f"   Direction: {signal_data.get('direction', 'Unknown')}")
            print(f"   Confidence: {signal_data.get('confidence', 0):.2%}")
            print(f"   Market Regime: {signal_data.get('regime', 'Unknown')}")
            print(f"   SuperTrend: ₹{signal_data.get('supertrend', 0):.2f}")
            
            # Strategy parameters
            print(f"\n⚙️  Current Strategy Parameters:")
            print(f"   ATR Period: {signal_data.get('atr_period', 'Unknown')}")
            print(f"   Factor: {signal_data.get('factor', 'Unknown')}")
            print(f"   Adaptive Mode: Enabled")
            
            # Price analysis
            current_price = latest_bees['close']
            supertrend_price = signal_data.get('supertrend', 0)
            if supertrend_price > 0:
                price_distance = current_price - supertrend_price
                price_distance_pct = (price_distance / supertrend_price) * 100
                
                print(f"\n📈 Price Analysis:")
                print(f"   Current Price: ₹{current_price:.2f}")
                print(f"   SuperTrend: ₹{supertrend_price:.2f}")
                print(f"   Distance: ₹{price_distance:+.2f} ({price_distance_pct:+.2f}%)")
                print(f"   Position: {signal_data.get('price_vs_supertrend', 'Unknown')}")
            
            # Risk metrics
            atr_value = signal_data.get('atr', 0)
            if atr_value > 0:
                print(f"\n🛡️  Risk Metrics:")
                print(f"   ATR: ₹{atr_value:.2f}")
                print(f"   ATR %: {(atr_value / current_price) * 100:.2f}%")
                
                # Suggested position sizing
                account_balance = 10000  # Example
                risk_per_trade = 0.02    # 2%
                risk_amount = account_balance * risk_per_trade
                stop_distance = atr_value * 2  # 2x ATR stop
                
                if stop_distance > 0:
                    suggested_quantity = int(risk_amount / stop_distance)
                    position_value = suggested_quantity * current_price
                    
                    print(f"\n💰 Suggested Position (2% risk):")
                    print(f"   Quantity: {suggested_quantity} shares")
                    print(f"   Position Value: ₹{position_value:,.2f}")
                    print(f"   Stop Loss Distance: ₹{stop_distance:.2f}")
            
            # Display recent candles with graceful fallback
            print(f"\n📊 Recent Market Data (Last 5 candles):")
            recent_bees = bees_df.tail(5)[['open', 'high', 'low', 'close', 'volume']]
            recent_bees['time'] = recent_bees.index.strftime('%H:%M')
            recent_bees = recent_bees[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            if TABULATE_AVAILABLE:
                print(tabulate(recent_bees, headers=recent_bees.columns, tablefmt='grid', floatfmt='.2f'))
            else:
                # Simple fallback table format
                print(f"{'Time':<8} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
                print("-" * 60)
                for _, row in recent_bees.iterrows():
                    print(f"{row['time']:<8} {row['open']:<8.2f} {row['high']:<8.2f} {row['low']:<8.2f} {row['close']:<8.2f} {row['volume']:<10,.0f}")
            
            # Action recommendation
            print(f"\n🎯 Action Recommendation:")
            if signal == "BUY":
                print(f"   📈 CONSIDER BUYING")
                print(f"   ✅ Strong uptrend signal detected")
                print(f"   ⚠️  Confidence: {signal_data.get('confidence', 0):.2%}")
            elif signal == "SELL":
                print(f"   📉 CONSIDER SELLING")
                print(f"   ❌ Downtrend signal detected")
                print(f"   ⚠️  Confidence: {signal_data.get('confidence', 0):.2%}")
            else:
                print(f"   ⏸️  HOLD POSITION")
                print(f"   📊 No clear signal at this time")
                if 'reason' in signal_data:
                    print(f"   💡 Reason: {signal_data['reason']}")
            
            print(f"\n⚠️  Disclaimer: This is for educational purposes only. Always do your own research!")
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
    
    def print_latest_candle(self, symbol="NIFTYBEES"):
        """Print the latest candle with enhanced info"""
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
            # Get enhanced signal
            signal, signal_data = self.strategy.get_signal(nifty50_df, has_position=False)
            
            # Get latest NIFTYBEES price
            latest_bees = bees_df.iloc[-1]
            latest_bees_time = pd.to_datetime(bees_df.index[-1]).strftime('%H:%M:%S')
            
            # Enhanced display
            trend = signal_data.get('trend', 'Unknown')
            confidence = signal_data.get('confidence', 0)
            regime = signal_data.get('regime', 'default')
            supertrend = signal_data.get('supertrend', 0)
            
            delta = latest_bees['close'] - supertrend if supertrend > 0 else 0
            delta_str = f"{delta:+.2f}"
            
            print(f"{latest_bees_time} | {trend} | {symbol}: ₹{latest_bees['close']:.2f} | "
                  f"ST: {supertrend:.2f} | Δ:{delta_str} | Conf:{confidence:.2%} | "
                  f"Regime:{regime} | {signal}")
            
        except Exception as e:
            print(f"❌ Error calculating enhanced SuperTrend: {e}")

def main():
    """Main function with enhanced monitoring"""
    parser = argparse.ArgumentParser(description="Enhanced SuperTrend Analysis")
    parser.add_argument("--symbol", default="NIFTYBEES", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds (default: 60)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    args = parser.parse_args()
    
    analyzer = SuperTrendAnalyzer()
    
    if not analyzer.setup():
        return
    
    if args.detailed:
        # Show detailed analysis once
        analyzer.analyze_current_market(args.symbol)
        return
    
    print(f"🚀 Enhanced SuperTrend Live Monitor: {args.symbol}")
    print(f"⏱️  Update interval: {args.interval} seconds")
    print(f"💡 Use --detailed for comprehensive analysis")
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