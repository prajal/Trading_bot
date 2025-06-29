#!/usr/bin/env python3
"""
SuperTrend Diagnostic Script
Analyzes why SuperTrend isn't generating signals on historical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.enhanced_strategy import EnhancedSuperTrendStrategy

class SuperTrendDiagnostic:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.strategy = EnhancedSuperTrendStrategy(atr_period=10, factor=3.0)
        
    def load_data(self):
        """Load and prepare data"""
        print(f"\n📁 Loading data from: {self.csv_file}")
        df = pd.read_csv(self.csv_file, parse_dates=['date'])
        df.set_index('date', inplace=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        print(f"✅ Loaded {len(df)} rows")
        print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        print(f"💰 Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")
        
        return df
    
    def analyze_supertrend(self, df):
        """Analyze SuperTrend calculations"""
        print("\n🔍 Analyzing SuperTrend Calculations...")
        
        # Calculate SuperTrend
        df_st = self.strategy.calculate_supertrend(df.copy())
        
        # Basic statistics
        print(f"\n📊 SuperTrend Statistics:")
        print(f"   ATR Mean: {df_st['atr'].mean():.2f}")
        print(f"   ATR Std: {df_st['atr'].std():.2f}")
        print(f"   ATR Range: {df_st['atr'].min():.2f} - {df_st['atr'].max():.2f}")
        
        # Direction changes
        direction_changes = (df_st['direction'].diff() != 0).sum() - 1  # -1 for first NaN
        print(f"\n🔄 Direction Changes: {direction_changes}")
        
        # Trend distribution
        uptrend_count = (df_st['direction'] == 1).sum()
        downtrend_count = (df_st['direction'] == -1).sum()
        uptrend_pct = uptrend_count / len(df_st) * 100
        downtrend_pct = downtrend_count / len(df_st) * 100
        
        print(f"\n📈 Trend Distribution:")
        print(f"   Uptrend: {uptrend_count} candles ({uptrend_pct:.1f}%)")
        print(f"   Downtrend: {downtrend_count} candles ({downtrend_pct:.1f}%)")
        
        # Find direction change points
        direction_changes_idx = df_st[df_st['direction'].diff() != 0].index[1:]  # Skip first
        
        print(f"\n🎯 Direction Change Dates:")
        for i, date in enumerate(direction_changes_idx[:10]):  # Show first 10
            prev_dir = df_st.loc[:date, 'direction'].iloc[-2]
            curr_dir = df_st.loc[date, 'direction']
            price = df_st.loc[date, 'close']
            st_value = df_st.loc[date, 'supertrend']
            
            signal = "BUY" if prev_dir == -1 and curr_dir == 1 else "SELL"
            print(f"   {i+1}. {date.strftime('%Y-%m-%d')}: {signal} @ ₹{price:.2f} (ST: ₹{st_value:.2f})")
        
        if len(direction_changes_idx) > 10:
            print(f"   ... and {len(direction_changes_idx) - 10} more")
        
        return df_st, direction_changes_idx
    
    def analyze_signals(self, df):
        """Analyze signal generation"""
        print("\n🎯 Analyzing Signal Generation...")
        
        # Track signals over time
        signals = []
        has_position = False
        
        for i in range(50, len(df)):  # Start after min_candles
            window = df.iloc[:i+1]
            signal, signal_data = self.strategy.get_signal(window, has_position)
            
            if signal != "HOLD":
                signals.append({
                    'date': df.index[i],
                    'signal': signal,
                    'price': df.iloc[i]['close'],
                    'confidence': signal_data.get('confidence', 0),
                    'direction': signal_data.get('direction', 0)
                })
                
                # Update position status
                if signal == "BUY":
                    has_position = True
                elif signal == "SELL":
                    has_position = False
        
        print(f"\n📊 Signals Generated: {len(signals)}")
        
        if signals:
            print("\n📋 Signal Details (First 10):")
            for i, sig in enumerate(signals[:10]):
                print(f"   {i+1}. {sig['date'].strftime('%Y-%m-%d')}: {sig['signal']} @ ₹{sig['price']:.2f} (Conf: {sig['confidence']:.2f})")
        else:
            print("   ❌ No signals generated!")
        
        return signals
    
    def diagnose_issues(self, df, df_st, signals):
        """Diagnose potential issues"""
        print("\n🔧 Diagnostic Analysis:")
        
        issues = []
        
        # Issue 1: Check if SuperTrend is too far from price
        avg_distance = abs(df_st['close'] - df_st['supertrend']).mean()
        avg_price = df_st['close'].mean()
        distance_pct = (avg_distance / avg_price) * 100
        
        print(f"\n1️⃣ SuperTrend Distance from Price:")
        print(f"   Average Distance: ₹{avg_distance:.2f} ({distance_pct:.1f}%)")
        
        if distance_pct > 5:
            issues.append("SuperTrend is too far from price - consider reducing factor")
        
        # Issue 2: Check ATR stability
        atr_cv = df_st['atr'].std() / df_st['atr'].mean()
        print(f"\n2️⃣ ATR Stability:")
        print(f"   Coefficient of Variation: {atr_cv:.2f}")
        
        if atr_cv > 0.5:
            issues.append("ATR is highly variable - market may be too volatile")
        
        # Issue 3: Check for flat periods
        flat_st = (df_st['supertrend'].diff() == 0).sum()
        flat_pct = flat_st / len(df_st) * 100
        
        print(f"\n3️⃣ Flat SuperTrend Periods:")
        print(f"   Flat Candles: {flat_st} ({flat_pct:.1f}%)")
        
        if flat_pct > 20:
            issues.append("SuperTrend has many flat periods - may indicate calculation issues")
        
        # Issue 4: Check confidence levels
        if signals:
            avg_confidence = np.mean([s['confidence'] for s in signals])
            print(f"\n4️⃣ Signal Confidence:")
            print(f"   Average Confidence: {avg_confidence:.2f}")
            
            if avg_confidence < 0.6:
                issues.append("Low signal confidence - signals may be filtered out")
        
        # Issue 5: Check for stuck direction
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, len(df_st)):
            if df_st['direction'].iloc[i] == df_st['direction'].iloc[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        print(f"\n5️⃣ Maximum Consecutive Same Direction:")
        print(f"   Max Consecutive: {max_consecutive} candles")
        
        if max_consecutive > len(df_st) * 0.8:
            issues.append("Direction rarely changes - SuperTrend may be stuck")
        
        # Summary
        print(f"\n🚨 Issues Found: {len(issues)}")
        for i, issue in enumerate(issues):
            print(f"   {i+1}. {issue}")
        
        return issues
    
    def plot_analysis(self, df_st, signals):
        """Plot diagnostic charts"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and SuperTrend
        ax1 = axes[0]
        ax1.plot(df_st.index, df_st['close'], label='Close Price', linewidth=1)
        ax1.plot(df_st.index, df_st['supertrend'], label='SuperTrend', linewidth=2)
        
        # Mark signals
        if signals:
            buy_signals = [s for s in signals if s['signal'] == 'BUY']
            sell_signals = [s for s in signals if s['signal'] == 'SELL']
            
            if buy_signals:
                buy_dates = [s['date'] for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
                ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            
            if sell_signals:
                sell_dates = [s['date'] for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
                ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Price and SuperTrend with Signals')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Direction
        ax2 = axes[1]
        ax2.plot(df_st.index, df_st['direction'], label='Direction', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('SuperTrend Direction (1=Up, -1=Down)')
        ax2.set_ylabel('Direction')
        ax2.set_ylim(-1.5, 1.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ATR
        ax3 = axes[2]
        ax3.plot(df_st.index, df_st['atr'], label='ATR', color='orange', linewidth=1)
        ax3.set_title('Average True Range (ATR)')
        ax3.set_ylabel('ATR')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('supertrend_diagnostic.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Diagnostic plot saved as 'supertrend_diagnostic.png'")
        
        # Additional analysis plot
        fig2, axes2 = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot 4: Price distance from SuperTrend
        ax4 = axes2[0]
        distance = df_st['close'] - df_st['supertrend']
        distance_pct = (distance / df_st['supertrend']) * 100
        ax4.plot(df_st.index, distance_pct, label='Price Distance %', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.fill_between(df_st.index, 0, distance_pct, alpha=0.3)
        ax4.set_title('Price Distance from SuperTrend (%)')
        ax4.set_ylabel('Distance %')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Rolling direction changes
        ax5 = axes2[1]
        direction_changes = df_st['direction'].diff().abs() / 2
        rolling_changes = direction_changes.rolling(window=20).sum()
        ax5.plot(df_st.index, rolling_changes, label='20-day Rolling Direction Changes', linewidth=1)
        ax5.set_title('Trend Change Frequency (20-day Rolling)')
        ax5.set_ylabel('Number of Changes')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('supertrend_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📊 Analysis plot saved as 'supertrend_analysis.png'")
        
        plt.show()
    
    def suggest_parameters(self, df):
        """Suggest better parameters"""
        print("\n💡 Parameter Optimization Suggestions:")
        
        # Calculate price volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        print(f"\n📊 Market Characteristics:")
        print(f"   Annualized Volatility: {volatility:.1%}")
        print(f"   Average Daily Range: {((df['high'] - df['low']) / df['close']).mean():.1%}")
        
        # Suggest parameters based on volatility
        if volatility < 0.20:  # Low volatility
            suggested_atr = 7
            suggested_factor = 2.0
            print(f"\n✅ Low volatility market - Suggested parameters:")
        elif volatility < 0.40:  # Medium volatility
            suggested_atr = 10
            suggested_factor = 2.5
            print(f"\n✅ Medium volatility market - Suggested parameters:")
        else:  # High volatility
            suggested_atr = 14
            suggested_factor = 3.5
            print(f"\n✅ High volatility market - Suggested parameters:")
        
        print(f"   ATR Period: {suggested_atr}")
        print(f"   Factor: {suggested_factor}")
        
        # Test suggested parameters
        print(f"\n🧪 Testing suggested parameters...")
        test_strategy = EnhancedSuperTrendStrategy(atr_period=suggested_atr, factor=suggested_factor)
        df_test = test_strategy.calculate_supertrend(df.copy())
        
        test_direction_changes = (df_test['direction'].diff() != 0).sum() - 1
        print(f"   Direction changes with suggested params: {test_direction_changes}")
        print(f"   Average signals per year: {test_direction_changes / (len(df) / 252):.1f}")
        
        return suggested_atr, suggested_factor
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis"""
        print("🏥 SuperTrend Diagnostic Tool")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Analyze SuperTrend
        df_st, direction_changes = self.analyze_supertrend(df)
        
        # Analyze signals
        signals = self.analyze_signals(df)
        
        # Diagnose issues
        issues = self.diagnose_issues(df, df_st, signals)
        
        # Suggest parameters
        suggested_atr, suggested_factor = self.suggest_parameters(df)
        
        # Plot analysis
        self.plot_analysis(df_st, signals)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Data Points: {len(df)}")
        print(f"Direction Changes: {len(direction_changes)}")
        print(f"Signals Generated: {len(signals)}")
        print(f"Issues Found: {len(issues)}")
        print(f"Suggested Parameters: ATR={suggested_atr}, Factor={suggested_factor}")
        
        if len(signals) == 0 or len(direction_changes) < 5:
            print("\n⚠️  CRITICAL: Strategy is not generating enough signals!")
            print("   Consider:")
            print("   1. Using suggested parameters")
            print("   2. Checking data quality")
            print("   3. Adjusting minimum confidence thresholds")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperTrend Diagnostic Tool')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"❌ File not found: {args.csv}")
        return
    
    diagnostic = SuperTrendDiagnostic(args.csv)
    diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    main()
