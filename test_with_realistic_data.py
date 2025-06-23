#!/usr/bin/env python3
"""
Test Backtesting with Realistic Data
====================================

This script tests the SuperTrend backtesting system with realistic market data
that should generate trading signals.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_strategy import SuperTrendBacktester
from generate_test_data import generate_realistic_nifty_data, generate_trending_data, generate_volatile_data

def test_with_realistic_data():
    """Test backtesting with realistic data"""
    print("🧪 TESTING SUPERTREND WITH REALISTIC DATA")
    print("=" * 50)
    
    # Test with realistic NIFTY data
    print("\n📈 Testing with Realistic NIFTY Data")
    print("-" * 40)
    
    nifty_df = generate_realistic_nifty_data(days=60)
    print(f"Data: {len(nifty_df)} days, Price range: ₹{nifty_df['close'].min():.2f} - ₹{nifty_df['close'].max():.2f}")
    
    # Configure backtest
    config = {
        'initial_capital': 4000.0,
        'leverage': 5.0,
        'stop_loss': 100.0,
        'commission_per_trade': 20.0,
        'atr_period': 10,
        'factor': 3.0
    }
    
    # Run backtest
    backtester = SuperTrendBacktester(**config)
    result_df = backtester.run_backtest(nifty_df)
    
    if len(backtester.trades) > 0:
        metrics = backtester.calculate_metrics(result_df)
        print(f"✅ NIFTY Data Results:")
        print(f"   Trades: {metrics.get('Total Trades', 0)}")
        print(f"   Win Rate: {metrics.get('Win Rate (%)', 0):.1f}%")
        print(f"   Return: {metrics.get('Total Return (%)', 0):.2f}%")
        print(f"   Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
    else:
        print("❌ No trades with NIFTY data")
    
    # Test with trending data
    print("\n📈 Testing with Trending Data")
    print("-" * 40)
    
    trending_df = generate_trending_data(days=60)
    print(f"Data: {len(trending_df)} days, Price range: ₹{trending_df['close'].min():.2f} - ₹{trending_df['close'].max():.2f}")
    
    backtester2 = SuperTrendBacktester(**config)
    result_df2 = backtester2.run_backtest(trending_df)
    
    if len(backtester2.trades) > 0:
        metrics2 = backtester2.calculate_metrics(result_df2)
        print(f"✅ Trending Data Results:")
        print(f"   Trades: {metrics2.get('Total Trades', 0)}")
        print(f"   Win Rate: {metrics2.get('Win Rate (%)', 0):.1f}%")
        print(f"   Return: {metrics2.get('Total Return (%)', 0):.2f}%")
        print(f"   Sharpe: {metrics2.get('Sharpe Ratio', 0):.2f}")
    else:
        print("❌ No trades with trending data")
    
    # Test with volatile data
    print("\n📊 Testing with Volatile Data")
    print("-" * 40)
    
    volatile_df = generate_volatile_data(days=60)
    print(f"Data: {len(volatile_df)} days, Price range: ₹{volatile_df['close'].min():.2f} - ₹{volatile_df['close'].max():.2f}")
    
    backtester3 = SuperTrendBacktester(**config)
    result_df3 = backtester3.run_backtest(volatile_df)
    
    if len(backtester3.trades) > 0:
        metrics3 = backtester3.calculate_metrics(result_df3)
        print(f"✅ Volatile Data Results:")
        print(f"   Trades: {metrics3.get('Total Trades', 0)}")
        print(f"   Win Rate: {metrics3.get('Win Rate (%)', 0):.1f}%")
        print(f"   Return: {metrics3.get('Total Return (%)', 0):.2f}%")
        print(f"   Sharpe: {metrics3.get('Sharpe Ratio', 0):.2f}")
    else:
        print("❌ No trades with volatile data")
    
    # Test different parameters
    print("\n🔬 Testing Different Parameters")
    print("-" * 40)
    
    # Test with more sensitive parameters
    sensitive_config = config.copy()
    sensitive_config.update({'atr_period': 7, 'factor': 2.0})
    
    backtester4 = SuperTrendBacktester(**sensitive_config)
    result_df4 = backtester4.run_backtest(nifty_df.copy())
    
    if len(backtester4.trades) > 0:
        metrics4 = backtester4.calculate_metrics(result_df4)
        print(f"✅ Sensitive Parameters (ATR=7, Factor=2.0):")
        print(f"   Trades: {metrics4.get('Total Trades', 0)}")
        print(f"   Win Rate: {metrics4.get('Win Rate (%)', 0):.1f}%")
        print(f"   Return: {metrics4.get('Total Return (%)', 0):.2f}%")
        print(f"   Sharpe: {metrics4.get('Sharpe Ratio', 0):.2f}")
    else:
        print("❌ No trades with sensitive parameters")
    
    # Save best results
    print("\n💾 Saving Results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the best performing backtest
    best_backtester = None
    best_return = -999
    
    for i, (bt, name) in enumerate([(backtester, "NIFTY"), (backtester2, "Trending"), (backtester3, "Volatile"), (backtester4, "Sensitive")]):
        if len(bt.trades) > 0:
            metrics = bt.calculate_metrics(bt.equity_curve)
            if metrics.get('Total Return (%)', 0) > best_return:
                best_return = metrics.get('Total Return (%)', 0)
                best_backtester = bt
    
    if best_backtester:
        report_file = f"realistic_backtest_report_{timestamp}.txt"
        chart_file = f"realistic_backtest_charts_{timestamp}.png"
        
        best_backtester.generate_report(best_backtester.equity_curve, report_file)
        best_backtester.plot_results(best_backtester.equity_curve, chart_file)
        
        print(f"📊 Best results saved:")
        print(f"   Report: {report_file}")
        print(f"   Charts: {chart_file}")
    
    return True

def main():
    """Main function"""
    print("🎯 Test SuperTrend with Realistic Data")
    print("=" * 50)
    
    success = test_with_realistic_data()
    
    if success:
        print("\n✅ Testing completed successfully!")
        print("\n💡 Next steps:")
        print("1. Review the generated reports and charts")
        print("2. Set up your Zerodha API credentials in .env file")
        print("3. Run: python cli.py auth")
        print("4. Then run: python run_backtest.py")
    else:
        print("\n❌ Testing failed!")

if __name__ == "__main__":
    main() 