"""
Quick test script for the SuperTrend backtesting system
=====================================================

This script tests the backtesting system with different parameter combinations
to ensure it's working correctly and generating trades.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_strategy import SuperTrendBacktester, create_sample_data

def test_parameter_sensitivity():
    """Test different parameter combinations to find working settings"""
    
    print("🧪 TESTING SUPERTREND BACKTEST SYSTEM")
    print("=" * 50)
    
    # Load sample data
    print("Loading sample data...")
    df = create_sample_data()
    
    # Test different parameter combinations
    test_configs = [
        {'atr_period': 10, 'factor': 2.0, 'name': 'Sensitive (Factor 2.0)'},
        {'atr_period': 10, 'factor': 2.5, 'name': 'Moderate (Factor 2.5)'},
        {'atr_period': 10, 'factor': 3.0, 'name': 'Conservative (Factor 3.0)'},
        {'atr_period': 7, 'factor': 2.5, 'name': 'Fast ATR (7)'},
        {'atr_period': 14, 'factor': 2.5, 'name': 'Slow ATR (14)'},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n📊 Testing: {config['name']}")
        print("-" * 30)
        
        # Base configuration
        backtest_config = {
            'initial_capital': 10000,
            'leverage': 5.0,
            'stop_loss': 100,
            'commission_per_trade': 20,
            'atr_period': config['atr_period'],
            'factor': config['factor']
        }
        
        # Run backtest
        backtester = SuperTrendBacktester(**backtest_config)
        result_df = backtester.run_backtest(df.copy())
        
        # Check results
        if len(backtester.trades) > 0:
            metrics = backtester.calculate_metrics(result_df)
            
            print(f"✅ SUCCESS: {len(backtester.trades)} trades executed")
            print(f"   Return: {metrics['Total Return (%)']:.2f}%")
            print(f"   Win Rate: {metrics['Win Rate (%)']:.1f}%")
            print(f"   Profit Factor: {metrics['Profit Factor']:.2f}")
            
            results.append({
                'config': config['name'],
                'trades': len(backtester.trades),
                'return': metrics['Total Return (%)'],
                'win_rate': metrics['Win Rate (%)'],
                'profit_factor': metrics['Profit Factor']
            })
        else:
            print("❌ FAILED: No trades executed")
            
            # Debug information
            signals = result_df['signal'].sum()
            print(f"   Buy signals detected: {signals}")
            
            if signals > 0:
                print("   Issue: Signals detected but no trades executed")
                print("   Possible causes: insufficient capital, position sizing issues")
            else:
                print("   Issue: No buy signals detected")
                print("   Suggestion: Try more sensitive parameters")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"✅ {len(results)} configurations generated trades")
        
        # Find best performer
        best_return = max(results, key=lambda x: x['return'])
        best_trades = max(results, key=lambda x: x['trades'])
        
        print(f"\n🏆 Best Return: {best_return['config']}")
        print(f"   {best_return['return']:.2f}% return, {best_return['trades']} trades")
        
        print(f"\n📈 Most Active: {best_trades['config']}")
        print(f"   {best_trades['trades']} trades, {best_trades['return']:.2f}% return")
        
        # Recommend configuration
        print(f"\n💡 RECOMMENDATION:")
        if best_return['return'] > 10 and best_return['trades'] > 5:
            print(f"   Use: {best_return['config']}")
            print(f"   This configuration provides good returns with reasonable activity")
        else:
            print("   Consider using Factor=2.0 or ATR=7 for more signals")
            print("   The sample data might not be ideal for this strategy")
        
    else:
        print("❌ No configurations generated trades")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Try factor values between 1.5-2.0")
        print("2. Try shorter ATR periods (5-7)")
        print("3. Check if sample data has sufficient trends")
        print("4. Verify SuperTrend calculation logic")

def create_trend_test_data():
    """Create data specifically designed to test SuperTrend signals"""
    
    print("\n🔧 Creating trend-specific test data...")
    
    # Create data with clear trends
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    data = []
    price = 100.0
    
    for i, date in enumerate(dates):
        # Create clear trend phases
        if i < len(dates) * 0.3:  # First 30% - uptrend
            daily_change = np.random.normal(0.8, 1.5)  # Strong uptrend
        elif i < len(dates) * 0.6:  # Next 30% - downtrend  
            daily_change = np.random.normal(-0.7, 1.5)  # Strong downtrend
        else:  # Last 40% - uptrend
            daily_change = np.random.normal(0.6, 1.5)  # Moderate uptrend
        
        price = max(price + daily_change, 50)  # Don't go below 50
        
        # Generate OHLC
        open_price = price + np.random.normal(0, 0.3)
        close = price + np.random.normal(0, 0.3)
        high = max(open_price, close) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, 0.5))
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': np.random.randint(10000, 50000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    print(f"✅ Created trending data: {len(df)} days")
    print(f"   Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    
    return df

def test_with_trending_data():
    """Test backtester with data designed to generate signals"""
    
    print("\n📈 TESTING WITH TRENDING DATA")
    print("=" * 40)
    
    # Create trending data
    df = create_trend_test_data()
    
    # Test with sensitive parameters
    config = {
        'initial_capital': 10000,
        'leverage': 5.0,
        'stop_loss': 100,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 2.0  # Sensitive
    }
    
    backtester = SuperTrendBacktester(**config)
    result_df = backtester.run_backtest(df)
    
    if len(backtester.trades) > 0:
        print(f"✅ SUCCESS: {len(backtester.trades)} trades with trending data")
        
        # Generate full report and charts
        backtester.generate_report(result_df, 'trending_test_report.txt')
        backtester.plot_results(result_df, 'trending_test_charts.png')
        
        metrics = backtester.calculate_metrics(result_df)
        print(f"📊 Quick Results:")
        print(f"   Return: {metrics['Total Return (%)']:.2f}%")
        print(f"   Win Rate: {metrics['Win Rate (%)']:.1f}%")
        print(f"   Trades: {metrics['Total Trades']}")
        
        return True
    else:
        print("❌ Still no trades with trending data")
        return False

def main():
    """Run all tests"""
    
    # Test 1: Parameter sensitivity
    test_parameter_sensitivity()
    
    # Test 2: Trending data
    success = test_with_trending_data()
    
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)
    
    if success:
        print("✅ Backtesting system is working correctly!")
        print("📁 Check generated files:")
        print("   - trending_test_report.txt")
        print("   - trending_test_charts.png")
        print("\n🚀 You can now run: python backtest_runner.py quick")
    else:
        print("❌ Backtesting system needs debugging")
        print("🔧 Check SuperTrend calculation logic")

if __name__ == "__main__":
    main()
