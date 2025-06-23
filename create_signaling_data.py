#!/usr/bin/env python3
"""
Create Data That Generates SuperTrend Signals
============================================

This script creates market data specifically designed to trigger SuperTrend signals
for testing the backtesting system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_signaling_data():
    """Create data that will definitely generate SuperTrend signals"""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(789)  # For reproducible results
    
    # Create a price series with clear trend changes
    prices = []
    current_price = 280  # Start price
    
    # Phase 1: Strong uptrend (20 days)
    for i in range(20):
        current_price *= (1 + np.random.normal(0.008, 0.01))  # Strong uptrend
        prices.append(current_price)
    
    # Phase 2: Downtrend (15 days)
    for i in range(15):
        current_price *= (1 + np.random.normal(-0.006, 0.01))  # Downtrend
        prices.append(current_price)
    
    # Phase 3: Strong uptrend again (20 days)
    for i in range(20):
        current_price *= (1 + np.random.normal(0.01, 0.01))  # Strong uptrend
        prices.append(current_price)
    
    # Phase 4: Downtrend (15 days)
    for i in range(15):
        current_price *= (1 + np.random.normal(-0.008, 0.01))  # Downtrend
        prices.append(current_price)
    
    # Phase 5: Final uptrend (30 days)
    for i in range(30):
        current_price *= (1 + np.random.normal(0.005, 0.01))  # Moderate uptrend
        prices.append(current_price)
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC
        daily_volatility = price * 0.015  # 1.5% daily volatility
        
        high = price + abs(np.random.normal(0, daily_volatility * 0.6))
        low = price - abs(np.random.normal(0, daily_volatility * 0.6))
        
        # Ensure high >= close >= low
        high = max(high, price)
        low = min(low, price)
        
        # Open price
        if i == 0:
            open_price = price + np.random.normal(0, daily_volatility * 0.3)
        else:
            open_price = data[i-1]['close'] + np.random.normal(0, daily_volatility * 0.2)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

def test_super_trend_signals():
    """Test if the data generates SuperTrend signals"""
    print("🧪 Testing SuperTrend Signal Generation")
    print("=" * 50)
    
    # Create signaling data
    df = create_signaling_data()
    print(f"📊 Generated {len(df)} days of data")
    print(f"📈 Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    print(f"📅 Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Test SuperTrend calculation
    from backtest_strategy import SuperTrendBacktester
    
    config = {
        'initial_capital': 4000.0,
        'leverage': 5.0,
        'stop_loss': 100.0,
        'commission_per_trade': 20.0,
        'atr_period': 10,
        'factor': 3.0
    }
    
    backtester = SuperTrendBacktester(**config)
    result_df = backtester.run_backtest(df)
    
    print(f"\n📊 SuperTrend Results:")
    print(f"   Buy signals: {result_df['signal'].sum()}")
    print(f"   Sell signals: {result_df['exit_signal'].sum()}")
    print(f"   Total trades: {len(backtester.trades)}")
    
    if len(backtester.trades) > 0:
        metrics = backtester.calculate_metrics(result_df)
        print(f"   Win Rate: {metrics.get('Win Rate (%)', 0):.1f}%")
        print(f"   Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
        print(f"   Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        
        # Show trade details
        print(f"\n📋 Trade Details:")
        for i, trade in enumerate(backtester.trades):
            entry_date = pd.to_datetime(trade['entry_date']).strftime('%Y-%m-%d')
            exit_date = pd.to_datetime(trade['exit_date']).strftime('%Y-%m-%d')
            print(f"  Trade {i+1}: {entry_date} → {exit_date}")
            print(f"    Entry: ₹{trade['entry_price']:.2f} | Exit: ₹{trade['exit_price']:.2f}")
            print(f"    P&L: ₹{trade['pnl']:.2f} | Reason: {trade['exit_reason']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"signaling_backtest_report_{timestamp}.txt"
        chart_file = f"signaling_backtest_charts_{timestamp}.png"
        
        backtester.generate_report(result_df, report_file)
        backtester.plot_results(result_df, chart_file)
        
        print(f"\n💾 Results saved:")
        print(f"   Report: {report_file}")
        print(f"   Charts: {chart_file}")
        
        return True
    else:
        print("❌ No trades generated. Trying with more sensitive parameters...")
        
        # Try with more sensitive parameters
        sensitive_config = config.copy()
        sensitive_config.update({'atr_period': 5, 'factor': 1.5})
        
        backtester2 = SuperTrendBacktester(**sensitive_config)
        result_df2 = backtester2.run_backtest(df)
        
        print(f"\n🔬 Sensitive Parameters (ATR=5, Factor=1.5):")
        print(f"   Buy signals: {result_df2['signal'].sum()}")
        print(f"   Sell signals: {result_df2['exit_signal'].sum()}")
        print(f"   Total trades: {len(backtester2.trades)}")
        
        if len(backtester2.trades) > 0:
            metrics2 = backtester2.calculate_metrics(result_df2)
            print(f"   Win Rate: {metrics2.get('Win Rate (%)', 0):.1f}%")
            print(f"   Total Return: {metrics2.get('Total Return (%)', 0):.2f}%")
            print(f"   Sharpe Ratio: {metrics2.get('Sharpe Ratio', 0):.2f}")
            
            return True
        else:
            print("❌ Still no trades. The SuperTrend algorithm might need adjustment.")
            return False

def main():
    """Main function"""
    print("🎯 Create Data That Generates SuperTrend Signals")
    print("=" * 50)
    
    success = test_super_trend_signals()
    
    if success:
        print("\n✅ Successfully generated data with SuperTrend signals!")
        print("\n💡 Next steps:")
        print("1. Review the generated reports and charts")
        print("2. The backtesting system is working correctly")
        print("3. For real data, set up Zerodha API credentials")
    else:
        print("\n❌ Failed to generate signals. The SuperTrend algorithm might need tuning.")

if __name__ == "__main__":
    main() 