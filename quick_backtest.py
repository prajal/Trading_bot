#!/usr/bin/env python3
"""
Quick SuperTrend Strategy Backtest with Sample Data
==================================================

This script runs a quick backtest using sample data so you can see the system working
without needing to authenticate with Kite Connect.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_strategy import SuperTrendBacktester, create_sample_data

def run_quick_backtest():
    """Run quick backtest with sample data"""
    print("📊 QUICK SUPERTREND STRATEGY BACKTEST")
    print("=" * 50)
    print("Using sample data (no authentication required)")
    
    # Create sample data
    print("📥 Generating sample market data...")
    df = create_sample_data()
    
    print(f"✅ Generated {len(df)} data points")
    print(f"📊 Price Range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    print(f"📅 Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Configure backtest parameters
    config = {
        'initial_capital': 4000.0,  # Match your live trading capital
        'leverage': 5.0,  # NIFTYBEES leverage
        'stop_loss': 100.0,  # Fixed stop loss
        'commission_per_trade': 20.0,  # Brokerage
        'atr_period': 10,
        'factor': 3.0
    }
    
    print("\n⚙️  Backtest Configuration:")
    print(f"   Initial Capital: ₹{config['initial_capital']:,.2f}")
    print(f"   Leverage: {config['leverage']}x")
    print(f"   Stop Loss: ₹{config['stop_loss']}")
    print(f"   ATR Period: {config['atr_period']}")
    print(f"   Factor: {config['factor']}")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    backtester = SuperTrendBacktester(**config)
    result_df = backtester.run_backtest(df)
    
    # Calculate and display metrics
    if len(backtester.trades) > 0:
        metrics = backtester.calculate_metrics(result_df)
        
        print("\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Trades: {metrics.get('Total Trades', 0)}")
        print(f"Winning Trades: {metrics.get('Winning Trades', 0)}")
        print(f"Win Rate: {metrics.get('Win Rate (%)', 0):.1f}%")
        print(f"Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
        print(f"Final Capital: ₹{backtester.equity_curve[-1]:,.2f}")
        print(f"Max Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"Average Trade P&L: ₹{metrics.get('Average Trade P&L', 0):.2f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"quick_backtest_report_{timestamp}.txt"
        chart_file = f"quick_backtest_charts_{timestamp}.png"
        
        backtester.generate_report(result_df, report_file)
        backtester.plot_results(result_df, chart_file)
        
        print(f"\n💾 Results saved:")
        print(f"   Report: {report_file}")
        print(f"   Charts: {chart_file}")
        
        # Show trade details
        if len(backtester.trades) > 0:
            print(f"\n📋 TRADE DETAILS")
            print("-" * 50)
            trades_df = pd.DataFrame(backtester.trades)
            for i, trade in trades_df.iterrows():
                entry_date = pd.to_datetime(trade['entry_date']).strftime('%Y-%m-%d')
                exit_date = pd.to_datetime(trade['exit_date']).strftime('%Y-%m-%d')
                print(f"Trade {i+1}: {entry_date} → {exit_date}")
                print(f"  Entry: ₹{trade['entry_price']:.2f} | Exit: ₹{trade['exit_price']:.2f}")
                print(f"  P&L: ₹{trade['pnl']:.2f} | Reason: {trade['exit_reason']}")
                print()
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print("-" * 50)
        if metrics.get('Total Return (%)', 0) > 0:
            print("✅ Strategy was profitable in the backtest period")
        else:
            print("❌ Strategy was not profitable in the backtest period")
        
        if metrics.get('Win Rate (%)', 0) > 50:
            print("✅ Win rate is above 50%")
        else:
            print("⚠️  Win rate is below 50%")
        
        if metrics.get('Max Drawdown (%)', 0) < 10:
            print("✅ Maximum drawdown is acceptable (< 10%)")
        else:
            print("⚠️  Maximum drawdown is high (> 10%)")
        
        if metrics.get('Sharpe Ratio', 0) > 1.0:
            print("✅ Sharpe ratio indicates good risk-adjusted returns")
        else:
            print("⚠️  Sharpe ratio indicates poor risk-adjusted returns")
            
    else:
        print("\n❌ No trades executed during the backtest period")
        print("💡 This could mean:")
        print("   - No SuperTrend signals generated")
        print("   - Market conditions not suitable for the strategy")
        print("   - Try adjusting ATR period or factor")
    
    return True

def test_different_parameters():
    """Test different parameter combinations"""
    print("\n🔬 TESTING DIFFERENT PARAMETERS")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Test different configurations
    configs = [
        {'atr_period': 7, 'factor': 2.0, 'name': 'Sensitive (ATR=7, Factor=2.0)'},
        {'atr_period': 10, 'factor': 3.0, 'name': 'Default (ATR=10, Factor=3.0)'},
        {'atr_period': 14, 'factor': 4.0, 'name': 'Conservative (ATR=14, Factor=4.0)'},
    ]
    
    base_config = {
        'initial_capital': 4000.0,
        'leverage': 5.0,
        'stop_loss': 100.0,
        'commission_per_trade': 20.0,
    }
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 30)
        
        # Combine base config with test config
        test_config = base_config.copy()
        test_config.update({
            'atr_period': config['atr_period'],
            'factor': config['factor']
        })
        
        # Run backtest
        backtester = SuperTrendBacktester(**test_config)
        result_df = backtester.run_backtest(df.copy())
        
        if len(backtester.trades) > 0:
            metrics = backtester.calculate_metrics(result_df)
            
            print(f"  Trades: {metrics.get('Total Trades', 0)}")
            print(f"  Win Rate: {metrics.get('Win Rate (%)', 0):.1f}%")
            print(f"  Return: {metrics.get('Total Return (%)', 0):.2f}%")
            print(f"  Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
            
            results.append({
                'name': config['name'],
                'trades': metrics.get('Total Trades', 0),
                'win_rate': metrics.get('Win Rate (%)', 0),
                'return': metrics.get('Total Return (%)', 0),
                'sharpe': metrics.get('Sharpe Ratio', 0)
            })
        else:
            print(f"  No trades executed")
            results.append({
                'name': config['name'],
                'trades': 0,
                'win_rate': 0,
                'return': 0,
                'sharpe': 0
            })
    
    # Summary
    print(f"\n📊 PARAMETER COMPARISON SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"{result['name']}:")
        print(f"  Trades: {result['trades']}, Win Rate: {result['win_rate']:.1f}%, Return: {result['return']:.2f}%, Sharpe: {result['sharpe']:.2f}")
        print()

def main():
    """Main function"""
    print("🎯 Quick SuperTrend Strategy Backtest")
    print("=" * 50)
    
    # Run quick backtest
    success = run_quick_backtest()
    
    if success:
        print("\n✅ Quick backtest completed successfully!")
        
        # Ask if user wants to test different parameters
        print("\n🔬 Would you like to test different parameter combinations?")
        print("This will help you understand how ATR period and factor affect performance.")
        
        # For now, just run it automatically
        test_different_parameters()
        
    else:
        print("\n❌ Quick backtest failed!")
    
    print("\n💡 Next steps:")
    print("1. Review the generated report and charts")
    print("2. Compare different parameter combinations")
    print("3. When ready for real data, authenticate with: python cli.py auth")
    print("4. Then run: python run_backtest.py")

if __name__ == "__main__":
    main() 