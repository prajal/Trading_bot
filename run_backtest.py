#!/usr/bin/env python3
"""
30-Day SuperTrend Strategy Backtest
===================================

This script runs a 30-day backtest of the SuperTrend strategy using real market data
from Zerodha Kite Connect API.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth.kite_auth import KiteAuth
from backtest_strategy import SuperTrendBacktester

def run_30_day_backtest():
    """Run 30-day backtest with real data"""
    print("📊 SUPERTREND STRATEGY BACKTEST")
    print("=" * 50)
    
    # Check authentication
    auth = KiteAuth()
    if not auth.test_connection():
        print("❌ Please authenticate first: python cli.py auth")
        return False
    
    kite = auth.get_kite_instance()
    if not kite:
        print("❌ Failed to get Kite instance")
        return False
    
    # Calculate date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("📈 Instrument: NIFTYBEES (2707457)")
    
    try:
        # Fetch historical data
        print("📥 Fetching historical data...")
        historical_data = kite.historical_data(
            instrument_token="2707457",  # NIFTYBEES
            from_date=start_date,
            to_date=end_date,
            interval="day"
        )
        
        if not historical_data:
            print("❌ No data received from Kite")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename columns to match expected format
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
        
        print(f"✅ Loaded {len(df)} data points")
        print(f"📊 Price Range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
        
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
            report_file = f"backtest_report_{timestamp}.txt"
            chart_file = f"backtest_charts_{timestamp}.png"
            
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
            print("\n💡 Suggestions:")
            print("   - Try a longer period (60-90 days)")
            print("   - Adjust ATR period (try 7 or 14)")
            print("   - Adjust factor (try 2.0 or 4.0)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎯 30-Day SuperTrend Strategy Backtest")
    print("=" * 50)
    
    success = run_30_day_backtest()
    
    if success:
        print("\n✅ Backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")
    
    print("\n💡 Next steps:")
    print("1. Review the generated report and charts")
    print("2. Adjust parameters if needed")
    print("3. Run with different time periods")
    print("4. Test with different instruments")

if __name__ == "__main__":
    main() 