#!/usr/bin/env python3
"""
Generate Test Data for SuperTrend Backtesting
============================================

This script generates realistic market data that will produce SuperTrend signals
for testing the backtesting system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_trending_data(days=100, base_price=300, volatility=0.02, trend_strength=0.001):
    """Generate trending market data that will produce SuperTrend signals"""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with trend
    np.random.seed(42)  # For reproducible results
    
    # Create a trending price series
    trend = np.linspace(0, trend_strength * days, days)
    noise = np.random.normal(0, volatility, days)
    
    # Generate OHLC data
    prices = base_price * (1 + trend + noise)
    
    # Create realistic OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_volatility = volatility * 0.5
        
        # High and low based on close
        high = price * (1 + abs(np.random.normal(0, daily_volatility)))
        low = price * (1 - abs(np.random.normal(0, daily_volatility)))
        
        # Ensure high >= close >= low
        high = max(high, price)
        low = min(low, price)
        
        # Open price (previous close with some gap)
        if i == 0:
            open_price = price * (1 + np.random.normal(0, daily_volatility * 0.5))
        else:
            open_price = data[i-1]['close'] * (1 + np.random.normal(0, daily_volatility * 0.3))
        
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

def generate_volatile_data(days=100, base_price=300, volatility=0.03):
    """Generate volatile market data with frequent trend changes"""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(123)  # For reproducible results
    
    # Create volatile price series with trend changes
    prices = [base_price]
    
    for i in range(1, days):
        # Add trend changes every 20-30 days
        if i % np.random.randint(20, 30) == 0:
            trend_direction = np.random.choice([-1, 1])
        
        # Generate price movement
        movement = np.random.normal(0, volatility)
        if 'trend_direction' in locals():
            movement += trend_direction * volatility * 0.5
        
        new_price = prices[-1] * (1 + movement)
        prices.append(new_price)
    
    # Create realistic OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_volatility = volatility * 0.8
        
        # High and low based on close
        high = price * (1 + abs(np.random.normal(0, daily_volatility)))
        low = price * (1 - abs(np.random.normal(0, daily_volatility)))
        
        # Ensure high >= close >= low
        high = max(high, price)
        low = min(low, price)
        
        # Open price
        if i == 0:
            open_price = price * (1 + np.random.normal(0, daily_volatility * 0.5))
        else:
            open_price = data[i-1]['close'] * (1 + np.random.normal(0, daily_volatility * 0.3))
        
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

def generate_realistic_nifty_data(days=100):
    """Generate realistic NIFTYBEES-like data"""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(456)  # For reproducible results
    
    # NIFTYBEES characteristics
    base_price = 280  # Typical NIFTYBEES price
    daily_volatility = 0.015  # ~1.5% daily volatility
    trend_changes = 3  # Number of major trend changes
    
    # Create price series with realistic movements
    prices = [base_price]
    trend_periods = days // trend_changes
    
    for i in range(1, days):
        # Determine trend direction based on period
        period = i // trend_periods
        if period == 0:
            trend = 0.0005  # Slight uptrend
        elif period == 1:
            trend = -0.0003  # Downtrend
        else:
            trend = 0.0008  # Strong uptrend
        
        # Add random movement
        movement = np.random.normal(trend, daily_volatility)
        new_price = prices[-1] * (1 + movement)
        prices.append(new_price)
    
    # Create realistic OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC
        daily_range = price * daily_volatility * 0.6
        
        high = price + abs(np.random.normal(0, daily_range * 0.5))
        low = price - abs(np.random.normal(0, daily_range * 0.5))
        
        # Ensure high >= close >= low
        high = max(high, price)
        low = min(low, price)
        
        # Open price
        if i == 0:
            open_price = price + np.random.normal(0, daily_range * 0.3)
        else:
            open_price = data[i-1]['close'] + np.random.normal(0, daily_range * 0.2)
        
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

def test_data_generation():
    """Test the data generation functions"""
    print("🧪 Testing Data Generation Functions")
    print("=" * 50)
    
    # Test trending data
    print("\n📈 Generating trending data...")
    trending_df = generate_trending_data(days=60)
    print(f"Trending data: {len(trending_df)} days, Price range: ₹{trending_df['close'].min():.2f} - ₹{trending_df['close'].max():.2f}")
    
    # Test volatile data
    print("\n📊 Generating volatile data...")
    volatile_df = generate_volatile_data(days=60)
    print(f"Volatile data: {len(volatile_df)} days, Price range: ₹{volatile_df['close'].min():.2f} - ₹{volatile_df['close'].max():.2f}")
    
    # Test realistic NIFTY data
    print("\n📈 Generating realistic NIFTY data...")
    nifty_df = generate_realistic_nifty_data(days=60)
    print(f"NIFTY data: {len(nifty_df)} days, Price range: ₹{nifty_df['close'].min():.2f} - ₹{nifty_df['close'].max():.2f}")
    
    # Save test data
    trending_df.to_csv('test_trending_data.csv')
    volatile_df.to_csv('test_volatile_data.csv')
    nifty_df.to_csv('test_nifty_data.csv')
    
    print(f"\n💾 Test data saved:")
    print(f"   test_trending_data.csv")
    print(f"   test_volatile_data.csv")
    print(f"   test_nifty_data.csv")
    
    return trending_df, volatile_df, nifty_df

def main():
    """Main function"""
    print("🎯 Generate Test Data for SuperTrend Backtesting")
    print("=" * 50)
    
    # Test data generation
    trending_df, volatile_df, nifty_df = test_data_generation()
    
    print(f"\n✅ Data generation completed!")
    print(f"\n💡 Next steps:")
    print(f"1. Use these datasets to test your backtesting system")
    print(f"2. Try different parameter combinations")
    print(f"3. Compare performance across different market conditions")

if __name__ == "__main__":
    main() 