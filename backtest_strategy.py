"""
SuperTrend Strategy Backtesting System
=====================================

A comprehensive backtesting framework for the SuperTrend trading strategy
with detailed performance analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SuperTrendBacktester:
    def __init__(self, initial_capital=10000, leverage=1.0, stop_loss=100, 
                 commission_per_trade=20, atr_period=10, factor=3.0):
        """
        Initialize backtester with strategy parameters
        
        Args:
            initial_capital: Starting capital
            leverage: Maximum leverage to use
            stop_loss: Fixed stop loss in rupees
            commission_per_trade: Brokerage per trade
            atr_period: ATR period for SuperTrend
            factor: SuperTrend factor
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.commission_per_trade = commission_per_trade
        self.atr_period = atr_period
        self.factor = factor
        
        # Trading state
        self.position = 0  # 0: No position, 1: Long
        self.entry_price = 0
        self.shares = 0
        self.capital = initial_capital
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def calculate_supertrend(self, df):
        """Calculate SuperTrend indicator with improved signal generation"""
        df = df.copy()
        
        # Calculate ATR
        df['h_l'] = df['high'] - df['low']
        df['h_c'] = abs(df['high'] - df['close'].shift(1))
        df['l_c'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h_l', 'h_c', 'l_c']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Calculate basic upper and lower bands
        df['hl2'] = (df['high'] + df['low']) / 2
        df['upper_band'] = df['hl2'] + (self.factor * df['atr'])
        df['lower_band'] = df['hl2'] - (self.factor * df['atr'])
        
        # Initialize SuperTrend columns
        df['final_upper_band'] = df['upper_band']
        df['final_lower_band'] = df['lower_band']
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 1
        
        # Calculate SuperTrend properly
        for i in range(1, len(df)):
            # Final Upper Band
            if (df['upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or 
                df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('final_upper_band')] = df['upper_band'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('final_upper_band')] = df['final_upper_band'].iloc[i-1]
            
            # Final Lower Band
            if (df['lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or 
                df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('final_lower_band')] = df['lower_band'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('final_lower_band')] = df['final_lower_band'].iloc[i-1]
            
            # SuperTrend
            if (df['close'].iloc[i] <= df['final_lower_band'].iloc[i] and 
                df['supertrend_direction'].iloc[i-1] == 1):
                df.iloc[i, df.columns.get_loc('supertrend')] = df['final_lower_band'].iloc[i]
                df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
            elif (df['close'].iloc[i] >= df['final_upper_band'].iloc[i] and 
                  df['supertrend_direction'].iloc[i-1] == -1):
                df.iloc[i, df.columns.get_loc('supertrend')] = df['final_upper_band'].iloc[i]
                df.iloc[i, df.columns.get_loc('supertrend_direction')] = 1
            else:
                df.iloc[i, df.columns.get_loc('supertrend')] = df['supertrend'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('supertrend_direction')] = df['supertrend_direction'].iloc[i-1]
        
        # Generate trading signals
        df['signal'] = 0
        df['exit_signal'] = 0
        
        # Buy signal: trend changes from down to up
        for i in range(1, len(df)):
            if (df['supertrend_direction'].iloc[i] == 1 and 
                df['supertrend_direction'].iloc[i-1] == -1):
                df.iloc[i, df.columns.get_loc('signal')] = 1
            
            # Sell signal: trend changes from up to down  
            if (df['supertrend_direction'].iloc[i] == -1 and 
                df['supertrend_direction'].iloc[i-1] == 1):
                df.iloc[i, df.columns.get_loc('exit_signal')] = 1
        
        # Print signal summary
        buy_signals = df['signal'].sum()
        sell_signals = df['exit_signal'].sum()
        print(f"SuperTrend calculated: {buy_signals} buy signals, {sell_signals} sell signals")
        
        return df
    
    def calculate_position_size(self, price, available_capital):
        """Calculate position size based on capital and leverage"""
        max_investment = available_capital * self.leverage
        shares = int(max_investment / price)
        actual_investment = shares * price
        
        return shares, actual_investment
    
    def run_backtest(self, df):
        """Run the complete backtest"""
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Initialize tracking
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        equity_values = [self.initial_capital]
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Add current equity to curve
            if self.position == 1:
                current_equity = self.capital + (self.shares * current_price) - (self.shares * self.entry_price)
            else:
                current_equity = self.capital
            equity_values.append(current_equity)
            
            # Check for stop loss if in position
            if self.position == 1:
                current_pnl = (current_price - self.entry_price) * self.shares
                if current_pnl <= -self.stop_loss:
                    # Stop loss hit
                    exit_price = current_price
                    pnl = (exit_price - self.entry_price) * self.shares - (2 * self.commission_per_trade)
                    
                    self.trades.append({
                        'entry_date': self.entry_date,
                        'exit_date': current_date,
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'shares': self.shares,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss',
                        'capital_before': self.capital,
                        'capital_after': self.capital + pnl
                    })
                    
                    self.capital += pnl
                    self.position = 0
                    self.shares = 0
                    continue
            
            # Check for buy signal
            if df['signal'].iloc[i] == 1 and self.position == 0:
                self.shares, investment = self.calculate_position_size(current_price, self.capital)
                if self.shares > 0:
                    self.entry_price = current_price
                    self.entry_date = current_date
                    self.position = 1
                    self.capital -= investment + self.commission_per_trade
            
            # Check for sell signal
            elif df['exit_signal'].iloc[i] == 1 and self.position == 1:
                exit_price = current_price
                pnl = (exit_price - self.entry_price) * self.shares - self.commission_per_trade
                
                self.trades.append({
                    'entry_date': self.entry_date,
                    'exit_date': current_date,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'shares': self.shares,
                    'pnl': pnl,
                    'exit_reason': 'SuperTrend Exit',
                    'capital_before': self.capital,
                    'capital_after': self.capital + pnl + (self.shares * exit_price)
                })
                
                self.capital += pnl + (self.shares * exit_price)
                self.position = 0
                self.shares = 0
        
        # Store equity curve
        self.equity_curve = equity_values
        
        # Calculate daily returns
        df['equity'] = equity_values
        df['daily_return'] = df['equity'].pct_change()
        self.daily_returns = df['daily_return'].dropna()
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Risk metrics
        returns = self.daily_returns
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown calculation
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Trade duration
        trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_trade_duration = trades_df['duration'].mean()
        
        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Total P&L': round(total_pnl, 2),
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Average P&L per Trade': round(avg_pnl, 2),
            'Average Win': round(avg_win, 2),
            'Average Loss': round(avg_loss, 2),
            'Profit Factor': round(profit_factor, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Average Trade Duration (days)': round(avg_trade_duration, 2),
            'Final Capital': round(self.equity_curve[-1], 2)
        }
        
        return metrics
    
    def plot_results(self, df, save_path=None):
        """Create comprehensive visualization of backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SuperTrend Strategy Backtest Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and SuperTrend
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
        ax1.plot(df.index, df['supertrend'], label='SuperTrend', linewidth=1.5)
        
        # Mark buy/sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['exit_signal'] == 1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], color='green', 
                   marker='^', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['close'], color='red', 
                   marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Price Chart with SuperTrend Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[0, 1]
        equity_dates = df.index[:len(self.equity_curve)]
        ax2.plot(equity_dates, self.equity_curve, color='blue', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade P&L Distribution
        ax3 = axes[1, 0]
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('P&L per Trade')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        ax4 = axes[1, 1]
        equity_series = pd.Series(self.equity_curve, index=equity_dates)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        ax4.fill_between(equity_dates, drawdown, 0, color='red', alpha=0.3)
        ax4.plot(equity_dates, drawdown, color='red', linewidth=1)
        ax4.set_title('Drawdown Chart')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, df, save_path=None):
        """Generate comprehensive backtest report"""
        metrics = self.calculate_metrics(df)
        
        if "error" in metrics:
            print(metrics["error"])
            return
        
        # Print metrics
        print("=" * 60)
        print("SUPERTREND STRATEGY BACKTEST REPORT")
        print("=" * 60)
        print(f"Backtest Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Strategy Parameters: ATR Period={self.atr_period}, Factor={self.factor}")
        print(f"Risk Management: Stop Loss=₹{self.stop_loss}, Leverage={self.leverage}x")
        print("-" * 60)
        
        # Performance metrics
        print("PERFORMANCE METRICS:")
        for key, value in metrics.items():
            print(f"{key:.<30} {value}")
        
        print("-" * 60)
        
        # Trade analysis
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            
            print("TRADE ANALYSIS:")
            print(f"Best Trade: ₹{trades_df['pnl'].max():.2f}")
            print(f"Worst Trade: ₹{trades_df['pnl'].min():.2f}")
            print(f"Consecutive Wins: {self._max_consecutive_wins(trades_df)}")
            print(f"Consecutive Losses: {self._max_consecutive_losses(trades_df)}")
            
            # Monthly performance
            trades_df['month'] = trades_df['exit_date'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['pnl'].sum()
            print(f"Best Month: ₹{monthly_pnl.max():.2f} ({monthly_pnl.idxmax()})")
            print(f"Worst Month: ₹{monthly_pnl.min():.2f} ({monthly_pnl.idxmin()})")
        
        print("=" * 60)
        
        # Save detailed trade log
        if save_path and len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(save_path.replace('.txt', '_trades.csv'), index=False)
            print(f"Detailed trade log saved to: {save_path.replace('.txt', '_trades.csv')}")
    
    def _max_consecutive_wins(self, trades_df):
        """Calculate maximum consecutive winning trades"""
        wins = (trades_df['pnl'] > 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _max_consecutive_losses(self, trades_df):
        """Calculate maximum consecutive losing trades"""
        losses = (trades_df['pnl'] <= 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


def create_sample_data():
    """Create realistic sample data with trending behavior for testing"""
    # Create 1 year of daily data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # Generate realistic NIFTYBEES-like data with trends
    np.random.seed(42)
    price = 280.0  # Start around NIFTYBEES price
    data = []
    
    trend_direction = 1  # Start with uptrend
    trend_strength = 0.02
    days_in_trend = 0
    
    for i, date in enumerate(dates):
        # Create trending behavior with reversals
        if days_in_trend > 20 + np.random.randint(0, 30):  # Trend lasts 20-50 days
            trend_direction *= -1  # Reverse trend
            days_in_trend = 0
            trend_strength = np.random.uniform(0.01, 0.04)  # Random trend strength
        
        # Calculate price movement
        trend_move = trend_direction * trend_strength
        noise = np.random.normal(0, 0.015)  # Daily volatility ~1.5%
        daily_change = trend_move + noise
        
        # Update price
        new_price = price * (1 + daily_change)
        new_price = max(new_price, 200)  # Floor price
        new_price = min(new_price, 400)  # Ceiling price
        
        # Generate OHLC based on daily change
        if daily_change > 0:  # Up day
            open_price = price + np.random.uniform(-0.5, 0.5)
            close = new_price + np.random.uniform(-0.3, 0.3)
            high = max(open_price, close) + abs(np.random.normal(0, 0.8))
            low = min(open_price, close) - abs(np.random.normal(0, 0.5))
        else:  # Down day
            open_price = price + np.random.uniform(-0.5, 0.5)
            close = new_price + np.random.uniform(-0.3, 0.3)
            high = max(open_price, close) + abs(np.random.normal(0, 0.5))
            low = min(open_price, close) - abs(np.random.normal(0, 0.8))
        
        # Ensure OHLC logic is correct
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': np.random.randint(50000, 200000)
        })
        
        price = close
        days_in_trend += 1
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Remove weekends (make it realistic)
    df = df[df.index.weekday < 5]
    
    print(f"Sample data generated: {len(df)} trading days")
    print(f"Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    print(f"Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
    
    return df


def load_data_from_kite(instrument_token, from_date, to_date):
    """
    Load historical data from Kite Connect API
    Replace this with your actual data loading implementation
    """
    # This is a placeholder - implement your Kite data loading here
    print("Loading data from Kite Connect...")
    print(f"Instrument: {instrument_token}")
    print(f"Period: {from_date} to {to_date}")
    
    # For now, return sample data
    return create_sample_data()


def main():
    """Main function to run backtest"""
    print("SuperTrend Strategy Backtester")
    print("=" * 50)
    
    # Configuration
    config = {
        'initial_capital': 10000,
        'leverage': 5.0,  # 5x leverage like NIFTYBEES
        'stop_loss': 100,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 3.0
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data (replace with your data source)
    print("Loading historical data...")
    df = create_sample_data()  # Replace with load_data_from_kite()
    print(f"Data loaded: {len(df)} days from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    # Initialize backtester
    backtester = SuperTrendBacktester(**config)
    
    # Run backtest
    print("Running backtest...")
    result_df = backtester.run_backtest(df)
    print()
    
    # Check if we have trades
    if len(backtester.trades) == 0:
        print("❌ No trades executed!")
        print("\nDebugging information:")
        print(f"ATR Period: {config['atr_period']}")
        print(f"Factor: {config['factor']}")
        
        # Show some sample calculations
        sample_atr = result_df['atr'].dropna().iloc[-10:].mean()
        sample_price = result_df['close'].iloc[-1]
        band_width = config['factor'] * sample_atr
        
        print(f"Recent average ATR: ₹{sample_atr:.2f}")
        print(f"Current price: ₹{sample_price:.2f}")
        print(f"Band width: ₹{band_width:.2f} ({band_width/sample_price*100:.1f}% of price)")
        
        # Try with more sensitive parameters
        print("\n🔧 Trying with more sensitive parameters...")
        sensitive_config = config.copy()
        sensitive_config['factor'] = 2.0  # More sensitive
        
        sensitive_backtester = SuperTrendBacktester(**sensitive_config)
        sensitive_result = sensitive_backtester.run_backtest(df.copy())
        
        if len(sensitive_backtester.trades) > 0:
            print(f"✅ With factor=2.0: {len(sensitive_backtester.trades)} trades executed")
            sensitive_backtester.generate_report(sensitive_result, 'sensitive_backtest_report.txt')
            sensitive_backtester.plot_results(sensitive_result, 'sensitive_backtest_charts.png')
        else:
            print("❌ Still no trades with sensitive parameters")
            
            # Show signal analysis
            signals = result_df['signal'].sum()
            exit_signals = result_df['exit_signal'].sum()
            print(f"Buy signals in data: {signals}")
            print(f"Sell signals in data: {exit_signals}")
            
            if signals == 0:
                print("\n💡 Suggestions:")
                print("- Try lower factor (1.5-2.5) for more signals")
                print("- Try shorter ATR period (5-7) for more sensitivity")
                print("- Check if data has sufficient volatility and trends")
        
        return
    
    # Generate report
    print("Generating comprehensive report...")
    backtester.generate_report(result_df, 'backtest_report.txt')
    
    # Plot results
    print("Generating charts...")
    backtester.plot_results(result_df, 'backtest_charts.png')
    
    print("\n🎉 Backtest completed successfully!")
    print(f"📊 {len(backtester.trades)} trades executed")
    print(f"💰 Final portfolio value: ₹{backtester.equity_curve[-1]:,.2f}")
    print(f"📈 Total return: {((backtester.equity_curve[-1] - config['initial_capital']) / config['initial_capital'] * 100):.2f}%")
    
    # Show top 3 winning and losing trades
    if len(backtester.trades) > 0:
        trades_df = pd.DataFrame(backtester.trades)
        
        print("\n🏆 Top 3 Winning Trades:")
        top_wins = trades_df.nlargest(3, 'pnl')
        for _, trade in top_wins.iterrows():
            print(f"  ₹{trade['pnl']:.2f} - {trade['entry_date'].strftime('%Y-%m-%d')} to {trade['exit_date'].strftime('%Y-%m-%d')}")
        
        print("\n📉 Top 3 Losing Trades:")
        top_losses = trades_df.nsmallest(3, 'pnl')
        for _, trade in top_losses.iterrows():
            print(f"  ₹{trade['pnl']:.2f} - {trade['entry_date'].strftime('%Y-%m-%d')} to {trade['exit_date'].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()