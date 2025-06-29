"""
SuperTrend Strategy Backtesting System
======================================================

A comprehensive backtesting framework for the SuperTrend trading strategy
with proper signal detection and position tracking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import argparse
from trading.strategies.strategy_factory import StrategyFactory
warnings.filterwarnings('ignore')

class SuperTrendBacktester:
    def __init__(self, strategy, initial_capital=10000, leverage=1.0, stop_loss=100, 
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
        self.strategy = strategy
        
        # Trading state
        self.position = 0  # 0: No position, 1: Long
        self.entry_price = 0
        self.shares = 0
        self.capital = initial_capital
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def calculate_position_size(self, price, available_capital):
        """Calculate position size based on capital and leverage"""
        max_investment = available_capital * self.leverage
        shares = int(max_investment / price)
        actual_investment = shares * price
        
        return shares, actual_investment
    
    def run_backtest(self, df):
        """Run the complete backtest using the strategy's signal generation"""
        print(f"\nRunning backtest with {len(df)} data points...")
        
        # Initialize tracking
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        equity_values = [self.initial_capital]
        
        # Track signals for debugging
        all_signals = []
        
        # Main backtest loop
        for i in range(50, len(df)):  # Start after minimum required candles
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Get data up to current point (no lookahead bias)
            historical_data = df.iloc[:i+1].copy()
            
            # Get signal from strategy
            signal, signal_data = self.strategy.get_signal(
                historical_data, 
                has_position=(self.position == 1)
            )
            
            # Track all signals for debugging
            if signal != "HOLD":
                all_signals.append({
                    'date': current_date,
                    'signal': signal,
                    'price': current_price,
                    'confidence': signal_data.get('confidence', 0),
                    'position': self.position
                })
            
            # Add current equity to curve
            if self.position == 1:
                current_equity = self.capital + (self.shares * (current_price - self.entry_price))
            else:
                current_equity = self.capital
            equity_values.append(current_equity)
            
            # Check for stop loss if in position
            if self.position == 1:
                current_pnl = (current_price - self.entry_price) * self.shares
                if current_pnl <= -self.stop_loss:
                    # Stop loss hit
                    exit_price = current_price
                    pnl = (exit_price - self.entry_price) * self.shares - self.commission_per_trade
                    
                    self.trades.append({
                        'entry_date': self.entry_date,
                        'exit_date': current_date,
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'shares': self.shares,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss',
                        'capital_before': self.capital - (self.shares * self.entry_price),
                        'capital_after': self.capital - (self.shares * self.entry_price) + (self.shares * exit_price) + pnl
                    })
                    
                    # Update capital
                    self.capital = self.capital + (self.shares * exit_price) - (self.shares * self.entry_price) - self.commission_per_trade
                    self.position = 0
                    self.shares = 0
                    
                    date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
                    print(f"🛑 STOP LOSS: Exit at ₹{exit_price:.2f} on {date_str} | P&L: ₹{pnl:.2f}")
                    continue
            
            # Process trading signals
            if signal == "BUY" and self.position == 0:
                self.shares, investment = self.calculate_position_size(current_price, self.capital)
                
                if self.shares > 0:
                    self.entry_price = current_price
                    self.entry_date = current_date
                    self.position = 1
                    # FIXED: Don't deduct share value, only commission
                    self.capital = self.capital - self.commission_per_trade
                    
                    date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
                    print(f"📈 BUY: {self.shares} shares at ₹{current_price:.2f} on {date_str}")
            
            elif signal == "SELL" and self.position == 1:
                exit_price = current_price
                gross_pnl = (exit_price - self.entry_price) * self.shares
                net_pnl = gross_pnl - self.commission_per_trade
                
                self.trades.append({
                    'entry_date': self.entry_date,
                    'exit_date': current_date,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'shares': self.shares,
                    'pnl': net_pnl,
                    'exit_reason': 'Signal',
                    'capital_before': self.capital,
                    'capital_after': self.capital + net_pnl  # FIXED
                })
                
                # FIXED: Just add the P&L
                self.capital = self.capital + net_pnl
                self.position = 0
                self.shares = 0
                
                date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
                print(f"📉 SELL: {self.shares} shares at ₹{exit_price:.2f} on {date_str} | P&L: ₹{net_pnl:.2f}")
        
        # Close any open position at the end
        if self.position == 1:
            exit_price = df['close'].iloc[-1]
            exit_date = df.index[-1]
            gross_pnl = (exit_price - self.entry_price) * self.shares
            net_pnl = gross_pnl - self.commission_per_trade
            
            self.trades.append({
                'entry_date': self.entry_date,
                'exit_date': exit_date,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'shares': self.shares,
                'pnl': net_pnl,
                'exit_reason': 'End of Data',
                'capital_before': self.capital,
                'capital_after': self.capital + (self.shares * exit_price) - self.commission_per_trade
            })
            
            self.capital = self.capital + (self.shares * exit_price) - self.commission_per_trade
            print(f"📊 END: Closing position at ₹{exit_price:.2f} | P&L: ₹{net_pnl:.2f}")
        
        # Store equity curve
        self.equity_curve = equity_values[:len(df)-49]  # Adjust for starting point
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            equity_series = pd.Series(self.equity_curve)
            self.daily_returns = equity_series.pct_change().dropna()
        
        print(f"\nBacktest completed: {len(self.trades)} trades executed")
        
        # Debug information
        if len(all_signals) > len(self.trades):
            print(f"\n⚠️  Note: {len(all_signals)} signals detected, {len(self.trades)} trades executed")
            unexecuted = len(all_signals) - len(self.trades)
            if unexecuted > 0:
                print(f"   Some signals were filtered out (already in position or no capital)")
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            volatility = self.daily_returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.06  # 6% annual risk-free rate
            daily_rf = risk_free_rate / 252
            excess_returns = self.daily_returns - daily_rf
            sharpe_ratio = (excess_returns.mean() * 252) / (self.daily_returns.std() * np.sqrt(252)) if self.daily_returns.std() > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown calculation
        if len(self.equity_curve) > 0:
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
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
            'Final Capital': round(self.capital, 2)
        }
        
        return metrics
    
    def plot_results(self, df, save_path=None):
        """Create comprehensive visualization of backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SuperTrend Strategy Backtest Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and trades
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, alpha=0.7)
        
        # Mark entry and exit points
        if self.trades:
            for trade in self.trades:
                # Entry point
                ax1.scatter(trade['entry_date'], trade['entry_price'], 
                           color='green', marker='^', s=100, zorder=5)
                # Exit point
                ax1.scatter(trade['exit_date'], trade['exit_price'], 
                           color='red', marker='v', s=100, zorder=5)
                # Connect entry and exit
                ax1.plot([trade['entry_date'], trade['exit_date']], 
                        [trade['entry_price'], trade['exit_price']], 
                        'k--', alpha=0.3, linewidth=1)
        
        ax1.set_title('Price Chart with Trade Entry/Exit Points')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[0, 1]
        if len(self.equity_curve) > 0:
            # Ensure we have the correct number of dates
            start_idx = 50  # Start after minimum required candles
            end_idx = start_idx + len(self.equity_curve)
            
            # Make sure we don't exceed the dataframe length
            if end_idx <= len(df):
                equity_dates = df.index[start_idx:end_idx]
                if len(equity_dates) == len(self.equity_curve):
                    ax2.plot(equity_dates, self.equity_curve, color='blue', linewidth=2)
                    ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
                    ax2.set_title('Portfolio Equity Curve')
                    ax2.set_ylabel('Portfolio Value (₹)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    # Fallback: plot with simple index
                    ax2.plot(range(len(self.equity_curve)), self.equity_curve, color='blue', linewidth=2)
                    ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
                    ax2.set_title('Portfolio Equity Curve (Index)')
                    ax2.set_ylabel('Portfolio Value (₹)')
                    ax2.set_xlabel('Trading Day')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            else:
                # Fallback: plot with simple index
                ax2.plot(range(len(self.equity_curve)), self.equity_curve, color='blue', linewidth=2)
                ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
                ax2.set_title('Portfolio Equity Curve (Index)')
                ax2.set_ylabel('Portfolio Value (₹)')
                ax2.set_xlabel('Trading Day')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade P&L Distribution
        ax3 = axes[1, 0]
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            pnl_values = trades_df['pnl'].values
            
            # Create histogram
            n, bins, patches = ax3.hist(pnl_values, bins=20, alpha=0.7, edgecolor='black')
            
            # Color code the bars
            for i, patch in enumerate(patches):
                if bins[i] >= 0:
                    patch.set_facecolor('green')
                else:
                    patch.set_facecolor('red')
            
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('P&L per Trade (₹)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative P&L
        ax4 = axes[1, 1]
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            ax4.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 
                    color='purple', linewidth=2, marker='o')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax4.set_title('Cumulative P&L')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative P&L (₹)')
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
            if len(monthly_pnl) > 0:
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
    
    # Remove weekends
    df = df[[d.weekday() < 5 for d in df.index]]
    
    print(f"Sample data generated: {len(df)} trading days")
    print(f"Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    
    return df


def main():
    """Main function to run backtest"""
    print("SuperTrend Strategy Backtester")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="SuperTrend Strategy Backtester")
    parser.add_argument('--csv', type=str, help='Path to historical OHLC CSV file')
    parser.add_argument('--strategy', type=str, default='enhanced', help='Strategy key to use')
    parser.add_argument('--list-strategies', action='store_true', help='List all available strategies')
    parser.add_argument('--sample-data', action='store_true', help='Use sample data for testing')
    args = parser.parse_args()

    if args.list_strategies:
        print("\nAvailable strategies:")
        strategies = StrategyFactory.list_strategies()
        for key, info in strategies.items():
            print(f"  {key}: {info['name']} - {info['description']}")
        return

    # Configuration
    config = {
        'initial_capital': 10000,
        'leverage': 5.0,
        'stop_loss': 2000,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 3.0
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load data
    if args.sample_data:
        print("Using sample data for testing...")
        df = create_sample_data()
    elif args.csv:
        print(f"Loading historical data from CSV: {args.csv}")
        df = pd.read_csv(args.csv, parse_dates=['date'])
        df.drop_duplicates(subset='date', inplace=True)
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        df.set_index('date', inplace=True)
        print(f"Data loaded: {len(df)} rows from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    else:
        print("No data source specified. Use --csv or --sample-data")
        return

    # Select strategy
    try:
        strategy = StrategyFactory.create_strategy(args.strategy)
        print(f"Using strategy: {args.strategy}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Use --list-strategies to see available options.")
        return

    # Initialize backtester
    backtester = SuperTrendBacktester(strategy, **config)

    # Run backtest
    result_df = backtester.run_backtest(df)
    print()

    # Generate report and charts
    print("Generating comprehensive report...")
    backtester.generate_report(result_df, 'backtest_report.txt')
    print("Generating charts...")
    backtester.plot_results(result_df, 'backtest_charts.png')
    
    print("\n🎉 Backtest completed successfully!")


if __name__ == "__main__":
    main()