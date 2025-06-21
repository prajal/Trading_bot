"""
Backtest Runner for SuperTrend Strategy
======================================

This script provides easy-to-use functions to run backtests with real Kite data
and different configurations for comprehensive strategy analysis.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
try:
    from auth.kite_auth import KiteAuth
    from trading.data_manager import DataManager
    HAS_KITE_MODULES = True
except ImportError:
    print("Warning: Kite modules not found. Using sample data only.")
    HAS_KITE_MODULES = False

from backtest_strategy import SuperTrendBacktester, create_sample_data


class BacktestRunner:
    """Enhanced backtest runner with real data integration"""
    
    def __init__(self):
        self.kite = None
        if HAS_KITE_MODULES:
            try:
                auth = KiteAuth()
                self.kite = auth.get_kite_instance()
                print("✅ Connected to Kite Connect")
            except Exception as e:
                print(f"❌ Kite connection failed: {e}")
                print("Will use sample data for backtesting")
    
    def load_real_data(self, instrument_token, from_date, to_date, interval="day"):
        """Load real historical data from Kite Connect"""
        if not self.kite:
            print("Using sample data (Kite not connected)")
            return create_sample_data()
        
        try:
            # Convert string dates to datetime if needed
            if isinstance(from_date, str):
                from_date = datetime.strptime(from_date, "%Y-%m-%d")
            if isinstance(to_date, str):
                to_date = datetime.strptime(to_date, "%Y-%m-%d")
            
            print(f"Loading data for instrument {instrument_token}...")
            print(f"Period: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
            
            # Fetch historical data
            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not historical_data:
                raise Exception("No data received from Kite")
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match expected format
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
            
            print(f"✅ Loaded {len(df)} data points")
            return df
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print("Falling back to sample data")
            return create_sample_data()
    
    def run_single_backtest(self, config, data_config):
        """Run a single backtest with given configuration"""
        
        # Load data
        df = self.load_real_data(**data_config)
        
        # Initialize backtester
        backtester = SuperTrendBacktester(**config)
        
        # Run backtest
        result_df = backtester.run_backtest(df)
        
        return backtester, result_df
    
    def run_parameter_optimization(self, base_config, data_config, param_ranges):
        """Run backtests across different parameter combinations"""
        
        print("Starting Parameter Optimization...")
        print("=" * 50)
        
        results = []
        total_combinations = 1
        for param_range in param_ranges.values():
            total_combinations *= len(param_range)
        
        current_test = 0
        
        # Test different ATR periods
        for atr_period in param_ranges.get('atr_periods', [10]):
            # Test different factors
            for factor in param_ranges.get('factors', [3.0]):
                # Test different stop losses
                for stop_loss in param_ranges.get('stop_losses', [100]):
                    
                    current_test += 1
                    print(f"Test {current_test}/{total_combinations}: ATR={atr_period}, Factor={factor}, SL={stop_loss}")
                    
                    # Create config for this test
                    test_config = base_config.copy()
                    test_config.update({
                        'atr_period': atr_period,
                        'factor': factor,
                        'stop_loss': stop_loss
                    })
                    
                    # Run backtest
                    backtester, result_df = self.run_single_backtest(test_config, data_config)
                    metrics = backtester.calculate_metrics(result_df)
                    
                    if "error" not in metrics:
                        result = {
                            'atr_period': atr_period,
                            'factor': factor,
                            'stop_loss': stop_loss,
                            **metrics
                        }
                        results.append(result)
                        
                        # Print key metrics
                        print(f"  Return: {metrics['Total Return (%)']}%, Trades: {metrics['Total Trades']}, Win Rate: {metrics['Win Rate (%)']}%")
                    else:
                        print(f"  {metrics['error']}")
        
        # Analyze results
        if results:
            results_df = pd.DataFrame(results)
            self.analyze_optimization_results(results_df)
            return results_df
        else:
            print("No valid results found")
            return None
    
    def analyze_optimization_results(self, results_df):
        """Analyze and display optimization results"""
        
        print("\n" + "=" * 60)
        print("PARAMETER OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Best configurations by different metrics
        best_return = results_df.loc[results_df['Total Return (%)'].idxmax()]
        best_sharpe = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
        best_winrate = results_df.loc[results_df['Win Rate (%)'].idxmax()]
        
        print("\n🏆 BEST CONFIGURATIONS:")
        print("-" * 40)
        
        print(f"Best Return: {best_return['Total Return (%)']}%")
        print(f"  Parameters: ATR={best_return['atr_period']}, Factor={best_return['factor']}, SL={best_return['stop_loss']}")
        print(f"  Trades: {best_return['Total Trades']}, Win Rate: {best_return['Win Rate (%)']}%")
        
        print(f"\nBest Sharpe Ratio: {best_sharpe['Sharpe Ratio']}")
        print(f"  Parameters: ATR={best_sharpe['atr_period']}, Factor={best_sharpe['factor']}, SL={best_sharpe['stop_loss']}")
        print(f"  Return: {best_sharpe['Total Return (%)']}%, Win Rate: {best_sharpe['Win Rate (%)']}%")
        
        print(f"\nBest Win Rate: {best_winrate['Win Rate (%)']}%")
        print(f"  Parameters: ATR={best_winrate['atr_period']}, Factor={best_winrate['factor']}, SL={best_winrate['stop_loss']}")
        print(f"  Return: {best_winrate['Total Return (%)']}%, Trades: {best_winrate['Total Trades']}")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print("-" * 40)
        print(f"Total Configurations Tested: {len(results_df)}")
        print(f"Profitable Configurations: {len(results_df[results_df['Total Return (%)'] > 0])}")
        print(f"Average Return: {results_df['Total Return (%)'].mean():.2f}%")
        print(f"Average Win Rate: {results_df['Win Rate (%)'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {results_df['Sharpe Ratio'].mean():.2f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
    
    def run_walkforward_analysis(self, config, data_config, window_months=6, step_months=1):
        """Run walk-forward analysis to test strategy robustness"""
        
        print("Starting Walk-Forward Analysis...")
        print("=" * 50)
        
        # Load full dataset
        df = self.load_real_data(**data_config)
        
        if len(df) < 30:  # Need reasonable amount of data
            print("❌ Insufficient data for walk-forward analysis")
            return None
        
        results = []
        
        # Calculate date ranges
        start_date = df.index[0]
        end_date = df.index[-1]
        current_date = start_date
        
        test_number = 1
        
        while current_date + timedelta(days=window_months*30) < end_date:
            # Define training and test periods
            train_end = current_date + timedelta(days=window_months*30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=step_months*30)
            
            if test_end > end_date:
                break
            
            print(f"\nTest {test_number}:")
            print(f"  Training: {current_date.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"  Testing:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Get test data
            test_data = df[test_start:test_end].copy()
            
            if len(test_data) < 10:  # Need minimum data points
                current_date += timedelta(days=step_months*30)
                continue
            
            # Run backtest on test period
            backtester = SuperTrendBacktester(**config)
            result_df = backtester.run_backtest(test_data)
            metrics = backtester.calculate_metrics(result_df)
            
            if "error" not in metrics:
                result = {
                    'test_number': test_number,
                    'start_date': test_start,
                    'end_date': test_end,
                    'days': len(test_data),
                    **metrics
                }
                results.append(result)
                
                print(f"  Return: {metrics['Total Return (%)']}%, Trades: {metrics['Total Trades']}")
            else:
                print(f"  {metrics['error']}")
            
            current_date += timedelta(days=step_months*30)
            test_number += 1
        
        if results:
            results_df = pd.DataFrame(results)
            self.analyze_walkforward_results(results_df)
            return results_df
        else:
            print("No valid walk-forward results")
            return None
    
    def analyze_walkforward_results(self, results_df):
        """Analyze walk-forward results"""
        
        print("\n" + "=" * 60)
        print("WALK-FORWARD ANALYSIS RESULTS")
        print("=" * 60)
        
        # Calculate stability metrics
        positive_periods = len(results_df[results_df['Total Return (%)'] > 0])
        total_periods = len(results_df)
        consistency = positive_periods / total_periods * 100
        
        print(f"Total Test Periods: {total_periods}")
        print(f"Profitable Periods: {positive_periods}")
        print(f"Consistency: {consistency:.1f}%")
        print(f"Average Return per Period: {results_df['Total Return (%)'].mean():.2f}%")
        print(f"Standard Deviation: {results_df['Total Return (%)'].std():.2f}%")
        print(f"Best Period: {results_df['Total Return (%)'].max():.2f}%")
        print(f"Worst Period: {results_df['Total Return (%)'].min():.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walkforward_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")


def run_comprehensive_analysis():
    """Run comprehensive backtest analysis"""
    
    runner = BacktestRunner()
    
    # Base configuration
    base_config = {
        'initial_capital': 10000,
        'leverage': 5.0,
        'stop_loss': 100,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 3.0
    }
    
    # Data configuration (modify for your needs)
    data_config = {
        'instrument_token': "256265",  # NIFTY 50 index
        'from_date': "2023-01-01",
        'to_date': "2024-12-31",
        'interval': "day"
    }
    
    print("🚀 COMPREHENSIVE SUPERTREND STRATEGY ANALYSIS")
    print("=" * 60)
    
    # 1. Single backtest with default parameters
    print("\n1️⃣ BASELINE BACKTEST")
    print("-" * 30)
    backtester, result_df = runner.run_single_backtest(base_config, data_config)
    backtester.generate_report(result_df)
    backtester.plot_results(result_df, 'baseline_backtest.png')
    
    # 2. Parameter optimization
    print("\n2️⃣ PARAMETER OPTIMIZATION")
    print("-" * 30)
    
    param_ranges = {
        'atr_periods': [7, 10, 14, 20],
        'factors': [2.0, 2.5, 3.0, 3.5, 4.0],
        'stop_losses': [50, 100, 150, 200]
    }
    
    optimization_results = runner.run_parameter_optimization(
        base_config, data_config, param_ranges
    )
    
    # 3. Walk-forward analysis with best parameters
    if optimization_results is not None:
        print("\n3️⃣ WALK-FORWARD ANALYSIS")
        print("-" * 30)
        
        # Use best parameters from optimization
        best_params = optimization_results.loc[optimization_results['Sharpe Ratio'].idxmax()]
        
        optimized_config = base_config.copy()
        optimized_config.update({
            'atr_period': int(best_params['atr_period']),
            'factor': best_params['factor'],
            'stop_loss': best_params['stop_loss']
        })
        
        print(f"Using optimized parameters: ATR={optimized_config['atr_period']}, "
              f"Factor={optimized_config['factor']}, SL={optimized_config['stop_loss']}")
        
        walkforward_results = runner.run_walkforward_analysis(
            optimized_config, data_config, window_months=6, step_months=1
        )


def run_quick_backtest():
    """Run a quick backtest with default parameters"""
    
    runner = BacktestRunner()
    
    config = {
        'initial_capital': 10000,
        'leverage': 5.0,  # NIFTYBEES leverage
        'stop_loss': 100,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 3.0
    }
    
    data_config = {
        'instrument_token': "256265",  # NIFTY 50
        'from_date': "2024-01-01",
        'to_date': "2024-12-31",
        'interval': "day"
    }
    
    print("🚀 QUICK SUPERTREND BACKTEST")
    print("=" * 40)
    
    backtester, result_df = runner.run_single_backtest(config, data_config)
    backtester.generate_report(result_df)
    backtester.plot_results(result_df)


def run_instrument_comparison():
    """Compare strategy performance across different instruments"""
    
    runner = BacktestRunner()
    
    # Configuration
    config = {
        'initial_capital': 10000,
        'leverage': 5.0,
        'stop_loss': 100,
        'commission_per_trade': 20,
        'atr_period': 10,
        'factor': 3.0
    }
    
    # Different instruments to test
    instruments = {
        'NIFTY_50': "256265",      # NIFTY 50 Index
        'BANK_NIFTY': "260105",    # Bank NIFTY Index  
        'NIFTYBEES': "2707457",    # NIFTY BeES ETF
        'BANKBEES': "1364739",     # Bank BeES ETF
    }
    
    comparison_results = []
    
    print("🏆 INSTRUMENT COMPARISON ANALYSIS")
    print("=" * 50)
    
    for name, token in instruments.items():
        print(f"\nTesting {name} (Token: {token})")
        print("-" * 30)
        
        data_config = {
            'instrument_token': token,
            'from_date': "2024-01-01",
            'to_date': "2024-12-31",
            'interval': "day"
        }
        
        try:
            backtester, result_df = runner.run_single_backtest(config, data_config)
            metrics = backtester.calculate_metrics(result_df)
            
            if "error" not in metrics:
                metrics['instrument'] = name
                metrics['token'] = token
                comparison_results.append(metrics)
                
                print(f"✅ {name}: {metrics['Total Return (%)']}% return, "
                      f"{metrics['Total Trades']} trades, {metrics['Win Rate (%)']}% win rate")
            else:
                print(f"❌ {name}: {metrics['error']}")
                
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    # Display comparison
    if comparison_results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Rank by different metrics
        print("\n🏆 RANKINGS:")
        print("-" * 20)
        
        # By total return
        return_ranking = comparison_df.sort_values('Total Return (%)', ascending=False)
        print("\nBy Total Return:")
        for i, row in return_ranking.iterrows():
            print(f"  {row['instrument']:12} {row['Total Return (%)']:8.2f}%")
        
        # By Sharpe ratio
        sharpe_ranking = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        print("\nBy Sharpe Ratio:")
        for i, row in sharpe_ranking.iterrows():
            print(f"  {row['instrument']:12} {row['Sharpe Ratio']:8.2f}")
        
        # By win rate
        winrate_ranking = comparison_df.sort_values('Win Rate (%)', ascending=False)
        print("\nBy Win Rate:")
        for i, row in winrate_ranking.iterrows():
            print(f"  {row['instrument']:12} {row['Win Rate (%)']:8.2f}%")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_df.to_csv(f"instrument_comparison_{timestamp}.csv", index=False)
        print(f"\n💾 Comparison saved to: instrument_comparison_{timestamp}.csv")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "quick":
            run_quick_backtest()
        elif mode == "comprehensive":
            run_comprehensive_analysis()
        elif mode == "compare":
            run_instrument_comparison()
        else:
            print("Usage: python backtest_runner.py [quick|comprehensive|compare]")
    else:
        # Default: run quick backtest
        run_quick_backtest()
