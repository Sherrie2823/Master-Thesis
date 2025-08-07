"""
Fixed Banking Data Collector for AI-Enhanced Portfolio Optimization
Author: MSc Banking and Digital Finance Student
Date: July 2025

This module collects banking sector stock data using Yahoo Finance API.
Fixed version that handles the new yfinance data structure.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os


class FixedBankingDataCollector:
    """
    Collects and processes banking sector stock data for portfolio optimization.
    """

    def __init__(self):
        # Major US Banking Stocks
        self.banking_stocks = [
            'JPM',  # JPMorgan Chase
            'BAC',  # Bank of America
            'WFC',  # Wells Fargo
            'C',  # Citigroup
            'GS',  # Goldman Sachs
            'MS',  # Morgan Stanley
            'USB',  # U.S. Bancorp
            'PNC',  # PNC Financial
            'TFC',  # Truist Financial
            'COF',  # Capital One
            'BK',  # Bank of New York Mellon
            'STT',  # State Street
            'AXP',  # American Express
            'SCHW',  # Charles Schwab
            'CB'  # Chubb Limited
        ]

        # Date range for data collection (10 years)
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # Create data directory if it doesn't exist
        self.data_dir = '../../../real_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def collect_stock_data(self, save_to_file=True):
        """
        Collect historical stock data for banking sector.

        Returns:
            dict: Dictionary containing price data and basic statistics
        """
        print("🏦 Starting FIXED banking sector data collection...")
        print(f"📅 Date range: {self.start_date} to {self.end_date} (10 years)")
        print(f"📊 Collecting data for {len(self.banking_stocks)} banking stocks")

        try:
            # Download data for all banking stocks
            print("\n⬇️  Downloading stock data...")
            data = yf.download(
                self.banking_stocks,
                start=self.start_date,
                end=self.end_date,
                progress=True,
                auto_adjust=True  # This simplifies the data structure
            )

            print(f"📊 Raw data shape: {data.shape}")

            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns (multiple stocks)
                print("📋 Multi-stock data detected")
                adj_close = data.xs('Close', axis=1, level=0)
                volume = data.xs('Volume', axis=1, level=0)
            else:
                # Single level columns (single stock or already processed)
                print("📋 Single-stock data detected")
                if 'Close' in data.columns:
                    adj_close = data[['Close']].copy()
                    adj_close.columns = [self.banking_stocks[0]]
                    volume = data[['Volume']].copy()
                    volume.columns = [self.banking_stocks[0]]
                else:
                    adj_close = data.copy()
                    volume = None

            # Clean the data - remove any stocks with insufficient data
            initial_stocks = len(adj_close.columns)
            adj_close = adj_close.dropna(axis=1, thresh=len(adj_close) * 0.8)
            final_stocks = len(adj_close.columns)

            if final_stocks < initial_stocks:
                print(f"⚠️  Removed {initial_stocks - final_stocks} stocks due to insufficient data")

            print(f"✅ Successfully collected data for {final_stocks} stocks")
            print(f"📊 Final data shape: {adj_close.shape[0]} days × {adj_close.shape[1]} stocks")
            print(f"📅 Actual date range: {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
            print(f"🏦 Stocks in dataset: {list(adj_close.columns)}")

            # Calculate daily returns
            returns = adj_close.pct_change().dropna()

            # Calculate basic statistics
            stats = self.calculate_basic_stats(adj_close, returns)

            # Save data if requested
            if save_to_file:
                self.save_data(adj_close, returns, stats, volume)

            print("✅ Data collection completed successfully!")
            return {
                'prices': adj_close,
                'returns': returns,
                'statistics': stats,
                'volume': volume
            }

        except Exception as e:
            print(f"❌ Error collecting data: {str(e)}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_basic_stats(self, prices, returns):
        """
        Calculate basic statistics for the banking stocks.
        """
        print("\n📊 Calculating basic statistics...")

        stats = {}

        # Price statistics
        current_prices = prices.iloc[-1]
        if len(prices) >= 252:
            price_1y_ago = prices.iloc[-252]
            price_change_1y = (current_prices / price_1y_ago - 1) * 100
        else:
            price_change_1y = pd.Series([np.nan] * len(current_prices), index=current_prices.index)

        stats['price_stats'] = {
            'current_prices': current_prices,
            'price_change_1y': price_change_1y,
            'max_prices': prices.max(),
            'min_prices': prices.min()
        }

        # Return statistics
        annual_factor = 252  # Trading days per year
        stats['return_stats'] = {
            'mean_annual_return': returns.mean() * annual_factor * 100,
            'annual_volatility': returns.std() * np.sqrt(annual_factor) * 100,
            'sharpe_ratio': (returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor)),
            'max_daily_return': returns.max() * 100,
            'min_daily_return': returns.min() * 100
        }

        # Correlation matrix
        stats['correlation_matrix'] = returns.corr()

        return stats

    def save_data(self, prices, returns, stats, volume=None):
        """
        Save collected data to CSV files.
        """
        print("\n💾 Saving data to files...")

        # Save price data
        prices_file = os.path.join(self.data_dir, 'banking_prices.csv')
        prices.to_csv(prices_file)
        print(f"📁 Prices saved to: {prices_file}")

        # Save returns data
        returns_file = os.path.join(self.data_dir, 'banking_returns.csv')
        returns.to_csv(returns_file)
        print(f"📁 Returns saved to: {returns_file}")

        # Save correlation matrix
        corr_file = os.path.join(self.data_dir, 'banking_correlation.csv')
        stats['correlation_matrix'].to_csv(corr_file)
        print(f"📁 Correlation matrix saved to: {corr_file}")

        # Save volume data if available
        if volume is not None:
            volume_file = os.path.join(self.data_dir, 'banking_volume.csv')
            volume.to_csv(volume_file)
            print(f"📁 Volume data saved to: {volume_file}")

        # Save summary statistics
        summary_file = os.path.join(self.data_dir, 'banking_summary_stats.txt')
        with open(summary_file, 'w') as f:
            f.write("=== BANKING SECTOR SUMMARY STATISTICS ===\n\n")
            f.write(f"Data Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Number of Stocks: {len(stats['return_stats']['mean_annual_return'])}\n")
            f.write(f"Stocks: {', '.join(stats['return_stats']['mean_annual_return'].index)}\n\n")

            f.write("ANNUAL RETURNS (%):\n")
            for stock, ret in stats['return_stats']['mean_annual_return'].items():
                f.write(f"{stock}: {ret:.2f}%\n")

            f.write("\nANNUAL VOLATILITY (%):\n")
            for stock, vol in stats['return_stats']['annual_volatility'].items():
                f.write(f"{stock}: {vol:.2f}%\n")

            f.write("\nSHARPE RATIOS:\n")
            for stock, sharpe in stats['return_stats']['sharpe_ratio'].items():
                f.write(f"{stock}: {sharpe:.3f}\n")

        print(f"📁 Summary statistics saved to: {summary_file}")

    def display_summary(self, data):
        """
        Display a quick summary of collected data.
        """
        if data is None:
            print("❌ No data to display")
            return

        prices = data['prices']
        returns = data['returns']
        stats = data['statistics']

        print("\n" + "=" * 60)
        print("📊 BANKING SECTOR DATA SUMMARY")
        print("=" * 60)

        print(f"\n📅 Data Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 Total Trading Days: {len(prices)}")
        print(f"🏦 Number of Banks: {len(prices.columns)}")

        print("\n🏆 TOP PERFORMERS (Annual Return):")
        top_performers = stats['return_stats']['mean_annual_return'].sort_values(ascending=False).head(3)
        for i, (stock, ret) in enumerate(top_performers.items(), 1):
            print(f"  {i}. {stock}: {ret:.2f}%")

        print("\n📊 RISK METRICS:")
        avg_vol = stats['return_stats']['annual_volatility'].mean()
        avg_sharpe = stats['return_stats']['sharpe_ratio'].mean()
        print(f"  Average Volatility: {avg_vol:.2f}%")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.3f}")

        print("\n🔗 AVERAGE CORRELATION:")
        avg_corr = stats['correlation_matrix'].mean().mean()
        print(f"  Banking Sector Correlation: {avg_corr:.3f}")

        print("\n✅ Data collection completed successfully!")
        print("📁 All files saved to 'data/' directory")

        # Show sample of current prices
        print(f"\n💰 LATEST PRICES (as of {prices.index[-1].strftime('%Y-%m-%d')}):")
        for stock, price in stats['price_stats']['current_prices'].items():
            change_1y = stats['price_stats']['price_change_1y'][stock]
            if not np.isnan(change_1y):
                print(f"  {stock}: ${price:.2f} ({change_1y:+.1f}% 1Y)")
            else:
                print(f"  {stock}: ${price:.2f}")


def main():
    """
    Main function to run the fixed banking data collector.
    """
    print("🚀 AI-Enhanced Portfolio Optimization - FIXED Banking Data Collector")
    print("=" * 70)

    # Create collector instance
    collector = FixedBankingDataCollector()

    # Collect data
    data = collector.collect_stock_data(save_to_file=True)

    # Display summary
    collector.display_summary(data)

    return data


if __name__ == "__main__":
    banking_data = main()