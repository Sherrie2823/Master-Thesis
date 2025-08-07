"""
Complete 10-Year Data Collection for AI-Enhanced Portfolio Optimization
Author: MSc Banking and Digital Finance Student
Date: July 2025

This script collects comprehensive 10-year datasets:
- 15 major banking stocks
- 25+ macroeconomic indicators via FRED
- Complete Treasury yield curve data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveDataCollector:
    """Complete data collection system for banking portfolio optimization."""

    def __init__(self):
        # 15 Major US Banking Stocks
        self.banking_stocks = [
            'JPM',  # JPMorgan Chase (largest US bank)
            'BAC',  # Bank of America (2nd largest)
            'WFC',  # Wells Fargo (3rd largest)
            'C',  # Citigroup (global presence)
            'GS',  # Goldman Sachs (investment banking)
            'MS',  # Morgan Stanley (investment banking)
            'USB',  # U.S. Bancorp (regional leader)
            'PNC',  # PNC Financial (regional)
            'TFC',  # Truist Financial (Southeast)
            'COF',  # Capital One (credit cards)
            'BK',  # Bank of New York Mellon (custody)
            'STT',  # State Street (asset management)
            'AXP',  # American Express (payments)
            'SCHW',  # Charles Schwab (brokerage)
            'CB'  # Chubb Limited (insurance)
        ]

        # 10-year date range
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # FRED API key
        self.fred_api_key = 'c9619c84919b1fb2be8d5a5dd96cd73c'

        # Directories
        self.data_dir = "/data"
        self.results_dir = "/results"

        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def collect_banking_data(self):
        """Collect comprehensive banking sector data."""
        print("🏦 COLLECTING BANKING SECTOR DATA")
        print("=" * 60)
        print(f"📊 Stocks: {len(self.banking_stocks)} major banks")
        print(f"📅 Period: {self.start_date} to {self.end_date} (10 years)")
        print(f"🎯 Target: ~2,500 trading days")

        try:
            # Download all banking stocks
            print("\n⬇️  Downloading stock data...")
            banking_data = yf.download(
                self.banking_stocks,
                start=self.start_date,
                end=self.end_date,
                progress=True,
                auto_adjust=True
            )

            print(f"📊 Raw data shape: {banking_data.shape}")

            # Extract price data
            if isinstance(banking_data.columns, pd.MultiIndex):
                prices = banking_data.xs('Close', axis=1, level=0)
                volume = banking_data.xs('Volume', axis=1, level=0)
            else:
                prices = banking_data[['Close']].copy()
                prices.columns = [self.banking_stocks[0]]
                volume = banking_data[['Volume']].copy()
                volume.columns = [self.banking_stocks[0]]

            # Clean data
            print("\n🔧 Processing data...")

            # Remove stocks with insufficient data (< 80% complete)
            threshold = len(prices) * 0.8
            initial_stocks = len(prices.columns)
            prices = prices.dropna(axis=1, thresh=threshold)
            volume = volume.reindex(columns=prices.columns)
            final_stocks = len(prices.columns)

            if final_stocks < initial_stocks:
                removed_stocks = initial_stocks - final_stocks
                print(f"⚠️  Removed {removed_stocks} stocks due to insufficient data")

            print(f"✅ Final banking data: {prices.shape[0]} days × {prices.shape[1]} stocks")
            print(f"📅 Actual range: {prices.index[0].date()} to {prices.index[-1].date()}")
            print(f"🏦 Stocks included: {', '.join(sorted(prices.columns))}")

            # Calculate returns and additional metrics
            returns = prices.pct_change().dropna()
            correlation = returns.corr()

            # Calculate rolling metrics
            print("\n📊 Calculating additional metrics...")

            # Rolling volatility (30-day)
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)

            # Rolling correlations with market (using SPY as proxy if available)
            try:
                spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
                spy_returns = spy_data['Close'].pct_change().dropna()

                # Align dates
                common_dates = returns.index.intersection(spy_returns.index)
                returns_aligned = returns.loc[common_dates]
                spy_aligned = spy_returns.loc[common_dates]

                # Calculate rolling betas
                rolling_betas = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns)

                for stock in returns_aligned.columns:
                    for i in range(60, len(returns_aligned)):  # 60-day rolling beta
                        window_data = returns_aligned.iloc[i - 60:i]
                        spy_window = spy_aligned.iloc[i - 60:i]

                        if len(window_data) == 60 and len(spy_window) == 60:
                            covariance = window_data[stock].cov(spy_window)
                            spy_variance = spy_window.var()
                            if spy_variance != 0:
                                rolling_betas.iloc[i][stock] = covariance / spy_variance

                print("✅ Market beta calculations completed")

                # Save beta data
                rolling_betas.to_csv(os.path.join(self.data_dir, 'banking_rolling_betas.csv'))

            except Exception as e:
                print(f"⚠️  Beta calculation failed: {e}")
                rolling_betas = None

            # Save all banking data
            print("\n💾 Saving banking data...")

            prices.to_csv(os.path.join(self.data_dir, 'banking_prices_10y.csv'))
            returns.to_csv(os.path.join(self.data_dir, 'banking_returns_10y.csv'))
            correlation.to_csv(os.path.join(self.data_dir, 'banking_correlation_10y.csv'))
            volume.to_csv(os.path.join(self.data_dir, 'banking_volume_10y.csv'))
            rolling_vol.to_csv(os.path.join(self.data_dir, 'banking_rolling_volatility.csv'))

            print(f"📁 Banking prices: banking_prices_10y.csv")
            print(f"📁 Banking returns: banking_returns_10y.csv")
            print(f"📁 Correlation matrix: banking_correlation_10y.csv")
            print(f"📁 Volume data: banking_volume_10y.csv")
            print(f"📁 Rolling volatility: banking_rolling_volatility.csv")
            if rolling_betas is not None:
                print(f"📁 Rolling betas: banking_rolling_betas.csv")

            # Calculate and display summary statistics
            self.display_banking_summary(prices, returns, correlation)

            return True

        except Exception as e:
            print(f"❌ Error collecting banking data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_fred_data(self):
        """Collect comprehensive FRED economic data."""
        print("\n📈 COLLECTING FRED ECONOMIC DATA")
        print("=" * 60)

        # Comprehensive FRED series for banking analysis
        fred_series = {
            # Interest Rates and Monetary Policy
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'treasury_3m': 'DGS3MO',
            'treasury_1y': 'DGS1',
            'treasury_5y': 'DGS5',
            'treasury_30y': 'DGS30',

            # Economic Growth and Employment
            'gdp_real': 'GDPC1',
            'unemployment_rate': 'UNRATE',
            'employment_ratio': 'EMRATIO',
            'industrial_production': 'INDPRO',
            'capacity_utilization': 'TCU',

            # Inflation and Prices
            'cpi_all': 'CPIAUCSL',
            'cpi_core': 'CPILFESL',
            'pce_inflation': 'PCEPI',
            'inflation_expectations_5y': 'T5YIE',
            'inflation_expectations_10y': 'T10YIE',

            # Banking and Credit
            'commercial_loans': 'BUSLOANS',
            'real_estate_loans': 'REALLN',
            'consumer_loans': 'CONSUMER',
            'total_reserves': 'TOTRESNS',
            'bank_credit': 'TOTBKCR',

            # Financial Market Indicators
            'vix': 'VIXCLS',
            'ted_spread': 'TEDRATE',
            'credit_spread_aaa': 'AAA10Y',
            'credit_spread_baa': 'BAA10Y',
            'term_spread_10y2y': 'T10Y2Y',
            'term_spread_10y3m': 'T10Y3M',

            # Economic Sentiment and Activity
            'consumer_sentiment': 'UMCSENT',
            'dollar_index': 'DTWEXBGS',
            'oil_price': 'DCOILWTICO',
            'housing_starts': 'HOUST',
            'retail_sales': 'RSXFS'
        }

        print(f"📊 Indicators: {len(fred_series)} economic series")
        print(f"📅 Period: {self.start_date} to {self.end_date}")

        try:
            fred = Fred(api_key=self.fred_api_key)
            economic_data = pd.DataFrame()
            failed_series = []

            print("\n⬇️  Downloading FRED data...")

            for i, (name, series_id) in enumerate(fred_series.items(), 1):
                try:
                    print(f"   {i:2d}/{len(fred_series)}: {name} ({series_id})...", end="")

                    data = fred.get_series(
                        series_id,
                        start=self.start_date,
                        end=self.end_date
                    )

                    if len(data) > 0:
                        economic_data[name] = data
                        print(f" ✅ ({len(data)} obs)")
                    else:
                        print(f" ⚠️  (No data)")
                        failed_series.append((name, series_id))

                    # Small delay to respect API limits
                    time.sleep(0.1)

                except Exception as e:
                    print(f" ❌ ({str(e)[:30]}...)")
                    failed_series.append((name, series_id))

            if failed_series:
                print(f"\n⚠️  {len(failed_series)} series failed:")
                for name, series_id in failed_series:
                    print(f"   - {name} ({series_id})")

            # Process and clean data
            print(f"\n🔧 Processing economic data...")
            processed_data = self.process_fred_data(economic_data)

            # Save data
            print(f"\n💾 Saving FRED data...")

            processed_data.to_csv(os.path.join(self.data_dir, 'fred_economic_data_10y.csv'))

            # Save metadata
            metadata_file = os.path.join(self.data_dir, 'fred_metadata_10y.txt')
            with open(metadata_file, 'w') as f:
                f.write("FRED Economic Data - 10 Year Collection\n")
                f.write("=" * 50 + "\n")
                f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Period: {self.start_date} to {self.end_date}\n")
                f.write(f"Total Series: {len(economic_data.columns)}\n")
                f.write(f"Final Shape: {processed_data.shape}\n\n")

                f.write("SUCCESSFULLY COLLECTED SERIES:\n")
                f.write("-" * 30 + "\n")
                for name in economic_data.columns:
                    series_id = [k for k, v in fred_series.items() if k == name]
                    if series_id:
                        original_id = fred_series[series_id[0]]
                        f.write(f"{name}: {original_id}\n")

                if failed_series:
                    f.write(f"\nFAILED SERIES:\n")
                    f.write("-" * 30 + "\n")
                    for name, series_id in failed_series:
                        f.write(f"{name}: {series_id}\n")

            print(f"📁 Economic data: fred_economic_data_10y.csv")
            print(f"📁 Metadata: fred_metadata_10y.txt")
            print(f"✅ FRED data: {processed_data.shape[0]} days × {processed_data.shape[1]} indicators")

            return True

        except Exception as e:
            print(f"❌ Error collecting FRED data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_fred_data(self, raw_data):
        """Process and clean FRED data."""
        print("   🔧 Cleaning and processing...")

        # Convert to daily frequency and forward fill
        daily_data = raw_data.resample('D').last()
        processed_data = daily_data.fillna(method='ffill')

        # Calculate derived indicators
        derived = pd.DataFrame(index=processed_data.index)

        # Real interest rates (approximate)
        if 'fed_funds_rate' in processed_data.columns and 'cpi_all' in processed_data.columns:
            # Calculate year-over-year inflation rate
            cpi_yoy = processed_data['cpi_all'].pct_change(periods=365) * 100
            derived['real_fed_funds_rate'] = processed_data['fed_funds_rate'] - cpi_yoy

        # Yield curve shapes
        if 'treasury_10y' in processed_data.columns and 'treasury_2y' in processed_data.columns:
            if 'term_spread_10y2y' not in processed_data.columns:
                derived['yield_curve_slope'] = processed_data['treasury_10y'] - processed_data['treasury_2y']

        # Credit conditions index (combine multiple credit indicators)
        credit_indicators = ['credit_spread_aaa', 'credit_spread_baa', 'ted_spread']
        available_credit = [col for col in credit_indicators if col in processed_data.columns]
        if len(available_credit) >= 2:
            credit_data = processed_data[available_credit]
            # Standardize and average
            standardized = (credit_data - credit_data.mean()) / credit_data.std()
            derived['credit_conditions_index'] = standardized.mean(axis=1)

        # Economic momentum indicators
        if 'unemployment_rate' in processed_data.columns:
            derived['unemployment_change'] = processed_data['unemployment_rate'].diff()
            derived['unemployment_trend'] = processed_data['unemployment_rate'].rolling(60).mean()

        if 'industrial_production' in processed_data.columns:
            derived['industrial_production_growth'] = processed_data['industrial_production'].pct_change(
                periods=252) * 100

        # Combine original and derived data
        final_data = pd.concat([processed_data, derived], axis=1)

        # Remove columns with too many missing values (>50%)
        threshold = len(final_data) * 0.5
        final_data = final_data.dropna(axis=1, thresh=threshold)

        print(f"   ✅ Final processed data: {final_data.shape[1]} indicators")

        return final_data

    def display_banking_summary(self, prices, returns, correlation):
        """Display banking sector summary statistics."""
        print(f"\n📊 BANKING SECTOR SUMMARY")
        print("-" * 40)

        # Performance metrics
        total_returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annual_returns = returns.mean() * 252 * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratios = annual_returns / annual_volatility

        print(f"🏆 TOP 5 PERFORMERS (10-Year Total Return):")
        for i, (stock, ret) in enumerate(total_returns.sort_values(ascending=False).head(5).items(), 1):
            annual_ret = annual_returns[stock]
            vol = annual_volatility[stock]
            sharpe = sharpe_ratios[stock]
            print(f"   {i}. {stock}: {ret:+6.1f}% total | {annual_ret:5.1f}% annual | {sharpe:.3f} Sharpe")

        print(f"\n📊 SECTOR AVERAGES (10-Year):")
        print(f"   Average Annual Return: {annual_returns.mean():5.1f}%")
        print(f"   Average Volatility: {annual_volatility.mean():8.1f}%")
        print(f"   Average Sharpe Ratio: {sharpe_ratios.mean():7.3f}")
        print(f"   Average Correlation: {correlation.mean().mean():.3f}")

        # Risk analysis
        max_drawdowns = self.calculate_max_drawdowns(prices)
        print(f"   Average Max Drawdown: {max_drawdowns.mean():.1f}%")

    def calculate_max_drawdowns(self, prices):
        """Calculate maximum drawdowns for each stock."""
        drawdowns = {}

        for stock in prices.columns:
            cumulative = (1 + prices[stock].pct_change().fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            drawdowns[stock] = drawdown.min() * 100

        return pd.Series(drawdowns)

    def run_complete_collection(self):
        """Run the complete data collection process."""
        print("🚀 AI-ENHANCED PORTFOLIO OPTIMIZATION")
        print("COMPREHENSIVE 10-YEAR DATA COLLECTION")
        print("=" * 80)
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"🎯 Target: 15 banking stocks + 30+ economic indicators")
        print(f"💾 Saving to: {self.data_dir}")

        start_time = datetime.now()

        # Step 1: Banking data
        banking_success = self.collect_banking_data()

        # Step 2: Economic data
        fred_success = self.collect_fred_data()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n" + "=" * 80)
        print("📊 COLLECTION SUMMARY")
        print("=" * 80)
        print(f"⏱️  Total Time: {duration}")
        print(f"🏦 Banking Data: {'✅ Success' if banking_success else '❌ Failed'}")
        print(f"📈 Economic Data: {'✅ Success' if fred_success else '❌ Failed'}")

        if banking_success and fred_success:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"✅ Ready for comprehensive AI portfolio optimization research")
            print(f"📁 All data saved to: {self.data_dir}")

            # List all created files
            print(f"\n📄 CREATED FILES:")
            data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') or f.endswith('.txt')]
            for file in sorted(data_files):
                filepath = os.path.join(self.data_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 ** 2)
                print(f"   📊 {file} ({size_mb:.1f} MB)")

        else:
            print(f"\n⚠️  PARTIAL SUCCESS")
            print(f"Some data collection failed. Check error messages above.")

        return banking_success and fred_success


def main():
    """Main function to run complete data collection."""
    collector = ComprehensiveDataCollector()
    success = collector.run_complete_collection()

    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run data explorer to analyze the 10-year dataset")
        print(f"2. Begin AI model development with comprehensive data")
        print(f"3. Implement traditional portfolio optimization baseline")

    return success


if __name__ == "__main__":
    collection_success = main()