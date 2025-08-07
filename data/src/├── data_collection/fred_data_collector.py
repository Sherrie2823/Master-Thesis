"""
FRED Economic Data Collector for AI-Enhanced Portfolio Optimization
Author: MSc Banking and Digital Finance Student
Date: July 2025

This module collects macroeconomic indicators from FRED (Federal Reserve Economic Data).
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class FREDDataCollector:
    """
    Collects macroeconomic data from FRED for portfolio optimization.
    """

    def __init__(self, api_key=None):
        """
        Initialize FRED data collector.

        Args:
            api_key (str): FRED API key. Get from https://fred.stlouisfed.org/docs/api/api_key.html
        """

        # FRED Economic Indicators for Banking Portfolio Optimization
        self.fred_series = {
            # Interest Rates and Monetary Policy
            'fed_funds_rate': 'FEDFUNDS',  # Federal Funds Rate
            'treasury_10y': 'DGS10',  # 10-Year Treasury Rate
            'treasury_2y': 'DGS2',  # 2-Year Treasury Rate
            'treasury_3m': 'DGS3MO',  # 3-Month Treasury Rate
            'real_fed_funds': 'REAINTRATREARAT10Y',  # Real Interest Rate

            # Economic Growth and Employment
            'gdp_growth': 'GDPC1',  # Real GDP
            'unemployment_rate': 'UNRATE',  # Unemployment Rate
            'employment_ratio': 'EMRATIO',  # Employment-Population Ratio
            'industrial_production': 'INDPRO',  # Industrial Production Index

            # Inflation Indicators
            'cpi_inflation': 'CPIAUCSL',  # Consumer Price Index
            'core_cpi': 'CPILFESL',  # Core CPI (ex food & energy)
            'pce_inflation': 'PCEPI',  # PCE Price Index
            'inflation_expectations': 'T5YIE',  # 5-Year Inflation Expectations

            # Banking Sector Specific
            'bank_lending': 'BOGZ1FL763165005Q',  # Bank Credit
            'commercial_loans': 'BUSLOANS',  # Commercial & Industrial Loans
            'real_estate_loans': 'REALLN',  # Real Estate Loans
            'credit_spread': 'BAA10Y',  # Corporate Bond Spread
            'ted_spread': 'TEDRATE',  # TED Spread (risk indicator)

            # Market and Financial Conditions
            'dollar_index': 'DTWEXBGS',  # Dollar Index
            'gold_price': 'GOLDAMGBD228NLBM',  # Gold Price
            'oil_price': 'DCOILWTICO',  # Oil Price
            'consumer_sentiment': 'UMCSENT',  # Consumer Sentiment

            # Volatility and Risk
            'vix': 'VIXCLS',  # VIX Volatility Index
            'term_spread': 'T10Y2Y',  # 10Y-2Y Term Spread
            'credit_conditions': 'DRBLACBS',  # Credit Conditions
        }

        # Date range (matching banking data)
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # Initialize FRED API
        if api_key:
            self.fred = Fred(api_key=api_key)
            self.api_key = api_key
        else:
            self.fred = None
            self.api_key = None
            print("⚠️  No FRED API key provided. Please get one from:")
            print("   https://fred.stlouisfed.org/docs/api/api_key.html")

        # Create data directory
        self.data_dir = '../../../real_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_api_key(self):
        """
        Interactive setup for FRED API key.
        """
        print("🔑 FRED API Key Setup")
        print("=" * 40)
        print("1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create a free account")
        print("3. Get your API key")
        print("4. Enter it below:")

        api_key = input("\n🔐 Enter your FRED API key: ").strip()

        if api_key:
            self.fred = Fred(api_key=api_key)
            self.api_key = api_key
            print("✅ API key configured successfully!")
            return True
        else:
            print("❌ No API key entered")
            return False

    def collect_economic_data(self, save_to_file=True):
        """
        Collect all economic indicators from FRED.

        Returns:
            dict: Dictionary containing economic data and metadata
        """
        if not self.fred:
            print("❌ FRED API not initialized. Please set up API key first.")
            if not self.setup_api_key():
                return None

        print("📊 Starting FRED economic data collection...")
        print(f"📅 Date range: {self.start_date} to {self.end_date}")
        print(f"📈 Collecting {len(self.fred_series)} economic indicators")

        economic_data = pd.DataFrame()
        metadata = {}
        failed_series = []

        print("\n⬇️  Downloading economic indicators...")

        for name, series_id in self.fred_series.items():
            try:
                print(f"   📊 {name} ({series_id})...", end="")

                # Get series data
                data = self.fred.get_series(
                    series_id,
                    start=self.start_date,
                    end=self.end_date
                )

                # Get series info for metadata
                info = self.fred.get_series_info(series_id)

                # Store data
                economic_data[name] = data
                metadata[name] = {
                    'series_id': series_id,
                    'title': info['title'],
                    'units': info['units'],
                    'frequency': info['frequency'],
                    'last_updated': info['last_updated']
                }

                print(f" ✅ ({len(data)} observations)")

            except Exception as e:
                print(f" ❌ Failed: {str(e)}")
                failed_series.append((name, series_id, str(e)))

        if failed_series:
            print(f"\n⚠️  {len(failed_series)} series failed to download:")
            for name, series_id, error in failed_series:
                print(f"   - {name} ({series_id}): {error}")

        # Process and clean data
        processed_data = self.process_economic_data(economic_data)

        # Calculate derived indicators
        derived_data = self.calculate_derived_indicators(processed_data)

        # Combine all data
        final_data = pd.concat([processed_data, derived_data], axis=1)

        print(f"\n✅ Successfully collected {len(final_data.columns)} economic indicators")
        print(f"📊 Data shape: {final_data.shape[0]} observations × {final_data.shape[1]} indicators")
        print(f"📅 Date range: {final_data.index[0]} to {final_data.index[-1]}")

        # Save data if requested
        if save_to_file:
            self.save_economic_data(final_data, metadata)

        return {
            'data': final_data,
            'metadata': metadata,
            'failed_series': failed_series
        }

    def process_economic_data(self, data):
        """
        Process and clean economic data.
        """
        print("\n🔧 Processing economic data...")

        # Convert to daily frequency and forward fill
        data_daily = data.resample('D').ffill()

        # Handle missing values
        data_clean = data_daily.interpolate(method='linear', limit=5)

        # Calculate some transformations for better stationarity
        processed = data_clean.copy()

        # Log transformations for price-level variables
        price_vars = ['gdp_growth', 'cpi_inflation', 'industrial_production',
                      'gold_price', 'oil_price', 'bank_lending']

        for var in price_vars:
            if var in processed.columns:
                # Calculate growth rates (log differences)
                processed[f'{var}_growth'] = np.log(processed[var]).diff() * 100

        return processed

    def calculate_derived_indicators(self, data):
        """
        Calculate derived economic indicators.
        """
        print("📊 Calculating derived indicators...")

        derived = pd.DataFrame(index=data.index)

        # Yield curve indicators
        if 'treasury_10y' in data.columns and 'treasury_2y' in data.columns:
            derived['yield_curve_slope'] = data['treasury_10y'] - data['treasury_2y']

        if 'treasury_10y' in data.columns and 'treasury_3m' in data.columns:
            derived['yield_curve_steep'] = data['treasury_10y'] - data['treasury_3m']

        # Real interest rates
        if 'fed_funds_rate' in data.columns and 'cpi_inflation' in data.columns:
            # Approximate real rate (nominal - inflation)
            cpi_growth = data['cpi_inflation'].pct_change(periods=12) * 100  # YoY inflation
            derived['real_fed_funds_approx'] = data['fed_funds_rate'] - cpi_growth

        # Credit conditions
        if 'credit_spread' in data.columns and 'treasury_10y' in data.columns:
            derived['credit_risk_premium'] = data['credit_spread'] - data['treasury_10y']

        # Economic momentum
        if 'unemployment_rate' in data.columns:
            derived['unemployment_change'] = data['unemployment_rate'].diff()
            derived['unemployment_trend'] = data['unemployment_rate'].rolling(60).mean()

        # Market stress indicator
        stress_components = []
        if 'vix' in data.columns:
            stress_components.append((data['vix'] - data['vix'].rolling(252).mean()) / data['vix'].rolling(252).std())
        if 'ted_spread' in data.columns:
            stress_components.append(
                (data['ted_spread'] - data['ted_spread'].rolling(252).mean()) / data['ted_spread'].rolling(252).std())

        if stress_components:
            derived['market_stress_index'] = pd.concat(stress_components, axis=1).mean(axis=1)

        return derived.dropna(how='all', axis=1)

    def save_economic_data(self, data, metadata):
        """
        Save economic data to files.
        """
        print("\n💾 Saving economic data...")

        # Save main economic data
        econ_file = os.path.join(self.data_dir, 'fred_economic_data.csv')
        data.to_csv(econ_file)
        print(f"📁 Economic data saved to: {econ_file}")

        # Save metadata
        metadata_file = os.path.join(self.data_dir, 'fred_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("=== FRED ECONOMIC DATA METADATA ===\n\n")
            f.write(f"Data Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Indicators: {len(data.columns)}\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for name, info in metadata.items():
                f.write(f"{name.upper()}:\n")
                f.write(f"  Series ID: {info['series_id']}\n")
                f.write(f"  Title: {info['title']}\n")
                f.write(f"  Units: {info['units']}\n")
                f.write(f"  Frequency: {info['frequency']}\n")
                f.write(f"  Last Updated: {info['last_updated']}\n\n")

        print(f"📁 Metadata saved to: {metadata_file}")

    def display_summary(self, data_dict):
        """
        Display summary of collected economic data.
        """
        if not data_dict:
            print("❌ No economic data to display")
            return

        data = data_dict['data']
        metadata = data_dict['metadata']

        print("\n" + "=" * 60)
        print("📊 FRED ECONOMIC DATA SUMMARY")
        print("=" * 60)

        print(f"\n📅 Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 Total Observations: {len(data)}")
        print(f"📊 Number of Indicators: {len(data.columns)}")
        print(f"✅ Successfully Collected: {len(metadata)} series")

        # Show recent values for key indicators
        print("\n📊 LATEST ECONOMIC INDICATORS:")
        key_indicators = ['fed_funds_rate', 'treasury_10y', 'unemployment_rate',
                          'cpi_inflation', 'vix', 'yield_curve_slope']

        latest_data = data.iloc[-1]
        for indicator in key_indicators:
            if indicator in latest_data.index and not pd.isna(latest_data[indicator]):
                value = latest_data[indicator]
                print(f"  {indicator.replace('_', ' ').title()}: {value:.2f}")

        print("\n✅ Economic data collection completed!")
        print("📁 All files saved to 'data/' directory")


def main():
    """
    Main function to run FRED data collection.
    """
    print("🚀 AI-Enhanced Portfolio Optimization - FRED Economic Data Collector")
    print("=" * 75)

    # Create collector instance
    collector = FREDDataCollector()

    # Collect economic data
    economic_data = collector.collect_economic_data(save_to_file=True)

    # Display summary
    collector.display_summary(economic_data)

    return economic_data


if __name__ == "__main__":
    fred_data = main()