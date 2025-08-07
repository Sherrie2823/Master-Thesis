"""
Treasury.gov Data Collector for AI-Enhanced Portfolio Optimization
Author: MSc Banking and Digital Finance Student
Date: July 2025

This module collects U.S. Treasury yield curve data from Treasury.gov API.
Provides precise risk-free rates for portfolio optimization and Sharpe ratio calculations.
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')


class TreasuryDataCollector:
    """
    Collects U.S. Treasury yield curve data from official Treasury.gov API.
    """

    def __init__(self):
        """
        Initialize Treasury data collector.
        """

        # Treasury API endpoints
        self.base_url = "https://api.fiscaldata.treasury.gov/services/api/v1"

        # Treasury yield curve maturities (in years)
        self.yield_maturities = {
            '1_month': '1 Mo',
            '2_month': '2 Mo',
            '3_month': '3 Mo',
            '4_month': '4 Mo',
            '6_month': '6 Mo',
            '1_year': '1 Yr',
            '2_year': '2 Yr',
            '3_year': '3 Yr',
            '5_year': '5 Yr',
            '7_year': '7 Yr',
            '10_year': '10 Yr',
            '20_year': '20 Yr',
            '30_year': '30 Yr'
        }

        # Date range (matching other data)
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # Create data directory
        self.data_dir = '../../../real_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def collect_treasury_yields(self, save_to_file=True):
        """
        Collect daily Treasury yield curve data.

        Returns:
            dict: Dictionary containing yield curve data and derived metrics
        """
        print("🏛️  Starting Treasury.gov data collection...")
        print(f"📅 Date range: {self.start_date} to {self.end_date}")
        print(f"📊 Collecting yield curve data for {len(self.yield_maturities)} maturities")

        try:
            # Get daily Treasury yield curve rates
            print("\n⬇️  Downloading Treasury yield curve data...")
            yield_data = self.fetch_yield_curve_data()

            if yield_data is None or yield_data.empty:
                print("❌ Failed to collect Treasury data")
                return None

            # Process and clean the data
            processed_data = self.process_treasury_data(yield_data)

            # Calculate derived metrics
            derived_metrics = self.calculate_treasury_metrics(processed_data)

            # Combine all Treasury data
            treasury_complete = pd.concat([processed_data, derived_metrics], axis=1)

            print(f"✅ Successfully collected Treasury data")
            print(f"📊 Data shape: {treasury_complete.shape[0]} days × {treasury_complete.shape[1]} metrics")
            print(f"📅 Date range: {treasury_complete.index[0]} to {treasury_complete.index[-1]}")

            # Save data if requested
            if save_to_file:
                self.save_treasury_data(treasury_complete, yield_data)

            return {
                'yield_curve': processed_data,
                'derived_metrics': derived_metrics,
                'complete_data': treasury_complete,
                'raw_data': yield_data
            }

        except Exception as e:
            print(f"❌ Error collecting Treasury data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_yield_curve_data(self):
        """
        Fetch daily Treasury yield curve data from Treasury.gov API.
        """
        # Treasury Daily Yield Curve Rates endpoint
        endpoint = f"{self.base_url}/accounting/od/avg_interest_rates"

        # API parameters
        params = {
            'filter': f'record_date:gte:{self.start_date},record_date:lte:{self.end_date}',
            'fields': 'record_date,security_desc,avg_interest_rate_amt',
            'format': 'json',
            'page[size]': '10000'  # Maximum page size
        }

        all_data = []
        page = 1

        while True:
            params['page[number]'] = page
            print(f"   📄 Fetching page {page}...", end="")

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if 'data' not in data or not data['data']:
                    print(" ✅ (No more data)")
                    break

                all_data.extend(data['data'])
                print(f" ✅ ({len(data['data'])} records)")

                # Check if we have more pages
                if len(data['data']) < params['page[size]']:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                print(f" ❌ Request failed: {str(e)}")
                break
            except Exception as e:
                print(f" ❌ Unexpected error: {str(e)}")
                break

        if not all_data:
            print("❌ No Treasury data retrieved")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Filter for yield curve securities only
        yield_curve_securities = [
            'Treasury Bills', 'Treasury Notes', 'Treasury Bonds',
            '1-Month Treasury Constant Maturity', '3-Month Treasury Constant Maturity',
            '6-Month Treasury Constant Maturity', '1-Year Treasury Constant Maturity',
            '2-Year Treasury Constant Maturity', '3-Year Treasury Constant Maturity',
            '5-Year Treasury Constant Maturity', '7-Year Treasury Constant Maturity',
            '10-Year Treasury Constant Maturity', '20-Year Treasury Constant Maturity',
            '30-Year Treasury Constant Maturity'
        ]

        # Alternative approach: Get Treasury Daily Yield Curve Rates directly
        return self.fetch_daily_yield_curve()

    def fetch_daily_yield_curve(self):
        """
        Fetch daily yield curve rates using the specific Treasury yield curve endpoint.
        """
        endpoint = f"{self.base_url}/accounting/od/daily_treasury_yield_curve"

        params = {
            'filter': f'record_date:gte:{self.start_date},record_date:lte:{self.end_date}',
            'format': 'json',
            'page[size]': '10000'
        }

        all_data = []
        page = 1

        print(f"   🎯 Using Daily Treasury Yield Curve endpoint...")

        while True:
            params['page[number]'] = page
            print(f"   📄 Fetching page {page}...", end="")

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if 'data' not in data or not data['data']:
                    print(" ✅ (Complete)")
                    break

                all_data.extend(data['data'])
                print(f" ✅ ({len(data['data'])} records)")

                if len(data['data']) < params['page[size]']:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                print(f" ❌ Request failed: {str(e)}")
                # Try alternative approach with smaller date ranges
                return self.fetch_yield_curve_chunked()
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
                break

        if not all_data:
            print("❌ No yield curve data retrieved")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        print(f"📊 Retrieved {len(df)} yield curve records")

        return df

    def fetch_yield_curve_chunked(self):
        """
        Fetch yield curve data in smaller chunks if main API fails.
        """
        print("   🔄 Trying chunked approach...")

        # Split date range into yearly chunks
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        all_data = []
        current_year = start.year

        while current_year <= end.year:
            year_start = f"{current_year}-01-01"
            year_end = f"{current_year}-12-31"

            if current_year == end.year:
                year_end = self.end_date

            print(f"   📅 Fetching {current_year} data...", end="")

            endpoint = f"{self.base_url}/accounting/od/daily_treasury_yield_curve"
            params = {
                'filter': f'record_date:gte:{year_start},record_date:lte:{year_end}',
                'format': 'json',
                'page[size]': '10000'
            }

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'data' in data and data['data']:
                    all_data.extend(data['data'])
                    print(f" ✅ ({len(data['data'])} records)")
                else:
                    print(" ⚠️ (No data)")

            except Exception as e:
                print(f" ❌ Failed: {str(e)}")

            current_year += 1

        if all_data:
            return pd.DataFrame(all_data)
        else:
            return None

    def process_treasury_data(self, raw_data):
        """
        Process raw Treasury data into clean yield curve matrix.
        """
        print("\n🔧 Processing Treasury yield curve data...")

        if raw_data is None or raw_data.empty:
            print("❌ No data to process")
            return pd.DataFrame()

        # Convert record_date to datetime
        raw_data['record_date'] = pd.to_datetime(raw_data['record_date'])

        # Create pivot table with dates as index and maturities as columns
        # The exact column names may vary, so we'll be flexible
        rate_columns = [col for col in raw_data.columns if
                        'rate' in col.lower() or 'yr' in col.lower() or 'mo' in col.lower()]

        # Set index to date
        processed_data = raw_data.set_index('record_date')

        # Select only numeric rate columns and convert to float
        numeric_columns = []
        for col in processed_data.columns:
            try:
                # Try to convert to numeric
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                if not processed_data[col].isna().all():
                    numeric_columns.append(col)
            except:
                continue

        # Keep only numeric columns
        yield_curve = processed_data[numeric_columns].copy()

        # Sort by date
        yield_curve = yield_curve.sort_index()

        # Forward fill missing values (Treasury markets are closed on weekends/holidays)
        yield_curve = yield_curve.fillna(method='ffill')

        # Rename columns to be more standardized
        column_mapping = {}
        for col in yield_curve.columns:
            if '1_mo' in col.lower() or '1 mo' in col.lower():
                column_mapping[col] = '1_month'
            elif '3_mo' in col.lower() or '3 mo' in col.lower():
                column_mapping[col] = '3_month'
            elif '6_mo' in col.lower() or '6 mo' in col.lower():
                column_mapping[col] = '6_month'
            elif '1_yr' in col.lower() or '1 yr' in col.lower():
                column_mapping[col] = '1_year'
            elif '2_yr' in col.lower() or '2 yr' in col.lower():
                column_mapping[col] = '2_year'
            elif '3_yr' in col.lower() or '3 yr' in col.lower():
                column_mapping[col] = '3_year'
            elif '5_yr' in col.lower() or '5 yr' in col.lower():
                column_mapping[col] = '5_year'
            elif '7_yr' in col.lower() or '7 yr' in col.lower():
                column_mapping[col] = '7_year'
            elif '10_yr' in col.lower() or '10 yr' in col.lower():
                column_mapping[col] = '10_year'
            elif '20_yr' in col.lower() or '20 yr' in col.lower():
                column_mapping[col] = '20_year'
            elif '30_yr' in col.lower() or '30 yr' in col.lower():
                column_mapping[col] = '30_year'

        yield_curve = yield_curve.rename(columns=column_mapping)

        print(f"✅ Processed yield curve data: {len(yield_curve)} days × {len(yield_curve.columns)} maturities")
        print(f"📊 Available maturities: {list(yield_curve.columns)}")

        return yield_curve

    def calculate_treasury_metrics(self, yield_curve):
        """
        Calculate derived Treasury metrics for portfolio optimization.
        """
        print("📊 Calculating Treasury-derived metrics...")

        metrics = pd.DataFrame(index=yield_curve.index)

        # Yield curve slopes
        if '10_year' in yield_curve.columns and '2_year' in yield_curve.columns:
            metrics['yield_slope_10y2y'] = yield_curve['10_year'] - yield_curve['2_year']

        if '10_year' in yield_curve.columns and '3_month' in yield_curve.columns:
            metrics['yield_slope_10y3m'] = yield_curve['10_year'] - yield_curve['3_month']

        if '2_year' in yield_curve.columns and '3_month' in yield_curve.columns:
            metrics['yield_slope_2y3m'] = yield_curve['2_year'] - yield_curve['3_month']

        # Yield curve curvature (butterfly)
        if all(col in yield_curve.columns for col in ['2_year', '5_year', '10_year']):
            metrics['yield_curvature'] = yield_curve['5_year'] - 0.5 * (yield_curve['2_year'] + yield_curve['10_year'])

        # Level (average of key rates)
        rate_cols = [col for col in yield_curve.columns if col in ['2_year', '5_year', '10_year']]
        if rate_cols:
            metrics['yield_level'] = yield_curve[rate_cols].mean(axis=1)

        # Risk-free rate (use 3-month Treasury as proxy)
        if '3_month' in yield_curve.columns:
            metrics['risk_free_rate'] = yield_curve['3_month']
        elif '1_year' in yield_curve.columns:
            metrics['risk_free_rate'] = yield_curve['1_year']

        # Rate volatility (rolling standard deviation)
        if '10_year' in yield_curve.columns:
            metrics['rate_volatility_10y'] = yield_curve['10_year'].rolling(30).std()

        # Rate momentum (change over past month)
        for col in yield_curve.columns:
            if col in ['3_month', '2_year', '10_year']:
                metrics[f'{col}_momentum'] = yield_curve[col].diff(21)  # 21-day change

        return metrics.dropna(how='all', axis=1)

    def save_treasury_data(self, complete_data, raw_data):
        """
        Save Treasury data to files.
        """
        print("\n💾 Saving Treasury data...")

        # Save complete Treasury data
        treasury_file = os.path.join(self.data_dir, 'treasury_yield_curve.csv')
        complete_data.to_csv(treasury_file)
        print(f"📁 Treasury data saved to: {treasury_file}")

        # Save raw data for reference
        raw_file = os.path.join(self.data_dir, 'treasury_raw_data.csv')
        if raw_data is not None:
            raw_data.to_csv(raw_file, index=False)
            print(f"📁 Raw Treasury data saved to: {raw_file}")

        # Save Treasury summary
        summary_file = os.path.join(self.data_dir, 'treasury_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== U.S. TREASURY YIELD CURVE DATA SUMMARY ===\n\n")
            f.write(f"Data Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Observations: {len(complete_data)}\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("AVAILABLE METRICS:\n")
            for col in complete_data.columns:
                non_null_count = complete_data[col].notna().sum()
                f.write(f"  {col}: {non_null_count} observations\n")

            if '10_year' in complete_data.columns:
                latest_10y = complete_data['10_year'].iloc[-1]
                f.write(f"\nLATEST 10-YEAR TREASURY YIELD: {latest_10y:.2f}%\n")

            if 'yield_slope_10y2y' in complete_data.columns:
                latest_slope = complete_data['yield_slope_10y2y'].iloc[-1]
                f.write(f"LATEST YIELD CURVE SLOPE (10Y-2Y): {latest_slope:.2f}%\n")

        print(f"📁 Treasury summary saved to: {summary_file}")

    def display_summary(self, data_dict):
        """
        Display summary of collected Treasury data.
        """
        if not data_dict:
            print("❌ No Treasury data to display")
            return

        complete_data = data_dict['complete_data']

        print("\n" + "=" * 60)
        print("🏛️  U.S. TREASURY YIELD CURVE DATA SUMMARY")
        print("=" * 60)

        print(
            f"\n📅 Data Period: {complete_data.index[0].strftime('%Y-%m-%d')} to {complete_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 Total Observations: {len(complete_data)}")
        print(f"📊 Number of Metrics: {len(complete_data.columns)}")

        # Show latest yield curve
        print("\n📊 LATEST YIELD CURVE:")
        latest_data = complete_data.iloc[-1]
        yield_columns = [col for col in complete_data.columns if
                         any(mat in col for mat in ['month', 'year']) and 'slope' not in col and 'momentum' not in col]

        for col in sorted(yield_columns):
            if not pd.isna(latest_data[col]):
                print(f"  {col.replace('_', ' ').title()}: {latest_data[col]:.2f}%")

        # Show derived metrics
        print("\n📊 LATEST DERIVED METRICS:")
        derived_columns = [col for col in complete_data.columns if
                           any(term in col for term in ['slope', 'curvature', 'level', 'volatility'])]

        for col in derived_columns[:5]:  # Show first 5 derived metrics
            if not pd.isna(latest_data[col]):
                print(f"  {col.replace('_', ' ').title()}: {latest_data[col]:.2f}%")

        print("\n✅ Treasury data collection completed!")
        print("📁 All files saved to 'data/' directory")


def main():
    """
    Main function to run Treasury data collection.
    """
    print("🚀 AI-Enhanced Portfolio Optimization - Treasury.gov Data Collector")
    print("=" * 75)

    # Create collector instance
    collector = TreasuryDataCollector()

    # Collect Treasury data
    treasury_data = collector.collect_treasury_yields(save_to_file=True)

    # Display summary
    collector.display_summary(treasury_data)

    return treasury_data


if __name__ == "__main__":
    treasury_data = main()