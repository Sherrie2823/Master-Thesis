"""
Treasury Yield Curve Data Collector using FRED API
Author: MSc Banking and Digital Finance Student
Date: July 2025

This module collects comprehensive U.S. Treasury yield curve data using FRED API.
More reliable than Treasury.gov direct API and provides complete yield curve.
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class TreasuryFREDCollector:
    """
    Collects comprehensive Treasury yield curve data using FRED API.
    """

    def __init__(self, api_key='c9619c84919b1fb2be8d5a5dd96cd73c'):
        """
        Initialize Treasury collector using FRED API.
        """

        # FRED Treasury yield series (daily data)
        self.treasury_series = {
            # Short-term rates
            '1_month': 'DGS1MO',  # 1-Month Treasury
            '3_month': 'DGS3MO',  # 3-Month Treasury
            '6_month': 'DGS6MO',  # 6-Month Treasury

            # Medium-term rates
            '1_year': 'DGS1',  # 1-Year Treasury
            '2_year': 'DGS2',  # 2-Year Treasury
            '3_year': 'DGS3',  # 3-Year Treasury
            '5_year': 'DGS5',  # 5-Year Treasury
            '7_year': 'DGS7',  # 7-Year Treasury

            # Long-term rates
            '10_year': 'DGS10',  # 10-Year Treasury
            '20_year': 'DGS20',  # 20-Year Treasury
            '30_year': 'DGS30',  # 30-Year Treasury

            # TIPS (inflation-protected)
            '5_year_tips': 'DFII5',  # 5-Year TIPS
            '10_year_tips': 'DFII10',  # 10-Year TIPS
            '30_year_tips': 'DFII30',  # 30-Year TIPS

            # Real rates
            '5_year_real': 'FII5',  # 5-Year Real Rate
            '10_year_real': 'FII10',  # 10-Year Real Rate
            '30_year_real': 'FII30',  # 30-Year Real Rate
        }

        # Additional Treasury-related series
        self.additional_series = {
            # Term spreads (already calculated by FRED)
            'term_spread_10y2y': 'T10Y2Y',  # 10Y-2Y spread
            'term_spread_10y3m': 'T10Y3M',  # 10Y-3M spread

            # Breakeven inflation rates
            'breakeven_5y': 'T5YIE',  # 5-Year breakeven inflation
            'breakeven_10y': 'T10YIE',  # 10-Year breakeven inflation
            'breakeven_30y': 'T30YIE',  # 30-Year breakeven inflation

            # Treasury yields in other contexts
            'aaa_treasury_spread': 'AAA10Y',  # AAA-Treasury spread
            'baa_treasury_spread': 'BAA10Y',  # BAA-Treasury spread
        }

        # Date range
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # Initialize FRED
        self.fred = Fred(api_key=api_key)

        # Create data directory
        self.data_dir = '../../../real_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def collect_treasury_data(self, save_to_file=True):
        """
        Collect complete Treasury yield curve and related data.

        Returns:
            dict: Dictionary containing yield curve and derived metrics
        """
        print("🏛️  Starting Treasury yield curve collection (via FRED)...")
        print(f"📅 Date range: {self.start_date} to {self.end_date}")
        print(f"📊 Collecting {len(self.treasury_series)} yield curve rates")
        print(f"📈 Plus {len(self.additional_series)} additional Treasury metrics")

        # Collect yield curve data
        yield_curve = self.collect_yield_curve()

        # Collect additional Treasury data
        additional_data = self.collect_additional_data()

        # Calculate derived metrics
        derived_metrics = self.calculate_treasury_metrics(yield_curve)

        # Combine all data
        complete_data = pd.concat([yield_curve, additional_data, derived_metrics], axis=1)

        # Clean and align data
        complete_data = self.clean_treasury_data(complete_data)

        print(f"\n✅ Successfully collected Treasury data")
        print(f"📊 Final data shape: {complete_data.shape[0]} days × {complete_data.shape[1]} metrics")
        print(f"📅 Date range: {complete_data.index[0]} to {complete_data.index[-1]}")

        # Save data if requested
        if save_to_file:
            self.save_treasury_data(complete_data)

        return {
            'yield_curve': yield_curve,
            'additional_data': additional_data,
            'derived_metrics': derived_metrics,
            'complete_data': complete_data
        }

    def collect_yield_curve(self):
        """
        Collect Treasury yield curve rates.
        """
        print("\n⬇️  Downloading yield curve rates...")

        yield_curve = pd.DataFrame()
        failed_series = []

        for name, series_id in self.treasury_series.items():
            try:
                print(f"   📊 {name} ({series_id})...", end="")

                data = self.fred.get_series(
                    series_id,
                    start=self.start_date,
                    end=self.end_date
                )

                if len(data) > 0:
                    yield_curve[name] = data
                    print(f" ✅ ({len(data)} observations)")
                else:
                    print(" ⚠️ (No data)")
                    failed_series.append((name, series_id))

            except Exception as e:
                print(f" ❌ Failed: {str(e)}")
                failed_series.append((name, series_id))

        if failed_series:
            print(f"\n⚠️  {len(failed_series)} yield curve series unavailable:")
            for name, series_id in failed_series:
                print(f"   - {name} ({series_id})")

        return yield_curve

    def collect_additional_data(self):
        """
        Collect additional Treasury-related metrics.
        """
        print("\n⬇️  Downloading additional Treasury metrics...")

        additional_data = pd.DataFrame()
        failed_series = []

        for name, series_id in self.additional_series.items():
            try:
                print(f"   📊 {name} ({series_id})...", end="")

                data = self.fred.get_series(
                    series_id,
                    start=self.start_date,
                    end=self.end_date
                )

                if len(data) > 0:
                    additional_data[name] = data
                    print(f" ✅ ({len(data)} observations)")
                else:
                    print(" ⚠️ (No data)")
                    failed_series.append((name, series_id))

            except Exception as e:
                print(f" ❌ Failed: {str(e)}")
                failed_series.append((name, series_id))

        if failed_series:
            print(f"\n⚠️  {len(failed_series)} additional series unavailable:")
            for name, series_id in failed_series:
                print(f"   - {name} ({series_id})")

        return additional_data

    def calculate_treasury_metrics(self, yield_curve):
        """
        Calculate derived Treasury metrics and risk factors.
        """
        print("\n📊 Calculating derived Treasury metrics...")

        metrics = pd.DataFrame(index=yield_curve.index)

        # Yield curve level (average of available rates)
        available_rates = [col for col in yield_curve.columns if col in ['2_year', '5_year', '10_year']]
        if available_rates:
            metrics['yield_level'] = yield_curve[available_rates].mean(axis=1)

        # Yield curve slopes (if not already available from FRED)
        if '10_year' in yield_curve.columns and '2_year' in yield_curve.columns:
            if 'term_spread_10y2y' not in yield_curve.columns:
                metrics['yield_slope_10y2y'] = yield_curve['10_year'] - yield_curve['2_year']

        if '10_year' in yield_curve.columns and '3_month' in yield_curve.columns:
            if 'term_spread_10y3m' not in yield_curve.columns:
                metrics['yield_slope_10y3m'] = yield_curve['10_year'] - yield_curve['3_month']

        if '5_year' in yield_curve.columns and '2_year' in yield_curve.columns:
            metrics['yield_slope_5y2y'] = yield_curve['5_year'] - yield_curve['2_year']

        # Yield curve curvature (butterfly spread)
        if all(col in yield_curve.columns for col in ['2_year', '5_year', '10_year']):
            metrics['yield_curvature'] = yield_curve['5_year'] - 0.5 * (yield_curve['2_year'] + yield_curve['10_year'])

        # Short-term rate volatility
        if '3_month' in yield_curve.columns:
            metrics['short_rate_volatility'] = yield_curve['3_month'].rolling(30).std()
            metrics['short_rate_momentum'] = yield_curve['3_month'].diff(21)  # Monthly change

        # Long-term rate volatility
        if '10_year' in yield_curve.columns:
            metrics['long_rate_volatility'] = yield_curve['10_year'].rolling(30).std()
            metrics['long_rate_momentum'] = yield_curve['10_year'].diff(21)  # Monthly change

        # Real vs nominal spread (if both available)
        if '10_year' in yield_curve.columns and '10_year_real' in yield_curve.columns:
            metrics['inflation_risk_premium'] = yield_curve['10_year'] - yield_curve['10_year_real']

        # Risk-free rate proxies
        if '3_month' in yield_curve.columns:
            metrics['risk_free_rate_3m'] = yield_curve['3_month']
        if '1_year' in yield_curve.columns:
            metrics['risk_free_rate_1y'] = yield_curve['1_year']

        # Rate change indicators
        for maturity in ['3_month', '2_year', '10_year']:
            if maturity in yield_curve.columns:
                # Daily change
                metrics[f'{maturity}_daily_change'] = yield_curve[maturity].diff()
                # Weekly change
                metrics[f'{maturity}_weekly_change'] = yield_curve[maturity].diff(5)

        return metrics.dropna(how='all', axis=1)

    def clean_treasury_data(self, data):
        """
        Clean and align Treasury data.
        """
        print("🔧 Cleaning and aligning Treasury data...")

        # Convert to daily frequency
        data_daily = data.resample('D').last()

        # Forward fill for weekends and holidays
        data_clean = data_daily.fillna(method='ffill')

        # Remove columns with too many missing values
        threshold = len(data_clean) * 0.5  # Keep columns with at least 50% data
        data_clean = data_clean.dropna(axis=1, thresh=threshold)

        # Interpolate remaining missing values (short gaps only)
        data_clean = data_clean.interpolate(method='linear', limit=5)

        return data_clean

    def save_treasury_data(self, complete_data):
        """
        Save Treasury data to files.
        """
        print("\n💾 Saving Treasury data...")

        # Save complete Treasury dataset
        treasury_file = os.path.join(self.data_dir, 'treasury_complete.csv')
        complete_data.to_csv(treasury_file)
        print(f"📁 Complete Treasury data saved to: {treasury_file}")

        # Save yield curve only (for easier analysis)
        yield_columns = [col for col in complete_data.columns
                         if any(term in col for term in ['month', 'year']) and
                         'spread' not in col and 'change' not in col and 'volatility' not in col]

        if yield_columns:
            yield_curve_file = os.path.join(self.data_dir, 'treasury_yield_curve.csv')
            complete_data[yield_columns].to_csv(yield_curve_file)
            print(f"📁 Yield curve data saved to: {yield_curve_file}")

        # Save Treasury summary
        summary_file = os.path.join(self.data_dir, 'treasury_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== U.S. TREASURY DATA SUMMARY (via FRED) ===\n\n")
            f.write(f"Data Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Observations: {len(complete_data)}\n")
            f.write(f"Total Metrics: {len(complete_data.columns)}\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Categorize metrics
            f.write("YIELD CURVE RATES:\n")
            for col in yield_columns:
                if col in complete_data.columns:
                    non_null = complete_data[col].notna().sum()
                    f.write(f"  {col}: {non_null} observations\n")

            spreads = [col for col in complete_data.columns if 'spread' in col or 'slope' in col]
            if spreads:
                f.write("\nYIELD SPREADS & SLOPES:\n")
                for col in spreads:
                    non_null = complete_data[col].notna().sum()
                    f.write(f"  {col}: {non_null} observations\n")

            # Latest values
            latest = complete_data.iloc[-1]
            f.write(f"\nLATEST VALUES (as of {complete_data.index[-1].date()}):\n")

            key_rates = ['3_month', '2_year', '10_year', '30_year']
            for rate in key_rates:
                if rate in latest.index and not pd.isna(latest[rate]):
                    f.write(f"  {rate.replace('_', '-').upper()}: {latest[rate]:.2f}%\n")

            if 'yield_slope_10y2y' in latest.index and not pd.isna(latest['yield_slope_10y2y']):
                f.write(f"  YIELD CURVE SLOPE (10Y-2Y): {latest['yield_slope_10y2y']:.2f}%\n")

        print(f"📁 Treasury summary saved to: {summary_file}")

    def display_summary(self, data_dict):
        """
        Display Treasury data summary.
        """
        if not data_dict:
            print("❌ No Treasury data to display")
            return

        complete_data = data_dict['complete_data']

        print("\n" + "=" * 60)
        print("🏛️  U.S. TREASURY DATA SUMMARY")
        print("=" * 60)

        print(
            f"\n📅 Data Period: {complete_data.index[0].strftime('%Y-%m-%d')} to {complete_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 Total Observations: {len(complete_data)}")
        print(f"📊 Total Metrics: {len(complete_data.columns)}")

        # Show current yield curve
        print("\n📊 LATEST YIELD CURVE:")
        latest = complete_data.iloc[-1]

        yield_rates = ['1_month', '3_month', '6_month', '1_year', '2_year', '3_year',
                       '5_year', '7_year', '10_year', '20_year', '30_year']

        for rate in yield_rates:
            if rate in latest.index and not pd.isna(latest[rate]):
                display_name = rate.replace('_', '-').upper()
                print(f"  {display_name}: {latest[rate]:.2f}%")

        # Show key derived metrics
        print("\n📊 KEY TREASURY METRICS:")

        key_metrics = ['yield_level', 'yield_slope_10y2y', 'yield_curvature',
                       'inflation_risk_premium', 'risk_free_rate_3m']

        for metric in key_metrics:
            if metric in latest.index and not pd.isna(latest[metric]):
                display_name = metric.replace('_', ' ').title()
                print(f"  {display_name}: {latest[metric]:.2f}%")

        print("\n✅ Treasury data collection completed!")
        print("📁 All files saved to 'data/' directory")


def main():
    """
    Main function to run Treasury data collection via FRED.
    """
    print("🚀 AI-Enhanced Portfolio Optimization - Treasury Data Collector (FRED)")
    print("=" * 80)

    # Create collector instance
    collector = TreasuryFREDCollector()

    # Collect Treasury data
    treasury_data = collector.collect_treasury_data(save_to_file=True)

    # Display summary
    collector.display_summary(treasury_data)

    return treasury_data


if __name__ == "__main__":
    treasury_data = main()