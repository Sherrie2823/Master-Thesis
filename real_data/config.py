"""
Configuration file for AI-Enhanced Portfolio Optimization Project
MSc Banking and Digital Finance

Updated configuration for rolling optimization framework with expanding windows.
"""

import os
from pathlib import Path
from XGboost_v5 import BankingXGBoostV5


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
DATA_PATH = PROJECT_ROOT / "data" / "raw"
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = PROJECT_ROOT / "models"
PLOTS_PATH = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for path in [RESULTS_PATH, MODELS_PATH, PLOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# BANKING SECTOR UNIVERSE (Updated to match your actual data)
# ============================================================================
# Primary banking stocks based on your actual implementation
BANKING_STOCKS = [
    'JPM',  # JPMorgan Chase & Co.
    'BAC',  # Bank of America Corp
    'WFC',  # Wells Fargo & Company
    'C',  # Citigroup Inc.
    'GS',  # Goldman Sachs Group Inc
    'MS',  # Morgan Stanley
    'USB',  # U.S. Bancorp
    'PNC',  # PNC Financial Services Group Inc
    'TFC',  # Truist Financial Corporation
    'COF',  # Capital One Financial Corp
    'BK',  # Bank of New York Mellon
    'STT',  # State Street Corporation
    'AXP',  # American Express
    'SCHW',  # Charles Schwab Corp
    'CB'  # Chubb Limited
]

# Banking sector ETFs for benchmarking
BANKING_ETFS = [
    'XLF',  # Financial Select Sector SPDR Fund
    'VFH',  # Vanguard Financials ETF
    'KBE',  # SPDR S&P Bank ETF
    'KRE'  # SPDR S&P Regional Banking ETF
]

# ============================================================================
# ECONOMIC INDICATORS (35 indicators as per your implementation)
# ============================================================================
FRED_INDICATORS = {
    # Interest Rates & Monetary Policy (7 indicators)
    'FEDFUNDS': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury Rate',
    'DGS2': '2-Year Treasury Rate',
    'DGS1MO': '1-Month Treasury Rate',
    'DGS3MO': '3-Month Treasury Rate',
    'DGS5': '5-Year Treasury Rate',
    'DGS30': '30-Year Treasury Rate',

    # Economic Growth & Employment (6 indicators)
    'GDPC1': 'Real GDP',
    'UNRATE': 'Unemployment Rate',
    'CIVPART': 'Labor Force Participation Rate',
    'PAYEMS': 'Total Nonfarm Payrolls',
    'INDPRO': 'Industrial Production Index',
    'HOUST': 'Housing Starts',

    # Inflation & Price Dynamics (5 indicators)
    'CPIAUCSL': 'Consumer Price Index',
    'PCEPI': 'Personal Consumption Expenditures Price Index',
    'T5YIE': '5-Year Breakeven Inflation Rate',
    'T10YIE': '10-Year Breakeven Inflation Rate',
    'DFEDTARU': 'Federal Reserve Target Rate Upper Limit',

    # Banking & Credit Conditions (5 indicators)
    'BAA10Y': 'Corporate BAA Yield relative to 10-Year Treasury',
    'AAA10Y': 'Corporate AAA Yield relative to 10-Year Treasury',
    'DRTSCILM': 'Charge-Off Rate on Commercial & Industrial Loans',
    'DRTSCIS': 'Charge-Off Rate on Consumer Loans',
    'BOGZ1FL763164103Q': 'Bank Total Loans and Leases',

    # Financial Market Indicators (6 indicators)
    'VIXCLS': 'VIX Volatility Index',
    'DEXUSEU': 'US/Euro Exchange Rate',
    'DEXJPUS': 'Japan/US Exchange Rate',
    'DGS10-DGS2': 'Yield Curve Slope (10Y-2Y)',
    'DGS10-DGS3MO': 'Yield Curve Slope (10Y-3M)',
    'TEDRATE': 'TED Spread',

    # Economic Sentiment & Activity (6 indicators)
    'UMCSENT': 'University of Michigan Consumer Sentiment',
    'DCOILWTICO': 'Crude Oil Prices (WTI)',
    'DTWEXBGS': 'Trade Weighted US Dollar Index',
    'CPILFESL': 'Core CPI',
    'PCEPILFE': 'Core PCE Price Index',
    'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average'
}


# ============================================================================
# ROLLING OPTIMIZATION PARAMETERS (Updated for new framework)
# ============================================================================
class RollingOptimizationConfig:
    """Configuration for rolling portfolio optimization with expanding windows"""

    # Rolling framework parameters
    REBALANCE_FREQUENCY = 1  # DAILY rebalancing 
    MIN_HISTORY = 252  # Minimum 1 year of data for first optimization
    TRANSACTION_COST = 0.001  # 0.1% transaction cost per rebalancing

    # Risk and return parameters
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

    # Portfolio constraints (based on your implementation)
    MIN_WEIGHT = 0.01  # Minimum 1% allocation per asset
    MAX_WEIGHT = 0.20  # Maximum 20% allocation per asset

    # Black-Litterman parameters
    BL_TAU = 0.025  # Uncertainty of prior
    BL_RISK_AVERSION = 3.0  # Risk aversion parameter
    BL_MOMENTUM_THRESHOLD = 0.02  # 2% threshold for creating views
    BL_VIEW_UNCERTAINTY = 0.05  # Base view uncertainty
    BL_RELATIVE_VIEW_UNCERTAINTY = 0.1  # Uncertainty for relative views

    # Risk Parity parameters
    RP_TOLERANCE = 1e-6  # Convergence tolerance for risk budgeting

    # Performance evaluation
    CONFIDENCE_LEVEL = 0.95  # For VaR calculations
    BOOTSTRAP_SAMPLES = 1000  # For confidence intervals


class BacktestConfig:
    """Configuration for backtesting and performance evaluation"""

    # Date ranges (aligned with your 2015-2024 data)
    START_DATE = '2015-01-01'  # 10 years of data
    END_DATE = '2024-12-31'

    # Walk-forward analysis
    INITIAL_WINDOW = 252  # Initial training window (1 year)
    STEP_SIZE = 1  # Step size for walk-forward (1 day)
    HOLDING_PERIOD = 15

    # Performance metrics
    BENCHMARK = 'XLF'  # Banking sector ETF benchmark

    # Portfolio evaluation parameters
    INITIAL_PORTFOLIO_VALUE = 100.0  # Starting portfolio value ($100)

    # Stress testing periods
    STRESS_PERIODS = {
        'covid_crisis': ('2020-03-01', '2020-04-30'),
        'regional_banking_crisis': ('2023-03-01', '2023-03-31'),
        'rate_hiking_cycle': ('2022-01-01', '2023-12-31')
    }


class AIModelConfig:
    """Configuration for AI model parameters (for future implementation)"""

    # LSTM parameters
    LSTM_LOOKBACK = 60  # 60 days lookback
    LSTM_EPOCHS = 100  # Training epochs
    LSTM_BATCH_SIZE = 32  # Batch size
    LSTM_HIDDEN_UNITS = [100, 50, 25]  # Hidden layer units
    LSTM_DROPOUT = 0.2  # Dropout rate

    # Reinforcement Learning parameters
    RL_EPISODES = 1000  # Training episodes
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.95  # Discount factor
    RL_EPSILON = 0.1  # Exploration rate

    # Feature engineering
    TECHNICAL_INDICATORS = [
        'SMA_20', 'SMA_50', 'SMA_200',  # Simple Moving Averages
        'EMA_12', 'EMA_26',  # Exponential Moving Averages
        'RSI_14',  # Relative Strength Index
        'MACD', 'MACD_Signal',  # MACD indicators
        'BB_Upper', 'BB_Lower',  # Bollinger Bands
        'Volume_SMA_20'  # Volume indicators
    ]


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
class PlotConfig:
    """Configuration for plotting and visualization"""

    # Figure settings
    FIGURE_SIZE = (15, 12)  # Larger for rolling optimization plots
    DPI = 300

    # Color schemes for different strategies
    STRATEGY_COLORS = {
        'markowitz_max_sharpe': '#1f77b4',
        'markowitz_min_vol': '#ff7f0e',
        'black_litterman': '#2ca02c',
        'risk_parity': '#d62728',
        'equal_weight': '#9467bd'
    }

    # Style settings
    STYLE = 'seaborn-v0_8'
    FONT_SIZE = 12
    TITLE_SIZE = 14
    LEGEND_SIZE = 10


# ============================================================================
# PERFORMANCE METRICS CONFIGURATION
# ============================================================================
class PerformanceMetrics:
    """Configuration for portfolio performance evaluation metrics"""

    # Core metrics for rolling optimization
    CORE_METRICS = [
        'total_return',
        'annual_return',
        'volatility',
        'sharpe_ratio',
        'max_drawdown',
        'var_95',
        'negative_days_pct',
        'total_transaction_costs'
    ]

    # Extended metrics for detailed analysis
    EXTENDED_METRICS = [
        'sortino_ratio',
        'calmar_ratio',
        'information_ratio',
        'var_99',
        'cvar_95',
        'skewness',
        'kurtosis'
    ]

    # Banking specific metrics
    BANKING_METRICS = [
        'stress_period_performance',
        'crisis_recovery_time',
        'sector_correlation',
        'concentration_risk'
    ]


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
import logging


def setup_logging():
    """Setup logging configuration for rolling optimization"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create formatters
    file_formatter = logging.Formatter(log_format)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Setup file handler
    file_handler = logging.FileHandler(RESULTS_PATH / 'rolling_optimization.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Setup logger
    logger = logging.getLogger('rolling_optimization')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_file_paths():
    """Get all relevant file paths for the project"""
    return {
        'data_path': DATA_PATH,
        'results_path': RESULTS_PATH,
        'models_path': MODELS_PATH,
        'plots_path': PLOTS_PATH,
        'stock_data': DATA_PATH / 'banking_stocks.csv',
        'economic_data': DATA_PATH / 'economic_indicators.csv'
    }


def validate_data_files():
    """Validate that required data files exist for rolling optimization"""
    required_files = []

    # Look for any CSV files in data directory
    csv_files = list(DATA_PATH.glob("*.csv"))
    if csv_files:
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  ✅ {file.name}")
            required_files.append(file)

    # Check if we have at least one data file
    if not required_files:
        print("❌ No CSV data files found")
        return False

    # Additional validation for file content
    try:
        import pandas as pd
        for file in required_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            if df.shape[1] >= len(BANKING_STOCKS):
                print(f"  ✅ {file.name}: {df.shape[1]} columns, {df.shape[0]} rows")
                return True
            elif df.shape[1] > 1:
                print(f"  ⚠️  {file.name}: Only {df.shape[1]} columns (expected {len(BANKING_STOCKS)})")
    except Exception as e:
        print(f"  ❌ Error reading files: {e}")

    return len(required_files) > 0


# ============================================================================
# MODEL VALIDATION CONFIGURATION
# ============================================================================
class ValidationConfig:
    """Configuration for model validation and testing"""

    # Statistical tests
    SIGNIFICANCE_LEVEL = 0.05
    BOOTSTRAP_SAMPLES = 1000

    # Performance comparison tests
    COMPARISON_TESTS = [
        'paired_t_test',
        'wilcoxon_signed_rank',
        'bootstrap_difference'
    ]

    # Robustness tests for rolling optimization
    ROBUSTNESS_TESTS = {
        'parameter_sensitivity': {
            'rebalance_frequency': [14, 21, 42],  # 2 weeks, 1 month, 2 months
            'transaction_cost': [0.0005, 0.001, 0.002],  # 0.05%, 0.1%, 0.2%
            'min_history': [189, 252, 378]  # 9 months, 1 year, 1.5 years
        },
        'subsample_analysis': {
            'start_dates': ['2015-01-01', '2016-01-01', '2017-01-01'],
            'end_dates': ['2022-12-31', '2023-12-31', '2024-12-31']
        }
    }


# ============================================================================
# EXPORT FUNCTIONS (Updated for rolling optimization)
# ============================================================================
def get_rolling_config():
    """Get rolling optimization configuration as dictionary"""
    return {
        'rebalance_frequency': RollingOptimizationConfig.REBALANCE_FREQUENCY,
        'min_history': RollingOptimizationConfig.MIN_HISTORY,
        'transaction_cost': RollingOptimizationConfig.TRANSACTION_COST,
        'risk_free_rate': RollingOptimizationConfig.RISK_FREE_RATE,
        'min_weight': RollingOptimizationConfig.MIN_WEIGHT,
        'max_weight': RollingOptimizationConfig.MAX_WEIGHT,
        'bl_tau': RollingOptimizationConfig.BL_TAU,
        'bl_risk_aversion': RollingOptimizationConfig.BL_RISK_AVERSION,
        'bl_momentum_threshold': RollingOptimizationConfig.BL_MOMENTUM_THRESHOLD
    }


def get_backtest_config():
    """Get backtesting configuration as dictionary"""
    return {
        'start_date': BacktestConfig.START_DATE,
        'end_date': BacktestConfig.END_DATE,
        'initial_window': BacktestConfig.INITIAL_WINDOW,
        'step_size': BacktestConfig.STEP_SIZE,
        'benchmark': BacktestConfig.BENCHMARK,
        'initial_portfolio_value': BacktestConfig.INITIAL_PORTFOLIO_VALUE,
        'stress_periods': BacktestConfig.STRESS_PERIODS
    }


def get_ai_config():
    """Get AI model configuration as dictionary"""
    return {
        'lstm_lookback': AIModelConfig.LSTM_LOOKBACK,
        'lstm_epochs': AIModelConfig.LSTM_EPOCHS,
        'lstm_batch_size': AIModelConfig.LSTM_BATCH_SIZE,
        'lstm_hidden_units': AIModelConfig.LSTM_HIDDEN_UNITS,
        'lstm_dropout': AIModelConfig.LSTM_DROPOUT,
        'rl_episodes': AIModelConfig.RL_EPISODES,
        'rl_learning_rate': AIModelConfig.RL_LEARNING_RATE,
        'rl_gamma': AIModelConfig.RL_GAMMA,
        'rl_epsilon': AIModelConfig.RL_EPSILON,
        'technical_indicators': AIModelConfig.TECHNICAL_INDICATORS
    }


# ============================================================================
# MAIN CONFIGURATION VALIDATION
# ============================================================================
def validate_configuration():
    """Validate configuration parameters for rolling optimization"""
    issues = []

    # Check rebalancing frequency
    if RollingOptimizationConfig.REBALANCE_FREQUENCY < 1:
        issues.append("Rebalancing frequency must be at least 1 day")

    # Check minimum history
    if RollingOptimizationConfig.MIN_HISTORY < 63:
        issues.append("Minimum history should be at least 3 months (63 days)")

    # Check weight constraints
    if RollingOptimizationConfig.MIN_WEIGHT >= RollingOptimizationConfig.MAX_WEIGHT:
        issues.append("Minimum weight must be less than maximum weight")

    # Check transaction costs
    if RollingOptimizationConfig.TRANSACTION_COST < 0:
        issues.append("Transaction cost cannot be negative")
        
    if BacktestConfig.HOLDING_PERIOD < 1:
        issues.append("HOLDING_PERIOD must be at least 1 day")
    if BacktestConfig.HOLDING_PERIOD < BacktestConfig.STEP_SIZE:
        issues.append(
            f"HOLDING_PERIOD ({BacktestConfig.HOLDING_PERIOD}) "
            f"must be >= STEP_SIZE ({BacktestConfig.STEP_SIZE}) to avoid overlap"
        )

    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True


# ============================================================================
# MAIN CONFIGURATION EXPORT
# ============================================================================
if __name__ == "__main__":
    print("🔧 Rolling Portfolio Optimization Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    print(f"Banking Stocks: {len(BANKING_STOCKS)} assets")
    print(f"Economic Indicators: {len(FRED_INDICATORS)} indicators")

    # Validate configuration
    print(f"\n⚙️  Configuration Validation:")
    validate_configuration()

    # Validate data files
    print(f"\n📁 Data File Validation:")
    if validate_data_files():
        print("✅ Data files found and ready for rolling optimization")
    else:
        print("❌ No suitable data files found - please check data directory")

    # Display key rolling optimization parameters
    print(f"\n📊 Rolling Optimization Parameters:")
    print(f"Rebalancing frequency: {RollingOptimizationConfig.REBALANCE_FREQUENCY} days")
    print(f"Minimum history: {RollingOptimizationConfig.MIN_HISTORY} days")
    print(f"Transaction cost: {RollingOptimizationConfig.TRANSACTION_COST:.2%}")
    print(f"Risk-free rate: {RollingOptimizationConfig.RISK_FREE_RATE:.1%}")
    print(
        f"Weight constraints: {RollingOptimizationConfig.MIN_WEIGHT:.1%} - {RollingOptimizationConfig.MAX_WEIGHT:.1%}")

    # Setup logging
    logger = setup_logging()
    logger.info("Rolling optimization configuration loaded successfully")