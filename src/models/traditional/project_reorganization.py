"""
Project Reorganization Script
Automatically reorganizes project structure and removes unnecessary files
"""

import os
import shutil
from pathlib import Path


class ProjectReorganizer:
    """Reorganize project into clean, professional structure."""

    def __init__(self):
        self.project_root = "/Users/sherrie/PycharmProjects/PythonProject/AI-Enhanced-Portfolio-Optimization"
        self.backup_dir = os.path.join(self.project_root, "backup_old_files")

    def create_new_structure(self):
        """Create the new, clean directory structure."""

        print("🏗️  Creating new project structure...")

        # New directory structure
        new_dirs = [
            "src/data_collection",
            "src/data_analysis",
            "src/models/traditional",
            "src/models/lstm",
            "src/models/reinforcement_learning",
            "src/optimization",
            "src/backtesting",
            "src/utils",
            "data/raw",
            "data/processed",
            "data/features",
            "results/charts",
            "results/reports",
            "results/models",
            "docs/literature",
            "docs/methodology",
            "docs/drafts",
            "notebooks/exploration",
            "notebooks/modeling",
            "tests"
        ]

        for dir_path in new_dirs:
            full_path = os.path.join(self.project_root, dir_path)
            os.makedirs(full_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")

    def backup_old_files(self):
        """Backup files before deletion."""

        print("\n💾 Creating backup of old files...")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Files to backup before deletion
        files_to_backup = [
            "banking_data_collector.py",
            "fixed_banking_data_collector.py",
            "fred_data_collector.py",
            "treasury_data_collector.py",
            "treasury_fred_collector.py",
            "quick_data_test.py",
            "test_data_collection.py",
            "data_exploration.py",
            "file_finder.py",
            "simple_data_explorer.py",
            "working_data_explorer.py",
            "absolute_path_explorer.py",
            "final_working_explorer.py",
            "simple_fix.py",
            "rerun_data_collection.py"
        ]

        for filename in files_to_backup:
            source = os.path.join(self.project_root, filename)
            if os.path.exists(source):
                destination = os.path.join(self.backup_dir, filename)
                shutil.copy2(source, destination)
                print(f"📦 Backed up: {filename}")

    def move_files_to_new_structure(self):
        """Move files to their new organized locations."""

        print("\n📁 Moving files to new structure...")

        # File movements mapping
        file_moves = {
            # Data collection scripts
            "complete_data_collection.py": "src/data_collection/",

            # Data analysis scripts
            "ten_year_data_explorer.py": "src/data_analysis/",
            "improved_visualization.py": "src/data_analysis/",

            # Raw data files (10-year versions)
            "data/banking_prices_10y.csv": "data/raw/",
            "data/banking_returns_10y.csv": "data/raw/",
            "data/banking_correlation_10y.csv": "data/raw/",
            "data/banking_volume_10y.csv": "data/raw/",
            "data/banking_rolling_volatility.csv": "data/raw/",
            "data/banking_rolling_betas.csv": "data/raw/",
            "data/fred_economic_data_10y.csv": "data/raw/",
            "data/fred_metadata_10y.txt": "data/raw/",

            # Results and charts
            "results/banking_performance_clean.png": "results/charts/",
            "results/risk_return_analysis.png": "results/charts/",
            "results/economic_indicators_clean.png": "results/charts/",
            "results/correlation_heatmap_clean.png": "results/charts/",
            "results/returns_distribution_clean.png": "results/charts/",
            "results/comprehensive_10year_analysis.png": "results/charts/",
            "results/comprehensive_analysis.png": "results/charts/",
            "results/final_analysis.png": "results/charts/",

            # Reports
            "results/comprehensive_10year_report.txt": "results/reports/",
            "results/comprehensive_analysis_report.txt": "results/reports/",
            "results/final_analysis_report.txt": "results/reports/",
        }

        for source_rel, dest_dir in file_moves.items():
            source = os.path.join(self.project_root, source_rel)
            dest_dir_full = os.path.join(self.project_root, dest_dir)

            if os.path.exists(source):
                filename = os.path.basename(source)
                destination = os.path.join(dest_dir_full, filename)

                try:
                    shutil.move(source, destination)
                    print(f"📁 Moved: {source_rel} → {dest_dir}")
                except Exception as e:
                    print(f"❌ Error moving {source_rel}: {e}")
            else:
                print(f"⚠️  File not found: {source_rel}")

    def delete_obsolete_files(self):
        """Delete obsolete and test files."""

        print("\n🗑️  Deleting obsolete files...")

        files_to_delete = [
            # Old data files (non-10y versions)
            "data/banking_correlation.csv",
            "data/banking_prices.csv",
            "data/banking_returns.csv",
            "data/banking_volume.csv",
            "data/fred_economic_data.csv",
            "data/treasury_complete.csv",
            "data/test_jpm_data.csv",

            # Old scripts (already backed up)
            "banking_data_collector.py",
            "fixed_banking_data_collector.py",
            "fred_data_collector.py",
            "treasury_data_collector.py",
            "treasury_fred_collector.py",
            "quick_data_test.py",
            "test_data_collection.py",
            "data_exploration.py",
            "file_finder.py",
            "simple_data_explorer.py",
            "working_data_explorer.py",
            "absolute_path_explorer.py",
            "final_working_explorer.py",
            "simple_fix.py",
            "rerun_data_collection.py",

            # Old analysis files
            "src/analysis/data_exploration.py",
            "src/analysis/file_finder.py",
            "src/analysis/simple_data_explorer.py",
            "src/analysis/working_data_explorer.py",
        ]

        for file_path in files_to_delete:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    print(f"🗑️  Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
            else:
                print(f"⚠️  File not found (already deleted): {file_path}")

    def remove_empty_directories(self):
        """Remove empty directories."""

        print("\n🧹 Removing empty directories...")

        # Directories that might be empty after cleanup
        dirs_to_check = [
            "src/analysis",
            "src/data_collection/data",
            "data/src",
        ]

        for dir_path in dirs_to_check:
            full_path = os.path.join(self.project_root, dir_path)
            if os.path.exists(full_path) and not os.listdir(full_path):
                try:
                    os.rmdir(full_path)
                    print(f"🗑️  Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"❌ Error removing {dir_path}: {e}")

    def create_readme_files(self):
        """Create README files for each directory."""

        print("\n📝 Creating README files...")

        readme_contents = {
            "src/data_collection/README.md": """# Data Collection
Scripts for collecting financial and economic data.

## Files:
- `complete_data_collection.py`: Main data collection script (10-year dataset)
""",

            "src/data_analysis/README.md": """# Data Analysis
Scripts for exploring and visualizing the collected data.

## Files:
- `ten_year_data_explorer.py`: Comprehensive 10-year data analysis
- `improved_visualization.py`: Clean, separated charts
""",

            "src/models/README.md": """# AI Models
Implementation of various AI models for portfolio optimization.

## Subdirectories:
- `traditional/`: Classical portfolio optimization methods
- `lstm/`: LSTM-based return prediction models  
- `reinforcement_learning/`: RL agents for portfolio allocation
""",

            "data/raw/README.md": """# Raw Data
Original, unprocessed data files from external sources.

## Files:
- Banking sector data (10-year): prices, returns, correlations, volume
- Economic indicators: FRED data with metadata
- Risk metrics: volatility, beta calculations
""",

            "data/processed/README.md": """# Processed Data
Cleaned and preprocessed data ready for model training.
""",

            "results/charts/README.md": """# Charts and Visualizations
Generated charts and graphs for analysis and presentation.
""",

            "results/reports/README.md": """# Analysis Reports
Comprehensive text reports with findings and insights.
""",

            "docs/README.md": """# Documentation
Academic writing, literature review, and methodology documentation.
"""
        }

        for file_path, content in readme_contents.items():
            full_path = os.path.join(self.project_root, file_path)
            with open(full_path, 'w') as f:
                f.write(content)
            print(f"📝 Created: {file_path}")

    def generate_project_summary(self):
        """Generate a summary of the reorganized project."""

        summary_file = os.path.join(self.project_root, "PROJECT_STRUCTURE.md")

        content = """# AI-Enhanced Portfolio Optimization Project Structure

## 📁 Directory Organization

```
AI-Enhanced-Portfolio-Optimization/
├── src/                          # Source code
│   ├── data_collection/          # Data collection scripts
│   ├── data_analysis/            # Data exploration and visualization
│   ├── models/                   # AI model implementations
│   │   ├── traditional/          # Classical optimization methods
│   │   ├── lstm/                 # LSTM neural networks
│   │   └── reinforcement_learning/ # RL agents
│   ├── optimization/             # Portfolio optimization algorithms
│   ├── backtesting/             # Backtesting framework
│   └── utils/                   # Utility functions
├── data/                        # Data storage
│   ├── raw/                     # Original data from sources
│   ├── processed/               # Cleaned data for modeling
│   └── features/                # Engineered features
├── results/                     # Analysis outputs
│   ├── charts/                  # Visualizations and plots
│   ├── reports/                 # Analysis reports
│   └── models/                  # Trained model files
├── docs/                        # Documentation
│   ├── literature/              # Literature review materials
│   ├── methodology/             # Methodology documentation
│   └── drafts/                  # Dissertation drafts
├── notebooks/                   # Jupyter notebooks
│   ├── exploration/             # Data exploration
│   └── modeling/                # Model development
└── tests/                       # Unit tests
```

## 🎯 Current Status

✅ **Data Collection**: Complete 10-year dataset
✅ **Data Analysis**: Comprehensive visualization suite
✅ **Project Structure**: Clean, professional organization
🔄 **Next Phase**: Traditional model implementation

## 📊 Dataset Summary

- **Banking Stocks**: 15 major US banks (2015-2024)
- **Economic Indicators**: 35+ macroeconomic variables
- **Market Data**: Prices, returns, volume, volatility
- **Risk Metrics**: Correlations, betas, drawdowns

## 🚀 Ready for Development

The project is now ready for:
1. Traditional portfolio optimization implementation
2. AI model development (LSTM, RL)
3. Backtesting framework creation
4. Academic writing and documentation
"""

        with open(summary_file, 'w') as f:
            f.write(content)

        print(f"📋 Created project summary: PROJECT_STRUCTURE.md")

    def run_full_reorganization(self):
        """Run the complete reorganization process."""

        print("🚀 AI-Enhanced Portfolio Optimization - Project Reorganization")
        print("=" * 80)

        # Step 1: Create new structure
        self.create_new_structure()

        # Step 2: Backup old files
        self.backup_old_files()

        # Step 3: Move files to new locations
        self.move_files_to_new_structure()

        # Step 4: Delete obsolete files
        self.delete_obsolete_files()

        # Step 5: Remove empty directories
        self.remove_empty_directories()

        # Step 6: Create documentation
        self.create_readme_files()

        # Step 7: Generate project summary
        self.generate_project_summary()

        print("\n" + "=" * 80)
        print("✅ PROJECT REORGANIZATION COMPLETED!")
        print("=" * 80)
        print("🎯 Your project is now professionally organized!")
        print("📁 Clean directory structure with proper categorization")
        print("💾 Old files backed up in: backup_old_files/")
        print("📝 README files created for each directory")
        print("📋 Project structure documented in: PROJECT_STRUCTURE.md")
        print("\n🚀 Ready for next phase: Traditional model implementation!")


def main():
    """Main function to run project reorganization."""
    reorganizer = ProjectReorganizer()
    reorganizer.run_full_reorganization()
    return True


if __name__ == "__main__":
    success = main()