"""
File Finder - Help locate data files
"""

import os


def find_files():
    """Find all CSV files in the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"🔍 Starting search from: {current_dir}")

    # Search in multiple possible locations
    search_locations = [
        current_dir,
        os.path.dirname(current_dir),  # parent directory
        os.path.dirname(os.path.dirname(current_dir)),  # grandparent
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data'),  # project_root/data
        '/Users/sherrie/PycharmProjects/PythonProject/AI-Enhanced-Portfolio-Optimization',  # absolute path
        '/Users/sherrie/PycharmProjects/PythonProject/AI-Enhanced-Portfolio-Optimization/data'  # absolute data path
    ]

    target_files = [
        'banking_prices.csv',
        'banking_returns.csv',
        'fred_economic_data.csv',
        'treasury_complete.csv'
    ]

    found_files = {}

    for location in search_locations:
        print(f"\n📁 Checking: {location}")

        if os.path.exists(location):
            print("  ✅ Directory exists")

            # List all files
            try:
                files = os.listdir(location)
                csv_files = [f for f in files if f.endswith('.csv')]

                if csv_files:
                    print(f"  📄 CSV files found: {csv_files}")

                    # Check for our target files
                    for target in target_files:
                        if target in csv_files:
                            full_path = os.path.join(location, target)
                            found_files[target] = full_path
                            print(f"  🎯 FOUND: {target}")
                else:
                    print("  ⚠️  No CSV files found")

            except PermissionError:
                print("  ❌ Permission denied")
        else:
            print("  ❌ Directory does not exist")

    print("\n" + "=" * 60)
    print("📊 SUMMARY OF FOUND FILES:")
    print("=" * 60)

    if found_files:
        for filename, path in found_files.items():
            print(f"✅ {filename}")
            print(f"   📁 Location: {path}")

            # Check file size
            try:
                size_mb = os.path.getsize(path) / (1024 ** 2)
                print(f"   📏 Size: {size_mb:.1f} MB")
            except:
                print(f"   📏 Size: Unknown")

        # Determine the data directory
        data_dirs = set(os.path.dirname(path) for path in found_files.values())
        if len(data_dirs) == 1:
            data_dir = list(data_dirs)[0]
            print(f"\n🎯 DATA DIRECTORY: {data_dir}")
            return data_dir
        else:
            print(f"\n⚠️  Files found in multiple directories: {data_dirs}")
            return list(data_dirs)[0]  # Return first one
    else:
        print("❌ No target files found!")
        return None


if __name__ == "__main__":
    data_directory = find_files()