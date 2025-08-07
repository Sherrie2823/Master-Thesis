import os
import pandas as pd


def debug_file_paths():
    """调试文件路径问题"""
    print("=" * 60)
    print("调试文件路径问题")
    print("=" * 60)

    # 1. 检查当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 2. 列出当前目录的所有文件和文件夹
    print(f"\n当前目录内容:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"📁 {item}/")
        else:
            print(f"📄 {item}")

    # 3. 检查可能的数据文件夹
    possible_folders = ['data', 'real_data', 'src/data_collection/data']

    print(f"\n检查可能的数据文件夹:")
    for folder in possible_folders:
        if os.path.exists(folder):
            print(f"✅ 找到文件夹: {folder}")
            # 列出文件夹内容
            try:
                files = os.listdir(folder)
                for file in files:
                    if file.endswith('.csv'):
                        print(f"    📄 {file}")
            except:
                print(f"    ❌ 无法读取文件夹内容")
        else:
            print(f"❌ 不存在: {folder}")

    # 4. 递归查找banking_prices.csv
    print(f"\n递归查找 banking_prices.csv:")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'banking_prices.csv':
                full_path = os.path.join(root, file)
                print(f"✅ 找到文件: {full_path}")

                # 尝试读取文件
                try:
                    df = pd.read_csv(full_path, nrows=5)
                    print(f"    文件可读，形状: {df.shape}")
                    print(f"    列名: {list(df.columns)}")
                    return os.path.dirname(full_path)  # 返回文件夹路径
                except Exception as e:
                    print(f"    ❌ 文件读取失败: {e}")

    print("❌ 没有找到 banking_prices.csv 文件")
    return None


def load_data_with_correct_path():
    """使用正确路径加载数据"""

    # 首先调试路径
    data_folder = debug_file_paths()

    if data_folder is None:
        print("无法找到数据文件，请检查文件位置")
        return None, None, None

    print(f"\n使用数据文件夹: {data_folder}")

    try:
        # 加载数据
        prices = pd.read_csv(f'{data_folder}/banking_prices.csv', index_col='Date', parse_dates=True)
        returns = pd.read_csv(f'{data_folder}/banking_returns.csv', index_col='Date', parse_dates=True)
        volume = pd.read_csv(f'{data_folder}/banking_volume.csv', index_col='Date', parse_dates=True)

        print(f"\n✅ 数据加载成功!")
        print(f"✅ 价格数据: {prices.shape}")
        print(f"✅ 收益数据: {returns.shape}")
        print(f"✅ 成交量数据: {volume.shape}")

        return prices, returns, volume

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None, None


if __name__ == "__main__":
    prices, returns, volume = load_data_with_correct_path()

    if prices is not None:
        print("\n🎉 太好了！数据加载成功，现在可以开始特征工程了！")
        print("请运行特征工程代码...")
    else:
        print("\n😭 数据加载失败，需要进一步排查...")