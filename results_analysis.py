# results_analysis.py - 分析结果并进行下一步工作

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


def analyze_results():
    """分析已生成的结果文件"""

    print("📊 开始分析结果...")

    # 1. 寻找结果文件
    results_dir = Path("results")
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        print(f"✅ 找到 {len(csv_files)} 个结果文件:")
        for file in csv_files:
            print(f"  - {file.name}")
    else:
        print("❌ 结果目录不存在")
        return

    # 2. 读取性能指标
    perf_files = [f for f in csv_files if 'performance' in f.name]
    if perf_files:
        perf_df = pd.read_csv(perf_files[0])
        print("\n📈 性能指标:")
        print(perf_df)

        # 比较性能
        if 'Sharpe_Ratio' in perf_df.columns:
            best_sharpe = perf_df.loc[perf_df['Sharpe_Ratio'].idxmax()]
            print(f"\n🏆 最佳Sharpe比率: {best_sharpe['Method']} = {best_sharpe['Sharpe_Ratio']:.3f}")

        if 'Annual_Return' in perf_df.columns:
            best_return = perf_df.loc[perf_df['Annual_Return'].idxmax()]
            print(f"💰 最佳年化收益: {best_return['Method']} = {best_return['Annual_Return']:.1%}")

    # 3. 读取信号分析
    signal_files = [f for f in csv_files if 'signal' in f.name]
    if signal_files:
        signal_df = pd.read_csv(signal_files[0], index_col=0, parse_dates=True)
        print(f"\n🎯 信号分析统计:")
        print(f"  平均活跃信号数: {signal_df['Active_Signals'].mean():.1f}")
        print(f"  平均信号强度: {signal_df['Avg_Alpha_Magnitude'].mean():.4f}")
        print(f"  平均置信度: {signal_df['Avg_Confidence'].mean():.3f}")

    return perf_df if perf_files else None


def create_enhanced_visualizations(perf_df=None):
    """创建增强的可视化图表"""

    print("\n📊 创建增强可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 性能对比图
    if perf_df is not None:
        if 'Sharpe_Ratio' in perf_df.columns:
            # Sharpe比率对比
            colors = ['red' if 'rf_enhanced' in method.lower() else 'blue'
                      for method in perf_df['Method']]

            bars = axes[0, 0].bar(range(len(perf_df)), perf_df['Sharpe_Ratio'], color=colors)
            axes[0, 0].set_title('Sharpe Ratio 比较')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].set_xticks(range(len(perf_df)))
            axes[0, 0].set_xticklabels(perf_df['Method'], rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, perf_df['Sharpe_Ratio']):
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

        # 风险收益散点图
        if 'Annual_Return' in perf_df.columns and 'Volatility' in perf_df.columns:
            for i, (ret, vol, method) in enumerate(zip(perf_df['Annual_Return'],
                                                       perf_df['Volatility'],
                                                       perf_df['Method'])):
                color = 'red' if 'rf_enhanced' in method.lower() else 'blue'
                axes[0, 1].scatter(vol, ret, c=color, s=100, alpha=0.7)
                axes[0, 1].annotate(method, (vol, ret), xytext=(5, 5),
                                    textcoords='offset points', fontsize=8)

            axes[0, 1].set_title('风险-收益分布')
            axes[0, 1].set_xlabel('波动率')
            axes[0, 1].set_ylabel('年化收益率')
            axes[0, 1].grid(True, alpha=0.3)

    # 如果有信号数据，绘制信号强度
    signal_files = list(Path("results").glob("*signal*.csv"))
    if signal_files:
        signal_df = pd.read_csv(signal_files[0], index_col=0, parse_dates=True)
        axes[1, 0].plot(signal_df.index, signal_df['Active_Signals'], 'b-', label='活跃信号数')
        axes[1, 0].set_title('RF信号活动')
        axes[1, 0].set_ylabel('活跃信号数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 信号置信度分布
        axes[1, 1].hist(signal_df['Avg_Confidence'], bins=20, alpha=0.7, color='green')
        axes[1, 1].set_title('信号置信度分布')
        axes[1, 1].set_xlabel('平均置信度')
        axes[1, 1].set_ylabel('频率')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    save_path = Path("results") / "enhanced_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {save_path}")

    plt.show()


def generate_performance_report():
    """生成性能报告"""

    print("\n📋 生成性能报告...")

    # 读取所有结果
    perf_df = None
    results_dir = Path("results")

    if results_dir.exists():
        perf_files = list(results_dir.glob("*performance*.csv"))
        if perf_files:
            perf_df = pd.read_csv(perf_files[0])

    if perf_df is None:
        print("❌ 无法找到性能数据")
        return

    # 创建报告
    report = []
    report.append("=" * 60)
    report.append("AI-Enhanced Portfolio Optimization 性能报告")
    report.append("=" * 60)
    report.append("")

    # 基本统计
    report.append("📊 基本性能统计:")
    for _, row in perf_df.iterrows():
        method = row['Method']
        is_rf = 'rf_enhanced' in method.lower()
        prefix = "🔴 [RF增强]" if is_rf else "🔵 [传统方法]"

        if 'Sharpe_Ratio' in row:
            report.append(f"{prefix} {method}:")
            report.append(f"    Sharpe比率: {row['Sharpe_Ratio']:.3f}")
            if 'Annual_Return' in row:
                report.append(f"    年化收益: {row['Annual_Return']:.1%}")
            if 'Volatility' in row:
                report.append(f"    波动率: {row['Volatility']:.1%}")
            if 'Max_Drawdown' in row:
                report.append(f"    最大回撤: {row['Max_Drawdown']:.1%}")
            report.append("")

    # RF增强效果分析
    rf_methods = perf_df[perf_df['Method'].str.contains('rf_enhanced', case=False)]
    traditional_methods = perf_df[~perf_df['Method'].str.contains('rf_enhanced', case=False)]

    if not rf_methods.empty and not traditional_methods.empty:
        report.append("🎯 RF增强效果分析:")

        if 'Sharpe_Ratio' in perf_df.columns:
            avg_rf_sharpe = rf_methods['Sharpe_Ratio'].mean()
            avg_trad_sharpe = traditional_methods['Sharpe_Ratio'].mean()
            improvement = (avg_rf_sharpe - avg_trad_sharpe) / avg_trad_sharpe * 100

            report.append(f"    RF增强方法平均Sharpe: {avg_rf_sharpe:.3f}")
            report.append(f"    传统方法平均Sharpe: {avg_trad_sharpe:.3f}")
            report.append(f"    改进幅度: {improvement:+.1f}%")
            report.append("")

    # 保存报告
    report_text = "\n".join(report)

    # 打印到控制台
    print(report_text)

    # 保存到文件
    report_file = results_dir / "performance_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ 报告已保存: {report_file}")


def prepare_dissertation_content():
    """为论文准备内容"""

    print("\n📝 准备论文内容...")

    # 1. 结果总结
    results_summary = {
        'rf_signal_generation': True,
        'portfolio_optimization': True,
        'performance_comparison': True,
        'statistical_significance': 'pending',
        'regulatory_compliance': 'theoretical'
    }

    # 2. 需要完成的工作
    next_steps = [
        "1. 统计显著性测试",
        "2. 更多基准方法比较",
        "3. 稳健性分析",
        "4. 风险分析",
        "5. 论文Results章节写作"
    ]

    print("✅ 当前完成状态:")
    for key, status in results_summary.items():
        status_icon = "✅" if status is True else "⚠️" if status == 'pending' else "📋"
        print(f"  {status_icon} {key}: {status}")

    print("\n📋 下一步工作:")
    for step in next_steps:
        print(f"  {step}")

    return results_summary, next_steps


def main():
    """主函数"""
    print("🚀 开始结果分析和下一步规划...")

    # 1. 分析现有结果
    perf_df = analyze_results()

    # 2. 创建可视化
    create_enhanced_visualizations(perf_df)

    # 3. 生成报告
    generate_performance_report()

    # 4. 规划下一步
    results_summary, next_steps = prepare_dissertation_content()

    print("\n🎉 分析完成!")
    print("📊 现在你有了:")
    print("  - 性能对比结果")
    print("  - 可视化图表")
    print("  - 详细报告")
    print("  - 下一步工作计划")


if __name__ == "__main__":
    main()