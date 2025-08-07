# final_test.py - 最终测试脚本
from rf_enhanced_production import RFEnhancedPortfolioOptimizer

print("🚀 开始运行修复版优化器...")

try:
    # 创建优化器（使用更少的重新平衡次数来测试）
    optimizer = RFEnhancedPortfolioOptimizer(
        data_path="./",
        rebalance_freq=63,  # 3个月一次，减少计算量
        min_history=252,
        transaction_cost=0.001
    )

    print("✅ 优化器初始化成功")
    print(f"📊 数据信息:")
    print(f"  - 股票数量: {len(optimizer.returns.columns)}")
    print(f"  - 数据期间: {optimizer.returns.index[0].date()} 到 {optimizer.returns.index[-1].date()}")
    print(f"  - RF模型数量: {len(optimizer.rf_models)}")
    print(f"  - 重新平衡次数: {len(optimizer.rebalance_dates)}")

    # 测试单个日期的RF信号生成
    print("\n🧪 测试RF信号生成...")
    test_date = optimizer.rebalance_dates[10]  # 选择中间的一个日期
    print(f"测试日期: {test_date.date()}")

    rf_signals = optimizer.generate_rf_alphas(test_date)
    active_signals = len([a for a in rf_signals if abs(a) > 0])

    print(f"✅ RF信号生成成功!")
    print(f"  - 活跃信号数: {active_signals}/{len(rf_signals)}")
    print(f"  - 平均信号强度: {rf_signals.abs().mean():.4f}")
    print(f"  - 信号范围: [{rf_signals.min():.4f}, {rf_signals.max():.4f}]")

    # 如果信号生成成功，运行完整优化
    if active_signals > 0:
        print("\n🎯 运行完整优化流程...")
        print("⚠️ 这可能需要几分钟时间...")

        results = optimizer.run_rf_enhanced_optimization()

        print("\n✅ 优化完成!")

        # 显示结果摘要
        print("\n📊 结果摘要:")
        for method_name, method_results in results.items():
            if method_results.get('portfolio_values') and len(method_results['portfolio_values']) > 1:
                initial_value = method_results['portfolio_values'][0]
                final_value = method_results['portfolio_values'][-1]
                total_return = (final_value / initial_value - 1) * 100
                num_rebalances = len(method_results.get('rebalance_dates', []))

                print(f"  {method_name}:")
                print(f"    - 总收益: {total_return:.2f}%")
                print(f"    - 最终价值: {final_value:.1f}")
                print(f"    - 重新平衡次数: {num_rebalances}")

        # 生成图表和导出结果
        print("\n📈 导出结果...")
        optimizer.export_rf_enhanced_results()
        optimizer.plot_rf_enhanced_performance()

        print("\n🎉 全部完成! 检查以下文件:")
        print("  - results/ 目录下的CSV文件")
        print("  - plots/ 目录下的PNG图表")

    else:
        print("❌ RF信号生成失败，无法继续")

except Exception as e:
    print(f"❌ 运行失败: {e}")
    import traceback

    traceback.print_exc()

    print("\n🔧 如果遇到问题，检查:")
    print("1. 是否正确替换了 _predict_single_stock_robust 函数")
    print("2. 所有依赖包是否安装完整")
    print("3. 数据文件是否在正确位置")