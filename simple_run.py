
# simple_run.py - 简单运行脚本
from rf_enhanced_production import RFEnhancedPortfolioOptimizer

print("🚀 开始运行修复版优化器...")

try:
    # 创建优化器
    optimizer = RFEnhancedPortfolioOptimizer(
        data_path="./",
        rebalance_freq=42,  # 2个月重新平衡一次，减少计算量
        min_history=252,
        transaction_cost=0.001
    )

    print("✅ 优化器初始化成功")
    print(f"📅 总重新平衡次数: {len(optimizer.rebalance_dates)}")

    # 只运行前5个重新平衡期，测试是否正常
    print("\n🧪 测试前5个重新平衡期...")

    test_results = optimizer.run_rf_enhanced_optimization()

    if test_results:
        print("\n✅ 测试运行成功!")
        print("\n📊 现在可以运行完整流程:")
        print("optimizer.run_rf_enhanced_optimization()")

        # 显示一些基本结果
        for method_name, results in test_results.items():
            if results.get('portfolio_values'):
                final_value = results['portfolio_values'][-1]
                print(f"  {method_name}: 最终价值 = {final_value:.1f}")
    else:
        print("❌ 测试运行失败")

except Exception as e:
    print(f"❌ 运行失败: {e}")
    import traceback
    traceback.print_exc()
