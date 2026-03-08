import numpy as np
import matplotlib.pyplot as plt
from functions import get_function_info
from ga_solver import GeneticAlgorithm

# 全局配置 matplotlib 中文字体支持及负号显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
    """
    主控程序，实例化 GA 求解器，分别对 30维 Rastrigin 和 Rosenbrock 函数进行寻优。
    打印结果并绘制收敛曲线。
    """
    dim = 30
    pop_size = 100
    max_iter = 500
    
    # 存储两次不同优化的历史结果以备绘图展示
    histories = {}
    
    functions_to_test = ["Rastrigin", "Rosenbrock"]
    
    for func_name in functions_to_test:
        print(f"--- 开始优化 30维 {func_name} 函数 ---")
        
        # 获取函数调用引用与推荐的标准搜索边界
        func, bounds = get_function_info(func_name)
        
        # 实例化 GA 优化器
        ga = GeneticAlgorithm(
            func=func,
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            crossover_rate=0.8,
            mutation_rate=0.2,   # 给定稍高的变异率跳出局部最优
            tournament_size=4    # 增加选择压力
        )
        
        # 运行优化过程
        best_sol, best_fit, history = ga.optimize()
        
        print(f"[{func_name}] 全局最优适应度值: {best_fit:.6e}")
        print(f"[{func_name}] 全局最优解向量 (前5维): {best_sol[:5]} ...\n")
        
        histories[func_name] = history
        
    # 绘制两者适应度随迭代次数的对比变化曲线
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        # 由于目标函数值差异巨大，建议使用对数变换轴
        plt.plot(range(len(hist)), hist, label=f"{name} Best Fitness")
        
    plt.yscale('log')
    plt.xlabel('迭代次数 (Generations)', fontname='sans-serif')
    plt.ylabel('最佳适应度值 (Fitness, Log Scale)', fontname='sans-serif')
    plt.title('遗传算法 (GA) 在 30维函数的收敛曲线', fontname='sans-serif')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # 保存结果图
    plot_file = 'convergence_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- 优化任务完成，收敛过程对比图已保存至: {plot_file} ---")

if __name__ == "__main__":
    # 为了保证结果复现，可配置随机种子（此处暂且随机以便测试工业适应性）
    np.random.seed(None)
    main()
