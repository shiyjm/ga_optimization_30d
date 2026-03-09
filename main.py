import numpy as np
import matplotlib.pyplot as plt
import os
from functions import get_function_info
from ga_solver import GeneticAlgorithm

# 全局配置 matplotlib 中文字体支持及负号显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d)

def main() -> None:
    """
    主控程序，实例化 GA 求解器，分别对 30维 Rastrigin 和 Rosenbrock 函数进行寻优。
    打印并保存统计结果，绘制收敛曲线并保存为 PDF。
    """
    dim = 30
    pop_size = 100
    max_iter = 500
    num_runs = 10
    
    ensure_dir("figures")
    ensure_dir("tables")
    
    functions_to_test = ["Rastrigin", "Rosenbrock"]
    
    results_tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Optimization Results over 10 independent runs}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Function} & \textbf{Best} & \textbf{Mean} & \textbf{Std} \\",
        r"\midrule"
    ]
    
    plt.figure(figsize=(10, 6))
    
    for func_name in functions_to_test:
        print(f"--- 开始优化 30维 {func_name} 函数 ({num_runs}次独立运行) ---")
        
        func, bounds = get_function_info(func_name)
        
        all_best_fits = []
        all_histories = []
        global_best_sol = None
        global_best_fit = np.inf
        
        for run in range(num_runs):
            ga = GeneticAlgorithm(
                func=func, dim=dim, bounds=bounds,
                pop_size=pop_size, max_iter=max_iter,
                crossover_rate=0.8, mutation_rate=0.2,
                tournament_size=4
            )
            
            best_sol, best_fit, history = ga.optimize()
            all_best_fits.append(best_fit)
            all_histories.append(history)
            
            if best_fit < global_best_fit:
                global_best_fit = best_fit
                global_best_sol = best_sol
                
        mean_fit = np.mean(all_best_fits)
        std_fit = np.std(all_best_fits)
        
        print(f"[{func_name}] 多次运行全局最优适应度值: {global_best_fit:.6e}")
        print(f"[{func_name}] 平均值: {mean_fit:.6e}, 标准差: {std_fit:.6e}\n")
        
        results_tex.append(rf"{func_name} & {global_best_fit:.2e} & {mean_fit:.2e} & {std_fit:.2e} \\")
        
        mean_history = np.mean(all_histories, axis=0)
        plt.plot(range(len(mean_history)), mean_history, label=f"{func_name} Mean Best Fitness")
        
    results_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    with open("tables/results_table.tex", "w") as f:
        f.write("\n".join(results_tex) + "\n")
        
    plt.yscale('log')
    plt.xlabel('迭代次数 (Generations)', fontname='sans-serif')
    plt.ylabel('平均最佳适应度值 (Fitness, Log Scale)', fontname='sans-serif')
    plt.title('遗传算法 (GA) 在 30维函数的平均收敛曲线', fontname='sans-serif')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    plot_file = 'figures/convergence_plot.pdf'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- 优化及统计完成，已输出收敛图: {plot_file}，和表格: tables/results_table.tex ---")

if __name__ == "__main__":
    np.random.seed(42)  # 给定随机种子便于复现
    main()
