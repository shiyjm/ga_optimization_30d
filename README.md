# GA Optimization for 30D Functions

这个项目是用 Python 实现的一个从零开始的高性能、模块化的遗传算法 (Genetic Algorithm, GA) 求解器。旨在对 30维的 Rastrigin 函数和 Rosenbrock 函数进行全局最小值寻优。

该算法采用了 Numpy 进行了完全向量化的实现，使其具备极高的工业级计算效率和鲁棒性。

## 目标函数说明

### 1. Rastrigin 函数
Rastrigin 函数是一个典型的非凸函数，拥有大量的局部极小值点，是测试全局寻优算法性能的经典基准函数。
- **数学公式**:
  $$ f(x) = A \cdot n + \sum_{i=1}^n [x_i^2 - A \cdot \cos(2\pi x_i)] $$
  (其中 $A = 10$, 本项目中 $n=30$)
- **全局最优解**: 当 $x = (0, 0, ..., 0)$ 时，获得全局最小值 $f(x) = 0$。
- **定义域边界**: $x_i \in [-5.12, 5.12]$

### 2. Rosenbrock 函数
Rosenbrock 函数 (也称香蕉函数) 的全局最小值位于一个平坦、狭长的抛物线形山谷中。由于山谷非常平缓，很难收敛到最优值。
- **数学公式**:
  $$ f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2] $$
- **全局最优解**: 当 $x = (1, 1, ..., 1)$ 时，获得全局最小值 $f(x) = 0$。
- **定义域边界**: 通常 $x_i \in [-5.0, 10.0]$

## 项目结构
- `functions.py`: 包含了目标函数 Rastrigin 和 Rosenbrock 的向量化 Numpy 实现。
- `ga_solver.py`: 实现了标准的实数编码连续遗传优化算法，包含：锦标赛选择 (Tournament Selection)、算术交叉 (Arithmetic Crossover) 与高斯变异 (Gaussian Mutation) 机制。
- `main.py`: 主控程序，负责读取配置、实例化求解器、执行寻优并绘制相关的适应度收敛图 (`convergence_plot.png`)。
- `requirements.txt`: 运行本项目的依赖列表。

## 如何运行项目

1. **环境准备 (基于 Linux 环境)**：
   建议在隔离的 Python 虚拟环境中运行：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **执行优化程序**:
   ```bash
   python3 main.py
   ```
   程序运行结束后，你将在控制台中看到最佳适应度以及对应解向量的输出。同时在当前目录将生成 `convergence_plot.png` 对收敛过程中的表现进行可视化展示。

## 注意事项
如果在 Linux 运行 `main.py` 绘制收敛图时，出现中文字体显示为空心方块或报警的问题，请先为您得操作系统安装中文字体：
```bash
sudo apt-get update
sudo apt-get install -y fonts-wqy-zenhei fonts-noto-cjk
rm -rf ~/.cache/matplotlib
```
重新运行 `python3 main.py` 即可正常显示。
