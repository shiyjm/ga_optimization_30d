import numpy as np
from typing import Callable, Tuple

class GeneticAlgorithm:
    """
    实数编码的遗传算法实现 (Genetic Algorithm for Continuous Optimization)
    用于求解多维连续函数的最小值。
    """
    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        bounds: Tuple[float, float],
        pop_size: int = 100,
        max_iter: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3
    ):
        """
        初始化 GA 求解器。

        Args:
            func: 目标优化函数 (即要求最小值的函数)，支持向量化输入
            dim: 问题维度 (如 30维)
            bounds: 搜索边界，格式为 (lower_bound, upper_bound)
            pop_size: 种群规模
            max_iter: 最大迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            tournament_size: 锦标赛选择时的竞争者数量
        """
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # 初始化种群
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        # 记录收敛过程，保存每次迭代的最佳适应度
        self.best_fitness_history = []
        self.global_best_solution = None
        self.global_best_fitness = np.inf

    def _evaluate(self) -> np.ndarray:
        """评估当前种群的适应度 (求最小值，因此值越小越好)"""
        fitness = self.func(self.population)
        
        # 更新全局最优
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.global_best_fitness:
            self.global_best_fitness = fitness[min_idx]
            self.global_best_solution = self.population[min_idx].copy()
            
        return fitness

    def _selection(self, fitness: np.ndarray) -> np.ndarray:
        """
        锦标赛选择 (Tournament Selection)
        求极小值问题，适应度值越小，被选中的概率越高。
        """
        selected_indices = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            # 随机挑选 tournament_size 个个体参与竞争
            competitors = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            # 在竞争者中选出适应度(函数值)最小的个体的索引
            best_competitor = competitors[np.argmin(fitness[competitors])]
            selected_indices[i] = best_competitor
            
        return self.population[selected_indices]

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        算术交叉 (Arithmetic Crossover)
        """
        offspring = np.empty_like(parents)
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            # 如果种群为奇数，最后一个个体直接复制
            if i + 1 >= self.pop_size:
                offspring[i] = p1
                continue
                
            p2 = parents[i+1]
            
            if np.random.rand() < self.crossover_rate:
                # 生成随机的交叉比例系数
                alpha = np.random.rand(self.dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
                offspring[i] = c1
                offspring[i+1] = c2
            else:
                offspring[i] = p1
                offspring[i+1] = p2
                
        # 确保交叉后不越界
        return np.clip(offspring, self.lb, self.ub)

    def _mutation(self, offspring: np.ndarray) -> np.ndarray:
        """
        高斯变异 (Gaussian Mutation)
        对个体的每个维度以 mutation_rate 的概率添加高斯噪声。
        """
        # 生成一个与 offspring 结构相同的随机矩阵，判断是否变异
        mutation_mask = np.random.rand(self.pop_size, self.dim) < self.mutation_rate
        
        # 变异步长：搜索空间的的一定比例 (动态或静态均可，此处用静态比例)
        sigma = 0.1 * (self.ub - self.lb)
        # 生成高斯噪声
        noise = np.random.normal(0, sigma, (self.pop_size, self.dim))
        
        # 应用变异
        offspring[mutation_mask] += noise[mutation_mask]
        
        # 确保变异后不越界
        return np.clip(offspring, self.lb, self.ub)

    def optimize(self) -> Tuple[np.ndarray, float, list]:
        """
        执行遗传算法优化过程
        
        Returns:
            Tuple[np.ndarray, float, list]: 
                - 全局最优解向量
                - 最优解对应的函数值
                - 每次迭代的最优值历史记录表
        """
        for iteration in range(self.max_iter):
            # 1. 评估种群
            fitness = self._evaluate()
            
            # 记录本代最佳
            self.best_fitness_history.append(self.global_best_fitness)
            
            # 2. 选择操作
            parents = self._selection(fitness)
            
            # 3. 交叉操作
            offspring = self._crossover(parents)
            
            # 4. 变异操作
            offspring = self._mutation(offspring)
            
            # 精英保留机制 (Elitism): 将种群中表现最差的个体替换为目前的全局最优解
            # 这里为了简单高效，将当前新种群再次评估前，预留一个位置给历史最佳。
            # 通常的做法是在新一代中随机替换一个或强制覆盖第一个。
            offspring[0] = self.global_best_solution
            
            # 更新种群为下一代
            self.population = offspring
            
        # 最后一次评估，确保全局最佳不遗漏最后一代
        self._evaluate()
        
        return self.global_best_solution, self.global_best_fitness, self.best_fitness_history
