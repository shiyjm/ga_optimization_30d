import numpy as np
from typing import Tuple

def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Rastrigin 函数的批量评估
    
    数学公式:
    f(x) = A * n + sum_{i=1}^n [x_i^2 - A * cos(2 * pi * x_i)]
    其中 A = 10，全局最优解在 x = (0, ..., 0) 处，最小值为 0。
    
    边界条件:
    x_i in [-5.12, 5.12]
    
    Args:
        x (np.ndarray): 输入的种群矩阵，形状为 (pop_size, dimensions)
        
    Returns:
        np.ndarray: 每个个体的适应度值组成的数组，形状为 (pop_size,)
    """
    # 确保输入是 numpy 数组
    x = np.asarray(x)
    
    A = 10.0
    n = x.shape[1] if x.ndim > 1 else x.shape[0]
    
    # 支持单个个体和种群的批量计算
    if x.ndim == 1:
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    else:
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock 函数的批量评估
    
    数学公式:
    f(x) = sum_{i=1}^{n-1} [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    全局最优解在 x = (1, ..., 1) 处，最小值为 0。
    
    边界条件限制:
    通常 x_i in [-5, 10]，此处也支持任意边界输入评估。
    
    Args:
        x (np.ndarray): 输入的种群矩阵，形状为 (pop_size, dimensions)
        
    Returns:
        np.ndarray: 每个个体的适应度值组成的数组，形状为 (pop_size,)
    """
    # 确保输入是 numpy 数组
    x = np.asarray(x)
    
    if x.ndim == 1:
        return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    else:
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, axis=1)

def get_function_info(name: str) -> Tuple[callable, Tuple[float, float]]:
    """
    获取目标函数及其推荐的搜索边界
    
    Args:
        name (str): 函数名称，"rastrigin" 或 "rosenbrock"
        
    Returns:
        Tuple[callable, Tuple[float, float]]: 
            - 目标函数的调用引用
            - 该函数的标准搜索边界 (lower_bound, upper_bound)
    """
    if name.lower() == "rastrigin":
        return rastrigin, (-5.12, 5.12)
    elif name.lower() == "rosenbrock":
        return rosenbrock, (-5.0, 10.0)
    else:
        raise ValueError(f"不支持的函数名: {name}")
