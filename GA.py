# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 20:27:14 2025

@author: sjliu
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# ETP模型求解函数
def etp_model(y, t, T_out, P, R1, R2, C1, C2, eta):
    T_i, T_e = y
    Q = eta * P
    dT_i = ((T_out - T_i) / R1 + (T_e - T_i) / R2 + Q) / C1
    dT_e = (T_i - T_e) / R2 / C2
    return [dT_i, dT_e]

def simulate_etp(T_out, P, R1, R2, C1, C2, eta, initial_T_i, initial_T_e, dt=0.25):
    T_i = initial_T_i
    T_e = initial_T_e
    T_i_sim = [T_i]
    for i in range(1, len(T_out)):
        t_interval = np.linspace(0, dt, 2)  # 自适应步长
        y0 = [T_i, T_e]
        sol = odeint(etp_model, y0, t_interval, args=(T_out[i-1], P[i-1], R1, R2, C1, C2, eta))
        T_i, T_e = sol[-1]
        T_i_sim.append(T_i)
    return np.array(T_i_sim)

# 适应度函数（MSE）
def fitness_function(params, T_out, P, T_in_true, initial_T_i, initial_T_e):
    R1, R2, C1, C2, eta = params
    T_i_sim = simulate_etp(T_out, P, R1, R2, C1, C2, eta, initial_T_i, initial_T_e)
    mse = np.mean((T_i_sim - T_in_true) ** 2)
    return mse

# 遗传算法主函数
def genetic_algorithm(T_out, P, T_in_true, initial_T_i, initial_T_e, pop_size=50, generations=100):
    # 参数范围
    param_bounds = [
        (1.0, 10.0),  # R1: °C/kW
        (0.1, 1.0),   # R2: °C/kW
        (0.1, 0.2),   # C1: kWh/°C
        (5.0, 50.0),  # C2: kWh/°C
        (1.0, 3.0)    # eta
    ]
    
    # 初始化种群
    np.random.seed(42)  # 固定随机种子
    population = np.array([
        [np.random.uniform(low, high) for low, high in param_bounds]
        for _ in range(pop_size)
    ])
    
    best_params = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        # 计算适应度
        fitness = np.array([fitness_function(individual, T_out, P, T_in_true, initial_T_i, initial_T_e) for individual in population])
        
        # 选择（锦标赛选择）
        parents = []
        for _ in range(pop_size):
            tournament_indices = np.random.choice(pop_size, 3, replace=False)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])
        parents = np.array(parents)
        
        # 交叉（单点交叉）
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                parent1, parent2 = parents[i], parents[i+1]
                if np.random.rand() < 0.8:  # 交叉概率
                    crossover_point = np.random.randint(1, len(param_bounds))
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])
            else:
                offspring.append(parent1)
        offspring = np.array(offspring)
        
        # 变异
        for i in range(len(offspring)):
            if np.random.rand() < 0.1:  # 变异概率
                mutation_idx = np.random.randint(len(param_bounds))
                low, high = param_bounds[mutation_idx]
                offspring[i][mutation_idx] = np.random.uniform(low, high)
        
        # 下一代
        population = offspring
        
        # 更新最佳解
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_params = population[best_idx]
        
        if gen % 10 == 0:
            print(f"Generation {gen}, Best Fitness (MSE): {best_fitness:.4f}")
    
    return best_params

# 主运行
file_path = './data/room7_data.csv'  # 替换为文件路径
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
t = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
T_in = df['temp'].values
T_out = df['outdoortemp'].values
P = df['Power'].values

# 初始状态
initial_T_i = T_in[0]
initial_T_e = initial_T_i  # 假设T_e初始等于T_i

# 运行遗传算法
start_time = time.time()
best_params = genetic_algorithm(T_out, P, T_in, initial_T_i, initial_T_e)
R1, R2, C1, C2, eta = best_params
print(f"Identified Parameters: R1 = {R1:.4f}, R2 = {R2:.4f}, C1 = {C1:.4f}, C2 = {C2:.4f}, eta = {eta:.4f}")

# 全区间验证
T_i_sim = simulate_etp(T_out, P, R1, R2, C1, C2, eta, initial_T_i, initial_T_e)

# 计算误差
mae = np.mean(np.abs(T_in - T_i_sim))
rmse = np.sqrt(np.mean((T_in - T_i_sim)**2))
mape = np.mean(np.abs((T_in - T_i_sim) / (T_in + 1e-6))) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(t, T_in, label='True T_in', color='blue')
plt.plot(t, T_i_sim, label='Simulated T_in (from Params)', color='red', linestyle='--')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Simulated Indoor Temperature (Full Range)')
plt.grid(True)
plt.show()

train_time = time.time() - start_time
print(f"Total Time: {train_time:.2f} seconds")