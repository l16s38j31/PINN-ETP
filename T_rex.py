# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:01:43 2025

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

# T-Rex算法主函数
def trex_optimization(T_out, P, T_in_true, initial_T_i, initial_T_e, n_individuals=50, max_iterations=100, early_stop_patience=10, early_stop_threshold=1e-5):
    # 参数范围
    param_bounds = [
        (1.0, 10.0),  # R1: °C/kW
        (0.1, 1.0),   # R2: °C/kW
        (0.1, 0.2),   # C1: kWh/°C
        (5.0, 50.0),  # C2: kWh/°C
        (2.2, 2.2)    # eta: 固定为2.2
    ]
    
    # 初始化种群
    np.random.seed(42)  # 固定随机种子
    n_params = len(param_bounds)
    individuals = np.array([[np.random.uniform(low, high) for low, high in param_bounds] for _ in range(n_individuals)])
    
    # 初始化个体和全局最优
    fitness = np.array([fitness_function(ind, T_out, P, T_in_true, initial_T_i, initial_T_e) for ind in individuals])
    gbest_idx = np.argmin(fitness)
    gbest_position = individuals[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]
    
    # T-Rex参数
    alpha = 0.1  # 探索步长
    beta = 0.05  # 开发步长
    
    # 早停记录
    fitness_history = []
    patience_counter = 0
    
    for iteration in range(max_iterations):
        # 探索阶段
        for i in range(n_individuals):
            step = alpha * np.random.randn(n_params)
            individuals[i] += step
            # 边界约束
            for j in range(n_params):
                individuals[i, j] = np.clip(individuals[i, j], param_bounds[j][0], param_bounds[j][1])
        
        # 开发阶段（围绕全局最优）
        for i in range(n_individuals):
            if np.random.rand() < 0.5:  # 50%概率进行局部搜索
                step = beta * (gbest_position - individuals[i]) + 0.01 * np.random.randn(n_params)
                individuals[i] += step
                for j in range(n_params):
                    individuals[i, j] = np.clip(individuals[i, j], param_bounds[j][0], param_bounds[j][1])
        
        # 计算适应度
        fitness = np.array([fitness_function(ind, T_out, P, T_in_true, initial_T_i, initial_T_e) for ind in individuals])
        
        # 更新全局最优
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = fitness[current_best_idx]
            gbest_position = individuals[current_best_idx].copy()
        
        # 早停检查
        fitness_history.append(gbest_fitness)
        if len(fitness_history) > 1:
            fitness_diff = abs(fitness_history[-1] - fitness_history[-2])
            if fitness_diff < early_stop_threshold:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at iteration {iteration} due to fitness difference < {early_stop_threshold}")
                    break
            else:
                patience_counter = 0
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Fitness (MSE): {gbest_fitness:.4f}")
    
    return gbest_position

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

# 运行T-Rex算法
start_time = time.time()
best_params = trex_optimization(T_out, P, T_in, initial_T_i, initial_T_e)
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