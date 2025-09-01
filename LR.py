import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# 一阶ET P模型求解函数（用于模拟温度）
def etp_model_1order(T_i, t, T_out, P, R, C, eta=2.2):
    Q = eta * P
    dT_i = ((T_out - T_i) / R + Q) / C
    return dT_i

def simulate_temperature_1order(T_out, P, R, C, eta, initial_T_i, dt=0.25):
    T_i = initial_T_i
    T_i_sim = [T_i]
    for i in range(1, len(T_out)):
        t_interval = np.linspace(0, dt, 2)  # 自适应步长
        sol = odeint(etp_model_1order, T_i, t_interval, args=(T_out[i-1], P[i-1], R, C, eta))
        T_i = sol[-1][0]  # 提取标量值
        T_i_sim.append(T_i)
    return np.array(T_i_sim)

# 线性回归参数估计函数（一阶ETP）
def linear_regression_etp_1order(T_o, T_i, P, dt=0.25, eta=2.2):
    # 数据清洗：插值填补NaN
    valid = ~np.isnan(T_o) & ~np.isnan(T_i) & ~np.isnan(P)
    T_o = np.interp(np.arange(len(T_o)), np.arange(len(T_o))[valid], T_o[valid])
    T_i = np.interp(np.arange(len(T_i)), np.arange(len(T_i))[valid], T_i[valid])
    P = np.interp(np.arange(len(P)), np.arange(len(P))[valid], P[valid])
    
    # 一阶方程：dT_i = (T_o - T_i)/ (R C) + eta * P / C
    dT_i = np.diff(T_i) / dt
    y = dT_i
    X = np.column_stack((T_o[:-1] - T_i[:-1], eta * P[:-1]))
    
    # 处理NaN
    y = np.nan_to_num(y, nan=0.0)
    X = np.nan_to_num(X, nan=0.0)
    
    # 避免除零：添加小epsilon
    X[X == 0] = 1e-6
    
    # 线性回归
    reg = LinearRegression().fit(X, y)
    a, b = reg.coef_  # a = 1/(R C), b = 1/C
    intercept = reg.intercept_  # 应接近0
    
    C = 1 / b if b != 0 else 0.15  # C = 1/b
    R = 1 / (a * C) if a * C != 0 else 5.0  # R = 1/(a C)
    
    return R, C

# 主运行
file_path = './data/room7_data.csv'  # 替换为您的文件路径
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
t = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
T_in = df['temp'].values
T_out = df['outdoortemp'].values
P = df['Power'].values

dt = 0.25  # 15分钟间隔

start_time = time.time()

# 使用线性回归估计参数
R, C = linear_regression_etp_1order(T_out, T_in, P, dt)

# 初始状态
initial_T_i = T_in[0]

# 模拟温度序列
T_i_sim = simulate_temperature_1order(T_out, P, R, C, 2.2, initial_T_i)

# 计算误差
mae = np.mean(np.abs(T_in - T_i_sim))
rmse = np.sqrt(np.mean((T_in - T_i_sim)**2))
mape = np.mean(np.abs((T_in - T_i_sim) / (T_in + 1e-6)) * 100)

print(f"Estimated Parameters: R={R:.4f}, C={C:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(t, T_in, label='Actual T_in', color='blue')
plt.plot(t, T_i_sim, label='Simulated T_in', color='red', linestyle='--')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Simulated Indoor Temperature')
plt.grid(True)
plt.show()

end_time = time.time()
print(f"Computation Time: {end_time - start_time:.2f} seconds")