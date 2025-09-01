import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# 二阶 ETP 模型求解函数（用于模拟温度）
def etp_model_2order(state, t, T_out, P, R1, R2, C1, C2, eta=2.2):
    T_i, T_s = state
    Q = eta * P
    dT_i = ((T_out - T_i) / R1 + (T_s - T_i) / R2 + Q) / C1
    dT_s = (T_i - T_s) / (C2 * R2)
    return [dT_i, dT_s]

# 模拟二阶 ETP 温度序列
def simulate_temperature_2order(T_out, P, R1, R2, C1, C2, eta, initial_state, dt=0.25):
    T_i, T_s = initial_state
    T_i_sim = [T_i]
    T_s_sim = [T_s]
    for i in range(1, len(T_out)):
        t_interval = np.linspace(0, dt, 2)
        sol = odeint(etp_model_2order, [T_i, T_s], t_interval, args=(T_out[i-1], P[i-1], R1, R2, C1, C2, eta))
        T_i, T_s = sol[-1]
        T_i_sim.append(T_i)
        T_s_sim.append(T_s)
    return np.array(T_i_sim), np.array(T_s_sim)

# 线性回归参数估计函数（二阶 ETP，简化版）
def linear_regression_etp_2order(T_o, T_i, P, dt=0.25, eta=2.2, R2=5.0, C2=0.15):
    # 数据清洗：插值填补 NaN
    valid = ~np.isnan(T_o) & ~np.isnan(T_i) & ~np.isnan(P)
    T_o = np.interp(np.arange(len(T_o)), np.arange(len(T_o))[valid], T_o[valid])
    T_i = np.interp(np.arange(len(T_i)), np.arange(len(T_i))[valid], T_i[valid])
    P = np.interp(np.arange(len(P)), np.arange(len(P))[valid], P[valid])

    # 近似 T_s：假设 T_s 随 T_i 缓慢变化，使用平滑 T_i 作为 T_s 估计
    T_s = np.convolve(T_i, np.ones(5)/5, mode='valid')  # 简单移动平均
    T_s = np.pad(T_s, (4, 0), mode='edge')[:len(T_i)]  # 填充至原始长度

    # 第一方程离散化：dT_i = (T_o - T_i)/(C1 R1) + (T_s - T_i)/(C1 R2) + eta P / C1
    dT_i = np.diff(T_i) / dt
    y = dT_i
    X = np.column_stack((T_o[:-1] - T_i[:-1], T_s[:-1] - T_i[:-1], eta * P[:-1]))

    # 处理 NaN
    y = np.nan_to_num(y, nan=0.0)
    X = np.nan_to_num(X, nan=0.0)

    # 避免除零
    X[X == 0] = 1e-6

    # 线性回归
    reg = LinearRegression().fit(X, y)
    a, b, c = reg.coef_  # a = 1/(C1 R1), b = 1/(C1 R2), c = 1/C1
    intercept = reg.intercept_  # 应接近 0

    C1 = 1 / c if c != 0 else 0.15
    R1 = 1 / (a * C1) if a * C1 != 0 else 5.0
    # R2, C2 固定，基于物理范围或文献值
    R2 = R2
    C2 = C2

    return R1, R2, C1, C2

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
# 使用线性回归估计二阶 ETP 参数（固定 R2, C2）
R1, R2, C1, C2 = linear_regression_etp_2order(T_out, T_in, P, dt, R2=5.0, C2=0.15)
# 初始状态
initial_state = [T_in[0], T_in[0]]  # 假设初始 T_s 近似 T_i
# 模拟温度序列
T_i_sim, T_s_sim = simulate_temperature_2order(T_out, P, R1, R2, C1, C2, 2.2, initial_state, dt)
# 计算误差
mae = np.mean(np.abs(T_in - T_i_sim))
rmse = np.sqrt(np.mean((T_in - T_i_sim) ** 2))
mape = np.mean(np.abs((T_in - T_i_sim) / (T_in + 1e-6)) * 100)
print(f"Estimated Parameters: R1={R1:.4f}, R2={R2:.4f}, C1={C1:.4f}, C2={C2:.4f}")
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
plt.title('Actual vs Simulated Indoor Temperature (2nd Order ETP)')
plt.grid(True)
plt.show()
end_time = time.time()
print(f"Computation Time: {end_time - start_time:.2f} seconds")