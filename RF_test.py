# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:57:01 2025

@author: sjliu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 数据读取
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    t = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0  # 时间（小时）
    T_in = df['temp'].values  # 室内温度 (°C)
    T_out = df['outdoortemp'].values  # 室外温度 (°C)
    P = df['Power'].values  # 加热功率 (kW)
    return t, T_in, T_out, P

# 数据划分（80%训练，20%测试）
def split_data(t, T_in, T_out, P, train_ratio=0.8):
    split = int(train_ratio * len(t))
    t_train, T_in_train = t[:split], T_in[:split]
    T_out_train, P_train = T_out[:split], P[:split]
    t_test, T_in_test = t[split:], T_in[split:]
    T_out_test, P_test = T_out[split:], P[split:]
    X_train = np.column_stack((t_train, T_out_train, P_train))
    X_test = np.column_stack((t_test, T_out_test, P_test))
    return X_train, T_in_train, X_test, T_in_test, t_test

# RF模型训练与预测
def train_rf(X_train, y_train, X_test, n_estimators=100):
    start_time = time.time()
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    end_time = time.time()
    return y_pred, end_time - start_time

# 结果评估
def evaluate_results(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true ))) * 100
    return mae, rmse, mape

# 主程序
if __name__ == "__main__":
    file_path = './data/room7_data.csv'  # 替换为您的CSV文件路径
    t, T_in, T_out, P = load_data(file_path)
    X_train, T_in_train, X_test, T_in_test, t_test = split_data(t, T_in, T_out, P)

    # 训练与预测
    T_in_pred, runtime = train_rf(X_train, T_in_train, X_test)

    # 评估
    mae, rmse, mape = evaluate_results(T_in_test, T_in_pred)
    print(f"Random Forest Results:")
    print(f"MAE: {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"MAPE: {mape:.2f}%")
    print(f"Runtime: {runtime:.2f} seconds")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(t_test, T_in_test, label='Actual $T_i$', color='blue')
    plt.plot(t_test, T_in_pred, label='Predicted $T_i$', color='red', linestyle='--')
    plt.xlabel('Time (hours)')
    plt.ylabel('Indoor Temperature (°C)')
    plt.title('Random Forest: Actual vs Predicted Indoor Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('rf_prediction.png')
    plt.show()

    # 保存结果
    results = pd.DataFrame({
        'Time': t_test,
        'Actual_T_in': T_in_test,
        'Predicted_T_in': T_in_pred
    })
    results.to_csv('rf_results.csv', index=False)