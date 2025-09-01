# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 22:08:39 2025

@author: sjliu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
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

# XGBoost模型训练与预测
def train_xgboost(X_train, y_train, X_test, n_estimators=100, learning_rate=0.1):
    start_time = time.time()
    xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    end_time = time.time()
    return y_pred, end_time - start_time

# 结果评估
def evaluate_results(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return mae, rmse, mape

# 主程序
if __name__ == "__main__":
    file_path = './data/room7_data.csv'  # 替换为您的CSV文件路径
    t, T_in, T_out, P = load_data(file_path)
    X_train, T_in_train, X_test, T_in_test, t_test = split_data(t, T_in, T_out, P)

    # 训练与预测
    T_in_pred, runtime = train_xgboost(X_train, T_in_train, X_test)

    # 评估
    mae, rmse, mape = evaluate_results(T_in_test, T_in_pred)
    print(f"XGBoost Results:")
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
    plt.title('XGBoost: Actual vs Predicted Indoor Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('xgboost_prediction.png')
    plt.show()

    # 保存结果
    results = pd.DataFrame({
        'Time': t_test,
        'Actual_T_in': T_in_test,
        'Predicted_T_in': T_in_pred
    })
    results.to_csv('xgboost_results.csv', index=False)