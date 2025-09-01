import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 数据读取
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    t = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0  # 时间（小时）
    T_in = df['temp'].values  # 室内温度 (°C)
    T_out = df['outdoortemp'].values  # 室外温度 (°C)
    P = df['Power'].values  # 加热功率 (kW)
    return t, T_in, T_out, P

# 数据准备为序列（用于LSTM）
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length=10):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_length], self.y[idx+self.seq_length]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

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

# LSTM训练与预测
def train_lstm(X_train, y_train, X_test, y_test, t_test, seq_length=10, epochs=1000, batch_size=32, lr=0.001):
    start_time = time.time()
    
    # 训练数据加载
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型、损失函数和优化器
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x.float())
            loss = criterion(output, batch_y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 测试集预测
    model.eval()
    y_pred = []
    with torch.no_grad():
        # 为测试集生成完整预测序列
        X_full = np.concatenate((X_train[-seq_length:], X_test), axis=0)  # 拼接训练末尾序列
        for i in range(len(X_test)):
            start_idx = i
            end_idx = i + seq_length
            if end_idx > len(X_full):
                break
            seq = X_full[start_idx:end_idx]
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            pred = model(seq).numpy().squeeze()
            y_pred.append(pred)
    
    # 对齐预测与测试集
    y_pred = np.array(y_pred)
    y_test = y_test[:len(y_pred)]  # 截断y_test以匹配y_pred长度
    t_test = t_test[:len(y_pred)]  # 截断t_test以匹配y_pred长度
    print(f"Prediction length: {len(y_pred)}, Test length: {len(y_test)}")
    
    end_time = time.time()
    return y_pred, y_test, t_test, end_time - start_time

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
    
    # 数据划分
    X_train, T_in_train, X_test, T_in_test, t_test = split_data(t, T_in, T_out, P)

    # 训练与预测
    T_in_pred, T_in_test_aligned, t_test_aligned, runtime = train_lstm(X_train, T_in_train, X_test, T_in_test, t_test)

    # 评估
    mae, rmse, mape = evaluate_results(T_in_test_aligned, T_in_pred)
    print(f"LSTM Results:")
    print(f"MAE: {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"MAPE: {mape:.2f}%")
    print(f"Runtime: {runtime:.2f} seconds")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(t_test_aligned, T_in_test_aligned, label='Actual $T_i$', color='blue')
    plt.plot(t_test_aligned, T_in_pred, label='Predicted $T_i$', color='red', linestyle='--')
    plt.xlabel('Time (hours)')
    plt.ylabel('Indoor Temperature (°C)')
    plt.title('LSTM: Actual vs Predicted Indoor Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_prediction.png')
    plt.show()

    # 保存结果
    results = pd.DataFrame({
        'Time': t_test_aligned,
        'Actual_T_in': T_in_test_aligned,
        'Predicted_T_in': T_in_pred
    })
    results.to_csv('lstm_results.csv', index=False)