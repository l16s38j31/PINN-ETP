import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time

# Step 1: 数据加载函数
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    t = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
    T_in = df['temp'].values
    T_out = df['outdoortemp'].values  # 调整列名
    P = df['Power'].values  # 调整列名，功率单位kW
    
    t_tensor = torch.tensor(t.values, dtype=torch.float32).unsqueeze(1)
    T_in_tensor = torch.tensor(T_in, dtype=torch.float32).unsqueeze(1)
    T_out_tensor = torch.tensor(T_out, dtype=torch.float32).unsqueeze(1)
    P_tensor = torch.tensor(P, dtype=torch.float32).unsqueeze(1)
    
    # 保存用于反归一化的参数
    min_t, max_t = t_tensor.min().item(), t_tensor.max().item()
    min_T_in, max_T_in = T_in_tensor.min().item(), T_in_tensor.max().item()
    # 统一0-1归一化
    for tensor in [t_tensor, T_in_tensor, T_out_tensor, P_tensor]:
        min_val, max_val = tensor.min().item(), tensor.max().item()
        if max_val - min_val != 0:
            tensor[:] = (tensor - min_val) / (max_val - min_val)
    
    
    
    return t_tensor, T_out_tensor, P_tensor, T_in_tensor, min_T_in, max_T_in, min_t, max_t

# Step 2: 数据划分（80%训练，20%测试）
def split_data(t, T_out, P, T_in, train_ratio=0.8):
    split = int(train_ratio * len(t))
    t_train, T_out_train, P_train, T_in_train = t[:split], T_out[:split], P[:split], T_in[:split]
    t_test, T_out_test, P_test, T_in_test = t[split:], T_out[split:], P[split:], T_in[split:]
    return t_train, T_out_train, P_train, T_in_train, t_test, T_out_test, P_test, T_in_test

# Step 3: PINN模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 2)  # 输出: T_e, T_i
        )
        # 参数随机初始化在广范围
        self.R1 = nn.Parameter(torch.rand(1) * 49.0 + 1.0)  # 1-50 °C/kW
        self.R2 = nn.Parameter(torch.rand(1) * 0.9 + 0.1)  # 0.1-1 °C/kW
        self.C1 = nn.Parameter(torch.rand(1) * 4.0 + 0.10)  # 5-50 kWh/°C
        self.C2 = nn.Parameter(torch.rand(1) * 11.9 + 0.1)  # 0.1-1 kWh/°C
        self.eta = nn.Parameter(torch.rand(1) * 2.0 + 1.0)  # 1-3

    def forward(self, t, T_out, P):
        inputs = torch.cat([t, T_out, P], dim=1)
        outputs = self.net(inputs)
        return outputs[:, 0:1], outputs[:, 1:2]  # T_e, T_i

# Step 4: 物理损失函数（基于新ETP模型）
def physics_loss(model, t, T_out, P):
    t.requires_grad = True
    T_e, T_i_pred = model(t, T_out, P)
    dT_i_dt = torch.autograd.grad(T_i_pred.sum(), t, create_graph=True)[0]
    dT_e_dt = torch.autograd.grad(T_e.sum(), t, create_graph=True)[0]
    
    Q = model.eta * P  # 热输入
    res1 = model.C1 * dT_i_dt - (T_out - T_i_pred) / model.R1 - (T_e - T_i_pred) / model.R2 - Q
    res2 = model.C2 * dT_e_dt - (T_i_pred - T_e) / model.R2
    
    return torch.mean(res1**2) + torch.mean(res2**2)

# Step 5: 训练函数
def train_model(model, t_train, T_out_train, P_train, T_in_train, epochs=5000, lr=0.0001, phys_weight=20.0, early_stop_patience=50, early_stop_threshold=1e-5):
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 使用AdamW
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    
    loss_history = []
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, T_i_pred_train = model(t_train, T_out_train, P_train)
        data_loss = torch.mean((T_i_pred_train - T_in_train)**2)
        
        phys_loss = physics_loss(model, t_train, T_out_train, P_train)
        
        total_loss = data_loss + phys_weight * phys_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(total_loss.item())
        
        # 早停检查
        if len(loss_history) > 1:
            loss_diff = abs(loss_history[-1] - loss_history[-2])
            if loss_diff < early_stop_threshold:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} due to loss difference < {early_stop_threshold}")
                    break
            else:
                patience_counter = 0
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Data Loss: {data_loss.item():.4f}, Phys Loss: {phys_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
    
    train_time = time.time() - start_time
    print(f"Training Time: {train_time:.2f} seconds")
    return model, train_time

# Step 6: 测试函数
def test_model(model, t_test, T_out_test, P_test, T_in_test, min_T_in, max_T_in, min_t, max_t):
    start_time = time.time()
    with torch.no_grad():
        _, T_i_pred_test = model(t_test, T_out_test, P_test)
        test_loss = torch.mean((T_i_pred_test - T_in_test)**2)
        print(f"Test Loss (MSE): {test_loss.item():.4f}")
        
        # 反归一化
        T_i_pred_test_denorm = T_i_pred_test * (max_T_in - min_T_in) + min_T_in
        T_in_test_denorm = T_in_test * (max_T_in - min_T_in) + min_T_in
        t_test_denorm = t_test * (max_t - min_t) + min_t
        
        true = T_in_test_denorm.numpy().squeeze()
        pred = T_i_pred_test_denorm.numpy().squeeze()
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        plt.figure(figsize=(12, 6))
        plt.plot(t_test_denorm.numpy(), true, label='True T_in', color='blue')
        plt.plot(t_test_denorm.numpy(), pred, label='Predicted T_in', color='red', linestyle='--')
        plt.legend()
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.title('Test Set Prediction')
        plt.grid(True)
        plt.show()
    
    test_time = time.time() - start_time
    print(f"Testing Time: {test_time:.2f} seconds")
    return test_time

# 多初值初始化
def multi_init_train(t_train, T_out_train, P_train, T_in_train, num_inits=5, epochs=5000, lr=0.0001, phys_weight=20.0):
    best_model = None
    best_loss = float('inf')
    total_train_time = 0
    for i in range(num_inits):
        model = PINN()
        model, train_time = train_model(model, t_train, T_out_train, P_train, T_in_train, epochs, lr, phys_weight)
        total_train_time += train_time
        with torch.no_grad():
            _, T_i_pred = model(t_train, T_out_train, P_train)
            loss = torch.mean((T_i_pred - T_in_train)**2).item()
        if loss < best_loss:
            best_loss = loss
            best_model = model
        print(f"Init {i+1} completed, loss: {loss:.4f}")
    print(f"Total Training Time: {total_train_time:.2f} seconds")
    return best_model, total_train_time

# 主运行
file_path = './data/room2_data.csv'  # 替换为文件路径
t, T_out, P, T_in, min_T_in, max_T_in, min_t, max_t = load_data(file_path)
t_train, T_out_train, P_train, T_in_train, t_test, T_out_test, P_test, T_in_test = split_data(t, T_out, P, T_in)

model, total_train_time = multi_init_train(t_train, T_out_train, P_train, T_in_train, num_inits=5)
test_time = test_model(model, t_test, T_out_test, P_test, T_in_test, min_T_in, max_T_in, min_t, max_t)

print(f"Identified Parameters: R1 = {model.R1.item():.4f}, R2 = {model.R2.item():.4f}, C1 = {model.C1.item():.4f}, C2 = {model.C2.item():.4f}, eta = {model.eta.item():.4f}")