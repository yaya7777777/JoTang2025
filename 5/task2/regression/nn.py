import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

from housing_loader import HousingDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt



class HousingRegression(nn.Module):
    # 线性回归模型
    def __init__(self, in_size, out_size):
        super(HousingRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        return self.linear(x)
    

# 定义计算环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 划分和加载数据集
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'housing.csv')
dataset = HousingDataset(data_path)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True)

# 定义推理函数，计算并返回均方误差和决定系数
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    
    criterion = nn.MSELoss()#均方误差损失函数
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            
            # 将数据收集移到循环内部
            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(outputs.cpu().numpy().tolist())
    
    # 计算决定系数 R²
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    losses = total_loss / len(data_loader.dataset)
    return losses, r2, all_targets, all_predictions

# 定义训练函数
def train_model(model, train_loader, val_loader, test_loader, device, epochs=200, learning_rate=0.001, save_path=None):
    
    # 确保保存路径存在
    if save_path is not None:
        weight_path = os.path.join(save_path, 'weights')
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

    model.train()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=learning_rate)  # sgd优化器,更新所有参数
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    val_r2_scores = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=80)
        for features, targets in train_bar:
            features, targets = features.to(device), targets.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(features)
            
            
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * features.size(0)#batch loss
            loss.backward()
            optimizer.step()
            
        
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_r2, _, _ = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        train_bar.desc = f"Train Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val R²: {val_r2:.4f}"
        print(f"Train Epoch [{epoch+1}/{epochs}] Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} Validation R²: {val_r2:.4f}")
        
        # 保存模型权重
        if save_path is not None:
            torch.save(model.state_dict(), os.path.join(weight_path, f'epoch_{epoch+1}.pth'))

    
    test_loss, test_r2, all_targets, all_predictions = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} Test R²: {test_r2:.4f}")
    
    print("Training complete.")
    
    return model, train_losses, val_losses, val_r2_scores, test_loss, test_r2, all_targets, all_predictions

#可视化训练过程
def visualize_training(train_losses, val_losses=None, val_r2_scores=None):
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制R²曲线
    if val_r2_scores:
        plt.subplot(1, 2, 2)
        plt.plot(val_r2_scores, label='Validation R²')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('Validation R² Score')
        plt.ylim(-1, 1)  # R²的取值范围通常在[-∞, 1]
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
# 可视化数据集
def visualize_data(targets, title="Dataset"):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(targets)), targets, c='blue', label='Targets')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(title)
    plt.legend()
    plt.show()
    
# 可视化预测结果
def visualize_predictions(model, data, targets, device):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        predictions = model(data_tensor).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(targets)), targets, c='blue', label='Actual Values')
    plt.scatter(range(len(predictions)), predictions, c='red', label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()
    
# 绘制真实房价vs预测房价的散点图
def visualize_true_vs_predicted(all_targets, all_predictions, r2_score):
    plt.figure(figsize=(10, 10))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    
    # 理想的预测线（取min和max确保覆盖范围）
    min_val = min(min(all_targets), min(all_predictions))
    max_val = max(max(all_targets), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
    
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title(f'True Price vs Predicted Price (R² = {r2_score:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def main():

    input_size = dataset.features.shape[1]  
    
    
    model = HousingRegression(in_size=input_size, out_size=1).to(device)
    epochs = 200
    learning_rate = 0.001
    save_path = os.path.dirname(os.path.abspath(__file__))
    
    model, train_losses, val_losses, val_r2_scores, test_loss, test_r2, all_targets, all_predictions = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path,
        
    )
    
     # 在测试集上评估模型
    test_loss, test_r2, _, _ = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # 可视化训练过程和R²曲线
    visualize_training(train_losses, val_losses, val_r2_scores)
    
    # 可视化数据集
    visualize_data(dataset.targets.numpy(), title='Housing Price Dataset')
    
    # 可视化预测结果
    visualize_predictions(model, dataset.features.numpy(), dataset.targets.numpy(), device)
    
    # 绘制真实房价vs预测房价的散点图
    visualize_true_vs_predicted(all_targets, all_predictions, test_r2)

if __name__ == "__main__":
    
    main()
        
    