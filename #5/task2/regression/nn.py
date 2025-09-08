import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from housing_loader import HousingDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


class NN(nn.Module):
    # 三层神经网络
    def __init__(self, in_size, hidden_size1, hidden_size2, out_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size2, out_size)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    
    
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

# 定义推理函数，计算并返回均方误差
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()#均方误差损失函数
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
    losses = total_loss / len(data_loader.dataset)
    return losses

# 定义训练函数
def train_model(model, train_loader, val_loader, device, epochs=100, learning_rate=0.001, save_path=None):

    # 确保保存路径存在
    if save_path is not None:
        weight_path = os.path.join(save_path, 'weights')
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

    model.train()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=learning_rate)  # Adam优化器,更新所有参数
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
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
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_bar.desc = f"Train Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"
        print(f"Train Epoch [{epoch+1}/{epochs}] Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}")
        
        # 保存模型权重
        if save_path is not None:
            torch.save(model.state_dict(), os.path.join(weight_path, f'epoch_{epoch+1}.pth'))


   # 测试模型
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    print("Training complete.")
    
    return model, train_losses, val_losses, test_loss

#可视化训练过程
def visualize_training(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
    
# 可视化数据集
def visualize_data(features, targets, title="Dataset"):
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
    

def main():

    input_size = dataset.features.shape[1]  # 输入特征数量
    
    
    model = NN(in_size=input_size, hidden_size1=64, hidden_size2=28, out_size=1).to(device)
    epochs = 100
    learning_rate = 0.001
    save_path = os.path.dirname(os.path.abspath(__file__))
    
    model, train_losses, val_losses, test_loss = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path
    )
    
    visualize_training(train_losses)
    
    visualize_data(dataset.features.numpy(), dataset.targets.numpy(), title='Housing Price Dataset')
    
    visualize_predictions(model, dataset.features.numpy(), dataset.targets.numpy(), device)

if __name__ == "__main__":
    main()
        
    