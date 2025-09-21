import os
import torch
import torch.nn as nn
import numpy as np
import sys
from tqdm import tqdm
import torch.optim as optim

from moons_loader import MoonsDataset
 

class MoonsMLP(nn.Module):
    # 三层神经网络
    def __init__(self, in_size):
        super(MoonsMLP, self).__init__()
        self.layer1 = nn.Linear(in_size, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 2)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

# 定义计算环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 划分和加载数据集
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'moons.csv')
dataset = MoonsDataset(data_path)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=14, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)





# 定义推理函数，计算并返回准确率
def evaluate(model, data_loader, device):
    model.eval()
    cor_num = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predicted = torch.max(outputs.data, dim=1)[1]
            total += labels.size(0)
            cor_num += (predicted == labels).sum().item()
    accuracy = cor_num / total
    return accuracy

# 训练函数
def train_model(model, train_loader, val_loader, device, learning_rate=0.001, epochs=100, save_path=None):

    # 初始化训练历史记录
    history = {
    'train_accuracy': [],
    'val_accuracy': []
    }
    losses = []  # 保存每轮的损失值
    
    # 确保保存路径存在
    if save_path is not None:
        weight_path = os.path.join(save_path, 'weights')
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
    
    model.train()
    pg=[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=learning_rate)  # Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 训练过程
    for epoch in range(epochs):
        cor_num = torch.zeros(1).to(device)
        sample_num = torch.zeros(1).to(device)
        epoch_loss = 0.0
        
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            labels = labels.squeeze(-1)
            sample_num += labels.size(0)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(data)
            predicted_class = torch.max(outputs.data, dim=1)[1]
            cor_num += (predicted_class == labels).sum().item()
            
            # 计算损失并反向传播
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        # 计算训练准确率
        train_accuracy = cor_num.item() / sample_num.item()
        history['train_accuracy'].append(train_accuracy)
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        
        # 计算验证准确率
        val_accuracy = evaluate(model, val_loader, device)
        history['val_accuracy'].append(val_accuracy)
        
        # 更新进度条描述
        train_bar.desc = f"Train Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # 保存模型权重
    if save_path is not None:
       torch.save(model.state_dict(), os.path.join(weight_path, 'nn.pth'))
            
    
    print("Training complete.")
    return model, history, losses


# 可视化数据集
import matplotlib.pyplot as plt
def visualize_data(data, labels, title="Dataset"):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=80)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
# 可视化训练损失和准确率
def visualize_training(history, losses):
        epochs = len(history['train_accuracy'])
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), history['train_accuracy'], label='Train Accuracy')
        plt.plot(range(epochs), history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
# 可视化决策边界（热力图）
def visualize_decision_boundary(model, data, labels, device):
        # 确定绘图范围
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        # 生成网格点（二维网格）
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        # 整理网格点数据并转换为张量
        grid_data = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_data, dtype=torch.float32).to(device)
        # 预测网格点类别
        model.eval()
        with torch.no_grad():
            outputs = model(grid_tensor)
        _, predicted = torch.max(outputs, 1)
        # 将预测结果转换为与网格形状相同的数组
        Z = predicted.cpu().numpy().reshape(xx.shape)
        # 绘制决策边界和数据点
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolor='k', s=80)
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
        plt.show()


def main():

        # 创建模型
        model = MoonsMLP(in_size=2).to(device)
        
 
        
        # 调用训练函数
        model, history, losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=0.005,
            epochs=100,
            save_path=os.getcwd()
        )
        
        # 测试模型
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.4f}")
      
      
        
        # 可视化数据集
        all_data = dataset.data.numpy()
        all_labels = dataset.labels.numpy()
        visualize_data(all_data, all_labels, title="make_moons Dataset")
                
        # 可视化训练过程
        visualize_training(history, losses)
                
        # 可视化决策边界
        visualize_decision_boundary(model, all_data, all_labels, device)
            
    

if __name__ == "__main__":
    main()
    

       

   
      
