from sched import scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import multiclass
import torch
from torch.utils.data import Dataset, DataLoader
import os

from iris_loader import IrisDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns


class IrisLogisticRegression(nn.Module):
    # 逻辑回归模型（多分类）
    def __init__(self, in_size):
        super(IrisLogisticRegression, self).__init__()
        
        self.linear = nn.Linear(in_size, 3)
        
    def forward(self, x):
        
        return self.linear(x)


# 定义计算环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 划分和加载数据集
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris.csv')
dataset = IrisDataset(data_path)

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
    total_acc = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
           
            # 前向传播，获取模型输出
            outputs = model(data)
            # 应用softmax获取概率
            probabilities = torch.softmax(outputs, dim=1)
            # 获取预测类别
            _, predicted = torch.max(probabilities, 1)
            
            # 更新计数
            batch_size = labels.size(0)
            total_samples += batch_size
            total_acc += (predicted == labels).sum().item()
            
            # 收集真实标签和预测标签
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
    accuracy = total_acc / total_samples
    return accuracy, all_labels, all_predictions

# 训练函数
def train_model(model, train_loader, val_loader, device, learning_rate=0.001, epochs=200, save_path=None):
    # 初始化训练历史记录
    history = {
        'train_accuracy': [],
        'val_accuracy': []
    }
    losses = []
    
    # 创建保存权重的文件夹
    if save_path is not None:
        weight_path = os.path.join(save_path, 'weights')
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
    
    model.train()
    pg=[p for p in model.parameters() if p.requires_grad]
    
    #使用Adam优化器
    optimizer = torch.optim.Adam(pg, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
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
        # 只获取evaluate函数的第一个返回值（准确率）
        val_accuracy, _, _ = evaluate(model, val_loader, device)
        history['val_accuracy'].append(val_accuracy)
        
        # 使用scheduler更新学习率
        scheduler.step(val_accuracy)
        
        # 更新进度条描述
        train_bar.desc = f"Train Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # 保存模型权重
    if save_path is not None:
        torch.save(model.state_dict(), os.path.join(weight_path, 'logistic_regression.pth'))
        print(f"Model weights saved to {os.path.join(weight_path, 'logistic_regression.pth')}")
    
    print("Training complete.")
    return model, history, losses


# 可视化数据集 - 使用PCA三维立体图
def visualize_data(data, labels, title="Dataset PCA 3D Visualization"):
    # 使用PCA将原始数据（4维特征）降维到3维
    principal_components = PCA(n_components=3).fit_transform(data)
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
    
    # 绘制3D散点图
    scatter = ax.scatter(principal_components[:, 0], 
                        principal_components[:, 1], 
                        principal_components[:, 2],
                        c=labels, 
                        cmap='viridis', 
                        edgecolor='k', 
                        s=80, 
                        alpha=0.8)
    
    # 设置图表标题和坐标轴标签
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Class')
    
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
    
    feature1_idx = 0
    feature2_idx = 1
    feature3_mean = data[:, 2].mean()#计算特征3的平均值，作为固定值
    feature4_mean = data[:, 3].mean()
    
    # 确定绘图范围
    x_min, x_max = data[:, feature1_idx].min() - 1, data[:, feature1_idx].max() + 1
    y_min, y_max = data[:, feature2_idx].min() - 1, data[:, feature2_idx].max() + 1
    # 生成网格点（二维网格）
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 创建4维的网格数据，前两个维度变化，后两个维度固定为均值
    grid_points = np.zeros((xx.ravel().shape[0], 4))
    grid_points[:, feature1_idx] = xx.ravel()
    grid_points[:, feature2_idx] = yy.ravel()
    grid_points[:, 2] = feature3_mean
    grid_points[:, 3] = feature4_mean
    
    # 转换为张量
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
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
    plt.scatter(data[:, feature1_idx], data[:, feature2_idx], c=labels, edgecolor='k', s=80)
    plt.title('Decision Boundary (Feature 1 vs Feature 2, others fixed to mean)')
    plt.xlabel(f'Feature {feature1_idx+1}')
    plt.ylabel(f'Feature {feature2_idx+1}')
    plt.colorbar()
    plt.show()

# 增加混淆矩阵可视化函数
def visualize_confusion_matrix(true_labels, predicted_labels, class_names=None):
   
    # 生成混淆矩阵
    # 混淆矩阵展示了真实标签与预测标签之间的对应关系
    # 对角线元素表示预测正确的样本数量，非对角线元素表示预测错误的样本数量
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 设置类别名称
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(true_labels)))]
    
    # 创建一个新的图形窗口
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图形式的混淆矩阵
    # annot=True: 在热力图中显示具体数值
    # fmt='d': 使用整数格式显示数值
    # cmap='Blues': 使用蓝色系的颜色映射
    # xticklabels, yticklabels: 设置坐标轴的标签为类别名称
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    # 添加坐标轴标签和图表标题
    plt.xlabel('Predicted Labels')  # x轴表示模型预测的标签
    plt.ylabel('True Labels')       # y轴表示真实的标签
    plt.title('Confusion Matrix')   # 图表标题
    
    plt.show()


def main():
    # 确保数据集已经加载完成
    input_size = dataset.features.shape[1]  # 输入特征数量
    model = IrisLogisticRegression(in_size=input_size).to(device)
    
    # 调整训练参数以提高性能
    learning_rate = 0.01  # 适当提高学习率
    epochs = 500  # 增加训练轮数
    save_path = os.path.dirname(os.path.abspath(__file__))
    
    # 训练模型
    model, history, losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        save_path=save_path
    )
    
    # 测试模型
    test_accuracy, true_labels, predicted_labels = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 可视化训练过程
    visualize_training(history, losses)
    
    # 可视化数据集
    visualize_data(dataset.features.numpy(), dataset.labels.numpy(), title='Iris Dataset PCA 3D Visualization')
    
    # 可视化决策边界
    visualize_decision_boundary(model, dataset.features.numpy(), dataset.labels.numpy(), device)
    
    # 可视化混淆矩阵
    iris_class_names = ['Setosa', 'Versicolor', 'Virginica']
    visualize_confusion_matrix(true_labels, predicted_labels, class_names=iris_class_names)

if __name__ == "__main__":
    main()
    

