import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from Titanic_loader import TitanicDataset, process_data
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import tqdm




class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super(TitanicMLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16)
        )
        self.layer6 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, data_loader, device, has_labels=True):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in data_loader:
            features = data['features'].to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            
            if has_labels and 'labels' in data:
                labels = data['labels'].to(device)
                all_labels.extend(labels.cpu().numpy().tolist())

    if has_labels and all_labels:
        # 计算准确率和F1分数
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        return accuracy, f1, all_labels, all_predictions
    else:
        # 没有标签时返回None
        return None, None, None, all_predictions


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=100, patience=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    model.to(device)
    
    counter = 0
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm.tqdm(train_loader, file=sys.stdout, ncols=80)
        for data in train_bar:
            features = data['features'].to(device)
            labels = data['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算本epoch的训练损失和准确率
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        
        # 验证集评估
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm.tqdm(val_loader, file=sys.stdout, ncols=80)
            for data in val_bar:
                features = data['features'].to(device)
                labels = data['labels'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算本epoch的验证损失和准确率
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # 更新学习率调度器
        scheduler.step(epoch_val_acc)
                
        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, '
              f'Best Val Acc: {best_val_acc:.4f}')
     
            
    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs    
def visualize_data(labels, title="Dataset"):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(labels)), labels, c='blue', label='Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
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



def visualize_training(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def handle_imbalance(data_df):

    if 'Survived' not in data_df.columns:
        raise ValueError("Dataframe is missing 'Survived' column")
    
    # 分析类别分布
    class_counts = data_df['Survived'].value_counts()
    print(f"Dataset size: {len(data_df)}")
    print(f"Class distribution:\n{class_counts}")
    print(f"Class ratio: 0:1 = {class_counts[0]/class_counts[1]:.2f}:1")
    
    # 计算权重 (基于类别占比)
    total = len(data_df)
    class_weight_0 = total / (2 * class_counts[0])
    class_weight_1 = total / (2 * class_counts[1])
    print(f"Class weights: 0={class_weight_0:.4f}, 1={class_weight_1:.4f}")
    
    # 创建权重张量
    class_weights = torch.tensor([class_weight_0, class_weight_1], dtype=torch.float32)
    
    # 移动到正确的设备
    if device.type == 'cuda':
        class_weights = class_weights.to(device)
    
    # 创建加权损失函数
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    return weighted_criterion

def main():
    # 数据预处理
    train_data, test_data, _ = process_data()
    
    # 可视化训练数据的标签分布

    visualize_data(train_data['Survived'].values, title='Titanic Training Data Survival Distribution')
    
    # 创建训练数据集
    dataset_df = train_data.copy()
    dataset = TitanicDataset(train_data)
    dataset.df = dataset_df  # 设置df属性以便handle_imbalance函数使用
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    # 获取输入特征维度
    sample_batch = next(iter(train_loader))
    input_size = sample_batch['features'].shape[1]
    
    # 创建模型
    model = TitanicMLP(input_size)
    model.to(device)
    
    # 处理类别不平衡并定义损失函数
    criterion = handle_imbalance(dataset.df)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.95, weight_decay=5e-5)  # 调整SGD优化器参数，降低学习率并增加动量
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=3, threshold=0.001
        )
    
    # 训练模型
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=300, patience=20
    )
    
    # 评估模型
    accuracy, f1, _, _ = evaluate(model, val_loader, device)
    print(f"Model evaluation results - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    

    
    # 可视化混淆矩阵
    _, _, all_labels, all_predictions = evaluate(model, val_loader, device)
    visualize_confusion_matrix(all_labels, all_predictions, class_names=['Not Survived', 'Survived'])
    
    # 可视化训练结果
    visualize_training(train_losses, val_losses, train_accs, val_accs)
    
    # 保存模型到bi_class目录下
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_model.pth')
    torch.save(model.state_dict(), model_path)
    
    test_dataset = TitanicDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 在测试集上评估模型 - 测试集可能没有标签
    _, _, _, all_predictions = evaluate(model, test_loader, device, has_labels=False)

    
    # 保存PassengerId用于提交
    passenger_ids = test_data['PassengerId'].astype(int).values
    
    # 提取特征并转换为张量
    test_features_df = test_data.drop(['PassengerId'], axis=1, errors='ignore')
        

        
        
    # 转换所有列为数值类型
    for col in test_features_df.columns:
                test_features_df[col] = pd.to_numeric(test_features_df[col], errors='coerce')
            
    test_features_df = test_features_df.fillna(0)
        
    # 转换为numpy数组并确保类型为float32
    test_features_np = test_features_df.values.astype(np.float32)
        
    # 转换为张量
    test_features = torch.tensor(test_features_np, dtype=torch.float32).to(device)
        
    # 进行预测
    model.eval()
    with torch.no_grad():
            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
        
    # 创建提交文件
    submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predicted.cpu().numpy().astype(int)
        })
        
    # 保存到当前目录
    submission_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Test set prediction results saved to {submission_path}")
    print(f"Successfully predicted {len(submission)} test samples")

if __name__ == '__main__':
    main()

    
    
    
    
    

    
    
   
    
    
    
    
        
        
    
    
    

    
    
    
    
    
    





    
    
    
    
    