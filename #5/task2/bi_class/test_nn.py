import torch
import torch.nn as nn
import torch.optim as optim
from nn import TitanicMLP, train_model, evaluate
import numpy as np

# 创建一个简单的测试数据集
def create_test_dataset():
    # 随机生成100个样本，每个样本有10个特征
    X = torch.randn(100, 10)
    # 二分类标签
    y = torch.randint(0, 2, (100,))
    
    # 分割训练集和验证集
    train_X, val_X = X[:80], X[80:]
    train_y, val_y = y[:80], y[80:]
    
    # 创建数据集和数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader

# 测试代码
if __name__ == "__main__":
    print("开始测试nn.py文件...")
    
    # 创建测试数据
    train_loader, val_loader = create_test_dataset()
    
    # 实例化模型
    input_size = 10
    model = TitanicMLP(input_size)
    print("模型创建成功")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # 训练模型（只训练1个epoch进行测试）
        print("开始训练模型...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=1
        )
        print("模型训练成功！")
        print(f"训练损失: {train_losses[-1]:.4f}")
        print(f"验证损失: {val_losses[-1]:.4f}")
        print(f"训练准确率: {train_accs[-1]:.4f}")
        print(f"验证准确率: {val_accs[-1]:.4f}")
        
        # 评估模型
        print("评估模型...")
        accuracy, f1, _, _ = evaluate(model, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"评估准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        print("nn.py文件测试成功！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()