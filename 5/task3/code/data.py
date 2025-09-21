import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

torch.manual_seed(123)

def create_dataloaders(batch_size=32, num_workers=4):
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据目录的绝对路径
    data_dir = os.path.join(current_dir, '..', 'data')
    
    # TODO: 定义训练集的数据预处理与增强
    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),  # 统一尺寸
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),  # 转化为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    
    # 定义验证集的数据预处理
    val_tf = transforms.Compose([
        transforms.Resize((32, 32)),  # 统一尺寸
        transforms.ToTensor(),  # 转化为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    
    # TODO 加载训练集，并确保应用训练集的 transform
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_tf)
      # TODO 加载验证集 
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_tf) 
    
    # TODO: 创建训练集 dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 同时四个程序一起运行
    # TODO: 创建验证集 dataloader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

    return train_loader, val_loader

