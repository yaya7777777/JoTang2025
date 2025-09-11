import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

torch.manual_seed(123)


def create_dataloaders():
    train_tf = None # TODO: 定义训练集的数据预处理与增强
    val_tf = None # TODO: 定义验证集的数据预处理

    train_dataset = None # TODO: 加载训练集，并确保应用训练集的 transform
    val_dataset = None # TODO: 加载验证集

    train_loader = None # TODO: 创建训练集 dataloader
    val_loader = None # TODO: 创建验证集 dataloader

    return train_loader, val_loader

