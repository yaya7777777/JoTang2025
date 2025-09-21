from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

class MoonsDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    
        # 从CSV文件加载数据
        df = pd.read_csv(self.data_path)
        data = df[['feature1', 'feature2']].values
        labels = df['label'].values
            
        # 标准化数据
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)#z值化
            
        # 保存到实例变量
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
            
        
            
            # 数据集大小
    def __len__(self):
        return len(self.labels)
    
    # 获取数据样本
    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]