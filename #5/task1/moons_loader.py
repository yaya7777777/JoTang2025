from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

class MoonsDataset(Dataset):
    def __init__(self, data_path, n_samples=1000, noise=0.1, random_state=42):
        self.data_path = data_path

        # 自动生成数据集
        if not os.path.exists(self.data_path):
           self._generate_dataset(n_samples, noise, random_state)
        else:
            # 从CSV文件加载数据
            df = pd.read_csv(self.data_path)
            data = df[['feature1', 'feature2']].values
            labels = df['label'].values
            
         
    
    def _generate_dataset(self, n_samples, noise, random_state):
        # 生成数据集
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

        # 创建DataFrame
        df = pd.DataFrame({
            'feature1': x[:, 0],
            'feature2': x[:, 1],
            'label': y
        })
        
        # 保存到CSV文件
        df.to_csv(self.data_path, index=False) # 保存数据集,不保存索引
    
        # 加载生成的数据
        data = df[['feature1', 'feature2']].values
        labels = df['label'].values
        
        # 标准化数据
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)#z值化
            
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
            
    
    
    # 数据集大小
    def __len__(self):
        return len(self.labels)
    
    # 获取数据样本
    def __getitem__(self, idx):
    
        return self.data[idx], self.labels[idx]