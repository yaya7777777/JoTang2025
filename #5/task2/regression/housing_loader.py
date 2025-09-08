import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing

class HousingDataset(Dataset):
    def __init__(self, data_path, n_samples=1000, noise=0.2, random_state=42):
        self.data_path = data_path

        # 自动生成数据集
        if not os.path.exists(self.data_path):
           self._generate_dataset()
        else:
            # 从CSV文件加载数据
            df = pd.read_csv(self.data_path)
            features = df.drop(columns=['labels']).values#删去标签列
            targets = df['labels'].values

            # 标准化数据
            features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # z值化
            
            # 保存到实例变量
            self.features = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def _generate_dataset(self):
        # 获取加州房价数据集
        housing = fetch_california_housing()
        data = housing.data
        target = housing.target
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=housing.feature_names)  # 指定列名
        df['labels'] = target
        
        # 保存到CSV文件
        df.to_csv(self.data_path, index=False)  # 保存数据集,不保存索引
        
        # 加载生成的数据
        features = df.drop(columns=['labels']).values
        targets = df['labels'].values


    # 数据集大小
    def __len__(self):
        return len(self.features)

    # 获取数据样本
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
       