import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing
import os

class HousingDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        # 检查CSV文件是否存在，如果不存在则下载数据
        if not os.path.exists(self.data_path):
            california_housing = fetch_california_housing()
            
            # 创建DataFrame
            df = pd.DataFrame(
                data=california_housing.data,
                columns=california_housing.feature_names
            )
            df['labels'] = california_housing.target
            
            # 保存到CSV文件
            df.to_csv(self.data_path, index=False)
           
        
        # 从CSV文件加载数据
        df = pd.read_csv(self.data_path)
        features = df.drop(columns=['labels']).values#删去标签列
        targets = df['labels'].values

        # 标准化数据
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # z值化
            
        # 保存到实例变量
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)


    # 数据集大小
    def __len__(self):
        return len(self.features)

    # 获取数据样本
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
       