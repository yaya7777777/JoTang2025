from lzma import FILTER_LZMA1
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.datasets import load_iris

class IrisDataset(Dataset):
    def __init__(self, data_path=None):
        self.data_path = data_path
       
        # 检查CSV文件是否存在，如果不存在则下载数据
        if data_path is not None and os.path.exists(data_path):
            # 从CSV文件加载数据
            df = pd.read_csv(self.data_path)
        else:
            # 使用sklearn加载数据
            iris = load_iris()
            
            # 创建DataFrame
            df = pd.DataFrame(
                    data=iris.data,
                    columns=iris.feature_names
                )
            df['labels'] = iris.target
            
    
            
            df.to_csv(self.data_path, index=False)
        
        features = df.iloc[:, :4].values
        labels = df.iloc[:, -1].values
        
        # 标准化数据
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # z值化
        
        # 保存到实例变量
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
       