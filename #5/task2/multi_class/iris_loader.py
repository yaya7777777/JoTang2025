from lzma import FILTER_LZMA1
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.datasets import load_iris

class IrisDataset(Dataset):
    def __init__(self, data_path=None, n_sample=100, noise=0.2, random_state=42):
        self.data_path = data_path
          
        
        if data_path is not None and os.path.exists(data_path):
            
            self.load_from_csv()
        else:
            
            self.generate_data()
            
            
            
    def generate_data(self):
        # 从sklearn加载鸢尾花数据集
        iris = load_iris()
        # 使用全部四个特征来提高模型准确率
        features = iris.data[:, :4]
        labels = iris.target
        
        # 标准化特征
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # 转换为PyTorch张量
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # 如果数据路径不存在且用户提供了路径，可以保存生成的数据到CSV
        if self.data_path is not None:
            # 创建DataFrame并保存
            df = pd.DataFrame({
                'feature1': features[:, 0],
                'feature2': features[:, 1],
                'feature3': features[:, 2],
                'feature4': features[:, 3],
                'label': labels
            })
            df.to_csv(self.data_path, index=False)
            print(f"Generated Iris dataset with 4 features saved to {self.data_path}")
    
    def load_from_csv(self):
        # 从CSV文件加载数据
        df = pd.read_csv(self.data_path)
        

        self.features = torch.tensor(df.iloc[:, :4].values, dtype=torch.float32)
  
        
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 返回正确的属性名：features和labels
        return self.features[idx], self.labels[idx]
       