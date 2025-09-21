
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

class TitanicDataset (Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        
        if isinstance(self.data_path, str):
            # 如果传入的是文件路径
            df = pd.read_csv(self.data_path)
        else:
            # 如果传入的是数据框
            df = self.data_path.copy()
        
            
        # 保存原始数据框供后续使用
        self.df = df.copy()
        
        # 提取特征和标签
        if 'Survived' in df.columns:
            # 移除PassengerId和Survived列作为特征
            features_df = df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
            # 确保所有特征都是数值类型
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # 填充任何NaN值
            features_df = features_df.fillna(0)
            # 转换为numpy数组
            self.features = features_df.values.astype(np.float32)
            # 确保标签是整数类型
            self.labels = df['Survived'].values.astype(int)
        else:
            # 测试集没有Survived列
            # 移除PassengerId列作为特征
            features_df = df.drop(['PassengerId'], axis=1, errors='ignore')
            # 确保所有特征都是数值类型
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # 填充任何NaN值
            features_df = features_df.fillna(0)
            # 转换为numpy数组
            self.features = features_df.values.astype(np.float32)
            self.labels = None
        
        
        
    def __len__(self):
            if self.labels is not None:
                return len(self.labels)
            else:
                return len(self.features)
        
        
    
    def __getitem__(self, idx):
        
        if self.labels is not None:
            return {
                'features': torch.tensor(self.features[idx], dtype=torch.float32), 
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            # 测试集没有标签
            return {
                'features': torch.tensor(self.features[idx], dtype=torch.float32)
            }


# 异常值处理函数
def change_outliers(data, col):
    # 确保列是数值类型
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if data[col].count() > 0:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        
        data[col] = np.where(data[col] < lower_bound, data[col].median(),
                            np.where(data[col] > upper_bound, data[col].median(), data[col]))
    
    return data


# 提取头衔函数
def extract_title(name):
    str1 = name.split(',')[1]  # Mr.XXXXX
    str2 = str1.split('.')[0]  # Mr
    str3 = str2.strip()        # 移走空格
    return str3


 # 提取Ticket数值部分
def process_ticket(ticket):
    # 处理缺失值
    if pd.isna(ticket) or ticket == '':
        return 0
    # 提取票号中的数字部分
    numeric_part = ''.join(filter(str.isdigit, str(ticket)))
    return int(numeric_part) if numeric_part else 0


# 数据预处理函数
def process_data(train_path='train.csv', test_path='test.csv'):
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整的文件路径
    train_data_path = os.path.join(current_dir, train_path)
    test_data_path = os.path.join(current_dir, test_path)
    
   
    # 读取数据
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # 合并数据
    full_data = pd.concat([train_data, test_data])
    
    # 异常值处理
    numerical_cols = ['Age', 'SibSp', 'Parch']
    for col in numerical_cols:
        full_data = change_outliers(full_data, col)
    
    # 数据清洗
    # 用平均值代替缺失
    full_data['Age'] = full_data['Age'].fillna(full_data['Age'].mean())
    
    full_data['Ticket'] = full_data['Ticket'].apply(process_ticket)
    # 计算Ticket的平均值，确保处理后没有缺失值
    ticket_mean = full_data['Ticket'].mean()
    # Z-score标准化
    ticket_std = full_data['Ticket'].std()
    if ticket_std > 0:  # 避免除以0的情况
        full_data['Ticket'] = (full_data['Ticket'] - ticket_mean) / ticket_std

    
    # 用众数代替缺失
    full_data['Embarked'] = full_data['Embarked'].fillna(full_data['Embarked'].mode()[0])
    # 用U表示unknown
    full_data['Cabin'] = full_data['Cabin'].fillna('U')
    
    # 特征工程
    # 性别转换为数字
    full_data['Sex'] = full_data['Sex'].map({'male': 0, 'female': 1})
    
    # 对Embarked和Pclass进行one-hot编码
    EmbarkedDf = pd.DataFrame()
    EmbarkedDf = pd.get_dummies(full_data['Embarked'], prefix='Embarked')
    full_data = pd.concat([full_data, EmbarkedDf], axis=1)
    full_data.drop('Embarked', axis=1, inplace=True)
    
    PclassDf = pd.DataFrame()
    PclassDf = pd.get_dummies(full_data['Pclass'], prefix='Pclass')
    full_data = pd.concat([full_data, PclassDf], axis=1)
    full_data.drop('Pclass', axis=1, inplace=True)
    
    # 从名字中提取头衔
    TitleDf = pd.DataFrame()
    TitleDf['Title'] = full_data['Name'].apply(extract_title)
        
    # 映射头衔到更广泛的类别
    Title_mapDict = {
            "Capt": "Officer", "Col": "Officer", "Major": "Officer",
            "Jonkheer": "Royalty", "Don": "Royalty", "Sir": "Royalty",
            "Dr": "Officer", "Rev": "Officer",
            "the Countess": "Royalty", "Dona": "Royalty",
            "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs",
            "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Royalty"
        }
        
    # 处理映射中可能不存在的头衔
    TitleDf["Title"] = TitleDf["Title"].apply(lambda x: Title_mapDict.get(x, "Other"))
    TitleDf = pd.get_dummies(TitleDf["Title"])
    
    # 处理Cabin特征
    CabinDf = pd.DataFrame()
   # 获取客舱首字母
    full_data['Cabin'] = full_data['Cabin'].map(lambda c: c[0] if pd.notnull(c) and c != 'U' else 'U')
    CabinDf = pd.get_dummies(full_data["Cabin"], prefix='Cabin')
    
    # 创建家庭大小特征
    FamilyDf = pd.DataFrame()
    full_data['FamilySize'] = full_data['Parch'] + full_data['SibSp'] + 1
    FamilyDf['FamilySize'] = full_data['FamilySize']
    FamilyDf['Family_Single'] = FamilyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    FamilyDf['Family_Small'] = FamilyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    FamilyDf['Family_Large'] = FamilyDf['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
    
    # 选择特征列
    selected_features = []
    
    # 添加创建的特征数据框
    selected_features.append(TitleDf)
    selected_features.append(PclassDf)
    selected_features.append(FamilyDf)
    
    # 添加原始特征
    selected_features.append(full_data[['Ticket']])
    selected_features.append(full_data[['Sex']])
    selected_features.append(full_data[['Age']])
    selected_features.append(CabinDf)
    selected_features.append(EmbarkedDf)
    
    # 合并所有特征
    full_features = pd.concat(selected_features, axis=1)
    
    # 归一化数值特征
    scaler = MinMaxScaler()
    numeric_cols = full_features.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        full_features[numeric_cols] = scaler.fit_transform(full_features[numeric_cols])
    
    # 将处理后的特征合并回原始数据
    # 保留PassengerId和Survived列
    
    result_data = pd.concat([full_data[['PassengerId']], full_features], axis=1)
    result_data['Survived'] = full_data['Survived']
    
    # 将所有非PassengerId和Survived的列转换为数值类型
    numeric_cols = result_data.columns.difference(['PassengerId', 'Survived'])
    for col in numeric_cols:
        result_data[col] = pd.to_numeric(result_data[col])
    
    
    # 分割回训练集和测试集
    if 'Survived' in result_data.columns:
        train_data = result_data[result_data['Survived'].notna()]
        # 确保Survived是整数类型
        if train_data['Survived'].dtype != 'int':
            train_data['Survived'] = train_data['Survived'].astype(int)
        
        test_data = result_data[result_data['Survived'].isna()]
        test_data = test_data.drop('Survived', axis=1)  # 移除测试集中的NaN Survived列
    else:
        # 如果没有Survived列（纯测试数据）
        train_data = pd.DataFrame()
        test_data = result_data.copy()
    
    # 确保测试集包含PassengerId
    test_data = test_data.drop('Survived', axis=1, errors='ignore')
    return train_data, test_data, scaler


    
       
       
       
       
       
       
       
       
       
       
        
        
        
        
        
        
        
        