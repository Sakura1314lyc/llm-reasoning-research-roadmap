import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import random_split

#定义数据集
class CSVDataset(Dataset):
    #导入数据集
    def __init__(self, path):
        df = pd.read_csv(path, header=None)
        # 输入特征
        self.X: NDArray[np.float32] = (
            df.iloc[:, :-1].to_numpy(dtype=np.float32)
        )

        # 原始标签
        raw_y = df.iloc[:, -1].to_numpy()

        # 标签编码，并明确转换成 NumPy 数组
        self.y: NDArray[np.int64] = np.asarray(
            LabelEncoder().fit_transform(raw_y),
            dtype=np.int64
        )

    # 定义获得数据集长度的方法
    def __len__(self):
        return len(self.X)

    # 定义获得某一行数据的方法
    def __getitem__(self, idx : int):
        return [self.X[idx], self.y[idx]]

    # 在类内部定义划分训练集和测试集的方法, 在本例中,训练集比例为0.67, 测试集的比例为0.33
    def get_splits(self, n_test = 0.33):
        # 确定训练集和测试集的尺寸
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        #根据尺寸划分训练集和测试集并返回
        return random_split(self, [train_size, test_size])
    





    
# 定义数据集路径（在本例中，数据集需为 csv 文件）
data_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
# 实例化数据集
dataset = CSVDataset(data_path)
print(f'输入矩阵的形状是：{dataset.X.shape}')
# dataset.X  # 查看输入矩阵 dataset.X
print(f'输出矩阵的形状是：{dataset.y.shape}')
# dataset.y  # 查看输出矩阵
# len() 方法本质上是调用类内部的 __len__() 方法，所以以下方法是等效的。
print(len(dataset))
# dataset[] 方法本质上是调用类内部的 __getitem__ 方法，所以以下方法是等效的。
print(dataset[149])