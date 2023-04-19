import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import os
import pickle
import pandas as pd
def read_csv_cols(filename, cols):
    # 读取CSV文件，指定要读取的列号
    df = pd.read_csv(filename, usecols=cols)
    # 将DataFrame对象转换为NumPy数组
    data = df.to_numpy()
    # 返回数组
    return data
# 读取两个CSV文件的数据
filename1 = '02癌症换特征加fid加编号.csv'
filename2 = '02合并后的污染源加fid加编号.csv'
cols = [2, 3]  # 读取前两列作为二维坐标
data1 = read_csv_cols(filename1, cols)
data2 = read_csv_cols(filename2, cols)

# 绘制散点图
data = np.concatenate([data1, data2], axis=0)
# 构造KNN图
adj = kneighbors_graph(data, n_neighbors=10, mode='connectivity', include_self=False)
adj = adj + adj.T.multiply(adj.T > adj)  # 对称化

# 制作数据集
train_idx = range(len(data1))
test_idx = range(len(data1), len(data))

train_mask = np.zeros(len(data), dtype=bool)
train_mask[train_idx] = True
test_mask = np.zeros(len(data), dtype=bool)
test_mask[test_idx] = True
labels = np.zeros((len(data),), dtype=int)

# 保存数据集
os.makedirs('./dataset/mydataset/raw', exist_ok=True)
sp.save_npz('./dataset/mydataset/raw/ind.cora.x', sp.csr_matrix(data[train_idx]))
sp.save_npz('./dataset/mydataset/raw/ind.cora.tx', sp.csr_matrix(data[test_idx]))
sp.save_npz('./dataset/mydataset/raw/ind.cora.allx', sp.csr_matrix(data))
np.save('./dataset/mydataset/raw/ind.cora.y', labels[train_idx])
np.save('./dataset/mydataset/raw/ind.cora.ty', labels[test_idx])
np.save('./dataset/mydataset/raw/ind.cora.ally', labels)
pickle.dump({i: adj[i].indices for i in range(adj.shape[0])}, open('./dataset/mydataset/raw/ind.cora.graph', 'wb'))
np.save('./dataset/mydataset/raw/ind.cora.test.index', test_idx)