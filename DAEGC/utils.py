# import numpy as np
# import torch
# from sklearn.preprocessing import normalize
#
# from torch_geometric.datasets import Planetoid
#
#
# def get_dataset(dataset):
#     datasets = Planetoid('./dataset', dataset)
#     return datasets
#
# def data_preprocessing(dataset):
#     dataset.adj = torch.sparse_coo_tensor(
#         dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
#     ).to_dense()
#     dataset.adj_label = dataset.adj
#
#     dataset.adj += torch.eye(dataset.x.shape[0])
#     dataset.adj = normalize(dataset.adj, norm="l1")
#     dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)
#
#     return dataset
#
# def get_M(adj):
#     adj_numpy = adj.cpu().numpy()
#     # t_order
#     t=2
#     tran_prob = normalize(adj_numpy, norm="l1", axis=0)
#     M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
#     return torch.Tensor(M_numpy)
#
#

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import normalize

import numpy as np
def get_dataset(dataset):
    x = np.load('./mydataset.npy')
    data = Data(x=torch.from_numpy(x).float())
    return data

def data_preprocessing(dataset, k=10):
    x = dataset.x
    dists = torch.cdist(x, x)  # 计算样本间距离
    knn_idx = torch.topk(dists, k=k, largest=False).indices  # 获取每个样本的前k个近邻
    edge_index = torch.stack([knn_idx.view(-1), torch.arange(len(knn_idx)).repeat_interleave(k)])
    edge_weight = torch.ones(len(edge_index[0]))
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([len(x), len(x)]))
    adj += adj.T  # 构造无向图

    adj = normalize(adj.to_dense(), norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)

    dataset.adj = adj
    dataset.adj_label = adj

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)




