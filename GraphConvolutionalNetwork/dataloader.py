# %% DATASETS
from time import time
import torch as t
import numpy as np
import scipy.sparse as sp


def encode_onehot(labels) -> np.ndarray:
    """
    将label变成独热向量
    例如这种[0,0,0,1,0,0]，代表样本是第4类的。
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx: sp.csr_matrix) -> sp.csr_matrix:
    """
    raw-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(path="./GraphConvolutionalNetwork/data/cora/", dataset="cora") -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    load data
    """
    start_time = time()
    print(f"Loading {dataset} dataset")
    # 提取源文件
    idx_features_labels = np.genfromtxt(
        f"{path}{dataset}.content", dtype=np.dtype(str))
    # 将特征向量提取出来
    # shape = [num_nodes, 1(node_idx)+feature_size+1(class)]
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 将类别，也就是标签（label）转换成独热编码如[0,0,0,1,0,0]以便训练
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
    """edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)"""
    # 下面三步操作是将边转换成节点下标连接的形式，而不是源数据中论文编号相连接
    list_map = list(map(idx_map.get, edges_unordered.flatten()))
    edges = np.array(list_map, dtype=np.int32)
    edges = edges.reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # 构建对称的邻接矩阵
    adj = adj+adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj+sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = t.tensor(np.array(features.todense()), dtype=t.float32)
    labels = t.tensor(np.where(labels)[1], dtype=t.long)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = t.LongTensor(idx_train)
    idx_val = t.LongTensor(idx_val)
    idx_test = t.LongTensor(idx_test)
    print(f"finished, used {time()-start_time:.4f}s")
    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.csr_matrix) -> t.Tensor:
    """
    将sp.csr_matrix转成t.sparse.FloatTensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64
                                                         ))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)


# %% TEST for DATALOADER
if __name__ == "__main__":
    data = load_data()
