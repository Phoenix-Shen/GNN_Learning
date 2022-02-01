import math
import torch as t
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # shape =[in_features, out_features]
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))
        # shape =[out_features]
        if bias:
            self.bias = nn.Parameter(t.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        weight normalization
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: Tensor, adj: Tensor) -> Tensor:
        """
        adjacency_matrix[n_nodes,n_nodes] * 
        (input_features[n_nodes,n_features] * 
        Weight[n_features,out_features] + 
        bias[out_features] = [n_features,out_features]
        """
        support = t.mm(input, self.weight)
        # adj稀疏矩阵在前，dense矩阵在后
        output = t.spmm(adj, support)
        if self.bias is not None:
            return output+self.bias
        else:
            return output

    def __repr__(self):
        """
        自我描述的类
        """
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_hiddens: int,
                 n_class: int,
                 dropout_rate: float) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(
            in_features=n_features, out_features=n_hiddens)
        self.gc2 = GraphConvolution(
            in_features=n_hiddens, out_features=n_class)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x = F.relu(self.gc1.forward(x, adj))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
