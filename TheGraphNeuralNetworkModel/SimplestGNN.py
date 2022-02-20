import torch as t
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import os

# (node, label)集
N = [("n{}".format(i), 0) for i in range(1, 7)] + \
    [("n{}".format(i), 1) for i in range(7, 13)] + \
    [("n{}".format(i), 2) for i in range(13, 19)]
# 边集
E = [("n1", "n2"), ("n1", "n3"), ("n1", "n5"),
     ("n2", "n4"),
     ("n3", "n6"), ("n3", "n9"),
     ("n4", "n5"), ("n4", "n6"), ("n4", "n8"),
     ("n5", "n14"),
     ("n7", "n8"), ("n7", "n9"), ("n7", "n11"),
     ("n8", "n10"), ("n8", "n11"), ("n8", "n12"),
     ("n9", "n10"), ("n9", "n14"),
     ("n10", "n12"),
     ("n11", "n18"),
     ("n13", "n15"), ("n13", "n16"), ("n13", "n18"),
     ("n14", "n16"), ("n14", "n18"),
     ("n15", "n16"), ("n15", "n18"),
     ("n17", "n18")]


class Xi(nn.Module):
    """
    实现Xi函数，输入一个batch的相邻节点特征向量对ln，返回的是s*s的A矩阵
    """

    def __init__(self, ln, s) -> None:
        super().__init__()
        self.ln = ln
        self.s = s
        # linear layer
        self.linear = nn.Linear(2*self.ln, s*s, bias=True)
        # activation layer
        self.tanh = nn.Tanh()

    def forward(self, X: t.Tensor):
        """
        X: (N,2*ln)输入节点特征以及邻居节点特征concat起来
        return : (N,S,S)，输出用于线性变换的参数矩阵
        """
        bs = X.size()[0]
        out = self.linear(X)
        out = self.tanh(out)
        out = out.view(bs, self.s, self.s)
        return out

# function Rou
# 产生bn的网络Rou Omega


class Rou(nn.Module):
    def __init__(self, ln, s) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(ln, s),
            nn.Tanh()
        )

    def forward(self, X: t.Tensor):
        out = self.linear(X)
        return out


class Hw(nn.Module):
    def __init__(self, ln, s, mu=0.9) -> None:
        super().__init__()
        self.ln = ln
        self.s = s
        self.mu = mu
        # init network layers
        self.Xi = Xi(self.ln, self.s)
        self.Rou = Rou(self.ln, self.s)

    def forward(self, X, H, dg_list):
        """
        X (N,2*ln)一个节点特征向量和该节点的某一个邻接向量concatenate得到的向量
        H (N,s)对应中心节点的状态向量
        dg_list : (N,) 对应中心节点的度的向量
        return :(N,s)
        """
        if type(dg_list) == list:
            dg_list = t.tensor(dg_list)
        elif isinstance(dg_list, t.Tensor):
            pass
        else:
            raise TypeError(
                "==> dg_list should be list or tensor, not {}".format(type(dg_list)))
        A = (self.Xi(X)*self.mu/self.s)/dg_list.view(-1, 1, 1)
        b = self.Rou(t.chunk(X, chunks=2, dim=1)[0])
        # (N, s, s) * (N, s) + (N, s)
        out = t.squeeze(t.matmul(A, t.unsqueeze(H, 2)), -1) + b
        return out    # (N, s)


class AggrSum(nn.Module):
    """
    信息聚合
    """

    def __init__(self, node_num: int) -> None:
        super().__init__()
        self.V = node_num

    def forward(self, H: t.Tensor, X_node: t.Tensor) -> t.Tensor:
        # H: (N,s)->(V,s)
        # X_node:(N,)
        mask = t.stack([X_node]*self.V, 0)
        mask = mask.float()-t.unsqueeze(t.arange(0, self.V).float(), 1)
        mask = (mask == 0).float()
        # (V,N)*(N,s)->(V,s)
        return t.mm(mask, H)


class OriLinearGNN(nn.Module):
    """
    The graph neural network model
    """

    def __init__(self, node_num, feat_dim, stat_dim, T) -> None:
        super().__init__()
        """
        初始化模型函数。
        
        :param node_num: 节点数量
        :param feat_dim: 节点特征维度
        :param stat_dim: 节点状态维度
        :param T       : GNN更新轮数
        """
        self.node_num = node_num
        self.feat_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T
        # 初始化节点的嵌入向量，也就是hv，(V,ln)
        self.node_features = nn.parameter.Parameter(
            data=t.randn((self.node_num, self.feat_dim), dtype=t.float32),
            requires_grad=True)
        # 输出层g
        self.linear = nn.Linear(feat_dim+stat_dim, 3)
        self.softmax = nn.Softmax(0)
        # Fw
        self.Hw = Hw(feat_dim, stat_dim)
        # H的分组求和
        self.Aggr = AggrSum(node_num)

    def forward(self, X_Node, X_Neis, dg_list):
        """前向计算函数。值得注意的是，这里输入的X_Node和N_Neis的第一个维度`N`表示边的个数。
        比如：
            X_Node: [0, 0, 0, 1, 1, ..., 18, 18]
            X_Neis: [1, 2, 4, 1, 4, ..., 11, 13]
        :param X_Node: 节点索引
        :param X_Neis: X_node对应节点邻居的索引
        :param dg_list: 节点的度列表
        """
        node_embeds = self.node_features[X_Node]
        neis_embeds = self.node_features[X_Neis]
        X = t.cat((node_embeds, neis_embeds), 1)
        # 初始化节点的状态向量
        node_states = t.zeros((self.node_num, self.stat_dim), dtype=t.float32)
        # 循环t次
        for _ in range(self.T):
            # (V,s)->(N,s)
            H = t.index_select(node_states, 0, X_Node)
            # (N,s)->(N,s)
            H = self.Hw(X, H, dg_list)
            # (N,s)->(V,s)
            node_states = self.Aggr(H, X_Node)
        out = self.linear(t.cat((self.node_features.data, node_states), 1))
        out = self.softmax(out)
        return out  # (V,3)


def CalclulateAccuracy(output, label):
    # output: (N,C)
    # label :(N)
    output = np.argmax(output, axis=1)
    res = output-label
    return list(res).count(0)/len(res)

# training


def train(node_list, edge_list, label_list, T, ndict_path="./data/node_dict.json"):
    if os.path.exists(ndict_path):
        with open(ndict_path) as fp:
            node_dict = json.load(fp)
    else:
        node_dict = dict([(node, ind) for ind, node in enumerate(node_list)])
        node_dict = {"stoi": node_dict,
                     "itos": node_list}
        with open(ndict_path, "w") as fp:
            json.dump(node_dict, fp)

    # 现在需要生成两个向量
    # 第一个向量类似于
    #   [0, 0, 0, 1, 1, ..., 18, 18]
    # 其中的值表示节点的索引，连续相同索引的个数为该节点的度
    # 第二个向量类似于
    #   [1, 2, 4, 1, 4, ..., 11, 13]
    # 与第一个向量一一对应，表示第一个向量节点的邻居节点

    # 得到节点的度 degree
    Degree = dict()
    for n1, n2 in edge_list:
        if n1 in Degree:
            Degree[n1].add(n2)
        else:
            Degree[n1] = {n2}
        if n2 in Degree:
            Degree[n2].add(n1)
        else:
            Degree[n2] = {n1}

    # 生成两个向量
    # 这两个向量就是所谓的adjacency list
    node_inds = []
    node_neis = []
    for n in node_list:
        node_inds += [node_dict["stoi"][n]]*len(Degree[n])
        node_neis += list(map(lambda x: node_dict["stoi"][x], list(Degree[n])))

    # 生成度向量
    dg_list = list(map(lambda x: len(Degree[node_dict["itos"][x]]), node_inds))

    # 训练集和测试集
    train_node_list = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    train_node_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    test_node_list = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    test_node_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # 开始训练
    model = OriLinearGNN(node_num=len(node_list), feat_dim=2, stat_dim=2, T=T)
    optimizer = t.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    loss_func = nn.CrossEntropyLoss()

    min_loss = float("inf")
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    node_inds_tensor = t.tensor(node_inds, dtype=t.long)
    node_neis_tensor = t.tensor(node_neis, dtype=t.long)
    train_label = t.tensor(train_node_label, dtype=t.long)

    for ep in range(50):
        # 运行模型得到结果
        with t.autograd.set_detect_anomaly(True):
            res = model(node_inds_tensor, node_neis_tensor, dg_list)  # (V, 3)
            train_res = t.index_select(
                res, 0, t.Tensor(train_node_list).long())
            test_res = t.index_select(
                res, 0, t.Tensor(test_node_list).long())
            loss = loss_func(input=train_res,
                             target=train_label)
            loss_val = loss.item()
            train_acc = CalclulateAccuracy(
                train_res.cpu().detach().numpy(), np.array(train_node_label))
            test_acc = CalclulateAccuracy(
                test_res.cpu().detach().numpy(), np.array(test_node_label))
            # 更新梯度
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        # 保存loss和acc
        train_loss_list.append(loss_val)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if loss_val < min_loss:
            min_loss = loss_val
        print("==> [Epoch {}] : loss {:.4f}, min_loss {:.4f}, train_acc {:.3f}, test_acc {:.3f}".format(
            ep, loss_val, min_loss, train_acc, test_acc))
    from tensorboardX import SummaryWriter

    sw = SummaryWriter("./SimplestGNN")
    sw.add_graph(model, input_to_model=[
        node_inds_tensor, node_neis_tensor, t.Tensor(dg_list)])
    sw.close()
    return train_loss_list, train_acc_list, test_acc_list


train_loss, train_acc, test_acc = train(node_list=list(map(lambda x: x[0], N)),
                                        edge_list=E,
                                        label_list=list(
                                            map(lambda x: x[1], N)),
                                        T=5)
