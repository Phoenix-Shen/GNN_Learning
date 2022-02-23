
# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class DGCNN(nn.Module):
    """
    DCGNN model 类，继承自nn.Module
    """

    def __init__(self, args: dict, output_channels=40) -> None:
        super().__init__()

        self.k = args["k"]

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args["emb_dims"])

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args["emb_dims"], kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(args["emb_dims"]*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args["dropout"])
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args["dropout"])
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        前向传播
        """
        # 提取特征
        # input -> x.shape = [batch_size, 3 , num_points]
        # x.shape = [batch_size, 6, num_points , 20]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)  # x.shape = [batch_size, 64, num_points, 20]
        # max Aggregation [batch_size, 64, num_points]
        x1 = x.max(dim=-1, keepdim=False)[0]

        # x.shape = [batch_size, 128, num_points, 20]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)  # x.shape = [batch_size, 64, num_points, 20]
        # max Aggregation [batch_size, 64, num_points]
        x2 = x.max(dim=-1, keepdim=False)[0]

        # x.shape = [batch_size, 128, num_points, 20]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)  # x.shape = [batch_size, 128, num_points, 20]
        # max Aggregation [batch_size, 128, num_points]
        x3 = x.max(dim=-1, keepdim=False)[0]

        # x.shape = [batch_size, 256, num_points, 20]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)  # x.shape = [batch_size, 256, num_points, 20]
        # max Aggregation [batch_size, 256 ,num_points]
        x4 = x.max(dim=-1, keepdim=False)[0]

        # concatenation operation [batch_size, 512, num_points]
        x = t.cat((x1, x2, x3, x4), dim=1)
        ###################################

        # 分类器
        x = self.conv5(x)  # x.shape = [batch_size, emb_dims, num_points]
        # maxpooling operation [batch_size, emb_dims]
        x1 = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        # avgpooling operation [batch_size, emb_dims]
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        # concatenation operation [batch_size, emb_dims*2]
        x = t.cat((x1, x2), 1)
        # linear layer [batch_size ,512 ]
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        # linear layer [batch_size ,256 ]
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        # linear layer [batch_size ,num_classes ]
        x = self.linear3(x)

        return x


def knn(x: t.Tensor, k: int) -> t.Tensor:
    """
    knn算法
    计算每个点之间的距离，并返回距离最小的前K个点的index
    """

    # input x [batch_size , 3 , num of points]
    # inner.shape = [batch_size, num_points, num_points]
    inner = -2*t.matmul(x.transpose(2, 1), x)
    # xx.shape = [batch_size, 1 , num_points]
    xx = t.sum(x**2, dim=1, keepdim=True)
    # pairwise_distance.shape = [batch_size, num_points, num_points] (broadcast mechanism)
    pairwise_distance = -xx-inner-xx.transpose(2, 1)
    # idx.shape = [batch_size, num_points, K] (selectet K points)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: t.Tensor, k: int, idx=None, dim9=False, cuda=True) -> t.Tensor:
    """
    提取边的特征
    """
    # extract the batch size and the point numbers
    # x.shape is [batch_size , 3 , num of points]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # 使用K近邻算法找到最近的点的下标
    # idx.shape = [batch_size, num_points, K] (selectet K points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)

    device = t.device("cuda" if cuda else "cpu")
    # idx_base.shape = [batch_size, 1, 1]
    # 通过广播机制，对不同batch的数据加上它们对应的idx，便于之后索引
    idx_base = t.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx+idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    # 拉伸成 [Batch*Num,Features] 的形状，然后再根据下标索引idx选出
    # feature.shape = []
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # [batch_size , num_points , k , features] 一个点占据K行，对应着包括自己在内的K个近邻点的属性
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # 然后跟X进行concatenate操作 shape= [batch_size , num_points, k ,features*2]
    feature = t.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # [batchsize, 2*num_dims, num_points, k]
    return feature


def calculate_loss(pred: t.Tensor, label: t.Tensor, smoothing=True) -> t.Tensor:
    """
    计算交叉熵损失，如果需要的话，还可以采取标签平滑化
    """
    label = label.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = t.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot*(1-eps)+(1-one_hot)*eps/(n_class-1)
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot*log_prob).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, label, reduction="mean")

    return loss

# # %% TEST of of the DGCNN model
# from utils import load_settings
# args = load_settings(r"C:\Users\ssk\Desktop\GNN\Code\DGCNN\settings.yaml")
# model = DGCNN(args, 40).to("cuda")

# tensor = t.randn((32, 3, 1024), device="cuda")

# result = model.forward(tensor)
# # %%
