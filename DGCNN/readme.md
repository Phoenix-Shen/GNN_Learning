# Dynamic Graph CNN for Learning on Point Clouds

代码从 An Tao 那儿重构过来的,[链接在此](https://github.com/AnTao97/dgcnn.pytorch)

## 要解决什么问题？

使用深度学习处理 3D 点云

设计一个直接使用点云作为输入的 CNN 架构，获取足够的局部信息，可使用于分类、分割等任务

## 使用的方法

提出了一个新的神经网络模块 EdgeConv

它是可微的，并且能够嵌入已有的网络架构中

EdgeConv 的优点是：1、包含了局部领域的信息；2、通过堆叠 EdgeConv 层模块或者循环使用，可以提取到全局的形状信息；3、在多层的系统中、特征空间的先对关系包含了语义特征

## 存在的问题

EdgeConv 考虑了点的坐标、与邻域点的距离，忽视了相邻点之间的向量方向，最终还是损失了一部分的局部几何信息。

## Structure

```
DGCNN_cls(
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn5): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv1): Sequential(
    (0): Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (conv2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (conv3): Sequential(
    (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (conv4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (conv5): Sequential(
    (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,), bias=False)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (linear1): Linear(in_features=2048, out_features=512, bias=False)
  (bn6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dp1): Dropout(p=0.5, inplace=False)
  (linear2): Linear(in_features=512, out_features=256, bias=True)
  (bn7): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dp2): Dropout(p=0.5, inplace=False)
  (linear3): Linear(in_features=256, out_features=40, bias=True)
)
```

## 杂谈

- 关于 Batch normalize
  是 2015 年一篇论文中提出的数据归一化方法，往往用在深度神经网络中激活层之前。其作用是可以加强模型训练的收敛速度，使得模型更加稳定，避免梯度爆炸或者是梯度消失，并且起到一定的正则化作用，几乎代替了 Dropout
- tensor.topk-> 求 tensor 中某个维度的前 K 大或者前 K 小的值以及对应的下标
  returns ： 数组最大或者最小的值以及它们的下标，需要两个东西来接收 return 值
- 关于数据集,可以在[这里](./data/LoadModelNet40Data.ipynb)看到一个小例子

## ref

- [dcgnn](https://blog.csdn.net/W1995S/article/details/113747174?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.1&utm_relevant_index=3)

- [dcgnn 中的 KNN 和 EDGECONV](https://blog.csdn.net/weixin_45482843/category_10835196.html)

- [EdgeConv 代码 TensorFlow 版本详解](https://blog.csdn.net/qq_39426225/article/details/101980690)
