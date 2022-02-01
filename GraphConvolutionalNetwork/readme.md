# GCN

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification

Original code can be found at [THERE](https://github.com/tkipf/pygcn)

# CORA DATASET

- 一共 2708 个样本点，，每个样本点都是一篇科学论文，一共有 8 个类别，类别分别是 1）基于案例；2）遗传算法；3）神经网络；4）概率方法；5）强化学习；6）规则学习；7）理论

- 每篇论文都有由一个 1433 维的词向量表示，所以每个样本点的 n_features = 1433，其中向量中的每个元素都对应一个词，只有 0 和 1 两种取值，取 0 表示该元素对应的词不在论文中，取 1 表示在论文中。所有的词来源于一个具有 1433 个词的字典。

- 每篇论文都至少引用了一篇其它论文，或者是被其他论文引用，也就是说，这个引用图是连通的，不存在孤立的点

- 文件格式：

  三个文件 core.cites cora.content README

  content 一共有 2708 行，一行代表一个样本点，每行由 3 部分组成，分别是论文编号，论文词向量，论文的类别

  cites 一共 5429 行，每行有两个论文编号，表示第一个编号的论文先写，第二个编号的论文引用第一个编号的论文
