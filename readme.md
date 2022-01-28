# Graph Neural Network

图的基本概念和 pytorch_geometric 库的使用

## 1.图

### 1.1 什么是图

- 图是表示实体之间的关系的一种数据结构

- 在数学中，图是描述于一组对象的结构，其中某些对象对在某种意义上是“相关的”。这些对象对应于称为顶点的数学抽象（也称为节点或点），并且每个相关的顶点对都称为边（也称为链接或线）。通常，图形以图解形式描绘为顶点的一组点或环，并通过边的线或曲线连接。 图形是离散数学的研究对象之一。

- 图分为有向图和无向图两类，是根据图中的边是否有向来判定(Directed/Undirected edge)

### 1.2 图的表示

- 生活中的图

  - 图片是一种特殊的图，图片的 shape 是(highth,width,channel)可以看做是一个**由 highth,width 组成的图**，每个顶点的属性是 RGB 值
  - 文本也是一种特殊的图，例如 I Love Deep Learning 其中每个单词可以视作一个顶点，它们的表示为**I → Love → Deep → Learning**
  - 分子也是一种图，每个原子视为一个顶点，分子之间的键是边
  - 社交网络是一种图，比如说每个人在社交中认识的人，是一种图
  - 引用图，是一种有向图，我的文章引用别人的文章，也是一张图

- 图 G 是一个有序二元组(V,E)，其中 V 称为顶集(Vertices Set)，E 称为边集(Edges set)，E 与 V 不相交。它们亦可写成 V(G)和 E(G)。其中，顶集的元素被称为顶点(Vertex)，边集的元素被称为边(edge)。

- 在存储上，可以使用数组（邻接矩阵 adjacency matrix）、邻接表、十字链表、邻接多重表等等表示。

### 1.3 相关概念

|                 概念                  | 解释                                                                                                   |
| :-----------------------------------: | ------------------------------------------------------------------------------------------------------ |
|               阶 order                | 顶点的个数                                                                                             |
|             子图 subgraph             | G'=(V',E') 其中 V'、E'分别是 V、E 的子集                                                               |
|               度 degree               | 无向图中节点 v 边的数量称为 v 的度                                                                     |
|       出度、入度 in\out degree        | 有向图中以 v 为起点和以 v 为终点的边的数量分别称为 v 的出度和入度                                      |
|                连通图                 | 无向图中任意节点 i 能够通过边到达节点 j，称为连通图                                                    |
|               连通分量                | 无向图 G 的一个极大连通子图称为 G 的一个连通分量，连通图只有一个连通分量，非连通的无向图有多个连通分量 |
|               强连通图                | 给定有向图 G=（V，E），任意两个节点 uv 都能够相互可达，G 是强连通图                                    |
|               弱连通图                | 如果有向图 G 去掉边的方向后满足无向图的连通标准，那么 G 是弱连通图                                     |
|               最短路径                | 顶点 u 到 v 所经过的最少边的数量称为最短路径                                                           |
|                图直径                 | 对于所有的顶点 uv，图直径是它们最短路径的最大值                                                        |
|               度中心性                | 度中心性=degree(v)/(n-1)其中 n 是顶点的数量                                                            |
| 特征向量中心性 eigenvector centrality | 一个节点的中心性是相邻节点中心性的函数，这是它的基本思想                                               |
|   中介中心性 betweenness centrality   | = 经过该节点的最短路径数/其余两两节点的最短路径数量                                                    |
|         连接中心性 closeness          | = (n-1)/节点到其它节点最短路径之和                                                                     |

---

### 1.4 什么问题需要用到图？

- graph level
  预测整个图的属性，例如分子，苯环、奈环的环数是不一样的
- node-level
  进行语义分割、分类等等
- edge-level
  给定顶点，预测边的属性

### 1.5 图与机器学习

- 要与深度学习连接起来，我们需要把图变成神经网络能够理解的格式
- 图有四个属性：顶点、边、全局信息和连接性
- 问题在于如何表达连接性，最直接的方法就是:可以使用邻接矩阵来表示，但是这个矩阵会非常大，而且它有很多冗余信息（稀疏矩阵）
- 邻接矩阵将行列的顺序交换之后是一样的意义，如何能够保证神经网络输入上述两个矩阵，结果不变呢
- 有一种方法:可以使用邻接列表来表示：详情见[GraphTest](./GraphTest.ipynb)

## 2. GNN

### 2.1 定义

GNN 是对图的所有属性进行可优化的转换，而且能够保留图的对称性

GNN 的输出和输入都是一个图，但是不会改变图的连接性

### 2.2 GNN 的起源

两种动机：一种来自于 CNN，一种来源于图嵌入，所谓嵌⼊，就是对图的节点、边或者⼦图(subgraph)学习得到⼀个低维的向 量表⽰，传统的机器学习⽅法通常基于⼈⼯特征⼯程来构建特征，但是这种⽅法受限于灵活性不⾜、表达能⼒不⾜以及⼯程量过⼤的问题

### 2.3 与传统 NN 的区别

CNN 和 RNN 不能够适当的处理图结构的输入，GNN 采⽤在每个节点上分别传播(propagate)的⽅式进⾏学习，由此忽略了 节点的顺序，相当于 GNN 的输出会随着输⼊的不同⽽不同。

### 2.4 分类

- 图卷积网络和图注意力网络，因为涉及到传播步骤。
- 图的空域网络，常用于动态图
- 图的自编码，该模型通常使用无监督学习的方式
- 图生成网络，因为是生成式网络

### 2.5 目标

GNN 的目标是学习得到一个状态的嵌入向量 hv∈Rs，这个向量包含每个节点的邻居节点信息，其中 hv 表示节点 v 的状态向量，这个向量可以用于产生节点的输出 ov，可以是节点的标签。

### 2.6 原始 GNN 的缺点

- 对不动点使用迭代的方法来更新节点的隐藏状态，效率不高
- 原始 GNN 使用相同的参数，另外在一些边上可能会存在某些信息特征不能够被有效地考虑进去
- 如果学习的是节点的向量表示而不是图的表示，使用不动点方法是不妥当的

### 2.7 GNN 的变体

- Spectral Methods
  - ChebNet
  - 1st-order model
  - Sigle Paramet
  - GCN
- Non-Spectral Methods
  - Neural FPs
  - DCNN
  - GraphSAGE
- Graph Attention Networks
  - GAT
- Gated Graph Neural Networks
  - GGNN
- Graph Long short-term memory
  - Tree LSTM (Child Sum)
  - Tree LSTM (N-ary)

### 2.8 GNN 一般框架

- MPNN (结合了 GNN 和 GCN 的方法)
- NLNN (结合几种 Self-attention 方法)
- GN (结合 MPNN 和 NLNN 以及某些 GNN 的变体)

### 2.9 GNN 的应用

- 物理系统应用：对现实世界中的物理系统进行建模
- 化学和生物应用：计算分子指纹、蛋白质结构等等
- 知识图谱应用：解决跨语言的知识图谱对齐任务
- 图像任务：视觉推理、语义分割、zero-shot and few-hot learning
- 文本任务：文本分类、序列标注、机器翻译、关系抽取、文本生成、关系推理
- 产生式模型：NETGAN、MOLGAN
- 组合优化： 解决在图上的 NP-Hard 问题，TSP、MSP 问题
- 推荐系统

### 2.10 GNN 存在的问题

- Shallow Structure
- Dynamic Graphs
- Non-Structure Scenarios
- Scalability

## 3. 最简单的 GNN

input Graph -> GNN blocks -> Transformed Graph -> Classification layer -> Prediction

- 在 GNN block 中并没有使用到图的结构信息（连通性等信息）,导致结果并不能完全表示信息

## 4. Passing messages between parts of the graph

在 3.的基础上，将边的信息传播给节点，将节点的信息传播给边，我们就实现了图连通性传播。详见[A Gentle Introduction to Graph Neural Networks](https://staging.distill.pub/2021/gnn-intro/?ref=https://githubhelp.com)

## 5. The Graph Neural Network Model -- 第一个 GNN 模型

## 5. references

[A Gentle Introduction to Graph Neural Networks](https://staging.distill.pub/2021/gnn-intro/?ref=https://githubhelp.com)
