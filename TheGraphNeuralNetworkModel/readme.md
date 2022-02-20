# data flow of the GNN

## data definition

x_Node:每条边的起始点 eg:[0,0,1,1,2,3,5,6,4] shape = [edge_num]

x_Neis:每条边的终点 eg:[0,0,1,1,2,3,5,6,4] shape = [edge_num]

dg_list: x_Node 的每个顶点的度数,比如 x_Node 的第一、二个元素是 0，假设度数为 9，那么 dg_list 的元素就是 [9,9,...]

node_feature: 神经网络内嵌的一个参数，它的维度是[node_num,feature_num]

node_states: 节点的状态向量, shape = [node_num,n_states]

## algorithm

1. 取得 x_Node,x_Neis 中顶点所对应的 features 并 concat，即
   ```python
   node_embeds = self.node_features[X_Node]
   neis_embeds = self.node_features[X_Neis]
   ```
2. 初始化节点的状态向量
   ```python
   node_states = t.zeros((self.node_num, self.stat_dim), dtype=t.float32)
   ```
3. 循环 T 次：
   1. 获取 X_Node 中顶点所对应的状态向量
      ```python
      H = t.index_select(node_states, 0, X_Node)
      ```
   2. 更新下一个状态
      ```python
      H = self.Hw(X, H, dg_list)
      ```
   3. 聚合：
      ```python
      node_states = self.Aggr(H, X_Node)
      ```
4. 将节点的 node_features 和 node_states 拼接起来通过全连接层获取结果
   ```python
       out = self.linear(t.cat((self.node_features.data, node_states), 1))
       out = self.softmax(out)
   ```
