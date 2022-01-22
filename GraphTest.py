import pandas as pd
import networkx as nx
import numpy as np

edges = pd.DataFrame()
edges["sources"] = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
edges["targets"] = [2, 4, 5, 3, 1, 2, 5, 1, 5, 1, 3, 4]
edges["weights"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

G = nx.from_pandas_edgelist(edges, source="sources",
                            target="targets", edge_attr="weights")
# 度
print(nx.degree(G))
# 连通分量
print(list(nx.connected_components(G)))
# 图直径
print(nx.diameter(G))
# 度中心性
print(nx.degree_centrality(G))
# 特征向量中心性
print(nx.eigenvector_centrality(G))
# 中介中心性
print(nx.betweenness_centrality(G))
# 链接中心性
print(nx.closeness_centrality(G))
# Pagerank
print(nx.pagerank(G))
# HITS
print(nx.hits(G))
