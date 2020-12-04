import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

A = torch.FloatTensor(np.load('A.npy')).to_sparse()
weights = A._values().numpy()	#symmetrically weighted
indices = A._indices().numpy()
weights1 = np.load('weights.npy')	#results from optimising Kemeny --lr 0.01 --epochs 200 --eps 0.001

n = 3
for i in range(n):
	node = np.random.randint(100)
	print (f'###{node}###')
	print ('initial dist.:\n', weights[np.where(indices[0]==node)])
	print ('optimised dist.:\n', weights1[np.where(indices[0]==node)])
	print ('==================')


G = nx.DiGraph()	#og graph
G1 = nx.DiGraph()	#optimised graph
edges = np.transpose(indices)

cnt=0
for edge in edges:
	G.add_nodes_from(edge)
	G1.add_nodes_from(edge)
	G.add_edge(edge[0], edge[1], weight=weights[cnt])
	G1.add_edge(edge[0], edge[1], weight=weights1[cnt])
	cnt+=1


#graphviz_layout(G)

nx.draw(G, width=weights)
plt.show()
nx.draw(G, pos=nx.spring_layout(G), width=weights)
plt.show()


nx.draw(G1, width=weights)
plt.show()
nx.draw(G1, pos=nx.spring_layout(G1), width=weights)
plt.show()










