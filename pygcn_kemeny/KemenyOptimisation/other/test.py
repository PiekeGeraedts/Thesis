import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from Markov_chain.Markov_chain_new import MarkovChain
from networkx import random_layout


path = "subtract_0.001_"

input = torch.load(path + 'data.pt')
V = input['V_norm']
indices = input['indices']
values = input['values']
size = input['size']
adj = torch.sparse.FloatTensor(indices, values, size)	#this is the adj using GCN aggregation

indices = indices.numpy()
values = values.numpy()
V = V.detach().numpy()

lst = np.where(V < 0.01)[0]
print (adj)
exit()
#print (np.where(V < 0.01)[0].shape[0])


G = nx.DiGraph()	#og graph
G1 = nx.DiGraph()	#optimised graph
edges = np.transpose(indices)

equal_weights = values.copy()
for i in range(size[0]):
	idx = np.where(indices[0] == i)[0]
	n = idx.shape[0]
	equal_weights[idx] = 1/n


cnt=-1
for edge in edges:
	cnt+=1
#	if cnt in lst:
#		print (cnt)
	#	continue
	G.add_nodes_from(edge)
	G1.add_nodes_from(edge)
	G.add_edge(edge[0], edge[1], weight=V[cnt])
	G1.add_edge(edge[0], edge[1], weight=equal_weights[cnt])


#graphviz_layout(G)
Y_colors = []
for i in range(5):
	Y_colors.append('yellow')
for i in range(9):
	Y_colors.append('red')
for i in range(10):
	Y_colors.append('blue')
for i in range(10):
	Y_colors.append('green')

#
Y_colors = np.array(['blue']*34, dtype=object)

#Y_colors[0] = 'yellow'
#Y_colors[4] = 'yellow'
Y_colors[5] = 'yellow'
#Y_colors[6] = 'yellow'
#Y_colors[10] = 'yellow'
#Y_colors[11] = 'yellow'
Y_colors[16] = 'yellow'
#Y_colors[22] = 'yellow'
#Y_colors[23] = 'yellow'
#Y_colors[24] = 'yellow'
#print (Y_colors)
#exit()


#nx.draw(G1, width=equal_weights, node_color=Y_colors)
#plt.savefig(path + 'ZacharysKarateClubGraph.jpg')
#nx.draw_networkx_labels(G, pos=nx.spring_layout(G1))
#plt.show()
#pos=random_layout(nx.spring_layout(G1), seed=42)  # local RNG just for this call
#nx.draw(G1, pos=pos, width=equal_weights, node_color=Y_colors)
nx.draw(G1, pos=nx.spring_layout(G1), width=equal_weights)#, node_color=Y_colors)
#nx.draw_networkx_labels(G, pos=nx.spring_layout(G1))
#plt.savefig(path + 'ZacharysKarateClubGraph_springlayout.jpg')
#plt.show()


nx.draw(G, width=V)
#plt.savefig(path + 'ZacharysKarateClubGraph_opt.jpg')
plt.show()

nx.draw(G, pos=nx.spring_layout(G), width=V)
#plt.savefig(path + 'ZacharysKarateClubGraph_opt_springlayout.jpg')
plt.show()



