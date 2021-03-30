"""
Date created: 
	18-01-2020
Purpose:
	Train a GCN with features X=I on a random graph. See if the (trained) output embeddings become linearly separable.
"""

import torch
import numpy as np
from models import GCN
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import torch.optim as optim
import torch.nn.functional as F
from utils import accuracy


def set_colors(Y):
	Y_colors = []
	for num in Y:
		if num == 0:
			Y_colors.append('yellow')
		elif num == 1:
			Y_colors.append('blue')
		elif num == 2:
			Y_colors.append('green')
		elif num == 3:
			Y_colors.append('red')
	return Y_colors

def depictA(rows, cols, n):
	G = nx.erdos_renyi_graph(n, 0.1)
	edges = np.transpose([rows,cols])
	G.add_node(31)
	G.add_edge(31,31)
	for edge in edges:
		G.add_nodes_from(edge)
		G.add_edge(edge[0], edge[1])

	nx.draw(G, pos=nx.spring_layout(G), node_color=Y_colors)
	plt.show()

def D_hat(A_hat, n):
	D = torch.zeros_like(A_hat)
	for i in range(n):
		D[i,i] = sum(A_hat[i])
	return D

def train(epoch):
	model.train()
	optimizer.zero_grad()
	output = model(X)
	loss_train = F.nll_loss(output[idx_train], Y[idx_train])
	acc_train = accuracy(output[idx_train], Y[idx_train])
	loss_train.backward()
	optimizer.step()
	if True:
		model.eval()
		output = model(X)
	loss_val = F.nll_loss(output[idx_val], Y[idx_val])
	acc_val = accuracy(output[idx_val], Y[idx_val])
	print ("train acc= {:.4f}".format(acc_train.item()),
		   "train loss= {:.4f}".format(loss_train.item()),
		   "val acc= {:.4f}".format(acc_val.item()),
		   "val loss= {:.4f}".format(loss_val.item())
		)

def test():
	model.eval()
	output = model(X)
	loss_test = F.nll_loss(output[idx_test], Y[idx_test])
	acc_test = accuracy(output[idx_test], Y[idx_test])
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    

# magic numbers
np.random.seed(123)
torch.manual_seed(1)
n = 50
nclasses = 2
nhid = 2
dropout = 0.5
lr = 0.01
weight_decay = 0.0005
idx_train = torch.LongTensor(range(5))
idx_val = torch.LongTensor(range(5,35))
idx_test = torch.LongTensor(range(25,50))

# initialisation 
Y = torch.randint(0, nclasses, size=(n,)) #between [0,nclasses-1] 
Y_colors = set_colors(Y)
X = torch.eye(n)
rows = np.random.randint(n, size=2*n)
cols = np.random.randint(n, size=2*n)
A = torch.zeros((n,n))
A[rows,cols] = 1
A[2,2] = 0
A[2,9] = 1
A[31,31] = 1

# Depict A
depictA(rows, cols, n)
exit()
# normalise A
A_hat = A + torch.eye(n)
D_hat = D_hat(A_hat, n)
D_2 = torch.pow(D_hat, -0.5)
D_2[D_2 == float("Inf")] = 0

aggregation = torch.mm(torch.mm(D_2, A_hat), D_2)	#this doesn't
aggregation = torch.mm(torch.inverse(D_hat), A_hat)	#this normalises everything to sum to one

# train model
adj = torch.FloatTensor(aggregation).to_sparse()
model = GCN(dims=[X.shape[1], 4, nclasses],
			dropout=dropout,
			adj=adj,
			nrm_mthd="softmax",
			learnable=False,
			projection=False,
			rand=False
	)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model(X)
h1 = torch.transpose(model.embeddings_dict['h2'],0,1).detach()
#plt.scatter(h1[0], h1[1], marker='o', color=Y_colors)
#plt.show()

for epoch in range(200):
	train(epoch)

test()

h1 = torch.transpose(model.embeddings_dict['h2'],0,1).detach()
plt.scatter(h1[0], h1[1], marker='o', color=Y_colors)
plt.show()


