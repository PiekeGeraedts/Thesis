import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from pygcn_kemeny.layers import GraphConvolution
#from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nnz, adj, rand=False):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # How do we want to initialise the edge weights?
        if rand:
            self.edge_weights = Parameter(torch.randn(nnz))
        else:
            self.edge_weights = Parameter(adj._values())
        
        self.dropout = dropout

    def init_adj():
        #initialise the adjacency weights
        pass
 
    def forward(self, x, indices, values, size):
        x = F.relu(self.gc1(x, indices, self.edge_weights, size))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, indices, self.edge_weights, size)
        return F.log_softmax(x, dim=1)



'''
import numpy as np
import matplotlib.pyplot as plt
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def squared(x):
    return x**2/sum(x**2)


N = 10000
K = 50
n = 5

for k in range(K):
    x1 = np.random.random(n)
    x2 = np.random.random(n)
    x1_lst = [x1]
    x2_lst = [x2]

    for i in range(N):
        x1 = softmax(x1)
        x2 = squared(x2)
        x1_lst.append(x1)
        #x2_lst.append(x2)
        x2_lst.append(x2)

   # plt.plot(x1_lst)
    plt.plot(x2_lst)
plt.show()

lst1 = np.zeros((2,N))
lst2 = np.zeros((2,N))
for i in range(N):
    rnd = np.random.random(2) #random in [0,1]
    p = rnd/sum(rnd)
    r = np.sqrt(p)
    lst1[0,i] = p[0]
    lst1[1,i] = p[1]
    lst2[0,i] = r[0]
    lst2[1,i] = r[1]

x_axis = np.linspace(0,1,N)
y_axis = 1 - x_axis
lst1 = np.array([x_axis, y_axis])
x1_axis = np.sqrt(x_axis)
y1_axis = np.sqrt(y_axis)
lst2 = np.array([x1_axis, y1_axis])

plt.scatter(lst1[0],lst1[1], s=2)
plt.scatter(lst2[0],lst2[1], s=2)
plt.show()

x_axis = np.linspace(0,1,100)
y_axis = 1 - x_axis
x1_axis = np.sqrt(x_axis)
y1_axis = np.sqrt(y_axis)
points = np.array([x1_axis, y1_axis]).T
lst2 = lst2.T
density = []
epsilon = 0.1

from scipy.spatial.distance import euclidean

for i in range(100):
    #count number of points in lst2 that have a euclidean distance of less than epsilon with points
    count = 0
    for point in lst2:
        if (euclidean(points[i], point) < epsilon):
            count+=1
    density.append(count)

plt.scatter(points.T[0], points.T[1])
plt.show()
plt.plot(density)
plt.show()

'''


