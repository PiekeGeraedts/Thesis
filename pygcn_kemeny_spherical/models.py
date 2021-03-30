import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
#from pygcn_kemeny.layers import GraphConvolution
from layers import GraphConvolution
from tools import AdjToSph, SphToAdj 

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # How do we want to initialise the edge weights?
        #self.edge_weights = Parameter(torch.FloatTensor(AdjToSph(adj._indices(), torch.randn(adj._nnz()), adj.size())))
        #-- random
        self.edge_weights = Parameter(torch.FloatTensor(AdjToSph(adj._indices(), adj._values(), adj.size())))
        #-- as Tkipf (1/degree)
        #self.edge_weights = Parameter(torch.zeros(adj._nnz()))
        #self.init_edgeweights(adj._indices())
        #-- self absorbing
        self.dropout = dropout

    def init_adj(self, adj):
        #initialise the adjacency weights: Convert to spherical
        pass
        
    def forward(self, x, indices, values, size):
        adj_values = SphToAdj(indices, self.edge_weights, size)        
        x = F.relu(self.gc1(x, indices, adj_values, size))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, indices, adj_values, size)
        return F.log_softmax(x, dim=1)
