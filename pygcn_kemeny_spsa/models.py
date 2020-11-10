import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from pygcn_kemeny.layers import GraphConvolution
#from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nnz, adj):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # How do we want to initialise the edge weights?
        #self.edge_weights = Parameter(torch.randn(nnz))
        #-- random
        self.edge_weights = Parameter(adj._values())
        #-- as Tkipf (1/degree)
        #self.edge_weights = Parameter(torch.zeros(nnz))
        #self.init_edgeweights(adj._indices())
        #-- self absorbing
        self.dropout = dropout
 
    def forward(self, x, indices, valuess, size):
        x = F.relu(self.gc1(x, indices, self.edge_weights, size))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, indices, self.edge_weights, size)
        return F.log_softmax(x, dim=1)
