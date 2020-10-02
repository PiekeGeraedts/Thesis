import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from pygcn_kemeny.layers import GraphConvolution
#from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nnz):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.weighted_adj = Parameter(torch.FloatTensor(nnz))
        self.dropout = dropout

    def init_adj():
        #initialise the adjacency weights
        pass
 
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj._indices(), self.weighted_adj, adj.size()))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj._indices(), self.weighted_adj, adj.size())
        return F.log_softmax(x, dim=1)
