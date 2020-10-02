import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nnz):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        print (nnz)
        exit()
        self.adj_weighted = Parameter(torch.DoubleTensor(nnz))
        self.dropout = dropout

    def init_adj():
        #initialise the adjacency weights
        pass
 
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj._indices(), adj_weighted, adj.size())
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj._indices(), adj_weighted, adj.size())
        return F.log_softmax(x, dim=1)
