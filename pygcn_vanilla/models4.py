import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.embedding_dict = {'h0':None, 'h1':None, 'h2':None, 'h3':None, 'h4':None}
        self.dropout = dropout

    def forward(self, x, adj):
        self.embedding_dict['h0'] = x
        x = F.relu(self.gc1(x, adj))
        self.embedding_dict['h1'] = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        #print ('1', torch.norm(x-self.embedding_dict['h1']))
        self.embedding_dict['h2'] = x
        x = F.relu(self.gc3(x, adj))
        #print ('2', torch.norm(x-self.embedding_dict['h2']))
        self.embedding_dict['h3'] = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        self.embedding_dict['h4'] = x

        return F.log_softmax(x, dim=1)

