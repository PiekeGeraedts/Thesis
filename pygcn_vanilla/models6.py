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
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nclass)
        self.embedding_dict = {'h0':None, 'h1':None, 'h2':None, 'h3':None, 'h4':None, 'h5':None, 'h6':None}
        self.dropout = dropout

    def forward(self, x, adj):
        self.embedding_dict['h0'] = x
        h1 = F.relu(self.gc1(x, adj))
        self.embedding_dict['h1'] = h1
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = F.relu(self.gc2(h1, adj))
        #print (torch.norm(h2-self.embedding_dict['h1']))
        self.embedding_dict['h2'] = h2
        h3 = F.relu(self.gc3(h2, adj))
        #print (torch.norm(h3-self.embedding_dict['h2']))
        self.embedding_dict['h3'] = h3
        h4 = F.relu(self.gc4(h3, adj))
        #print (torch.norm(h4-self.embedding_dict['h3']))
        self.embedding_dict['h4'] = h4
        h5 = F.relu(self.gc5(h4, adj))
        #print (torch.norm(h5-self.embedding_dict['h4']))
        self.embedding_dict['h5'] = h5
        h5 = F.dropout(h5, self.dropout, training=self.training)
        h6 = self.gc6(h5, adj)
        self.embedding_dict['h6'] = h6

        return F.log_softmax(h6, dim=1)

