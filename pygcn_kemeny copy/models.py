import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
#from pygcn_kemeny.layers import GraphConvolution
from layers import GraphConvolution
from tools import AdjToSph, SphToAdj
from utils import softmax_normalisation, squared_normalisation, subtract_normalisation, SpecialSpmm

class GCN(nn.Module):
    def __init__(self, dims, dropout, adj, nrm_mthd, projection, learnable=True, rand=False):
        #dims should be a list, dropout a float, and adj the sparse adjencency. 
        #dims = [nfeat, nhid1, .., nhid, nclass], i.e., size of nlayer+1, hence, dims also says the amount of layers.
        #learnable=False doesn't work yet.
        super(GCN, self).__init__()
        self.sparsemm = SpecialSpmm()
        self.gcn, self.embeddings_dict = self.init_layers(dims)    
        self.indices = adj._indices()
        self.size = adj.size()  
        self.learnable = learnable
        self.projection= projection
        self.spherical, self.normalise = self.check_substitution(nrm_mthd)
        self.edge_weights = self.init_adj(adj, rand)
        self.dropout = dropout
        
    def check_substitution(self, nrm_mthd):
        spherical = False
        if nrm_mthd == 'spherical':
            spherical = True
            normalise = SphToAdj
        elif nrm_mthd == 'softmax':
            normalise = softmax_normalisation
        elif nrm_mthd == 'squared':
            normalise = squared_normalisation
        elif nrm_mthd == 'subtract':
            normalise = subtract_normalisation
        else:
            assert False, 'Invalid normalisation type'
        return spherical, normalise 

    def init_layers(self, dims):
        nlayers = len(dims) - 1
        gcn = {}
        embeddings_dict = {'h0':None}
        for i in range(nlayers):
            gc_layer = GraphConvolution(dims[i], dims[i+1]) #optional: bias=False 
            gcn['gc'+str(i+1)] = gc_layer
            #This does not feel like the best solution, given that the GC already makes the parameters.
            #they aren't added to parameter bcs gc is not a global variable in this class and,
            #bcs the gcn dict is a dict of classes.
            self.register_parameter(f'gc{i+1}_weight', gc_layer.weight)
            self.register_parameter(f'gc{i+1}_bias', gc_layer.bias)
            embeddings_dict['h'+str(i+1)] = None
        return gcn, embeddings_dict

    def init_adj(self, adj, rand):
        if self.spherical:
            if not self.learnable:
                edge_weights = AdjToSph(self.indices, adj._values(), self.size)
            else:
                if rand:
                    edge_weights = Parameter(torch.randn(adj._nnz()-self.size[0]))
                else:
                    edge_weights = Parameter(AdjToSph(self.indices, adj._values(), self.size))
        else:
            if not self.learnable:
                edge_weights = adj._values()
            else:
                if rand:
                    edge_weights = Parameter(torch.randn(adj._nnz()))
                else:
                    edge_weights = Parameter(adj._values())
        return edge_weights

    def update_embeddings_dict(self, lst):
        #note no activation has been applied to the output of the last layer.
        for i in range(len(lst)):
            self.embeddings_dict['h'+str(i)] = lst[i]

    def D_inv(self):
        D = torch.zeros(self.size)
        V = torch.sparse.FloatTensor(self.indices, self.edge_weights, self.size)
        #D[[range(self.size[0]), range(self.size[0])]] = torch.sparse.mm(V, torch.ones(self.size)).diag()
        D[[range(self.size[0]), range(self.size[0])]] = self.sparsemm(self.indices, self.edge_weights, self.size, torch.ones(self.size)).diag()
        
        return torch.inverse(D) #this is 1/sum_row_weight on the diag and zero elsewhere

    def forward(self, x): 
        #normalise and convert spherical to adjacency if needed
        '''
        if self.learnable:
            if self.projection and not self.spherical:  #cannot use spherical to do projections, if spherical and projection then just do spherical.
                with torch.no_grad():
                    self.edge_weights.data = self.normalise(self.indices, self.edge_weights, self.size)
                values = self.edge_weights  #does it matter if I use .clone() here?
            else:   
                values = self.normalise(self.indices, self.edge_weights, self.size)
        else:
            values = self.edge_weights
        '''
        #R=(W@D^-1)
        values = torch.sparse.FloatTensor(self.indices, self.edge_weights, self.size).to_dense()    #sum of these rows still match D
        #D_inv@W should work!!!
        values = torch.mm(self.D_inv(), values).to_sparse()
        self.edge_weights.data = values._values()
        values = self.edge_weights


        lst = [x]
        for gc in self.gcn:
            gc_layer = self.gcn[gc]
            if gc == 'gc' + str(len(self.gcn)):                    
                x = gc_layer(x, self.indices, values, self.size)
            else:
                x = F.relu(gc_layer(x, self.indices, values, self.size))
                x = F.dropout(x, self.dropout, training=self.training)
            lst.append(x)
        self.update_embeddings_dict(lst)

        return F.log_softmax(x, dim=1)




