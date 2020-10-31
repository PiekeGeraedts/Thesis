import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Markov_chain.Markov_chain_new import MarkovChain

class SpecialSpmmFunction(torch.autograd.Function):
    """
        Special function for only sparse region backpropataion layer.

        source: PyGAT, https://github.com/Diego999/pyGAT
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]

        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

#spmm = SpecialSpmm()

class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weighted_adj'):
            w = module.weighted_adj
            w = w.clamp(0, 1)
            module.weighted_adj.data = w


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
 

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float64)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Kemeny(indices, values, size):
    """Calculate the Kemeny constant."""
    P = torch.sparse.DoubleTensor(indices, values, size).detach().to_dense().numpy()
    return torch.DoubleTensor([MarkovChain(P).K])


def Kemeny_spsa(indices, values, size, eta):
    """Calculate gradient of Kemeny constant using SPSA."""
    #NOTE: the gradients are only calculated for the edges in the network.
    nnz = values.shape[0]
    delta = torch.FloatTensor(np.random.choice([-1,1], nnz))

    values1 = normalisation1(indices, torch.add(values, delta, alpha=eta), size)
    values2 = normalisation1(indices, torch.sub(values, delta, alpha=eta), size)

    P1 = torch.sparse.FloatTensor(indices, values1, size) 
    P2 = torch.sparse.FloatTensor(indices, values2, size)

    K1 = MarkovChain(P1.to_dense().detach().numpy()).K
    K2 = MarkovChain(P2.to_dense().detach().numpy()).K
    
    return torch.mul(torch.pow(delta, exponent=-1), (K1-K2)/2*eta)

def softmax_norm(indices, values, size):
    """Normalise the adjacency matrix with softmax. Problem with using softmax on the entire row is that the zeros are transformed to nnz."""
    for i in range(size[0]):
        idx = torch.where(indices[0] == i)[0]
        values[idx] = F.softmax(values[idx], dim=0)
    return values
    
def normalisation1(indices, values, size):
    """This normalisation squares the input then divides by sum of the squares"""
    #NOTE: this normalisation has a pool at zero, e.g., if the parameter is a zero row then this does not scale to probability dist.
    for i in range(size[0]):
        idx = torch.where(indices[0] == i)[0]
        if (torch.sum(torch.mul(values[idx], values[idx])) == 0):
            values[idx] = F.softmax(values[idx], dim=0)
        values[idx] = torch.mul(values[idx], values[idx])/torch.sum(torch.mul(values[idx], values[idx]))
    return values

def normalisation2(indices, values, size):
    """This normalisation is as in arXiv:1309.1541: "a rigid shift of the points to the right of the Y-asis"."""
    for i in range(size[0]):
        idx = torch.where(indices[0] == i)[0]
        tmp = torch.sort(values[idx], descending=True)[0]
        for j in len(tmp):
            if (tmp[j] + 1/j(1-sum(tmp[:j])) > 0):
                rho = j
        lbd = 1/rho*(1-sum(tmp[:rho]))
        values[idx] = values[idx] + lbd
        values[idx] = max(values[idx, torch.zeros_like(values[idx])])

def nan_to_zero(mx):
    """Set all nan values to zero"""
    coo = np.where(torch.isnan(mx) == True)
    #print (coo)
    return len(coo[0])
    #print (len(coo[0]) + len(coo[1])) 
    if (len(coo[0]) + len(coo[1]) == 0):
        #there where no inf numbers, continues as usual
        pass
    else:
        print (f'There are {len(coo[0]) + len(coo[1])} nan numbers, fit it!')
        exit()
        mx[coo[0], coo[1]] = 0
    #return mx

def inf_to_large(mx):
    """Set all inf values to large(st) float value"""
    import sys
    max_float = sys.float_info.max
    #print (torch.sum(torch.isinf(mx).view(-1) == True))
    
    coo = np.where(torch.isinf(mx) == True)
    if (len(coo[0]) + len(coo[1]) == 0):
        #there where no inf numbers, continues as usual
        pass
    else:
        print ('There are inf numbers, fit it!')
        exit()
        mx[coo[0], coo[1]] = max_float
    #return mx

def constraint(weights, initial, mu):
    """Clamps all values of the weighted adjacency to the mu nieghbourhood of initial"""
    #NOTE: takes roughly 0.033 seconds for nnz=896, might be a bigger problem when using this on Cora.
    nnz = weights.shape[0]
    for i in range(nnz):
        weights[i] = weights[i].clamp(initial[i]-mu, initial[i]+mu)
    return weights




    
