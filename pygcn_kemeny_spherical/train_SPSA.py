from __future__ import division
from __future__ import print_function

import time
#allows to write user-friendly command line interface
import argparse
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from pygcn_kemeny.utils import load_data, accuracy, Kemeny, Kemeny_spsa, WeightClipper
from utils import load_data, accuracy, Kemeny, Kemeny_spsa, WeightClipper, toP
#from pygcn_kemeny.models import GCN
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
#Example: Can use 'python3 train.py -epochs 500' to let the network train with 500 epochs.
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
#parameters
edge_weights = nn.Parameter(torch.randn(adj._nnz()))
adj_indices = adj._indices()
adj_size = adj.size()
nnz = adj._nnz()

print (adj)
A = adj.to_dense()

print (A[0,0], A[0,8], A[0,14], A[0,258], A[0,435], A[0,544])
print (A[0,0] + A[0,8] + A[0,14] + A[0,258] + A[0,435] + A[0,544])
print (A[0,0], A[8,0], A[14,0], A[258,0], A[435,0], A[544,0])
print (A[0,0] + A[8,0] + A[14,0] + A[258,0] + A[435,0] + A[544,0])
exit()
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nnz=nnz)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
Clipper = WeightClipper()
gamma = 10**-5
eta = 10**-6    #should be decreasing with epochs
import random
node_lst = [random.randint(0, len(adj)), random.randint(0, len(adj)), random.randint(0, len(adj))]

print ('!!!!!!!!!!!!CHECK!!!!!!!!!!!!')
print ('save (location) of plots')
print ('Gamma:', gamma)
print ('Eta:', eta)
print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

print ('================================================================================')
A = adj.to_dense()
for node in node_lst:
    print ('Considering node:', node)
    print (np.where(A[node] != 0)[0].shape[0])

t = time.time()
toP(A[:,:len(A)-1].numpy())
print (time.time() - t)

print ('================================================================================')
exit()
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def test_check(tensor1):
    return tensor1

def train(epoch):
    global edge_weights
    t = time.time()
    model.train()  
    optimizer.zero_grad()
    edge_weights = F.softmax(edge_weights, dim=0)
    output = model(features, adj_indices, edge_weights, adj_size)
    gcn_loss = F.nll_loss(output[idx_train], labels[idx_train])

    with torch.no_grad():
        perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz)) * eta
        edge_weights0, edge_weights1 = edge_weights + perturbation, edge_weights - perturbation
        
        normalized0, normalized1 = F.softmax(edge_weights0, dim=0), F.softmax(edge_weights1, dim=0)

        kemeny_loss0 = - gamma * Kemeny(adj_indices, normalized0, adj_size)
        kemeny_loss1 = - gamma * Kemeny(adj_indices, normalized1, adj_size)
    
    loss_train = gcn_loss + sum(edge_weights * ((kemeny_loss0 - kemeny_loss1)/2*perturbation))
    acc_train = accuracy(output[idx_train], labels[idx_train])  
    loss_train.backward()
    #plot gradient flow and check analytical gradient with FD gradient
    plot_grad_flow(model.named_parameters())
    '''
    print ('checking gradient...')
    t = time.time()
    print (torch.autograd.gradcheck(Kemeny, (adj_indices, edge_weights.double(), adj_size)))
    print (time.time()-t)
    exit()
    '''
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_indices, edge_weights, adj_size)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #Keep track of train and val. accuracy
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    K0_lst.append(kemeny_loss0)
    K1_lst.append(kemeny_loss1)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(): #similar to train, but in eval mode and on test data.
    model.eval()
    output = model(features, adj_indices, edge_weights, adj_size)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test


# Train model
t_total = time.time()
acctrn_lst = []
accval_lst = []
K0_lst = []
K1_lst = []
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
loss_test, acc_test = test()

#Plotting
path = 'Results/spsa/'
plt.plot(acctrn_lst)
plt.title('Accuracy Train Set')
plt.savefig(path + '{:.3f}_accuracytrain.jpg'.format(t_total))
plt.show()

plt.plot(accval_lst)
plt.title('Accuracy Validation Set')
plt.savefig(path + '{:.3f}_accuracyvalidation.jpg'.format(t_total))
plt.show()

plt.plot(K0_lst)
plt.title('Progress Kemeny Loss 0')
plt.savefig(path + '{:.3f}_kemenyloss0.jpg'.format(t_total))
plt.show()

plt.plot(K1_lst)
plt.title('Progress Kemeny Loss 1')
plt.savefig(path + '{:.3f}_kemenyloss1.jpg'.format(t_total))
plt.show()

#save log file: t_total is added so that the corresponding graphs can be found.
eta_info = f"Eta is chosen fixed at: {eta}"
log = np.array([date.today(), gamma, eta_info, float(loss_test), float(acc_test)])
np.save(path + '{:.3f}_log.npy'.format(t_total), log)

for node in node_lst:
    print ('Considering node:', node)
    print (np.where(A[node] != 0)[0].shape[0])