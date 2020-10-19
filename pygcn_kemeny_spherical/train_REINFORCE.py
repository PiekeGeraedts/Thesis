from __future__ import division
from __future__ import print_function

import time
import argparse
#allows to write user-friendly command line interface
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

#from pygcn_kemeny.utils import load_data, accuracy, Kemeny, Kemeny_spsa, WeightClipper
from utils import load_data, accuracy, Kemeny, Kemeny_spsa, WeightClipper
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

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nnz=adj._nnz())
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
gamma = 10**-4
eta = 10**-6    #should be decreasing with epochs

print ('!!!!!!!!!!!!CHECK!!!!!!!!!!!!')
print ('save (location) of plots')
print ('Gamma:', gamma)
print ('Eta:', eta)
print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

import random
node_lst = [random.randint(0, len(adj)), random.randint(0, len(adj)), random.randint(0, len(adj))]
print ('================================================================================')
A = adj.to_dense()
for node in node_lst:
    print ('Considering node:', node)
    print (np.where(A[node] != 0)[0].shape[0])

print ('================================================================================')

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#parameters
ew_mean = nn.Parameter(torch.randn(adj._nnz()))
ew_std = nn.Parameter(torch.ones(adj._nnz()))

adj_indices = adj._indices()
adj_size = adj.size()

def train(epoch):
    global edge_weights
    t = time.time()
    model.train()  
    optimizer.zero_grad()
    # reparametrized sample from the normal dist
    ew_dist = dist.Normal(ew_mean, ew_std)
    edge_weights = ew_dist.rsample()
    # normalisation or spherical normalization or whatever
    edge_weights = F.softmax(edge_weights, dim=0)
    output = model(features, adj_indices, edge_weights, adj_size)
    #gcn loss
    gcn_loss = F.nll_loss(output[idx_train], labels[idx_train])
    #kemeny loss
    with torch.no_grad():
        kemeny_loss = - gamma * Kemeny(adj_indices, edge_weights, adj_size)
    #total loss
    print (kemeny_loss)
    loss_train = gcn_loss + sum(ew_dist.log_prob(edge_weights) * kemeny_loss)
    # -- The expectation over the second term, we rewrite using the score function (so this becomes a score function with a
    #    single sample). The variable `kemeny_loss` is just a constant, but the log probability over the edge_weight will
    #    get a gradient for the REINFORCE loss


    acc_train = accuracy(output[idx_train], labels[idx_train])  
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_indices, edge_weights, adj_size)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) #add kemeny loss to this
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #Keep track of train and val. accuracy
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    Kloss_lst.append(kemeny_loss)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(): #similar to train, but in eval mode and on test data.
    model.eval()
    edge_weights = dist.Normal(ew_mean, ew_std).rsample()
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
Kloss_lst = []
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
loss_test, acc_test = test()

#Plotting
path = 'Results/reinforce/'
plt.plot(acctrn_lst)
plt.title('Accuracy Train Set')
plt.savefig(path + '{:.3f}_accuracytrain.jpg'.format(t_total))
plt.show()

plt.plot(accval_lst)
plt.title('Accuracy Validation Set')
plt.savefig(path + '{:.3f}_accuracyvalidation.jpg'.format(t_total))
plt.show()

plt.plot(Kloss_lst)
plt.title('Progress Kemeny Loss')
plt.savefig(path + '{:.3f}_kemenyloss.jpg'.format(t_total))
plt.show()


#save log file: t_total is added so that the corresponding graphs can be found.
eta_info = f"Eta is chosen fixed at: {eta}"
log = np.array([date.today(), gamma, eta_info, float(loss_test), float(acc_test)])
np.save(path + '{:.3f}_log.npy'.format(t_total), log)

for node in node_lst:
    print ('Considering node:', node)
    print (np.where(A[node] != 0)[0].shape[0])