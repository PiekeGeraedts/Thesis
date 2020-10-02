from __future__ import division
from __future__ import print_function

import time
import argparse
#allows to write user-friendly command line interface
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn_kemeny.utils import load_data, accuracy, Kemeny, Kemeny_spsa, WeightClipper
from pygcn_kemeny.models import GCN

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

print ('!!!!!!!!!!!!!!!!!!!!!!!!')
print ('WARNING: Check save (location) of plots')
print ('!!!!!!!!!!!!!!!!!!!!!!!!')

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nnz=adj._nnz())
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
Clipper = WeightClipper()
gamma = 10**-5
eta = 10**-6    #should be decreasing with epochs

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()  
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    K = Kemeny(adj._indices(), model.weighted_adj.detach(), adj.size())
    loss_train += - gamma*K
    acc_train = accuracy(output[idx_train], labels[idx_train])  
    loss_train.backward()
    K_spsa = Kemeny_spsa(adj._indices(), model.weighted_adj.detach(), adj.size(), eta)
    model.weighted_adj.grad += - gamma*K_spsa
    optimizer.step()
    model.apply(Clipper)
    #Should I still normalize the weighted_adj after step?
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #Keep track of train and val. accuracy
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    K_lst.append(K)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'Kemeny: {:.4f}'.format(K))


def test(): #similar to train, but in eval mode and on test data.
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
acctrn_lst = []
accval_lst = []
K_lst = []
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

#Plotting
import matplotlib.pyplot as plt
plt.plot(acctrn_lst)
plt.title('Accuracy Train Set')
plt.savefig('accuracytrain_{:.3f}.jpg'.format(time.time()))
plt.show()

plt.plot(accval_lst)
plt.title('Accuracy Validation Set')
plt.savefig('accuracyvalidation_{:.3f}.jpg'.format(time.time()))
plt.show()

plt.plot(K_lst)
plt.title('Progress Kemeny Constant')
plt.savefig('kemeny_{:.3f}.jpg'.format(time.time()))
plt.show()
