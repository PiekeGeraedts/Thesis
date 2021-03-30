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
from utils import load_data, accuracy, normalisation1, constraint1
from tools import AdjToSph, SphToAdj, toCartesian, KemenySpherical, plot_grad_flow
#from pygcn_kemeny.models import GCN
from models import GCN
from scipy.spatial.distance import euclidean


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
parser.add_argument('--gamma', type=float, default=10**-6,  
                    help='Penalty coefficient for Kemeny constant')
parser.add_argument('--mu', type=float, default=0.1,  
                    help='Neighbourhood of initial adjacency')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj = torch.FloatTensor(np.load('A.npy')).to_sparse()
labels = torch.LongTensor(np.load('labels.npy'))     
features = torch.FloatTensor(np.load('features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))
#adj, features, labels, idx_train, idx_val, idx_test = load_data()

#parameters
adj_indices = adj._indices()
adj_size = adj.size()
adj_nnz = adj._nnz()
sph_nnz = adj_nnz - adj_size[0]
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            adj=adj)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
gamma = args.gamma
mu = args.mu
eta_info = "Eta is chosen decreasing with 1/(n+1)"
model_info = f"the model contains {2} layers"
print ('!!!!!!!!!!!!CHECK!!!!!!!!!!!!')
print ('save (location) of plots and log information')
print ('Gamma:', gamma)
print ('Mu:', mu)
print (eta_info)
print (model_info)
print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
time.sleep(3)

# Check the adjacency for random set of nodes, compare this with after optimisation results.
import random
initial_edge_weights = model.edge_weights.detach().clone()
random.seed(args.seed)
node_lst = [random.randint(0, len(idx_train)-1), random.randint(0, len(idx_train)-1), random.randint(0, len(idx_train)-1)]
neighboor_lst = [np.where(adj_indices[0].cpu().numpy() == node)[0] for node in node_lst]

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
    eta = 1/(epoch+1)
    model.train()  
    optimizer.zero_grad()
    #with torch.no_grad():
    #    model.edge_weights = constraint1(model.edge_weights, initial_edge_weights, mu)
    output = model(features, adj_indices, adj._values(), adj_size)
    gcn_loss = F.nll_loss(output[idx_train], labels[idx_train])
    
    with torch.no_grad():
        perturbation = torch.FloatTensor(np.random.choice([-1,1], sph_nnz)) * eta
        edge_weights0, edge_weights1 = model.edge_weights + perturbation, model.edge_weights - perturbation

        kemeny_loss0 = KemenySpherical(adj_indices, edge_weights0, adj_size)
        kemeny_loss1 = KemenySpherical(adj_indices, edge_weights1, adj_size)

        K_lst.append(KemenySpherical(adj_indices, model.edge_weights, adj_size))
        K0_lst.append(kemeny_loss0)
        K1_lst.append(kemeny_loss1)
    
    loss_train = gcn_loss - gamma*sum(model.edge_weights * ((kemeny_loss0 - kemeny_loss1)/(2*perturbation)))
    acc_train = accuracy(output[idx_train], labels[idx_train])  
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        model.eval()
        output = model(features, adj_indices, adj._values(), adj_size)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #Keep track of train and val. accuracy
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    plot_grad_flow(model.named_parameters())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
def test(): 
    model.eval()
    output = model(features, adj_indices, adj._values(), adj_size)
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
K_lst = []
K0_lst = []
K1_lst = []
pert_lst = []
tmp1_lst = []
tmp2_lst = []
for epoch in range(args.epochs):
    train(epoch)
plt.show()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
loss_test, acc_test = test()

#Plotting and Logging
from datetime import date
t = time.time()
path = 'Results/spsa/smallgraph/'

fig, ax = plt.subplots()
ax.plot(acctrn_lst)
ax.set_title('Accuracy Train Set')
fig.savefig(path + '{:.0f}_accuracytrain.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(accval_lst)
ax.set_title('Accuracy Validation Set')
fig.savefig(path + '{:.0f}_accuracyvalidation.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_title('Progress Kemeny')
fig.savefig(path + '{:.0f}_kemeny.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(K0_lst)
ax.set_title('Progress Kemeny 0')
fig.savefig(path + '{:.0f}_kemeny0.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(K1_lst)
ax.set_title('Progress Kemeny 1')
fig.savefig(path + '{:.0f}_kemeny1.jpg'.format(t_total))
#plt.show()


log = np.array([str(date.today()), str(gamma), eta_info, str(loss_test), str(acc_test)])
#reasoning log list name: t is added so that the corresponding graphs can be found again.
#np.save('Graphs/{:.3f}_log.npy'.format(t), log)
with open(path + "{:.0f}_log.txt".format(t), "a") as file1:
    for line in log:
        file1.write(line)
        file1.write('\n')

print ("============SUBGRAPH INFORMATION============")
lst = list(map(list, zip(*[neighboor_lst, node_lst])))
for neighboor, node in lst:
    print ('Considering node:', node)
    print ('Equal distribution is:', 1/len(neighboor))
    print ('original spherical adjacency:', initial_edge_weights[neighboor])
    print ('original adjacency:', toCartesian(initial_edge_weights[neighboor]))
    print ('spherical adjacency:', model.edge_weights[neighboor])
    print ('adjacency:', np.round(toCartesian(model.edge_weights[neighboor]).detach(),3))
    print (labels[node], model(features, adj_indices, adj._values(), adj_size)[node])
    print ('================================')



    