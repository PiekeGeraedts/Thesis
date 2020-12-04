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
from utils import load_data, accuracy, Kemeny, Kemeny_spsa, subtract_normalisation, squared_normalisation, softmax_normalisation
#from pygcn_kemeny.models import GCN
from models import GCN
from Markov_chain.Markov_chain_new import MarkovChain

# Training settings
parser = argparse.ArgumentParser()
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
parser.add_argument('--eta', type=float, default=1e-4,
                    help='coefficient for spsa')
parser.add_argument('--eps', type=float, default=1e-3,
                    help='Minimum value for edge weights.')
parser.add_argument('--clipper', action='store_true', default=False,
                    help='gradient clip on or off')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

###input###
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj = torch.FloatTensor(np.load('A.npy')).to_sparse()
labels = torch.LongTensor(np.load('labels.npy'))     
features = torch.FloatTensor(np.load('features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))

###lists to track variables###
K_lst = []
Kdiff_lst = []
grad_lst = []
edgeweights_lst = []
t_lst = []
accval_lst = []
acctrn_lst = []
losstrn_lst = []

###Variables###
M = 10**5   #max number for kemeny constant, makes the plots better interpretable
eps = args.eps  #for stability we do not let any edge weight become < eps.
gamma = args.gamma
eta = args.eta
normalise = subtract_normalisation
adj_indices = adj._indices()
adj_values = adj._values()
adj_size = adj.size()
nnz = adj._nnz()
V_init = adj._values().clone().detach()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nnz=nnz,
            adj=adj)
optimizer = optim.Adam([{'params': model.gc1.parameters()},
                        {'params': model.gc2.parameters()},
                        {'params': model.edge_weights, 'lr': args.lr, 'weight_decay': 0}],
                        lr=args.lr, weight_decay=args.weight_decay)

# Log information
eta_info = f"Eta is chosen fixed at {eta}"
model_info = f"the model contains {2} layers"
print ('!!!!!!!!!!!!CHECK!!!!!!!!!!!!')
print ('save (location) of plots and log information')
print ('Gamma:', gamma)
print ('Eta:', eta)
print ('Epsilon:', eps)
print ('normalisation method:', subtract_normalisation)
print (eta_info)
print (model_info)
print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
time.sleep(3)

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
    with torch.no_grad():
        model.edge_weights = normalise(adj_indices, model.edge_weights, adj_size, eps)
    output = model(features, adj_indices, model.edge_weights, adj_size)
    gcn_loss = F.nll_loss(output[idx_train], labels[idx_train])

    with torch.no_grad():
        perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz)) * eta

        edge_weights0, edge_weights1 = torch.add(model.edge_weights, perturbation), torch.sub(model.edge_weights, perturbation) 
        normalized0, normalized1 = normalise(adj_indices, edge_weights0, adj_size, eps), normalise(adj_indices, edge_weights1, adj_size, eps)

        K0, K1 = Kemeny(adj_indices, normalized0, adj_size), Kemeny(adj_indices, normalized1, adj_size)
        K = Kemeny(adj_indices, model.edge_weights, adj_size)

    loss_train =  gcn_loss - gamma * torch.sum(torch.mul(model.edge_weights, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2))) 
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    #clip gradients
    if args.clipper:
        #still need to check if (0,1) is a good range to clip to
        model.edge_weights.clamp(0,1)

    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj_indices, model.edge_weights, adj_size)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    #keep track of values
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    cnt = 1
    if (epoch>0):
        if (abs(K/K_lst[cnt-1]) < 10):
            K_lst.append(K)
            cnt+=1
        else:
            print ('to big')
    else:
        K_lst.append(K)
    #Kdiff_lst.append(K0.clamp(-M,M) - K1.clamp(-M,M))
    grad_lst.append(torch.norm(model.edge_weights.grad))
    edgeweights_lst.append(torch.norm(model.edge_weights))
    losstrn_lst.append(loss_train.item())
    t_lst.append(time.time()-t)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test(): #similar to train, but in eval mode and on test data.
    model.eval()
    output = model(features, adj_indices, model.edge_weights, adj_size)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
loss_test, acc_test = test()

###Plotting and Logging###
from datetime import date
t = time.time()
path = f'Results/lr={args.lr}_eps={args.eps}_'
print (path)

fig, ax = plt.subplots()
ax.plot(acctrn_lst)
ax.set_title('Accuracy Train Set')
fig.savefig(path + '{:.0f}_accuracytrain.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(accval_lst)
ax.set_title('Accuracy Validation Set')
fig.savefig(path + '{:.0f}_accuracyvalidation.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(edgeweights_lst)
ax.set_title('Progress norm edge weights')
fig.savefig(path + '{:.0f}_edgeweights.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_title('Progress Kemeny')
fig.savefig(path + '{:.0f}_kemeny.jpg'.format(t_total))

fig, ax = plt.subplots()
ax.plot(losstrn_lst)
ax.set_title('Training Loss')
fig.savefig(path + '{:.0f}_loss.jpg'.format(t_total))

#fig, ax = plt.subplots()
#ax.plot(K0_lst)
#ax.set_title('Progress Kemeny 0')
#fig.savefig(path + '{:.0f}_kemeny0.jpg'.format(t_total))

#fig, ax = plt.subplots()
#ax.plot(K1_lst)
#ax.set_title('Progress Kemeny 1')
#fig.savefig(path + '{:.0f}_kemeny1.jpg'.format(t_total))
#plt.show()

log = np.array([str(date.today()), "clipper:"+str(args.clipper), "gamma:"+str(gamma), eta_info, "normalisation method:" + str(normalise), "loss test:" + str(loss_test), "acc_test:" + str(acc_test)])
#reasoning log list name: t is added so that the corresponding graphs can be found again.
#np.save('Graphs/{:.3f}_log.npy'.format(t), log)
with open(path +  "{:.0f}_log.txt".format(t_total),"a") as file1:
    for line in log:
        file1.write(line)
        file1.write('\n')

###Check few rows###
torch.set_printoptions(sci_mode=False)
rand = False
if rand:
    print ('============initial weights============')
    #values row 0
    print ('Row 0:\n', V_init[torch.where(adj_indices[0]==0)[0]])
    #values row 25
    print ('Row 25:\n', V_init[torch.where(adj_indices[0]==25)[0]])
    #values row 50
    print ('Row 50:\n', V_init[torch.where(adj_indices[0]==50)[0]])
    #values row 75
    print ('Row 75:\n', V_init[torch.where(adj_indices[0]==75)[0]])
    V_nrmld = normalise(adj_indices, model.edge_weights.detach(), adj_size, eps)
    print ('\n============initial distribution============')
    #values row 0
    print ('Row 0:\n', V_nrmld[torch.where(adj_indices[0]==0)[0]])
    #values row 25
    print ('Row 25:\n', V_nrmld[torch.where(adj_indices[0]==25)[0]])
    #values row 50
    print ('Row 50:\n', V_nrmld[torch.where(adj_indices[0]==50)[0]])
    #values row 75
    print ('Row 75:\n', V_nrmld[torch.where(adj_indices[0]==75)[0]])
    print ('\n============Final distribution============')
else:
    print ("\n\nRecall, V starts of with equal weights for all neighbours!")
V = normalise(adj_indices, model.edge_weights.detach(), adj_size, eps)
#values row 0
print ('Row 0:\n', V[torch.where(adj_indices[0]==0)[0]])
#values row 25
print ('Row 25:\n', V[torch.where(adj_indices[0]==25)[0]])
#values row 50
print ('Row 50:\n', V[torch.where(adj_indices[0]==50)[0]])
#values row 75
print ('Row 75:\n', V[torch.where(adj_indices[0]==75)[0]])
print ('smallest value in V:', torch.min(V))
print ('Average time per epoch:', sum(t_lst)/len(t_lst))    

###Apply KDA on the result###
#maybe later

