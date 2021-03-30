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
from utils import load_data, accuracy, Kemeny, Kemeny_spsa, subtract_normalisation, squared_normalisation, softmax_normalisation, paper_normalisation
#from pygcn_kemeny.models import GCN
from tools import AdjToSph, SphToAdj, KemenySpherical, plot_grad_flow, plot_grad_flow1
from models import GCN
from Markov_chain.Markov_chain_new import MarkovChain
from MC_derivative import derivativeK, toTheta

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
parser.add_argument('--lrV', type=float, default=0.01,
                    help='Initial learning rate edge weights.')
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
                    help='gradient clip on or off.')
parser.add_argument('--N', type=int, default=0,
                    help='number of epochs with Kemeny regularizer')
parser.add_argument('--nlayer', type=int, default=2,
                    help='number of identical layers')
parser.add_argument('--subtract', action='store_true', default=False,
                    help='subtract normalisation')
parser.add_argument('--squared', action='store_true', default=False,
                    help='squared normalisation')
parser.add_argument('--softmax', action='store_true', default=False,
                    help='softmax normalisation')
parser.add_argument('--spherical', action='store_true', default=False,
                    help='spherical parameterisation')
parser.add_argument('--learnable', action='store_false', default=True,
                    help='the adjacency is not learnable')
parser.add_argument('--projection', action='store_true', default=False,
                    help='use projection method')
parser.add_argument('--rand', action='store_true', default=False,
                    help='random initialisation for V')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

###input###
adj = torch.FloatTensor(np.load('data/A.npy')).to_sparse()
labels = torch.LongTensor(np.load('data/labels.npy'))
features = torch.FloatTensor(np.load('data/features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))
adj, features, labels, idx_train, idx_val, idx_test = load_data()
dims = [args.hidden]*(args.nlayer-1)
dims.insert(0, features.shape[1])
dims.append(labels.max().item() + 1)

'''
# Zacharys Karate Club
P = MarkovChain('ZacharysKarateClub').P
# make P an equally weighted matrix over neighbours
P = np.nan_to_num(P/P)
P = np.transpose(P/sum(P))
adj = torch.FloatTensor(P).to_sparse()
labels = labels[:adj.size()[0]]
features = features[:adj.size()[0]]
idx_train = torch.LongTensor(range(15))
idx_val = torch.LongTensor(range(15,20))
idx_test = torch.LongTensor(range(20,int(adj.size()[0])))
'''

###lists to track variables###
K_lst = []
grad_lst = []
edgeweightsgrad_lst = []
t_lst = []
accval_lst = []
acctrn_lst = []
losstrn_lst = []
n_eps_lst = []

###Variables###
M = 10**5   #max number for kemeny constant, makes the plots better interpretable
N = args.N
eps = args.eps  #for stability we do not let any edge weight become < eps.
gamma = args.gamma
eta = args.eta
adj_indices = adj._indices()
adj_values = adj._values()
adj_size = adj.size()
nnz = adj._nnz()
V_init = adj._values().clone().detach()

if args.softmax:
    normalise = softmax_normalisation
    name = "softmax"
elif args.subtract:
    normalise = subtract_normalisation
    name = "subtract"
elif args.squared: 
    normalise = squared_normalisation
    name = "squared"
elif args.spherical:
    normalise = SphToAdj
    name = "spherical"
    Kemeny = KemenySpherical
    nnz = nnz - adj_size[0]
else:
    assert False, 'Have to specify a normalisation method'
    
# Model and optimizer
print ('!!!!!CHECK WHAT YOU ARE RUNNING!!!!!')
model = GCN(dims=dims,
            dropout=args.dropout,
            adj=adj,
            nrm_mthd=name,
            learnable=args.learnable,
            projection=args.projection,
            rand=args.rand)

#THIS ONLY WORKS FOR A 6 LAYER GCN!!!
lr_edgeweights = args.lrV

optimizer = optim.Adam([{'params': model.gcn['gc1'].parameters()},
                        {'params': model.gcn['gc2'].parameters()},
                        {'params': model.gcn['gc3'].parameters()},
                        {'params': model.gcn['gc4'].parameters()},
                        {'params': model.gcn['gc5'].parameters()},
                        {'params': model.gcn['gc6'].parameters()},
                        {'params': model.edge_weights, 'lr': lr_edgeweights, 'weight_decay': args.weight_decay}],
                        lr=args.lr, weight_decay=args.weight_decay)

lr_info = str(args.lr) + '+' + str(lr_edgeweights)

#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

################################################################################################################################################
#determine one step neighbourhood of the training set
step1_nghd = set()
for i in idx_train:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step1_nghd.add(int(node))
print ('one step size:', len(step1_nghd))

#two+ step neighbourhood of the training set
complement_nghd = []
for i in range(adj_size[0]):
    if (i not in step1_nghd):
        complement_nghd.append(i)
print ('1 step nghd size=', len(step1_nghd))
print ('2+ step nghd size=', len(complement_nghd))

nz_edges = 0
nnz_edges = 0
zerograd_edges = []
nzgrad_edges = []
for i in complement_nghd:
    for j in adj_indices[1][torch.where(adj_indices[0] == i)].tolist():
        if j in complement_nghd:
            zerograd_edges.append([i,j])
            nz_edges += 1
for i in step1_nghd:
    for j in adj_indices[1][torch.where(adj_indices[0] == i)].tolist():
        nzgrad_edges.append([i,j])
        nnz_edges += 1
print ('number of edges in 1 step nghd=', nnz_edges)
print ('number of edges in 2+ step nghd=', nz_edges)
################################################################################################################################################

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def update_print(epoch, acc_val, acc_train, loss_val, loss_train, t):
    #keep track of values
    V = normalise(adj_indices, model.edge_weights.clone().detach(), adj_size)
    n_eps = torch.where(V <= 10*eps)[0].shape[0]
    n_eps_lst.append(n_eps)
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
 #   grad_lst.append(torch.norm(model.edge_weights.grad))
    edgeweightsgrad_lst.append(torch.norm(model.edge_weights.grad))
    losstrn_lst.append(loss_train.item())
    t_lst.append(time.time()-t)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'sum_edgeW:{:.4f}'.format(torch.sum(model.edge_weights).item()),
          'n_eps:{:d}'.format(n_eps),
          'time: {:.4f}s'.format(time.time() - t))

def train(epoch):
    t = time.time()
    model.train()  
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    #clip gradients
    #if args.clipper:
        #still need to check if (0,1) is a good range to clip to. Based on the gcn gradients it is not too low. Their average gradient is mostly smaller than 0.2.
    #    model.edge_weights.grad.clamp(0,1)
    #--not necessary here!
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    update_print(epoch, acc_val, acc_train, loss_val, loss_train, t)
    plot_grad_flow(model.named_parameters())
    plot_grad_flow1(model.named_parameters())
    if epoch == args.epochs-1:
        plot_grad_flow(model.named_parameters(), True, path)
        plot_grad_flow1(model.named_parameters(), True, path)


def train_kemeny(epoch):
    t = time.time()
    model.train()  
    optimizer.zero_grad()
    output = model(features)
    gcn_loss = F.nll_loss(output[idx_train], labels[idx_train])
    kem_loss =torch.zeros_like(model.edge_weights)
    with torch.no_grad():
        perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz)) * eta
        edge_weights0, edge_weights1 = torch.add(model.edge_weights, perturbation), torch.sub(model.edge_weights, perturbation) 
        normalized, normalized0, normalized1 = normalise(adj_indices, model.edge_weights, adj_size, eps), normalise(adj_indices, edge_weights0, adj_size, eps), normalise(adj_indices, edge_weights1, adj_size, eps)
        K, K0, K1 = Kemeny(adj_indices, normalized, adj_size), Kemeny(adj_indices, normalized0, adj_size), Kemeny(adj_indices, normalized1, adj_size)
    kem_loss = torch.mul(model.edge_weights, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2))
    loss_train =  gcn_loss - gamma * torch.sum(kem_loss) 
    
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    #clip gradients
    if args.clipper:
        #still need to check if (0,1) is a good range to clip to
        model.edge_weights.grad = model.edge_weights.grad.clamp(0,0.5)
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    # make kemeny list more interpretable
    cnt = 1
    if (epoch>args.epochs):
        if (abs(K/K_lst[cnt-1]) < 20):
            K_lst.append(K)
            cnt+=1
        else:
            print ('to big')
    else:
        K_lst.append(K)
    update_print(epoch, acc_val, acc_train, loss_val, loss_train, t)
    plot_grad_flow(model.named_parameters())
    plot_grad_flow1(model.named_parameters())
    if epoch == args.N+args.epochs-1:
        plot_grad_flow(model.named_parameters(), True, path + 'K_')
        plot_grad_flow1(model.named_parameters(), True, path + 'K_')

def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test


# Train model
t_total = time.time()
#path = f'Results/2021/FakeGraph/{int(t_total)}_{name}_{args.projection}'
path = f'Results/2021/Cora/performanceGCNKem/{args.projection}/seed={args.seed}/{name}_'
print (path)
for epoch in range(args.epochs):
    train(epoch)

GCN_loss_test, GCN_acc_test = test()
#N = 100
for epoch in range(N):
    train_kemeny(epoch+args.epochs)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
loss_test, acc_test = test()

###Plotting and Logging###
fig, ax = plt.subplots()
ax.plot(acctrn_lst, label='test accuracy')

#fig.savefig(path + 'accuracytrain.jpg')

#fig, ax = plt.subplots()
ax.plot(accval_lst, label='validation accuracy')
ax.set_title('Accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('iteration')
ax.legend()
#ax.set_title('Accuracy Validation Set')
#ax.set_ylabel('accuracy')
#ax.set_xlabel('iteration')
fig.savefig(path + 'accuracyvalidation.jpg')
'''
fig, ax = plt.subplots()
ax.plot(edgeweightsgrad_lst)
ax.set_title('Progress norm edge weights gradient')
ax.set_ylabel('norm of gradient')
ax.set_xlabel('iteration')
fig.savefig(path + 'edgeweightsgrad.jpg')
'''
fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_title('Progress Kemeny')
ax.set_ylabel('kemeny\'s constant')
ax.set_xlabel('iteration')
fig.savefig(path + 'kemeny.jpg')

'''
fig, ax = plt.subplots()
ax.plot(losstrn_lst)
ax.set_title('Training Loss')
fig.savefig(path + 'loss.jpg')
'''

fig, ax = plt.subplots()
ax.plot(n_eps_lst)
ax.set_title('Number of edge weights near zero')
ax.set_ylabel(r'#$V_{ij} > \epsilon$')
ax.set_xlabel('iteration')
fig.savefig(path + 'neps.jpg')
#plt.show()
# log information
eta_info = f"Eta is chosen fixed at {eta}"
avg_grad = sum(edgeweightsgrad_lst)/len(edgeweightsgrad_lst)
log = np.array([str(date.today()), "nlayers:"+str(args.nlayer), "epochs:"+str(args.epochs)+"+"+str(args.N), "clipper:"+str(args.clipper), "random:"+str(args.rand), "lr:"+lr_info, "gamma:"+str(gamma), "eps:"+str(eps), eta_info, "projection:" + str(args.projection), "normalisation method:" + str(name), "average grad norm edge weights:" + str(avg_grad), "GCN loss test:" + str(GCN_loss_test), "GCN acc_test:" + str(GCN_acc_test), "loss test:" + str(loss_test), "acc_test:" + str(acc_test)])
#reasoning log list name: t is added so that the corresponding graphs can be found again.
#np.save('Graphs/{:.3f}_log.npy'.format(t), log)
with open(path +  "log.txt","a") as file1:
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

torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch':epoch}, path+'model-optimised.pt')
torch.save({'projection': args.projection, 'learnable': args.learnable, 'nrm_mthd': name, 'indices': adj._indices(), 'values': adj._values(), 'size': adj.size(), 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test, 'dims': dims, 'embeddings': model.embeddings_dict }, path +'data.pt')

###Apply KDA on the result###
#maybe later

