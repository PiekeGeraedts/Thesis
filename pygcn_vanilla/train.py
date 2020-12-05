from __future__ import division
from __future__ import print_function

import time
import argparse
#allows to write user-friendly command line interface
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

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
#write adjacency matrix to log file
'''
with open("adjacency.log", "a") as f:
    n,m = adj.shape
    for i in range(m):
        #the adj.item() of each row sum to one.
        f.write(f"{i}-")
        for j in range(n):   
            if (adj[i,j].item() != 0.0):
                f.write(f"{j}:{adj[i,j].item()} ")
        f.write('\n')
'''

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

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
    model.train()   #this sets the module in training mode. Some layers (e.g., dropout and batchnorm) behave different under train and eval (test) mode
    optimizer.zero_grad()
    output = model(features, adj)   #this is the same as model.forward(features, adj), i.e., a forward pass. Hence returns final embedding
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])   #negative log likelihood loss, (input: torch.Tensor, target: torch.Tensor)
                                #NOTE: it is enough that the shape of output is (-1, C: number of classes), no need to specify
    acc_train = accuracy(output[idx_train], labels[idx_train])  #(I think) The max of the elements is taken to find classification prediction and compute accuracy.
    loss_train.backward() #performs backpropogation, make sure gradients are zeroed beforehand.
    optimizer.step()      #performs optimisation step
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    accval_lst.append(acc_val.item())
    acctrn_lst.append(acc_train.item())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

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
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
print (dir(model))
print ('##Printing model.buffers')
for buffer in model.buffers():
    print (buffer)
print ('##Printing model.named_buffers')
for named_buffer in model.named_buffers():
    print (named_buffer)
print ('##Printing model.children')
for child in model.children():
    print (child)

torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch':epoch}, 'model-optimised.pt')

print ('###Parameter norms')
for param in model.parameters():
    print (torch.norm(param[0].detach()))
    print ('===')

#Plotting
plt.plot(acctrn_lst)
plt.title('Accuracy Train Set')
#plt.savefig('accuracytrain_{:.3f}.jpg'.format(time.time()))
#plt.show()

plt.plot(accval_lst)
plt.title('Accuracy Validation Set')
#plt.savefig('accuracyvalidation_{:.3f}.jpg'.format(time.time()))
#plt.show()

#import save_load

