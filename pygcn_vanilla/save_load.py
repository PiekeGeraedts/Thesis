"""
Purpose:	
	Load a trained model for inference. Currently, as check I want that the test accuracies coincide, however this is not the case.
Date:
	04-12-2020
	
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import GCN
from utils import load_data, accuracy
import time
#import train
import numpy as np
np.random.seed(42)
torch.manual_seed(42)

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

#print ('###in save-load.py###')
#test()
#print ('========')
#exit()

adj, features, labels, idx_train, idx_val, idx_test = load_data()
input = torch.load('input.pt')
adj = torch.sparse.FloatTensor(input['indices']input['values'], input['size'])
features, labels, idx_train, idx_val, idx_test = input['features'], input['labels'], input['idx_train'], input['idx_val'], input['idx_test']

kwargs = {"dropout":0.5, "nfeat":features.shape[1], "nclass":labels.max().item()+1, "nhid":16}
model = GCN(**kwargs)
kwargs = {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4}
optimizer = optim.Adam(**kwargs)

checkpoint = torch.load('model-optimised.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']

#testing
model.eval()
test()


