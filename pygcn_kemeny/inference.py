"""
Purpose:
	Try to make sense of the output of Kemeny optimisation. Can we see desirable effects?
Date:	
	05-12-2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from models import GCN
from utils import load_data, accuracy, rowdiscrep, subtract_normalisation, softmax_normalisation, squared_normalisation
from tools import SphToAdj
from collections import Counter 
import time
import matplotlib.pyplot as plt

#import train

def test(): 
    model.eval()
    output = model(features, indices, model.edge_weights, size)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test

def distance_unequal_dim(x,y):
	n1 = x.shape[0]
	n2 = y.shape[0]
	if (n1 > n2):
		tmp = torch.zeros_like(x)
		tmp[:n2] = y
		y = tmp
	elif (n1 < n2):
		tmp = torch.zeros_like(y)
		tmp[:n1] = x
		x = tmp
	return torch.cdist(x.view(1,-1),y.view(1,-1)) #can also use torch.norm

def over_smoothing(dictionary):
	if dictionary['h0'] == None:
		return

	nlayers = len(dictionary)-1
	diff_lst = []
	diff_lst1 = []
	dim_lst = []
	for i in range(nlayers):
		if i == 0 or i == 1 or i == nlayers:
			if i!= 0: dim_lst.append(np.linalg.matrix_rank(dictionary['h'+str(i)].detach()))
			#print (np.linalg.matrix_rank(dictionary['h'+str(i)].detach()))
			continue
		h_current = 'h' + str(i)
		h_previous = 'h' + str(i-1)

		diff = torch.norm(dictionary[h_current] - dictionary[h_previous])
		diff_lst.append(diff)
		#print (f'Difference between output layer {i} and layer {i-1}: {diff}')
		dim_lst.append(np.linalg.matrix_rank(dictionary[h_current].detach()))
	return diff_lst, dim_lst


def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

#TODO: step is not implemented
def neighbourhood(node, step=2):
	"""return the <step>-step neighbourhood of node <node>"""
	nghd = []
	nghd.append(indices[1][torch.where(indices[0] == node)].tolist())
	for ngh in nghd[0]:
		nghd.append(indices[1][torch.where(indices[0] == ngh)].tolist())
	return [ngh for nghs in nghd for ngh in nghs]

def inference():
	V = model.edge_weights.detach()
	n = int(size[0])
	cnt_K = 0
	cnt_mf = 0
	cnt_mf2 = 0
	cnt_first = 0
	for node in range(4,n):
		label = labels[node]
		neighbours = indices[1][torch.where(indices[0] == node)]
		idx = torch.where(indices[0] == node)[0]

		label_ngh = labels[neighbours]
		label_ngh2 = labels[neighbourhood(node)]

		dist_ngh = V[idx]
		maxprob_label = labels[neighbours[torch.argmax(V[idx])]]
		label_mf = most_frequent(label_ngh.tolist())
		label_mf2 = most_frequent(label_ngh2.tolist())

		if maxprob_label == label:
			cnt_K +=1
		if label_mf == label:
			cnt_mf +=1
		if label_mf2 == label:
			cnt_mf2 +=1
		if labels[neighbours][0] == label:
			cnt_first +=1
		
		'''
		print ('###INFORMATION###')
		print ('node=', node)
		print ('label=', label)
		print ('label neighbours=', label_ngh)
		print ('distribution over neighbours=', dist_ngh)
		print ('label max prob.=', maxprob_label)
		print ('label mf=', label_mf)
		print ('label mf2=', label_mf2)
		print ('label first=', labels[neighbours][0])
		time.sleep(1)
		'''
		
	print ('Max. prob. acc.:', cnt_K/n)
	print ('First label acc.:', cnt_first/n)
	print ('Most frequent acc.:', cnt_mf/n)
	print ('Most frequent 2 acc.:', cnt_mf2/n)


# Magic numbers
seed = 42
lr = 0.01
weight_decay = 5e-4
dropout = 0.5
eps = 1e-3

# Input
#path = 'Results/spsa/Cora/gcn200-gcnK50-1/'
#path = ''

# Initialisation
torch.manual_seed(42)	
#path = 'Users/piekegeraedts/Documents/MScEOR-OR/Thesis/Pieke_git/pygcn_vanilla/'
path = 'Results/2021/Cora/performanceGCNKem/substitution/extra/True_0.01_spherical_'


input = torch.load(path + 'data.pt')
indices = input['indices']
values = input['values']
size = input['size']
adj = torch.sparse.FloatTensor(indices, values, size)	#this is the adj using GCN aggregation
features, labels, idx_train, idx_val, idx_test, dims, embeddings_dict, bprojection, blearnable, nrm_mthd = input['features'], input['labels'], input['idx_train'], input['idx_val'], input['idx_test'], input['dims'], input['embeddings'], input['projection'], input['learnable'], input['nrm_mthd']
normalise = softmax_normalisation #manual entry



# Model and optimizer
model = GCN(dims=dims,
            dropout=dropout,
            adj=adj,
            nrm_mthd=nrm_mthd,
            learnable=blearnable,
            projection=bprojection)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#results on the untrained version of the model.
#model(features)
#over_smoothing(model.embeddings_dict)

checkpoint = torch.load(path + 'model-optimised.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print (values.shape)
nrm = SphToAdj(indices, model.edge_weights.detach(), size)

print ('-')
print (nrm[torch.where(indices[0] == 2)[0]])
print (nrm[torch.where(indices[0] == 3)[0]])
print (nrm[torch.where(indices[0] == 4)[0]])
print (nrm[torch.where(indices[0] == 5)[0]])
print (nrm[torch.where(indices[0] == 6)[0]])
print (nrm[torch.where(indices[0] == 7)[0]])
print (nrm[torch.where(indices[0] == 8)[0]])
print (nrm[torch.where(indices[0] == 9)[0]])

exit()

#results on the trained version of the model.
H_diff, H_dim = over_smoothing(embeddings_dict)
print (H_diff)
print (H_dim)

plt.plot(H_diff, label='H Distance', marker='o')
plt.legend()
#plt.savefig(f'Results/2021/FakeGraph/inference/H_diff.jpg')
plt.show()

plt.plot(H_dim, label='H Dimension', marker='o', color='orange')
plt.legend()
#plt.savefig(f'Results/2021/FakeGraph/inference/H_dim.jpg')
plt.show()



exit()
#path = 'Results/spsa/Cora/new/spherical/8_300_0_False_0.005'
path = 'Results/2021/Cora/performanceGCNKem/False_0.0005_softmax_'
input = torch.load(path + 'data.pt')
indices = input['indices']
values = input['values']
size = input['size']
adj = torch.sparse.FloatTensor(indices, values, size)	#this is the adj using GCN aggregation
features, labels, idx_train, idx_val, idx_test, dims, embeddings_dict = input['features'], input['labels'], input['idx_train'], input['idx_val'], input['idx_test'], input['dims'], input['embeddings']
normalise = softmax_normalisation #manual entry
#-- is the normalise method needed
spherical = True	#manual entry

model = GCN(dims=dims,
			dropout=dropout,
			adj=adj,
			spherical=spherical)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#results on the untrained version of the model.
#model(features)
#over_smoothing(model.embeddings_dict)
checkpoint = torch.load(path + 'model-optimised.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#results on the trained version of the model.
plt.plot(over_smoothing(embeddings_dict), label='300-0')
plt.legend()
plt.show()





