import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from models6 import GCN


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
			print (np.linalg.matrix_rank(dictionary['h'+str(i)].detach()))
			continue
		h_current = 'h' + str(i)
		h_previous = 'h' + str(i-1)

		diff = torch.norm(dictionary[h_current] - dictionary[h_previous])
		diff_lst.append(diff)
		print (f'Difference between output layer {i} and layer {i-1}: {diff}')
		dim_lst.append(np.linalg.matrix_rank(dictionary[h_current].detach()))
	return diff_lst, dim_lst


seed = 42
lr = 0.01
weight_decay = 5e-4
dropout = 0.5
hidden = 160
eps = 1e-3
torch.manual_seed(42)	
path = 'Results/'
nlayer = '6'
input = torch.load(path + nlayer + 'input.pt')
#print (input)
#exit()
indices = input['indices']
values = input['values']
size = input['size']
adj = torch.sparse.FloatTensor(indices, values, size)	#this is the adj using GCN aggregation
features, labels, idx_train, idx_val, idx_test, embedding_dict = input['features'], input['labels'], input['idx_train'], input['idx_val'], input['idx_test'], input['embeddings']

model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)


#checkpoint = torch.load(path + nlayer + 'model-optimised.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#results on the trained version of the model.

model(features, adj)
#exit()
lst1, lst2 = over_smoothing(model.embedding_dict)
plt.plot(lst1, label= nlayer + ' layer', marker='o')
plt.legend()
plt.savefig(f'Results/{nlayer}H_diff.jpg')
plt.show()

plt.plot(lst2, label= 'Embedding Dimension', marker='x', color='orange')
plt.legend()
plt.savefig(f'Results/{nlayer}H_dim.jpg')
plt.show()



