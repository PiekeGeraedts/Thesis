"""
Purpose:	
	Optimise Kemeny constant using different normalisation methods. Both for Cora and random graph.
Date:	
	01-12-2020
"""

import torch
import torch.optim as optim
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from simplex import mx_to_sph
from MC_derivative import gradientK
from utils import load_data, Kemeny, Kemeny_spsa, squared_normalisation, subtract_normalisation, softmax_normalisation, paper_normalisation
from Markov_chain.Markov_chain_new import MarkovChain
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--linear', action='store_true', default=False,
                    help='linearly decreasing eta.')
parser.add_argument('--sqrt', action='store_true', default=False,
                    help='square root decreasing eta.')
parser.add_argument('--log', action='store_true', default=False,
                    help='logarithmic decreasing eta')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--gamma', type=float, default=1e-4,
                    help='Penalty coefficient for Kemeny.')
parser.add_argument('--eta', type=float, default=1e-4,
                    help='coefficient for spsa')
parser.add_argument('--eps', type=float, default=1e-3,
                    help='Minimum value for edge weights.')
parser.add_argument('--clipper', action='store_true', default=False,
                    help='gradient clip on or off')
parser.add_argument('--subtract', action='store_true', default=False,
                    help='gradient clip on or off')
parser.add_argument('--squared', action='store_true', default=False,
                    help='gradient clip on or off')
parser.add_argument('--softmax', action='store_true', default=False,
                    help='gradient clip on or off')
args = parser.parse_args()

###input###
adj = torch.FloatTensor(np.load('data/A.npy')).to_sparse()
labels = torch.LongTensor(np.load('data/labels.npy'))     
features = torch.FloatTensor(np.load('data/features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))
np.random.seed(42)
torch.manual_seed(42)
#adj, features, labels, idx_train, idx_val, idx_test = load_data()

def write_to_csv(np_arr, path):
	'''format before saving'''
	if type(np_arr) is not np.ndarray:
		np_arr = np.array(np_arr)
	lst = []
	for i in range(size[0]):
		idx = np.where(indices[0] == i)[0]
		lst.append(np_arr[idx].tolist())
	with open(path, 'w', newline='\n') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',')
	    for sublist in lst:
	    	spamwriter.writerow(sublist)

def dist_center(indices, values, size):
	V = normalise(indices, values, size)
	dist = 0
	for i in range(size[0]):
		idx = torch.where(indices[0] == i)[0]
		dist += torch.norm(V[idx] - (1/idx.shape[0])*torch.ones_like(idx))
	return dist

def derivativePi(MC, i, j):
    derivP = np.zeros((MC.P.shape[0],MC.P.shape[0]))
    derivP[i,j] = 1
    return MC.Pi @ derivP @ MC.D

def derivativeD(MC, i, j):
    derivPi = derivativePi(MC, i, j)
    derivP = np.zeros((MC.P.shape[0],MC.P.shape[0]))
    derivP[i,j] = 1
    return -1*derivPi @ MC.D + MC.D @ derivP @ MC.D
    
def derivativeK(indices, values, size, idx):
    P = torch.sparse.DoubleTensor(indices, values, size)
    MC = MarkovChain(P.to_dense().numpy())
    i, j = indices[:, idx]
    derivD = derivativeD(MC, i, j)
    return np.trace(derivD)

def gradK(indices, values, size):
	grad = torch.zeros_like(values)
	for idx in range(size[0]):
		grad[idx] = derivativeK(indices, values, size, idx)
	return grad

###lists to track variables###
K_lst = []
Kdiff_lst = []
grad_lst = []
V_lst = []
loss_lst = []
t_lst = []
edge_weight_lst = []
n_eps_lst = []
n_eps_lst2 = []
dist_cnt_lst = []

###Variables###
N = args.epochs
M = 10**5	#max number for kemeny constant, makes the plots better interpretable
eps = args.eps	#for stability we do not let any edge weight become <eps.
gamma = args.gamma	

variance = []
eta_lst = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
for eta in eta_lst:
	if args.softmax:
		normalise = softmax_normalisation
	elif args.subtract:
		normalise = subtract_normalisation
	elif args.squared: 
		normalise = squared_normalisation
	else: 
		assert False, "have to specify a normalise method"

	P = MarkovChain('Courtois').P
	# make P an equally weighted matrix over neighbours
	P = np.nan_to_num(P/P)
	P = np.transpose(P/sum(P))
	adj = torch.FloatTensor(P).to_sparse()

	######
	indices = adj._indices()
	values = adj._values()
	size = adj.size()
	nnz = adj._nnz()
	rand = False		#True: weights are initiated randomly, False: weights are equal across neighbours
	if rand:
		V = torch.nn.Parameter(torch.randn(nnz))
		V_init = V.clone().detach()
	else:
		V = torch.nn.Parameter(values)
		V_init = adj._values().clone()

	optimizer = optim.Adam([V], lr=args.lr)

	SPSA = []	# for index 0,0
	analytical = []
	moving_average = []

	###Optimisation###
	for n in range(N):
		t = time.time()
		'''
		if args.linear:
			eta = 1/(n+10)
		elif args.sqrt:
			eta = 1/(np.sqrt(n)+10)
		elif args.log:
			eta = 1/(np.log(n+1)+10)
		else: #constant eta
			eta = args.eta
		'''
		optimizer.zero_grad()
		with torch.no_grad():
			#normalise V is not super relevant here, but in the gcn model we need to call this before calling forward.
			#V = normalise(indices, V, size, eps)		
			perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz)) * eta
			
			V0, V1 = torch.add(V, perturbation), torch.sub(V, perturbation) 
			V_norm, normalized0, normalized1 = normalise(indices, V.clone(), size, eps), normalise(indices, V0, size, eps), normalise(indices, V1, size, eps)
			dist_cnt_lst.append(dist_center(indices, V_norm, size))
			try:
				K, K0, K1 = Kemeny(indices, V_norm, size), Kemeny(indices, normalized0, size), Kemeny(indices, normalized1, size)
			except np.linalg.LinAlgError as err:	
				assert False, "This should NOT happen!"
		#loss = -gamma*torch.sum(torch.mul(V, gradK(indices, V_norm, size)))
		P = torch.sparse.FloatTensor(indices, V_norm.clone().detach(), size).to_dense().numpy()
		analytical.append(gradientK(mx_to_sph(P))[0,0])
		SPSA.append(torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2)[0])
		moving_average.append(sum(SPSA)/len(SPSA))
		loss = -gamma * torch.sum(torch.mul(V, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2))) 
		
		###loss = sum_i V_{ii}###
		'''
		loss = 0
		for i in range(size[0]):
			combined = torch.cat((torch.where(indices[0] == i)[0], torch.where(indices[1] == i)[0]))
			uniques, counts = combined.unique(return_counts=True)
			#should only contain one value with counts>1, namely intersection of row i with col i which is V_ii.
			loss -= V[uniques[counts > 1]]	#Here, I try both '+' and '-'. The former, I expect V_ii=0 for all i and for the latter I expect V_ii=1 for all i.
		'''
		loss.backward()
		
		#clip gradients
		if args.clipper:
			V.grad = V.grad.clamp(0,1)
		#with torch.no_grad():
		#	V_norm = normalise(indices, V.clone(), size)
		n_eps = torch.where(V_norm <= 10*eps)[0].shape[0]
		n_eps_lst.append(n_eps)
		#n_eps_lst2.append(torch.where(V_norm <= 4*eps)[0].shape[0])
		cnt = 1	
		if (n>0):
			if (abs(K/K_lst[cnt-1]) < 100):
				K_lst.append(K)
				cnt+=1
			else:
				print ('to big')
		else:
			K_lst.append(K)

		Kdiff_lst.append(K0.clamp(-M,M) - K1.clamp(-M,M))
		grad_lst.append(torch.norm(V.grad))	#is constant if optimising V_ii directly
		V_lst.append(torch.norm(V))
		loss_lst.append(abs(loss.item()))
		t_lst.append(time.time()-t)
		edge_weight_lst.append(torch.norm(normalise(indices, V, size)))
		V_lst.append(torch.norm(V))

		#how can we find the minima for each row here?
		#print ('average of minima:', torch.min(V.detach().dense(), dim=1))
		optimizer.step()
		
		print ('epoch:{:d}'.format(n),
			'loss_train:{:.4f}'.format(loss.item()),
			'K:{:.4f}'.format(K.item()),
			'K_diff:{:.4f}'.format(abs(K0-K1).item()),
			'n_eps:{:d}'.format(n_eps),
			'sum_V:{:.4f}'.format(torch.sum(V).item()),
			'eta:{:.4f}'.format(eta),
			'time:{:.4f}'.format(time.time()-t))
	SPSA = np.array(SPSA)
	moving_average = np.array(moving_average)
	variance.append(sum((SPSA-moving_average[N-1])**2)/(N-1))
	
path = f'KemenyOptimisation/other/'
plt.scatter(np.log10(np.array(eta_lst)), variance, label=str(N))
plt.ylabel('variance')
plt.xlabel(r'$log(\eta)$')
plt.title(r'variance vs $\eta$')
#plt.legend()
plt.savefig(path + str(int(time.time())) + 'softmax_variance.jpg')#str(int(time.time())))#'squaredK12.jpg')
plt.show()

'''	
###Plots###
t = time.time()
#save_path = f'KemenyOptimisation/epochs={N}/{eps}_'
if args.softmax:
	name = "softmax"
elif args.subtract:
	name = "subtract"
else: 
	name = "squared"

save_path = f'KemenyOptimisation/other/{name}_{args.eps}_'
#save_path = f'KemenyOptimisation/{t}_'
print (save_path)

fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_ylabel('Kemeny\'s constant')
ax.set_xlabel('iteration')
ax.set_title('Kemeny during optimisation')
fig.savefig(save_path + 'Kemeny.jpg')


fig, ax = plt.subplots()
ax.plot(Kdiff_lst)
ax.set_title('Kemeny Difference')
fig.savefig(save_path + '{:.0f}_KemenyDiff.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(grad_lst)
ax.set_title('Gradient Norm')
fig.savefig(save_path + '{:.0f}_grad.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(V_lst)
ax.set_title('V norm')
fig.savefig(save_path + '{:.0f}_V.jpg'.format(t))

#fig, ax = plt.subplots()
#ax.plot(loss_lst)
#ax.set_title('Training Loss')
#fig.savefig(save_path + '{:.0f}_loss.jpg'.format(t))


#fig, ax = plt.subplots()
#ax.plot(edge_weight_lst)
#ax.set_title('Edge Weights norm')
#fig.savefig(save_path + 'Edgeweightnorm.jpg')

#fig, ax = plt.subplots()
#ax.plot(V_lst)
#ax.set_title('V norm')
#fig.savefig(save_path + 'Vnorm.jpg')

fig, ax = plt.subplots()
#ax.plot(SPSA, label='spsa gradient')
max1 = np.max(moving_average)
max2 = np.max(analytical)

ax.plot(moving_average/max1, label='moving average spsa')
ax.plot(analytical/max2, label='analytical gradient')
plt.ylabel('gradient')
plt.xlabel('iteration')
plt.legend()
plt.title('Convergence SPSA estimator')
plt.legend()
plt.savefig(save_path + 'mvgavg.jpg')
#plt.show()

fig, ax = plt.subplots()
ax.plot(dist_cnt_lst)
ax.set_title('distance to center')
ax.set_ylabel('euclidean distance')
ax.set_xlabel('iteration')
fig.savefig(save_path + 'dCenter.jpg')

fig, ax = plt.subplots()
ax.plot(n_eps_lst)
ax.set_title('Number of edge weights near zero')
ax.set_ylabel(r'#$V_{ij} > \epsilon$')
ax.set_xlabel('iteration')
fig.savefig(save_path + 'neps.jpg')

#fig, ax = plt.subplots()
#ax.plot(n_eps_lst2)
#ax.set_title('Number V_ij<4eps')
#fig.savefig(save_path + '2neps.jpg')
#plt.show()
exit()
###Graph Analysis###
G = nx.DiGraph()
G_init = nx.DiGraph()
weights = normalise(indices, V.clone().detach(), size, eps).numpy()
weights_init = V_init.detach().numpy()
np.save('weights.npy', weights)
edges = torch.transpose(indices, 1, 0).numpy()
cnt = 0
for edge in edges:
	G.add_nodes_from(edge)
	G.add_edge(edge[0], edge[1], weight=weights[cnt])
	G_init.add_edge(edge[0], edge[1], weight=weights_init[cnt])
	cnt+=1

#print (weights[:50])
#print (nx.clustering(G))

#weights = [G[u][v]['weight'] for u,v in G.edges()]
#weights_init = [G_init[u][v]['weight'] for u,v in G_init.edges()]

nx.draw(G, width=weights)
#plt.show()
nx.draw(G_init, width=weights_init)
#plt.show()
values = normalise(indices, V.clone().detach(), size, eps)
#print (torch.sparse.FloatTensor(indices, values, size).to_dense())
for i in range(size[0]):
	idx = torch.where(indices[0] == i)[0]
	string1 = ''
	string2 = ''
	for i in idx:
		string1 += ' ' + str(np.round(values[i].numpy(),3))
		string2 += ' ' + str(np.round(V_init[i].numpy(),3))
	print (string2 + ' ----going to---- ' + string1)

#write_to_csv(values, save_path + 'optimised_weights2.csv')
#exit()

###Check few rows###
torch.set_printoptions(sci_mode=False)
if rand:
	print ('============initial weights============')
	#values row 0
	print ('Row 0:\n', V_init[torch.where(indices[0]==0)[0]])
	#values row 25
	print ('Row 25:\n', V_init[torch.where(indices[0]==25)[0]])
	#values row 50
	print ('Row 50:\n', V_init[torch.where(indices[0]==50)[0]])
	#values row 75
	print ('Row 75:\n', V_init[torch.where(indices[0]==75)[0]])
	V_nrmld = normalise(indices, V_init, size, eps)
	print ('\n============initial distribution============')
	#values row 0
	print ('Row 0:\n', V_nrmld[torch.where(indices[0]==0)[0]])
	#values row 25
	print ('Row 25:\n', V_nrmld[torch.where(indices[0]==25)[0]])
	#values row 50
	print ('Row 50:\n', V_nrmld[torch.where(indices[0]==50)[0]])
	#values row 75
	print ('Row 75:\n', V_nrmld[torch.where(indices[0]==75)[0]])
	print ('\n============Final distribution============')
else:
	print ("\n\nRecall, V starts of with equal weights for all neighbours!")
V = normalise(indices, V.clone().detach(), size, eps)
#values row 0
print ('Row 0:\n', V[torch.where(indices[0]==0)[0]])
#values row 25
print ('Row 25:\n', V[torch.where(indices[0]==25)[0]])
#values row 50
print ('Row 50:\n', V[torch.where(indices[0]==50)[0]])
#values row 75
print ('Row 75:\n', V[torch.where(indices[0]==75)[0]])
print ('smallest value in V:', torch.min(V))
print ('Average time per epoch:', sum(t_lst)/len(t_lst))	

###Apply KDA on the result###
#maybe later
'''