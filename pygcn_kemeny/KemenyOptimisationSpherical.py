"""
Purpose:	
	Optimise Kemeny constant using the spherical coordinate parameterization. Both for Cora and random graph.
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

from tools import KemenySpherical, AdjToSph, SphToAdj
from MC_derivative import gradientK
from simplex import mx_to_sph
from utils import load_data, Kemeny, Kemeny_spsa, squared_normalisation, subtract_normalisation, softmax_normalisation, paper_normalisation
from Markov_chain.Markov_chain_new import MarkovChain


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for random number')
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
args = parser.parse_args()

###input###
adj = torch.FloatTensor(np.load('data/A.npy')).to_sparse()
labels = torch.LongTensor(np.load('data/labels.npy'))     
features = torch.FloatTensor(np.load('data/features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
#adj, features, labels, idx_train, idx_val, idx_test = load_data()

###lists to track variables###
K_lst = []
K0_lst = []
K1_lst = []
Kdiff_lst = []
denominator_lst = []
grad_lst = []
V_lst = []
loss_lst = []
nEc_lst = []
nTc_lst = []
t_lst = []
n_eps_lst = []
delta_lst = []
dist_cnt_lst = []

P = MarkovChain('Courtois').P
# make P an equally weighted matrix over neighbours
P = np.nan_to_num(P/P)
P = np.transpose(P/sum(P))
adj = torch.FloatTensor(P).to_sparse()

###Variables###
N = args.epochs
M = 10**5	#max number for kemeny constant, makes the plots better interpretable
eps = args.eps	#for stability we do not let any edge weight become <eps.
gamma = args.gamma	
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
	V_init = V.clone().detach()
V_sph = torch.nn.Parameter(AdjToSph(indices, V.detach(), size))
nnz_sph = len(V_sph)

optimizer = optim.Adam([V_sph], lr=args.lr)

SPSA = []
analytical = []
moving_average = []
exp_mvg_avg = []
beta = 0.9

def dist_center(indices, values, size):
	V = values
	dist = 0
	for i in range(size[0]):
		idx = torch.where(indices[0] == i)[0]
		dist += torch.norm(V[idx] - (1/idx.shape[0])*torch.ones_like(idx))
	return dist

###Optimisation###
for n in range(N):
	t = time.time()
	if args.linear:
		eta = 1/(n+10)
	elif args.sqrt:
		eta = 1/(np.sqrt(n)+10)
	elif args.log:
		eta = 1/(np.log(n+1)+10)
	else: #constant eta
		eta = args.eta
	
	optimizer.zero_grad()
	with torch.no_grad():
		#print (V[torch.where(indices[0]==0)[0]]) #gives an idea of the behaviour of the optimisation
		delta = torch.FloatTensor(np.random.choice([-1,1], nnz_sph))
		perturbation = delta * eta
		#print (indices[0])
		#print (indices[1])
		#exit()
		delta_lst.append(delta[3])
		V0, V1 = torch.add(V_sph, perturbation), torch.sub(V_sph, perturbation) 
		V_norm = SphToAdj(indices, V_sph, size)
		dist_cnt_lst.append(dist_center(indices, V_norm, size))
		try:
			K, K0, K1 = KemenySpherical(indices, V_sph, size, eps), KemenySpherical(indices, V0, size, eps), KemenySpherical(indices, V1, size, eps)
		except np.linalg.LinAlgError as err:	
			print ('This should NOT happen!')
			exit()
	#P = torch.sparse.FloatTensor(indices, V_norm.clone().detach(), size).to_dense().numpy()
	P = torch.sparse.FloatTensor(indices, SphToAdj(indices, V_sph.detach(), size, eps), size).to_dense().numpy()
	analytical.append(gradientK(mx_to_sph(P))[0,4])
	SPSA.append(torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2)[3])
	moving_average.append(sum(SPSA)/len(SPSA))
	if n!= 0:
		exp_mvg_avg.append(beta*exp_mvg_avg[n-1] + (1-beta)*SPSA[n])
	else: 
		exp_mvg_avg.append(SPSA[n])
	loss = -gamma * torch.sum(torch.mul(V_sph, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2))) 
	
	###loss = sum_i V_{ii}###
	'''
	loss = 0
	V = SphToAdj(indices, V_sph, size)	#there should be no need to normalize this
	for i in range(size[0]):
		combined = torch.cat((torch.where(indices[0] == i)[0], torch.where(indices[1] == i)[0]))
		uniques, counts = combined.unique(return_counts=True)
		#should only contain one value with counts>1, namely intersection of row i with col i which is V_ii.
		loss -= V[uniques[counts > 1]]	#Here, I try both '+' and '-'. The former, I expect V_ii=0 for all i and for the latter I expect V_ii=1 for all i.
	'''

	loss.backward()
	#clip gradients
	if args.clipper:
		V_sph.grad = V_sph.grad.clamp(0,1)
	V = SphToAdj(indices, V_sph.detach(), size, eps)
	n_eps = torch.where(V <= 10*eps)[0].shape[0]
	n_eps_lst.append(n_eps)

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
	grad_lst.append(torch.norm(V_sph.grad))	#is constant if optimising V_ii directly
	V_lst.append(torch.norm(V_sph))
	loss_lst.append(loss.item())
	t_lst.append(time.time()-t)
	
	optimizer.step()
	
	print ('epoch:{:d}'.format(n),
		'loss_train:{:.4f}'.format(loss.item()),
		'K:{:.4f}'.format(K.item()),
		'K_diff:{:.4f}'.format(abs(K0-K1).item()),
		'n_eps:{:d}'.format(n_eps),
		'sum_V:{:.4f}'.format(torch.sum(V).item()),
		'time:{:.4f}'.format(time.time()-t))
	
###Plots###
t = time.time()
#save_path = f'KemenyOptimisation/epochs={N}/{eps}_'
save_path = f'KemenyOptimisation/other/spherical/spherical_{eps}_{args.lr}_{args.seed}_'
print (save_path)

#fig, ax = plt.subplots()
#ax.plot(K_lst)
#ax.set_ylabel('Kemeny\'s constant')
#ax.set_xlabel('iteration')
#ax.set_title('Kemeny during optimisation')
#fig.savefig(save_path + 'Kemeny.jpg')

analytical = np.array(analytical)
SPSA = np.array(SPSA)
moving_average = np.array(moving_average)
exp_mvg_avg = np.array(exp_mvg_avg)
delta = np.array(delta_lst)

max1 = np.max(np.abs(moving_average))
max2 = np.max(np.abs(analytical))
max3 = np.max(np.abs(exp_mvg_avg))


fig, ax = plt.subplots()
#ax.plot(SPSA, label='spsa')
#ax.plot(delta, label=r'$\Delta$')
ax.plot(moving_average/max1, label='moving average spsa')
ax.plot(analytical/max2, label='analytical gradient')
ax.plot(exp_mvg_avg/max3, label='exponential average spsa')
plt.ylabel('gradient')
plt.xlabel('iteration')
plt.legend()
plt.title('Convergence SPSA estimator')
plt.legend()
plt.savefig(save_path + 'mvgavg.jpg')
#plt.show()

#fig, ax = plt.subplots()
#ax.plot(dist_cnt_lst)
#plt.ylabel('distance')
#plt.xlabel('iteration')
#plt.show()
#print (dist_cnt_lst[-1])

#exit()

'''
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

fig, ax = plt.subplots()
ax.plot(loss_lst)
ax.set_title('Training Loss')
fig.savefig(save_path + '{:.0f}_loss.jpg'.format(t))
plt.show()
'''

fig, ax = plt.subplots()
ax.plot(n_eps_lst)
ax.set_title('Number of edge weights near zero')
ax.set_ylabel(r'#$V_{ij} > \epsilon$')
ax.set_xlabel('iteration')
#fig.savefig(save_path + 'neps.jpg')
plt.show()
exit()
###Graph Analysis###
V = SphToAdj(indices, V_sph.detach(), size, eps)	#shouldn't need to normalize
G = nx.DiGraph()
G_init = nx.DiGraph()
weights = V.numpy()
weights_init = V_init.numpy()
edges = torch.transpose(indices, 1, 0).numpy()
cnt = 0
for edge in edges:
	G.add_nodes_from(edge)
	G_init.add_nodes_from(edge)
	G.add_edge(edge[0], edge[1], weight=weights[cnt])
	G_init.add_edge(edge[0], edge[1], weight=weights_init[cnt])
	cnt+=1
#print (weights[:50])
#print (nx.clustering(G))
weights_init = [G_init[u][v]['weight'] for u,v in G_init.edges()]
nx.clustering(G_init)
nx.draw(G_init, width=weights_init)
#plt.show()
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.clustering(G)
nx.draw(G, width=weights_init)
#plt.show()

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





	