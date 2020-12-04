import torch
import torch.optim as optim
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from tools import AdjToSph, KemenySpherical
from utils import Kemeny, Kemeny_spsa#, squared_normalisation, subtract_normalisation, softmax_normalisation, paper_normalisation
from Markov_chain.Markov_chain_new import MarkovChain


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--linear', action='store_true', default=False,
                    help='linearly decreasing eta.')
parser.add_argument('--sqrt', action='store_true', default=False,
                    help='square root decreasing eta.')
parser.add_argument('--log', action='store_true', default=False,
                    help='logarithmic decreasing eta')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--gamma', type=float, default=1e-4,
                    help='Penalty coefficient for Kemeny.')
args = parser.parse_args()

###input###
adj = torch.FloatTensor(np.load('A.npy')).to_sparse()
labels = torch.LongTensor(np.load('labels.npy'))     
features = torch.FloatTensor(np.load('features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))

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

###Variables###
N = args.epochs
M = 10**5	#max number for kemeny constant, makes the plots better interpretable
gamma = args.gamma	
normalise = 0#squared_normalisation 	#NOTE: paper_normalisation does not work yet, though it seemed so because it is 'close'
indices = adj._indices()
values = adj._values()
size = adj.size()
nnz = adj._nnz()
nnz_sph = nnz - size[0]
rand = False		#True: weights are initiated randomly, False: weights are equal across neighbours
if rand:
	V = torch.nn.Parameter(torch.randn(nnz_sph))
	V_init = V.clone().detach()
else:
	V = torch.nn.Parameter(AdjToSph(indices, values, size))

optimizer = optim.Adam([V], lr=args.lr)

###Optimisation###
for n in range(N):
	t = time.time()
	if args.linear:
		eta = 1/(n+10)
	elif args.sqrt:
		eta = 1/(np.sqrt(n)+10)
	elif args.log:
		eta = 1/(np.log(n)+10)
	else: #constant eta
		eta = 1e-3
	
	optimizer.zero_grad()
	with torch.no_grad():
		#normalise V is not super relevant here, but in the gcn model we need to call this before calling forward.
		#V = normalise(indices, V, size)
		perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz_sph)) * eta

		V0, V1 = torch.add(V, perturbation), torch.sub(V, perturbation) 
		#normalized0, normalized1 = normalise(indices, V0, size), normalise(indices, V1, size)
		K, K0, K1 = KemenySpherical(indices, V, size), KemenySpherical(indices, V0, size), KemenySpherical(indices, V1, size)
		#try:
		#	(K, nEc, nTc), (K0, _, _), (K1, _, _) = Kemeny(indices, V, size), Kemeny(indices, normalized0, size), Kemeny(indices, normalized1, size)
		#except np.linalg.LinAlgError as err:	
		#	print ('This should NOT happen!')
		#	break 
	loss = -gamma * torch.sum(torch.mul(V, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2)))

	###loss = sum_i V_{ii}###
	'''
	loss = 0
	for i in range(size[0]):
		combined = torch.cat((torch.where(indices[0] == i)[0], torch.where(indices[1] == i)[0]))
		uniques, counts = combined.unique(return_counts=True)
		#should only contain one value with counts>1, namely intersection of row i with col i which is V_ii.
		loss -= V[uniques[counts > 1]]	#Here, I try both '+' and '-'. The former, I expect V_ii=0 for all i and for the latter I expect V -> I_N.
	'''
	loss.backward()

	K_lst.append(K.clamp(-M,M))
	K0_lst.append(K0.clamp(-M,M))
	K1_lst.append(K1.clamp(-M,M))
	Kdiff_lst.append(torch.abs(K0.clamp(-M,M) - K1.clamp(-M,M)))
	denominator_lst.append(torch.norm(2*perturbation))
	#nEc_lst.append(nEc)
	#nTc_lst.append(nTc)
	grad_lst.append(torch.norm(V.grad))	#is constant if optimising V_ii directly
	V_lst.append(torch.norm(V))
	loss_lst.append(loss.item())
	t_lst.append(time.time()-t)
	
	optimizer.step()

	print ('epoch:{:d}'.format(n),
		'loss_train:{:.4f}'.format(loss.item()),
		'time:{:.4f}'.format(time.time()-t))

###Plots###
t = time.time()
#save_path = f'tmp/-gammaK_subtract_symmetric_Adam/gamma={gamma}_lr={args.lr}/'
save_path = f'tmp/'
print (save_path)

fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_title('Kemeny')
fig.savefig(save_path + '{:.0f}_Kemeny.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(K0_lst)
ax.set_title('Kemeny 0')
fig.savefig(save_path + '{:.0f}_Kemeny0.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(K1_lst)
ax.set_title('Kemeny 1')
fig.savefig(save_path + '{:.0f}_Kemeny1.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(Kdiff_lst)
ax.set_title('Kemeny Difference')
fig.savefig(save_path + '{:.0f}_KemenyDiff.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(denominator_lst)
ax.set_title('Denominator Norm')
fig.savefig(save_path + '{:.0f}_Denominator.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(nEc_lst)
ax.set_title('Number of Ergodic Classes')
fig.savefig(save_path + '{:.0f}_nEc.jpg'.format(t))

fig, ax = plt.subplots()
ax.plot(nTc_lst)
ax.set_title('Number of Transient Classes')
fig.savefig(save_path + '{:.0f}_nTc.jpg'.format(t))

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


###Plot graph###
G = nx.DiGraph()
weights = normalise(indices, V.clone().detach(), size).numpy()
edges = torch.transpose(indices, 1, 0).numpy()
cnt = 0
for edge in edges:
	G.add_nodes_from(edge)
	G.add_edge(edge[0], edge[1], weight=weights[cnt])
	cnt+=1

###Graph analysis###
#print (weights[:50])
#print (nx.clustering(G))
weights = [G[u][v]['weight'] for u,v in G.edges()]
#nx.draw(G, width=weights)
#plt.show()

###Check few rows###
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
	V_nrmld = normalise(indices, V_init, size)
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
V = normalise(indices, V.clone().detach(), size)
#values row 0
print ('Row 0:\n', V[torch.where(indices[0]==0)[0]])
#values row 25
print ('Row 25:\n', V[torch.where(indices[0]==25)[0]])
#values row 50
print ('Row 50:\n', V[torch.where(indices[0]==50)[0]])
#values row 75
print ('Row 75:\n', V[torch.where(indices[0]==75)[0]])

print ('Average time per epoch:', sum(t_lst)/len(t_lst))	

###Apply KDA on the result###
#first optimise V_ii





	