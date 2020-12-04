import torch
import torch.optim as optim
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from tools import KemenySpherical, AdjToSph, SphToAdj
from utils import load_data, Kemeny, Kemeny_spsa, squared_normalisation, subtract_normalisation, softmax_normalisation, paper_normalisation
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
parser.add_argument('--eta', type=float, default=1e-4,
                    help='coefficient for spsa')
parser.add_argument('--eps', type=float, default=1e-3,
                    help='Minimum value for edge weights.')
parser.add_argument('--clipper', action='store_true', default=False,
                    help='gradient clip on or off')
args = parser.parse_args()

###input###
adj = torch.FloatTensor(np.load('A.npy')).to_sparse()
labels = torch.LongTensor(np.load('labels.npy'))     
features = torch.FloatTensor(np.load('features.npy'))
idx_train = torch.LongTensor(range(30))
idx_val = torch.LongTensor(range(30,50))
idx_test = torch.LongTensor(range(50,100))
np.random.seed(42)
torch.manual_seed(42)
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
		perturbation = torch.FloatTensor(np.random.choice([-1,1], nnz_sph)) * eta

		V0, V1 = torch.add(V_sph, perturbation), torch.sub(V_sph, perturbation) 
		
		try:
			K, K0, K1 = KemenySpherical(indices, V_sph, size), KemenySpherical(indices, V0, size), KemenySpherical(indices, V1, size)
		except np.linalg.LinAlgError as err:	
			#torch.save(indices, 'tmp/indices.pt')
			#torch.save(V, 'tmp/V.pt')
			#torch.save(normalized0, 'tmp/nrld0.pt')
			#torch.save(normalized1, 'tmp/nrld1.pt')
			#torch.save(size, 'tmp/size.pt')
			print ('This should NOT happen!')
			exit()
	loss = -gamma * torch.sum(torch.mul(V_sph, torch.mul(torch.pow(perturbation, exponent=-1), (K0-K1)/2))) 
	
	###loss = sum_i V_{ii}###

	loss = 0
	V = SphToAdj(indices, V_sph, size)	#there should be no need to normalize this
	for i in range(size[0]):
		combined = torch.cat((torch.where(indices[0] == i)[0], torch.where(indices[1] == i)[0]))
		uniques, counts = combined.unique(return_counts=True)
		#should only contain one value with counts>1, namely intersection of row i with col i which is V_ii.
		loss -= V[uniques[counts > 1]]	#Here, I try both '+' and '-'. The former, I expect V_ii=0 for all i and for the latter I expect V_ii=1 for all i.


	loss.backward()
	#clip gradients
	if args.clipper:
		V_sph.grad = V_sph.grad.clamp(0,1)

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
		'time:{:.4f}'.format(time.time()-t))
	
###Plots###
t = time.time()
#save_path = f'KemenyOptimisation/epochs={N}/{eps}_'
save_path = f'KemenyOptimisation/'
print (save_path)

fig, ax = plt.subplots()
ax.plot(K_lst)
ax.set_title('Kemeny')
fig.savefig(save_path + '{:.0f}_Kemeny.jpg'.format(t))
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
'''
fig, ax = plt.subplots()
ax.plot(loss_lst)
ax.set_title('Training Loss')
fig.savefig(save_path + '{:.0f}_loss.jpg'.format(t))
plt.show()

###Graph Analysis###
V = SphToAdj(indices, V_sph.detach(), size)	#shouldn't need to normalize
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
plt.show()
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.clustering(G)
nx.draw(G, width=weights_init)
plt.show()

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





	