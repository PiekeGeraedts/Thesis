"""
Date created: 
	19-01-2020
Purpose:
	Calculate difference in bias and variance of Kemeny's constant, and hence the SPSA estimator, using vairous normalisation methods.
	Do not use this to calculate the bias. The code works, but the idea behind it is not good. Here we simply calculate SPSA estimator for a single transition matrix and
	repeat this. In other words, eps=0. But we need epsilon to roam.
"""
import time
import numpy as np
import torch
from Markov_chain.Markov_chain_new import MarkovChain
from MC_derivative import gradientK
from simplex import softmax, squared, subtract, paper, mx_to_sph, sph_to_mx, SPSA_kem, SPSA_kem_sph
import matplotlib.pyplot as plt
#import torch.nn.functional as F


def g(x):
	return np.sum(x**2)

def dg(x, i):
	"""derivative of g w.r.t. x_i"""
	return 2*x[i]

def f(x):
	return np.exp(x)/np.sum(np.exp(x))

def df(x, i, j):
	"""derivative of f_i w.r.t. x_j"""
	return f(x)[j]*(1 - f(x)[i])

def SPSA_f(x, eta):
	n = x.shape[0]
	#eta = 1e-4
	Delta = np.random.choice([-1,1], n)
	f1 = g(x + eta*Delta)
	f2 = g(x - eta*Delta)
	return (f1-f2)/(2*eta*Delta)

def sample(V, N, normalise, eta):
	beta = 0.9
	SPSA_lst = []
	moving_average = []
	exponential_mvg_avg = []
	K1_lst = []
	K2_lst = []
	for i in range(N):
		print (i)
		#eta = 1/np.sqrt(i+1)
		K1, K2, SPSA = SPSA_kem(V, normalise, eta)
		K1_lst.append(K1)
		K2_lst.append(K2)
		SPSA_lst.append(SPSA)
		moving_average.append(sum(SPSA_lst)/(i+1))
		if i!= 0:
			exponential_mvg_avg.append(beta*exponential_mvg_avg[i-1] + (1-beta)*moving_average[i])
		else: 
			exponential_mvg_avg.append(moving_average[i])

	return np.array(K1_lst), np.array(K2_lst), np.array(SPSA_lst), np.array(moving_average), np.array(exponential_mvg_avg)

def sample_spherical(V, N, normalise, eta):
	beta = 0.9
	SPSA_lst = []
	moving_average = []
	exponential_mvg_avg = []
	K1_lst = []
	K2_lst = []
	for i in range(N):
		print (i)
		eta = 1/np.sqrt(i+1)
		K1, K2, SPSA = SPSA_kem_sph(V, eta)
		K1_lst.append(K1)
		K2_lst.append(K2)
		SPSA_lst.append(SPSA)
		moving_average.append(sum(SPSA_lst)/(i+1))
		if i!= 0:
			exponential_mvg_avg.append(beta*exponential_mvg_avg[i-1] + (1-beta)*moving_average[i])
		else: 
			exponential_mvg_avg.append(moving_average[i])

	return np.array(K1_lst), np.array(K2_lst), np.array(SPSA_lst), np.array(moving_average), np.array(exponential_mvg_avg)

# tmp
'''
N= 2000	#for N=1e4 slightly did not converge, see Desktop snapshot (for f)
np.random.seed(1234)
x = np.random.randn(3)
print ('x=',x)
SPSA_lst = []
moving_average = []
for i in range(N):
	eta = 1/(i+1)
	SPSA_lst.append(SPSA_f(x, eta))
	moving_average.append(sum(SPSA_lst)/(i+1))

spsa = np.array(SPSA_lst)[:,0]

moving_average = np.array(moving_average)[:,0]

plt.plot([dg(x,0)]*N, label='g')
#plt.plot(spsa, label='spsa')
plt.plot(moving_average, label='moving average')
plt.legend()
plt.show()
exit()
plt.plot(spsa, label='moving average')
plt.legend()
plt.show()
exit()
'''

# we consider a 3x3 matrix
N = 200
n = 3 
normalise = softmax	#sq, sub, sft

# Projection
np.random.seed(1234456)	
#--seed=1 gives decent convergence for squared
#--for many many different seeds between 0-5000 we did not have convergence for softmax
#--seed=9 gives decent convergence for subtract, sometimes though we have slight divergence as N gets larger.

V = np.random.rand(3,3)
#V = np.array([[3,3,3],[3,3,3],[3,3,3]])
P =  np.transpose(np.array([[0.52132701, 0.69107643, 0.13155063],
 			   				[0.34230885, 0.14292281, 0.51690021],
 			   				[0.13636415, 0.16600076, 0.35154916]]))
P = normalise(V)
#Spherical
V = np.random.rand(3,2)
P = sph_to_mx(V)

K = MarkovChain(P).K
for N in [500]:#,500,750]:
	lst = []
	eta_lst = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
	variance = []
	for eta in eta_lst:
		#K1, K2, spsa, moving_average, exp_mvg_avg = sample(V, N, normalise, eta)
		K1, K2, spsa, moving_average, exp_mvg_avg = sample_spherical(V, N, normalise, eta)
		variance.append(sum((spsa-moving_average[N-1])**2)/(N-1))
		#gradient = gradientK(mx_to_sph(P))
		gradient = gradientK(V)
	variance = np.array(variance)

	#
	print ('transition matrix:\n', P)
	print ('Kemeny constant:', K)
	print ('gradient Kemeny:\n', gradient)
	print ('mean matrix:\n', moving_average[N-1])
	print ('std. dev. matrix:\n', np.sqrt(variance[-1]))

	# plotting
	path = 'Results/2021/other/'

	# spsa converging to unbiased estimator
	'''
	plt.plot(spsa[:,0,0], label='spsa gradient')
	plt.plot([gradient[0,0]]*N, label='analytical gradient')
	plt.plot(moving_average[:,0,0], label='moving average spsa')
	#plt.plot(exp_mvg_avg[:,0,0], label='exp. mvg. avg. spsa')
	plt.ylabel('gradient')
	plt.xlabel('iteration')
	plt.legend()
	plt.title('Convergence SPSA estimator')
	plt.legend()
	#plt.savefig(path+str(int(time.time())))
	plt.show()

	# K plots
	K_avg = (K1+K2)/2
	plt.plot(K1, label='K1')
	plt.plot(K2, label='K2')
	plt.plot([K]*N, label='K')
	#plt.plot(K_avg, label='average K1 & K2')
	plt.legend()
	plt.ylabel('kemeny')
	plt.xlabel('iteration')
	plt.title('Kemeny Samples')
	plt.legend()
	#plt.savefig(path+str(int(time.time())))#'squaredK12.jpg')
	plt.show()
	'''

	# variance w.r.t. eta

	plt.scatter(np.log10(np.array(eta_lst)), variance[:,0,0], label=str(N))
	plt.ylabel('variance')
	plt.xlabel(r'$log(\eta)$')
	plt.title(r'variance vs $\eta$')
	#plt.legend()
plt.savefig(path + str(int(time.time())) + 'spherical_variance.jpg')#str(int(time.time())))#'squaredK12.jpg')
plt.show()

'''
plt.scatter(np.log10(np.array(eta_lst)), variance[:,0,0])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,0,1])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,0,2])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,1,0])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,1,1])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,1,2])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,2,0])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,2,1])
plt.scatter(np.log10(np.array(eta_lst)), variance[:,2,2])
plt.ylabel('variance')
plt.xlabel(r'$log(\eta)$')
plt.show()
'''

exit()
# spherical substitution
N = 250
n = 3 

V = np.random.rand(3,2)
#V = np.array([[3,3,3],[3,3,3],[3,3,3]])
P = sph_to_mx(V)
#P =  np.transpose(np.array([[0.52132701, 0.69107643, 0.13155063],
# 			   				[0.34230885, 0.14292281, 0.51690021],
# 			   				[0.13636415, 0.16600076, 0.35154916]]))
#--obtained randomly
#P = normalise(V)
K = MarkovChain(P).K
lst = []
eta_lst = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
variance = []
for eta in eta_lst:
	K1, K2, spsa, moving_average, exp_mvg_avg = sample_spherical(V, N, normalise, eta)
	variance.append(sum((spsa-moving_average[N-1])**2)/(N-1))
	gradient = gradientK(mx_to_sph(P))
variance = np.array(variance)

plt.plot(spsa[:,0,0], label='spsa gradient')
plt.plot([gradient[0,0]]*N, label='analytical gradient')
plt.plot(moving_average[:,0,0], label='moving average spsa')
#plt.plot(exp_mvg_avg[:,0,0], label='exp. mvg. avg. spsa')
plt.ylabel('gradient')
plt.xlabel('iteration')
plt.legend()
plt.title('Convergence SPSA estimator')
plt.legend()
#plt.savefig(path+str(int(time.time())))
plt.show()


