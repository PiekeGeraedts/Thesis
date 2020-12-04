"""
Check behaviour of the normalisation methods we use
"""


import numpy as np
from Markov_chain.Markov_chain_new import MarkovChain
import matplotlib.pyplot as plt

def softmax_normalisation(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def squared_normalisation(mx):
	return mx**2/sum(mx**2)

def subtract_normalisation(mx, eps=0):
	min_mx = min(mx)
	if (min_mx < 0):
		mx -= min_mx
	mx += eps
	mx = mx/sum(mx)	
	return mx

def subtract_normalisation2(mx, eps=0):	
	min_mx = min(mx)
	if (min_mx < 0):
		mx -= min_mx	
	x = sum(mx)/(1-len(mx)*eps)
	return mx/x + eps
	

def spsa(mx, eta):
	n = mx.shape[0]
	delta = np.random.choice([-1,1], (n,n))
	P1, P2 = normalise1(mx + eta*delta), normalise1(mx - eta*delta)
	K1, K2 = MarkovChain(P1).K, MarkovChain(P2).K

	return (K1-K2)/2*eta*delta

'''
N = 100
A = np.load('A_50.npy')
eps = 10**-8
A_lst = []
grad_lst = []

for i in range(N):
	eta = 1/(i+1)
	grad = spsa(A, eta)
	A = A - eps*grad
	#keep track
	grad_lst.append(np.linalg.norm(grad))
'''

K = 10
#SOFTMAX
for k in range(K):
	n = 4
	N = 10
	lst = np.zeros((N+1, n))
	lst[0] = np.random.randn(n)

	for i in range(1,N+1):
		lst[i] = softmax_normalisation(lst[i-1])
	
	plt.plot(lst)
	plt.title('softmax normalisation')
plt.show()

#SQUARED
for k in range(K):
	n = 4
	N = 10
	lst = np.zeros((N+1, n))	
	lst[0] = np.random.randn(n)
	for i in range(1,N+1):	lst[i] = squared_normalisation(lst[i-1])
		
	plt.plot(lst)
	plt.title('squared normalisation')
plt.show()

#SUBTRACT
for k in range(K):
	n = 4
	N = 10
	lst = np.zeros((N+1, n))
	lst[0] = np.random.randn(n)

	for i in range(1,N+1):
		lst[i] = subtract_normalisation2(lst[i-1],0.01)

	plt.plot(lst)
	plt.title('subtract normalisation')
plt.show()
print (lst[N])
exit()

#testing subtract2
N = 100
n=3
eps = 0.001
lst1 = []
lst2 = []
lst3 = []

for i in range(N):	
	vec = np.random.randn(n)
	lst1.append(min(subtract_normalisation2(vec.copy(),eps)))

	
plt.plot(lst1, label='minima eps=0.01')
plt.plot([eps], marker='.', label=eps)
plt.legend()
plt.title('minimum of adjusted subtract normalisation')
plt.show()

print (f'for eps={eps} minimum of minima is: {min(lst1)}')


