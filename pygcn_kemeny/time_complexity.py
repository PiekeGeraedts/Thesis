"""
Date created:
	26-01-2021
Purpose:
	Compare running time of the SPSA gradient estimator with the analytical gradient and with FD gradient estimator.
	ToDO: add the FD implementation, use pytorch.
"""
import numpy as np
from Markov_chain.Markov_chain_new import MarkovChain
import torch
from torch.nn.parameter import Parameter
from torch.autograd import gradcheck
from simplex import squared, softmax, subtract, paper, SPSA_kem
import time

def derivP(P, i, j):
	dP = np.zeros_like(P)
	dP[i,j] = 1
	return dP

def derivPi(P, i, j):
	MC = MarkovChain(P)
	dP = derivP(P,i,j)
	return MC.Pi@dP@MC.D

def derivD(P, i, j):
	D = MarkovChain(P).D
	dPi = derivPi(P, i, j)
	return -dPi@D + D@dPi@D

def derivK(P, i, j):
	return np.trace(derivD(P, i, j))

def gradK(P):
	n = P.shape[0]
	gradK = np.zeros_like(P)
	for i in range(n):
		for j in range(n):
			gradK[i,j] = derivK(P,i,j)
	return gradK

def derivFD_K(P, normalise, i ,j):
	h=1e-4
	P1 = P.copy()
	P1[i,j] += h
	#P[i,j] -= h
	K1 = MarkovChain(normalise(P1)).K
	K2 = MarkovChain(normalise(P)).K
	return (K1-K2)/2*h

def FD_K(P, normalise):
	n = P.shape[0]
	gradK = np.zeros_like(P)
	for i in range(n):
		for j in range(n):
			gradK[i,j] = derivFD_K(P,normalise,i,j)
	return gradK

def Kemeny(P):
	with torch.no_grad():
		tmp = P.clone().detach().numpy()
	
		tmp = torch.from_numpy(tmp)  
		K = torch.FloatTensor(MarkovChain(tmp).K)
	return K


# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
#input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))


if __name__ == '__main__':
	N= 100
	n= 10
	eta = 1e-4
	normalise = subtract
	random_mx = np.random.randn(N,n,n)
	t1_lst = []
	t2_lst = []
	t3_lst = []

	for i in range(N):
		P = normalise(random_mx[0])
		t = time.time()
		SPSA_kem(P, normalise, eta)
		t1_lst.append(time.time()-t)
		t = time.time()
		gradK(P)
		t2_lst.append(time.time()-t)
		t = time.time()
		FD_K(P, normalise)
		#input = torch.randn(n,n,dtype=torch.double,requires_grad=True)
		#input = Parameter(torch.randn(n,n, dtype=torch.double))
		
		#input.data = torch.DoubleTensor(normalise(input.clone().detach().numpy()))
		#test = gradcheck(Kemeny, input, eps=1e-6, atol=1e-4, check_undefined_grad=False)
		#torch.autograd.functional.jacobian(Kemeny, input)
		t3_lst.append(time.time()-t)
		print(i)

		
	print ('average computation time SPSA:', sum(t1_lst)/N)
	print ('average computation time analytical:', sum(t2_lst)/N)
	print ('average computation FD:', sum(t3_lst)/N)






