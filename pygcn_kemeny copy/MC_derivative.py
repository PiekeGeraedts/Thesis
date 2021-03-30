# -*- coding: utf-8 -*-
"""
Created on 05-04-2020 (last checked all derivative functions on 08-04-2020)

Author: Pieke Geraedts

Description: This script uses the MarkovChain class functionality to provide derivatives for relevant Markov chain matrices (i.e. P, D, M, Pi, pi) w.r.t. a single element of Theta.

"""
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
from scipy.spatial.distance import euclidean
import time
import scipy.linalg as la
import sys
import random
from matplotlib import pyplot as plt
import copy

def toTheta(P):
    '''
    Purpose: 
        Convert the P^0.5 transition matrix using cartesian coordinates to a Theta transition matrix using spherical coordinates
    Input:
        P - a (nxn)-matrix with transition probabilities
    Output:
        Theta - a (nxn)-matrix representing each row of the P matrix in spherical coordinates, corresponding to a row in mSph. 
                i.e., each row in Theta respresents a row in P.
    '''
    n, _ = P.shape  
    P = np.sqrt(P)
    Theta = np.zeros((n,n-1))
    for j in range(n):
       # Theta[j,0] = euclidean(P[j,:],np.zeros(n))   #radius=1
        for i in range(n-1):    #determine angles
            if (np.array_equal(P[j,i:], np.zeros(n-i))):
                break #All angles remain at zero. Using below formulas will give division by 0
            if (i == n-2):  
                if (P[j,i] >= 0): 
                    Theta[j,i] = np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))   
                else:
                    Theta[j,i] = 2*np.pi - np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))   
            else:
                Theta[j,i] = np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))

    return Theta

def toP(Theta):
    '''
    Purpose: 
        Convert the Theta transition matrix using spherical coordinates to a P transition matrix using cartesian coordinates
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
    Output:
        P - a (nxn)-matrix representing each row of the Theta matrix in cartesian coordinates, corresponding to a row in P. 
                i.e., each row in P respresents a row in Theta.
    '''
    n,_ = Theta.shape   
    P = np.zeros((n,n))
    for j in range(n):
        for i in range(n):  
            if (i == n-1):
                P[j,i] = np.prod([np.sin(Theta[j,:]), np.ones(n-1)])
            else:
                P[j,i] = np.prod([np.sin(Theta[j,:i]), np.ones(i)])*np.cos(Theta[j,i])
        
    return P**2

def diag(A):
    '''
    Purpose:
        Return dg(A), where all off-diagonals are set to zero
    Input:
        A - (n,n)-matrix
    Output:
        dg_A - (n,n)-matrix
    '''
    n,_ = A.shape
    dg_A = np.zeros((n,n))
    np.fill_diagonal(dg_A, np.diag(A))

    return dg_A

def stationaryDist(Theta, mu):
    '''
    Purpose:
        Determine the stationary distribution of the Markov chain described by the transition matrix P
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        mu - a (nx1)-vector (numpy array) for the Markov chain's 'start values' or None 
    Output:
        pi - a (nx1)-vector (numpy array) with the (unique) steady state probabilities
    '''
    MC = MarkovChain(toP(Theta))
    if MC.bUniChain:   #P is a uni-chain
        pi = MC.pi[0]
    else:               #P is a multi-chain
        Pi = MC.Pi
        pi = mu@Pi    

    return pi

def derivativeP(Theta, r, c):
    '''
    Purpose:
        Calculate the partial derivative of P w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        derivP - (nxn)-matrix (numpy array) of the derivative of the P matrix 
    '''
    P = toP(Theta)
    n,_ = P.shape
    theta_r = Theta[r]  #only angels of interest
    grad_P_r = np.zeros((n-1,n))
    
    for j in range(n):
        deriv = np.zeros(n-1)
        if (j == 0):
            deriv[0] = -2*np.sin(theta_r[0])*np.cos(theta_r[0])
        elif (j == n-1):
            for k in range(n-1):  
                lst = list(set(range(n-1)) - set([k]))  
                deriv[k] = 2*np.sin(theta_r[k])*np.cos(theta_r[k])*np.prod([np.sin(theta_r[l])**2 for l in lst])
        else:
            for k in range(j):
                lst = list(set(range(j)) - set([k]))
                deriv[k] = 2*np.sin(theta_r[k])*np.cos(theta_r[k])*np.cos(theta_r[j])**2 * np.prod([np.sin(theta_r[l])**2 for l in lst])
            deriv[j] = -2*np.sin(theta_r[j])*np.cos(theta_r[j])*np.prod([np.sin(theta_r[l])**2 for l in range(j)])
        grad_P_r[:,j] = deriv
    derivP = np.zeros((n,n))
    derivP[r] = grad_P_r[c]    
    
    return derivP

def derivative_pi(Theta, r, c):
    '''
    Purpose:
        Determine the derivative of the stationary distribution w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        deriv_pi - (nx1)-vector (numpy array) the derivative of pi
    '''
    n,_ = Theta.shape
    MC = MarkovChain(toP(Theta))
    pi = MC.pi[0]
    D = MC.D
    deriv_P = derivativeP(Theta, r, c)
    #deriv_P = np.zeros((n,n))
    #deriv_P[r,c] = 1.0
    deriv_pi = pi @ deriv_P @ D

    return deriv_pi

def derivativePi(Theta, r ,c):
    '''
    Purpose:
        Calculate the derivative of the ergodic projector of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivPi - (nxn)-matrix (numpy array) of the derivative of the Pi matrix 
    '''
    n,_ = Theta.shape
    MC = MarkovChain(toP(Theta))
    Pi = MC.Pi
    
    if (MC.bUniChain == False):
        print ('The MC is not a unichain, derivative not yet determined')
        exit()

    #We have uni-chain
    return np.array([derivative_pi(Theta,r,c)]*n)

def derivativeD(Theta, r, c):
    '''
    Purpose:
        Calculate the derivative of the deviation matrix of P w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivD - (nxn)-matrix (numpy array) of the derivative of the D matrix 
    '''
    n,_ = Theta.shape
    P = toP(Theta)
    MC = MarkovChain(P)
    Pi = MC.Pi
    derivP = derivativeP(Theta, r, c)
    derivPi = derivativePi(Theta, r, c)
    inv_term = np.linalg.inv((np.identity(n) - P + Pi))
    derivD = -inv_term @ (-derivP + derivPi) @ inv_term - derivPi
    
    return derivD

def derivativeD_H(Theta, r, c):
    '''
    Purpose:
        Calculate the derivative of the deviation matrix of P w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivD - (nxn)-matrix (numpy array) of the derivative of the D matrix 
    '''
    n,_ = Theta.shape
    P = toP(Theta)
    MC = MarkovChain(P)
    Pi = MC.Pi
    D = MC.D
    derivP = derivativeP(Theta, r, c)
    derivPi = derivativePi(Theta, r, c)
    derivD = -derivPi@D + D@derivP@D
    
    return derivD

def derivativeD_SPSA(Theta, r, c, Delta):
    '''
    Purpose:
        Estimate the derivative w.r.t. Theta(r,c) of the deviation matrix using SPSA
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivD - (nxn)-matrix (numpy array) of the derivative of the D matrix 
    '''
    n, _ = Theta.shape
    nabla = 0.0001
    Theta1 = copy.deepcopy(Theta)
    Theta2 = copy.deepcopy(Theta)
    Theta1[r] = Theta[r] - nabla*Delta
    Theta2[r] = Theta[r] + nabla*Delta
    D1 = MarkovChain(toP(Theta1)).D
    D2 = MarkovChain(toP(Theta2)).D

    derivD = (D2-D1)/(2*nabla*Delta[c])
    
    return derivD

def derivativeM(Theta, r, c, Delta):
    '''
    Purpose:
        Calculate the derivative of the mean first passage time matrix of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivM - (nxn)-matrix (numpy array) of the derivative of the M matrix 
    '''
    n,_ = Theta.shape
    Ones = np.ones((n,n))
    MC = MarkovChain(toP(Theta))
    pi = MC.pi[0]
    Pi = MC.Pi
    D = MC.D
    dg_Pi = diag(Pi)
    dg_D = diag(D)
    #derivD = derivativeD(Theta,r,c)
    #derivD = derivativeD_SPSA(Theta,r,c, Delta)
    derivD = derivativeD_H(Theta,r,c)
    dg_derivD = diag(derivD)
    #derivative of the inverse of dg_Pi
    deriv_dg_Pi_inv = np.zeros((n,n))
    deriv_dg_Pi_inv[r,r]  = -1*(derivative_pi(Theta,r,c)[r]) / (pi[r]**2)
    #using derivative of M formula
    derivM = (-derivD + Ones@dg_derivD) @ np.linalg.inv(dg_Pi) + deriv_dg_Pi_inv@(np.identity(n) - D + Ones@dg_D)
    
    return derivM

def derivativeK(Theta, r, c):
    '''
    Purpose:
        Calculate the derivative of the Kemeny constant of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivK - single value for the derivative of the K matrix w.r.t. Theta(r,c)
    '''
    derivD = derivativeD(Theta, r, c)
    return np.trace(derivD)

def gradientK(Theta):
    n = Theta.shape[0]
    gradK = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            gradK[i,j] = derivativeK(Theta, i, j)
            print (j)
        print (i)
    return gradK

def derivative_pen_subset(Theta_current, r, c, subset, Theta_ini):
    '''
    Purpose:
        Calculate the derivative of the difference between intial P(Theta) and current P(Theta) w.r.t. Theta(r,c)
        - Note: derivative of initial P(Theta) is zero.
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
        subset - list of states for which outgoing transition probabiities may not be altered.
        Theta_ini - the (nx(n-1))-matrix (numpy array) with angles corresponding initial transition matrix P.
    Output: 
        derivS - (nxn)-matrix (numpy array) as derivative of defined purpose
    '''
    P_current = toP(Theta_current)
    P_ini = toP(Theta_ini)
    P_der = derivativeP(Theta_current, r, c)

    return np.sum( (P_current[r,j] - P_ini[r,j]) / max(abs(P_current[r,j] - P_ini[r,j]), 0.000001) * P_der[r,j] for j in subset )

'''
import time
Theta = np.random.rand(500,499)
n, _ = Theta.shape
Delta = np.array(random.choices([-1,1], k=n-1))# for i in range(n)])

s = time.time()
derivativeD(Theta, 2, 4)
print ("Time elapsed for normal derivative:", time.time()-s)

s = time.time()
derivativeD_SPSA(Theta, 2, 4, Delta)
print ("Time elapsed for SPSA derivative:", time.time()-s)
'''


