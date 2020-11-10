'''
Author: 
    Pieke
Date created: 
    19/10/2020    
Purpose:
    Make tools (for spherical coordinate transformation).
'''
import numpy as np
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Markov_chain.Markov_chain_new import MarkovChain

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    

#NOTE: The four methods for the spherical transformation and back work. However, the error from rounding might be a bit large.
def toSpherical(vec):
    """n-dimensional cartesian vector to (n-1)-dimensional spherical vector"""
    n = vec.shape[0]
    sph = torch.zeros(n-1)
    vec = torch.sqrt(vec)
    for i in range(n-1):
        if (torch.equal(vec[i:], torch.zeros(n-i))):
            break #All angles remain at zero. Using below formulas will give division by 0
        if (i == n-2):  
            if (vec[i] >= 0): 
                sph[i] = torch.acos(vec[i]/euclidean(vec[i:], torch.zeros(n-i)))   
            else:
                sph[i] = 2*torch.pi - torch.acos(vec[i]/euclidean(vec[i:], torch.zeros(n-i)))   
        else:
            sph[i] = torch.acos(vec[i]/euclidean(vec[i:], torch.zeros(n-i)))
    return sph

def AdjToSph(indices, values, size):
    """convert the sparse adjacency matrix to a spherical coordinate parameterization
        Note: only returns the values."""
    sph_values = []
    for i in range(size[0]):
        idx = torch.where(indices[0] == i)[0]
        sph_values.append(toSpherical(values[idx]))
    #Note: the angles are sorted as: first all angles of row 0, then all angles of row 1, etc.
    sph_values = torch.FloatTensor([item for sublist in sph_values for item in sublist])
    return torch.FloatTensor(sph_values)

def toP(sph):
    """n-1-dimensional spherical vector to (n)-dimensional cartesian vector"""
    n = sph.shape[0]+1
    cart = torch.zeros(n)
    for i in range(n):
        if i==0:
            cart[0] = torch.cos(sph[0])
        elif i == n-1:
            cart[n-1] = torch.prod(torch.sin(sph))
        else:
            cart[i] = torch.cos(sph[i])*torch.prod(torch.sin(sph[:i]))
    return cart**2

def SphToAdj(indices, values, size):
    """convert the sparse spherical matrix to a sparse adjacency matrix.
        Note: only returns the values."""
    values = values
    cart_values = torch.zeros(len(indices[0]))
    sum_idx = 0
    for i in range(size[0]):
        idx = torch.where(indices[0] == i)[0]
        cart_values[idx] = toP(values[sum_idx:sum_idx+len(idx)-1])
        sum_idx += len(idx) - 1
    return cart_values

def KemenySpherical(indices, values, size):
    """Calculate Kemeny's constant. values should be spherical coordinates """
    P = torch.sparse.DoubleTensor(indices, SphToAdj(indices, values.detach(), size), size)
    return torch.DoubleTensor([MarkovChain(P.detach().to_dense().numpy()).K])




