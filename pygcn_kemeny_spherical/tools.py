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
from scipy.spatial.distance import euclidean

def toSpherical(vec):
    """n-dimensional cartesian vector to (n-1)-dimensional spherical vector"""
    n = vec.shape[0]
    sph = np.zeros(n-1)
    vec = np.sqrt(vec)
    for i in range(n-1):
        if (np.array_equal(vec[i:], np.zeros(n-i))):
            break #All angles remain at zero. Using below formulas will give division by 0
        if (i == n-2):  
            if (vec[i] >= 0): 
                sph[i] = np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))   
            else:
                sph[i] = 2*np.pi - np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))   
        else:
            sph[i] = np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))
    return sph

def AdjToSph(indices, values, size):
    """convert the sparse adjacency matrix to a spherical coordinate parameterization"""
    sph_values = []
    for i in range(size):
        idx = np.where(indices[0] == i)[0]
        sph_values.append(toSpherical(adj_values[idx]))
    sph_values = np.array([item for sublist in sph_values for item in sublist])
    return sph_values

indices = np.load('indices.npy')
values = np.load('values.npy')
adj = np.load('adj.npy')
size = np.load('size.npy')
adj_values = np.load('adj_values.npy')

AdjToSph(indices, values, size)



