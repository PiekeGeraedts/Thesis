"""
Purpose:
	project random 3-dim vector on the probability simplex using softmax, subtract, squared and spherical. 
Date:
	13-12-2020
"""
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tools import toCartesian
from sklearn.neighbors import BallTree
from scipy.spatial.distance import euclidean
from Markov_chain.Markov_chain_new import MarkovChain

def scatter_plot(data, norm_str):
    # spherical scatter plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(data[:,0], data[:,1], data[:,2], cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
    ax.set_xlabel('p_1')
    ax.set_ylabel('p_2')
    ax.set_zlabel('p_3')
    
    plt.title(norm_str)
    #plt.savefig('spherical_simplex.jpg')
    plt.show()
    

def calc_point_vicinity(data, point, eps):
	"""determine the number of points form data in an eps-ball neighbourhood of point"""
	tree = BallTree(data, leaf_size=2)	#leaf_size impacts running time not query result
	return tree.query_radius(point, r=eps, count_only=True)[0]

def calc_line_vicinity(data, line, eps):
	"""determine the number of points in a eps-rectangle neighbourhood of corner"""
	#line = 0 : line between y+z=1
	#line = 1 : line between x+z=1
	#line = 2 : line between x+y=1
	cnt = 0
	for point in data:
		if (point[line] <= eps):
			cnt +=1
	return cnt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def squared(mx):
	return mx**2/sum(mx**2)

def subtract(mx):
    if len(mx.shape) > 1:
        normalised = np.zeros_like(mx)
        for i in range(mx.shape[0]):
            normalised[i] = subtract(mx[i])
        return normalised
    else:
        min_mx = min(mx)
        if (min_mx < 0):
            mx -= min_mx
        mx = mx/sum(mx)	
        return mx

def paper(mx):
    if len(mx.shape) > 1:
       if len(mx.shape) > 1:
        normalised = np.zeros_like(mx)
        for i in range(mx.shape[0]):
            i = np.random.randint(mx.shape[0])

            normalised[i] = paper(mx[i])
        return normalised
    else:
        T = np.transpose(np.eye(3)[np.argsort(mx)[::-1]])   #this transformation makes sure the vector is sorted back to original
        mx = np.sort(mx)[::-1]  #NOTE: the [::-1] operation mirrors the array we have!
        
        for j in range(len(mx)):
            if (mx[j] + 1/(j+1)*(1-sum(mx[:(j+1)])) > 0):
                rho = j+1
            lbd = 1/rho*(1-sum(mx[:rho]))
            mx = mx + lbd
            mx = np.maximum(mx, np.zeros_like(mx))
    return T@mx

def toCartesian(sph, eps=0):
	"""the given mx (sph) is in spherical, return prob. vector."""
	#NOTE: random spherical is not possible with substitution. Is is if we use projection.
	n = sph.shape[0]+1    
	cart = np.zeros(n)
	r = np.sqrt(1 - n*eps)  #could also let r be input to the function and give it 1-n*eps in main.
	for i in range(n):
	    if i==0:
	        cart[0] = r*np.cos(sph[0])
	    elif i == n-1:
	        cart[n-1] = r*np.prod(np.sin(sph))
	    else:
	        cart[i] = r*np.cos(sph[i])*np.prod(np.sin(sph[:i]))
	return cart**2

def toSpherical(vec, eps=0):
    """the given mx is a prob. vector, return spherical vector."""
    n = vec.shape[0]
    sph = np.zeros(n-1)
    vec = np.sqrt(vec)
    for i in range(n-1):
        if (np.equal(vec[i:], np.zeros(n-i)).all()):
            break #All angles remain at zero. Using below formulas will give division by 0
        if (i == n-2):  
            if (vec[i] >= 0): 
                sph[i] = np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))   
            else:
                sph[i] = 2*np.pi - np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))   
        else:
            sph[i] = np.arccos(vec[i]/euclidean(vec[i:], np.zeros(n-i)))
    return sph

def mx_to_sph(V):
    n = V.shape[0]
    sph = np.zeros((n,n-1))
    for i in range(n):
        sph[i] = toSpherical(V[i])
    return sph

def sph_to_mx(sph):
    n = sph.shape[0]
    mx = np.zeros((n,n))
    for i in range(n):
        mx[i] = toCartesian(sph[i])
    return mx

def sum_norm(V):
    """calc sum of norm of rows of V. Assuming V is (nxn)."""
    s = 0
    for i in range(V.shape[0]):
        s += np.linalg.norm(V[i])
    return s

def SPSA_kem(V, normalise, eta):
#    eta = 1e-4
    #print(eta)
    Delta = np.random.choice([-1,1], V.shape) * eta
    V0, V1 = V+eta*Delta, V-eta*Delta
    normalised0, normalised1 = normalise(V0), normalise(V1)
    K0, K1 = MarkovChain(normalised0).K, MarkovChain(normalised1).K
    #return (K0-K1)/(2*eta*Delta)
    return K0, K1, (K0-K1)/(2*eta*Delta)

def SPSA_kem_sph(sph, eta):
    #eta = 1e-4
    Delta = np.random.choice([-1,1], sph.shape) * eta
    sph0, sph1 = sph+eta*Delta, sph-eta*Delta
    K0, K1 = MarkovChain(sph_to_mx(sph0)).K, MarkovChain(sph_to_mx(sph1)).K
    return K0, K1, (K0-K1)/(2*eta*Delta)

if __name__ == '__main__':
    # magic numbers
    N = 1000
    n = 3
    eps_ball = 0.15
    eps_rect = 0.01
    corner1 = np.array([[1,0,0]])
    corner2 = np.array([[0,1,0]])
    corner3 = np.array([[0,0,1]])
    center = np.array([[1/3,1/3,1/3]])

    # init 
    arr_sub = np.zeros((N,n))
    Vnrm_sub = np.zeros(N)
    Knrm_sub = np.zeros(N)
    arr_sqrd = np.zeros((N,n))
    Vnrm_sqrd = np.zeros(N)
    Knrm_sqrd = np.zeros(N)
    arr_sft = np.zeros((N,n))
    Vnrm_sft = np.zeros(N)
    Knrm_sft = np.zeros(N)
    arr_sph = np.zeros((N,n))
    Vnrm_sph = np.zeros(N)
    Knrm_sph = np.zeros(N)
    arr_paper = np.zeros((N,n))
    Vnrm_paper = np.zeros(N)
    Knrm_paper = np.zeros(N)


    normalise = squared
    random_mx = np.random.rand(N,n)*4-2
    random_sph = np.random.rand(N,n-1)*np.pi/2

    for i in range(N):
        i = np.random.randint(N)
        tmp = random_mx[i]
        arr_sub[i] = subtract(tmp.copy())
        arr_sqrd[i] = squared(tmp.copy())
        arr_sft[i] = softmax(tmp.copy())
        arr_paper[i] = paper(tmp.copy())
        arr_sph[i] = toCartesian(random_sph[i])


    # norm vs norm plots

    #NOTE: we take the sum of the norm of the kemeny derivatives not the norm of the sum
    for i in range(int(N/3)):
        #V = np.random.randn(n,n)
        V = random_mx[3*i:3*(i+1)]
        sph = random_sph[3*i:3*(i+1)]
        '''
        data_sub = np.sum(subtract(V.copy()), axis=0)
        Vnrm_sub[i] = np.linalg.norm(data_sub)
        Knrm_sub[i] = sum_norm(SPSA_kem(V, subtract))
        #print ('=sub=')
        #print (data_sub)
        #print (Vnrm_sub[i])
        #print (Knrm_sub[i])

        data_sqrd = np.sum(squared(V.copy()), axis=1)
        Vnrm_sqrd[i] = np.linalg.norm(data_sqrd)
        Knrm_sqrd[i] = sum_norm(SPSA_kem(V, squared))
        #print ('=sqrd=')
        #print (data_sqrd)
        #print (Vnrm_sqrd[i])
        #print (Knrm_sqrd[i])

        data_sft = np.sum(softmax(V.copy()), axis=1)
        Vnrm_sft[i] = np.linalg.norm(data_sft)
        Knrm_sft[i] = sum_norm(SPSA_kem(V, softmax))
        #print ('=sft=')
        #print (data_sft)
        #print (Vnrm_sft[i])
        #print (Knrm_sft[i])
        '''
        data_paper = np.sum(paper(V.copy()), axis=1)
        Vnrm_paper[i] = np.linalg.norm(data_paper)
        Knrm_paper[i] = sum_norm(SPSA_kem(V, paper, 1e-4)[2])
        #print ('=sft=')
        #print (data_sft)
        #print (Vnrm_sft[i])
        #print (Knrm_sft[i])

        data_sph = np.sum(sph_to_mx(sph), axis=0)
        Vnrm_sph[i] = np.linalg.norm(data_sph)
        Knrm_sub[i] = sum_norm(SPSA_kem_sph(sph, 1e-4)[2])
        #print ('=sph=')
        #print (data_sph)
        #print (Vnrm_sph[i])
        #print (Knrm_sph[i])



    # eps-ball neighbourhood
    #NOTE: two vicinity lines have a corner points as intersection.

    print (f'#####INFO AFTER {N} UNIFORMLY RANDOM GENERATED POINTS IN 3-D#####')
    print ('\u03B5-BALL NEIGHBOURHOOD =', eps_ball)
    print ('\u03B5-RECTANGLE NEIGHBOURHOOD =', eps_rect)
    print (f'=======Neighbourhood of subtract=======')
    print ('vicinity corner one=', calc_point_vicinity(arr_sub, corner1, eps_ball))
    print ('vicinity corner two=', calc_point_vicinity(arr_sub, corner2, eps_ball))
    print ('vicinity corner three=', calc_point_vicinity(arr_sub, corner3, eps_ball))
    print ('vicinity center=', calc_point_vicinity(arr_sub, center, eps_ball))
    print ('vicinity line y+z=1=', calc_line_vicinity(arr_sub, 0, eps_rect))
    print ('vicinity line x+z=1=', calc_line_vicinity(arr_sub, 1, eps_rect))
    print ('vicinity line x+y=1=', calc_line_vicinity(arr_sub, 2, eps_rect))
    '''
    print (f'=======Neighbourhood of squared=======')
    print ('vicinity corner one=', calc_point_vicinity(arr_sqrd, corner1, eps_ball))
    print ('vicinity corner two=', calc_point_vicinity(arr_sqrd, corner2, eps_ball))
    print ('vicinity corner three=', calc_point_vicinity(arr_sqrd, corner3, eps_ball))
    print ('vicinity center=', calc_point_vicinity(arr_sqrd, center, eps_ball))
    print ('vicinity line y+z=1=', calc_line_vicinity(arr_sqrd, 0, eps_rect))
    print ('vicinity line x+z=1=', calc_line_vicinity(arr_sqrd, 1, eps_rect))
    print ('vicinity line x+y=1=', calc_line_vicinity(arr_sqrd, 2, eps_rect))
    print (f'=======Neighbourhood of softmax=======')
    print ('vicinity corner one=', calc_point_vicinity(arr_sft, corner1, eps_ball))
    print ('vicinity corner two=', calc_point_vicinity(arr_sft, corner2, eps_ball))
    print ('vicinity corner three=', calc_point_vicinity(arr_sft, corner3, eps_ball))
    print ('vicinity center=',calc_point_vicinity( arr_sft, center, eps_ball))
    print ('vicinity line y+z=1=', calc_line_vicinity(arr_sft, 0, eps_rect))
    print ('vicinity line x+z=1=', calc_line_vicinity(arr_sft, 1, eps_rect))
    print ('vicinity line x+y=1=', calc_line_vicinity(arr_sft, 2, eps_rect))
    '''
    print (f'=======Neighbourhood of paper=======')
    print ('vicinity corner one=', calc_point_vicinity(arr_paper, corner1, eps_ball))
    print ('vicinity corner two=', calc_point_vicinity(arr_paper, corner2, eps_ball))
    print ('vicinity corner three=', calc_point_vicinity(arr_paper, corner3, eps_ball))
    print ('vicinity center=',calc_point_vicinity( arr_paper, center, eps_ball))
    print ('vicinity line y+z=1=', calc_line_vicinity(arr_paper, 0, eps_rect))
    print ('vicinity line x+z=1=', calc_line_vicinity(arr_paper, 1, eps_rect))
    print ('vicinity line x+y=1=', calc_line_vicinity(arr_paper, 2, eps_rect))
    #print (f'=======Neighbourhood of spherical=======')
    #print ('vicinity corner one=', calc_point_vicinity(arr_sph, corner1, eps_ball))
    #print ('vicinity corner two=', calc_point_vicinity(arr_sph, corner2, eps_ball))
    #print ('vicinity corner three=', calc_point_vicinity(arr_sph, corner3, eps_ball))
    #print ('vicinity center=', calc_point_vicinity(arr_sph, center, eps_ball))
    #print ('vicinity line y+z=1=', calc_line_vicinity(arr_sph, 0, eps_rect))
    #print ('vicinity line x+z=1=', calc_line_vicinity(arr_sph, 1, eps_rect))
    #print ('vicinity line x+y=1=', calc_line_vicinity(arr_sph, 2, eps_rect))


    #scatter_plot(arr_sub, 'subtract')
    #scatter_plot(arr_sqrd, 'squared')
    #scatter_plot(arr_sft, 'softmax')
    #scatter_plot(arr_paper, 'paper')
    scatter_plot(arr_sph, 'spherical')
    exit()


    #plt.scatter(Vnrm_sub, Knrm_sub)
    #plt.show()

    #plt.scatter(Vnrm_sqrd, Knrm_sqrd)
    #plt.show()

    #plt.scatter(Vnrm_sft, Knrm_sft)
    #plt.show()

    #plt.scatter(Vnrm_paper, Knrm_paper)
    #plt.show()

    plt.scatter(Vnrm_sph, Knrm_sph)
    plt.show()




