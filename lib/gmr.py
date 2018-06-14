import copy
import numba
from numba import prange
import numpy as np
from sklearn.neighbors import NearestNeighbors

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max

################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True)
def _outer(x, y):
    """
    Computes the outer production between 1d-ndarrays x and y.
    """
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res


@numba.jit('float64 (float64[:,:])', nopython=True)
def _det(X):
    """
    Direct computation of determinant for matrices of size 2x2 and 3x3
    """
    n = X.shape[0]
    if n==2:
        return X[0,0]*X[1,1] - X[0,1]*X[1,0]
    else:
        return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
               X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
               X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])



@numba.jit('float64 (float64[:], float64[:], float64[:,:])', nopython=True)
def normal(x, mu, cov):
    """
    Normal distribution with parameters mu (mean) and cov (covariance matrix)
    """
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * np.linalg.det(cov))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(cov), x-mu)))



def normalize(w, mu, cov):
    pass


def ncomp_finder(kl_hist, w_size=10):
    """
    Heuristic: If the actual diff is 1 order of magnitude
    greater than the mean of the 10 last diff values, we 
    consider this points as the estimate of the number of components
    """
    diff = np.diff(kl_hist)
    diff -= diff.min()
    diff /= diff.max()
    reached_flag = False
    
    for i in range(w_size, len(diff)):
        # If actual diff is 1 order of magnitude
        if diff[i] > 10*np.mean(diff[i-w_size:i]):
            reached_flag = True
            break
    if not reached_flag:
        # in case of no high increase is detected
        i += 1
    return len(kl_hist)-i



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the moment preserving merge of components (w1,mu1,cov1) and
    (w2,mu2,cov2)
    """
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2)*_outer(mu1-mu2, mu1-mu2)
    return (w_m, mu_m, cov_m)



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isomorphic_merge(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the isomorphic moment preserving merge of components (w1,mu1,cov1) and
    (w2,mu2,cov2)
    """
    d = len(mu1)
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2) * np.abs(_det(_outer(mu1-mu2, mu1-mu2)))**(1./d) * np.identity(d)
    return (w_m, mu_m, cov_m)



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64[:], \
            float64[:,:], float64[:,:,:])', nopython=True)
def merge_full(w, mu, cov):
    n = mu.shape[0]
    d = mu.shape[1]
    w_m = np.sum(w)
    mu_m = np.zeros(d)
    cov_m = np.zeros((d,d))

    #mean calculation
    for i in range(n):
        mu_m += w[i]*mu[i]
    mu_m /= w_m

    #covariance calculation
    for i in range(n):
        cov_m += w[i] * ( cov[i] + _outer(mu[i]-mu_m, mu[i]-mu_m) )
    cov_m /= w_m

    return (w_m, mu_m, cov_m)



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def kl_diss(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computation of the KL-divergence (dissimilarity) upper bound between components 
    [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
    ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    return 0.5*((w1+w2)*np.log(_det(cov_m)) - w1*np.log(_det(cov1)) - w2*np.log(_det(cov2)))



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the ISD (Integral Square Difference between components [(w1,mu1,cov1), (w2,mu2,cov2)])
    and its moment preserving merge. Ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    Jhr = w1*w_m * normal(mu1, mu_m, cov1+cov_m) + w2*w_m * normal(mu2, mu_m, cov2+cov_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov_m)))
    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov1))) + \
          (w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov2))) + \
          2*w1*w2*normal(mu1, mu2, cov1+cov2)
    return Jhh - 2*Jhr + Jrr



@numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:])', nopython=True)
def isd_diss_full(w, mu, sig):
    # number of components
    c = len(w)
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge_full(w, mu, sig)
    # ISD computation between merge and components
    Jhr = 0.
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = 0.
    for i in range(c):  
        Jhr += w[i]*w_m * normal(mu[i], mu_m, sig[i]+sig_m)   
    for i in range(c):
        for j in range(c):
            Jhh += w[i]*w[j] * normal(mu[i], mu[j], sig[i]+sig[j])
    return Jhh - 2*Jhr + Jrr



# def gaussian_reduction(c, mu, cov, n_comp, metric=kl_diss, verbose=True):
#     """
#     Gaussian Mixture Reduction Through KL-upper bound approach
#     """
#     d = mu.shape[1]
#     c = c.tolist()
#     mu = list(map(np.array, mu.tolist()))
#     if d==2: cov = [(s**2)*np.identity(2) for s in cov]
#     elif d==3: cov = [(s**2)*np.identity(3) for s in cov]

#     # indexes of the actual gaussian components
#     components = [i for i in range(len(c))]
#     structs_dict = {i:[i] for i in range(len(c))}
#     htree = {}
#     new_comp = len(c)

#     # main loop
#     while len(components)>n_comp:
#         m = len(components)
#         diss_min = np.inf
#         for i in range(m):
#             ii = components[i]
#             for j in range(i+1,m):
#                 jj = components[j]
#                 diss = metric(c[ii], mu[ii], cov[ii], c[jj], mu[jj], cov[jj])
#                 if diss < diss_min: 
#                     i_min = i; j_min = j
#                     ii_min = ii; jj_min = jj 
#                     diss_min = diss
#         # compute the moment preserving  merged gaussian
#         w_m, mu_m, cov_m = merge(c[ii_min], mu[ii_min], cov[ii_min], 
#                                  c[jj_min], mu[jj_min], cov[jj_min])
          
#         if verbose:
#             ISD_diss = isd_diss(c[ii_min], mu[ii_min], cov[ii_min], c[jj_min], mu[jj_min], cov[jj_min])
#             print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(ii_min, jj_min, diss_min, ISD_diss))

#         # updating structures   
#         del components[max(i_min, j_min)]
#         del components[min(i_min, j_min)]
#         components.append(new_comp)
#         c.append(w_m); mu.append(mu_m); cov.append(cov_m)
#         htree[new_comp] = (min(ii_min,jj_min), max(ii_min,jj_min))
#         tmp = structs_dict[min(ii_min,jj_min)] + structs_dict[max(ii_min,jj_min)]
#         tmp.sort()
#         structs_dict[new_comp] = tmp
#         new_comp += 1



# @numba.jit('(float64[:], float64)')
# def _compute_neighbors(mu_center, maxsig):
#     nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
#     nn.fit(mu_center)
#     neigh_indexes_arr = nn.radius_neighbors(mu_center, return_distance=False)
    
#     # creating the initial array
#     maxlen = 0
#     for arr in neigh_indexes_arr:
#         if len(arr)>maxlen:
#             maxlen = len(arr)
#     neigh_indexes = MAXINT*np.ones((len(neigh_indexes_arr),maxlen-1), dtype=np.int32)
    
#     # filling it with the correct indexes
#     for i,arr in enumerate(neigh_indexes_arr):
#         ll = arr.tolist(); ll.remove(i); ll.sort()
#         for j,index in enumerate(ll):
#             neigh_indexes[i,j] = index      
#     return nn,neigh_indexes


@numba.jit('(float64[:], float64)')
def _compute_neighbors(mu_center, maxsig):
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
    nn.fit(mu_center)
    neigh_indexes_arr = nn.radius_neighbors(mu_center, return_distance=False)
    
    # creating the initial array
    maxlen = 0
    for arr in neigh_indexes_arr:
        if len(arr)>maxlen: maxlen = len(arr)
    neigh_indexes = MAXINT*np.ones((len(neigh_indexes_arr),maxlen-1), dtype=np.int32)
    
    # filling it with the correct indexes
    for i,neigh in enumerate(neigh_indexes_arr):
        neigh = neigh[neigh>i]
        for j,neigh_index in enumerate(neigh):
            neigh_indexes[i,j] = neigh_index   
    return nn,neigh_indexes



@numba.jit('float64[:,:] (float64[:], float64[:,:], float64[:,:,:], int32[:,:])', nopython=True)
def build_diss_matrix(w, mu, cov, nn_indexes):
    M,max_neigh = nn_indexes.shape
    diss_matrix = np.inf*np.ones((M,max_neigh))
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            diss_matrix[i,j] = kl_diss(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])  
    return diss_matrix



@numba.jit('Tuple((int32,int32)) (float64[:,:], int32[:], int32[:,:])', nopython=True)
def least_dissimilar(diss_matrix, indexes, nn_indexes):
    max_neigh = diss_matrix.shape[1]
    i_min = -1; j_min = -1
    diss_min = np.inf
    for i in indexes:
        for j in range(max_neigh):
            if diss_matrix[i,j]==-1: continue
            if diss_matrix[i,j]==np.inf: break
            if diss_matrix[i,j]<diss_min:
                diss_min = diss_matrix[i,j]
                i_min = i
                j_min = nn_indexes[i,j]
    return i_min,j_min


@numba.jit('int32 (int32[:], int32)', nopython=True)
def get_index(array, value):
    n = len(array)
    for i in range(n):
        if array[i]==value: return i
    return -1


@numba.jit('(int32[:], int32, int32)', nopython=True)
def update_merge_mapping(merge_mapping, nindex, dindex):
    n = len(merge_mapping)
    for i in range(n):
        if merge_mapping[i]==dindex:
            merge_mapping[i] = nindex


@numba.jit()
def radius_search(nn, mu, max_neigh, merge_mapping, nindex):
    neigh_array = nn.radius_neighbors([mu], return_distance=False)[0]
    neigh_array = merge_mapping[neigh_array]
    neigh_array = np.unique(neigh_array)
    # just in case...
    if len(neigh_array)>max_neigh:
        neigh_array = nn.kneighbors([mu], n_neighbors=max_neigh, return_distance=False)[0]
        neigh_array = merge_mapping[neigh_array]
        neigh_array = np.unique(neigh_array)
    # removing nindex and dindex from neighbors
    neigh_array = np.delete(neigh_array, get_index(neigh_array,nindex))
    ret = MAXINT*np.ones(max_neigh, dtype=np.int32)
    ret[0:len(neigh_array)] = neigh_array
    return ret


@numba.jit('(int32[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], int32[:], int32, int32)', nopython=True, parallel=True, nogil=True)
def update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex):
    """
    Updates the nn_indexes and diss_matrix structs by removing the items
    corresponding to nindex and dindex
    """
    num_comp = len(indexes)
    max_neigh = nn_indexes.shape[1]
    for i in prange(num_comp):
        i = indexes[i]
        if i==nindex: continue # this is an special case (see below)
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            if jj==nindex or jj==dindex:
                nn_indexes[i,j] = -1
                diss_matrix[i,j] = -1       

    # the special case...
    for j in prange(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj!=MAXINT:
            diss_matrix[nindex,j] = kl_diss(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])
        else:
            diss_matrix[nindex,j] = np.inf


################################################################
# MAIN FUNCTION
################################################################

def mixture_reduction(w, mu, cov, n_comp, verbose=True):
    """
    Gaussian Mixture Reduction Through KL-upper bound approach
    """
    w = np.copy(w)
    mu = np.copy(mu)
    cov = np.copy(cov)

    # original size of the mixture
    M = len(w) 
    # target size of the mixture
    N = n_comp
    # dimensionality of data
    d = mu.shape[1]

    # we consider neighbors at a radius equivalent to the lenght of 5 pixels
    if cov.ndim==1:
        maxsig = 5*np.max(cov)
        # if cov is 1-dimensional we convert it to its covariance matrix form
        cov = np.asarray( [(val**2)*np.identity(d) for val in cov] )
    else:
        maxsig = 5*max([np.max(np.linalg.eig(_cov)[0])**(1./2) for _cov in cov])

    indexes = np.arange(M, dtype=np.int32)
    nn,nn_indexes = _compute_neighbors(mu,maxsig)

    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(M, dtype=np.int32)

    # max number of neighbors
    max_neigh = nn_indexes.shape[1]
    
    # computing the initial dissimilarity matrix
    diss_matrix = build_diss_matrix(w, mu, cov, nn_indexes)  
    
    # main loop
    while M>N:
        i_min, j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
        if verbose: print('Merged components {0} and {1}'.format(i_min, j_min))  
        w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                                 w[j_min], mu[j_min], cov[j_min])
 
        # updating structures
        nindex = min(i_min,j_min) # index of the new component
        dindex = max(i_min,j_min) # index of the del component
        w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
        indexes = np.delete(indexes, get_index(indexes,dindex))
        update_merge_mapping(merge_mapping, nindex, dindex)
        nn_indexes[nindex] = radius_search(nn, mu_m, max_neigh, merge_mapping, nindex)
        update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)
        M -= 1

    # indexes of "alive" mixture components
    return w[indexes],mu[indexes],cov[indexes]


