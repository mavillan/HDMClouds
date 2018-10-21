#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True 
#cython: infer_types=True
import copy
import numpy as np
cimport numpy as np
from sklearn.neighbors import NearestNeighbors

from libc.math cimport exp,cos,sin,pi,sqrt,log
from cython.parallel import prange

ii32 = np.iinfo(np.int32)
cdef int MAXINT = ii32.max



cdef inline double[:,:] _outer(double[:] x, double[:] y):
    """
    Computes the outer product between 1d-ndarrays x and y.
    """
    cdef int m = x.shape[0]
    cdef int n = y.shape[0]
    cdef int i,j
    cdef double[:,:] res = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            res[i,j] = x[i]*y[j]
    return res


cdef inline double _det(double[:,:] X):
    """
    Direct computation of determinant for matrices of size 2x2 and 3x3
    """
    cdef int n = X.shape[0]
    if n==2:
        return X[0,0]*X[1,1] - X[0,1]*X[1,0]
    else:
        return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
               X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
               X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])


cdef inline double[:,:] _merge(double w1, double[:] mu1, double[:,:] cov1,
                               double w2, double[:] mu2, double[:,:] cov2):
    """
    Only computes the covariance matrix of the moment preserving merge gaussian
    """
    cdef int i,j
    cdef int n = len(mu1)
    cdef double w_m = w1+w2
    cdef double[:,:] cov_m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cov_m[i,j] =  (w1/w_m)*cov1[i,j] 
            cov_m[i,j] += (w2/w_m)*cov2[i,j] 
            cov_m[i,j] += (w1*w2/w_m**2)*(mu1[i]-mu2[i])*(mu1[j]-mu2[j])
    return cov_m


cdef inline merge(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the moment preserving merge of components (w1,mu1,cov1) and
    (w2,mu2,cov2)
    """
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2)*_outer(mu1-mu2, mu1-mu2)
    return (w_m, mu_m, cov_m)


cdef inline double KLdiv(double w1, double[:] mu1, double[:,:] cov1, 
                         double w2, double[:] mu2, double[:,:] cov2):
    """
    Computation of the KL-divergence (dissimilarity) upper bound between components 
    [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
    ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    """
    cdef double[:,:] cov_m = _merge(w1, mu1, cov1, w2, mu2, cov2) 
    # computing the KL-div (upper bound) between the merge and the sum of the components
    return 0.5*((w1+w2)*log(_det(cov_m)) - w1*log(_det(cov1)) - w2*log(_det(cov2)))


def _compute_neighbors(double[:,:] mu_center, int n_neighbors):
    nn = NearestNeighbors(algorithm="ball_tree", n_jobs=-1)
    nn.fit(mu_center)
    nn_indexes = nn.kneighbors(mu_center, n_neighbors=n_neighbors+1, return_distance=False)
    # first column removed, since correspond to the index of the same row
    nn_indexes = nn_indexes[:,1:]
    # removing pairs of repeated neighbors
    cdef int i,j,jj
    cdef int M = nn_indexes.shape[0]
    cdef int max_neigh = nn_indexes.shape[1]
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if i>jj and np.any(i==nn_indexes[jj,:]):
                nn_indexes[i,j] = MAXINT
    return nn,nn_indexes.astype(np.int32) 


cdef void sort_by_indexes1(int[:] array, long[:] indexes): 
    cdef int i
    cdef int n = array.shape[0]
    for i in range(n):
        array[i] = array[indexes[i]]


cdef void sort_by_indexes2(double[:] array, long[:] indexes): 
    cdef int i,j
    cdef int n = array.shape[0]
    for i in range(n):
        array[i] = array[indexes[i]]


def build_diss_matrix(double[:] w, double[:,:] mu, double[:,:,:] cov, int[:,:] nn_indexes):
    cdef int i,j,jj
    cdef int M = nn_indexes.shape[0]
    cdef int max_neigh = nn_indexes.shape[1]
    cdef long[:]  sorted_indexes
    cdef double[:,:] diss_matrix = np.inf*np.ones((M,max_neigh),dtype=np.float64)
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: continue
            diss_matrix[i,j] = KLdiv(w[i],mu[i,:],cov[i,:,:],w[jj],mu[jj,:],cov[jj,:,:])
        sorted_indexes = np.argsort(diss_matrix[i,:])
        sort_by_indexes1(nn_indexes[i,:], sorted_indexes)
        sort_by_indexes2(diss_matrix[i,:], sorted_indexes)
        
    return diss_matrix.base


def least_dissimilar(double[:,:] diss_matrix, int[:] indexes, int[:,:] nn_indexes):
    # number of mixture components still alive
    cdef int _i,i,j
    cdef int M = len(indexes)
    cdef int max_neigh = diss_matrix.shape[1]
    cdef int i_min = -1
    cdef int j_min = -1
    cdef double diss_min = np.inf
    for _i in range(M):
        i = indexes[_i]
        for j in range(max_neigh):
            if diss_matrix[i,j]==-1: continue
            if diss_matrix[i,j]==np.inf: break
            if diss_matrix[i,j]<diss_min:
                diss_min = diss_matrix[i,j]
                i_min = i
                j_min = nn_indexes[i,j]
            break
    return i_min,j_min


cdef int get_index(int[:] array, int value):
    cdef int i
    cdef int n = len(array)
    for i in range(n):
        if array[i]==value: return i
    return -1


cdef update_merge_mapping(int[:] merge_mapping, int nindex, int dindex):
    cdef int i
    cdef int n = len(merge_mapping)
    for i in range(n):
        if merge_mapping[i]==dindex:
            merge_mapping[i] = nindex


def neighbors_search(nn, mu, n_neighbors, merge_mapping, nindex):
    if len(mu)==2: radius=np.sqrt(2)
    if len(mu)==3: radius=np.sqrt(3)
    dist,ind = nn.radius_neighbors([mu], radius=radius, return_distance=True)
    dist = dist[0]; ind = ind[0]
    # sorting the results
    sorted_indexes = np.argsort(dist)
    neigh_array = ind[sorted_indexes]
    # applying the mapping
    neigh_array = merge_mapping[neigh_array]
    # removing repeated neighbors and mainting the order!
    _,unique_indexes = np.unique(neigh_array, return_index=True)
    unique_indexes = np.sort(unique_indexes)
    neigh_array = neigh_array[unique_indexes]
    # removing nindex from neighbors
    neigh_array = np.delete(neigh_array, get_index(neigh_array,nindex))
    # returning the first n_neighbors neighbors
    ret = MAXINT*np.ones(n_neighbors, dtype=np.int32)
    ret[0:len(neigh_array)] = neigh_array[0:n_neighbors]
    return ret



def update_structs(int[:,:] nn_indexes, double[:,:] diss_matrix, double[:] w, double[:,:] mu, 
                      double[:,:,:] cov, int[:] indexes, int nindex, int dindex):
    """
    Updates the nn_indexes and diss_matrix structs by removing the items
    corresponding to nindex and dindex
    """
    # number of mixture components still alive
    cdef int _i,i,j,jj
    cdef int num_comp = len(indexes)
    cdef int max_neigh = nn_indexes.shape[1]
    for _i in range(num_comp):
        i = indexes[_i]
        if i==nindex: continue # this is an special case (see below)
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            if jj==nindex or jj==dindex:
                nn_indexes[i,j] = -1
                diss_matrix[i,j] = -1       

    # the special case...
    for j in range(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj!=MAXINT:
            diss_matrix[nindex,j] = KLdiv(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])
        else:
            diss_matrix[nindex,j] = np.inf

    cdef long[:] sorted_indexes = np.argsort(diss_matrix[nindex,:])
    sort_by_indexes1(nn_indexes[nindex,:], sorted_indexes)
    sort_by_indexes2(diss_matrix[nindex,:], sorted_indexes)



def mixture_reduction(w, mu, cov, n_comp=1, n_neighbors=None, 
    verbose=True, build_htree=False):
    """
    Gaussian Mixture Reduction Through KL-upper bound approach
    """
    # current mixture size
    cdef int cur_mixture_size = len(w)
    # target mixture size
    cdef int tar_mixture_size = n_comp
    # dimensionality of data
    cdef int d = mu.shape[1]

    # needed conversions
    w = np.copy(w)
    mu = np.copy(mu)
    cov = np.copy(cov)
    if cov.ndim==1:
        # if cov is 1-dimensional we convert it to its covariance matrix form
        cov = np.asarray( [(sig**2)*np.identity(d) for sig in cov] )

    if build_htree:
        # hierarchical tracking data structures
        decomp_dict = dict()
        join_dict = dict()
        entity_dict = {i:[i] for i in range(cur_mixture_size)}
        # the below dict maps the indexes of the current GM components,
        # to the indexes of the current cloud entities
        entity_key_mapping = {i:i for i in range(cur_mixture_size)}
        # label for the next entity to be added
        new_entity = cur_mixture_size 

    # we consider neighbors at a radius equivalent to the lenght of k_sigma*sigma_max
    if n_neighbors is None:
        # one neighbor for each considered degree of freedom
        if d==2: n_neighbors=8
        if d==3: n_neighbors=24

    indexes = np.arange(cur_mixture_size, dtype=np.int32)
    nn,nn_indexes = _compute_neighbors(mu,n_neighbors)

    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(cur_mixture_size, dtype=np.int32)
    
    # computing the initial dissimilarity matrix
    diss_matrix= build_diss_matrix(w, mu, cov, nn_indexes)

    # main loop
    while cur_mixture_size>tar_mixture_size:
        # approximated GMR for improved performance
        i_min,j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
        w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                             w[j_min], mu[j_min], cov[j_min])
        # updating structures
        nindex = min(i_min,j_min) # index of the new component
        dindex = max(i_min,j_min) # index of the del component
        w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
        indexes = np.delete(indexes, get_index(indexes,dindex))
        update_merge_mapping(merge_mapping, nindex, dindex)
        nn_indexes[nindex] = neighbors_search(nn, mu_m, n_neighbors, merge_mapping, nindex)
        update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)

        cur_mixture_size -= 1
        if verbose: print('{2}: Merged components {0} and {1}'.format(i_min, j_min, cur_mixture_size)) 

        if build_htree:
            # updating the hierarchical tracking structures
            i_min = entity_key_mapping[i_min]
            j_min = entity_key_mapping[j_min] 
            decomp_dict[new_entity] = (i_min,j_min)
            join_dict[(i_min,j_min)] = new_entity
            entity_dict[new_entity] = entity_dict[i_min]+entity_dict[j_min]

            entity_key_mapping[nindex] = new_entity
            del entity_key_mapping[dindex]
            new_entity += 1

    if not build_htree: 
        return w[indexes],mu[indexes],cov[indexes]
    return decomp_dict,join_dict,entity_dict



