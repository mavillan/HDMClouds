import copy
import numba
import numpy as np
from sklearn.neighbors import BallTree

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max

################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True, fastmath=True, nogil=True)
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
    

@numba.jit('float64 (float64[:,:])', nopython=True, fastmath=True, nogil=True)
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



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', 
            nopython=True, fastmath=True, nogil=True)
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
            float64[:,:], float64, float64[:], float64[:,:])', 
            nopython=True, fastmath=True, nogil=True)
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
            float64[:,:], float64[:,:,:])', 
            nopython=True, fastmath=True, nogil=True)
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



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', 
           nopython=True, fastmath=True, nogil=True)
def KLdiv(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computation of the KL-divergence (dissimilarity) upper bound between components 
    [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
    ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    return 0.5*((w1+w2)*np.log(_det(cov_m)) - w1*np.log(_det(cov1)) - w2*np.log(_det(cov2)))



#@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True, nogil=True)
#def isd_diss(w1, mu1, cov1, w2, mu2, cov2):
#    """
#    Computes the ISD (Integral Square Difference between components [(w1,mu1,cov1), (w2,mu2,cov2)])
#    and its moment preserving merge. Ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
#    """
#    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
#    Jhr = w1*w_m * normal(mu1, mu_m, cov1+cov_m) + w2*w_m * normal(mu2, mu_m, cov2+cov_m)
#    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov_m)))
#    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov1))) + \
#          (w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*cov2))) + \
#          2*w1*w2*normal(mu1, mu2, cov1+cov2)
#    return Jhh - 2*Jhr + Jrr

# @numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:])', nopython=True, nogil=True)
# def isd_diss_full(w, mu, sig):
#     # number of components
#     c = len(w)
#     # merged moment preserving gaussian
#     w_m, mu_m, sig_m = merge_full(w, mu, sig)
#     # ISD computation between merge and components
#     Jhr = 0.
#     Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
#     Jhh = 0.
#     for i in range(c):  
#         Jhr += w[i]*w_m * normal(mu[i], mu_m, sig[i]+sig_m)   
#     for i in range(c):
#         for j in range(c):
#             Jhh += w[i]*w[j] * normal(mu[i], mu[j], sig[i]+sig[j])
#     return Jhh - 2*Jhr + Jrr


@numba.jit('(float64[:,:], int32)')
def _compute_neighbors(mu_center, n_neighbors):
    BTree = BallTree(mu_center)
    n_samples = mu_center.shape[0]
    n_neighbors = min(n_neighbors+1, n_samples)
    nn_indexes = BTree.query(mu_center, 
                             k=n_neighbors, 
                             return_distance=False, 
                             sort_results=True)
    # first column removed, since correspond to the index of the same row
    nn_indexes = nn_indexes[:,1:]
    # removing pairs of repeated neighbors
    M,max_neigh = nn_indexes.shape
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if i>jj and np.any(i==nn_indexes[jj,:]):
                nn_indexes[i,j] = MAXINT
    return BTree,nn_indexes.astype(np.int32)   


@numba.jit('float64[:,:] (float64[:], float64[:,:], float64[:,:,:], int32[:,:])', 
           nopython=True, fastmath=True, nogil=True, parallel=True)
def build_diss_matrix(w, mu, cov, nn_indexes):
    M,max_neigh = nn_indexes.shape
    diss_matrix = np.empty((M,max_neigh),dtype=np.float64)
    diss_matrix[:,:] = np.inf
    for i in numba.prange(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: continue
            diss_matrix[i,j] = KLdiv(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
        sorted_indexes = np.argsort(diss_matrix[i,:])
        diss_matrix[i,:] = (diss_matrix[i,:])[sorted_indexes]
        nn_indexes[i,:] = (nn_indexes[i,:])[sorted_indexes]
    return diss_matrix


@numba.jit('float64[:,:] (float64[:], float64[:,:], float64[:,:,:])', 
           nopython=True, fastmath=True, nogil=True, parallel=True)
def _build_diss_matrix(w, mu, cov):
    M = len(w)
    diss_matrix = np.empty((M,M), dtype=np.float64)
    diss_matrix[:] = np.inf
    for i in numba.prange(M):
        for j in range(i+1,M):
            diss_matrix[i,j] = KLdiv(w[i],mu[i],cov[i],w[j],mu[j],cov[j])
    return diss_matrix


@numba.jit('Tuple((int32,int32)) (float64[:,:], int32[:], int32[:,:])', 
           nopython=True, fastmath=True)
def least_dissimilar(diss_matrix, indexes, nn_indexes):
    # number of mixture components still alive
    max_neigh = diss_matrix.shape[1]
    i_min = -1; j_min = -1
    diss_min = np.inf
    for _i in range(len(indexes)):
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


@numba.jit('Tuple((int32,int32)) (float64[:,:], int32[:])', 
           nopython=True, fastmath=True)
def _least_dissimilar(diss_matrix, indexes):
    # number of mixture components still alive
    i_min = -1; j_min = -1
    diss_min = np.inf
    for i in indexes:
        for j in indexes:
            if (i==j) or diss_matrix[i,j]==np.inf: continue
            if diss_matrix[i,j]<diss_min:
                diss_min = diss_matrix[i,j]
                i_min=i
                j_min=j
    return i_min,j_min


@numba.jit('int32 (int32[:], int32)', nopython=True, fastmath=True, nogil=True)
def get_index(array, value):
    n = len(array)
    for i in range(n):
        if array[i]==value: return i
    return -1


@numba.jit('(int32[:], int32, int32)', nopython=True, fastmath=True, nogil=True)
def update_merge_mapping(merge_mapping, nindex, dindex):
    n = len(merge_mapping)
    for i in range(n):
        if merge_mapping[i]==dindex:
            merge_mapping[i] = nindex


@numba.jit()
def radius_search(BTree, mu, n_neighbors, merge_mapping, nindex):
    # performing nn-search through BallTree
    n_samples = BTree.data.shape[0]
    k_query = min(2*n_neighbors, n_samples)
    neigh_array = BTree.query([mu], k=k_query, return_distance=False, sort_results=True)[0]
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


@numba.jit('(int32[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], int32[:], int32, int32)',
           nopython=True, fastmath=True, parallel=True)
def update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex):
    """
    Updates the nn_indexes and diss_matrix structs by removing the items
    corresponding to nindex and dindex
    """
    # number of mixture components still alive
    num_comp = len(indexes)
    max_neigh = nn_indexes.shape[1]
    for _i in numba.prange(num_comp):
        i = indexes[_i]
        if i==nindex: continue # this is an special case (see below)
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            if jj==nindex or jj==dindex:
                nn_indexes[i,j] = -1
                diss_matrix[i,j] = -1       

    # the special case...
    diss_matrix[nindex,:] = np.inf
    for j in numba.prange(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj==MAXINT: break
        diss_matrix[nindex,j] = KLdiv(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])

    sorted_indexes = np.argsort(diss_matrix[nindex,:])
    diss_matrix[nindex,:] = (diss_matrix[nindex,:])[sorted_indexes]
    nn_indexes[nindex,:] = (nn_indexes[nindex,:])[sorted_indexes]


@numba.jit('(float64[:,:], float64[:], float64[:,:], float64[:,:,:], int32[:], int32)', 
           nopython=True, fastmath=True, parallel=True)
def _update_structs(diss_matrix, w, mu, cov, indexes, nindex):
    # the dissimilarity between the added Gaussian and the alive Gaussians are re-computed
    for j in indexes:
        if j==nindex: 
            diss_matrix[nindex,j] = np.inf
            continue
        diss_matrix[nindex,j] = KLdiv(w[nindex],mu[nindex],cov[nindex],w[j],mu[j],cov[j])


def reduce_mixture():
    pass


def agglomerate():
    pass


def mixture_reduction(w, mu, cov, n_comp=1, n_neighbors=None,
                      verbose=True, build_htree=False):
    """
    Gaussian Mixture Reduction Through KL-upper bound approach
    """
    # current mixture size
    cur_mixture_size = len(w)
    # target mixture size
    tar_mixture_size = n_comp
    # dimensionality of data
    d = mu.shape[1]

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

    # indexes of "alive" mixture components
    indexes = np.arange(cur_mixture_size, dtype=np.int32)
    
    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(cur_mixture_size, dtype=np.int32)
    
    # BTree for efficient searches
    BTree,nn_indexes = _compute_neighbors(mu, n_neighbors)
    diss_matrix = build_diss_matrix(w, mu, cov, nn_indexes)

    # main mixture reduction loop
    while cur_mixture_size > tar_mixture_size:
        # approximated GMR for improved performance
        i_min,j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
        if i_min==-1:
            print(cur_mixture_size)
            print(diss_matrix[i_min])
            print(nn_indexes[i_min])
        w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                                 w[j_min], mu[j_min], cov[j_min])
        # updating structures
        nindex = min(i_min,j_min) # index of the new component
        dindex = max(i_min,j_min) # index of the del component
        w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
        indexes = np.delete(indexes, get_index(indexes,dindex))
        update_merge_mapping(merge_mapping, nindex, dindex)
        if cur_mixture_size <= n_neighbors+1:
            alive_neighbors = np.delete(indexes, get_index(indexes,nindex))
            nn_indexes[nindex,:] = MAXINT
            nn_indexes[nindex,0:len(alive_neighbors)] = alive_neighbors 
        else:
            nn_indexes[nindex] = radius_search(BTree, mu_m, n_neighbors, merge_mapping, nindex)
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
