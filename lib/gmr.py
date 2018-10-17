import copy
import numba
import numpy as np
from sklearn.neighbors import NearestNeighbors

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max

################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True, nogil=True, cache=True)
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


@numba.jit('float64 (float64[:,:])', nopython=True, nogil=True, cache=True)
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



@numba.jit('float64 (float64[:], float64[:], float64[:,:])', nopython=True, nogil=True, cache=True)
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
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True, nogil=True, cache=True)
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
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True, nogil=True, cache=True)
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
            float64[:,:], float64[:,:,:])', nopython=True, nogil=True, cache=True)
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



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True, nogil=True, cache=True)
def KLdiv(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computation of the KL-divergence (dissimilarity) upper bound between components 
    [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
    ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    return 0.5*((w1+w2)*np.log(_det(cov_m)) - w1*np.log(_det(cov1)) - w2*np.log(_det(cov2)))



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True, nogil=True, cache=True)
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

# @numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:])', nopython=True, nogil=True, cache=True)
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


@numba.jit('float64[:,:] (float64[:], float64[:,:], float64[:,:,:], int32[:,:])', nopython=True, nogil=True, parallel=True)
def build_diss_matrix_nn(w, mu, cov, nn_indexes):
    M,max_neigh = nn_indexes.shape
    diss_matrix = np.inf*np.ones((M,max_neigh))
    for i in numba.prange(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            diss_matrix[i,j] = KLdiv(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
        sorted_indexes = np.argsort(diss_matrix[i,:])
        diss_matrix[i,:] = (diss_matrix[i,:])[sorted_indexes]
        nn_indexes[i,:] = (nn_indexes[i,:])[sorted_indexes]
    return diss_matrix


@numba.jit('float64[:,:] (float64[:], float64[:,:], float64[:,:,:])', nopython=True, nogil=True, parallel=True)
def build_diss_matrix_full(w, mu, cov):
    M = len(w)
    diss_matrix = np.inf*np.ones((M,M))
    for i in numba.prange(M):
        for j in range(i+1,M):
            diss_matrix[i,j] = KLdiv(w[i],mu[i],cov[i],w[j],mu[j],cov[j])
    return diss_matrix


@numba.jit('Tuple((int32,int32)) (float64[:,:], int32[:], int32[:,:])', nopython=True, cache=True)
def least_dissimilar_nn(diss_matrix, indexes, nn_indexes):
    # number of mixture components still alive
    num_comp = indexes.shape[0]
    max_neigh = diss_matrix.shape[1]
    i_min = -1; j_min = -1
    diss_min = np.inf
    for _i in range(num_comp):
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


@numba.jit('Tuple((int32,int32)) (float64[:,:], int32[:])', nopython=True, cache=True)
def least_dissimilar_full(diss_matrix, indexes):
    # number of mixture components still alive
    i_min = -1; j_min = -1
    diss_min = np.inf
    for i in indexes:
        for j in indexes:
            if (j<=i) or (diss_matrix[i,j]==np.inf): continue
            if diss_matrix[i,j]<diss_min:
                diss_min = diss_matrix[i,j]
                i_min = i; j_min=j
    return i_min,j_min


@numba.jit('int32 (int32[:], int32)', nopython=True, nogil=True, cache=True)
def get_index(array, value):
    n = len(array)
    for i in range(n):
        if array[i]==value: return i
    return -1


@numba.jit('(int32[:], int32, int32)', nopython=True, nogil=True, cache=True)
def update_merge_mapping(merge_mapping, nindex, dindex):
    n = len(merge_mapping)
    for i in range(n):
        if merge_mapping[i]==dindex:
            merge_mapping[i] = nindex


@numba.jit()
def radius_search(nn, mu, max_neigh, merge_mapping, nindex, rho):
    if rho is not None:
        nn.radius *= rho
    neigh_array = nn.radius_neighbors([mu], return_distance=False)[0]
    neigh_array = merge_mapping[neigh_array]
    neigh_array = np.unique(neigh_array)
    # just in case...
    if len(neigh_array)>max_neigh:
        neigh_array = nn.kneighbors([mu], n_neighbors=max_neigh, return_distance=False)[0]
        neigh_array = merge_mapping[neigh_array]
        neigh_array = np.unique(neigh_array)
    # removing nindex from neighbors
    neigh_array = np.delete(neigh_array, get_index(neigh_array,nindex))
    ret = MAXINT*np.ones(max_neigh, dtype=np.int32)
    ret[0:len(neigh_array)] = neigh_array
    return ret


@numba.jit('(int32[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], int32[:], int32, int32)', nopython=True, parallel=True)
def update_structs_nn(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex):
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
    for j in numba.prange(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj!=MAXINT:
            diss_matrix[nindex,j] = KLdiv(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])
        else:
            diss_matrix[nindex,j] = np.inf

    sorted_indexes = np.argsort(diss_matrix[nindex,:])
    diss_matrix[nindex,:] = (diss_matrix[nindex,:])[sorted_indexes]
    nn_indexes[nindex,:] = (nn_indexes[nindex,:])[sorted_indexes]


@numba.jit('(float64[:,:], float64[:], float64[:,:], float64[:,:,:], int32[:], int32)', nopython=True, parallel=True)
def update_structs_full(diss_matrix, w, mu, cov, indexes, nindex):
    # the dissimilarity between the added Gaussian and the alive Gaussians are re-computed
    for j in indexes:
        if j==nindex: continue
        # the next is done to respect the diss_matrix structure:
        # only the KLdiv between minor_index -> major_index is present
        if nindex<j:
            diss_matrix[nindex,j] = KLdiv(w[nindex],mu[nindex],cov[nindex],w[j],mu[j],cov[j])
        elif nindex>j:
            diss_matrix[j,nindex] = KLdiv(w[j],mu[j],cov[j],w[nindex],mu[nindex],cov[nindex])



################################################################
# MAIN FUNCTION
################################################################

# def mixture_reduction(w, mu, cov, n_comp, k_sig=2, verbose=True):
#     """
#     Gaussian Mixture Reduction Through KL-upper bound approach
#     """
#     w = np.copy(w)
#     mu = np.copy(mu)
#     cov = np.copy(cov)

#     # original size of the mixture
#     M = len(w) 
#     # target size of the mixture
#     N = n_comp
#     # dimensionality of data
#     d = mu.shape[1]

#     # we consider neighbors at a radius equivalent to the lenght of k_sigma*sigma_max
#     if cov.ndim==1:
#         maxsig = k_sig*np.max(cov)
#         # if cov is 1-dimensional we convert it to its covariance matrix form
#         cov = np.asarray( [(sig**2)*np.identity(d) for sig in cov] )
#     else:
#         maxsig = k_sig*max([np.max(np.linalg.eig(_cov)[0])**(1./2) for _cov in cov])

#     indexes = np.arange(M, dtype=np.int32)
#     nn,nn_indexes = _compute_neighbors(mu,maxsig)

#     # idea: keep track that the k-th component was merged into the l-th positon
#     merge_mapping = np.arange(M, dtype=np.int32)

#     # max number of neighbors
#     max_neigh = nn_indexes.shape[1]
    
#     # computing the initial dissimilarity matrix
#     diss_matrix = build_diss_matrix(w, mu, cov, nn_indexes)
    
#     # main loop
#     while M>N:
#         i_min,j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
#         if verbose: print('Merged components {0} and {1}'.format(i_min, j_min))  
#         w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
#                                  w[j_min], mu[j_min], cov[j_min])
 
#         # updating structures
#         nindex = min(i_min,j_min) # index of the new component
#         dindex = max(i_min,j_min) # index of the del component
#         w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
#         indexes = np.delete(indexes, get_index(indexes,dindex))
#         update_merge_mapping(merge_mapping, nindex, dindex)
#         nn_indexes[nindex] = radius_search(nn, mu_m, max_neigh, merge_mapping, nindex)
#         update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)
#         M -= 1

#     # indexes of "alive" mixture components
#     return w[indexes],mu[indexes],cov[indexes]








def mixture_reduction(w, mu, cov, n_comp=1, break_point=100, k_sig=2, 
    verbose=True, build_htree=False, adaptive_maxsig=False):
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

    if build_htree:
        # hierarchical tracking data structures
        decomp_dict = dict()
        join_dict = dict()
        entity_dict = {i:[i] for i in range(M)}
        # the below dict maps the indexes of the current GM components,
        # to the indexes of the current cloud entities
        entity_key_mapping = {i:i for i in range(M)}
        # label for the next entity to be added
        new_entity = M 

    # we consider neighbors at a radius equivalent to the lenght of k_sigma*sigma_max
    if cov.ndim==1:
        maxsig = k_sig*np.max(cov)
        # if cov is 1-dimensional we convert it to its covariance matrix form
        cov = np.asarray( [(sig**2)*np.identity(d) for sig in cov] )
    else:
        maxsig = k_sig*max([np.max(np.linalg.eig(_cov)[0])**(1./2) for _cov in cov])

    indexes = np.arange(M, dtype=np.int32)
    nn,nn_indexes = _compute_neighbors(mu,maxsig)
    if adaptive_maxsig:
        if d==2: rho = (np.sqrt(2)/maxsig)**(1./(M-1))
        if d==3: rho = (np.sqrt(3)/maxsig)**(1./(M-1))
    else: rho = None

    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(M, dtype=np.int32)

    # max number of neighbors
    max_neigh = nn_indexes.shape[1]
    
    # computing the initial dissimilarity matrix
    diss_matrix= build_diss_matrix_nn(w, mu, cov, nn_indexes)
    
    # main loop
    while M>N:
        if M==break_point:
            del diss_matrix
            diss_matrix = build_diss_matrix_full(w, mu, cov)
        if M<=break_point:
            # full GMR for improved accuracy
            i_min,j_min = least_dissimilar_full(diss_matrix, indexes)
            w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                                 w[j_min], mu[j_min], cov[j_min])
            # updating structures
            nindex = min(i_min,j_min) # index of the new component
            dindex = max(i_min,j_min) # index of the del component
            w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
            indexes = np.delete(indexes, get_index(indexes,dindex))
            update_structs_full(diss_matrix, w, mu, cov, indexes, nindex)
        else:
            # approximated GMR for improved performance
            i_min,j_min = least_dissimilar_nn(diss_matrix, indexes, nn_indexes)
            w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                                 w[j_min], mu[j_min], cov[j_min])
            # updating structures
            nindex = min(i_min,j_min) # index of the new component
            dindex = max(i_min,j_min) # index of the del component
            w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
            indexes = np.delete(indexes, get_index(indexes,dindex))
            update_merge_mapping(merge_mapping, nindex, dindex)
            nn_indexes[nindex] = radius_search(nn, mu_m, max_neigh, merge_mapping, nindex, rho)
            update_structs_nn(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)

        M -= 1
        if verbose: print('{2}: Merged components {0} and {1}'.format(i_min, j_min, M)) 

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

    # 
    #inv = {kmax-i:i for i in range(kmax+1)}
    #_decomp_dict = dict()
    #_join_dict = dict()
    #_entity_dict = dict()
    #for key in decomp_dict: 
    #    _decomp_dict[inv[key]]
    if not build_htree: 
        return w[indexes],mu[indexes],cov[indexes]
    else: 
        return decomp_dict,join_dict,entity_dict
