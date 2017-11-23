import copy
import numba
import numpy as np



################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True)
def _outer(x, y):
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res

@numba.jit('float64 (float64[:,:])', nopython=True)
def _det2D(X):
    return X[0,0]*X[1,1] - X[0,1]*X[1,0]


@numba.jit('float64 (float64[:,:])', nopython=True)
def _det3D(X):
    return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
           X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
           X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])


@numba.jit('float64 (float64[:], float64[:], float64[:,:])', nopython=True)
def normal(x, mu, sig):
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * np.linalg.det(sig))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(sig), x-mu)))


def remove(l, indexes):
    """
    Remove elements from list
    yielding a new list
    """
    return [val for i,val in enumerate(l) if i not in indexes]


def ncomp_finder(kl_hist, w_size=10):
    """
    Heuristic: If the actual diff is 1 order of magnitude
    greater than the mean of the 10 last diff values, we 
    consider this points as the number of components estimate
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


#################################################################
# MOMENT PRESERVING GAUSSIAN
#################################################################

@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(c1, mu1, sig1, c2, mu2, sig2):
    c_m = c1+c2
    mu_m = (c1/c_m)*mu1 + (c2/c_m)*mu2
    sig_m = (c1/c_m)*sig1 + (c2/c_m)*sig2 + (c1/c_m)*(c2/c_m)*_outer(mu1-mu2, mu1-mu2)
    return (c_m, mu_m, sig_m)


@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64[:], \
            float64[:,:], float64[:,:,:])', nopython=True)
def merge_full(w, mu, sig):
    n = mu.shape[0]
    d = mu.shape[1]
    w_m = np.sum(w)
    mu_m = np.zeros(d)
    sig_m = np.zeros((d,d))

    #mean calculation
    for i in range(n):
        mu_m += w[i]*mu[i]
    mu_m /= w_m

    #covariance calculation
    for i in range(n):
        sig_m += w[i] * ( sig[i] + _outer(mu[i]-mu_m, mu[i]-mu_m) )
    sig_m /= w_m

    return (w_m, mu_m, sig_m)



###########################################################################
# ISD: Integral Square Difference
# ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
###########################################################################
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss(w1, mu1, sig1, w2, mu2, sig2):
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(w1, mu1, sig1, w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = w1*w_m * normal(mu1, mu_m, sig1+sig_m) + w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*w1*w2*normal(mu1, mu2, sig1+sig2)
    return Jhh - 2*Jhr + Jrr


#normalized version
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss_(w1, mu1, sig1, w2, mu2, sig2):
    _w1 = w1 / (w1 + w2)
    _w2 = w2 / (w1 + w2)
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(_w1, mu1, sig1, _w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = _w1*w_m * normal(mu1, mu_m, sig1+sig_m) + _w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (_w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (_w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*_w1*_w2*normal(mu1, mu2, sig1+sig2)
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


@numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:], \
                     float64[:], float64[:,:], float64[:,:,:])', nopython=True)
def isd_diss_full_(w1, mu1, sig1, w2, mu2, sig2):
    # number of components
    h = len(w1)
    r = len(w2)

    # ISD computation between merge and components
    Jhr = 0.
    Jrr = 0.
    Jhh = 0.

    for i in range(h):
        for j in range(r):
            Jhr += w1[i]*w2[j] * normal(mu1[i], mu2[j], sig1[i]+sig2[j])

    for i in range(r):
        for j in range(r):
            Jrr += w2[i]*w2[j] * normal(mu2[i], mu2[j], sig2[i]+sig2[j])
    
    for i in range(h):
        for j in range(h):
            Jhh += w1[i]*w1[j] * normal(mu1[i], mu1[j], sig1[i]+sig1[j])

    return Jhh - 2*Jhr + Jrr



#################################################################
# KL-DIVERGENCE UPPER BOUND
# ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
#################################################################
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def kl_diss(c1, mu1, sig1, c2, mu2, sig2):
    # merged moment preserving gaussian
    c_m, mu_m, sig_m = merge(c1, mu1, sig1, c2, mu2, sig2)
    # KL divergence upper bound as proposed in: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    if len(sig_m)==2:
        return 0.5*((c1+c2)*np.log(_det2D(sig_m)) - c1*np.log(_det2D(sig1)) - c2*np.log(_det2D(sig2)))
    else:
        return 0.5*((c1+c2)*np.log(_det3D(sig_m)) - c1*np.log(_det3D(sig1)) - c2*np.log(_det3D(sig2)))


#################################################################
# MAIN GAUSSIAN REDUCTION FUNCTION
#################################################################

def gaussian_reduction(c, mu, sig, n_comp, metric='KL', verbose=True):
    if metric=='KL': 
        _metric = kl_diss
        isd_hist = list(); kl_hist = list()
    elif metric=='ISD': 
        _metric = isd_diss
        isd_hist = list(); kl_hist = None
    elif metric=='ISD_':
        _metric = isd_diss_
        isd_hist = list(); kl_hist = None
    else: return None

    d = mu.shape[1]
    c = c.tolist()
    mu = list(map(np.array, mu.tolist()))
    if d==2: sig = [(s**2)*np.identity(2) for s in sig]
    elif d==3: sig = [(s**2)*np.identity(3) for s in sig]

    # indexes of the actual gaussian components
    components = [i for i in range(len(c))]
    components_dict = {i:[i] for i in range(len(c))}
    htree = {}
    new_comp = len(c)

    # main loop
    while len(components)>n_comp:
        m = len(components)
        diss_min = np.inf
        for i in range(m):
            ii = components[i]
            for j in range(i+1,m):
                jj = components[j]
                diss = _metric(c[ii], mu[ii], sig[ii], c[jj], mu[jj], sig[jj])
                if diss < diss_min: 
                    i_min = i; j_min = j
                    ii_min = ii; jj_min = jj 
                    diss_min = diss
        # compute the moment preserving  merged gaussian
        c_m, mu_m, sig_m = merge(c[ii_min], mu[ii_min], sig[ii_min], 
                                 c[jj_min], mu[jj_min], sig[jj_min])
        
        if (metric=='ISD' or metric=='ISD_') and verbose:
            print('Merged components {0} and {1} with {2} ISD dist'.format(ii_min, jj_min, diss_min))
            isd_hist.append(diss_min)    
        elif metric=='KL' and verbose:
            ISD_diss = isd_diss(c[ii_min], mu[ii_min], sig[ii_min], c[jj_min], mu[jj_min], sig[jj_min])
            print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(ii_min, jj_min, diss_min, ISD_diss))
            isd_hist.append(ISD_diss), kl_hist.append(diss_min)

        # updating structures   
        del components[max(i_min, j_min)]
        del components[min(i_min, j_min)]
        components.append(new_comp)
        c.append(c_m); mu.append(mu_m); sig.append(sig_m)
        htree[new_comp] = (min(ii_min,jj_min), max(ii_min,jj_min))
        tmp = components_dict[min(ii_min,jj_min)] + components_dict[max(ii_min,jj_min)]
        tmp.sort()
        components_dict[new_comp] = tmp
        new_comp += 1
    return (components_dict,htree)

def _gaussian_reduction_(c, mu, sig, n_comp, metric='KL', verbose=True):
    if metric=='KL': 
        _metric = kl_diss
        isd_hist = list(); kl_hist = list()
    elif metric=='ISD': 
        _metric = isd_diss
        isd_hist = list(); kl_hist = None
    elif metric=='ISD_':
        _metric = isd_diss_
        isd_hist = list(); kl_hist = None
    else: return None

    d = mu.shape[1]
    c = c.tolist()
    mu = list(map(np.array, mu.tolist()))
    if d==2: sig = [(s**2)*np.identity(2) for s in sig]
    elif d==3: sig = [(s**2)*np.identity(3) for s in sig]
    # indexes of the actual gaussian components
    components = [[i] for i in range(len(c))]
    components_dict = {len(components) : copy.deepcopy(components)}

    # main loop
    while len(components)>n_comp:
        m = len(c)
        diss_min = np.inf
        i_min = -1; j_min = -1
        for i in range(m):
            for j in range(i+1,m):
                diss = _metric(c[i], mu[i], sig[i], c[j], mu[j], sig[j])
                if diss < diss_min: i_min = i; j_min = j; diss_min = diss
        # compute the moment preserving  merged gaussian
        c_m, mu_m, sig_m = merge(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
        # updating structures
        if (metric=='ISD' or metric=='ISD_') and verbose:
            print('Merged components {0} and {1} with {2} ISD dist'.format(i_min, j_min, diss_min))
            isd_hist.append(diss_min)    
        elif metric=='KL' and verbose:
            ISD_diss = isd_diss(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
            print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(i_min, j_min, diss_min, ISD_diss))
            isd_hist.append(ISD_diss), kl_hist.append(diss_min)
        del c[max(i_min, j_min)]; del c[min(i_min, j_min)]; c.append(c_m)
        del mu[max(i_min, j_min)]; del mu[min(i_min, j_min)]; mu.append(mu_m)
        del sig[max(i_min, j_min)]; del sig[min(i_min, j_min)]; sig.append(sig_m)
        new_component = components.pop(max(i_min,j_min)) + components.pop(min(i_min,j_min))
        new_component.sort()
        components.append(new_component)
        components_dict[m-1] = copy.deepcopy(components)
    return components_dict, np.array(isd_hist), np.array(kl_hist)


def gaussian_reduction_(c, mu, _sig, n_comp, metric='KL', verbose=True):
    """
    Maray's approach for Gaussian reduction with ISD
    """

    # putting it in the correct format
    sig = np.zeros((len(c),2,2))
    sig[:,0,0] = _sig**2; sig[:,1,1] = _sig**2

    # indexes of the actual gaussian components
    components = [[i] for i in range(len(c))]
    components_dict = {len(components) : copy.deepcopy(components)}
    isd_hist = list()

    # main loop
    while len(components)>n_comp:
        m = len(components)
        diss_min = np.inf
        i_min = -1; j_min = -1
        for i in range(m):
            for j in range(i+1,m):
                indexes_i = components[i]
                indexes_j = components[j]
                c_ij = np.hstack( [c[indexes_i], c[indexes_j]] )
                mu_ij = np.vstack( [mu[indexes_i], mu[indexes_j]] )
                sig_ij = np.vstack( [sig[indexes_i], sig[indexes_j]] )
                diss = isd_diss_full(c_ij, mu_ij, sig_ij)
                if diss < diss_min: i_min = i; j_min = j; diss_min = diss

        print('Merged components {0} and {1} with {2} ISD dist'.format(i_min, j_min, diss_min))
        isd_hist.append(diss_min)    

        new_component = components.pop(max(i_min,j_min)) + components.pop(min(i_min,j_min))
        new_component.sort()
        components.append(new_component)
        components_dict[m-1] = copy.deepcopy(components)
    return components_dict, isd_hist


def gaussian_reduction__(c, mu, _sig, n_comp, metric='KL', verbose=True):
    # putting it in the correct format
    sig = np.zeros((len(c),2,2))
    sig[:,0,0] = _sig**2; sig[:,1,1] = _sig**2

    d = mu.shape[1]
    c = c.tolist()
    mu = list(map(np.array, mu.tolist()))
    sig = list(map(np.array, sig.tolist()))

    # indexes of the actual gaussian components
    components = [[i] for i in range(len(c))]
    components_dict = {len(components) : copy.deepcopy(components)}
    isd_hist = list()

    # main loop
    while len(components)>n_comp:
        m = len(c)
        diss_min = np.inf
        i_min = -1; j_min = -1
        for i in range(m):
            for j in range(i+1,m):
                c_m, mu_m, sig_m = merge(c[i], mu[i], sig[i], c[j], mu[j], sig[j])
                _c = remove(c, [i,j]); _c.append(c_m)
                _mu = remove(mu, [i,j]); _mu.append(mu_m)
                _sig = remove(sig, [i,j]); _sig.append(sig_m)
                diss = isd_diss_full_(np.array(c), np.array(mu), np.array(sig), 
                                     np.array(_c), np.array(_mu), np.array(_sig))
                if diss < diss_min: i_min = i; j_min = j; diss_min = diss
        # compute the moment preserving  merged gaussian
        c_m, mu_m, sig_m = merge(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
        print('Merged components {0} and {1} with {2} ISD dist'.format(i_min, j_min, diss_min))
        isd_hist.append(diss_min)    

        # updating structures
        del c[max(i_min, j_min)]; del c[min(i_min, j_min)]; c.append(c_m)
        del mu[max(i_min, j_min)]; del mu[min(i_min, j_min)]; mu.append(mu_m)
        del sig[max(i_min, j_min)]; del sig[min(i_min, j_min)]; sig.append(sig_m)
        new_component = components.pop(max(i_min,j_min)) + components.pop(min(i_min,j_min))
        new_component.sort()
        components.append(new_component)
        components_dict[m-1] = copy.deepcopy(components)
    return components_dict, np.array(isd_hist)




