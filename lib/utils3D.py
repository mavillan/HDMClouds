import sys
import numba
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits


@numba.jit('float64[:] (float64[:], float64[:], float64[:], float64[:], float64[:], \
           float64[:], float64[:], float64[:], float64)', nopython=True, nogil=True)
def u_eval(c, sig, xc, yc, zc, xe, ye, ze, support=5.):
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2 + (ze[i]-zc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            ret[i] += c[j] * exp( -dist2 / (2* sig[j]**2) )
    return ret



# @numba.jit('void (float64[:], float64[:], float64[:,:], float64[:,:], int64, int64, \
#            float64[:], float64, float64)', nopython=True, nogil=True)
# def u_eval(c, sig, epoints, cpoints, start, chunk_size, ret, supp=5., sig0=0.001):
#     m = len(epoints)
#     n = len(cpoints)
#     for i in range(start, min(start+chunk_size, m)):
#         for j in range(n):
#             dist2 = np.sum((epoints[i] - cpoints[i])**2)
#             if  dist2 > supp**2 * sig[j]**2: continue
#             ret[i] += c[j] * exp( -dist2 / (2* (sig0**2 + sig[j]**2)) )

# def thread_eval(c, sig, epoints, cpoints, n_thread=2):
#     chunk_size = len(epoints)/n_thread
    
#     ret = np.zeros(len(epoints))
    
#     start_index = [i*chunk_size for i in range(n_thread)]
    
#     threads = [Thread(target=u_eval, args=(c, sig, epoints, cpoints, start, chunk_size, ret, 5, 0.001)) for start in start_index]
    
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
    
#     return ret



def compute_solution(c, sig, xc, yc, zc, dims, base_level=0., support=5.):
    _xe = np.linspace(0., 1., dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., dims[1]+2)[1:-1]
    _ze = np.linspace(0., 1., dims[2]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye); len_ze = len(_ze)
    Xe,Ye,Ze = np.meshgrid(_xe, _ye, _ze, indexing='ij', sparse=False)
    xe = Xe.ravel(); ye = Ye.ravel(); ze = Ze.ravel()
    
    u = u_eval(c, sig, xc, yc, zc, xe, ye, ze, support=support) + base_level
    return u.reshape(len_xe, len_ye, len_ze)



def estimate_rms(data):
    """
    Computes RMS value of an N-dimensional numpy array
    """

    if isinstance(data, ma.MaskedArray):
        ret = np.sum(data*data) / (np.size(data) - np.sum(data.mask)) 
    else: 
        ret = np.sum(data*data) / np.size(data)
    return np.sqrt(ret)



def estimate_entropy(data):
    """
    Computes Entropy of an N-dimensional numpy array
    """

    # estimation of probabilities
    p = np.histogram(data.ravel(), bins=256, density=False)[0].astype(float)
    # little fix for freq=0 cases
    p = (p+1.)/(p.sum()+256.)
    # computation of entropy 
    return -np.sum(p * np.log2(p))



def estimate_variance(data):
    """
    Computes variance of an N-dimensional numpy array
    """

    return np.std(data)**2



def build_dist_matrix(points):
    """
    Builds a distance matrix from points array.
    It returns a (n_points, n_points) distance matrix. 

    points: NumPy array with shape (n_points, 2) 
    """
    xp = points[:,0]
    yp = points[:,1]
    zp = points[:,2]
    N = points.shape[0]
    Dx = np.empty((N,N))
    Dy = np.empty((N,N))
    Dz = np.empty((N,N))
    for k in range(N):
        Dx[k,:] = xp[k]-xp
        Dy[k,:] = yp[k]-yp
        Dz[k,:] = zp[k]-zp
    return np.sqrt(Dx**2+Dy**2+Dz**2)



def logistic(x):
    return 1. / (1. + np.exp(-x))



def logit(x):
    mask0 = x==0.
    mask1 = x==1.
    mask01 = np.logical_and(~mask0, ~mask1)
    res = np.empty(x.shape[0])
    res[mask0] = -np.inf
    res[mask1] = np.inf
    res[mask01] = np.log(x[mask01] / (1-x[mask01]))
    return res



def load_data(fits_path):
    hdulist = fits.open(fits_path)
    data = hdulist[0].data
    # droping out the stokes dimension
    data = np.ascontiguousarray(data[0])
    
    # in case NaN values exist on cube
    mask = np.isnan(data)
    if np.any(mask): data = ma.masked_array(data, mask=mask)

    # map to 0-1 intensity range
    data -= data.min()
    data /= data.max()
    
    if data.shape[0]==1:
        data = np.ascontiguousarray(data[0])
        if np.any(mask): 
            mask = np.ascontiguousarray(mask[0])
            data = ma.masked_array(data, mask=mask)
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x,y), data, method='linear', bounds_error=False, fill_value=0.)
        return x,y,data,dfunc

    else:
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        z = np.linspace(0., 1., data.shape[2]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x, y, z), data, method='linear', bounds_error=False, fill_value=0.)
        return x,y,z,data,dfunc


def logistic(x):
    return 1. / (1. + np.exp(-x))


def logit(x):
    mask0 = x==0.
    mask1 = x==1.
    mask01 = np.logical_and(~mask0, ~mask1)
    res = np.empty(x.shape[0])
    res[mask0] = -np.inf
    res[mask1] = np.inf
    res[mask01] = np.log(x[mask01] / (1-x[mask01]))
    return res


def mean_min_dist(points1, points2):
    x1 = points1[:,0]; y1 = points1[:,1]
    x2 = points2[:,0]; y2 = points2[:,1]
    M = points1.shape[0]
    N = points2.shape[0]
    Dx = np.empty((M,N))
    Dy = np.empty((M,N))
    for k in range(M):
        Dx[k] = x1[k] - x2
        Dy[k] = y1[k] - y2
    D = np.sqrt(Dx**2 + Dy**2)
    return np.mean( np.min(D, axis=1) )


def prune(vec):
    mean = np.mean(vec)
    median = np.median(vec)
    #all values greater than 1e-3 the mean/median
    mask = vec > 1e-3*min(mean,median)
    return mask, vec[mask]


def sig_mapping(sig, minsig=0., maxsig=1.):
    return np.sqrt( (maxsig**2-minsig**2)*np.tanh(sig**2) + minsig**2 )


def _inv_tanh(x):
    return 0.5*np.log((1+x)/(1-x))


def inv_sig_mapping(sig, minsig=0., maxsig=1.):
    return np.sqrt( _inv_tanh((sig**2-minsig**2)/(maxsig**2-minsig**2)) )
