import sys
import warnings
import numba
import scipy
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import NearestNeighbors
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)




@numba.jit(nopython=True)
def gm_eval(c, sig, xc, yc, xe, ye):    
    """
    Fast evaluation of the Gaussian Mixture

    w     : 1D np.ndarray - weight values
    sig   : 1D np.ndarray - sigma values
    xc    : 1D np.ndarray - x coordinate of center points
    yc    : 1D np.ndarray - y coordinate of center points
    xe    : 1D np.ndarray - x coordinate of eval points
    ye    : 1D np.ndarray - y coordinate of eval points
    """
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            ret[i] += c[j] * exp( -0.5 * dist2 / sig[j]**2 )
    return ret


@numba.jit(nopython=True)
def gm_eval_fast(w, sig, xc, yc, xe, ye, 
                neigh_indexes, neigh_indexes_aux):
    """
    Fast evaluation of the Gaussian Mixture

    w     : 1D np.ndarray - weight values
    sig   : 1D np.ndarray - sigma values
    xc    : 1D np.ndarray - x coordinate of center points
    yc    : 1D np.ndarray - y coordinate of center points
    xe    : 1D np.ndarray - x coordinate of eval points
    ye    : 1D np.ndarray - y coordinate of eval points
    neigh_indexes : 1D np.ndarray - indexes of neighbors of all eval points
    neigh_indexes_aux : 1D np.ndarray - limit indices for neighbors of eval points in neigh_indexes
    """
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    sind = 0 # start index
    for i in range(m):
        eind = neigh_indexes_aux[i] # end index
        for j in neigh_indexes[sind:eind]:
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            ret[i] += w[j] * exp( -0.5 * dist2 / sig[j]**2 )
        sind = eind
    return ret


@numba.jit(nopython=True)
def grad_eval(c, sig, xc, yc, xe, ye, support=5):
    m = len(xe)
    n = len(xc)
    ret = np.zeros((3,m))
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            # common terms
            coef = (-1./(sig[j]**2))
            # u evaluation
            aux = c[j] * exp( -dist2 / (2* sig[j]**2 ) )
            ret[0,i] += aux
            # ux evaluation
            ret[1,i] += coef * aux * (xe[i]-xc[j])
            # uy evaluation
            ret[2,i] += coef * aux * (ye[i]-yc[j])
    return np.sqrt(ret[1,:]**2 + ret[2,:]**2)


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


def snr_estimation(data, noise=None, points=1000, full_output=False):
    """
    Heurustic that uses the inflexion point of the thresholded RMS to estimate where signal is dominant w.r.t. noise
    
    Parameters
    ---------- 
    data : (M,N,Z) numpy.ndarray or numpy.ma.MaskedArray


    noise : float (default=None)
        Noise level, if not given will use rms of the data.
    
    points : (default=1000)

    full_output : boolean (default=False)
        Gives verbose results if True

    Returns
    --------

    "Signal to Noise Radio" value
    
    """
    if noise is None:
        noise = estimate_rms(data)
    x = []
    y = []
    n = []
    sdata = data[data > noise]
    for i in range(1, int(points)):
        val = 1.0 + 2.0 * i / points
        sdata = sdata[sdata > val * noise]
        if sdata.size < 2:
            break
        n.append(sdata.size)
        yval = sdata.mean() / noise
        x.append(val)
        y.append(yval)
    y = np.array(y)
    v = y[1:] - y[0:-1]
    p = v.argmax() + 1
    snrlimit = x[p]
    if full_output == True:
        return snrlimit, noise, x, y, v, n, p
    return snrlimit


def build_dist_matrix(points, inf=False):
    """
    Builds a distance matrix from points array.
    It returns a (n_points, n_points) distance matrix. 

    points: NumPy array with shape (n_points, 2) 
    """
    m,n = points.shape
    D = np.empty((m,m))
    for i in range(m):
        for j in range(m):
            if inf and i==j: 
                D[i,j] = np.inf
                continue 
            D[i,j] = np.linalg.norm(points[i]-points[j], ord=2)
    return D

        
def load_data(fits_path):
    hdu = fits.open(fits_path)[0]
    data = hdu.data
    wcs = WCS(hdu.header)

    if data.ndim>3:
        # droping out the stokes dimension
        data = np.ascontiguousarray(data[0])
        wcs = wcs.dropaxis(3)

        if data.shape[0]==1:
            # in case data is not a cube but an image
            data = np.ascontiguousarray(data[0])
            wcs = wcs.dropaxis(2)
    
    # in case NaN values exist on data
    mask = np.isnan(data)
    if np.any(mask): data = ma.masked_array(data, mask=mask)

    return data,wcs,hdu


def sig_mapping(sig, minsig=0., maxsig=1.):
    return np.sqrt( (maxsig**2-minsig**2)*np.tanh(sig**2) + minsig**2 )


def _inv_tanh(x):
    return 0.5*np.log((1+x)/(1-x))


def inv_sig_mapping(sig, minsig=0., maxsig=1.):
    return np.sqrt( _inv_tanh((sig**2-minsig**2)/(maxsig**2-minsig**2)) )


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


def gradient(img):
    gx, gy = np.gradient(img)
    img_grad = np.sqrt(gx**2 + gy**2)
    return img_grad


def compute_neighbors(mu_center, mu_eval, maxsig):
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=2)
    nn.fit(mu_center)
    neigh_indexes_arr = nn.radius_neighbors(mu_eval, return_distance=False)
    neigh_indexes = []
    neigh_indexes_aux = []
    last = 0
    for arr in neigh_indexes_arr:
        l = arr.tolist(); l.sort()
        last += len(l)
        for index in l: neigh_indexes.append(index)
        neigh_indexes_aux.append(last)
    return np.asarray(neigh_indexes),np.asarray(neigh_indexes_aux)  


def compute_neighbors2(mu_center, mu_eval, maxsig):
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=2)
    nn.fit(mu_center)
    neigh_indexes_arr = nn.radius_neighbors(mu_eval, return_distance=False)
    
    # creating the initial array
    maxlen = 0
    for arr in neigh_indexes_arr:
        if len(arr)>maxlen:
            maxlen = len(arr)
    neigh_indexes = -1*np.ones((len(neigh_indexes_arr),maxlen), dtype=np.int64)
    
    # filling it with the correct indexes
    for i,arr in enumerate(neigh_indexes_arr):
        ll = arr.tolist(); ll.sort()
        for j,index in enumerate(ll):
            neigh_indexes[i,j] = index
            
    return neigh_indexes
