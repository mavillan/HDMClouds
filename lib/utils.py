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

# max int32 value
ii32 = np.iinfo(np.int32)
INAN = ii32.max


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
    return  (maxsig-minsig)*((np.tanh(sig)+1)/2) + minsig

def theta_mapping(theta, theta_lim=np.pi):
    return theta_lim*np.tanh(theta)

def inv_tanh(x):
    return 0.5*np.log((1+x)/(1-x))

def inv_sig_mapping(sig, minsig=0., maxsig=1.):
    return inv_tanh( (2*(sig-minsig)/(maxsig-minsig))-1 )

def inv_theta_mapping(theta, theta_lim=np.pi):
    return inv_tanh(theta/theta_lim)


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
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
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
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
    nn.fit(mu_center)
    neigh_indexes_arr = nn.radius_neighbors(mu_eval, return_distance=False)
    
    # creating the initial array
    maxlen = 0
    for arr in neigh_indexes_arr:
        if len(arr)>maxlen:
            maxlen = len(arr)
    neigh_indexes = INAN*np.ones((len(neigh_indexes_arr),maxlen), dtype=np.int32)
    
    # filling it with the correct indexes
    for i,arr in enumerate(neigh_indexes_arr):
        ll = arr.tolist(); ll.sort()
        for j,index in enumerate(ll):
            neigh_indexes[i,j] = index
            
    return nn,neigh_indexes


@numba.jit('float64 (float64[:,:])', nopython=True)
def det(X):
    """
    Direct computation of determinant for 
    matrices of size 2x2 and 3x3
    """
    n = X.shape[0]
    if n==2:
        return X[0,0]*X[1,1] - X[0,1]*X[1,0]
    else:
        return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
               X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
               X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])
            


@numba.jit('float64[:,:] (float64[:,:])', nopython=True)
def inv(X):
    """
    Direct computation of inverse for 
    matrices of size 2x2 and 3x3
    """
    n = X.shape[0]
    if n==2:
        ret = np.empty((2,2))
        ret[0,0] = X[1,1]; ret[0,1] = -X[0,1]
        ret[1,0] = -X[1,0]; ret[1,1] = X[0,0]
        return 1./det(X) * ret
    else:
        ret = np.empty((3,3))
        ret[0,0]=X[2,2]*X[1,1]-X[2,1]*X[1,2]; ret[0,1]=-X[2,2]*X[0,1]+X[2,1]*X[0,2]; ret[0,2]=X[1,2]*X[0,1]-X[1,1]*X[0,2]
        ret[1,0]=-X[2,2]*X[1,0]+X[2,0]*X[1,2]; ret[1,1]=X[2,2]*X[0,0]-X[2,0]*X[0,2]; ret[1,2]=-X[1,2]*X[0,0]+X[1,0]*X[0,2]
        ret[2,0]=X[2,1]*X[1,0]-X[2,0]*X[1,1]; ret[2,1]=-X[2,1]*X[0,0]+X[2,0]*X[0,1]; ret[2,2]=X[1,1]*X[0,0]-X[1,0]*X[0,1]
        return 1./det(X) * ret
