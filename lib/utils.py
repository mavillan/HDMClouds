import sys
import numba
import scipy
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits


@numba.jit(nopython=True)
def u_eval(c, sig, xc, yc, xe, ye, support=5):
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            ret[i] += c[j] * exp( -0.5 * dist2 / sig[j]**2 )
    return ret


@numba.jit('float64[:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', nopython=True)
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


@numba.jit('float64[:,:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', nopython=True)
def u_eval_full(c, sig, xc, yc, xe, ye, support=5):
    m = len(xe)
    n = len(xc)
    ret = np.zeros((6,m))
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            # common terms
            coef = (-1./(sig[j]**2))
            coef2 = coef**2
            # u evaluation
            aux = c[j] * exp( -dist2 / (2* sig[j]**2 ) )
            ret[0,i] += aux
            # ux evaluation
            ret[1,i] += coef * aux * (xe[i]-xc[j])
            # uy evaluation
            ret[2,i] += coef * aux * (ye[i]-yc[j])
            # uxy evaluation
            ret[3,i] += coef2 * aux * (xe[i]-xc[j])*(ye[i]-yc[j])
            # uxx evaluation
            ret[4,i] += coef2 * aux * ((xe[i]-xc[j])**2 - sig[j]**2)
            # uyy evaluation
            ret[5,i] += coef2 * aux * ((ye[i]-yc[j])**2 - sig[j]**2)
    return ret


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
    hdulist = fits.open(fits_path)
    data = hdulist[0].data

    if data.ndim>3:
        # droping out the stokes dimension
        data = np.ascontiguousarray(data[0])

        if data.shape[0]==1:
            # in case data is an image and not a cube
            data = np.ascontiguousarray(data[0])
    
    # in case NaN values exist on data
    mask = np.isnan(data)
    if np.any(mask): data = ma.masked_array(data, mask=mask)

    # map to 0-1 intensity range
    data -= data.min()
    data /= data.max()
    
    if data.ndim==2:
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x,y), data, method='linear', bounds_error=False, fill_value=0.)

    elif data.ndim==3:
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        z = np.linspace(0., 1., data.shape[2]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x, y, z), data, method='linear', bounds_error=False, fill_value=0.)

    return data,dfunc





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


def prune(vec):
    mean = np.mean(vec)
    median = np.median(vec)
    #all values greater than 1e-3 the mean/median
    mask = vec > 1e-3*min(mean,median)
    return mask, vec[mask]


def gradient(img):
    gx, gy = np.gradient(img)
    img_grad = np.sqrt(gx**2 + gy**2)
    return img_grad


def stat_extractor(r_stats, stat):
    """
    Function to extract a single residual stat from 
    the r_stats structure (list of tuples)
    """
    if stat=='variance':
        return np.array([var for (var,_,_,_,_,_,_,_,_) in r_stats])
    elif stat=='entropy':
        return np.array([entr for (_,entr,_,_,_,_,_,_,_) in r_stats])
    elif stat=='rms':
        return np.array([rms for (_,_,rms,_,_,_,_,_,_) in r_stats])
    elif stat=='flux_addition':
        return np.array([flux for (_,_,_,flux,_,_,_,_,_) in r_stats])
    elif stat=='flux_lost':
        return np.array([flux for (_,_,_,_,flux,_,_,_,_) in r_stats])
    elif stat=='sharpness':
        return np.array([sharp for (_,_,_,_,_,_,_,sharp,_) in r_stats])


@numba.jit('int64[:,:] (int64[:,:], float64, int32, int32)')
def remove_isolate(inp, thresh, on, off):
    """
    Cellular automaton from FellWalker Algorithm to remove isolated
    group of pixels
    """
    out = np.empty_like(inp)
    shape = inp.shape

    for ox in range(shape[0]):
        for oy in range(shape[1]):
            if inp[ox,oy] == off:
                # If the corresponding input pixel is off, then the output must be also be off
                out[ox,oy] = off
            else:
                # Otherwise, loop round all input pixels in the neighbourhood of the current
                # output pixel, this is a cube of 3x3x3 input pixels, centred on the current
                # output pixel. Count how many of these input pixels are set to "on". If
                # the current output pixel is close to an edge of the array, there will be
                # fewer than 3x3x3 pixels in the cube. Count the total number of pixels
                # in the cube.
                s = 0
                tot = 0
                for ix in range(ox-1, ox+2):
                    if ix < 0 or ix >= shape[0] : continue
                    for iy in range(oy-1, oy+2):
                        if iy < 0 or iy >= shape[1]: continue
                        if ix==ox and iy==oy: continue
                        tot += 1
                        if inp[ix,iy] == on: s += 1
                if float(s)/float(tot) > thresh:
                    out[ox,oy] = on
                else:
                    out[ox,oy] = off
    return out
