#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True 
import numpy as np
from libc.math cimport exp, cos, sin
from cython.parallel import prange



def gm_eval(double[:] w, double[:] sig, double[:] xc, double[:] yc, double[:] xe, double[:] ye):    
    """
    Fast Gaussian Mixture Evaluation: No truncation

    parameters
    ----------
    w     : 1D np.ndarray - weight values
    sig   : 1D np.ndarray - sigma values
    xc    : 1D np.ndarray - x coordinate of center points
    yc    : 1D np.ndarray - y coordinate of center points
    xe    : 1D np.ndarray - x coordinate of eval points
    ye    : 1D np.ndarray - y coordinate of eval points
    """
    cdef int m = len(xe)
    cdef int n = len(xc)
    cdef int i,j
    cdef double quad
    cdef double[:] ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            quad = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            ret[i] += w[j] * exp( -quad/sig[j]**2 )
    return ret.base


def gm_eval_3d(double[:] w, double[:] sig, double[:] xc, double[:] yc, double[:] zc,
               double[:] xe, double[:] ye, double[:] ze):    
    """
    Fast Gaussian Mixture Evaluation: No truncation

    parameters
    ----------
    w     : 1D np.ndarray - weight values
    sig   : 1D np.ndarray - sigma values
    xc    : 1D np.ndarray - x coordinate of center points
    yc    : 1D np.ndarray - y coordinate of center points
    zc    : 1D np.ndarray - z coordinate of center points
    xe    : 1D np.ndarray - x coordinate of eval points
    ye    : 1D np.ndarray - y coordinate of eval points
    ze    : 1D np.ndarray - z coordinate of eval points
    """
    cdef int m = len(xe)
    cdef int n = len(xc)
    cdef int i,j
    cdef double quad
    cdef double[:] ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            quad = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2 +  (ze[i]-zc[j])**2
            ret[i] += w[j] * exp( -quad/sig[j]**2 )
    return ret.base


def gm_eval_trunc(double[:] w, double[:] sig, double[:] xc, double[:] yc, double[:] xe, 
                 double[:] ye, long[:] neigh_indexes, long[:] neigh_indexes_aux):
    """
    Fast Gaussian Mixture Evaluation: Truncated Gaussians

    parameters
    ----------
    w     : 1D np.ndarray - weight values
    sig   : 1D np.ndarray - sigma values
    xc    : 1D np.ndarray - x coordinate of center points
    yc    : 1D np.ndarray - y coordinate of center points
    xe    : 1D np.ndarray - x coordinate of eval points
    ye    : 1D np.ndarray - y coordinate of eval points
    neigh_indexes : 1D np.ndarray - indexes of neighbors of all eval points
    neigh_indexes_aux : 1D np.ndarray - limit indices for neighbors of eval points in neigh_indexes
    """
    cdef int m = len(xe)
    cdef int i,j,sind,eind
    cdef double quad
    cdef double[:] ret = np.zeros(m)
    sind = 0 # start index
    for i in range(m):
        eind = neigh_indexes_aux[i] # end index
        for j in range(sind,eind):
            j = neigh_indexes[j]
            quad = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            ret[i] += w[j] * exp( -quad/sig[j]**2 )
        sind = eind
    return ret.base


def gm_eval_trunc_thread(double[:] w, double[:] sig, double[:] xc, double[:] yc, double[:] xe, 
                   double[:] ye, long[:] neigh_indexes, long[:] neigh_indexes_aux):
    """
    Gaussian Mixture Evaluation with threads: Low memory consumption
    """ 
    cdef int m = xe.shape[0]
    cdef int i,j,sind,eind
    cdef double quad
    cdef double[:] ret = np.zeros(m)
    
    for i in prange(m, nogil=True):
        if i==0: sind = 0
        else: sind = neigh_indexes_aux[i-1]
        eind = neigh_indexes_aux[i]
        for j in range(sind,eind):
            j = neigh_indexes[j]
            quad = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            ret[i] += w[j] * exp( -quad/sig[j]**2 )
    return ret.base


def gm_eval_trunc_thread2(double[:] w, double[:] sig, double[:] xc, double[:] yc, double[:] xe, 
                    double[:] ye, long[:,:] neigh_indexes):
    """
    Gaussian Mixture Evaluation with threads
    """
    cdef int m = neigh_indexes.shape[0]
    cdef int n = neigh_indexes.shape[1]
    cdef int i,j
    cdef double quad = 0
    cdef double[:] ret = np.zeros(m)
    
    for i in prange(m, nogil=True):
        for j in range(n):
            j = neigh_indexes[i,j]
            if j==-1: break
            quad = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2 
            ret[i] += w[j] * exp( -quad/sig[j]**2 )
    return ret.base


def gm_eval_full_(double[:] w, double[:,:] cov, double[:] xc, double[:] yc, double[:] xe, double[:] ye):
    """
    Gaussian Mixture Evaluation with full Covariance matrix
    """
    cdef int m = len(xe)
    cdef int n = len(xc)
    cdef int i,j
    cdef double quad
    cdef double[:] ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            quad = cov[j,2]*(xe[i]-xc[j])**2 - 2*cov[j,1]*(xe[i]-xc[j])*(ye[i]-yc[j]) + cov[j,0]*(ye[i]-yc[j])**2
            det = cov[j,0]*cov[j,2]-cov[j,1]**2
            ret[i] += w[j]*exp(-quad/det)
    return ret.base


def gm_eval_full(double[:] w, double[:,:] sig, double[:] theta, double[:] xc, double[:] yc, 
                  double[:] xe, double[:] ye):
    """
    Gaussian Mixture Evaluation with full Covariance matrix (parameterized by sig and theta)
    """
    cdef int m = len(xe)
    cdef int n = len(xc)
    cdef int i,j
    cdef double a,b,c,quad
    cdef double[:] ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            a = cos(theta[j])**2/sig[j,0]**2 + sin(theta[j])**2/sig[j,1]**2
            b = -sin(2*theta[j])/sig[j,0]**2 + sin(2*theta[j])/sig[j,1]**2
            c = sin(theta[j])**2/sig[j,0]**2 + cos(theta[j])**2/sig[j,1]**2
            quad = a*(xe[i]-xc[j])**2 + b*(xe[i]-xc[j])*(ye[i]-yc[j]) + c*(ye[i]-yc[j])**2
            ret[i] += w[j]*exp(-quad)
    return ret.base


def gm_eval_full_thread(double[:] w, double[:,:] sig, double[:] theta, double[:] xc, double[:] yc, 
                  double[:] xe, double[:] ye, long[:] neigh_indexes, long[:] neigh_indexes_aux):
    """
    Gaussian Mixture Evaluation with full Covariance matrix (parameterized by sig and theta): Multithreading version 
    """
    cdef int m = len(xe)
    cdef int i,j,sind,eind
    cdef double a,b,c,quad
    cdef double[:] ret = np.zeros(m)
    for i in prange(m, nogil=True):
        if i==0: sind=0
        else: sind = neigh_indexes_aux[i-1]
        eind = neigh_indexes_aux[i]
        for j in range(sind,eind):
            j = neigh_indexes[j]
            a = cos(theta[j])**2/sig[j,0]**2 + sin(theta[j])**2/sig[j,1]**2
            b = -sin(2*theta[j])/sig[j,0]**2 + sin(2*theta[j])/sig[j,1]**2
            c = sin(theta[j])**2/sig[j,0]**2 + cos(theta[j])**2/sig[j,1]**2
            quad = a*(xe[i]-xc[j])**2 + b*(xe[i]-xc[j])*(ye[i]-yc[j]) + c*(ye[i]-yc[j])**2
            ret[i] += w[j]*exp(-quad)
    return ret.base