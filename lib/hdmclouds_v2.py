import time
import scipy
import numba
import scipy as sp
import numpy as np
import numexpr as ne
from math import sqrt, exp
import matplotlib.pyplot as plt

import graph as gp
from utils import *
from points_generation import *
from preprocessing import *
from gmr import *

from fgm_eval import gm_eval, gm_eval_full
from fgm_eval import gm_eval_full_thread as gm_eval_full_fast


#################################################################
# General Psi penalizing function
#################################################################
def psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 1.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('10*x**3 - 15*x**4 + 6*x**5')
    return ret

def d1psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 0.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('30*x**2 - 60*x**3 + 30*x**4')
    return lamb*ret




#################################################################
# HDMClouds class definition
#################################################################
class HDMClouds():
    def __init__(self, data, freq=None, alpha=0., lamb=1., n_center=200, back_level=None,
        minsig=None, maxsig=None, pix_freedom=1., verbose=False, wcs=None):

        #############################################################
        # Preprocessing: Estimation of back_level, computing mask, 
        # standardizing the data, building the linear interpolator
        #############################################################
        self.orig_data = data
        if back_level is None:
            back_level = 0.25*estimate_rms(data)
        data = np.copy(data)
        mask = compute_mask(data, back_level)
        data[mask] -= back_level
        vmin = data[mask].min(); vmax = data[mask].max()
        data -= vmin
        data /= vmax-vmin
        data[~mask] = 0
    
        if data.ndim==2:
            # generating the data function
            _x = np.linspace(0., 1., data.shape[0]+1, endpoint=True)
            _y = np.linspace(0., 1., data.shape[1]+1, endpoint=True)
            x = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
            y = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
            dfunc = RegularGridInterpolator((x,y), data, method='linear', bounds_error=False, fill_value=0.)

        elif data.ndim==3:
            # generating the data function
            _x = np.linspace(0., 1., data.shape[0]+1, endpoint=True)
            _y = np.linspace(0., 1., data.shape[1]+1, endpoint=True)
            _z = np.linspace(0., 1., data.shape[2]+1, endpoint=True)
            x = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
            y = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
            z = np.asarray( [(_z[i]+_z[i+1])/2 for i in range(len(_z)-1)] )
            dfunc = RegularGridInterpolator((x, y, z), data, method='linear', bounds_error=False, fill_value=0.)

        self.data = data
        self.freq = freq
        self.mask = mask
        self.dims = data.shape
        self.vmin = vmin
        self.vmax = vmax
        self.dfunc = dfunc
        self.wcs = wcs
        
        # Verification of consistency between n_center and number of significant emission pixels
        npix = np.sum(mask)
        if 4*n_center > npix:
            n_center = npix//4
            print("[WARNING] The number of evaluation points (4*n_center) cannot"
                  "be greater than the number of significant emission pixels: {0}".format(npix))
            print("[WARNING] n_center was set to: {0}".format(n_center))

        #######################################
        # Evaluation regular-grid points
        ####################################### 
        _x = np.linspace(0., 1., data.shape[0]+1, endpoint=True)
        _y = np.linspace(0., 1., data.shape[1]+1, endpoint=True)
        _xe = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
        _ye = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
        Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
        xgrid = Xe.ravel(); ygrid = Ye.ravel()
        self.xgrid = xgrid
        self.ygrid = ygrid
        points_grid = np.vstack([xgrid,ygrid]).T


        #######################################
        # Computing boundary points
        #######################################
        Nb = 0
        boundary_points = boundary_points_generation(data, mask, Nb)
        #gp.points_plot(data, points=boundary_points, color="green", wcs=wcs)
        # right format
        xb = boundary_points[:,0]
        yb = boundary_points[:,1]
        points_bound = np.vstack([xb,yb]).T

        #######################################
        # Estimating initial guess
        #######################################
        # initial w
        w = (data[mask]).astype(np.float64)

        # initial mu
        xc = Xe[mask]; yc = Ye[mask]
        mu = np.vstack([xc,yc]).T

        # initial sigma
        k = 0.25
        pix_lenght = min(1./data.shape[0], 1./data.shape[1])
        sig = (pix_lenght/(2.*k))*np.ones(w.shape[0])

        # re-normalizing w
        u = gm_eval(w, sig, xc, yc, xgrid, ygrid)
        w *= data.max()/u.max()

        # agglomeration
        w_red,mu_red,sig_red = mixture_reduction(w, mu, sig, 4*n_center, verbose=False)

        # evaluation points
        xe = mu_red[:,0]
        ye = mu_red[:,1]
        eval_points = np.vstack([xe,ye]).T
        
        # agglomeration must go on
        w_red,mu_red,sig_red = mixture_reduction(w_red, mu_red, sig_red, n_center, verbose=False)

        # gaussian center points
        xc = mu_red[:,0]
        yc = mu_red[:,1]
        center_points = np.vstack([xc,yc]).T

        # Extracting sigx, sigy and theta from covariances
        sig = []
        theta = []
        for cov in sig_red:
            lam,V = np.linalg.eig(cov)
            sig.append( np.sqrt(np.sort(lam)[::-1]) )
            theta.append( np.arccos(V.T[0,0]) )
        sig = np.asarray(sig).ravel()
        theta = np.asarray(theta)


        if verbose:
            # visualizing the choosen points
            gp.points_plot(data, points=center_points, color="red", wcs=wcs)
            gp.points_plot(data, points=eval_points, color="blue", wcs=wcs)
            
        
        ########################################
        # Computing neighbor indexes for 
        # fast evaluation
        ########################################
        minsig = np.min(np.abs(sig))
        maxsig = 3*np.max(np.abs(sig))
        epsilon = 1e-6 # little shift to avoid NaNs in inv_sig_mapping
        neigh_indexes,neigh_indexes_aux = compute_neighbors(mu_red, eval_points, 5*maxsig)
        self.nind1 = neigh_indexes
        self.nind_aux1 = neigh_indexes_aux

        #neigh_indexes,neigh_indexes_aux = compute_neighbors(mu_red, points_bound, 5*np.max(sig_red))
        #self.nind2 = neigh_indexes
        #self.nind_aux2 = neigh_indexes_aux

        neigh_indexes,neigh_indexes_aux = compute_neighbors(mu_red, points_grid, 5*maxsig)
        self.nind3 = neigh_indexes
        self.nind_aux3 = neigh_indexes_aux


        ########################################
        # HDMClouds internals
        ########################################
        self.f0 = dfunc( np.vstack([xe,ye]).T )
        self.fb = dfunc( np.vstack([xb,yb]).T )
        self.xb = xb; self.yb = yb
        self.xe = xe; self.ye = ye
        self.xc = xc; self.yc = yc
        self.minsig = minsig
        self.maxsig = maxsig
        ###########################
        # optimization parameters
        self.w = np.sqrt(w_red)
        self.w0 = np.copy(w_red)
        self.sig = inv_sig_mapping(sig+epsilon, minsig, maxsig)
        self.sig0 = np.copy(sig)
        self.theta = inv_theta_mapping(theta)
        self.theta0 = np.copy(theta)
        ###########################
        self.d1psi1 = d1psi1
        self.alpha = alpha
        self.lamb = lamb
        self.back_level = back_level
        self.scipy_sol = None
        self.scipy_tol = None
        self.elapsed_time = None
        self.residual_stats = None

    def set_w(self, w):
        self.w = w

    def set_sig(self, sig):
        self.sig = sig

    def set_theta(self, theta):
        self.theta = theta

    def set_params(self, params):
        N = len(params)//4
        self.w = params[0:N]
        self.sig = params[N:3*N]
        self.theta = params[3*N:]

    def get_params(self):
        """
        Get the parameter of the function F (to optimize): 
        theta_xc, theta_yc, c, sig
        """ 
        return np.concatenate([self.w, self.sig, self.theta])

    def get_params_mapped(self):
        """
        Get the real parameters of the model (mapped/bounded):
        xc, yc, c, sig
        """
        #xc = self.xc0 + self.deltax * np.sin(self.theta_xc)
        #yc = self.yc0 + self.deltay * np.sin(self.theta_yc)
        w = self.w**2
        sig = sig_mapping(self.sig, self.minsig, self.maxsig)
        theta = theta_mapping(self.theta) 
        return w, sig, theta

    def normalized_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal functions. 
        """
        w,sig,theta = self.get_params_mapped()
        d = len(self.dims)
        w = w * (2*np.pi*sig**2)**(d/2.)
        return w


    def get_approximation(self):
        w,sig,theta = self.get_params_mapped()
        sig = sig.reshape((-1,2))
        u = gm_eval_full_fast(w, sig, theta, self.xc, self.yc, self.xgrid, self.ygrid, self.nind3, self.nind_aux3)
        u = u.reshape(self.dims)
        return u


    def get_residual_stats(self, plot=True):
        u = self.get_approximation()
        if self.mask is not None:
            residual = np.zeros(self.dims)
            residual[self.mask] = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u

        # visualization of residual
        if plot:
            gp.solution_plot(self.data, u, residual)
        # computing residual stats
        total_flux = np.sum(self.data[self.mask])
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])

        # output residuals
        out = (estimate_rms(residual), estimate_variance(residual), \
               flux_addition/total_flux, flux_lost/total_flux)

        print("RESIDUAL STATS")
        print("RMS of residual: {0}".format(out[0]))
        print("Inf norm of residual: {0}".format(np.max(np.abs(residual))))
        print("Variance of residual: {0}".format(out[1]))
        print("Normalized flux addition: {0}".format(out[2]))
        print("Normalized flux lost: {0}".format(out[3]))

        return out


    def prune(self):
        w = self.w
        mean = np.mean(w)
        median = np.median(w)
        # all values greater than 1e-3 the mean/median
        mask = w > 1e-3*min(mean,median)
        # update all the arrays
        self.xc = self.xc[mask]
        self.yc = self.yc[mask]
        self.w = self.w[mask]
        self.sig = self.sig[mask]


    def solver_output(self):
        if self.scipy_sol is not None:
            print('Solver Output:')
            print('success: {0}'.format(self.scipy_sol['success']))
            print('status: {0}'.format(self.scipy_sol['status']))
            print('message: {0}'.format(self.scipy_sol['message']))
            print('nfev: {0}'.format(self.scipy_sol['nfev']))
            print("xtol: {0}".format(self.scipy_tol))
            print("ftol: {0}".format(self.scipy_tol))


    
    def summarize(self, solver_output=True, residual_stats=True, solution_plot=True,
                  params_plot=True, histograms_plot=True):
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90 + '\n')
        _w,_sig,_theta = self.get_params_mapped()

        out = self.get_residual_stats(plot=False)
        var,rms,flux_addition,flux_lost = out
        
        if solution_plot:
            gp.solution_plot(self.dfunc, _w, _sig, self.xc, self.yc, dims=self.dims)
        
        if params_plot:
            gp.params_plot(_w, _sig, self.xc, self.yc)
            gp.params_distribution_plot(_w, _sig)

        if histograms_plot:
            u = self.get_approximation()
            term1 = u[self.mask]-self.data[self.mask]
            plt.figure(figsize=(8,8))
            plt.hist(term1.ravel(), bins=10, facecolor='seagreen', edgecolor='black', lw=2)
            plt.title('u-f')
            plt.show()

            
    def F(self, params):
        N = len(params)//4

        # parameters transform/mapping
        w = params[0:N]**2
        sig = sig_mapping(params[N:3*N], self.minsig, self.maxsig).reshape((-1,2))
        theta = theta_mapping(params[3*N:]) 
        
        # computing the EL equation
        u = gm_eval_full_fast(w, sig, theta, self.xc, self.yc, self.xe, self.ye, self.nind1, self.nind_aux1)

        # evaluation of the el equation
        f0 = self.f0
        alpha = self.alpha; lamb = self.lamb
        tmp1 = ne.evaluate('u-f0')
        tmp2 = self.d1psi1(tmp1, lamb)
        el = ne.evaluate('2*tmp1 + alpha*tmp2')
            
        # evaluating at the boundary
        fb = self.fb
        u_boundary = gm_eval_full(w, sig, theta, self.xc, self.yc, self.xb, self.yb)
        
        return np.concatenate([el,u_boundary-fb])


    def build_gmr(self, max_nfev=None, verbose=True, tol=1.e-7):
        """
        Build the Gaussian Mixture Representation for the
        input data
        """
        t0 = time.time()

        # lm optimization from scipy.optimize.root
        #if max_nfev is None:
        #    max_nfev = 100*(len(self.get_params())+1)
        #options = {'maxiter':max_nfev, 'xtol':xtol, 'ftol':ftol}
        #sol = sp.optimize.root(self.F, self.get_params(), method='lm', options=options)
        while True:
            options = {'xtol':tol, 'ftol':tol}
            sol = sp.optimize.root(self.F, self.get_params(), method='lm', options=options)
            if sol["status"]!=2:
                self.scipy_tol = tol
                break
            tol /= 10
        sol_length = len(sol.x)//4
        opt_w = sol.x[0:sol_length]
        opt_sig = sol.x[sol_length:3*sol_length]
        opt_theta = sol.x[3*sol_length:]

        # update to best parameters
        self.set_w(opt_w)
        self.set_sig(opt_sig)
        self.set_theta(opt_theta)

        # prune of gaussians from mixture
        #self.prune()
    
        # storing results    
        self.scipy_sol = sol
        self.elapsed_time = time.time() - t0
        #self.summarize()


    def build_htree(self):
        pass

