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
from initial_guess import *
from preprocessing import *


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
# HDSources class definition
#################################################################
class HDSources():
    def __init__(self, data, alpha=0., lamb=1., n_center=200, back_level=None, 
        minsig=None, maxsig=None, pix_freedom=1., verbose=False, wcs=None):

        #############################################################
        # Preprocessing: Estimation of back_level, computing mask, 
        # standardizing the data, building the linear interpolator
        #############################################################
        if back_level is None:
            back_level = 0.25*estimate_rms(data)
        mask = compute_mask(data, back_level)
        data = np.copy(data)
        data[mask] -= back_level
        vmin = data[mask].min(); vmax = data[mask].max()
        data -= vmin
        data /= vmax-vmin
        data[~mask] = 0
    
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

        self.data = data
        self.mask = mask
        self.dims = data.shape
        self.vmin = vmin
        self.vmax = vmax
        self.dfunc = dfunc

        #######################################
        # Evaluation regular-grid points
        ####################################### 
        _xe = np.linspace(0., 1., data.shape[0]+2)[1:-1]
        _ye = np.linspace(0., 1., data.shape[1]+2)[1:-1]
        Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
        xgrid = Xe.ravel(); ygrid = Ye.ravel()
        self.xgrid = xgrid
        self.ygrid = ygrid

        #######################################
        # Computing points
        #######################################
        Nc = n_center
        Nb = int(0.2*Nc)
        Ne = 4*Nc-Nb

        points = qrandom_centers_generation(dfunc, Ne, ndim=2)
        center_points = points[0:Nc]
        collocation_points = points[0:Ne]
        boundary_points = boundary_points_generation(data, mask, Nb)

        # right format
        xc = center_points[:,0]
        yc = center_points[:,1]
        xe = collocation_points[:,0]
        ye = collocation_points[:,1]
        xb = boundary_points[:,0]
        yb = boundary_points[:,1]

        if verbose:
            # visualizing the choosen points
            gp.points_plot(data, points=center_points, label="Center points", color="red", wcs=wcs)
            gp.points_plot(data, points=collocation_points, label="Collocation points", color="blue", wcs=wcs)
            gp.points_plot(data, points=boundary_points, label="Boundary points", color="green", wcs=wcs)

        #######################################
        # Estimating initial guess
        #######################################
        if minsig is None:
            minsig = ( 0.5*(1./data.shape[0] + 1./data.shape[1]) ) / 6.
        if maxsig is None:
            # 30 is hardcoded...
            maxsig = 30*minsig
        c0, sig0 = estimate_initial_guess(center_points, dfunc, minsig=minsig, 
                   maxsig=maxsig, method='min_dist')

        if verbose:
            # visualizing the initial guess
            gp.solution_plot(dfunc, c0, sig0, xc, yc, dims=data.shape)
            gp.params_plot(c0, sig0, xc, yc)
            gp.params_distribution_plot(c0, sig0)

        ########################################
        # HDSources internals
        ########################################
        self.f0 = dfunc( np.vstack([xe,ye]).T )
        self.fb = dfunc( np.vstack([xb,yb]).T )
        self.xb = xb; self.yb = yb
        self.xe = xe; self.ye = ye
        self.xc = xc; self.yc = yc
        self.xc0 = xc; self.yc0 = yc
        self.theta_xc = np.zeros(Nc)
        self.theta_yc = np.zeros(Nc)
        self.deltax = pix_freedom * 1./data.shape[0]
        self.deltay = pix_freedom * 1./data.shape[1]
        self.minsig = minsig
        self.maxsig = maxsig
        self.c = np.sqrt(c0)
        self.sig = inv_sig_mapping(sig0, minsig, maxsig)
        self.d1psi1 = d1psi1
        self.alpha = alpha
        self.lamb = lamb
        self.back_level = back_level
        self.scipy_sol = None
        self.elapsed_time = None
        self.residual_stats = None

    def set_centers(self, theta_xc, theta_yc):
        self.xc = self.xc0 + self.deltax * np.sin(theta_xc)
        self.yc = self.yc0 + self.deltay * np.sin(theta_yc)   

    def set_theta(self, theta_xc, theta_yc):
        self.theta_xc = theta_xc
        self.theta_yc = theta_yc

    def set_c(self, c):
        self.c = c

    def set_sig(self, sig):
        self.sig = sig

    def set_params(self, params):
        N = len(params)//4
        self.theta_xc = params[0:N]
        self.theta_yc = params[N:2*N]
        self.c = params[2*N:3*N]
        self.sig = params[3*N:4*N]

    def get_params(self):
        """
        Get the parameter of the function F (to optimize): 
        theta_xc, theta_yc, c, sig
        """
        return np.concatenate([self.theta_xc, self.theta_yc, self.c, self.sig])

    def get_params_mapped(self):
        """
        Get the real parameters of the model (mapped/bounded):
        xc, yc, c, sig
        """
        #xc = self.xc0 + self.deltax * np.sin(self.theta_xc)
        #yc = self.yc0 + self.deltay * np.sin(self.theta_yc)
        xc = self.xc
        yc = self.yc
        c = self.c**2
        sig = sig_mapping(self.sig, self.minsig, self.maxsig)
        return xc, yc, c, sig

    def get_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal functions. 
        """
        xc, yc, c, sig = self.get_params_mapped()
        d = len(self.dims)
        w = c * (2*np.pi*sig**2)**(d/2.)
        return w


    def get_residual_stats(self):
        xc, yc, c, sig = self.get_params_mapped()
        u = u_eval(c, sig, xc, yc, self.xgrid, self.ygrid)
        u = u.reshape(self.dims)

        if self.mask is not None:
            residual = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u
        
        # first term of Lagrangian stats
        total_flux = np.sum(self.data[self.mask])
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])
        npix = np.sum(flux_mask)

        residual_stats = estimate_variance(residual), estimate_rms(residual), \
                         flux_addition/total_flux, flux_lost/total_flux, npix
        self.residual_stats = residual_stats
        return residual_stats


    def get_approximation(self):
        xc, yc, c, sig = self.get_params_mapped()
        u = u_eval(c, sig, xc, yc, self.xgrid, self.ygrid)
        u = u.reshape(self.dims)
        return u


    def get_gradient(self):
        xc, yc, c, sig = self.get_params_mapped()
        grad = grad_eval(c, sig, xc, yc, self.xgrid, self.ygrid)
        grad = grad.reshape(self.dims)
        return grad


    def prune(self):
        w = self.get_w()
        mask, _ = prune(w)
        #update all the arrays
        self.xc = self.xc[mask]; self.xc0 = self.xc0[mask]
        self.yc = self.yc[mask]; self.yc0 = self.yc0[mask]
        self.theta_xc = self.theta_xc[mask]
        self.theta_yc = self.theta_yc[mask]
        self.c = self.c[mask]
        self.sig = self.sig[mask]

    
    def summarize(self, solver_output=True, residual_stats=True, solution_plot=True,
                  params_plot=True, histograms_plot=True):
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90 + '\n')
        _xc, _yc, _c, _sig = self.get_params_mapped()

        out = self.get_residual_stats()
        var,rms,flux_addition,flux_lost,npix = out
        
        if solver_output:
            print('Solver Output:')
            print('success: {0}'.format(self.scipy_sol['success']))
            print('status: {0}'.format(self.scipy_sol['status']))
            print('message: {0}'.format(self.scipy_sol['message']))
            print('nfev: {0}'.format(self.scipy_sol['nfev']))
        
        if residual_stats:
            print('\nResidual stats:')
            print('Residual RMS: {0}'.format(rms))
            print('Residual Variance: {0}'.format(var))
            print('Flux Lost: {0}'.format(flux_lost))
            print('Flux Addition: {0}'.format(flux_addition))
            print('Exceeded Pixels: {0}'.format(npix))
            print('Total elapsed time: {0} [s]'.format(self.elapsed_time))
        
        if solution_plot:
            gp.solution_plot(self.dfunc, _c, _sig, _xc, _yc, dims=self.dims)
        
        if params_plot:
            gp.params_plot(_c, _sig, _xc, _yc)
            gp.params_distribution_plot(_c, _sig)

        if histograms_plot:
            u = self.get_approximation()
            term1 = u[self.mask]-self.data[self.mask]
            plt.figure(figsize=(8,8))
            plt.hist(term1.ravel(), bins=10, facecolor='seagreen', edgecolor='black', lw=2)
            plt.title('u-f')
            plt.show()

            
    def F(self, params):
        N = len(params)//4
        theta_xc = params[0:N]
        theta_yc = params[N:2*N]

        # parameters transform/mapping
        xc = self.xc0 + self.deltax * np.sin(theta_xc)
        yc = self.yc0 + self.deltay * np.sin(theta_yc)
        c = params[2*N:3*N]**2
        sig = sig_mapping(params[3*N:4*N], self.minsig, self.maxsig)

        # evaluation points
        xe = np.hstack([self.xe, xc]); ye = np.hstack([self.ye, yc])
        xb = self.xb; yb = self.yb
        
        # computing the EL equation
        u = u_eval(c, sig, xc, yc, xe, ye)
        f0 = np.hstack([ self.f0, self.dfunc(np.vstack([xc,yc]).T) ])
        alpha = self.alpha; lamb = self.lamb

        tmp1 = ne.evaluate('u-f0')
        tmp2 = self.d1psi1(tmp1, lamb)
        el = ne.evaluate('2*tmp1 + alpha*tmp2')
            
        # evaluating at boundary
        fb = self.fb
        u_boundary = u_eval(c, sig, xc, yc, self.xb, self.yb)
        
        return np.concatenate([el,u_boundary-fb])


    def build_gmr(self, max_nfev=None, n_iter=100, verbose=True, xtol=1.e-7, ftol=1.e-7):
        """
        Build the Gaussian Mixture Representation for the
        input data
        """
        t0 = time.time()

        # lm optimization from scipy.optimize.root
        if max_nfev is None:
            max_nfev = 100*(len(self.get_params())+1)
        options = {'maxiter':max_nfev, 'xtol':xtol, 'ftol':ftol}
        sol = sp.optimize.root(self.F, self.get_params(), method='lm', options=options)
        sol_length = len(sol.x)//4
        opt_theta_xc = sol.x[0:sol_length]
        opt_theta_yc = sol.x[sol_length:2*sol_length]
        opt_c = sol.x[2*sol_length:3*sol_length]
        opt_sig = sol.x[3*sol_length:4*sol_length]

        # update to best parameters
        self.set_theta(opt_theta_xc, opt_theta_yc)
        self.set_centers(opt_theta_xc, opt_theta_yc)
        self.set_c(opt_c)
        self.set_sig(opt_sig)

        # prune of gaussians from mixture
        self.prune()
    
        # storing results    
        self.scipy_sol = sol
        self.elapsed_time = time.time() - t0
        #self.summarize()


    def build_htree(self):
        pass

