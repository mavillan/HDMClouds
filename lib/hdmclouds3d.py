import time
import scipy
import numba
import scipy as sp
import numpy as np
import numexpr as ne
from math import sqrt, exp
import matplotlib.pyplot as plt
import graph as gp
from utils3D import *



#################################################################
# General Psi penalizing function (applicable in both cases)
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


def d2psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 0.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('60*x - 180*x**2 + 120*x**3')
    return (lamb**2)*ret


def psi2(x, lamb=1.):
    return lamb**2 * np.log(1 + x/lamb**2)


def d1psi2(x, lamb=1.):
    return 1./(1.+x/lamb**2)


def d2psi2(x, lamb=1.):
    return -lamb**2/(lamb**2+x)**2




#################################################################
# HDMClouds 3D class definition
#################################################################
class HDMClouds():
    def __init__(self, data, dfunc, dims, xe, ye, ze, xc, yc, zc, xb, yb, zb, c0, sig0, d1psi1=d1psi1, 
                 d1psi2=d1psi2, d2psi2=d2psi2, a=0., b=0., lamb1=1., lamb2=1., base_level=0., minsig=None,
                 maxsig=None, pix_freedom=1., support=5., mask=None):
        f0 = dfunc( np.vstack([xe,ye,ze]).T )
        fb = dfunc( np.vstack([xb,yb,zb]).T )
        len_f0 = len(f0)
        len_c0 = len(c0)
        len_sig0 = len(sig0)
        Ne = len(xe)
        Nc = len(xc)
        Nb = len(xb)
        
        # important attribute
        self.data = data
        if mask is None:
            self.mask = data > base_level
        else:
            self.mask = mask
        self.dfunc = dfunc
        self.dims = dims
        self.f0 = f0
        self.fb = fb
        self.xb = xb; self.yb = yb; self.zb = zb
        self.xe = xe; self.ye = ye; self.ze = ze
        self.xc = xc; self.yc = yc; self.zc = zc
        self.xc0 = xc; self.yc0 = yc; self.zc0 = zc
        self.theta_xc = np.zeros(Nc)
        self.theta_yc = np.zeros(Nc)
        self.theta_zc = np.zeros(Nc)
        self.deltax = pix_freedom * 1./dims[0]
        self.deltay = pix_freedom * 1./dims[1]
        self.deltaz = pix_freedom * 1./dims[2]
        # minimal and maximum broadening
        if minsig is None:
            # reasoning: 3 minsig = pixsize / 2
            self.minsig = ( (1./dims[0] + 1./dims[1] + 1./dims[2])/3. ) / 6.
        else:
            self.minsig = minsig
        if maxsig is None:
            K = np.sum(self.mask)//Nc
            self.maxsig = K*self.minsig
        else:
            self.maxsig = maxsig
        # inverse transformation to (real) model parameters
        self.c = np.sqrt(c0)
        self.sig = inv_sig_mapping(sig0, minsig, maxsig)
        self.d1psi1 = d1psi1
        self.d1psi2 = d1psi2
        self.d2psi2 = d2psi2
        self.a = a
        self.b = b
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.base_level = base_level
        self.support = support
        # solution variables
        self.scipy_sol = None
        self.elapsed_time = None
        self.residual_stats = None

        
    def set_centers(self, theta_xc, theta_yc, theta_zc):        
        self.xc = self.xc0 + self.deltax * np.sin(theta_xc)
        self.yc = self.yc0 + self.deltay * np.sin(theta_yc)
        self.zc = self.zc0 + self.deltaz * np.sin(theta_zc)
    
    
    def set_theta(self, theta_xc, theta_yc, theta_zc):
        self.theta_xc = theta_xc
        self.theta_yc = theta_yc
        self.theta_zc = theta_zc
      
    
    def set_c(self, c):
        self.c = c

        
    def set_sig(self, sig):
        self.sig = sig
    
    
    def set_params(self, params):
        N = len(params)//5
        self.theta_xc = params[0:N]
        self.theta_yc = params[N:2*N]
        self.theta_zc = params[2*N:3*N]
        self.c = params[3*N:4*N]
        self.sig = params[4*N:5*N]

    
    def get_params(self):
        """
        Get the parameter of the function F (to optimize): 
        theta_xc, theta_yc, theta_zc, c, sig
        """
        return np.concatenate([self.theta_xc, self.theta_yc, self.theta_zc, self.c, self.sig])

    
    def get_params_mapped(self):
        """
        Get the real parameters of the model (mapped/bounded):
        xc, yc, zc, c, sig
        """
        #xc = self.xc0 + self.deltax * np.sin(self.theta_xc)
        #yc = self.yc0 + self.deltay * np.sin(self.theta_yc)
        xc = self.xc
        yc = self.yc
        zc = self.zc
        c = self.c**2
        sig = sig_mapping(self.sig, self.minsig, self.maxsig)
        return xc, yc, zc, c, sig


    def get_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal distributions. 
        """
        xc, yc, zc, c, sig = self.get_params_mapped()
        d = len(self.dims)
        w = c * (2*np.pi*sig**2)**(d/2.)
        return w
    

    def get_residual_stats(self):
        _xe = np.linspace(0., 1., self.dims[0]+2)[1:-1]
        _ye = np.linspace(0., 1., self.dims[1]+2)[1:-1]
        _ze = np.linspace(0., 1., self.dims[2]+2)[1:-1]
        Xe,Ye,Ze = np.meshgrid(_xe, _ye, _ze, sparse=False, indexing='ij')
        xe = Xe.ravel(); ye = Ye.ravel(); ze = Ze.ravel()

        xc, yc, zc, c, sig = self.get_params_mapped()

        u = u_eval(c, sig, xc, yc, zc, xe, ye, ze, support=self.support)
        u = u.reshape(self.dims)

        if self.mask is not None:
            residual = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u
        
        return (estimate_variance(residual), 
                estimate_entropy(residual),
                estimate_rms(residual))


    def prune(self):
        w = self.get_w()
        mask, _ = prune(w)
        #update all the arrays
        self.xc = self.xc[mask]; self.xc0 = self.xc0[mask]
        self.yc = self.yc[mask]; self.yc0 = self.yc0[mask]
        self.zc = self.zc[mask]; self.zc0 = self.zc0[mask]
        self.theta_xc = self.theta_xc[mask]
        self.theta_yc = self.theta_yc[mask]
        self.theta_zc = self.theta_zc[mask]
        self.c = self.c[mask]
        self.sig = self.sig[mask]

    
    def summarize(self, solver_output=True, residual_stats=True, solution_plot=True, params_plot=True):
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90 + '\n')
        _xc, _yc, _zc, _c, _sig = self.get_params_mapped()
        
        if solver_output:
            print('Solver Output:')
            print('\nsuccess: {0}'.format(self.scipy_sol['success']))
            print('\nstatus: {0}'.format(self.scipy_sol['status']))
            print('\nmessage: {0}'.format(self.scipy_sol['message']))
            print('\nnfev: {0}'.format(self.scipy_sol['nfev']))
            
        if residual_stats:
            var,entr,rms  = self.get_residual_stats()
            print('Residual RMS: {0}'.format(rms))
            print('Residual Variance: {0}'.format(var))
            print('Residual Entropy: {0}'.format(entr))
            print('Total elapsed time: {0} [s]'.format(self.elapsed_time))
            
        
        if solution_plot:
            gp.solution_plot3D(self)

        if params_plot:
            gp.params_distribution_plot(_c, _sig)
    
        
    def F(self, params):
        N = len(params)//5
        theta_xc = params[0:N]
        theta_yc = params[N:2*N]
        theta_zc = params[2*N:3*N]
        
        # parameters transform/mapping
        xc = self.xc0 + self.deltax * np.sin(theta_xc)
        yc = self.yc0 + self.deltay * np.sin(theta_yc)
        zc = self.zc0 + self.deltaz * np.sin(theta_zc)
        c = params[3*N:4*N]**2
        sig = sig_mapping(params[4*N:5*N], self.minsig, self.maxsig)
        
        # evaluation points
        xe = np.hstack([self.xe, xc]); ye = np.hstack([self.ye, yc]); ze = np.hstack([self.ze, zc])
        #xe = self.xe; ye = self.ye; ze = self.ze
        xb = self.xb; yb = self.yb; zb = self.zb
        
        # computing u, ux, uy, ...
        u = u_eval(c, sig, xc, yc, zc, xe, ye, ze, support=5.) + self.base_level
        
        # computing the EL equation
        #f0 = self.f0
        f0 = np.hstack([ self.f0, self.dfunc(np.vstack([xc,yc,zc]).T) ])
        el = 2.*(u-f0) + self.a*self.d1psi1(u-f0, self.lamb1)
        
        return el


#################################################################
# Euler-Lagrange instansiation solver
#################################################################
def elm_solver(elm, method='standard', max_nfev=None, n_iter=100, verbose=True):
    t0 = time.time()

    if method=='standard':
        # lm optimization from scipy.optimize.root
        options = {'maxiter':max_nfev, 'xtol':1.e-7, 'ftol':1.e-7}
        sol = sp.optimize.root(elm.F, elm.get_params(), method='lm', options=options)
        sol_length = len(sol.x)/5
        opt_theta_xc = sol.x[0:sol_length]
        opt_theta_yc = sol.x[sol_length:2*sol_length]
        opt_theta_zc = sol.x[2*sol_length:3*sol_length]
        opt_c = sol.x[3*sol_length:4*sol_length]
        opt_sig = sol.x[4*sol_length:5*sol_length]

        # update of best parameters
        elm.set_theta(opt_theta_xc, opt_theta_yc, opt_theta_zc)
        elm.set_centers(opt_theta_xc, opt_theta_yc, opt_theta_zc)
        elm.set_c(opt_c)
        elm.set_sig(opt_sig)
    
    elif method=='iterative':
        for it in range(n_iter):
            print('\n'+'#'*90)
            print('Results after {0} iterations'.format(it+1))
            print('#'*90)
            
            # lm optimization from scipy.optimize.root
            options = {'maxiter':max_nfev, 'xtol':1.e-7, 'ftol':1.e-7}
            sol = sp.optimize.root(elm.F, elm.get_params(), method='lm', options=options)
            sol_length = len(sol.x)/5
            opt_theta_xc = sol.x[0:sol_length]
            opt_theta_yc = sol.x[sol_length:2*sol_length]
            opt_theta_zc = sol.x[2*sol_length:3*sol_length]
            opt_c = sol.x[3*sol_length:4*sol_length]
            opt_sig = sol.x[4*sol_length:5*sol_length]

            # update of best parameters
            elm.set_theta(opt_theta_xc, opt_theta_yc, opt_theta_zc)
            elm.set_centers(opt_theta_xc, opt_theta_yc, opt_theta_zc)
            elm.set_c(opt_c)
            elm.set_sig(opt_sig)
            
            print('\nsuccess: {0}'.format(sol['success']))
            print('\nstatus: {0}'.format(sol['status']))
            print('\nmessage: {0}'.format(sol['message']))
            print('\nnfev: {0}'.format(sol['nfev']))
            if sol['success']: break    

    elm.scipy_sol = sol
    elm.elapsed_time = time.time() - t0
    if verbose: elm.summarize()	

