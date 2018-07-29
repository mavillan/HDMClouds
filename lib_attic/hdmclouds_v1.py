import time
import scipy
import numba
import scipy as sp
import numpy as np
import numexpr as ne
from math import sqrt, exp
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import graph as gp
from utils import *
from points_generation import *
from preprocessing import *
from gmr import *
from fgm_eval import *




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



class HDMClouds():
    """
    Hierarchical Decomposition of Molecular Clouds
    """
    def __init__(self, data, freq=None, alpha=0., lamb=1., compression=0.25, back_level=None,
        minsig=None, maxsig=None, kappa=5., verbose=False, wcs=None):

        #############################################################
        # Preprocessing: Estimation of back_level, computing mask, 
        # standardizing the data, building the linear interpolator
        #############################################################
        self.orig_data = data
        if back_level is None:
            back_level = estimate_rms(data)
        data = np.copy(data)
        mask = compute_mask(data, back_level)
        data[mask] -= back_level
        vmin = data[mask].min(); vmax = data[mask].max()
        data -= vmin
        data /= vmax-vmin
        data[~mask] = 0
        pix_lenght = 1./max(data.shape)
    
        if data.ndim==2:
            # generating the data function
            _x = (data.shape[0]*pix_lenght) * np.linspace(0., 1., data.shape[0]+1, endpoint=True)
            _y = (data.shape[1]*pix_lenght) * np.linspace(0., 1., data.shape[1]+1, endpoint=True)
            _xe = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
            _ye = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
            dfunc = RegularGridInterpolator((_xe,_ye), data, method='linear', bounds_error=False, fill_value=0.)

        elif data.ndim==3:
            # generating the data function
            _x = (data.shape[0]*pix_lenght) * np.linspace(0., 1., data.shape[0]+1, endpoint=True)
            _y = (data.shape[1]*pix_lenght) * np.linspace(0., 1., data.shape[1]+1, endpoint=True)
            _z = (data.shape[2]*pix_lenght) * np.linspace(0., 1., data.shape[2]+1, endpoint=True)
            _xe = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
            _ye = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
            _ze = np.asarray( [(_z[i]+_z[i+1])/2 for i in range(len(_z)-1)] )
            dfunc = RegularGridInterpolator((_xe, _ye, _ze), data, method='linear', bounds_error=False, fill_value=0.)

        self.data = data
        self.freq = freq
        self.mask = mask
        self.ndim = data.ndim
        self.shape = data.shape
        self.back_level = back_level
        self.vmin = vmin
        self.vmax = vmax
        self.dfunc = dfunc
        self.wcs = wcs
        self.pix_lenght = pix_lenght
        
        # Verification of consistency between n_center and number of significant emission pixels
        npix = np.sum(mask)
        n_gaussians = int(compression*npix)
        print("[INFO] Number of pixels with significant emission: {0}".format(npix))
        if 2*n_gaussians > npix:
            n_gaussians = npix//2
            print("[WARNING] The number of evaluation points (2*n_gaussians) cannot"
                  "be greater than the number of significant emission pixels: {0}".format(npix))
            print("[WARNING] n_gaussians was set to: {0}".format(n_gaussians))

        #######################################
        # Regular-grid points
        #######################################
        if self.ndim==2:
            Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
            xgrid = Xe.ravel(); ygrid = Ye.ravel()
            self.xgrid = xgrid
            self.ygrid = ygrid
            grid_points = np.vstack([xgrid,ygrid]).T
            self.grid_points = grid_points
        if self.ndim==3:
            Xe,Ye,Ze = np.meshgrid(_xe, _ye, _ze, sparse=False, indexing='ij')
            xgrid = Xe.ravel(); ygrid = Ye.ravel(); zgrid = Ze.ravel()
            self.xgrid = xgrid
            self.ygrid = ygrid
            self.zgrid = zgrid
            grid_points = np.vstack([xgrid,ygrid,zgrid]).T
            self.grid_points = grid_points


        #######################################
        # Computing boundary points
        #######################################
        #Nb = 0
        #boundary_points = boundary_points_generation(data, mask, Nb)
        #gp.points_plot(data, points=boundary_points, color="green", wcs=wcs)
        # right format
        #xb = boundary_points[:,0]
        #yb = boundary_points[:,1]
        #points_bound = np.vstack([xb,yb]).T


        ##########################################
        # Finding the isolated cloud entities
        ##########################################

        # initial weights
        w_init = (data[mask]).astype(np.float64)
        # initial mu
        if self.ndim==2: mu_init = np.vstack([Xe[mask], Ye[mask]]).T
        if self.ndim==3: mu_init = np.vstack([Xe[mask], Ye[mask], Ze[mask]]).T
        # initial sigma
        k = 0.25
        sig_init = (pix_lenght/(2.*k))*np.ones(w_init.shape[0])

        # DBSAN to find isolated structures
        if self.ndim==2: radio = np.sqrt(2)*pix_lenght
        if self.ndim==3: radio = np.sqrt(3)*pix_lenght
        db = DBSCAN(eps=radio, min_samples=4, n_jobs=-1)
        db.fit(mu_init)
        # visualizing the results
        if verbose:
            if self.ndim==2: gp.points_clusters(data, mu_init, db.labels_, wcs=wcs, title="Isolated Cloud Entities")

        # ensure that no -1 labels are present here!!!
        if np.any(db.labels_==-1): print("-1 labels are present!")

        # instantiating HDICE for each Isolated Cloud Entity
        hdice_list = list()
        for i in range(db.labels_.min(), db.labels_.max()+1):
            _mask = db.labels_==i
            print("Isolated Cloud Entity {0}: {1} pixels of significant emission.".format(i, np.sum(_mask)))
            hdice = HDICE(w_init[_mask], mu_init[_mask], sig_init[_mask], back_level, alpha, lamb, compression)
            hdice.set_f0(dfunc(hdice.eval_points))
            hdice.set_fgrid(dfunc(hdice.grid_points))
            hdice_list.append(hdice)

        # retrieving the parameters for the global GM
        w_global = list(); mu_global = list(); sig_global = list()
        eval_points_global = list()
        for hdice in hdice_list:
            w,sig = hdice.get_params_mapped()
            mu = hdice.center_points
            eval_points = hdice.eval_points
            w_global.append(w)
            mu_global.append(mu)
            sig_global.append(sig)
            eval_points_global.append(eval_points)

        w_global = np.hstack(w_global)
        mu_global = np.vstack(mu_global)
        sig_global = np.hstack(sig_global)
        eval_points_global = np.vstack(eval_points_global)

        # computing neighborhood indexes to perform fast GM evaluation
        minsig = np.min(np.abs(sig_global))
        maxsig = 3*np.max(np.abs(sig_global))
        nn_indexes,nn_indexes_aux = compute_neighbors(mu_global, grid_points, kappa*maxsig)
        self.nn_ind = nn_indexes
        self.nn_ind_aux = nn_indexes_aux

        if verbose:
            # visualizing the choosen points
            if self.ndim==2:
                print("#"*100)
                gp.points_plot(data, points=mu_global, color="red", wcs=wcs, title="Gaussian centers")
                gp.points_plot(data, points=eval_points_global, color="blue", wcs=wcs, title="Evaluation Points")

        ########################################
        # HDMClouds internals
        ########################################
        # list with HDICE objetcs for each ICE
        self.hdice_list = hdice_list
        # gaussian centers points
        self.xc = mu_global[:,0]; self.yc = mu_global[:,1] 
        if self.ndim==3: self.zc = mu_global[:,2]
        # minsig and maxsig for the global GM
        self.minsig = minsig
        self.maxsig = maxsig
        self.kappa = kappa
        # parameters for the global GM
        self.w = w_global
        self.w0 = np.copy(w_global)
        self.sig = sig_global
        self.sig0 = np.copy(sig_global)
        # some other useful data 
        self.compression = compression
        self.alpha = alpha
        self.lamb = lamb
        self.elapsed_time = None
        self.residual_stats = None


    def set_params(self,w,sig):
        """
        Set the parameters of the global GM
        """
        self.w = w
        self.sig = sig

    def get_params(self):
        """
        Get the parameters of the global GM
        """
        return self.w,self.sig


    def normalized_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal functions. 
        """
        w,sig = self.get_params()
        d = self.ndim
        w = w * (2*np.pi*sig**2)**(d/2.)
        return w


    def get_approximation(self):
        w,sig = self.get_params()
        if self.ndim==2:
            u = gm_eval2d_2(w, sig, self.xc, self.yc, self.xgrid, self.ygrid, self.nn_ind, self.nn_ind_aux)
        if self.ndim==3:
            u = gm_eval3d_2(w, sig, self.xc, self.yc, self.zc, self.xgrid, self.ygrid, self.zgrid, self.nn_ind, self.nn_ind_aux)
        u = u.reshape(self.shape)
        return u


    def get_residual_stats(self, plot=True):
        u = self.get_approximation()
        if self.mask is not None:
            residual = np.zeros(self.shape)
            residual[self.mask] = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u

        # visualization original data, u and residual
        if plot and self.ndim==2:
            gp.solution_plot(self.data, u, residual)
            gp.residual_histogram(residual[self.mask])

        if plot and self.ndim==3:
            print("-"*118)
            print("ORIGINAL DATA")
            gp.cube_plot(self.data, wcs=self.wcs, freq=self.freq)
            print("-"*118)
            print("GAUSSIAN MIXTURE")
            gp.cube_plot(u, wcs=self.wcs, freq=self.freq)
            print("-"*118)
            print("RESIDUAL")
            gp.cube_plot(residual, wcs=self.wcs, cmap=plt.cm.RdBu_r, freq=self.freq)
            print("-"*118)

        # computing residual stats
        total_flux = np.sum(self.data[self.mask])
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])

        # output residuals
        out = (estimate_rms(residual), estimate_variance(residual), \
               flux_addition/total_flux, flux_lost/total_flux)
        
        # printing output
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
            
    
    def summarize(self, solver_output=True, residual_stats=True, solution_plot=True,
                  params_plot=True, histograms_plot=True):
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90 + '\n')
        _w,_sig = self.get_params_mapped()

        out = self.get_residual_stats(plot=False)
        var,rms,flux_addition,flux_lost = out
        
        if solution_plot:
            gp.solution_plot(self.dfunc, _w, _sig, self.xc, self.yc, shape=self.shape)
        
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

            


    def build_gmr(self, max_nfev=None, verbose=True, tol=1.e-7):
        """
        Build the Gaussian Mixture Representation for the
        input data
        """
        t0 = time.time()

        w_global = list(); sig_global = list()
        for i,hdice in enumerate(self.hdice_list):
            print("-"*45)
            print("Building GM for Isolated Cloud Entity {0}".format(i))
            print("-"*45)
            hdice.build_gmr(max_nfev=max_nfev, tol=tol)
            _w,_sig = hdice.get_params_mapped()
            w_global.append(_w)
            sig_global.append(_sig)
            if verbose: hdice.get_residual_stats(); print("\n")

        # updating the global GM parameters
        w_global = np.hstack(w_global)
        sig_global = np.hstack(sig_global)
        self.set_params(w_global, sig_global)

        # prune of gaussians from mixture
        #self.prune()
    
        # storing the total elapsed time    
        self.elapsed_time = time.time() - t0



class HDICE():
    """
    Hierarchical Decomposition of Independent/Isolated Cloud Entities
    """
    def __init__(self, w_init, mu_init, sig_init, back_level, alpha, lamb, compression,
        minsig=None, maxsig=None, kappa=5., verbose=False,):

        self.ndim = mu_init.shape[1]
        # Max intensity and grid positions in the region
        data_max = w_init.max()
        xgrid = mu_init[:,0]
        ygrid = mu_init[:,1]
        if self.ndim==3: zgrid = mu_init[:,2]
        grid_points = mu_init

        # target number of gaussians (ceil rounding)
        n_gaussians = int(1+len(w_init)*compression)

        w_red, mu_red, cov_red = mixture_reduction(w_init, mu_init, sig_init, 2*n_gaussians, verbose=False)
        xe = mu_red[:,0]
        ye = mu_red[:,1]
        if self.ndim==3: ze = mu_red[:,2]
        eval_points = mu_red

        w, mu, cov = mixture_reduction(w_red, mu_red, cov_red, n_gaussians, verbose=False)
        xc = mu[:,0]
        yc = mu[:,1]
        if self.ndim==3: ze = mu_red[:,2]
        center_points = mu

        # truncation of the covariance matrices
        sig = np.asarray( [np.mean((np.linalg.eig(_cov)[0]))**(1./2) for _cov in cov] )

        minsig = np.min(np.abs(sig))
        maxsig = 3*np.max(np.abs(sig))
        epsilon = 1e-6 # little shift to avoid NaNs in inv_sig_mapping

        nn_indexes,nn_indexes_aux = compute_neighbors(center_points, eval_points, kappa*maxsig)
        self.nn_ind1 = nn_indexes
        self.nn_ind1_aux = nn_indexes_aux

        #neigh_indexes,neigh_indexes_aux = compute_neighbors(mu_red, points_bound, 5*np.max(sig_red))
        #self.nind2 = neigh_indexes
        #self.nind_aux2 = neigh_indexes_aux

        nn_indexes,nn_indexes_aux = compute_neighbors(center_points, grid_points, kappa*maxsig)
        self.nn_ind3 = nn_indexes
        self.nn_ind3_aux = nn_indexes_aux

        if self.ndim==2:
            # normalizing w
            u = gm_eval2d_2(w, sig, xc, yc, xgrid, ygrid, self.nn_ind3, self.nn_ind3_aux)
            w *= data_max/u.max()
        if self.ndim==3:
            # normalizing w
            u = gm_eval3d_2(w, sig, xc, yc, zc, xgrid, ygrid, zgrid, self.nn_ind3, self.nn_ind3_aux)
            w *= data_max/u.max()


        ########################################
        # HDICE internals
        ########################################

        # data values at the position of center_points and grid_points
        self.f0 = None
        self.fgrid = None
        # center points
        self.xc = xc; self.yc = yc 
        if self.ndim==3: self.zc = zc
        self.center_points = center_points
        # boundary points
        #self.xb = xb; self.yb = yb
        #if self.ndim==3: self.zb = zb
        # evaluation points
        self.xe = xe; self.ye = ye
        if self.ndim==3: self.ze = ze
        self.eval_points = eval_points
        # grid points
        self.xgrid = xgrid; self.ygrid = ygrid
        self.grid_points = grid_points
        # minimal and maximal extend of gaussians
        self.minsig = minsig
        self.maxsig = maxsig
        self.kappa = kappa
        # optimization parameters
        self.w0 = np.copy(w)
        self.w = np.sqrt(w)
        self.sig0 = np.copy(sig)
        self.sig = inv_sig_mapping(sig+epsilon, minsig, maxsig)
        # some other useful data
        self.d1psi1 = d1psi1
        self.alpha = alpha
        self.lamb = lamb
        self.back_level = back_level
        self.scipy_sol = None
        self.scipy_tol = None
        self.elapsed_time = None
        self.residual_stats = None

    def set_f0(self, f0):
        self.f0 = f0

    def set_fgrid(self, fgrid):
        self.fgrid = fgrid

    def set_w(self, w):
        self.w = w

    def set_sig(self, sig):
        self.sig = sig

    def set_params(self, params):
        N = len(params)//2
        self.w = params[0:N]
        self.sig = params[N:]

    def get_params(self):
        """
        Get the parameter of the function F (to optimize): 
        theta_xc, theta_yc, c, sig
        """
        return np.concatenate([self.w, self.sig])

    def get_params_mapped(self):
        """
        Get the real parameters of the model (mapped/bounded):
        xc, yc, c, sig
        """
        #xc = self.xc0 + self.deltax * np.sin(self.theta_xc)
        #yc = self.yc0 + self.deltay * np.sin(self.theta_yc)
        w = self.w**2
        sig = sig_mapping(self.sig, self.minsig, self.maxsig)
        return w,sig

    def normalized_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal functions. 
        """
        w,sig = self.get_params_mapped()
        d = self.ndim
        w = w * (2*np.pi*sig**2)**(d/2.)
        return w

    def get_approximation(self):
        w,sig = self.get_params_mapped()
        if self.ndim==2:
            u = gm_eval2d_2(w, sig, self.xc, self.yc, self.xgrid, self.ygrid, self.nn_ind3, self.nn_ind3_aux)
        if self.ndim==3:
            u = gm_eval3d_2(w, sig, self.xc, self.yc, self.zc, self.xgrid, self.ygrid, self.zgrid, self.nn_ind3, self.nn_ind3_aux)
        return u

    def get_residual_stats(self, plot=True):
        u = self.get_approximation()
        residual = self.fgrid-u

        # computing residual stats
        total_flux = np.sum(self.fgrid)
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])

        # output residuals
        out = (estimate_rms(residual), estimate_variance(residual), \
               flux_addition/total_flux, flux_lost/total_flux)
        
        # printing output        
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
        _w,_sig = self.get_params_mapped()

        out = self.get_residual_stats(plot=False)
        var,rms,flux_addition,flux_lost = out
        
        if solution_plot:
            gp.solution_plot(self.dfunc, _w, _sig, self.xc, self.yc, shape=self.shape)
        
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
        N = len(params)//2

        # parameters transform/mapping
        w = params[0:N]**2
        sig = sig_mapping(params[N:], self.minsig, self.maxsig)
        
        # computing the EL equation
        if self.ndim==2:
            u = gm_eval2d_2(w, sig, self.xc, self.yc, self.xe, self.ye, self.nn_ind1, self.nn_ind1_aux)
        if self.ndim==3:
            u = gm_eval3d_2(w, sig, self.xc, self.yc, self.zc, self.xe, self.ye, self.ze, self.nn_ind1, self.nn_ind1_aux)

        # evaluation of the EL equation
        f0 = self.f0
        alpha = self.alpha; lamb = self.lamb
        tmp1 = ne.evaluate('u-f0')
        tmp2 = self.d1psi1(tmp1, lamb)
        el = ne.evaluate('2*tmp1 + alpha*tmp2')
            
        # evaluating at the boundary
        #fb = self.fb
        #u_boundary = gm_eval_fast(w, sig, self.xc, self.yc, self.xb, self.yb, self.nind2, self.nind_aux2)
        #if self.ndim==2:
        #    u_boundary = gm_eval2d_1(w, sig, self.xc, self.yc, self.xb, self.yb)
        #if self.ndim==3:
        #    u_boundary = gm_eval3d_1(w, sig, self.xc, self.yc, self.zc, self.xb, self.yb, self.zb)
        
        #return np.concatenate([el,u_boundary-fb])
        return el


    def build_gmr(self, max_nfev=None, tol=1.e-7):
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
        sol_length = len(sol.x)//2
        opt_w = sol.x[0:sol_length]
        opt_sig = sol.x[sol_length:]

        # update to best parameters
        self.set_w(opt_w)
        self.set_sig(opt_sig)

        # prune of gaussians from mixture
        #self.prune()
    
        # storing results    
        self.scipy_sol = sol
        self.elapsed_time = time.time() - t0

